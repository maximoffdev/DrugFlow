import os
os.environ["WANDB_SERVICE_WAIT"] = "300"

import argparse
from argparse import Namespace
from pathlib import Path
import warnings
from typing import List

import torch
import pytorch_lightning as pl
import yaml

from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

pl_version = getattr(pl, "__version__", "0.0.0")

torch.set_float32_matmul_precision("high")   # or "medium"


import sys
basedir = Path(__file__).resolve().parent.parent
sys.path.append(str(basedir))

from src.model.lightning import DrugFlow
from src.model.dpo import DPO
from src.model.energy_force_diffusion import EnergyForceDiffusion
from src.utils import set_deterministic, disable_rdkit_logging, dict_to_namespace, namespace_to_dict
from src.file_logger import PlainFileLogger


class DataloaderShardDebugCallback(Callback):
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not bool(getattr(trainer, "is_global_zero", True)):
            return
        # Important: inspect the dataloader that the Trainer is actually using,
        # because this is where Lightning may have injected a DistributedSampler.
        dl = getattr(trainer, "train_dataloader", None)
        if dl is None:
            print("[DL SHARD DEBUG] trainer.train_dataloader is None", flush=True)
            return

        # Some PL versions expose it as a method.
        if callable(dl):
            try:
                dl = dl()
            except Exception as e:
                print(f"[DL SHARD DEBUG] failed to resolve trainer.train_dataloader(): {e}", flush=True)
                return

        def _safe_len(obj):
            try:
                return len(obj)
            except Exception:
                return None

        # Handle CombinedLoader (multiple loaders) vs. a single DataLoader.
        combined_loaders = getattr(dl, "loaders", None)
        if isinstance(combined_loaders, dict):
            loaders = combined_loaders
        else:
            loaders = {"train": dl}

        for name, dli in loaders.items():
            sampler = getattr(dli, "sampler", None)
            batch_sampler = getattr(dli, "batch_sampler", None)
            dataset = getattr(dli, "dataset", None)

            msg = {
                "name": str(name),
                "world_size": getattr(trainer, "world_size", None),
                "global_rank": getattr(trainer, "global_rank", None),
                "local_rank": int(os.environ.get("LOCAL_RANK", -1)),
                "len(dataset)": (_safe_len(dataset) if dataset is not None else None),
                "len(dataloader)": _safe_len(dli),
                "sampler": (type(sampler).__name__ if sampler is not None else None),
                "batch_sampler": (type(batch_sampler).__name__ if batch_sampler is not None else None),
            }

            # If DistributedSampler is present, expose its key attributes.
            if sampler is not None and type(sampler).__name__ == "DistributedSampler":
                msg.update({
                    "sampler.num_replicas": getattr(sampler, "num_replicas", None),
                    "sampler.rank": getattr(sampler, "rank", None),
                    "sampler.shuffle": getattr(sampler, "shuffle", None),
                    "sampler.drop_last": getattr(sampler, "drop_last", None),
                })

            print("[DL SHARD DEBUG]", msg, flush=True)


def merge_args_and_yaml(args, config_dict):
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if key in arg_dict:
            warnings.warn(
                f"Command line argument '{key}' (value: {arg_dict[key]}) "
                f"will be overwritten with value {value} provided in the config file."
            )
        arg_dict[key] = dict_to_namespace(value)

    return args
def merge_configs(config, resume_config):
    """Merge a checkpoint config into the current config.

    When resuming (not finetuning), we generally want to keep hyperparameters
    consistent with the original run stored in the checkpoint.
    """

    for key, value in resume_config.items():
        if isinstance(value, Namespace):
            value = value.__dict__

        if isinstance(value, dict) and key in config and isinstance(config.get(key), dict):
            config[key] = merge_configs(config[key], value)
            continue

        if key in config and config[key] != value:
            print(f'[CONFIG UPDATE] {key}: {config[key]} -> {value}')
        config[key] = value

    return config


# ------------------------------------------------------------------------------
# Training
# ______________________________________________________________________________
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--backoff', action='store_true')
    p.add_argument('--finetune', action='store_true')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--overfit', action='store_true')
    p.add_argument('--test', action='store_true', help='Run trainer.test() after fit finishes')
    args = p.parse_args()

    set_deterministic(seed=42)
    disable_rdkit_logging()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    assert 'resume' not in config
    assert not (args.resume is not None and args.backoff)
    config['dpo_mode'] = config.get('dpo_mode', None)
    assert not (config['dpo_mode'] and 'checkpoint' not in config), 'DPO mode requires a reference checkpoint'

    if args.debug:
        config['run_name'] = 'debug'

    out_dir = Path(config['train_params']['logdir'], config['run_name'])
    checkpoints_root_dir = Path(out_dir, 'checkpoints')
    if args.backoff:
        last_checkpoint = Path(checkpoints_root_dir, 'last.ckpt')
        print(f'Checking if there is a checkpoint at: {last_checkpoint}')
        if last_checkpoint.exists():
            print(f'Found existing checkpoint: {last_checkpoint}')
            args.resume = str(last_checkpoint)
        else:
            print(f'Did not find {last_checkpoint}')

    # Get main config
    ckpt_path = None if args.resume is None else Path(args.resume)
    if args.resume is not None and not args.finetune:
        assert ckpt_path is not None
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        print(f'Resuming from epoch {ckpt["epoch"]}')
        resume_config = ckpt['hyper_parameters']
        config = merge_configs(config, resume_config)

    args = merge_args_and_yaml(args, config)

    print(
        "[train] run_name=", args.run_name,
        " out_dir=", out_dir,
        " max_epochs=", getattr(args.train_params, "n_epochs", None),
        " resume=", args.resume,
        " backoff=", args.backoff,
        " finetune=", args.finetune,
        " ckpt_path=", str(ckpt_path) if ckpt_path is not None else None,
        flush=True,
    )

    if args.debug:
        print('DEBUG MODE')
        args.wandb_params.mode = 'disabled'
        args.train_params.enable_progress_bar = True
        args.train_params.num_workers = 0

    if args.overfit:
        print('OVERFITTING MODE')

    args.eval_params.outdir = out_dir
    model_type = getattr(args, 'model_type', None)
    if model_type in {'energy_force', 'energy_force_diffusion'}:
        model_class = EnergyForceDiffusion
    else:
        model_class = DPO if args.dpo_mode else DrugFlow
    model_args = {
        'pocket_representation': args.pocket_representation,
        'train_params': args.train_params,
        'loss_params': args.loss_params,
        'eval_params': args.eval_params,
        'predictor_params': args.predictor_params,
        'simulation_params': args.simulation_params,
        'virtual_nodes': args.virtual_nodes,
        'flexible': args.flexible,
        'flexible_bb': args.flexible_bb,
        'debug': args.debug,
        'overfit': args.overfit,
    }
    if args.dpo_mode:
        print('DPO MODE')
        model_args.update({
            'dpo_mode': args.dpo_mode,
            'ref_checkpoint_p': args.checkpoint,
        })
    pl_module = model_class(**model_args)

    wandb_mode = getattr(args.wandb_params, 'mode', 'online')
    if wandb_mode == 'disabled':
        logger = PlainFileLogger(save_dir=args.train_params.logdir, filename=f"logfile_{args.run_name}")
    else:
        resume_logging = False
        if args.finetune:
            resume_logging = 'allow'
        elif args.resume is not None:
            resume_logging = 'must'

        logger = WandbLogger(
            save_dir=args.train_params.logdir,
            project='FlexFlow',
            group=args.wandb_params.group,
            name=args.run_name,
            id=args.run_name,
            resume=resume_logging,
            entity=args.wandb_params.entity,
            mode=wandb_mode,
        )

    checkpoint_callbacks: List[Callback] = [
        ModelCheckpoint(
            dirpath=checkpoints_root_dir,
            save_last=True,
            save_on_train_epoch_end=True,
        ),
        ModelCheckpoint(
            dirpath=Path(checkpoints_root_dir, 'val_loss'),
            filename="epoch_{epoch:04d}_loss_{loss/val:.3f}",
            monitor="loss/val",
            save_top_k=5,
            mode="min",
            auto_insert_metric_name=False,
        ),
    ]

    # For learning rate logging
    lr_monitor = LearningRateMonitor(logging_interval='step')

    default_strategy = 'auto' if pl_version >= '2.0.0' else None

    trainer_kwargs = {}
    # EnergyForceDiffusion currently expects Trainer-level gradient clipping (its config
    # exposes train_params.clip_grad but does not implement internal clipping like DrugFlow).
    if getattr(args, "model_type", None) == "energy_force_diffusion" and bool(getattr(args.train_params, "clip_grad", False)):
        clip_val = getattr(args.train_params, "gradient_clip_val", None)
        if clip_val is None:
            clip_val = 1.0
        trainer_kwargs["gradient_clip_val"] = float(clip_val)
        trainer_kwargs["gradient_clip_algorithm"] = "norm"

    trainer_kwargs_strategy = {}
    if args.train_params.gpus > 1:
        trainer_kwargs_strategy["strategy"] = 'ddp_find_unused_parameters_true'
    elif default_strategy is not None:
        trainer_kwargs_strategy["strategy"] = default_strategy

    callbacks: List[Callback] = checkpoint_callbacks + [lr_monitor, DataloaderShardDebugCallback()]

    trainer = pl.Trainer(
        max_epochs=args.train_params.n_epochs,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=args.train_params.enable_progress_bar,
        check_val_every_n_epoch=args.eval_params.eval_epochs,
        num_sanity_val_steps=args.train_params.num_sanity_val_steps,
        accumulate_grad_batches=args.train_params.accumulate_grad_batches,
        accelerator='gpu' if args.train_params.gpus > 0 else 'cpu',
        devices=args.train_params.gpus if args.train_params.gpus > 0 else 'auto',
        use_distributed_sampler=True,
        **trainer_kwargs_strategy,
        **trainer_kwargs,
    )

    # Add all arguments as a dictionary.
    # - W&B: store in run config.
    # - File logger (and other loggers): log as hyperparams.
    args_as_dict = {'as_dict': namespace_to_dict(args)}
    if isinstance(logger, WandbLogger):
        logger.experiment.config.update(args_as_dict, allow_val_change=True)
    else:
        try:
            logger.log_hyperparams(args_as_dict)
        except Exception:
            pass

    trainer.fit(model=pl_module, ckpt_path=ckpt_path)

    if args.test:
        trainer.test(model=pl_module, ckpt_path='best')
