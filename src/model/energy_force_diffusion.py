from __future__ import annotations

from argparse import Namespace
from typing import Literal, cast
from pathlib import Path
from typing import Optional, Union
import warnings

import numpy as np
import pytorch_lightning as pl
import torch
from torch_scatter import scatter_add, scatter_mean

from src.data.pkl_dataset import PKLEnergyForceDataset, PKLKeys, split_pkl_file_paths
from src.model.diffusion_utils import DistributionNodes
from src.model.dynamics_radius_gvp import RadiusGVPDynamics, RadiusGVPParams
from src.model.dynamics_radius_mlp import RadiusMLPDynamics, RadiusMLPParams
from src.model.flows import CategoricalLogitScoreDiffusion, CoordScoreDiffusion, SDEParams
from src.model.markov_bridge import UniformPriorMarkovBridge
from src import utils
from src.utils import num_nodes_to_batch_mask


def set_default(namespace: Namespace, key: str, default_val):
    val = vars(namespace).get(key, default_val)
    setattr(namespace, key, val)


class EnergyForceDiffusion(pl.LightningModule):
    """Ligand-only diffusion on (x, h) for OMol25-style PKLs.

    This model is intentionally independent from DrugFlow because DrugFlow assumes a
    fixed atom vocabulary from src.constants and also includes bond diffusion.

    Current scope (Step 1):
      - Diffuse coordinates (continuous) and atom types (discrete)
      - No bond diffusion
      - Ligand-only (empty pocket placeholder)
      - Uses a minimal per-node MLP dynamics as a baseline

      - Replace dynamics with a radius-graph message passing backbone
      - Add energy + force heads and the boundary condition loss tying force_head to -∇E
      - Optionally include pocket nodes
    """

    def __init__(
        self,
        pocket_representation: str,
        train_params: Namespace,
        loss_params: Namespace,
        eval_params: Namespace,
        predictor_params: Namespace,
        simulation_params: Namespace,
        virtual_nodes: Union[list, None],
        flexible: bool,
        flexible_bb: bool = False,
        debug: bool = False,
        overfit: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Keep references to the merged Namespaces (avoid relying on hparams typing).
        self._train_params = train_params
        self._loss_params = loss_params
        self._eval_params = eval_params
        self._predictor_params = predictor_params
        self._simulation_params = simulation_params

        # Dataset defaults tailored to OMol25 PKLs
        set_default(train_params, "pkl_pos_key", "pos")
        set_default(train_params, "pkl_atom_types_key", "atomic_numbers")
        set_default(train_params, "pkl_energy_key", "e_total")
        set_default(train_params, "pkl_forces_key", "forces")
        set_default(train_params, "pkl_atom_types_format", "atomic_numbers")
        # OM25_1M_HCNOSPClF -> Z = [1, 6, 7, 8, 9, 15, 16, 17]
        set_default(train_params, "pkl_allowed_atomic_numbers", [1, 6, 7, 8, 9, 15, 16, 17])
        set_default(train_params, "pkl_file_limit", None)
        set_default(train_params, "pkl_val_fraction", 0.01)
        set_default(train_params, "pkl_val_n", None)
        set_default(train_params, "pkl_split_seed", 42)

        # Training defaults
        set_default(train_params, "lr_step_size", None)
        set_default(train_params, "lr_gamma", None)
        set_default(loss_params, "lambda_x", 1.0)
        set_default(loss_params, "lambda_h", 1.0)
        set_default(loss_params, "lambda_energy", 1.0)
        # Optional HJB PDE loss + force/energy consistency (at noisy time)
        set_default(loss_params, "lambda_hjb", 0.0)
        set_default(loss_params, "lambda_consistency", 0.0)
        set_default(loss_params, "reduce", "mean")
        # Must be one of {'CE', 'VLB'} to match src/model/markov_bridge.py
        set_default(loss_params, "discrete_loss", "CE")

        # Core training params
        self.datadir = train_params.datadir
        self.batch_size = train_params.batch_size
        self.eval_batch_size = getattr(eval_params, "eval_batch_size", train_params.batch_size)
        self.num_workers = train_params.num_workers
        self.lr = train_params.lr
        self.lr_step_size = train_params.lr_step_size
        self.lr_gamma = train_params.lr_gamma

        # Loss params
        self.loss_reduce = loss_params.reduce
        self.lambda_x = loss_params.lambda_x
        self.lambda_h = loss_params.lambda_h
        self.lambda_energy = loss_params.lambda_energy
        self.lambda_hjb = loss_params.lambda_hjb
        self.lambda_consistency = loss_params.lambda_consistency

        # Diffusion params
        self.n_steps = simulation_params.n_steps
        self.train_step_size = 1.0 / float(self.n_steps)

        # Timestep sampling range (t=0 noisy -> t=1 clean). Useful to avoid very-clean
        # regimes early in training.
        set_default(simulation_params, "t_min", 0.0)
        set_default(simulation_params, "t_max", 1.0)
        self.t_min = float(simulation_params.t_min)
        self.t_max = float(simulation_params.t_max)
        if not (0.0 <= self.t_min <= self.t_max <= 1.0):
            raise ValueError(f"simulation_params.t_min/t_max must satisfy 0<=t_min<=t_max<=1; got {self.t_min}, {self.t_max}")

        # Optional training diagnostics
        set_default(train_params, "log_diffusion_stats", False)
        self.log_diffusion_stats = bool(train_params.log_diffusion_stats)

        # Option to disable atom-type diffusion (train with diffusion on x only).
        set_default(simulation_params, "diffuse_h", True)
        self.diffuse_h = bool(simulation_params.diffuse_h)

        # Temperature is treated as a true conditioning feature.
        # If the predictor is not conditioned on temperature, then temperature must not
        # affect training, evaluation, sampling, or the diffusion schedule.
        self.use_temperature = bool(getattr(predictor_params, "condition_temperature", True))
        set_default(simulation_params, "use_temperature", self.use_temperature)
        if bool(getattr(simulation_params, "use_temperature")) != self.use_temperature:
            raise ValueError(
                "simulation_params.use_temperature must match predictor_params.condition_temperature "
                "(temperature conditioning is an all-or-nothing feature)."
            )

        # Temperature conditioning (batch-level, like t)
        set_default(simulation_params, "temperature", 1.0)
        set_default(simulation_params, "temperature_min", float(simulation_params.temperature))
        set_default(simulation_params, "temperature_max", float(simulation_params.temperature))
        self.temperature = float(simulation_params.temperature)
        self.temperature_min = float(simulation_params.temperature_min)
        self.temperature_max = float(simulation_params.temperature_max)
        if not self.use_temperature:
            # Ensure we never introduce random temperatures when the feature is disabled.
            self.temperature_min = self.temperature
            self.temperature_max = self.temperature

        # Atom vocabulary for this dataset
        self.allowed_atomic_numbers = list(train_params.pkl_allowed_atomic_numbers)
        self.atom_nf = len(self.allowed_atomic_numbers)
        self.x_dim = 3

        # Modules
        # Score-based diffusion on coordinates (t=0 noise, t=1 clean)
        set_default(simulation_params, "sigma_min", 0.01)
        set_default(simulation_params, "sigma_max", 50.0)
        set_default(simulation_params, "coord_sde_kind", "ve")
        set_default(simulation_params, "vp_beta_min", 0.1)
        set_default(simulation_params, "vp_beta_max", 20.0)
        set_default(simulation_params, "vp_sigma_scale", 1.0)
        set_default(simulation_params, "enforce_zero_com_noise", False)
        set_default(simulation_params, "project_predicted_force_zero_com", False)
        set_default(loss_params, "x_score_weight_by_sigma", False)

        coord_sde = SDEParams(
            kind=cast(Literal["ve", "vp"], str(simulation_params.coord_sde_kind)),
            sigma_min=float(simulation_params.sigma_min),
            sigma_max=float(simulation_params.sigma_max),
            beta_min=float(simulation_params.vp_beta_min),
            beta_max=float(simulation_params.vp_beta_max),
            vp_sigma_scale=float(simulation_params.vp_sigma_scale),
        )

        self.module_x = CoordScoreDiffusion(
            sigma_min=float(simulation_params.sigma_min),
            sigma_max=float(simulation_params.sigma_max),
            dim=self.x_dim,
            sde=coord_sde,
            enforce_zero_com_noise=bool(simulation_params.enforce_zero_com_noise),
            project_predicted_score_zero_com=bool(simulation_params.project_predicted_force_zero_com),
        )
        self.module_h = UniformPriorMarkovBridge(self.atom_nf, loss_type=loss_params.discrete_loss)
        self.x_score_weight_by_sigma = bool(loss_params.x_score_weight_by_sigma)

        # Optional score-based diffusion on atom types via logit space.
        # This is for sampling support (requested); training remains Markov bridge for now.
        set_default(simulation_params, "sigma_min_h", 0.01)
        set_default(simulation_params, "sigma_max_h", 5.0)
        set_default(simulation_params, "logit_sde_kind", str(simulation_params.coord_sde_kind))

        logit_sde = SDEParams(
            kind=cast(Literal["ve", "vp"], str(simulation_params.logit_sde_kind)),
            sigma_min=float(simulation_params.sigma_min_h),
            sigma_max=float(simulation_params.sigma_max_h),
            beta_min=float(simulation_params.vp_beta_min),
            beta_max=float(simulation_params.vp_beta_max),
            vp_sigma_scale=float(simulation_params.vp_sigma_scale),
        )
        self.module_h_score = CategoricalLogitScoreDiffusion(
            sigma_min=float(simulation_params.sigma_min_h),
            sigma_max=float(simulation_params.sigma_max_h),
            n_classes=self.atom_nf,
            sde=logit_sde,
        )

        backbone = getattr(predictor_params, "backbone", "radius_mlp")
        if backbone == "gvp":
            gvp_params = getattr(predictor_params, "gvp_params", Namespace())
            params = RadiusGVPParams(
                hidden_scalar_nf=int(getattr(predictor_params, "hidden_nf", 128)),
                condition_time=bool(getattr(predictor_params, "condition_time", True)),
                condition_temperature=self.use_temperature,
                edge_cutoff_ligand=getattr(predictor_params, "edge_cutoff_ligand", None),
                knn_k=int(getattr(predictor_params, "knn_k", 30)),
                n_layers=int(getattr(gvp_params, "n_layers", 5)),
                node_h_dim_s=int(getattr(gvp_params, "node_h_dim", [128, 32])[0]),
                node_h_dim_v=int(getattr(gvp_params, "node_h_dim", [128, 32])[1]),
                edge_h_dim_s=int(getattr(gvp_params, "edge_h_dim", [128, 32])[0]),
                edge_h_dim_v=int(getattr(gvp_params, "edge_h_dim", [128, 32])[1]),
                dropout=float(getattr(gvp_params, "dropout", 0.1)),
                vector_gate=bool(getattr(gvp_params, "vector_gate", True)),
                reflection_equivariant=bool(getattr(predictor_params, "reflection_equivariant", False)),
                d_max=float(getattr(predictor_params, "d_max", 15.0)),
                num_rbf=int(getattr(predictor_params, "num_rbf", 16)),
            )
            self.dynamics = RadiusGVPDynamics(atom_nf=self.atom_nf, x_dim=self.x_dim, params=params)
            self.condition_time = bool(getattr(predictor_params, "condition_time", True))
        elif backbone == "radius_mlp":
            radius_mlp = getattr(predictor_params, "radius_mlp", Namespace())
            hidden_dim = getattr(radius_mlp, "hidden_dim", 256)
            n_layers = getattr(radius_mlp, "n_layers", 3)
            condition_time = getattr(radius_mlp, "condition_time", True)
            # Single source of truth for temperature conditioning:
            # if disabled, temperature must not be required or consumed by the backbone.
            condition_temperature = self.use_temperature
            params = RadiusMLPParams(
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                condition_time=condition_time,
                condition_temperature=condition_temperature,
            )
            self.dynamics = RadiusMLPDynamics(atom_nf=self.atom_nf, x_dim=self.x_dim, params=params)
            self.condition_time = bool(condition_time)
        else:
            raise ValueError(f"Unknown predictor_params.backbone: {backbone}")

        if (float(self.lambda_x) > 0.0 or float(self.lambda_h) > 0.0) and (not getattr(self, "condition_time", True)):
            warnings.warn(
                "Training diffusion without time conditioning (predictor_params.condition_time=False) often causes a flat/plateauing loss, "
                "because the model must fit incompatible score targets across all noise levels without knowing t. "
                "Consider setting predictor_params.condition_time=True.",
                stacklevel=2,
            )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Sampling support (match DrugFlow UX)
        self.size_distribution: DistributionNodes | None = None

    def configure_optimizers(self):
        optimizers = [
            torch.optim.AdamW(self.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12),
        ]
        if self.lr_step_size is None or self.lr_gamma is None:
            lr_schedulers = []
        else:
            lr_schedulers = [
                torch.optim.lr_scheduler.StepLR(optimizers[0], step_size=self.lr_step_size, gamma=self.lr_gamma),
            ]
        return optimizers, lr_schedulers

    def _make_dataset(self, stage: str) -> PKLEnergyForceDataset:
        datadir = Path(self.datadir)
        stage_dir = datadir / stage

        keys = PKLKeys(
            pos=self._train_params.pkl_pos_key,
            atom_types=self._train_params.pkl_atom_types_key,
            energy=self._train_params.pkl_energy_key,
            forces=self._train_params.pkl_forces_key,
        )

        allowed_z = list(self._train_params.pkl_allowed_atomic_numbers)

        if stage_dir.exists():
            return PKLEnergyForceDataset(
                stage_dir,
                keys=keys,
                atom_types_format=self._train_params.pkl_atom_types_format,
                allowed_atomic_numbers=allowed_z,
                num_atom_types=len(allowed_z),
                device="cpu",
                file_limit=self._train_params.pkl_file_limit,
            )

        all_files = sorted(datadir.glob("*.pkl"))
        if len(all_files) == 0:
            raise FileNotFoundError(
                f"No .pkl files found in {datadir} and {stage_dir} does not exist. "
                f"Either create {stage_dir} or point train_params.datadir to a folder containing .pkl files."
            )

        train_files, val_files = split_pkl_file_paths(
            all_files,
            val_fraction=float(self._train_params.pkl_val_fraction),
            seed=int(self._train_params.pkl_split_seed),
            val_n=self._train_params.pkl_val_n,
        )

        if stage == "train":
            chosen = train_files
        elif stage in {"val", "test"}:
            chosen = val_files
        else:
            raise ValueError(f"Unknown stage: {stage}")

        return PKLEnergyForceDataset(
            datadir,
            file_paths=chosen,
            keys=keys,
            atom_types_format=self._train_params.pkl_atom_types_format,
            allowed_atomic_numbers=allowed_z,
            num_atom_types=len(allowed_z),
            device="cpu",
            file_limit=self._train_params.pkl_file_limit,
        )

    def train_dataloader(self):
        from torch.utils.data import DataLoader

        if self.train_dataset is None:
            self.train_dataset = self._make_dataset("train")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=PKLEnergyForceDataset.collate_fn,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self):
        from torch.utils.data import DataLoader

        if self.val_dataset is None:
            self.val_dataset = self._make_dataset("val")

        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=PKLEnergyForceDataset.collate_fn,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        from torch.utils.data import DataLoader

        if self.test_dataset is None:
            self.test_dataset = self._make_dataset("test")

        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=PKLEnergyForceDataset.collate_fn,
            pin_memory=True,
            drop_last=False,
        )

    def setup(self, stage: Optional[str] = None):
        # Mirror DrugFlow stage semantics.
        if stage in {None, "fit"}:
            self.train_dataset = self._make_dataset("train")
            self.val_dataset = self._make_dataset("val")
        elif stage == "val":
            self.val_dataset = self._make_dataset("val")
        elif stage == "test":
            self.test_dataset = self._make_dataset("test")
        elif stage == "generation":
            # sampling-only stage; lazy init size distribution
            pass
        else:
            raise NotImplementedError(f"Unknown stage: {stage}")

    def setup_sampling(self):
        # Match DrugFlow behavior: sample molecule sizes from histogram.
        histogram_file = Path(self.datadir, "size_distribution.npy")
        if not histogram_file.exists():
            # Fallback to repo default histogram.
            src_dir = Path(__file__).resolve().parents[1]
            histogram_file = src_dir / "default" / "size_distribution.npy"

        size_histogram = np.load(histogram_file).tolist()
        self.size_distribution = DistributionNodes(size_histogram)

    def parse_num_nodes_spec(self, batch: dict, *, spec, n_samples: int) -> int | torch.Tensor:
        """Parse num_nodes spec in the same spirit as DrugFlow.

        Supported:
          - None: sample from size distribution
          - int: fixed size
          - torch.Tensor: shape (n_samples,) sizes
          - 'ground_truth': use batch['ligand']['size']
          - 'uniform_L_R': uniform integer in [L, R]
        """
        if spec is None:
            if self.size_distribution is None:
                self.setup_sampling()
            assert self.size_distribution is not None
            num_nodes_lig, _ = self.size_distribution.sample(n_samples=n_samples)
            return num_nodes_lig.to(self.device)

        if isinstance(spec, int):
            return int(spec)

        if isinstance(spec, torch.Tensor):
            if spec.ndim != 1 or int(spec.numel()) != n_samples:
                raise ValueError(f"num_nodes tensor must have shape (n_samples,), got {tuple(spec.shape)}")
            return spec.to(self.device)

        if isinstance(spec, str):
            if spec == "ground_truth":
                sizes = batch["ligand"]["size"]
                if isinstance(sizes, torch.Tensor):
                    sizes = sizes.to(self.device)
                else:
                    sizes = torch.as_tensor(sizes, device=self.device)
                # sizes is shape (B,). Repeat each size n_samples times.
                return sizes.repeat_interleave(n_samples)

            if spec.startswith("uniform_"):
                parts = spec.split("_")
                if len(parts) != 3:
                    raise ValueError(f"Invalid uniform spec: {spec}")
                left = int(parts[1])
                right = int(parts[2])
                if left > right or left <= 0:
                    raise ValueError(f"Invalid uniform range in {spec}")
                return torch.randint(low=left, high=right + 1, size=(n_samples,), device=self.device)

        raise ValueError(f"Unsupported num_nodes spec: {spec}")

    def compute_loss(
        self,
        ligand,
        return_info: bool = False,
        *,
        energy: torch.Tensor | None = None,
        force: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        temperature: torch.Tensor | float | None = None,
    ):
        # Center per-molecule for stability.
        com = scatter_mean(ligand["x"], ligand["mask"], dim=0)
        ligand["x"] = ligand["x"] - com[ligand["mask"]]

        # Timestep t for each example in batch
        if t is None:
            if self.t_min == 0.0 and self.t_max == 1.0:
                t = torch.rand(ligand["size"].size(0), device=ligand["x"].device).unsqueeze(-1)
            else:
                u = torch.rand(ligand["size"].size(0), device=ligand["x"].device).unsqueeze(-1)
                t = u * (self.t_max - self.t_min) + self.t_min
        else:
            if t.ndim == 1:
                t = t.unsqueeze(-1)
            if t.shape != (ligand["size"].size(0), 1):
                raise ValueError(f"t must have shape (B,1); got {tuple(t.shape)}")
            t = t.to(device=ligand["x"].device, dtype=ligand["x"].dtype)
        t = torch.clamp(t, 0.0, 1.0)

        # Temperature conditioning (batch-level, like t)
        if not self.use_temperature:
            # Temperature conditioning disabled: do not pass temperature to the model or diffusion.
            temp_val = float(self.temperature)
            temperature = None
        else:
            if temperature is None:
                # Sample a single temperature for the whole batch during training.
                # Validation uses the fixed temperature from config.
                if self.training and (self.temperature_min != self.temperature_max):
                    temp_val = float(
                        torch.empty((), device=ligand["x"].device).uniform_(self.temperature_min, self.temperature_max).item()
                    )
                else:
                    temp_val = self.temperature
                temperature = torch.full((ligand["size"].size(0), 1), temp_val, device=ligand["x"].device, dtype=t.dtype)
            else:
                if isinstance(temperature, (int, float)):
                    temp_val = float(temperature)
                    temperature = torch.full((ligand["size"].size(0), 1), temp_val, device=ligand["x"].device, dtype=t.dtype)
                else:
                    if temperature.ndim == 1:
                        temperature = temperature.unsqueeze(-1)
                    if temperature.shape != (ligand["size"].size(0), 1):
                        raise ValueError(f"temperature must have shape (B,1); got {tuple(temperature.shape)}")
                    temperature = temperature.to(device=ligand["x"].device, dtype=t.dtype)
                    temp_val = float(temperature[0].detach().cpu().item())

        # Noise
        # Coordinates: perturb the clean sample x (t=1 clean) with sigma(t) (t=0 most noisy)
        zt_x, _ = self.module_x.sample_zt(ligand["x"], t, ligand["mask"], temperature=temperature)

        # Atom types: optionally diffuse, otherwise condition on clean one_hot.
        if self.diffuse_h and (self.lambda_h is None or float(self.lambda_h) != 0.0):
            z0_h = self.module_h.sample_z0(ligand["mask"])
            zt_h = self.module_h.sample_zt(z0_h, ligand["one_hot"], t, ligand["mask"])
        else:
            zt_h = ligand["one_hot"]

        zt_x = zt_x.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)

        pred_ligand, _ = self.dynamics(
            zt_x,
            zt_h,
            ligand["mask"],
            pocket=None,
            t=t,
            temperature=temperature,
            bonds_ligand=None,
            sc_transform=None,
        )

        # Coordinate score matching loss
        # Convention: dynamics returns `force` which we interpret as the coordinate score.

        ######## Compute coordinate score loss ########

        loss_x = self.module_x.score_loss(
            pred_ligand["force"],
            ligand["x"],
            zt_x,
            t,
            ligand["mask"],
            reduce=self.loss_reduce,
            temperature=temperature,
            weight_by_sigma=self.x_score_weight_by_sigma,
        )

        ######## Compute atom-type loss ########

        if self.diffuse_h and (self.lambda_h is None or float(self.lambda_h) != 0.0):
            t_next = torch.clamp(t + self.train_step_size, max=1.0)
            loss_h = self.module_h.compute_loss(
                pred_ligand["logits_h"],
                zt_h,
                ligand["one_hot"],
                ligand["mask"],
                t,
                t_next,
                reduce=self.loss_reduce,
            )
        else:
            loss_h = torch.zeros_like(loss_x)

        # Supervised energy head (computed on clean x/h).
        # Use t=1.0 for the energy head to represent the clean configuration.
        loss_energy = torch.zeros_like(loss_x)
        loss_hjb = torch.zeros_like(loss_x)
        loss_consistency = torch.zeros_like(loss_x)

        # compute_ef = (
        #     (energy is not None)
        #     or (force is not None)
        #     or (float(self.lambda_bc) > 0.0)
        #     or (float(self.lambda_hjb) > 0.0)
        #     or (float(self.lambda_consistency) > 0.0)
        # )
        # if compute_ef:

        ######## Compute energy match loss (boundary condition 1) ########

        if (energy is not None) and (float(self.lambda_energy) > 0.0):
            t_clean = torch.ones_like(t)
            pred_clean, _ = self.dynamics(
                ligand["x"],
                ligand["one_hot"],
                ligand["mask"],
                pocket=None,
                t=t_clean,
                temperature=temperature,
                bonds_ligand=None,
                sc_transform=None,
            )

            energy_tgt = energy.to(device=ligand["x"].device, dtype=ligand["x"].dtype).view(-1)
            energy_pred = pred_clean["energy"].to(dtype=ligand["x"].dtype).view(-1)
            if energy_pred.shape != energy_tgt.shape:
                raise ValueError(f"energy_pred must have shape (B,), got {tuple(energy_pred.shape)} vs target {tuple(energy_tgt.shape)}")
            loss_energy = (energy_pred - energy_tgt) ** 2

            print("True energy:", energy_tgt[:5].detach().cpu().numpy())
            print("Pred energy:", energy_pred[:5].detach().cpu().numpy())
            print()

        ######## HJB loss + consistency condition at the *noisy* state (zt_x, t). ########

        # This implements:
        #  - L_hjb: mean squared HJB residual (per molecule)
        #  - L_consistency: mean squared (F + ∇_x u) (per molecule)
        # Boundary condition u(x, t=1) = u_ref is handled by loss_energy above.
        if (float(self.lambda_hjb) > 0.0) or (float(self.lambda_consistency) > 0.0):
            loss_hjb, loss_consistency = self.module_x.hjb_loss(
                pred_ligand["force"].to(dtype=ligand["x"].dtype),
                pred_ligand["energy"].to(dtype=ligand["x"].dtype).view(-1),
                zt_x,
                t,
                ligand["mask"],
                temperature=temperature,
                reduce=self.loss_reduce,
            )

        loss = (
            self.lambda_x * loss_x
            + self.lambda_h * loss_h
            + self.lambda_energy * loss_energy
            + self.lambda_hjb * loss_hjb
            + self.lambda_consistency * loss_consistency
        )
        loss = loss.mean(0)

        info = {
            "loss_x": float(loss_x.mean().detach().cpu()),
            "loss_h": float(loss_h.mean().detach().cpu()),
            "loss_energy": float(loss_energy.mean().detach().cpu()),
            "loss_hjb": float(loss_hjb.mean().detach().cpu()),
            "loss_consistency": float(loss_consistency.mean().detach().cpu()),
            "temperature": float(temp_val),
            "diffuse_h": bool(self.diffuse_h and (self.lambda_h is None or float(self.lambda_h) != 0.0)),
        }

        if self.log_diffusion_stats:
            with torch.no_grad():
                sigma_b = self.module_x.sigma(t.detach(), temperature=temperature).view(-1)
                target = self.module_x.score_target(zt_x.detach(), ligand["x"].detach(), t.detach(), ligand["mask"], temperature=temperature)
                pred = pred_ligand["force"].detach()

                target_norm = torch.sqrt(torch.sum(target * target, dim=-1) + 1e-12)
                pred_norm = torch.sqrt(torch.sum(pred * pred, dim=-1) + 1e-12)
                cos = torch.sum(pred * target, dim=-1) / torch.clamp(pred_norm * target_norm, min=1e-12)

                info.update(
                    {
                        "t/mean": float(t.detach().mean().cpu()),
                        "t/min": float(t.detach().min().cpu()),
                        "t/max": float(t.detach().max().cpu()),
                        "sigma/mean": float(sigma_b.mean().cpu()),
                        "sigma/min": float(sigma_b.min().cpu()),
                        "sigma/max": float(sigma_b.max().cpu()),
                        "score_target_norm/mean": float(target_norm.mean().cpu()),
                        "score_pred_norm/mean": float(pred_norm.mean().cpu()),
                        "score_cos/mean": float(cos.mean().cpu()),
                    }
                )
        return (loss, info) if return_info else loss

    def training_step(self, batch, *args):
        loss, info = self.compute_loss(
            batch["ligand"],
            energy=batch.get("energy", None),
            force=batch.get("force", None),
            return_info=True,
        )
        self.log("loss/train", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_x/train", info["loss_x"], on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_h/train", info["loss_h"], on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_energy/train", info["loss_energy"], on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_hjb/train", info["loss_hjb"], on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log(
            "loss_consistency/train",
            info["loss_consistency"],
            on_step=True,
            on_epoch=True,
            batch_size=len(batch["ligand"]["size"]),
        )
        self.log("temperature/train", info["temperature"], on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        if self.log_diffusion_stats:
            for k in (
                "t/mean",
                "t/min",
                "t/max",
                "sigma/mean",
                "sigma/min",
                "sigma/max",
                "score_target_norm/mean",
                "score_pred_norm/mean",
                "score_cos/mean",
            ):
                if k in info:
                    self.log(f"{k}/train", info[k], on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        return loss

    def validation_step(self, batch, *args):
        loss, info = self.compute_loss(
            batch["ligand"],
            energy=batch.get("energy", None),
            force=batch.get("force", None),
            return_info=True,
        )
        self.log("loss/val", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_x/val", info["loss_x"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_h/val", info["loss_h"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_energy/val", info["loss_energy"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_hjb/val", info["loss_hjb"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log(
            "loss_consistency/val",
            info["loss_consistency"],
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["ligand"]["size"]),
        )
        self.log("temperature/val", info["temperature"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        if self.log_diffusion_stats:
            for k in (
                "t/mean",
                "t/min",
                "t/max",
                "sigma/mean",
                "sigma/min",
                "sigma/max",
                "score_target_norm/mean",
                "score_pred_norm/mean",
                "score_cos/mean",
            ):
                if k in info:
                    self.log(f"{k}/val", info[k], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        return {"loss": loss, **info}

    def test_step(self, batch, *args):
        loss, info = self.compute_loss(
            batch["ligand"],
            energy=batch.get("energy", None),
            force=batch.get("force", None),
            return_info=True,
        )
        self.log("loss/test", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_x/test", info["loss_x"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_h/test", info["loss_h"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_energy/test", info["loss_energy"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_hjb/test", info["loss_hjb"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log(
            "loss_consistency/test",
            info["loss_consistency"],
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["ligand"]["size"]),
        )
        self.log("temperature/test", info["temperature"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        return {"loss": loss, **info}

    @torch.no_grad()
    def sample(
        self,
        data: dict,
        n_samples: int,
        num_nodes=None,
        timesteps: int | None = None,
        *,
        x_sampler: str = "ode",
        h_sampler: str = "markov_bridge",
        temperature: float | None = None,
        stochastic_x: bool = False,
        stochastic_h: bool = False,
        **kwargs,
    ):
        """Framework-style sampling entrypoint.

        Returns (ligands, pockets, names) to match DrugFlow's call sites.
        For this ligand-only model, pockets is an empty list.
        """
        timesteps = int(self.n_steps if timesteps is None else timesteps)

        # Determine number of ligand nodes per sampled molecule
        # If conditioning batch has B examples, we produce B * n_samples molecules.
        batch_sizes = data["ligand"]["size"]
        if not isinstance(batch_sizes, torch.Tensor):
            batch_sizes = torch.as_tensor(batch_sizes, device=self.device)
        B = int(batch_sizes.numel())
        total = B * int(n_samples)

        # Expand num_nodes spec
        if num_nodes is None:
            num_nodes_expanded = self.parse_num_nodes_spec(data, spec=None, n_samples=total)
        elif isinstance(num_nodes, str) and num_nodes == "ground_truth":
            num_nodes_expanded = self.parse_num_nodes_spec(data, spec="ground_truth", n_samples=int(n_samples))
        else:
            # If user passes an int, apply to all total samples.
            if isinstance(num_nodes, int):
                num_nodes_expanded = int(num_nodes)
            else:
                num_nodes_expanded = self.parse_num_nodes_spec(data, spec=num_nodes, n_samples=total)

        sample_batch = self.sample_ligand(
            n_samples=total,
            num_nodes=num_nodes_expanded,
            timesteps=timesteps,
            temperature=temperature,
            x_sampler=x_sampler,
            h_sampler=h_sampler,
            stochastic_x=stochastic_x,
            stochastic_h=stochastic_h,
        )

        # Split into per-molecule structures
        coords_list = utils.batch_to_list(sample_batch["x"].detach().cpu(), sample_batch["mask"].detach().cpu())
        one_hot_list = utils.batch_to_list(sample_batch["one_hot"].detach().cpu(), sample_batch["mask"].detach().cpu())
        allowed_z = torch.tensor(self.allowed_atomic_numbers, dtype=torch.long)

        ligands = []
        for x_i, h_i in zip(coords_list, one_hot_list):
            idx = torch.argmax(h_i, dim=-1).to(torch.long)
            atomic_numbers = allowed_z[idx]
            ligands.append({
                "x": x_i,
                "one_hot": h_i,
                "atomic_numbers": atomic_numbers,
            })

        # Names: repeat each conditioning name n_samples times (if available)
        names = data["ligand"].get("name", None)
        if names is None:
            names = ["sample"] * total
        else:
            # names is a list of length B
            names = [str(n) for n in names]
            names = [n for n in names for _ in range(int(n_samples))]

        pockets: list = []
        return ligands, pockets, names

    @torch.no_grad()
    def sample_ligand(
        self,
        *,
        n_samples: int,
        num_nodes: int | torch.Tensor,
        timesteps: int | None = None,
        temperature: float | None = None,
        x_sampler: str = "ode",
        h_sampler: str = "markov_bridge",
        stochastic_x: bool = False,
        stochastic_h: bool = False,
    ):
        """Sample ligand-only (x, one_hot) with two atom-type sampling modes.

        Args:
          n_samples: number of molecules
          num_nodes: int (fixed size) or tensor (n_samples,) with per-mol sizes
          timesteps: Euler steps from t=0 -> t=1
          temperature: scalar conditioning value (if None, uses config temperature)
          x_sampler: 'ode' or 'sde'
          h_sampler: 'markov_bridge' or 'score'
          stochastic_x/h: only used when sampler uses stochastic steps
        """
        assert x_sampler in {"ode", "sde"}
        assert h_sampler in {"markov_bridge", "score"}

        device = self.device
        dtype = next(self.parameters()).dtype
        timesteps = int(timesteps or self.n_steps)
        if timesteps <= 0:
            raise ValueError("timesteps must be > 0")

        batch_mask = num_nodes_to_batch_mask(n_samples, num_nodes, device=device)
        if batch_mask.numel() == 0:
            raise ValueError("num_nodes produced empty batch_mask")

        sizes = torch.unique(batch_mask, return_counts=True)[1]

        if not self.use_temperature:
            t_temperature = None
        else:
            if temperature is None:
                temperature = float(self.temperature)
            t_temperature = torch.full((n_samples, 1), float(temperature), device=device, dtype=dtype)

        # Initialize x at t=0 (noisy prior). Use COM=0 for each molecule.
        com = torch.zeros((n_samples, 3), device=device, dtype=dtype)
        x = self.module_x.sample_z1(com, batch_mask, temperature=t_temperature)

        # Initialize atoms.
        if h_sampler == "markov_bridge":
            h = self.module_h.sample_z0(batch_mask)  # one-hot
            h_logits = None
        else:
            h_logits = self.module_h_score.sample_prior(batch_mask, temperature=t_temperature, dtype=dtype)
            h = self.module_h_score.logits_to_probs(h_logits)

        dt = 1.0 / float(timesteps)
        for i in range(timesteps):
            s_val = i * dt
            t_val = (i + 1) * dt
            s = torch.full((n_samples, 1), s_val, device=device, dtype=dtype)
            t = torch.full((n_samples, 1), t_val, device=device, dtype=dtype)

            pred_ligand, _ = self.dynamics(
                x,
                h,
                batch_mask,
                pocket=None,
                t=s,
                temperature=t_temperature,
                bonds_ligand=None,
                sc_transform=None,
            )

            # Coordinates: interpret dynamics 'force' as score in x-space.
            if x_sampler == "ode":
                x = self.module_x.ode_step(x, pred_ligand["force"], s=s, t=t, batch_mask=batch_mask, temperature=t_temperature)
            else:
                x = self.module_x.reverse_step(
                    x,
                    pred_ligand["force"],
                    s=s,
                    t=t,
                    batch_mask=batch_mask,
                    temperature=t_temperature,
                    stochastic=stochastic_x,
                )

            # Enforce translation invariance (match training which centers per molecule).
            com = scatter_mean(x, batch_mask, dim=0)
            x = x - com[batch_mask]

            # Atom types.
            if h_sampler == "markov_bridge":
                h = self.module_h.sample_zt_given_zs(
                    h,
                    pred_ligand["logits_h"],
                    s=s,
                    t=t,
                    batch_mask=batch_mask,
                )
            else:
                # Interpret dynamics logits_h as score in logit-space.
                assert h_logits is not None
                if x_sampler == "ode":
                    h_logits = self.module_h_score.ode_step(
                        h_logits,
                        pred_ligand["logits_h"],
                        s=s,
                        t=t,
                        batch_mask=batch_mask,
                        temperature=t_temperature,
                    )
                else:
                    h_logits = self.module_h_score.reverse_step(
                        h_logits,
                        pred_ligand["logits_h"],
                        s=s,
                        t=t,
                        batch_mask=batch_mask,
                        temperature=t_temperature,
                        stochastic=stochastic_h,
                    )
                h = self.module_h_score.logits_to_probs(h_logits)

        # Finalize categorical sample
        if h_sampler == "score":
            idx = torch.argmax(h, dim=-1)
            h = torch.nn.functional.one_hot(idx, num_classes=self.atom_nf).to(h.dtype)

        return {
            "x": x,
            "one_hot": h,
            "mask": batch_mask,
            "size": sizes,
        }
