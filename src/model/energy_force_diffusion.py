from __future__ import annotations

import os
os.environ["WANDB_SERVICE_WAIT"] = "300"

from argparse import Namespace
from typing import Literal, cast
from pathlib import Path
from typing import Optional, Union
import warnings
from contextlib import ExitStack, nullcontext

import numpy as np
import pytorch_lightning as pl
import torch
from torch_scatter import scatter_add, scatter_mean

from src.data.pkl_dataset import PKLEnergyForceDataset, PKLKeys, split_pkl_file_paths
from src.data.ani_h5_dataset import make_ani_h5_dataset_for_stage
from src.model.diffusion_utils import DistributionNodes
from src.model.dynamics_radius_gvp import RadiusGVPDynamics, RadiusGVPParams
from src.model.dynamics_radius_mlp import RadiusMLPDynamics, RadiusMLPParams
from src.model.dynamics_scorenet_mlp import ScoreNetMLPDynamics, ScoreNetMLPParams
from src.model.flows import CategoricalLogitScoreDiffusion, CoordScoreDiffusion, SDEParams
from src.model.markov_bridge import UniformPriorMarkovBridge
from src import utils
from src.utils import num_nodes_to_batch_mask


def set_default(namespace: Namespace, key: str, default_val):
    val = vars(namespace).get(key, default_val)
    setattr(namespace, key, val)


BOLTZMANN_CONSTANT_EV_PER_K = 8.617333262145e-5


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
        set_default(train_params, "dataset_stats", None)
        set_default(train_params, "pretrain_diff_epochs", 0)
        set_default(loss_params, "lambda_x", 1.0)
        set_default(loss_params, "lambda_h", 1.0)
        set_default(loss_params, "lambda_energy", 1.0)
        set_default(loss_params, "lambda_force_dsm", 1.0)
        set_default(loss_params, "lambda_energy_dsm", 1.0)
        set_default(loss_params, "use_gamma_scaling", True)
        set_default(loss_params, "gamma_envelope_eps", 1.0e-2)
        # Boundary condition at t=0: match predicted force (-∇E) to true forces.
        set_default(loss_params, "lambda_force_t0", 0.0)
        # Optional boundary condition at t=0: match energy-gradient force -dU/dx to true forces.
        set_default(loss_params, "lambda_force_energy_t0", 0.0)
        # Optional conditional flow-matching (CFM) velocity loss (VE kinematics).
        set_default(loss_params, "lambda_cfm", 0.0)
        # Optional HJB PDE loss + force/energy consistency (at noisy time)
        set_default(loss_params, "lambda_hjb", 0.0)
        set_default(loss_params, "lambda_consistency", 0.0)
        set_default(loss_params, "hjb_use_target_force_fm", False)
        set_default(loss_params, "consistency_weight_by_snr", False)
        set_default(loss_params, "consistency_use_min_snr", True)
        set_default(loss_params, "consistency_min_snr_gamma", 5.0)
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
        self.pretrain_diff_epochs = int(getattr(train_params, "pretrain_diff_epochs", 0))

        # UMA-style target scaling + (optional) energy referencing.
        self.target_sigma, self.omol_elem_ref = utils.parse_dataset_stats(getattr(train_params, "dataset_stats", None))

        # Optional diffusion-output thermodynamic scaling.
        # Old target-side U/(k_B T_ref) scaling is retired; when enabled now, only the
        # diffusion-model outputs are multiplied by k_B T_ref to enter physical units.
        # This fixed thermo reference temperature is also the single temperature reported for
        # scaling when temperature conditioning itself is commented out.
        set_default(train_params, "scale_diffusion_by_thermo", False)
        set_default(train_params, "thermo_reference_temperature_K", 300.0)
        self.scale_diffusion_by_thermo = bool(getattr(train_params, "scale_diffusion_by_thermo", False))
        self.thermo_reference_temperature_K = float(getattr(train_params, "thermo_reference_temperature_K", 300.0))
        if self.scale_diffusion_by_thermo:
            if not np.isfinite(self.thermo_reference_temperature_K) or self.thermo_reference_temperature_K <= 0.0:
                raise ValueError(
                    "train_params.thermo_reference_temperature_K must be a positive finite float when "
                    "train_params.scale_diffusion_by_thermo=True"
                )
            self.thermo_kbt_ev = float(BOLTZMANN_CONSTANT_EV_PER_K * self.thermo_reference_temperature_K)
            if not np.isfinite(self.thermo_kbt_ev) or self.thermo_kbt_ev <= 0.0:
                raise ValueError(
                    "Computed k_B T_ref must be a positive finite float; got "
                    f"{self.thermo_kbt_ev} eV from T_ref={self.thermo_reference_temperature_K} K"
                )
        else:
            self.thermo_kbt_ev = 1.0

        # Loss params
        self.loss_reduce = loss_params.reduce
        self.lambda_x = loss_params.lambda_x
        self.lambda_h = loss_params.lambda_h
        self.lambda_energy = loss_params.lambda_energy
        self.lambda_force_dsm = float(getattr(loss_params, "lambda_force_dsm", 1.0))
        self.lambda_energy_dsm = float(getattr(loss_params, "lambda_energy_dsm", 1.0))
        self.use_gamma_scaling = bool(getattr(loss_params, "use_gamma_scaling", True))
        self.gamma_envelope_eps = float(getattr(loss_params, "gamma_envelope_eps", 1.0e-2))
        self.lambda_force_t0 = float(getattr(loss_params, "lambda_force_t0", 0.0))
        self.lambda_force_energy_t0 = float(getattr(loss_params, "lambda_force_energy_t0", 0.0))
        # Fraction of graphs in each batch forced to t=0 (used for boundary energy/force losses).
        # The remaining graphs keep their sampled diffusion times.
        set_default(loss_params, "t0_batch_fraction", 1.0)
        self.t0_batch_fraction = float(getattr(loss_params, "t0_batch_fraction", 1.0))

        # Avoid conditioning exactly at t=0 for the boundary subset by using a small epsilon time.
        # This keeps coordinates clean (we still override zt_x for the boundary graphs) but can
        # reduce endpoint numerical issues in time embeddings / schedules.
        # Set to 0.0 to recover exact t=0 conditioning.
        set_default(loss_params, "t0_time_epsilon", 1.0e-3)
        self.t0_time_epsilon = float(getattr(loss_params, "t0_time_epsilon", 1.0e-3))
        self.lambda_cfm = loss_params.lambda_cfm
        self.lambda_hjb = loss_params.lambda_hjb
        self.lambda_consistency = loss_params.lambda_consistency
        self.hjb_use_target_force_fm = bool(getattr(loss_params, "hjb_use_target_force_fm", False))
        self.consistency_weight_by_snr = bool(getattr(loss_params, "consistency_weight_by_snr", False))
        self.consistency_use_min_snr = bool(getattr(loss_params, "consistency_use_min_snr", True))
        self.consistency_min_snr_gamma = float(getattr(loss_params, "consistency_min_snr_gamma", 5.0))

        # Diffusion params
        self.n_steps = simulation_params.n_steps
        self.train_step_size = 1.0 / float(self.n_steps)

        # Timestep sampling range in diffusion time (t=0 clean -> t=1 noisy).
        # Useful to avoid very-noisy regimes early in training.
        set_default(simulation_params, "t_min", 0.0)
        set_default(simulation_params, "t_max", 1.0)

        # Optional timestep curriculum: ramp up the max sampled diffusion time over epochs.
        # Mirrors notebooks/test.ipynb (Cell 15 + Cell 21): warmup -> sinusoidal growth -> plateau.
        set_default(simulation_params, "time_horizon_schedule", "constant")  # 'constant' | 'sinusoidal'
        set_default(simulation_params, "time_horizon_warmup_fraction", 0.1)
        set_default(simulation_params, "time_horizon_plateau_start_fraction", 0.8)
        set_default(simulation_params, "time_horizon_t_min", 0.01)
        set_default(simulation_params, "time_horizon_t_final", 1.0)
        self.t_min = float(simulation_params.t_min)
        self.t_max = float(simulation_params.t_max)
        if not (0.0 <= self.t_min <= self.t_max <= 1.0):
            raise ValueError(f"simulation_params.t_min/t_max must satisfy 0<=t_min<=t_max<=1; got {self.t_min}, {self.t_max}")

        self.time_horizon_schedule = str(getattr(simulation_params, "time_horizon_schedule", "constant"))
        self.time_horizon_warmup_fraction = float(getattr(simulation_params, "time_horizon_warmup_fraction", 0.1))
        self.time_horizon_plateau_start_fraction = float(getattr(simulation_params, "time_horizon_plateau_start_fraction", 0.8))
        self.time_horizon_t_min = float(getattr(simulation_params, "time_horizon_t_min", 0.01))
        self.time_horizon_t_final = float(getattr(simulation_params, "time_horizon_t_final", 1.0))
        if self.time_horizon_schedule not in {"constant", "sinusoidal"}:
            raise ValueError("simulation_params.time_horizon_schedule must be one of: constant, sinusoidal")
        if not (0.0 <= self.time_horizon_t_min <= self.time_horizon_t_final <= 1.0):
            raise ValueError(
                "simulation_params.time_horizon_t_min/t_final must satisfy 0<=t_min<=t_final<=1; "
                f"got {self.time_horizon_t_min}, {self.time_horizon_t_final}"
            )

        # Optional training diagnostics
        set_default(train_params, "log_diffusion_stats", False)
        self.log_diffusion_stats = bool(train_params.log_diffusion_stats)

        # Option to disable atom-type diffusion (train with diffusion on x only).
        set_default(simulation_params, "diffuse_h", True)
        self.diffuse_h = bool(simulation_params.diffuse_h)

        # Temperature conditioning is optional and currently disabled by default.
        # Boltzmann scaling uses train_params.thermo_reference_temperature_K separately from
        # any model-side temperature conditioning.
        # self.use_temperature = bool(getattr(predictor_params, "condition_temperature", True))
        self.use_temperature = bool(getattr(predictor_params, "condition_temperature", False))
        set_default(simulation_params, "use_temperature", self.use_temperature)
        if bool(getattr(simulation_params, "use_temperature")) != self.use_temperature:
            raise ValueError(
                "simulation_params.use_temperature must match predictor_params.condition_temperature "
                "(temperature conditioning is an all-or-nothing feature)."
            )

        # Temperature conditioning (batch-level, like t). In the current setup we keep only
        # a single fixed scalar for logging/debugging, derived from thermo_reference_temperature_K
        # whenever diffusion-output Boltzmann scaling is enabled.
        set_default(simulation_params, "temperature", self.thermo_reference_temperature_K)
        # set_default(simulation_params, "temperature_min", float(simulation_params.temperature))
        # set_default(simulation_params, "temperature_max", float(simulation_params.temperature))
        self.temperature = float(simulation_params.temperature)
        # self.temperature_min = float(simulation_params.temperature_min)
        # self.temperature_max = float(simulation_params.temperature_max)
        if not self.use_temperature:
            # Ensure we never introduce random temperatures when conditioning is disabled.
            if self.scale_diffusion_by_thermo:
                self.temperature = float(self.thermo_reference_temperature_K)
            # self.temperature_min = self.temperature
            # self.temperature_max = self.temperature

        # Atom vocabulary for this dataset
        self.allowed_atomic_numbers = list(train_params.pkl_allowed_atomic_numbers)
        self.atom_nf = len(self.allowed_atomic_numbers)

        # Coordinate dimension (default 3 for molecules; allow 2D toy tasks like swiss roll).
        set_default(simulation_params, "x_dim", 3)
        self.x_dim = int(getattr(simulation_params, "x_dim", 3))
        if self.x_dim < 2:
            raise ValueError(f"simulation_params.x_dim must be >= 2; got {self.x_dim}")

        # Modules
        # Score-based diffusion on coordinates (diffusion time: t=0 clean, t=1 noisy)
        set_default(simulation_params, "sigma_min", 0.01)
        set_default(simulation_params, "sigma_max", 50.0)
        set_default(simulation_params, "coord_sde_kind", "ve")
        set_default(simulation_params, "vp_beta_min", 0.1)
        set_default(simulation_params, "vp_beta_max", 20.0)
        set_default(simulation_params, "vp_sigma_scale", 1.0)
        # Centering (translation invariance) for molecular data.
        # NOTE: For toy tasks where absolute position matters (e.g., 2D energy landscapes),
        # you may want to disable this via simulation_params.center_ligand=False.
        set_default(simulation_params, "center_ligand", True)
        self.center_ligand = bool(getattr(simulation_params, "center_ligand", True))
        set_default(simulation_params, "enforce_zero_com_noise", False)
        # Renamed: predicted coordinate vector is velocity `v` (was `force` in score-matching).
        # Keep backward compatibility with older configs.
        if getattr(simulation_params, "project_predicted_v_zero_com", None) is None:
            legacy = getattr(simulation_params, "project_predicted_force_zero_com", None)
            if legacy is not None:
                setattr(simulation_params, "project_predicted_v_zero_com", bool(legacy))
        set_default(simulation_params, "project_predicted_v_zero_com", False)
        set_default(loss_params, "x_score_weight_by_sigma", False)
        set_default(loss_params, "use_min_snr", False)
        set_default(loss_params, "min_snr_gamma", 5.0)

        # Optional SNR weighting for the conditional flow-matching (CFM) velocity loss.
        # For VE (alpha=1), define SNR(t) = 1 / sigma(t)^2.
        # When enabled, we weight per-node squared error by SNR(t) (optionally clamped via Min-SNR).
        set_default(loss_params, "cfm_weight_by_snr", False)
        set_default(loss_params, "cfm_use_min_snr", bool(loss_params.use_min_snr))
        set_default(loss_params, "cfm_min_snr_gamma", float(loss_params.min_snr_gamma))
        # HJB settings (divergence estimator + higher-order autodiff)
        # set_default(loss_params, "hjb_divergence", "hutchinson")
        # set_default(loss_params, "hjb_trace_samples", 1)
        # set_default(loss_params, "hjb_enable_higher_order", True)
        # set_default(loss_params, "hjb_divergence_create_graph", True)

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
            project_predicted_score_zero_com=bool(simulation_params.project_predicted_v_zero_com),
        )
        self.module_h = UniformPriorMarkovBridge(self.atom_nf, loss_type=loss_params.discrete_loss)
        self.x_score_weight_by_sigma = bool(loss_params.x_score_weight_by_sigma)
        self.use_min_snr = bool(loss_params.use_min_snr)
        self.min_snr_gamma = float(loss_params.min_snr_gamma)

        self.cfm_weight_by_snr = bool(loss_params.cfm_weight_by_snr)
        self.cfm_use_min_snr = bool(loss_params.cfm_use_min_snr)
        self.cfm_min_snr_gamma = float(loss_params.cfm_min_snr_gamma)

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

        self.dynamics_diff, self.condition_time = self._build_dynamics_module(predictor_params)
        self.dynamics_corr, condition_time_corr = self._build_dynamics_module(predictor_params)
        if bool(self.condition_time) != bool(condition_time_corr):
            raise ValueError("Diffusion and correction predictors must agree on condition_time")

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

        # HJB configuration
        self.hjb_divergence = str(getattr(loss_params, "hjb_divergence", "hutchinson"))
        self.hjb_trace_samples = int(getattr(loss_params, "hjb_trace_samples", 1))
        self.hjb_enable_higher_order = bool(getattr(loss_params, "hjb_enable_higher_order", True))
        self.hjb_divergence_create_graph = bool(getattr(loss_params, "hjb_divergence_create_graph", True))
        self.hjb_trace_batch_size = int(getattr(loss_params, "hjb_trace_batch_size", 1))
        self.hjb_debug = bool(getattr(loss_params, "hjb_debug", False))
        if self.hjb_divergence not in {"exact", "hutchinson"}:
            raise ValueError("loss_params.hjb_divergence must be one of: exact, hutchinson")
        if self.hjb_trace_samples <= 0:
            raise ValueError("loss_params.hjb_trace_samples must be >= 1")

        self._active_training_phase: str | None = None
        self._apply_training_phase(self._phase_name_for_epoch(0))

    def _build_dynamics_module(self, predictor_params: Namespace):
        backbone = getattr(predictor_params, "backbone", "radius_mlp")
        if backbone == "gvp":
            set_default(predictor_params, "force_fm_scale", 1.0)
            set_default(predictor_params, "force_corr_scale", 1.0)
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
                force_fm_scale=float(getattr(predictor_params, "force_fm_scale", 1.0)),
                force_corr_scale=float(getattr(predictor_params, "force_corr_scale", 1.0)),
                force_corr_schedule=bool(getattr(predictor_params, "force_corr_schedule", False)),
            )
            return RadiusGVPDynamics(atom_nf=self.atom_nf, x_dim=self.x_dim, params=params), bool(getattr(predictor_params, "condition_time", True))

        if backbone == "radius_mlp":
            radius_mlp = getattr(predictor_params, "radius_mlp", Namespace())
            hidden_dim = getattr(radius_mlp, "hidden_dim", 256)
            n_layers = getattr(radius_mlp, "n_layers", 3)
            condition_time = getattr(radius_mlp, "condition_time", True)
            params = RadiusMLPParams(
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                condition_time=condition_time,
                condition_temperature=self.use_temperature,
            )
            return RadiusMLPDynamics(atom_nf=self.atom_nf, x_dim=self.x_dim, params=params), bool(condition_time)

        if backbone == "scorenet_mlp":
            scorenet_mlp = getattr(predictor_params, "scorenet_mlp", Namespace())
            hidden_dim = int(getattr(scorenet_mlp, "hidden_dim", 128))
            time_emb_dim = int(getattr(scorenet_mlp, "time_emb_dim", 64))
            num_layers = int(getattr(scorenet_mlp, "num_layers", 3))
            condition_time = bool(getattr(scorenet_mlp, "condition_time", True))

            set_default(predictor_params, "force_fm_scale", float(getattr(scorenet_mlp, "force_fm_scale", 1.0)))
            set_default(predictor_params, "force_corr_scale", float(getattr(scorenet_mlp, "force_corr_scale", 1.0)))
            set_default(predictor_params, "force_corr_schedule", bool(getattr(scorenet_mlp, "force_corr_schedule", False)))

            params = ScoreNetMLPParams(
                input_dim=2,
                hidden_dim=hidden_dim,
                time_emb_dim=time_emb_dim,
                num_layers=num_layers,
                condition_time=condition_time,
                condition_temperature=bool(self.use_temperature),
                force_fm_scale=float(getattr(predictor_params, "force_fm_scale", 1.0)),
                force_corr_scale=float(getattr(predictor_params, "force_corr_scale", 1.0)),
                force_corr_schedule=bool(getattr(predictor_params, "force_corr_schedule", False)),
            )
            return ScoreNetMLPDynamics(atom_nf=self.atom_nf, x_dim=self.x_dim, params=params), bool(condition_time)

        raise ValueError(f"Unknown predictor_params.backbone: {backbone}")

    @staticmethod
    def _set_module_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
        for param in module.parameters():
            param.requires_grad = enabled

    def _phase_name_for_epoch(self, epoch: int | None = None) -> str:
        epoch_idx = int(self.current_epoch if epoch is None else epoch)
        return "pretrain_diff" if epoch_idx < self.pretrain_diff_epochs else "correction"

    def _apply_training_phase(self, phase: str) -> None:
        if phase == self._active_training_phase:
            return
        if phase == "pretrain_diff":
            self._set_module_requires_grad(self.dynamics_diff, True)
            self._set_module_requires_grad(self.dynamics_corr, False)
        elif phase == "correction":
            self._set_module_requires_grad(self.dynamics_diff, False)
            self._set_module_requires_grad(self.dynamics_corr, True)
        else:
            raise ValueError(f"Unknown training phase: {phase}")
        self._active_training_phase = phase

    def _gamma_envelope(self, t: torch.Tensor, temperature: torch.Tensor | float | None = None) -> torch.Tensor:
        if not self.use_gamma_scaling:
            return torch.ones_like(t)
        tau = 1.0 - torch.clamp(t, 0.0, 1.0)
        if isinstance(temperature, (int, float)):
            temperature_t = torch.full_like(t, float(temperature))
        else:
            temperature_t = temperature
        sigma = self.module_x.sigma(tau, temperature=temperature_t)
        sigma2 = sigma * sigma
        return sigma2 / torch.clamp(sigma2 + float(self.gamma_envelope_eps), min=1e-12)

    def _scale_diffusion_outputs_to_physical_units(self, x: torch.Tensor) -> torch.Tensor:
        if not self.scale_diffusion_by_thermo:
            return x
        return x * float(self.thermo_kbt_ev)

    def _unscale_physical_outputs_to_score_units(self, x: torch.Tensor) -> torch.Tensor:
        if not self.scale_diffusion_by_thermo:
            return x
        return x / float(self.thermo_kbt_ev)

    def _combine_predictions(
        self,
        pred_diff: dict,
        pred_corr: dict | None,
        batch_mask: torch.Tensor,
        t: torch.Tensor,
        temperature: torch.Tensor | float | None = None,
    ) -> dict:
        energy_diff_raw = pred_diff["energy"]
        force_diff_raw = pred_diff["force"]
        energy_diff_physical = self._scale_diffusion_outputs_to_physical_units(energy_diff_raw)
        force_diff_physical = self._scale_diffusion_outputs_to_physical_units(force_diff_raw)

        if pred_corr is None:
            gamma_graph = torch.ones((t.size(0), 1), device=t.device, dtype=t.dtype)
            gamma_node = gamma_graph[batch_mask]
            return {
                **pred_diff,
                "energy": energy_diff_physical,
                "force": force_diff_physical,
                "energy_diff": energy_diff_raw,
                "energy_diff_physical": energy_diff_physical,
                "energy_corr": torch.zeros_like(energy_diff_raw),
                "force_diff": force_diff_raw,
                "force_diff_physical": force_diff_physical,
                "force_corr_model": torch.zeros_like(force_diff_raw),
                "gamma_graph": gamma_graph,
                "gamma_node": gamma_node,
            }

        gamma_graph = self._gamma_envelope(t, temperature=temperature).to(dtype=pred_diff["energy"].dtype)
        gamma_node = gamma_graph[batch_mask].to(dtype=force_diff_raw.dtype)
        return {
            **pred_diff,
            "energy": gamma_graph.view(-1) * energy_diff_physical.view(-1) + pred_corr["energy"].view(-1),
            "force": gamma_node * force_diff_physical + pred_corr["force"],
            "logits_h": pred_diff.get("logits_h", pred_corr.get("logits_h")),
            "energy_diff": energy_diff_raw,
            "energy_diff_physical": energy_diff_physical,
            "energy_corr": pred_corr["energy"],
            "force_diff": force_diff_raw,
            "force_diff_physical": force_diff_physical,
            "force_corr_model": pred_corr["force"],
            "gamma_graph": gamma_graph,
            "gamma_node": gamma_node,
        }

    def _forward_model_components(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        batch_mask: torch.Tensor,
        t: torch.Tensor,
        temperature: torch.Tensor | float | None = None,
        *,
        correction_active: bool,
    ) -> tuple[dict, dict | None, dict]:
        pred_diff, _ = self.dynamics_diff(
            x,
            h,
            batch_mask,
            pocket=None,
            t=t,
            temperature=temperature,
            bonds_ligand=None,
            sc_transform=None,
        )
        pred_corr = None
        if correction_active:
            pred_corr, _ = self.dynamics_corr(
                x,
                h,
                batch_mask,
                pocket=None,
                t=t,
                temperature=temperature,
                bonds_ligand=None,
                sc_transform=None,
            )
        pred_total = self._combine_predictions(pred_diff, pred_corr, batch_mask, t, temperature=temperature)
        return pred_diff, pred_corr, pred_total

    def on_train_epoch_start(self) -> None:
        self._apply_training_phase(self._phase_name_for_epoch())

    def on_validation_epoch_start(self) -> None:
        self._apply_training_phase(self._phase_name_for_epoch())

    def on_test_epoch_start(self) -> None:
        self._apply_training_phase(self._phase_name_for_epoch())

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

    def _time_horizon_t_max_for_step(self, step: int, total_steps: int) -> float:
        """Sinusoidal curriculum for the maximum sampled diffusion time.

        Matches the notebook schedule but interprets warmup/plateau fractions over
        *batch steps* (total_steps = n_epochs * n_batches_per_epoch).
        """
        if total_steps <= 0:
            return float(self.time_horizon_t_final)

        warmup_steps = int(total_steps * float(self.time_horizon_warmup_fraction))
        plateau_steps = int(total_steps * float(self.time_horizon_plateau_start_fraction))
        if plateau_steps <= warmup_steps:
            plateau_steps = warmup_steps + 1

        t_min = float(self.time_horizon_t_min)
        t_final = float(self.time_horizon_t_final)

        if step < warmup_steps:
            return t_min
        if step >= plateau_steps:
            return t_final

        current_step = step - warmup_steps
        total_growth_steps = plateau_steps - warmup_steps
        progress = float(current_step) / float(total_growth_steps)
        sine_factor = 0.5 * (1.0 - float(np.cos(progress * float(np.pi))))
        return t_min + (t_final - t_min) * sine_factor

    def _make_dataset(self, stage: str):
        datadir = Path(self.datadir)

        dataset_format = str(getattr(self._train_params, "dataset_format", "pkl")).lower()
        if dataset_format == "auto":
            dataset_format = "ani_h5" if (datadir.is_file() and datadir.suffix.lower() in {".h5", ".hdf5"}) else "pkl"

        if dataset_format == "ani_h5":
            allowed_z = list(self._train_params.pkl_allowed_atomic_numbers)
            return make_ani_h5_dataset_for_stage(
                datadir,
                stage=stage,
                allowed_atomic_numbers=allowed_z,
                num_atom_types=len(allowed_z),
                val_fraction=float(self._train_params.pkl_val_fraction),
                seed=int(self._train_params.pkl_split_seed),
                val_n=self._train_params.pkl_val_n,
                atomic_numbers_key=str(getattr(self._train_params, "h5_atomic_numbers_key", "atomic_numbers")),
                coordinates_key=str(getattr(self._train_params, "h5_coordinates_key", "coordinates")),
                energy_key=str(getattr(self._train_params, "h5_energy_key", "wb97x_dz.energy")),
                forces_key=str(getattr(self._train_params, "h5_forces_key", "wb97x_dz.forces")),
                filter_nan=bool(getattr(self._train_params, "h5_filter_nan", True)),
                scan_limit_groups=getattr(self._train_params, "h5_scan_limit_groups", None),
                scan_chunk_size=int(getattr(self._train_params, "h5_scan_chunk_size", 128)),
                conformation_limit=getattr(self._train_params, "h5_conformation_limit", None),
                device="cpu",
            )

        if dataset_format != "pkl":
            raise ValueError("train_params.dataset_format must be one of: pkl, ani_h5, auto")

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
            collate_fn=type(self.train_dataset).collate_fn,
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
            collate_fn=type(self.val_dataset).collate_fn,
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
            collate_fn=type(self.test_dataset).collate_fn,
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
        datadir = Path(self.datadir)
        base_dir = datadir.parent if datadir.is_file() else datadir
        histogram_file = base_dir / "size_distribution.npy"
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
        step: int | None = None,
        total_steps: int | None = None,
    ):
        # Center per-molecule for stability / translation invariance (molecular setting).
        # For 1-node toy graphs this would collapse all inputs to zero, so we make it optional.
        if getattr(self, "center_ligand", True):
            com = scatter_mean(ligand["x"], ligand["mask"], dim=0)
            ligand["x"] = ligand["x"] - com[ligand["mask"]]

        # Timestep t for each example in batch
        if t is None:
            t_low = float(self.t_min)
            t_high = float(self.t_max)

            if t_high < t_low:
                t_high = t_low
            u = torch.rand(ligand["size"].size(0), device=ligand["x"].device).unsqueeze(-1)
            t = u * (t_high - t_low) + t_low
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
                # Current setup uses a single fixed conditioning temperature.
                # Old range-based behavior is intentionally commented out:
                # if self.training and (self.temperature_min != self.temperature_max):
                #     temp_val = float(
                #         torch.empty((), device=ligand["x"].device).uniform_(self.temperature_min, self.temperature_max).item()
                #     )
                # else:
                #     temp_val = self.temperature
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

        phase_name = self._phase_name_for_epoch()
        correction_active = phase_name == "correction"

        need_x = float(self.lambda_x) > 0.0
        need_h = float(self.lambda_h) > 0.0 and bool(self.diffuse_h)
        need_cfm = float(getattr(self, "lambda_cfm", 0.0)) > 0.0
        need_force_dsm = float(self.lambda_force_dsm) > 0.0
        need_energy_dsm = float(self.lambda_energy_dsm) > 0.0
        need_hjb = correction_active and float(self.lambda_hjb) > 0.0
        need_consistency = correction_active and float(self.lambda_consistency) > 0.0

        # Boundary conditions at the clean endpoint (flow-time t=1).
        need_energy_t0 = correction_active and (energy is not None) and (float(self.lambda_energy) > 0.0)
        need_force_t0 = correction_active and (force is not None) and (float(getattr(self, "lambda_force_t0", 0.0)) > 0.0)
        need_force_energy_t0 = correction_active and (force is not None) and (float(self.lambda_force_energy_t0) > 0.0)

        need_main_pass = (
            need_x
            or need_h
            or need_cfm
            or need_force_dsm
            or need_energy_dsm
            or need_hjb
            or need_consistency
            or need_energy_t0
            or need_force_t0
            or need_force_energy_t0
        )

        # Apply a fixed fraction of graphs at t≈1 (for boundary losses), keep the rest at their sampled t.
        # This reduces cost by computing du/dx once and reusing it for both boundary force loss and HJB.
        if need_main_pass:
            B = int(ligand["size"].size(0))
            if B <= 0:
                raise ValueError("Empty batch: ligand['size'] has zero length")

            t_used = t
            is_t0_graph = torch.zeros((B,), device=ligand["x"].device, dtype=torch.bool)
            if need_energy_t0 or need_force_t0 or need_force_energy_t0:
                # Clean-boundary graph fraction schedule (legacy name: t0_*):
                #  - Start at 100% early in training.
                #  - Linearly decay to `t0_batch_fraction` until `time_horizon_warmup_fraction` of training.
                #  - Plateau at `t0_batch_fraction` afterwards.
                frac_target = float(getattr(self, "t0_batch_fraction", 1.0))
                frac_target = float(np.clip(frac_target, 0.0, 1.0))
                frac_used = frac_target

                if self.training and frac_target < 1.0:
                    warmup_frac = float(getattr(self, "time_horizon_warmup_fraction", 0.0))
                    warmup_frac = float(np.clip(warmup_frac, 0.0, 1.0))

                    if warmup_frac > 0.0:
                        step_i: int | None = int(step) if step is not None else None
                        total_i: int | None = int(total_steps) if total_steps is not None else None

                        # Best-effort fallback when caller doesn't provide step/total_steps.
                        if (step_i is None) or (total_i is None) or (total_i <= 0):
                            step_i = int(getattr(self, "global_step", 0) or 0)
                            trainer = getattr(self, "trainer", None)
                            total_i = int(getattr(trainer, "estimated_stepping_batches", 0) or 0) if trainer is not None else 0

                        if (total_i is not None) and (total_i > 0) and (step_i is not None):
                            progress = float(step_i) / float(total_i)
                            progress = float(np.clip(progress, 0.0, 1.0))
                            if progress < warmup_frac:
                                ramp = progress / warmup_frac
                                frac_used = 1.0 - (1.0 - frac_target) * float(ramp)
                            else:
                                frac_used = frac_target

                frac_used = float(np.clip(frac_used, 0.0, 1.0))
                n_t0 = int(round(frac_used * B))
                if n_t0 > 0:
                    # Random subset each step during training; deterministic (first n) during eval.
                    if self.training:
                        idx = torch.randperm(B, device=ligand["x"].device)[:n_t0]
                    else:
                        idx = torch.arange(n_t0, device=ligand["x"].device)
                    is_t0_graph[idx] = True
                    t_used = t.clone()
                    t0_eps = float(np.clip(float(getattr(self, "t0_time_epsilon", 0.0)), 0.0, 1.0))
                    # Enforce the *clean* boundary at flow-time t=1 (or t=1-eps for numerical stability).
                    t_used[is_t0_graph] = 1.0 - t0_eps

            # Coordinates: flow-time convention is t=0 most noisy -> t=1 clean.
            # CoordScoreDiffusion utilities are parameterized in *diffusion forward-time*
            # tau=0 clean -> tau=1 most noisy, so we use tau = 1 - t.
            tau_used = 1.0 - t_used
            zt_x, eps_x = self.module_x.sample_zt(ligand["x"], tau_used, ligand["mask"], temperature=temperature)
            if (need_energy_t0 or need_force_t0 or need_force_energy_t0) and bool(is_t0_graph.any()):
                is_t0_node = is_t0_graph[ligand["mask"]]
                zt_x = zt_x.clone()
                zt_x[is_t0_node] = ligand["x"][is_t0_node]
                if eps_x is not None:
                    eps_x = eps_x.clone()
                    eps_x[is_t0_node] = 0.0

            # Atom types: optionally diffuse, otherwise condition on clean one_hot.
            if self.diffuse_h and (self.lambda_h is None or float(self.lambda_h) != 0.0):
                z0_h = self.module_h.sample_z0(ligand["mask"])
                # Markov-bridge time convention is 0 noisy -> 1 clean, same as flow-time t.
                zt_h = self.module_h.sample_zt(z0_h, ligand["one_hot"], t_used, ligand["mask"])
            else:
                zt_h = ligand["one_hot"]
        else:
            zt_x = None
            zt_h = None
            eps_x = None
            t_used = t
            is_t0_graph = None

        # Lightning runs validation/test under torch.no_grad(), so we selectively re-enable
        # autograd for the energy-gradient and HJB terms.
        need_time_space_grads = (
            need_hjb or need_x or need_consistency or need_energy_dsm or need_cfm or need_force_energy_t0
        )

        # Only train with higher-order autodiff; during eval we can compute the residuals
        # with create_graph=False to avoid massive memory usage.
        enable_higher_order = bool(self.hjb_enable_higher_order and self.training)
        divergence_create_graph = bool(self.hjb_divergence_create_graph and self.training)
        trace_batch_size = int(self.hjb_trace_batch_size) if enable_higher_order else 1
        hjb_debug = bool(self.hjb_debug and self.training)

        grad_ctx = nullcontext()
        if need_time_space_grads:
            grad_contexts = []
            if torch.is_inference_mode_enabled():
                grad_contexts.append(torch.inference_mode(False))
            if not torch.is_grad_enabled():
                grad_contexts.append(torch.enable_grad())

            if len(grad_contexts) == 1:
                grad_ctx = grad_contexts[0]
            elif len(grad_contexts) > 1:
                grad_stack = ExitStack()
                for ctx in grad_contexts:
                    grad_stack.enter_context(ctx)
                grad_ctx = grad_stack

        if need_main_pass:
            assert zt_x is not None
            assert zt_h is not None
            with grad_ctx:
                zt_x = zt_x.detach().requires_grad_(need_time_space_grads)
                t_used = t_used.detach().requires_grad_(need_hjb)

                pred_diff, pred_corr, pred_total = self._forward_model_components(
                    zt_x,
                    zt_h,
                    ligand["mask"],
                    t_used,
                    temperature=temperature,
                    correction_active=correction_active,
                )
        else:
            pred_diff = None
            pred_corr = None
            pred_total = None

        # Coordinate flow-matching losses (computed at the *noisy* state (zt_x, t)).
        # This implements:
        #  - L_relation (constitutive): v = f - 0.5 g^2 ∇u / T
        #  - L_continuity: du/dt + v·∇u - div(v) = 0
        # Boundary condition on energy is handled by loss_energy below.
        loss_x = torch.zeros((ligand["size"].size(0),), device=ligand["x"].device, dtype=ligand["x"].dtype)

        ######## Compute atom-type loss ########

        if need_h and self.diffuse_h and (self.lambda_h is None or float(self.lambda_h) != 0.0):
            assert pred_total is not None
            assert zt_h is not None
            # Markov-bridge time convention matches flow-time: 0 noisy -> 1 clean.
            t_mb = t
            t_next = torch.clamp(t_mb + self.train_step_size, max=1.0)
            loss_h = self.module_h.compute_loss(
                pred_total["logits_h"],
                zt_h,
                ligand["one_hot"],
                ligand["mask"],
                t_mb,
                t_next,
                reduce=self.loss_reduce,
            )
        else:
            loss_h = torch.zeros_like(loss_x)

        # Supervised energy head / boundary condition losses.
        loss_energy = torch.zeros_like(loss_x)
        loss_energy_t0 = torch.zeros_like(loss_x)
        loss_force_t0 = torch.zeros_like(loss_x)
        loss_force_energy_t0 = torch.zeros_like(loss_x)
        loss_force_dsm = torch.zeros_like(loss_x)
        loss_energy_dsm = torch.zeros_like(loss_x)
        loss_cfm = torch.zeros_like(loss_x)
        loss_hjb = torch.zeros_like(loss_x)
        loss_consistency = torch.zeros_like(loss_x)

        ######## Phase 1: diffusion-base DSM pretraining. ########
        grad_energy_total = None
        if need_force_dsm or need_energy_dsm:
            assert pred_diff is not None
            assert zt_x is not None
            tau_noisy = 1.0 - t_used
            force_target = self.module_x.score_target(
                zt_x,
                ligand["x"],
                tau_noisy,
                ligand["mask"],
                temperature=temperature,
            ).to(dtype=ligand["x"].dtype)
            sigma_tau = self.module_x.sigma(tau_noisy, temperature=temperature)[ligand["mask"]].squeeze(-1)
            dsm_weight = torch.clamp(sigma_tau * sigma_tau, min=1e-12)

            if need_force_dsm:
                force_diff = pred_diff.get("force", pred_diff.get("v", None))
                if force_diff is None:
                    raise RuntimeError("Internal error: diffusion predictor must return a force field")
                per_node_force_dsm = torch.mean((force_diff.to(dtype=ligand["x"].dtype) - force_target) ** 2, dim=-1)
                loss_force_dsm = scatter_mean(per_node_force_dsm * dsm_weight, ligand["mask"], dim=0)

            if need_energy_dsm:
                energy_diff = pred_diff["energy"].to(dtype=ligand["x"].dtype).view(-1)
                grad_energy_diff = torch.autograd.grad(
                    outputs=energy_diff.sum(),
                    inputs=zt_x,
                    create_graph=bool(self.training),
                    retain_graph=True,
                    allow_unused=False,
                )[0]
                per_node_energy_dsm = torch.mean(((-grad_energy_diff) - force_target) ** 2, dim=-1)
                loss_energy_dsm = scatter_mean(per_node_energy_dsm * dsm_weight, ligand["mask"], dim=0)

        ######## Phase 2: correction consistency and total-field HJB. ########
        if need_consistency:
            assert pred_total is not None
            assert zt_x is not None
            if grad_energy_total is None:
                energy_total = pred_total["energy"].to(dtype=ligand["x"].dtype).view(-1)
                grad_energy_total = torch.autograd.grad(
                    outputs=energy_total.sum(),
                    inputs=zt_x,
                    create_graph=bool(self.training),
                    retain_graph=True,
                    allow_unused=False,
                )[0]
            per_node_cons = torch.mean((pred_total["force"].to(dtype=ligand["x"].dtype) + grad_energy_total) ** 2, dim=-1)
            loss_consistency = scatter_mean(per_node_cons, ligand["mask"], dim=0)

        if need_hjb:
            assert pred_total is not None
            assert zt_x is not None
            loss_hjb = self.module_x.hjb_loss(
                pred_total["force"].to(dtype=ligand["x"].dtype),
                pred_total["energy"].to(dtype=ligand["x"].dtype).view(-1),
                zt_x,
                t_used,
                ligand["mask"],
                temperature=temperature,
                reduce=self.loss_reduce,
                divergence=cast(Literal["exact", "hutchinson"], self.hjb_divergence),
                n_trace_samples=self.hjb_trace_samples,
                enable_higher_order=enable_higher_order,
                divergence_create_graph=divergence_create_graph,
                trace_batch_size=trace_batch_size,
                compute_hjb=True,
                debug=hjb_debug,
            )

        ######## Boundary conditions at t=0 (clean), computed on a subset of graphs. ########
        if need_energy_t0 or need_force_t0 or need_force_energy_t0:
            if pred_total is None:
                raise RuntimeError("Internal error: boundary losses requested but total prediction is missing")
            if is_t0_graph is None:
                raise RuntimeError("Internal error: is_t0_graph missing")

            if bool(is_t0_graph.any()):
                if need_energy_t0:
                    assert energy is not None
                    energy_tgt = energy.to(device=ligand["x"].device, dtype=ligand["x"].dtype).view(-1)
                    energy_pred = pred_total["energy"].to(dtype=ligand["x"].dtype).view(-1)
                    if energy_pred.shape != energy_tgt.shape:
                        raise ValueError(
                            f"energy_pred must have shape (B,), got {tuple(energy_pred.shape)} vs target {tuple(energy_tgt.shape)}"
                        )

                    # Energy referencing: E_ref = E_DFT - sum_i E_atom(Z_i). (Charge ignored: use 0.)
                    if len(getattr(self, "omol_elem_ref", {})) > 0:
                        e_atom_sum = utils.sum_atomref_per_graph(
                            ligand["one_hot"],
                            ligand["mask"],
                            allowed_atomic_numbers=self.allowed_atomic_numbers,
                            elem_ref=self.omol_elem_ref,
                        ).to(dtype=energy_tgt.dtype)
                        energy_tgt = energy_tgt - e_atom_sum

                    loss_energy = torch.abs(energy_pred - energy_tgt)
                    loss_energy = loss_energy * is_t0_graph.to(dtype=loss_energy.dtype)

                if need_force_t0:
                    assert force is not None
                    force_pred = pred_total.get("force", None)
                    if force_pred is None:
                        raise RuntimeError("Internal error: force boundary requested but force output is missing")
                    force_tgt = force.to(device=ligand["x"].device, dtype=ligand["x"].dtype)
                    if force_tgt.shape != force_pred.shape:
                        raise ValueError(
                            f"force must have shape {tuple(force_pred.shape)} to match -dE/dx, got {tuple(force_tgt.shape)}"
                        )

                    is_t0_node = is_t0_graph[ligand["mask"]]
                    per_node = torch.mean((force_pred.to(dtype=ligand["x"].dtype) - force_tgt) ** 2, dim=-1)
                    per_node = per_node * is_t0_node.to(dtype=per_node.dtype)
                    loss_force_t0 = scatter_mean(per_node, ligand["mask"], dim=0)

                if need_force_energy_t0:
                    assert force is not None
                    assert zt_x is not None
                    if grad_energy_total is None:
                        energy_pred = pred_total["energy"].to(dtype=ligand["x"].dtype).view(-1)
                        grad_energy_total = torch.autograd.grad(
                            outputs=energy_pred.sum(),
                            inputs=zt_x,
                            create_graph=bool(self.training),
                            retain_graph=True,
                            allow_unused=False,
                        )[0]
                    assert grad_energy_total is not None
                    force_tgt = force.to(device=ligand["x"].device, dtype=ligand["x"].dtype)
                    if force_tgt.shape != grad_energy_total.shape:
                        raise ValueError(
                            f"force must have shape {tuple(grad_energy_total.shape)} to match -dU/dx, got {tuple(force_tgt.shape)}"
                        )

                    is_t0_node = is_t0_graph[ligand["mask"]]
                    per_node = torch.mean(((-grad_energy_total) - force_tgt) ** 2, dim=-1)
                    per_node = per_node * is_t0_node.to(dtype=per_node.dtype)
                    loss_force_energy_t0 = scatter_mean(per_node, ligand["mask"], dim=0)
            else:
                loss_energy = torch.zeros_like(loss_x)
                loss_force_t0 = torch.zeros_like(loss_x)
                loss_force_energy_t0 = torch.zeros_like(loss_x)

        ######## Conditional flow matching (CFM) velocity loss (VE schedule). ########
        if need_cfm:
            if self.module_x.sde.kind != "ve":
                raise ValueError("lambda_cfm is only implemented for coord_sde_kind='ve' (VE kinematics)")
            assert pred_diff is not None
            assert eps_x is not None
            assert zt_x is not None
            # Build target velocity in flow time t (0 noisy -> 1 clean).
            # We parameterize the perturbation in diffusion forward-time tau = 1 - t:
            #   x(t) = x0 + sigma(tau) * eps
            # so v_target = dx/dt = - d_sigma/dtau * eps.
            t_det = t_used.detach()  # avoid building higher-order graphs unnecessarily
            sigma_min = float(self.module_x.sde.sigma_min)
            sigma_max = float(self.module_x.sde.sigma_max)
            log_ratio = float(np.log(sigma_max / sigma_min))

            tau_det = 1.0 - t_det
            sigma_tau = self.module_x.sigma(tau_det, temperature=temperature)  # (B,1)
            d_sigma_dt = -sigma_tau * log_ratio
            v_target = d_sigma_dt[ligand["mask"]] * eps_x

            force_pred = pred_diff.get("force", pred_diff.get("v", None))
            if force_pred is None:
                raise KeyError("Dynamics must return 'force' (preferred) or legacy 'v'")

            # Phase-1 flow matching is supervised in score/force units, so it consumes the
            # raw diffusion predictor output before any thermo scaling for phase 2.
            force_pred_model = force_pred.to(dtype=ligand["x"].dtype)

            # Velocity for flow-matching loss is computed directly from predicted force:
            #   v = f + 0.5 * g^2 * F
            tau_used = 1.0 - t_used
            f = self.module_x._sde_f_forward(
                zt_x,
                tau_used,
                ligand["mask"],
                temperature=temperature,
            )
            g2 = self.module_x._sde_g2_forward(tau_used, temperature=temperature)  # (B,)
            g2_node = g2[ligand["mask"]].unsqueeze(-1)
            v_pred = f + 0.5 * g2_node * force_pred_model

            per_node = torch.sum((v_pred - v_target) ** 2, dim=-1)

            # Optional SNR weighting (VE: SNR = 1/sigma(t)^2) to emphasize noisier steps
            # less and clamp gradients at very high SNR (near-clean) if requested.
            if self.cfm_weight_by_snr:
                sigma_t = self.module_x.sigma(1.0 - t_used, temperature=temperature)[ligand["mask"]].squeeze(-1)
                snr = 1.0 / torch.clamp(sigma_t * sigma_t, min=1e-12)
                if self.cfm_use_min_snr:
                    snr = torch.clamp(snr, max=float(self.cfm_min_snr_gamma))
                per_node = per_node * snr

            loss_cfm = scatter_mean(per_node / float(self.x_dim), ligand["mask"], dim=0)

        loss = (
            self.lambda_x * loss_x
            + self.lambda_h * loss_h
            + self.lambda_force_dsm * loss_force_dsm
            + self.lambda_energy_dsm * loss_energy_dsm
            + self.lambda_energy * loss_energy
            + float(getattr(self, "lambda_force_t0", 0.0)) * loss_force_t0
            + self.lambda_force_energy_t0 * loss_force_energy_t0
            + self.lambda_cfm * loss_cfm
            + self.lambda_hjb * loss_hjb
            + self.lambda_consistency * loss_consistency
        )
        loss = loss.mean(0)

        info = {
            "loss_x": float(loss_x.mean().detach().cpu()),
            "loss_h": float(loss_h.mean().detach().cpu()),
            "loss_force_dsm": float(loss_force_dsm.mean().detach().cpu()),
            "loss_energy_dsm": float(loss_energy_dsm.mean().detach().cpu()),
            "loss_energy": float(loss_energy.mean().detach().cpu()),
            "loss_force_t0": float(loss_force_t0.mean().detach().cpu()),
            "loss_force_energy_t0": float(loss_force_energy_t0.mean().detach().cpu()),
            "loss_cfm": float(loss_cfm.mean().detach().cpu()),
            "loss_hjb": float(loss_hjb.mean().detach().cpu()),
            "loss_consistency": float(loss_consistency.mean().detach().cpu()),
            "temperature": float(temp_val),
            "diffuse_h": bool(self.diffuse_h and (self.lambda_h is None or float(self.lambda_h) != 0.0)),
            "phase_is_correction": float(1.0 if correction_active else 0.0),
            "use_gamma_scaling": float(1.0 if self.use_gamma_scaling else 0.0),
            "gamma_mean": float(pred_total["gamma_graph"].mean().detach().cpu()) if pred_total is not None else 1.0,
            "scale_diffusion_by_thermo": float(1.0 if self.scale_diffusion_by_thermo else 0.0),
            "thermo_reference_temperature_K": float(self.thermo_reference_temperature_K),
            "thermo_kbt_ev": float(self.thermo_kbt_ev),
        }

        # Optional: log the (scheduled) t=0 fraction used for boundary losses.
        if (need_energy_t0 or need_force_t0 or need_force_energy_t0) and need_main_pass:
            # frac_used is defined only in that branch; recompute a safe proxy here.
            info["t0_batch_fraction_target"] = float(np.clip(float(getattr(self, "t0_batch_fraction", 1.0)), 0.0, 1.0))
            if is_t0_graph is not None:
                info["t0_batch_fraction_effective"] = float(is_t0_graph.float().mean().detach().cpu().item())

        if self.log_diffusion_stats and (pred_total is not None) and (zt_x is not None):
            with torch.no_grad():
                sigma_b = self.module_x.sigma(1.0 - t.detach(), temperature=temperature).view(-1)
                v = pred_total.get("force", pred_total.get("v", None))
                if v is None:
                    raise RuntimeError("Internal error: total predictor must return 'force' or legacy 'v'")
                v = v.detach()
                v_norm = torch.sqrt(torch.sum(v * v, dim=-1) + 1e-12)
                info.update(
                    {
                        "t/mean": float(t.detach().mean().cpu()),
                        "t/min": float(t.detach().min().cpu()),
                        "t/max": float(t.detach().max().cpu()),
                        "sigma/mean": float(sigma_b.mean().cpu()),
                        "sigma/min": float(sigma_b.min().cpu()),
                        "sigma/max": float(sigma_b.max().cpu()),
                        "v_norm/mean": float(v_norm.mean().cpu()),
                    }
                )
        return (loss, info) if return_info else loss

    def training_step(self, batch, batch_idx: int = 0, *args):
        # Implement optional time-horizon curriculum over *batch steps*.
        # If enabled, we pass the sampled t explicitly into compute_loss.
        t = None
        t_low = float(self.t_min)
        t_high = float(self.t_max)

        # Step bookkeeping (used by both time-horizon curriculum and t=0-fraction schedule).
        step: int | None = None
        total_steps: int | None = None
        try:
            n_batches = int(getattr(self.trainer, "num_training_batches", 0) or 0)
            n_epochs = int(getattr(self.trainer, "max_epochs", 0) or getattr(self._train_params, "n_epochs", 0) or 0)
        except Exception:
            n_batches = 0
            n_epochs = int(getattr(self._train_params, "n_epochs", 0) or 0)
        if n_batches > 0 and n_epochs > 0:
            total_steps = int(n_epochs * n_batches)
            step = int(self.current_epoch) * int(n_batches) + int(batch_idx)

        if self.time_horizon_schedule == "sinusoidal":
            if (step is not None) and (total_steps is not None) and (total_steps > 0):
                t_high = min(float(self.t_max), float(self._time_horizon_t_max_for_step(step, total_steps)))
                if t_high < t_low:
                    t_high = t_low
                B = int(batch["ligand"]["size"].numel())
                u = torch.rand((B, 1), device=batch["ligand"]["x"].device, dtype=batch["ligand"]["x"].dtype)
                t = u * (t_high - t_low) + t_low

        loss, info = self.compute_loss(
            batch["ligand"],
            energy=batch.get("energy", None),
            force=batch.get("force", None),
            t=t,
            step=step,
            total_steps=total_steps,
            return_info=True,
        )
        # Log the timestep sampling bounds used for this batch.
        # (For sinusoidal schedule, these reflect the curriculum; otherwise they are the static config bounds.)
        self.log("t_low/train", float(t_low), on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("t_high/train", float(t_high), on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss/train", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_x/train", info["loss_x"], on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_h/train", info["loss_h"], on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_force_dsm/train", info["loss_force_dsm"], on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_energy_dsm/train", info["loss_energy_dsm"], on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_energy/train", info["loss_energy"], on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_force_t0/train", info["loss_force_t0"], on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_force_energy_t0/train", info["loss_force_energy_t0"], on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_cfm/train", info["loss_cfm"], on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_hjb/train", info["loss_hjb"], on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log(
            "loss_consistency/train",
            info["loss_consistency"],
            on_step=True,
            on_epoch=True,
            batch_size=len(batch["ligand"]["size"]),
        )
        self.log("temperature/train", info["temperature"], on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("phase_is_correction/train", info["phase_is_correction"], on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("gamma_mean/train", info["gamma_mean"], on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("scale_diffusion_by_thermo/train", info["scale_diffusion_by_thermo"], on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("thermo_reference_temperature_K/train", info["thermo_reference_temperature_K"], on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("thermo_kbt_ev/train", info["thermo_kbt_ev"], on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        if self.log_diffusion_stats:
            for k in (
                "t/mean",
                "t/min",
                "t/max",
                "sigma/mean",
                "sigma/min",
                "sigma/max",
                "v_norm/mean",
            ):
                if k in info:
                    self.log(f"{k}/train", info[k], on_step=True, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        return loss

    def validation_step(self, batch, *args):
        needs_eval_grad = (
            float(self.lambda_x) > 0.0
            or float(self.lambda_energy_dsm) > 0.0
            or float(self.lambda_cfm) > 0.0
            or (
                self._phase_name_for_epoch() == "correction"
                and float(self.lambda_force_energy_t0) > 0.0
            )
            or (
                self._phase_name_for_epoch() == "correction"
                and (float(self.lambda_hjb) > 0.0 or float(self.lambda_consistency) > 0.0)
            )
        )

        eval_ctx = nullcontext()
        if needs_eval_grad:
            eval_stack = ExitStack()
            if torch.is_inference_mode_enabled():
                eval_stack.enter_context(torch.inference_mode(False))
            if not torch.is_grad_enabled():
                eval_stack.enter_context(torch.enable_grad())
            eval_ctx = eval_stack

        with eval_ctx:
            loss, info = self.compute_loss(
                batch["ligand"],
                energy=batch.get("energy", None),
                force=batch.get("force", None),
                return_info=True,
            )
        self.log("loss/val", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_x/val", info["loss_x"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_h/val", info["loss_h"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_force_dsm/val", info["loss_force_dsm"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_energy_dsm/val", info["loss_energy_dsm"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_energy/val", info["loss_energy"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_force_t0/val", info["loss_force_t0"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_force_energy_t0/val", info["loss_force_energy_t0"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_cfm/val", info["loss_cfm"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_hjb/val", info["loss_hjb"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log(
            "loss_consistency/val",
            info["loss_consistency"],
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["ligand"]["size"]),
        )
        self.log("temperature/val", info["temperature"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("phase_is_correction/val", info["phase_is_correction"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("gamma_mean/val", info["gamma_mean"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("scale_diffusion_by_thermo/val", info["scale_diffusion_by_thermo"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("thermo_reference_temperature_K/val", info["thermo_reference_temperature_K"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("thermo_kbt_ev/val", info["thermo_kbt_ev"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        if self.log_diffusion_stats:
            for k in (
                "t/mean",
                "t/min",
                "t/max",
                "sigma/mean",
                "sigma/min",
                "sigma/max",
                "v_norm/mean",
            ):
                if k in info:
                    self.log(f"{k}/val", info[k], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        return {"loss": loss, **info}

    def test_step(self, batch, *args):
        needs_eval_grad = (
            float(self.lambda_x) > 0.0
            or float(self.lambda_energy_dsm) > 0.0
            or float(self.lambda_cfm) > 0.0
            or (
                self._phase_name_for_epoch() == "correction"
                and float(self.lambda_force_energy_t0) > 0.0
            )
            or (
                self._phase_name_for_epoch() == "correction"
                and (float(self.lambda_hjb) > 0.0 or float(self.lambda_consistency) > 0.0)
            )
        )

        eval_ctx = nullcontext()
        if needs_eval_grad:
            eval_stack = ExitStack()
            if torch.is_inference_mode_enabled():
                eval_stack.enter_context(torch.inference_mode(False))
            if not torch.is_grad_enabled():
                eval_stack.enter_context(torch.enable_grad())
            eval_ctx = eval_stack

        with eval_ctx:
            loss, info = self.compute_loss(
                batch["ligand"],
                energy=batch.get("energy", None),
                force=batch.get("force", None),
                return_info=True,
            )
        self.log("loss/test", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_x/test", info["loss_x"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_h/test", info["loss_h"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_force_dsm/test", info["loss_force_dsm"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_energy_dsm/test", info["loss_energy_dsm"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_energy/test", info["loss_energy"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_force_t0/test", info["loss_force_t0"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_force_energy_t0/test", info["loss_force_energy_t0"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_cfm/test", info["loss_cfm"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("loss_hjb/test", info["loss_hjb"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log(
            "loss_consistency/test",
            info["loss_consistency"],
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["ligand"]["size"]),
        )
        self.log("temperature/test", info["temperature"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("phase_is_correction/test", info["phase_is_correction"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("gamma_mean/test", info["gamma_mean"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("scale_diffusion_by_thermo/test", info["scale_diffusion_by_thermo"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("thermo_reference_temperature_K/test", info["thermo_reference_temperature_K"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
        self.log("thermo_kbt_ev/test", info["thermo_kbt_ev"], on_step=False, on_epoch=True, batch_size=len(batch["ligand"]["size"]))
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
        atom_type_prior: Literal["uniform"] = "uniform",
        stochastic_x: bool = False,
        stochastic_h: bool = False,
        temperature_sampler: float | None = None,
    ):
        """Sample ligand-only (x, one_hot) with two atom-type sampling modes.

        Args:
          n_samples: number of molecules
          num_nodes: int (fixed size) or tensor (n_samples,) with per-mol sizes
          timesteps: Euler steps from t=0 -> t=1 (denoising)
          temperature: scalar conditioning value (if None, uses config temperature)
                    x_sampler: 'ode' for velocity integration using the force head, or 'sde'
                        for reverse diffusion using the score induced by the energy head.
                    h_sampler: 'markov_bridge' or 'score'
                    stochastic_x/h: only used when the corresponding sampler uses stochastic steps
        """
        assert x_sampler in {"ode", "sde"}
        assert h_sampler in {"markov_bridge", "score"}
        assert atom_type_prior in {"uniform"}

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
        if temperature_sampler is not None:
            sampler_temperature = torch.full((n_samples, 1), float(temperature_sampler), device=device, dtype=dtype)
        else:
            sampler_temperature = t_temperature

        # Initialize x at the noisiest state (tau=1). In flow time, this corresponds to t=0.
        com = torch.zeros((n_samples, self.x_dim), device=device, dtype=dtype)
        x = self.module_x.sample_z1(com, batch_mask, temperature=t_temperature)

        # Initialize atoms.
        # If atom-type diffusion is disabled, we must provide a "clean" categorical conditioning
        # signal to the dynamics. Since the model is not trained to generate atom types (and
        # logits_h may be untrained), sample atom types from a simple prior and keep them fixed.
        if not self.diffuse_h:
            if atom_type_prior == "uniform":
                idx = torch.randint(self.atom_nf, (batch_mask.shape[0],), device=device)
            else:
                raise ValueError(f"Unsupported atom_type_prior: {atom_type_prior}")
            h = torch.nn.functional.one_hot(idx, num_classes=self.atom_nf).to(dtype)
            h_logits = None
        else:
            if h_sampler == "markov_bridge":
                h = self.module_h.sample_z0(batch_mask)  # one-hot
                h_logits = None
            else:
                h_logits = self.module_h_score.sample_prior(batch_mask, temperature=t_temperature, dtype=dtype)
                h = self.module_h_score.logits_to_probs(h_logits)

        dt = 1.0 / float(timesteps)
        correction_active = self._phase_name_for_epoch() == "correction"
        for i in range(timesteps):
            # Denoising integration in flow time: start at s=0 and step up to t=1.
            s_val = i * dt
            t_val = (i + 1) * dt
            s = torch.full((n_samples, 1), s_val, device=device, dtype=dtype)
            t = torch.full((n_samples, 1), t_val, device=device, dtype=dtype)

            if x_sampler == "ode":
                _, _, pred_total = self._forward_model_components(
                    x,
                    h,
                    batch_mask,
                    s,
                    temperature=t_temperature,
                    correction_active=correction_active,
                )

                # Coordinates (flow matching): explicit Euler integration dx/dt = v(x,t).
                # Convert predicted force F into velocity using: v = f + 0.5 * g^2 * F.
                step = (t - s)[batch_mask]
                F = pred_total.get("force", pred_total.get("v", None))
                if F is None:
                    raise KeyError("Dynamics must return 'force' (preferred) or legacy 'v'")

                # The combined phase-2 field is represented in physical units; convert back
                # to score units only at the sampler boundary.
                F = self._unscale_physical_outputs_to_score_units(F.to(dtype=x.dtype))

                s_tau = 1.0 - s
                f = self.module_x._sde_f_forward(x, s_tau, batch_mask, temperature=sampler_temperature).to(dtype=x.dtype)
                g2 = self.module_x._sde_g2_forward(s_tau, temperature=sampler_temperature)  # (B,)
                g2_node = g2[batch_mask].unsqueeze(-1).to(dtype=x.dtype)
                v = f + 0.5 * g2_node * F
                x = x + step * v
            else:
                # Reverse diffusion sampling uses the predicted force/score field directly.
                _, _, pred_total = self._forward_model_components(
                    x,
                    h,
                    batch_mask,
                    s,
                    temperature=t_temperature,
                    correction_active=correction_active,
                )
                score_s = pred_total.get("force", pred_total.get("v", None))
                if score_s is None:
                    raise KeyError("Dynamics must return 'force' (preferred) or legacy 'v'")

                score_s = self._unscale_physical_outputs_to_score_units(score_s.to(dtype=x.dtype))
                s_tau = 1.0 - s
                t_tau = 1.0 - t
                x = self.module_x.reverse_step(
                    x,
                    score_s,
                    s=s_tau,
                    t=t_tau,
                    batch_mask=batch_mask,
                    temperature=sampler_temperature,
                    stochastic=stochastic_x,
                )

            # Enforce translation invariance (match training which centers per molecule).
            # IMPORTANT: for toy settings (e.g., 1-node graphs learning an absolute-position
            # energy field), centering will collapse all samples to the origin.
            if getattr(self, "center_ligand", True):
                com = scatter_mean(x, batch_mask, dim=0)
                x = x - com[batch_mask]

            # Atom types.
            if self.diffuse_h:
                if h_sampler == "markov_bridge":
                    # Markov-bridge time matches flow time (0 noisy -> 1 clean).
                    s_mb = s
                    t_mb = t
                    h = self.module_h.sample_zt_given_zs(
                        h,
                        pred_total["logits_h"],
                        s=s_mb,
                        t=t_mb,
                        batch_mask=batch_mask,
                    )
                else:
                    # Interpret dynamics logits_h as score in logit-space.
                    assert h_logits is not None
                    # Logit-score diffusion utilities are parameterized in diffusion forward-time
                    # tau=0 clean -> tau=1 most noisy, so we use tau = 1 - t (flow time).
                    s_tau = 1.0 - s
                    t_tau = 1.0 - t
                    if not stochastic_h:
                        h_logits = self.module_h_score.ode_step(
                            h_logits,
                            pred_total["logits_h"],
                            s=s_tau,
                            t=t_tau,
                            batch_mask=batch_mask,
                            temperature=t_temperature,
                        )
                    else:
                        h_logits = self.module_h_score.reverse_step(
                            h_logits,
                            pred_total["logits_h"],
                            s=s_tau,
                            t=t_tau,
                            batch_mask=batch_mask,
                            temperature=t_temperature,
                            stochastic=stochastic_h,
                        )
                    h = self.module_h_score.logits_to_probs(h_logits)

        # Finalize categorical sample
        if self.diffuse_h and h_sampler == "score":
            idx = torch.argmax(h, dim=-1)
            h = torch.nn.functional.one_hot(idx, num_classes=self.atom_nf).to(h.dtype)

        return {
            "x": x,
            "one_hot": h,
            "mask": batch_mask,
            "size": sizes,
        }
