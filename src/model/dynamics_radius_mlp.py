from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean


@dataclass(frozen=True)
class RadiusMLPParams:
    hidden_dim: int = 256
    n_layers: int = 3
    condition_time: bool = True
    condition_temperature: bool = True
    # For now we build edges but do not use them in this baseline.
    # A proper radius-graph message passing backbone will replace this in the next step.

    # Include coordinates in energy/force heads so that E depends on x and
    # the boundary condition force_from_E = -∂E/∂x is meaningful.
    use_x_in_energy_force: bool = True

    # Time-conditioning architecture.
    # This is an internal backbone choice; existing call sites do not need to change.
    time_emb_dim: int = 64


class SinusoidalTimeEmbeddings(nn.Module):
    """Standard sinusoidal embedding for a scalar time t (diffusion-style)."""

    def __init__(self, dim: int):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if dim % 2 != 0:
            raise ValueError("dim must be even for sinusoidal embeddings")
        self.dim = int(dim)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # time: (..., 1) or (...,)
        if time.ndim == 1:
            time = time.unsqueeze(-1)
        if time.shape[-1] != 1:
            raise ValueError(f"time must have trailing dim 1; got {tuple(time.shape)}")

        device = time.device
        dtype = time.dtype

        half_dim = self.dim // 2
        if half_dim <= 1:
            raise ValueError("dim must be >= 4 for stable sinusoidal frequencies")

        # exp(-log(10000) * i/(half_dim-1))
        exponent = -torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) * (
            torch.arange(half_dim, device=device, dtype=dtype) / float(half_dim - 1)
        )
        freqs = torch.exp(exponent)  # (half_dim,)

        args = time * freqs.unsqueeze(0)  # (..., half_dim)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeAwareResBlock(nn.Module):
    """Residual MLP block with additive conditioning (time/temperature embedding)."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.cond_proj = nn.Linear(cond_dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.linear1(self.act(self.norm(x)))
        h = h + self.cond_proj(self.act(cond))
        h = self.linear2(self.act(h))
        return x + h


def _make_mlp(in_dim: int, out_dim: int, hidden_dim: int, n_layers: int) -> nn.Sequential:
    layers = []
    if n_layers <= 1:
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    layers.append(nn.Linear(in_dim, hidden_dim))
    layers.append(nn.SiLU())
    for _ in range(n_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.SiLU())
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class RadiusMLPDynamics(nn.Module):
    """Ligand-only baseline dynamics.

    Purpose (Step 1): provide the minimal interface expected by the training loop
    while we integrate the PKL dataset and run x+h diffusion losses without bond diffusion.

    Later steps will replace this with a real radius-graph GNN and add energy/force heads.
    """

    def __init__(self, atom_nf: int, x_dim: int = 3, params: RadiusMLPParams = RadiusMLPParams()):
        super().__init__()
        self.atom_nf = atom_nf
        self.x_dim = x_dim
        self.params = params

        in_dim = atom_nf
        if params.condition_time:
            in_dim += 1
        if params.condition_temperature:
            in_dim += 1

        # --- Time-conditioned residual backbone (keeps same I/O as the previous MLP) ---
        time_emb_dim = int(getattr(params, "time_emb_dim", 64))
        if time_emb_dim % 2 != 0:
            # Keep it safe even if a user config sets an odd value.
            time_emb_dim += 1
        time_emb_dim = max(4, time_emb_dim)
        self.cond_dim = time_emb_dim

        self.input_proj = nn.Linear(in_dim, params.hidden_dim)

        # Condition embedding combines (optional) time + temperature.
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.temperature_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        n_blocks = max(int(params.n_layers), 2)
        self.blocks = nn.ModuleList([TimeAwareResBlock(params.hidden_dim, time_emb_dim) for _ in range(n_blocks)])
        self.final_norm = nn.LayerNorm(params.hidden_dim)
        self.final_act = nn.SiLU()

        # Simple per-atom heads
        self.vel_head = _make_mlp(params.hidden_dim, x_dim, params.hidden_dim, 2)
        self.logits_h_head = _make_mlp(params.hidden_dim, atom_nf, params.hidden_dim, 2)

        # Energy/force heads (baseline).
        ef_in_dim = params.hidden_dim + (x_dim if params.use_x_in_energy_force else 0)
        self.energy_node_head = _make_mlp(ef_in_dim, 1, params.hidden_dim, 2)
        self.force_head = _make_mlp(ef_in_dim, x_dim, params.hidden_dim, 2)

    def forward(
        self,
        x_atoms: torch.Tensor,
        h_atoms: torch.Tensor,
        mask_atoms: torch.Tensor,
        pocket: Optional[Dict] = None,
        t: Optional[torch.Tensor] = None,
        temperature: Optional[torch.Tensor] = None,
        bonds_ligand: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        sc_transform=None,
    ):
        # h_atoms is expected to be (N, K) one-hot/probabilities at time t.
        feats = h_atoms
        if self.params.condition_time:
            if t is None:
                raise ValueError("t is required when condition_time=True")
            if t.numel() == 1:
                # Preserve autograd: avoid converting to Python float.
                t_node = t.to(device=h_atoms.device, dtype=h_atoms.dtype).view(1, 1).expand(h_atoms.size(0), 1)
            else:
                t_node = t[mask_atoms].to(h_atoms.dtype)
            feats = torch.cat([feats, t_node], dim=-1)
        else:
            t_node = None

        if self.params.condition_temperature:
            if temperature is None:
                raise ValueError("temperature is required when condition_temperature=True")
            if temperature.numel() == 1:
                # Preserve autograd: avoid converting to Python float.
                temp_node = temperature.to(device=h_atoms.device, dtype=h_atoms.dtype).view(1, 1).expand(h_atoms.size(0), 1)
            else:
                temp_node = temperature[mask_atoms].to(h_atoms.dtype)
            feats = torch.cat([feats, temp_node], dim=-1)
        else:
            temp_node = None

        # Backbone
        h = self.input_proj(feats)

        # Build conditioning embedding (always present for simplicity).
        # If a condition is disabled, it contributes zeros.
        cond = torch.zeros((h.size(0), self.cond_dim), device=h.device, dtype=h.dtype)
        if self.params.condition_time:
            cond = cond + self.time_mlp(t_node)
        if self.params.condition_temperature:
            cond = cond + self.temperature_mlp(temp_node)

        for block in self.blocks:
            h = block(h, cond)
        h = self.final_act(self.final_norm(h))

        # Predict velocity in the same coordinate frame as x_atoms
        vel = self.vel_head(h)
        logits_h = self.logits_h_head(h)

        if self.params.use_x_in_energy_force:
            ef_feats = torch.cat([h, x_atoms.to(h.dtype)], dim=-1)
        else:
            ef_feats = h

        # Per-node energy contribution pooled to per-molecule energy.
        energy_node = self.energy_node_head(ef_feats).squeeze(-1)  # (N,)
        energy = scatter_mean(energy_node, mask_atoms, dim=0)  # (B,)

        # Direct force head.
        force = self.force_head(ef_feats)  # (N,3)

        pred_ligand = {
            "vel": vel,
            "logits_h": logits_h,
            "energy": energy,
            "force": force,
        }
        pred_residues = {}
        return pred_ligand, pred_residues

    @staticmethod
    def center_of_mass(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return scatter_mean(x, mask, dim=0)
