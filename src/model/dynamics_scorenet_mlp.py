from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch_scatter import scatter_mean


@dataclass(frozen=True)
class ScoreNetMLPParams:
    input_dim: int = 2
    hidden_dim: int = 128
    time_emb_dim: int = 64
    num_layers: int = 3

    # Conditioning flags (kept for API symmetry with other dynamics)
    condition_time: bool = True
    condition_temperature: bool = False

    # Force head scaling / scheduling (match GVP dynamics contract)
    force_fm_scale: float = 1.0
    force_corr_scale: float = 1.0
    force_corr_schedule: bool = False


class SinusoidalTimeEmbeddings(nn.Module):
    """Creates a high-dimensional vector representation of scalar time t."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # time: (B,1) or (B,)
        if time.dim() == 1:
            time = time.unsqueeze(-1)
        if time.shape[-1] != 1:
            raise ValueError(f"time must have shape (B,1) or (B,); got {tuple(time.shape)}")

        device = time.device
        dtype = time.dtype

        half_dim = self.dim // 2
        if half_dim <= 1:
            raise ValueError("dim must be >= 4")

        embeddings = torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / float(half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * (-embeddings))
        embeddings = time * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeAwareResBlock(nn.Module):
    """Residual block that injects time embedding via additive projection."""

    def __init__(self, dim: int, time_emb_dim: int):
        super().__init__()
        self.act = nn.SiLU()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(time_emb_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.linear1(self.act(self.norm(x)))
        h = h + self.time_proj(self.act(t_emb))
        h = self.linear2(self.act(h))
        return x + h


class ScoreNet(nn.Module):
    """Time-conditioned MLP trunk producing hidden features h(x,t)."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, time_emb_dim: int = 64, num_layers: int = 3):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.blocks = nn.ModuleList([TimeAwareResBlock(hidden_dim, time_emb_dim) for _ in range(int(num_layers))])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final_act = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        t_emb = self.time_mlp(t)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        h = self.final_act(self.final_norm(h))
        return h


class ScoreNetMLPDynamics(nn.Module):
    """Drop-in dynamics module for EnergyForceDiffusion using ScoreNet MLP.

        Matches the dynamics forward signature and returns pred_ligand with keys:
            - logits_h: (N, atom_nf)
            - energy: (B,)
            - force: (N, x_dim)
            - force_fm: (N, x_dim)
            - force_corr: (N, x_dim)
            - v: (N, x_dim) legacy alias for `force`

    Notes:
      - For swiss-roll notebook we only use x/y; z is ignored and force_z is zero.
      - Energy is predicted per-node then pooled to per-graph via scatter_mean.
    """

    def __init__(self, atom_nf: int, x_dim: int = 3, params: ScoreNetMLPParams = ScoreNetMLPParams()):
        super().__init__()
        self.atom_nf = int(atom_nf)
        self.x_dim = int(x_dim)
        self.params = params

        if self.x_dim < 2:
            raise ValueError(f"x_dim must be >= 2; got {self.x_dim}")
        if int(self.params.input_dim) != 2:
            raise ValueError("ScoreNetMLPParams.input_dim must be 2 (x,y)")

        self.net = ScoreNet(
            input_dim=int(self.params.input_dim),
            hidden_dim=int(self.params.hidden_dim),
            time_emb_dim=int(self.params.time_emb_dim),
            num_layers=int(self.params.num_layers),
        )

        # Optional atom-type logits head (kept for compatibility with diffuse_h=True paths).
        self.logits_h_head = nn.Linear(int(self.params.hidden_dim), self.atom_nf)
        self.energy_head = nn.Linear(int(self.params.hidden_dim), 1)
        self.force_fm_head = nn.Linear(int(self.params.hidden_dim), 2)
        self.force_corr_head = nn.Linear(int(self.params.hidden_dim), 2)

    def _t_to_node(self, t: torch.Tensor, mask_atoms: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if t.shape[-1] != 1:
            raise ValueError(f"t must have trailing dim 1; got {tuple(t.shape)}")
        if t.numel() == 1:
            return t.to(dtype=dtype).view(1, 1).expand(mask_atoms.size(0), 1)
        return t[mask_atoms].to(dtype=dtype)

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
        if not self.params.condition_time:
            raise ValueError("ScoreNetMLPDynamics requires condition_time=True")
        if t is None:
            raise ValueError("t is required")

        # Node-wise inputs (N,2)
        x2 = x_atoms[:, 0:2].to(dtype=torch.float32)

        # Time per node (N,1), preserve autograd
        t_node = self._t_to_node(t, mask_atoms, dtype=x2.dtype)

        # Forward through ScoreNet trunk
        h = self.net(x2, t_node)

        force_fm2 = self.force_fm_head(h) * float(getattr(self.params, "force_fm_scale", 1.0))
        force_corr2 = self.force_corr_head(h) * float(getattr(self.params, "force_corr_scale", 1.0))

        final_force_corr2 = force_corr2
        if bool(getattr(self.params, "force_corr_schedule", False)):
            final_force_corr2 = force_corr2 * t_node.to(dtype=force_corr2.dtype)

        force2 = force_fm2 + final_force_corr2

        # Pad forces to (N, x_dim) for the diffusion module, keep dtype consistent with x_atoms.
        force = torch.zeros((x_atoms.size(0), self.x_dim), device=x_atoms.device, dtype=x_atoms.dtype)
        force_fm = torch.zeros_like(force)
        force_corr = torch.zeros_like(force)
        force[:, 0:2] = force2.to(dtype=x_atoms.dtype)
        force_fm[:, 0:2] = force_fm2.to(dtype=x_atoms.dtype)
        force_corr[:, 0:2] = final_force_corr2.to(dtype=x_atoms.dtype)

        # Pool node energies -> per-graph energy (B,)
        energy_node = self.energy_head(h).squeeze(-1).to(dtype=x_atoms.dtype)
        energy = scatter_mean(energy_node, mask_atoms, dim=0)

        logits_h = self.logits_h_head(h)

        pred_ligand = {
            "logits_h": logits_h,
            "energy": energy,
            "force": force,
            "force_fm": force_fm,
            "force_corr": force_corr,
            # Backward-compatible alias; callers now prefer `force` explicitly.
            "v": force,
        }
        pred_residues = {}
        return pred_ligand, pred_residues
