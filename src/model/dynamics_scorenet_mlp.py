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
    """Time-conditioned MLP predicting velocity v(x,t) and energy u(x,t)."""

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

        self.head_force = nn.Linear(hidden_dim, 2)
        self.head_energy = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        t_emb = self.time_mlp(t)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        h = self.final_act(self.final_norm(h))

        force = self.head_force(h)
        energy = self.head_energy(h)
        return force, energy


class ScoreNetMLPDynamics(nn.Module):
    """Drop-in dynamics module for EnergyForceDiffusion using ScoreNet MLP.

        Matches the dynamics forward signature and returns pred_ligand with keys:
            - logits_h: (N, atom_nf)
            - energy: (B,)
            - v: (N, 3)  (velocity)

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

        # Helper to reuse the backbone's input embedding when producing logits.
        # We keep it minimal and consistent: x->input_proj->blocks->final->h.
        self._input_proj = self.net.input_proj
        self._time_mlp = self.net.time_mlp
        self._blocks = self.net.blocks
        self._final_norm = self.net.final_norm
        self._final_act = self.net.final_act

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

        # Forward through ScoreNet
        v2, energy_node_1 = self.net(x2, t_node)

        # Pad velocity to (N,3) for the diffusion module, keep dtype consistent with x_atoms
        v = torch.zeros((x_atoms.size(0), self.x_dim), device=x_atoms.device, dtype=x_atoms.dtype)
        v[:, 0:2] = v2.to(dtype=x_atoms.dtype)

        # Pool node energies -> per-graph energy (B,)
        energy_node = energy_node_1.squeeze(-1).to(dtype=x_atoms.dtype)
        energy = scatter_mean(energy_node, mask_atoms, dim=0)

        # Build a hidden representation for logits_h by re-running the trunk (cheap vs adding hooks)
        # This keeps logits_h defined even if diffuse_h is enabled.
        t_emb = self._time_mlp(t_node)
        h = self._input_proj(x2)
        for block in self._blocks:
            h = block(h, t_emb)
        h = self._final_act(self._final_norm(h))
        logits_h = self.logits_h_head(h)

        pred_ligand = {
            "logits_h": logits_h,
            "energy": energy,
            "v": v,
        }
        pred_residues = {}
        return pred_ligand, pred_residues
