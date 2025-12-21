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

        # Simple per-atom heads
        self.node_mlp = _make_mlp(in_dim, params.hidden_dim, params.hidden_dim, max(params.n_layers, 2))
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
                t_node = torch.full((h_atoms.size(0), 1), float(t.item()), device=h_atoms.device, dtype=h_atoms.dtype)
            else:
                t_node = t[mask_atoms].to(h_atoms.dtype)
            feats = torch.cat([feats, t_node], dim=-1)

        if self.params.condition_temperature:
            if temperature is None:
                raise ValueError("temperature is required when condition_temperature=True")
            if temperature.numel() == 1:
                temp_node = torch.full(
                    (h_atoms.size(0), 1), float(temperature.item()), device=h_atoms.device, dtype=h_atoms.dtype
                )
            else:
                temp_node = temperature[mask_atoms].to(h_atoms.dtype)
            feats = torch.cat([feats, temp_node], dim=-1)

        h = self.node_mlp(feats)

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
