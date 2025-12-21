from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from src.model.gvp import GVPModel
from src.model.graph_builders import build_batched_biknn_edges, build_batched_fully_connected_edges


@dataclass(frozen=True)
class RadiusGVPParams:
    # Node feature sizes
    hidden_scalar_nf: int = 128

    # Conditioning
    condition_time: bool = True
    condition_temperature: bool = True

    # Radius graph
    edge_cutoff_ligand: float | None = None
    knn_k: int = 30

    # GVP backbone params (match DrugFlow configs)
    n_layers: int = 5
    node_h_dim_s: int = 128
    node_h_dim_v: int = 32
    edge_h_dim_s: int = 128
    edge_h_dim_v: int = 32
    dropout: float = 0.0
    vector_gate: bool = True
    reflection_equivariant: bool = False
    d_max: float = 15.0
    num_rbf: int = 16


class RadiusGVPDynamics(nn.Module):
    """Ligand-only radius-graph dynamics using the same GVP backbone as DrugFlow.

    This mirrors DrugFlow's use of `GVPModel` (via `Dynamics`), but simplified to a
    ligand-only radius graph with no bond types and no pocket nodes.

    Outputs:
      - logits_h: (N, atom_nf)
      - energy: (B,)
      - force: (N, 3)
    """

    def __init__(self, atom_nf: int, x_dim: int = 3, params: RadiusGVPParams = RadiusGVPParams()):
        super().__init__()
        self.atom_nf = int(atom_nf)
        self.x_dim = int(x_dim)
        self.params = params

        # Encode atom type probabilities/one-hot into scalar node features.
        self.atom_encoder = nn.Sequential(
            nn.Linear(self.atom_nf, params.hidden_scalar_nf),
            nn.SiLU(),
            nn.Linear(params.hidden_scalar_nf, params.hidden_scalar_nf),
        )

        extra = (1 if params.condition_time else 0) + (1 if params.condition_temperature else 0)
        node_in_scalar = params.hidden_scalar_nf + extra

        self.net = GVPModel(
            node_in_dim=(node_in_scalar, 0),
            node_h_dim=(params.node_h_dim_s, params.node_h_dim_v),
            node_out_nf=params.hidden_scalar_nf,
            edge_in_nf=0,
            edge_h_dim=(params.edge_h_dim_s, params.edge_h_dim_v),
            edge_out_nf=0,
            num_layers=int(params.n_layers),
            drop_rate=float(params.dropout),
            vector_gate=bool(params.vector_gate),
            reflection_equiv=bool(params.reflection_equivariant),
            d_max=float(params.d_max),
            num_rbf=int(params.num_rbf),
            update_edge_attr=False,
        )

        self.atom_decoder = nn.Sequential(
            nn.Linear(params.hidden_scalar_nf, 2 * params.hidden_scalar_nf),
            nn.SiLU(),
            nn.Linear(2 * params.hidden_scalar_nf, self.atom_nf),
        )

        ef_in_dim = params.hidden_scalar_nf + self.x_dim
        self.energy_node_head = nn.Sequential(
            nn.Linear(ef_in_dim, params.hidden_scalar_nf),
            nn.SiLU(),
            nn.Linear(params.hidden_scalar_nf, 1),
        )

    def _build_edges(self, x: torch.Tensor, batch_mask: torch.Tensor) -> torch.Tensor:
        # biKNN hard radius graph (matches FairChem/OMol-style selection logic).
        # If no cutoff is provided, treat as effectively infinite.
        cutoff = float(self.params.edge_cutoff_ligand) if self.params.edge_cutoff_ligand is not None else 1e9

        # Old behavior (biKNN radius graph):
        # return build_batched_biknn_edges(x, batch_mask, cutoff=cutoff, k=int(self.params.knn_k))

        # New behavior: fully-connected graph within each molecule (no self-loops).
        return build_batched_fully_connected_edges(x, batch_mask, cutoff=cutoff, k=int(self.params.knn_k))

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
        # Node scalars
        h = self.atom_encoder(h_atoms)

        if self.params.condition_time:
            if t is None:
                raise ValueError("t is required when condition_time=True")
            if t.numel() == 1:
                t_node = torch.full((h.size(0), 1), float(t.item()), device=h.device, dtype=h.dtype)
            else:
                t_node = t[mask_atoms].to(h.dtype)
            h = torch.cat([h, t_node], dim=-1)

        if self.params.condition_temperature:
            if temperature is None:
                raise ValueError("temperature is required when condition_temperature=True")
            if temperature.numel() == 1:
                temp_node = torch.full((h.size(0), 1), float(temperature.item()), device=h.device, dtype=h.dtype)
            else:
                temp_node = temperature[mask_atoms].to(h.dtype)
            h = torch.cat([h, temp_node], dim=-1)

        edges = self._build_edges(x_atoms, mask_atoms)

        h_final, force, _ = self.net(h, x_atoms, edges, v=None, batch_mask=mask_atoms, edge_attr=None)

        logits_h = self.atom_decoder(h_final)

        ef_feats = torch.cat([h_final, x_atoms.to(h_final.dtype)], dim=-1)
        energy_node = self.energy_node_head(ef_feats).squeeze(-1)
        energy = scatter_mean(energy_node, mask_atoms, dim=0)

        pred_ligand = {
            "logits_h": logits_h,
            "energy": energy,
            "force": force,
        }
        pred_residues = {}
        return pred_ligand, pred_residues
