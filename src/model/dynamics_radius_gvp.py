from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from src.model.gvp import GVPModel
from src.model.graph_builders import build_batched_biknn_edges, build_batched_fully_connected_edges
from src.model.dynamics_radius_mlp import SinusoidalTimeEmbeddings


@dataclass(frozen=True)
class RadiusGVPParams:
    # Node feature sizes
    hidden_scalar_nf: int = 128

    # Conditioning
    condition_time: bool = True
    condition_temperature: bool = True

    # Conditioning architecture (diffusion-style)
    time_emb_dim: int = 64

    # Radius graph
    edge_cutoff_ligand: float | None = None
    knn_k: int = 30

    # Graph construction
    #  - "biknn": biKNN + cutoff radius graph (default)
    #  - "fc": fully-connected within each molecule (no self loops)
    graph_kind: str = "biknn"

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
            - v: (N, 3)
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

        # Diffusion-style conditioning embedding.
        # Keep this internal to the dynamics module; call sites only pass scalars t/temperature.
        self.cond_dim = 0
        if params.condition_time or params.condition_temperature:
            time_emb_dim = int(getattr(params, "time_emb_dim", 64))
            if time_emb_dim % 2 != 0:
                time_emb_dim += 1
            time_emb_dim = max(4, time_emb_dim)
            self.cond_dim = time_emb_dim

        self.time_mlp = None
        if params.condition_time:
            self.time_mlp = nn.Sequential(
                SinusoidalTimeEmbeddings(self.cond_dim),
                nn.Linear(self.cond_dim, self.cond_dim),
                nn.SiLU(),
                nn.Linear(self.cond_dim, self.cond_dim),
            )

        self.temperature_mlp = None
        if params.condition_temperature:
            self.temperature_mlp = nn.Sequential(
                nn.Linear(1, self.cond_dim),
                nn.SiLU(),
                nn.Linear(self.cond_dim, self.cond_dim),
            )

        node_in_scalar = params.hidden_scalar_nf + self.cond_dim

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

        graph_kind = str(getattr(self.params, "graph_kind", "biknn")).lower()
        if graph_kind in {"fc", "fully_connected", "fully-connected"}:
            return build_batched_fully_connected_edges(x, batch_mask, cutoff=cutoff, k=int(self.params.knn_k))
        if graph_kind in {"biknn", "knn", "radius"}:
            return build_batched_biknn_edges(x, batch_mask, cutoff=cutoff, k=int(self.params.knn_k))
        raise ValueError(f"Unknown graph_kind={graph_kind!r}; expected 'biknn' or 'fc'")

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

        # Conditioning embedding (time/temperature). If disabled, it contributes zeros.
        if self.cond_dim > 0:
            cond = torch.zeros((h.size(0), self.cond_dim), device=h.device, dtype=h.dtype)
        else:
            cond = None

        if self.params.condition_time:
            if t is None:
                raise ValueError("t is required when condition_time=True")
            if t.numel() == 1:
                # Preserve autograd: avoid converting to Python float.
                t_node = t.to(device=h.device, dtype=h.dtype).view(1, 1).expand(h.size(0), 1)
            else:
                t_node = t[mask_atoms].to(h.dtype)
            if cond is not None and self.time_mlp is not None:
                cond = cond + self.time_mlp(t_node)

        if self.params.condition_temperature:
            if temperature is None:
                raise ValueError("temperature is required when condition_temperature=True")
            if temperature.numel() == 1:
                # Preserve autograd: avoid converting to Python float.
                temp_node = temperature.to(device=h.device, dtype=h.dtype).view(1, 1).expand(h.size(0), 1)
            else:
                temp_node = temperature[mask_atoms].to(h.dtype)
            if cond is not None and self.temperature_mlp is not None:
                cond = cond + self.temperature_mlp(temp_node)

        if cond is not None:
            h = torch.cat([h, cond], dim=-1)

        edges = self._build_edges(x_atoms, mask_atoms)

        h_final, v, _ = self.net(h, x_atoms, edges, v=None, batch_mask=mask_atoms, edge_attr=None)

        logits_h = self.atom_decoder(h_final)

        ef_feats = torch.cat([h_final, x_atoms.to(h_final.dtype)], dim=-1)
        energy_node = self.energy_node_head(ef_feats).squeeze(-1)
        energy = scatter_mean(energy_node, mask_atoms, dim=0)

        pred_ligand = {
            "logits_h": logits_h,
            "energy": energy,
            "force": v,
        }
        pred_residues = {}
        return pred_ligand, pred_residues
