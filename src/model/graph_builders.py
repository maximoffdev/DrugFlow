from __future__ import annotations

import torch


@torch.jit.script
def safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    # TorchScript-friendly norm with epsilon.
    return torch.sqrt(torch.clamp(torch.sum(x * x, dim=dim), min=eps))


@torch.jit.script
def _ranks_from_argsort(order: torch.Tensor) -> torch.Tensor:
    """Convert argsort indices to ranks.

    Args:
        order: (N, M) where each row is a permutation of 0..M-1

    Returns:
        ranks: (N, M) where ranks[i, j] is rank of j in row i.
    """
    n = order.size(0)
    m = order.size(1)
    ranks = torch.empty((n, m), device=order.device, dtype=torch.long)
    arange = torch.arange(m, device=order.device, dtype=torch.long).unsqueeze(0).expand(n, m)
    ranks.scatter_(1, order, arange)
    return ranks


@torch.jit.script
def build_biknn_radius_graph_no_pbc(
    pos: torch.Tensor,
    cutoff: float,
    k: int,
    start_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build biKNN radius graph for a single system (no PBC).

    keep (i->j) iff
      - j is among i's k nearest neighbors
      - i is among j's k nearest neighbors
      - dist(i,j) < cutoff
      - i != j

    Returns:
        index1: (E,) source indices (global, with start_index offset)
        index2: (E,) destination indices (global, with start_index offset)
    """
    n = pos.size(0)
    if n == 0:
        empty = torch.empty((0,), device=pos.device, dtype=torch.long)
        return empty, empty

    # Pairwise distances.
    disp = pos[:, None, :] - pos[None, :, :]
    dist = safe_norm(disp, dim=-1)

    # Exclude self.
    diag = torch.eye(n, device=pos.device, dtype=torch.bool)
    dist = dist.masked_fill(diag, float("inf"))

    # Ranks: smaller distance => smaller rank.
    order = torch.argsort(dist, dim=1)
    ranks = _ranks_from_argsort(order)

    # biKNN + cutoff.
    src_ok = ranks < int(k)
    dst_ok = src_ok.transpose(0, 1)
    within = dist < float(cutoff)
    mask = src_ok & dst_ok & within

    index1, index2 = torch.where(mask)

    if start_index != 0:
        index1 = index1 + int(start_index)
        index2 = index2 + int(start_index)

    return index1.to(torch.long), index2.to(torch.long)


def build_batched_biknn_edges(
    pos: torch.Tensor,
    batch_mask: torch.Tensor,
    *,
    cutoff: float,
    k: int,
) -> torch.Tensor:
    """Build a (2, E) edge_index for a batched set of molecules.

    Uses the TorchScript kernel per-molecule to match biKNN selection.

    Note: this assumes atoms of the same molecule are contiguous in `pos`.
    """
    device = pos.device
    edge_src: list[torch.Tensor] = []
    edge_dst: list[torch.Tensor] = []

    unique = torch.unique(batch_mask, sorted=True)
    for b in unique.tolist():
        idx = torch.where(batch_mask == int(b))[0]
        if idx.numel() == 0:
            continue
        start_index = int(idx.min().item())
        pos_b = pos[idx]
        i1, i2 = build_biknn_radius_graph_no_pbc(pos_b, cutoff=float(cutoff), k=int(k), start_index=start_index)
        edge_src.append(i1)
        edge_dst.append(i2)

    if len(edge_src) == 0:
        return torch.empty((2, 0), device=device, dtype=torch.long)

    src = torch.cat(edge_src, dim=0)
    dst = torch.cat(edge_dst, dim=0)
    return torch.stack([src, dst], dim=0)


def build_batched_fully_connected_edges(
    pos: torch.Tensor,
    batch_mask: torch.Tensor,
    *,
    cutoff: float,
    k: int,
) -> torch.Tensor:
    """Build a (2, E) fully-connected directed edge_index per molecule (no self loops).

    This has the same signature as `build_batched_biknn_edges` for easy swapping.
    The `cutoff` and `k` arguments are accepted but ignored.

    Returns:
        edge_index: (2, E) with all pairs (i -> j) for i != j within each molecule.
    """
    del pos, cutoff, k

    device = batch_mask.device
    edge_src: list[torch.Tensor] = []
    edge_dst: list[torch.Tensor] = []

    unique = torch.unique(batch_mask, sorted=True)
    for b in unique.tolist():
        idx = torch.where(batch_mask == int(b))[0]  # global indices for molecule b
        n = int(idx.numel())
        if n <= 1:
            continue

        # Create all ordered pairs (r, c) with r != c, in local indexing.
        r = torch.arange(n, device=device, dtype=torch.long).repeat_interleave(n)
        c = torch.arange(n, device=device, dtype=torch.long).repeat(n)
        keep = r != c

        edge_src.append(idx[r[keep]])
        edge_dst.append(idx[c[keep]])

    if len(edge_src) == 0:
        return torch.empty((2, 0), device=device, dtype=torch.long)

    src = torch.cat(edge_src, dim=0)
    dst = torch.cat(edge_dst, dim=0)
    return torch.stack([src, dst], dim=0)
