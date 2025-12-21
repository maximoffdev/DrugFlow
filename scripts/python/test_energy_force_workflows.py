"""Comprehensive tensor-only workflow test for EnergyForceDiffusion.

Covers:
  - Training smoke: one forward/backward/optimizer step
  - Sampling matrix:
      * Coordinates: ODE vs SDE-like (stochastic reverse step)
      * Atom types: Markov bridge vs score-based logit diffusion

This script is intentionally tensor-only (no RDKit/SDF).

Example (synthetic batch):
  python scripts/python/test_energy_force_workflows.py --device cpu --n-samples 2 --num-nodes 16

Example (real PKL batch for training smoke):
  python scripts/python/test_energy_force_workflows.py \
    --datadir /path/to/OM25_pkl/OM25_1M_HCNOSPClF_pkl \
    --allowed-z 1,6,7,8,9,15,16,17 --batch-size 4

Example (load checkpoint and sample):
  python scripts/python/test_energy_force_workflows.py --checkpoint path/to.ckpt --n-samples 4 --num-nodes 16,18,20,22
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, List, Literal, Sequence, cast

import torch

# Ensure imports work when called from repo root or elsewhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from src.model.energy_force_diffusion import EnergyForceDiffusion
from src.model.flows import CategoricalLogitScoreDiffusion, CoordScoreDiffusion, SDEParams
from src.model.graph_builders import build_batched_biknn_edges, build_batched_fully_connected_edges


def _assert_monotonic_nonincreasing(name: str, x: torch.Tensor, *, atol: float = 0.0):
    if x.numel() <= 1:
        return
    # allow tiny numerical noise via atol
    dx = x[1:] - x[:-1]
    if not torch.all(dx <= atol):
        i = int(torch.where(dx > atol)[0][0].item())
        raise AssertionError(f"{name} is not non-increasing at i={i}: {float(x[i])} -> {float(x[i+1])}")


def _assert_monotonic_nondecreasing(name: str, x: torch.Tensor, *, atol: float = 0.0):
    if x.numel() <= 1:
        return
    dx = x[1:] - x[:-1]
    if not torch.all(dx >= -atol):
        i = int(torch.where(dx < -atol)[0][0].item())
        raise AssertionError(f"{name} is not non-decreasing at i={i}: {float(x[i])} -> {float(x[i+1])}")


def _run_schedule_sanity(args, device: torch.device):
    """Sanity-check VE/VP schedules under DrugFlow time convention.

    Convention: t=0 noisy -> t=1 clean.
    We assert:
      - sigma(t) is non-increasing
      - for VP: alpha(t) is non-decreasing and alpha(1)=1, sigma(1)=0
      - for VE: alpha(t)=1, sigma(0)=sigma_max, sigma(1)=sigma_min
    """
    n = int(getattr(args, "schedule_n", 200))
    if n < 3:
        raise ValueError("--schedule-n must be >= 3")

    coord_kind = str(getattr(args, "coord_sde_kind", "vp"))
    logit_kind = str(getattr(args, "logit_sde_kind", coord_kind))

    vp_beta_min = float(getattr(args, "vp_beta_min", 0.1))
    vp_beta_max = float(getattr(args, "vp_beta_max", 20.0))
    vp_sigma_scale = float(getattr(args, "vp_sigma_scale", 1.0))

    t = torch.linspace(0.0, 1.0, n, device=device, dtype=torch.get_default_dtype()).unsqueeze(-1)

    coord = CoordScoreDiffusion(
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
        dim=3,
        sde=SDEParams(
            kind=cast(Literal["ve", "vp"], coord_kind),
            sigma_min=float(args.sigma_min),
            sigma_max=float(args.sigma_max),
            beta_min=vp_beta_min,
            beta_max=vp_beta_max,
            vp_sigma_scale=vp_sigma_scale,
        ),
    )
    logit = CategoricalLogitScoreDiffusion(
        sigma_min=float(args.sigma_min_h),
        sigma_max=float(args.sigma_max_h),
        n_classes=8,
        sde=SDEParams(
            kind=cast(Literal["ve", "vp"], logit_kind),
            sigma_min=float(args.sigma_min_h),
            sigma_max=float(args.sigma_max_h),
            beta_min=vp_beta_min,
            beta_max=vp_beta_max,
            vp_sigma_scale=vp_sigma_scale,
        ),
    )

    def check_one(name: str, diffusion) -> None:
        temps = [None]
        if bool(getattr(args, "condition_temperature", False)):
            temps = [torch.tensor([[0.5]], device=device, dtype=t.dtype), torch.tensor([[2.0]], device=device, dtype=t.dtype)]

        for temp in temps:
            a = diffusion.alpha(t, temperature=temp).squeeze(-1)
            s = diffusion.sigma(t, temperature=temp).squeeze(-1)

            _assert_finite(f"{name}.alpha", a)
            _assert_finite(f"{name}.sigma", s)
            if (a < -1e-6).any() or (a > 1.0 + 1e-6).any():
                raise AssertionError(f"{name}.alpha out of expected range [0,1]")
            if (s < -1e-6).any():
                raise AssertionError(f"{name}.sigma has negative values")

            _assert_monotonic_nonincreasing(f"{name}.sigma", s, atol=1e-10)

            kind = str(diffusion.sde.kind)
            if kind == "ve":
                if not torch.allclose(a, torch.ones_like(a), atol=0.0, rtol=0.0):
                    raise AssertionError(f"{name} VE should have alpha(t)=1")
                if not torch.isclose(s[0], torch.tensor(float(diffusion.sde.sigma_max), device=device, dtype=s.dtype), atol=1e-6, rtol=0.0):
                    raise AssertionError(f"{name} VE sigma(0) != sigma_max")
                if not torch.isclose(s[-1], torch.tensor(float(diffusion.sde.sigma_min), device=device, dtype=s.dtype), atol=1e-6, rtol=0.0):
                    raise AssertionError(f"{name} VE sigma(1) != sigma_min")
            elif kind == "vp":
                _assert_monotonic_nondecreasing(f"{name}.alpha", a, atol=1e-10)
                if not torch.isclose(a[-1], torch.tensor(1.0, device=device, dtype=a.dtype), atol=1e-6, rtol=0.0):
                    raise AssertionError(f"{name} VP alpha(1) != 1")
                # sigma at t=1 should be 0 even with vp_sigma_scale (since alpha(1)=1)
                if not torch.isclose(s[-1], torch.tensor(0.0, device=device, dtype=s.dtype), atol=1e-6, rtol=0.0):
                    raise AssertionError(f"{name} VP sigma(1) != 0")
            else:
                raise AssertionError(f"Unknown kind: {kind}")

    check_one("coord", coord)
    check_one("logit", logit)


def _parse_int_list(csv: str) -> List[int]:
    parts = [p.strip() for p in csv.split(",") if p.strip()]
    return [int(x) for x in parts]


def _parse_num_nodes(num_nodes: str, n_samples: int) -> int | torch.Tensor:
    if "," not in num_nodes:
        return int(num_nodes)
    sizes = _parse_int_list(num_nodes)
    if len(sizes) != n_samples:
        raise ValueError(f"--num-nodes provides {len(sizes)} sizes but --n-samples is {n_samples}")
    return torch.tensor(sizes, dtype=torch.long)


def _make_batch_mask_from_sizes(sizes: Sequence[int], *, device: torch.device) -> torch.Tensor:
    masks = []
    for i, n in enumerate(sizes):
        n = int(n)
        if n < 0:
            raise ValueError("sizes must be non-negative")
        if n == 0:
            continue
        masks.append(torch.full((n,), i, device=device, dtype=torch.long))
    if not masks:
        return torch.empty((0,), device=device, dtype=torch.long)
    return torch.cat(masks, dim=0)


def run_graph_builder_checks(args, device: torch.device):
    """Check graph batching correctness for FC and biKNN builders.

    Invariants:
      - No cross-molecule edges: batch[src] == batch[dst]
      - No self loops: src != dst
      - Edge count matches expectation for special parameter choices
    """
    graph_kind = str(getattr(args, "graph_kind", "both"))
    if graph_kind not in {"fc", "biknn", "both"}:
        raise ValueError("--graph-kind must be one of: fc, biknn, both")

    # Synthetic sizes; keep small to make checks cheap.
    sizes = [3, 5, 1]
    batch_mask = _make_batch_mask_from_sizes(sizes, device=device)
    n_total = int(batch_mask.numel())
    if n_total == 0:
        raise AssertionError("Empty synthetic batch_mask")

    # Make positions deterministic and separated per molecule.
    torch.manual_seed(7)
    pos = torch.randn((n_total, 3), device=device, dtype=torch.get_default_dtype())
    # shift each molecule far apart so cutoff tests are robust
    offsets = torch.tensor([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [200.0, 0.0, 0.0]], device=device, dtype=pos.dtype)
    pos = pos + offsets[batch_mask]

    def assert_basic(edge_index: torch.Tensor, name: str):
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise AssertionError(f"{name}: edge_index must have shape (2,E)")
        if edge_index.dtype != torch.long:
            raise AssertionError(f"{name}: edge_index must be torch.long")
        if edge_index.numel() == 0:
            return
        src = edge_index[0]
        dst = edge_index[1]
        if (src < 0).any() or (dst < 0).any() or (src >= n_total).any() or (dst >= n_total).any():
            raise AssertionError(f"{name}: edge indices out of bounds")
        if torch.any(src == dst):
            raise AssertionError(f"{name}: found self loops")
        if not torch.all(batch_mask[src] == batch_mask[dst]):
            bad = torch.where(batch_mask[src] != batch_mask[dst])[0][0].item()
            raise AssertionError(
                f"{name}: found cross-molecule edge at e={bad}: src_batch={int(batch_mask[src[bad]])} dst_batch={int(batch_mask[dst[bad]])}"
            )

    # Fully-connected: expect sum n*(n-1) for n>1
    if graph_kind in {"fc", "both"}:
        fc = build_batched_fully_connected_edges(pos, batch_mask, cutoff=1e9, k=999)
        assert_basic(fc, "fully_connected")
        expected = sum(n * (n - 1) for n in sizes if n > 1)
        if int(fc.size(1)) != int(expected):
            raise AssertionError(f"fully_connected: expected E={expected}, got {int(fc.size(1))}")

    # biKNN: if k >= n and cutoff large, it should also yield all non-self directed pairs per molecule.
    if graph_kind in {"biknn", "both"}:
        cutoff = float(getattr(args, "graph_cutoff", 1e9))
        k = int(getattr(args, "graph_k", 999))
        biknn = build_batched_biknn_edges(pos, batch_mask, cutoff=cutoff, k=k)
        assert_basic(biknn, "biknn")

        # Only assert edge count when k is large enough to include all neighbors.
        expected_all = 0
        for n in sizes:
            if n <= 1:
                continue
            if k >= n:
                expected_all += n * (n - 1)
            else:
                expected_all = None
                break
        if expected_all is not None:
            if int(biknn.size(1)) != int(expected_all):
                raise AssertionError(f"biknn: expected E={expected_all} for k>=n, got {int(biknn.size(1))}")

        # Negative checks: ensure cutoff and k are actually enforced.
        biknn_zero_cutoff = build_batched_biknn_edges(pos, batch_mask, cutoff=0.0, k=max(1, k))
        assert_basic(biknn_zero_cutoff, "biknn_zero_cutoff")
        if int(biknn_zero_cutoff.size(1)) != 0:
            raise AssertionError(f"biknn_zero_cutoff: expected E=0, got {int(biknn_zero_cutoff.size(1))}")

        biknn_k0 = build_batched_biknn_edges(pos, batch_mask, cutoff=float(cutoff), k=0)
        assert_basic(biknn_k0, "biknn_k0")
        if int(biknn_k0.size(1)) != 0:
            raise AssertionError(f"biknn_k0: expected E=0, got {int(biknn_k0.size(1))}")


def _to_device(batch: dict, device: torch.device) -> dict:
    # batch is a dict: ligand/pocket are TensorDict (tensor-like mapping)
    out = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out[k] = v.to(device)
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _load_any(path: Path) -> Any:
    """Load a PKL that may be a torch-saved object or a pickled python object."""
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)


def _get(obj: Any, key: str):
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)


def _collate_items(items: list[dict], *, device: torch.device, dtype: torch.dtype) -> dict:
    """Collate a list of per-molecule items into the format expected by EnergyForceDiffusion."""
    xs: list[torch.Tensor] = []
    hs: list[torch.Tensor] = []
    forces: list[torch.Tensor] = []
    energies: list[torch.Tensor] = []
    names: list[str] = []
    masks: list[torch.Tensor] = []
    sizes: list[int] = []

    offset = 0
    for i, it in enumerate(items):
        lig = it["ligand"]
        x = lig["x"].to(device=device, dtype=dtype)
        h = lig["one_hot"].to(device=device, dtype=dtype)
        n = int(x.size(0))
        if h.ndim != 2 or h.size(0) != n:
            raise ValueError("Invalid one_hot shape")

        xs.append(x)
        hs.append(h)
        sizes.append(n)
        names.append(str(lig.get("name", f"sample_{i}")))
        masks.append(torch.full((n,), i, device=device, dtype=torch.long))

        f = it.get("force", torch.zeros((n, 3), device=device, dtype=dtype))
        forces.append(f.to(device=device, dtype=dtype))

        e = it.get("energy", torch.zeros((1,), device=device, dtype=dtype))
        e = torch.as_tensor(e, device=device, dtype=dtype).view(-1)
        if e.numel() != 1:
            raise ValueError("Energy must be scalar per molecule")
        energies.append(e)

        offset += n

    x_cat = torch.cat(xs, dim=0) if xs else torch.empty((0, 3), device=device, dtype=dtype)
    h_cat = torch.cat(hs, dim=0) if hs else torch.empty((0, 0), device=device, dtype=dtype)
    mask = torch.cat(masks, dim=0) if masks else torch.empty((0,), device=device, dtype=torch.long)
    force = torch.cat(forces, dim=0) if forces else torch.empty((0, 3), device=device, dtype=dtype)
    energy = torch.cat(energies, dim=0) if energies else torch.empty((0,), device=device, dtype=dtype)
    size = torch.tensor(sizes, device=device, dtype=torch.long)

    ligand = {
        "name": names,
        "x": x_cat,
        "one_hot": h_cat,
        "mask": mask,
        "size": size,
    }
    pocket = {
        "name": names,
        "x": torch.zeros((0, 3), device=device, dtype=dtype),
        "one_hot": torch.zeros((0, 0), device=device, dtype=dtype),
        "mask": torch.zeros((0,), device=device, dtype=torch.long),
        "size": torch.zeros((len(names),), device=device, dtype=torch.long),
    }
    return {"ligand": ligand, "pocket": pocket, "energy": energy, "force": force}


def _load_pkl_items(
    datadir: Path,
    *,
    allowed_z: list[int],
    batch_size: int,
    file_limit: int | None,
    pos_key: str = "pos",
    atom_key: str = "atomic_numbers",
    energy_key: str = "e_total",
    forces_key: str = "forces",
) -> list[dict]:
    files = sorted(datadir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No *.pkl files found in {datadir}")
    if file_limit is not None:
        files = files[: int(file_limit)]
    files = files[: int(batch_size)]

    z_to_idx = {int(z): i for i, z in enumerate(allowed_z)}
    atom_nf = len(allowed_z)

    items: list[dict] = []
    for fp in files:
        obj = _load_any(fp)
        pos = torch.as_tensor(_get(obj, pos_key), dtype=torch.float32)
        z = torch.as_tensor(_get(obj, atom_key))
        if z.dtype.is_floating_point:
            z = z.round().to(torch.long)
        else:
            z = z.to(torch.long)

        # Map to contiguous indices
        idx = torch.empty_like(z)
        for i in range(int(z.numel())):
            zi = int(z[i].item())
            if zi not in z_to_idx:
                raise ValueError(f"Unknown atomic number {zi} in {fp.name}; allowed={allowed_z}")
            idx[i] = int(z_to_idx[zi])
        one_hot = torch.nn.functional.one_hot(idx, num_classes=atom_nf).to(torch.float32)

        energy = torch.as_tensor(_get(obj, energy_key), dtype=torch.float32).view(-1)
        if energy.numel() != 1:
            raise ValueError(f"Energy must be scalar in {fp.name}")

        force = torch.as_tensor(_get(obj, forces_key), dtype=torch.float32)
        if force.shape != pos.shape:
            raise ValueError(f"forces shape {tuple(force.shape)} != pos shape {tuple(pos.shape)} in {fp.name}")

        ligand = {"name": fp.name, "x": pos, "one_hot": one_hot, "size": int(pos.size(0))}
        items.append({"ligand": ligand, "energy": energy, "force": force})

    return items


def _cast_batch_floats(batch: dict, dtype: torch.dtype) -> dict:
    # Cast only floating tensors; keep indices/masks as-is.
    for entity in ("ligand", "pocket"):
        if entity not in batch:
            continue
        td = batch[entity]
        # TensorDict behaves like a mapping.
        for k in list(td.keys()):
            v = td[k]
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                td[k] = v.to(dtype)
        batch[entity] = td

    for k in ("energy", "force"):
        if k in batch and isinstance(batch[k], torch.Tensor) and batch[k].is_floating_point():
            batch[k] = batch[k].to(dtype)
    return batch


def _make_synthetic_items(
    *,
    batch_size: int,
    num_nodes: Sequence[int],
    atom_nf: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict]:
    items: list[dict] = []
    for i in range(batch_size):
        n = int(num_nodes[i])
        pos = torch.randn((n, 3), device=device, dtype=dtype)
        labels = torch.randint(low=0, high=atom_nf, size=(n,), device=device)
        one_hot = torch.nn.functional.one_hot(labels, num_classes=atom_nf).to(dtype)

        ligand = {"name": f"synthetic_{i}", "x": pos, "one_hot": one_hot, "size": n}
        pocket = {
            "name": f"synthetic_{i}",
            "x": torch.zeros((0, 3), device=device, dtype=pos.dtype),
            "one_hot": torch.zeros((0, 0), device=device, dtype=one_hot.dtype),
            "size": 0,
        }
        energy = torch.zeros((1,), device=device, dtype=dtype)
        forces = torch.zeros((n, 3), device=device, dtype=dtype)
        items.append({"ligand": ligand, "pocket": pocket, "energy": energy, "force": forces})
    return items


def _assert_finite(name: str, tensor: torch.Tensor):
    if not torch.isfinite(tensor).all():
        bad = (~torch.isfinite(tensor)).nonzero(as_tuple=False)
        raise AssertionError(f"{name} contains non-finite values; first bad index: {bad[0].tolist() if bad.numel() else 'unknown'}")


def _validate_sample(sample: dict, *, atom_nf: int, n_samples: int):
    x = sample["x"]
    one_hot = sample["one_hot"]
    mask = sample["mask"]
    size = sample["size"]

    assert isinstance(x, torch.Tensor) and x.ndim == 2 and x.size(-1) == 3
    assert isinstance(one_hot, torch.Tensor) and one_hot.ndim == 2 and one_hot.size(-1) == atom_nf
    assert isinstance(mask, torch.Tensor) and mask.ndim == 1
    assert isinstance(size, torch.Tensor) and size.ndim == 1 and int(size.numel()) == n_samples

    assert int(mask.max().item()) + 1 == n_samples
    assert x.size(0) == one_hot.size(0) == mask.size(0)

    _assert_finite("x", x)
    _assert_finite("one_hot", one_hot)

    row_sums = one_hot.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=0.0):
        # allow minor numerical drift for the probability version; but finalizer should make it exact
        # if it isn't close, report
        raise AssertionError(f"one_hot rows do not sum to 1 (min={float(row_sums.min())}, max={float(row_sums.max())})")

    # COM per molecule should be near 0 because sampler initializes COM=0.
    # We keep a loose tolerance because stochastic sampling can drift slightly.
    coms = []
    for b in range(n_samples):
        xb = x[mask == b]
        com = xb.mean(dim=0)
        coms.append(com)
    coms = torch.stack(coms, dim=0)
    max_com = float(coms.norm(dim=-1).max().item())
    if math.isnan(max_com) or max_com > 5.0:
        raise AssertionError(f"COM drift too large: max ||COM|| = {max_com}")


def _build_fresh_model(args, device: torch.device) -> EnergyForceDiffusion:
    train_params = Namespace(
        datadir=str(args.datadir) if args.datadir is not None else str(REPO_ROOT),
        batch_size=int(args.batch_size),
        num_workers=0,
        lr=float(args.lr),
        pkl_allowed_atomic_numbers=_parse_int_list(args.allowed_z) if args.allowed_z else [1, 6, 7, 8, 9, 15, 16, 17],
        pkl_file_limit=int(args.file_limit) if args.file_limit is not None else None,
        pkl_val_fraction=0.01,
        pkl_val_n=None,
        pkl_split_seed=42,
        pkl_pos_key="pos",
        pkl_atom_types_key="atomic_numbers",
        pkl_energy_key="e_total",
        pkl_forces_key="forces",
        pkl_atom_types_format="atomic_numbers",
    )

    loss_params = Namespace(
        lambda_x=1.0,
        lambda_h=1.0,
        reduce="mean",
        discrete_loss="CE",
        x_score_weight_by_sigma=bool(args.x_score_weight_by_sigma),
    )

    eval_params = Namespace(eval_batch_size=int(args.batch_size))

    condition_time = not bool(getattr(args, "no_condition_time", False))
    predictor_params = Namespace(
        backbone=str(getattr(args, "backbone", "radius_mlp")),
        condition_time=condition_time,
        condition_temperature=bool(getattr(args, "condition_temperature", False)),
        radius_mlp=Namespace(
            hidden_dim=int(args.hidden_dim),
            n_layers=int(args.n_layers),
            condition_time=condition_time,
            condition_temperature=bool(getattr(args, "condition_temperature", False)),
        ),
    )

    simulation_params = Namespace(
        n_steps=int(args.timesteps),
        temperature=float(args.temperature),
        temperature_min=float(args.temperature),
        temperature_max=float(args.temperature),
        sigma_min=float(args.sigma_min),
        sigma_max=float(args.sigma_max),
        sigma_min_h=float(args.sigma_min_h),
        sigma_max_h=float(args.sigma_max_h),
        # Schedule selection
        coord_sde_kind=str(getattr(args, "coord_sde_kind", "vp")),
        logit_sde_kind=str(getattr(args, "logit_sde_kind", str(getattr(args, "coord_sde_kind", "vp")))),
        vp_beta_min=float(getattr(args, "vp_beta_min", 0.1)),
        vp_beta_max=float(getattr(args, "vp_beta_max", 20.0)),
        # Also set legacy names used by some YAMLs for compatibility.
        beta_min=float(getattr(args, "vp_beta_min", 0.1)),
        beta_max=float(getattr(args, "vp_beta_max", 20.0)),
        vp_sigma_scale=float(getattr(args, "vp_sigma_scale", 1.0)),
        use_temperature=bool(getattr(args, "condition_temperature", False)),
    )

    model = EnergyForceDiffusion(
        pocket_representation="none",
        train_params=train_params,
        loss_params=loss_params,
        eval_params=eval_params,
        predictor_params=predictor_params,
        simulation_params=simulation_params,
        virtual_nodes=None,
        flexible=False,
        flexible_bb=False,
        debug=True,
        overfit=False,
    )

    model.to(device)
    return model


def run_training_smoke(model: EnergyForceDiffusion, args, device: torch.device):
    model.train()
    dtype = next(model.parameters()).dtype

    if args.datadir is not None:
        allowed_z = _parse_int_list(args.allowed_z) if args.allowed_z else model.allowed_atomic_numbers
        items = _load_pkl_items(Path(args.datadir), allowed_z=allowed_z, batch_size=int(args.batch_size), file_limit=args.file_limit)
        batch = _collate_items(items, device=device, dtype=dtype)
    else:
        # synthetic batch
        bs = int(args.batch_size)
        sizes = [int(args.synthetic_nodes)] * bs
        items = _make_synthetic_items(batch_size=bs, num_nodes=sizes, atom_nf=model.atom_nf, device=device, dtype=dtype)
        batch = _collate_items(items, device=device, dtype=dtype)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), amsgrad=True, weight_decay=1e-12)

    loss, info = model.compute_loss(
        batch["ligand"],
        return_info=True,
        temperature=float(args.temperature),
        t=torch.full((int(args.batch_size), 1), float(args.t_value), device=device, dtype=next(model.parameters()).dtype)
        if args.freeze_t
        else None,
    )
    if not torch.isfinite(loss):
        raise AssertionError(f"compute_loss loss is not finite: {loss}")
    if not (math.isfinite(info["loss_x"]) and math.isfinite(info["loss_h"])):
        raise AssertionError(f"Non-finite loss components: {info}")

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    # Validation step should also be finite
    model.eval()
    with torch.no_grad():
        val_loss, val_info = model.compute_loss(
            batch["ligand"],
            return_info=True,
            temperature=float(args.temperature),
            t=torch.full((int(args.batch_size), 1), float(args.t_value), device=device, dtype=next(model.parameters()).dtype)
            if args.freeze_t
            else None,
        )
        if not torch.isfinite(val_loss):
            raise AssertionError(f"validation compute_loss loss is not finite: {val_loss}")
        if not (math.isfinite(val_info["loss_x"]) and math.isfinite(val_info["loss_h"])):
            raise AssertionError(f"Non-finite validation loss components: {val_info}")


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def run_training_loss_decrease(model: EnergyForceDiffusion, args, device: torch.device):
    """Run a short training loop and check the loss trend on real PKLs.

    This is not meant to be a full training benchmark; it is a regression guard:
    - confirms forward/backward works on real batches
    - checks that the optimizer can reduce the loss on a small slice of data
    """
    if args.datadir is None:
        raise ValueError("--datadir is required for loss-decrease test")

    model.train()
    dtype = next(model.parameters()).dtype

    datadir = Path(args.datadir)
    allowed_z = _parse_int_list(args.allowed_z) if args.allowed_z else model.allowed_atomic_numbers
    files = sorted(datadir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No *.pkl files found in {datadir}")
    if args.file_limit is not None:
        files = files[: int(args.file_limit)]
    if bool(args.shuffle_train):
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(files), generator=g).tolist()
        files = [files[i] for i in perm]
    if len(files) < int(args.batch_size):
        raise ValueError(f"Need at least batch-size={args.batch_size} PKLs, found {len(files)}")

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), amsgrad=True, weight_decay=1e-12)

    losses: list[float] = []
    losses_x: list[float] = []
    losses_h: list[float] = []
    first_batch = None
    t_fixed = None
    file_i = 0
    for step in range(int(args.train_iters)):
        if args.overfit_single_batch:
            if first_batch is None:
                items = _load_pkl_items(
                    datadir,
                    allowed_z=allowed_z,
                    batch_size=int(args.batch_size),
                    file_limit=args.file_limit,
                )
                first_batch = _collate_items(items, device=device, dtype=dtype)
            batch = first_batch
        else:
            start = file_i
            end = file_i + int(args.batch_size)
            if end > len(files):
                file_i = 0
                start = 0
                end = int(args.batch_size)
            batch_files = files[start:end]
            file_i = end
            items = []
            for fp in batch_files:
                obj = _load_any(fp)
                pos = torch.as_tensor(_get(obj, "pos"), dtype=torch.float32)
                z = torch.as_tensor(_get(obj, "atomic_numbers"))
                if z.dtype.is_floating_point:
                    z = z.round().to(torch.long)
                else:
                    z = z.to(torch.long)
                z_to_idx = {int(zv): i for i, zv in enumerate(allowed_z)}
                idx = torch.empty_like(z)
                for ii in range(int(z.numel())):
                    zi = int(z[ii].item())
                    if zi not in z_to_idx:
                        raise ValueError(f"Unknown atomic number {zi} in {fp.name}; allowed={allowed_z}")
                    idx[ii] = int(z_to_idx[zi])
                one_hot = torch.nn.functional.one_hot(idx, num_classes=len(allowed_z)).to(torch.float32)
                energy = torch.as_tensor(_get(obj, "e_total"), dtype=torch.float32).view(-1)
                force = torch.as_tensor(_get(obj, "forces"), dtype=torch.float32)
                ligand = {"name": fp.name, "x": pos, "one_hot": one_hot, "size": int(pos.size(0))}
                items.append({"ligand": ligand, "energy": energy, "force": force})
            batch = _collate_items(items, device=device, dtype=dtype)

        opt.zero_grad(set_to_none=True)
        if args.freeze_t and t_fixed is None:
            if args.t_value is not None:
                t_fixed = torch.full((int(args.batch_size), 1), float(args.t_value), device=device, dtype=dtype)
            else:
                # sample once and reuse
                t_fixed = torch.rand((int(args.batch_size), 1), device=device, dtype=dtype)
                # keep away from boundary to avoid clamping effects on t_next
                t_fixed = torch.clamp(t_fixed, 0.0, 1.0 - (1.0 / float(model.n_steps)) - 1e-4)

        loss, info = model.compute_loss(
            batch["ligand"],
            return_info=True,
            temperature=float(args.temperature) if args.freeze_temperature else None,
            t=t_fixed if args.freeze_t else None,
        )
        if not torch.isfinite(loss):
            raise AssertionError(f"Non-finite loss at step={step}: {loss}")
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))
        losses_x.append(float(info["loss_x"]))
        losses_h.append(float(info["loss_h"]))

        if args.print_every and (step % int(args.print_every) == 0):
            print(
                f"step={step:04d} loss={losses[-1]:.6f} "
                f"loss_x={losses_x[-1]:.6f} loss_h={losses_h[-1]:.6f}"
            )

    # Trend check: compare mean of first/last window.
    w = int(args.trend_window)
    w = max(1, min(w, len(losses) // 2))
    first = _mean(losses[:w])
    last = _mean(losses[-w:])

    first_x = _mean(losses_x[:w])
    last_x = _mean(losses_x[-w:])
    first_h = _mean(losses_h[:w])
    last_h = _mean(losses_h[-w:])

    print(f"loss trend:   first_mean({w})={first:.6f} last_mean({w})={last:.6f}")
    print(f"loss_x trend: first_mean({w})={first_x:.6f} last_mean({w})={last_x:.6f}")
    print(f"loss_h trend: first_mean({w})={first_h:.6f} last_mean({w})={last_h:.6f}")
    if args.require_decrease:
        min_frac = float(args.min_decrease_frac)
        if not (last < first * (1.0 - min_frac)):
            raise AssertionError(
                f"Loss did not decrease enough: first={first:.6f} last={last:.6f} "
                f"(required last < first*(1-{min_frac}))"
            )

    if args.require_decrease_x:
        min_frac = float(args.min_decrease_frac_x)
        if not (last_x < first_x * (1.0 - min_frac)):
            raise AssertionError(
                f"loss_x did not decrease enough: first={first_x:.6f} last={last_x:.6f} "
                f"(required last < first*(1-{min_frac}))"
            )

    if args.require_decrease_h:
        min_frac = float(args.min_decrease_frac_h)
        if not (last_h < first_h * (1.0 - min_frac)):
            raise AssertionError(
                f"loss_h did not decrease enough: first={first_h:.6f} last={last_h:.6f} "
                f"(required last < first*(1-{min_frac}))"
            )


def run_sampling_matrix(model: EnergyForceDiffusion, args, device: torch.device):
    model.eval()

    n_samples = int(args.n_samples)
    num_nodes = _parse_num_nodes(args.num_nodes, n_samples)
    timesteps = int(args.timesteps)

    cases = [
        ("ode", "markov_bridge", False, False),
        ("ode", "score", False, False),
        ("sde", "markov_bridge", True, False),
        ("sde", "score", True, True),
    ]

    results = []
    for x_sampler, h_sampler, stoch_x, stoch_h in cases:
        name = f"x={x_sampler} h={h_sampler} stoch_x={stoch_x} stoch_h={stoch_h}"
        sample = model.sample_ligand(
            n_samples=n_samples,
            num_nodes=num_nodes,
            timesteps=timesteps,
            temperature=float(args.temperature),
            x_sampler=x_sampler,
            h_sampler=h_sampler,
            stochastic_x=stoch_x,
            stochastic_h=stoch_h,
        )
        _validate_sample(sample, atom_nf=model.atom_nf, n_samples=n_samples)
        results.append((name, "OK"))

    # Print a compact summary table
    print("\nSampling matrix results:")
    for name, status in results:
        print(f"  - {status}: {name}")


def run_temperature_gating_invariance(model: EnergyForceDiffusion, args, device: torch.device):
    """If temperature conditioning is disabled, ensure it has *zero* effect.

    We check both:
      - compute_loss invariance under different requested temperatures
      - sample_ligand invariance under different requested temperatures

    Determinism: we reset RNG seeds before each call.
    """
    if getattr(model, "use_temperature", True):
        # Only a strict requirement when conditioning is disabled.
        # If enabled, the model is allowed (and expected) to depend on temperature.
        if not bool(getattr(args, "check_temperature_gating", False)):
            return

    model.eval()
    dtype = next(model.parameters()).dtype

    # Always run on a synthetic batch to keep this check lightweight and deterministic.
    bs = max(1, int(args.batch_size))
    sizes = [int(args.synthetic_nodes)] * bs
    items = _make_synthetic_items(batch_size=bs, num_nodes=sizes, atom_nf=model.atom_nf, device=device, dtype=dtype)
    batch = _collate_items(items, device=device, dtype=dtype)

    t_val = float(getattr(args, "temp_gate_t", 0.2))
    t_fixed = torch.full((bs, 1), t_val, device=device, dtype=dtype)

    temp_a = float(getattr(args, "temp_gate_a", 0.5))
    temp_b = float(getattr(args, "temp_gate_b", 2.0))

    # Loss invariance (same noise by resetting RNG).
    torch.manual_seed(123)
    loss_a, info_a = model.compute_loss(batch["ligand"], return_info=True, temperature=temp_a, t=t_fixed)
    torch.manual_seed(123)
    loss_b, info_b = model.compute_loss(batch["ligand"], return_info=True, temperature=temp_b, t=t_fixed)

    if not torch.isfinite(loss_a) or not torch.isfinite(loss_b):
        raise AssertionError("Non-finite loss in temperature gating check")

    if not getattr(model, "use_temperature", True):
        if not torch.allclose(loss_a, loss_b, atol=0.0, rtol=0.0):
            raise AssertionError(
                f"Temperature gating violated: loss differs with condition_temperature=False: {float(loss_a)} vs {float(loss_b)}"
            )
        if info_a.get("temperature") != info_b.get("temperature"):
            raise AssertionError(
                f"Temperature gating violated: logged temperature differs: {info_a.get('temperature')} vs {info_b.get('temperature')}"
            )

    # Sampling invariance (same priors/noise by resetting RNG).
    n_samples = int(getattr(args, "n_samples", 2))
    num_nodes = _parse_num_nodes(str(getattr(args, "num_nodes", "16")), n_samples)
    timesteps = int(getattr(args, "timesteps", 8))

    torch.manual_seed(456)
    s_a = model.sample_ligand(
        n_samples=n_samples,
        num_nodes=num_nodes,
        timesteps=timesteps,
        temperature=temp_a,
        x_sampler="ode",
        h_sampler="markov_bridge",
        stochastic_x=False,
        stochastic_h=False,
    )
    torch.manual_seed(456)
    s_b = model.sample_ligand(
        n_samples=n_samples,
        num_nodes=num_nodes,
        timesteps=timesteps,
        temperature=temp_b,
        x_sampler="ode",
        h_sampler="markov_bridge",
        stochastic_x=False,
        stochastic_h=False,
    )

    if not getattr(model, "use_temperature", True):
        if not torch.equal(s_a["x"], s_b["x"]):
            raise AssertionError("Temperature gating violated: sampled x differs with condition_temperature=False")
        if not torch.equal(s_a["one_hot"], s_b["one_hot"]):
            raise AssertionError("Temperature gating violated: sampled one_hot differs with condition_temperature=False")


def main():
    p = argparse.ArgumentParser(description="Tensor-only workflow tests for EnergyForceDiffusion")
    p.add_argument("--checkpoint", type=str, default=None, help="Optional Lightning checkpoint for EnergyForceDiffusion")
    p.add_argument("--datadir", type=str, default=None, help="Optional folder of OMol25-style *.pkl files")

    p.add_argument("--device", type=str, default="cpu", help="cpu | cuda | cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])

    # Training smoke
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--file-limit", type=int, default=None, help="Limit number of PKL files scanned/used")
    p.add_argument("--synthetic-nodes", type=int, default=16)

    # Training decrease test (real data)
    p.add_argument("--train-iters", type=int, default=0, help="If >0 and --datadir provided, run optimizer for N steps")
    p.add_argument("--trend-window", type=int, default=5, help="Window size for first/last mean loss trend")
    p.add_argument("--require-decrease", action="store_true", help="Fail if loss does not decrease by min fraction")
    p.add_argument("--min-decrease-frac", type=float, default=0.01, help="Required fractional decrease (e.g. 0.01 = 1%%)")
    p.add_argument("--require-decrease-x", action="store_true", help="Fail if loss_x does not decrease by min fraction")
    p.add_argument("--min-decrease-frac-x", type=float, default=0.0, help="Required fractional decrease for loss_x")
    p.add_argument("--require-decrease-h", action="store_true", help="Fail if loss_h does not decrease by min fraction")
    p.add_argument("--min-decrease-frac-h", type=float, default=0.0, help="Required fractional decrease for loss_h")
    p.add_argument("--overfit-single-batch", action="store_true", help="Reuse the first real batch every step")
    p.add_argument("--shuffle-train", action="store_true", help="Shuffle PKL dataset during loss-decrease test")
    p.add_argument("--print-every", type=int, default=0, help="Print loss every N steps (0 disables)")

    # Model params
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--backbone", type=str, default="radius_mlp", choices=["radius_mlp", "gvp"])
    p.add_argument(
        "--no-condition-time",
        action="store_true",
        help="Disable time conditioning (default: enabled)",
    )
    p.add_argument(
        "--condition-temperature",
        action="store_true",
        help="Condition backbone + diffusion schedule on temperature (strictly off unless set)",
    )

    # Diffusion params
    p.add_argument("--timesteps", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--sigma-min", type=float, default=0.01)
    p.add_argument("--sigma-max", type=float, default=50.0)
    p.add_argument("--sigma-min-h", type=float, default=0.01)
    p.add_argument("--sigma-max-h", type=float, default=5.0)
    p.add_argument("--x-score-weight-by-sigma", action="store_true")

    # Schedule selection (DrugFlow convention: t=0 noisy -> t=1 clean)
    p.add_argument("--coord-sde-kind", type=str, default="vp", choices=["ve", "vp"])
    p.add_argument("--logit-sde-kind", type=str, default=None, choices=["ve", "vp"], help="Defaults to coord-sde-kind")
    p.add_argument("--vp-beta-min", type=float, default=0.1)
    p.add_argument("--vp-beta-max", type=float, default=20.0)
    p.add_argument("--vp-sigma-scale", type=float, default=1.0)

    # Extra checks
    p.add_argument("--check-schedule", action="store_true", help="Run schedule monotonicity/endpoints checks")
    p.add_argument("--schedule-n", type=int, default=200, help="Number of t points for schedule checks")
    p.add_argument("--check-graph", action="store_true", help="Run graph batching correctness checks")
    p.add_argument("--graph-kind", type=str, default="both", choices=["fc", "biknn", "both"])
    p.add_argument("--graph-cutoff", type=float, default=1e9, help="Cutoff for biKNN graph check")
    p.add_argument("--graph-k", type=int, default=999, help="k for biKNN graph check")
    p.add_argument(
        "--check-temperature-gating",
        action="store_true",
        help="Run strict invariance checks when temperature conditioning is disabled",
    )
    p.add_argument("--temp-gate-t", type=float, default=0.2, help="Fixed t for temperature gating loss check")
    p.add_argument("--temp-gate-a", type=float, default=0.5)
    p.add_argument("--temp-gate-b", type=float, default=2.0)

    # Freeze stochastic conditioning during training decrease test
    p.add_argument("--freeze-t", action="store_true", help="Reuse a fixed per-sample t across optimizer steps")
    p.add_argument(
        "--t-value",
        type=float,
        default=None,
        help="If set and --freeze-t enabled, use this fixed t in [0,1] (else sample once)",
    )
    p.add_argument("--freeze-temperature", action="store_true", help="Use --temperature as fixed conditioning during training steps")

    # Sampling
    p.add_argument("--n-samples", type=int, default=2)
    p.add_argument("--num-nodes", type=str, default="16", help="int or comma list length n_samples")

    # Dataset atom vocab
    p.add_argument("--allowed-z", type=str, default=None, help="Comma-separated atomic numbers, e.g. 1,6,7,8,9,15,16,17")

    args = p.parse_args()

    if args.logit_sde_kind is None:
        args.logit_sde_kind = args.coord_sde_kind

    # If the user requests float64, set default dtype early so utilities that
    # create tensors without explicit dtype (e.g. Markov bridge priors) match.
    if args.dtype == "float64":
        torch.set_default_dtype(torch.float64)

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    device = torch.device(args.device)

    if args.checkpoint is not None:
        model = EnergyForceDiffusion.load_from_checkpoint(args.checkpoint, map_location=device, strict=False)
        model.to(device)
    else:
        model = _build_fresh_model(args, device)

    if args.dtype == "float64":
        model = model.double()

    if bool(getattr(args, "check_schedule", False)):
        print("=== Schedule sanity ===")
        _run_schedule_sanity(args, device)
        print("OK: schedule monotonicity/endpoints")

    if bool(getattr(args, "check_graph", False)):
        print("=== Graph batching ===")
        run_graph_builder_checks(args, device)
        print("OK: graph batching")

    print("=== Temperature gating ===")
    run_temperature_gating_invariance(model, args, device)
    print("OK: temperature gating")

    print("=== Training smoke ===")
    run_training_smoke(model, args, device)
    print("OK: training_step + validation_step")

    if args.train_iters and int(args.train_iters) > 0:
        print("\n=== Training loss decrease (real data) ===")
        run_training_loss_decrease(model, args, device)
        print("OK: loss decrease test")

    print("\n=== Sampling matrix ===")
    run_sampling_matrix(model, args, device)

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
