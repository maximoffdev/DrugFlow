#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def _iter_pkl_files(input_dir: Path) -> Iterable[Path]:
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix == ".pkl":
            yield p


def _as_numpy(x: Any) -> np.ndarray:
    # torch.Tensor
    try:
        import torch  # type: ignore

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass

    # numpy array
    if isinstance(x, np.ndarray):
        return x

    return np.asarray(x)


def _get(obj: Any, k: str):
    if isinstance(obj, dict):
        return obj[k]
    return getattr(obj, k)


def _sanitize_group_name(name: str) -> str:
    # HDF5 allows many characters, but keep it simple and stable.
    # Avoid slashes (path separator in HDF5).
    return name.replace("/", "_")


def _atomic_numbers_key(z: np.ndarray) -> str:
    """Stable key for grouping molecules with identical atomic numbers.

    ANI-style groups require a fixed atom count + fixed atomic numbers array.
    We hash the atomic numbers bytes to keep group names compact.
    """
    z = np.asarray(z, dtype=np.int64)
    h = hashlib.sha1(z.tobytes()).hexdigest()[:16]
    return f"na{int(z.shape[0])}_{h}"


def _read_single_mol_pkl(fp: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(fp, "rb") as f:
        obj = pickle.load(f)

    pos = _as_numpy(_get(obj, "pos")).astype(np.float32, copy=False)
    z = _as_numpy(_get(obj, "atomic_numbers"))
    e = _as_numpy(_get(obj, "e_total"))
    forces = _as_numpy(_get(obj, "forces")).astype(np.float32, copy=False)

    if pos.ndim != 2 or pos.shape[-1] != 3:
        raise ValueError(f"Expected pos shape (N,3), got {pos.shape} in {fp.name}")
    if forces.shape != pos.shape:
        raise ValueError(f"Expected forces shape to match pos, got {forces.shape} vs {pos.shape} in {fp.name}")

    z = np.asarray(z)
    if z.ndim != 1 or z.shape[0] != pos.shape[0]:
        raise ValueError(f"Expected atomic_numbers shape (N,), got {z.shape} in {fp.name}")

    # OMol25 sometimes stores atomic_numbers as float; round safely.
    if np.issubdtype(z.dtype, np.floating):
        z = np.rint(z).astype(np.int64)
    else:
        z = z.astype(np.int64, copy=False)

    e = np.asarray(e, dtype=np.float32)
    if e.ndim == 0:
        e = e.reshape(1)
    elif e.ndim == 1 and e.size == 1:
        e = e.reshape(1)
    else:
        raise ValueError(f"Expected e_total scalar, got {e.shape} in {fp.name}")

    return pos, z, e, forces


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a directory of single-structure PKL files into one ANI-style HDF5 file, "
            "compatible with src/data/ani_h5_dataset.py without modifications."
        )
    )
    parser.add_argument("input_dir", type=str, help="Directory containing single-molecule .pkl files")
    parser.add_argument("output_h5", type=str, help="Output .h5 path")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output_h5 if it exists",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally only convert the first N input pkls (debugging)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed used to shuffle molecule order before packing into groups",
    )
    parser.add_argument(
        "--max_conformations_per_group",
        type=int,
        default=256,
        help=(
            "Maximum number of molecules (Nc) stored in a single HDF5 group. "
            "If exceeded, additional molecules with the same atomic_numbers are written to a new part group."
        ),
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_h5 = Path(args.output_h5)

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"input_dir does not exist or is not a directory: {input_dir}")

    if output_h5.exists():
        if args.overwrite:
            output_h5.unlink()
        else:
            raise SystemExit(f"output_h5 already exists (use --overwrite): {output_h5}")

    try:
        import h5py  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing dependency 'h5py'. Install it (e.g. `conda install -c conda-forge h5py`)."
        ) from e

    output_h5.parent.mkdir(parents=True, exist_ok=True)

    max_nc = int(args.max_conformations_per_group)
    if max_nc <= 0:
        raise SystemExit(f"--max_conformations_per_group must be > 0, got {max_nc}")

    # Shuffle input files for random packing into groups.
    files: List[Path] = list(_iter_pkl_files(input_dir))
    if args.limit is not None:
        files = files[: int(args.limit)]
    rng = np.random.default_rng(int(args.seed))
    if len(files) > 1:
        perm = rng.permutation(len(files)).tolist()
        files = [files[i] for i in perm]

    n_written = 0
    n_groups = 0

    # Track per-key group parts so each group remains a reasonable size.
    # base_key -> (part_idx, current_nc)
    group_state: Dict[str, Tuple[int, int]] = {}
    # base_key -> atomic_numbers array (to validate consistency)
    group_z: Dict[str, np.ndarray] = {}

    # For appending, keep open datasets per currently-active (base_key, part_idx).
    # group_name -> dict of datasets
    active: Dict[str, Dict[str, Any]] = {}

    with h5py.File(output_h5, "w") as h5:
        h5.attrs["format"] = "drugflow_ani_h5_v1"
        h5.attrs["source"] = str(input_dir)

        str_dt = h5py.string_dtype(encoding="utf-8")

        def _ensure_group(base_key: str, z_arr: np.ndarray) -> str:
            nonlocal n_groups

            if base_key not in group_state:
                group_state[base_key] = (0, 0)
                group_z[base_key] = np.asarray(z_arr, dtype=np.int64)

            part_idx, cur_nc = group_state[base_key]
            if cur_nc >= max_nc:
                part_idx += 1
                cur_nc = 0
                group_state[base_key] = (part_idx, cur_nc)

            group_name = _sanitize_group_name(f"{base_key}__part{part_idx:04d}")
            if group_name in active:
                return group_name

            # Create a new group + empty resizable datasets.
            grp = h5.create_group(group_name)
            grp.create_dataset("atomic_numbers", data=np.asarray(z_arr, dtype=np.int64), dtype="i8")

            na = int(z_arr.shape[0])
            coords = grp.create_dataset(
                "coordinates",
                shape=(0, na, 3),
                maxshape=(None, na, 3),
                chunks=(min(64, max_nc), na, 3),
                dtype="f4",
            )
            forces_ds = grp.create_dataset(
                "wb97x_dz.forces",
                shape=(0, na, 3),
                maxshape=(None, na, 3),
                chunks=(min(64, max_nc), na, 3),
                dtype="f4",
            )
            energy_ds = grp.create_dataset(
                "wb97x_dz.energy",
                shape=(0,),
                maxshape=(None,),
                chunks=(min(1024, max_nc),),
                dtype="f4",
            )
            names_ds = grp.create_dataset(
                "source_pkl",
                shape=(0,),
                maxshape=(None,),
                chunks=(min(1024, max_nc),),
                dtype=str_dt,
            )

            active[group_name] = {
                "grp": grp,
                "coordinates": coords,
                "forces": forces_ds,
                "energy": energy_ds,
                "names": names_ds,
            }
            n_groups += 1
            return group_name

        for fp in files:
            pos, z, e, forces = _read_single_mol_pkl(fp)
            base_key = _atomic_numbers_key(z)

            # Validate the atomic_numbers are consistent within each base_key
            if base_key in group_z:
                if not np.array_equal(group_z[base_key], np.asarray(z, dtype=np.int64)):
                    # Extremely unlikely due to hashing, but protect against collisions.
                    base_key = f"{base_key}_{hashlib.sha1(fp.name.encode('utf-8')).hexdigest()[:8]}"
                    group_z[base_key] = np.asarray(z, dtype=np.int64)

            group_name = _ensure_group(base_key, z)
            ds = active[group_name]

            # Append as a new conformation (Nc += 1)
            cur = int(ds["energy"].shape[0])
            ds["coordinates"].resize((cur + 1, ds["coordinates"].shape[1], 3))
            ds["forces"].resize((cur + 1, ds["forces"].shape[1], 3))
            ds["energy"].resize((cur + 1,))
            ds["names"].resize((cur + 1,))

            ds["coordinates"][cur, :, :] = pos
            ds["forces"][cur, :, :] = forces
            ds["energy"][cur] = float(np.asarray(e, dtype=np.float32).reshape(-1)[0])
            ds["names"][cur] = fp.name

            # Update group_state counts
            part_idx, cur_nc = group_state[base_key]
            group_state[base_key] = (part_idx, cur_nc + 1)

            n_written += 1
            if n_written % 1000 == 0:
                print(f"Converted {n_written} molecules into {n_groups} groups...")

    print(f"Wrote {n_written} molecules to {output_h5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
