#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def _load_any(path: Path) -> Any:
    # Try torch.load first (common for PyG Data), then fallback to pickle.
    try:
        import torch

        return torch.load(path, map_location="cpu")
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)


def _get_n_atoms(obj: Any) -> int:
    # PyG Data
    if hasattr(obj, "pos") and getattr(obj, "pos") is not None:
        pos = getattr(obj, "pos")
        try:
            return int(pos.shape[0])
        except Exception:
            pass

    for key in ("atomic_numbers", "z", "atom_type"):
        if hasattr(obj, key):
            v = getattr(obj, key)
            try:
                return int(len(v))
            except Exception:
                pass
        if isinstance(obj, dict) and key in obj:
            try:
                return int(len(obj[key]))
            except Exception:
                pass

    raise ValueError("Could not infer number of atoms from object")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate DrugFlow-compatible size_distribution.npy from a directory of .pkl files. "
            "Creates a 2D histogram over (n_ligand_nodes, n_pocket_nodes). For ligand-only datasets, "
            "pocket is fixed to size 0 => histogram shape [max_n+1, 1]."
        )
    )
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--glob", type=str, default="*.pkl")
    parser.add_argument("--out", type=str, default="size_distribution.npy")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of files to scan (0 = no limit)")

    args = parser.parse_args()

    datadir = Path(args.datadir)
    if not datadir.exists():
        raise FileNotFoundError(datadir)

    files = sorted(datadir.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No files matched {args.glob} in {datadir}")

    if args.limit and args.limit > 0:
        files = files[: args.limit]

    counts = {}
    max_n = 0

    for p in files:
        obj = _load_any(p)
        n = _get_n_atoms(obj)
        max_n = max(max_n, n)
        counts[n] = counts.get(n, 0) + 1

    # 2D histogram expected by DistributionNodes: [n_lig_sizes, n_pocket_sizes]
    hist = np.zeros((max_n + 1, 1), dtype=np.float64)
    for n, c in counts.items():
        hist[n, 0] = float(c)

    out_path = datadir / args.out
    np.save(out_path, hist)
    print(f"Wrote {out_path} with shape={hist.shape}; scanned {len(files)} files; max_n={max_n}")


if __name__ == "__main__":
    main()
