#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path
from typing import Any


def _iter_pkl_files(input_dir: Path):
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix == ".pkl":
            yield p


def _dump_bundle(out_path: Path, items: list[Any], names: list[str]) -> None:
    payload = {
        "__bundle__": "drugflow_pkl_v1",
        "items": items,
        "names": names,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Bundle many single-molecule .pkl files into fewer multi-molecule .pkl files. "
            "This reduces filesystem I/O overhead on HPC."
        )
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing single-molecule .pkl files.",
    )
    parser.add_argument(
        "batch_size",
        type=int,
        help="How many molecules to store per bundled .pkl.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for bundled .pkl files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally only bundle the first N input pkls (for debugging).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    batch_size = int(args.batch_size)

    if batch_size <= 0:
        raise SystemExit(f"batch_size must be > 0, got {batch_size}")
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"input_dir does not exist or is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    items: list[Any] = []
    names: list[str] = []
    bundle_idx = 0

    n_in = 0
    for fp in _iter_pkl_files(input_dir):
        if args.limit is not None and n_in >= int(args.limit):
            break

        with open(fp, "rb") as f:
            obj = pickle.load(f)

        items.append(obj)
        names.append(fp.name)
        n_in += 1

        if len(items) >= batch_size:
            out_fp = output_dir / f"bundle_{bundle_idx:06d}.pkl"
            _dump_bundle(out_fp, items, names)
            bundle_idx += 1
            items = []
            names = []

    if items:
        out_fp = output_dir / f"bundle_{bundle_idx:06d}.pkl"
        _dump_bundle(out_fp, items, names)

    print(f"Bundled {n_in} molecules into {bundle_idx + (1 if items else 0)} files at {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
