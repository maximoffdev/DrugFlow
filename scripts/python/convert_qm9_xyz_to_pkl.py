#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import Any, Iterable
import re

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from torch_geometric.data import Data
except Exception:  # pragma: no cover
    Data = None


# Minimal periodic table mapping for QM9 (+ common halogens)
_ATOMIC_NUMBERS = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}


@dataclass(frozen=True)
class ParsedXYZ:
    name: str
    pos: np.ndarray  # (N,3) float32
    atomic_numbers: np.ndarray  # (N,) float32 (OMol25-style)
    e_total: float  # float32-compatible


def _parse_float(token: str) -> float:
    """Parse float tokens that may include nonstandard scientific notation.

    QM9-curated xyz files sometimes contain values like '2.1997E^-6', which is
    not accepted by Python's float(). We normalize it to '2.1997e-6'.
    """
    t = token.strip()
    if "E^" in t or "e^" in t:
        base, exp = t.replace("e^", "E^").split("E^", 1)
        return float(f"{base}e{exp}")
    return float(t)


def _parse_xyz(path: Path) -> ParsedXYZ:
    # IMPORTANT: QM9 xyz files include many extra lines (frequencies, SMILES, InChI, ...).
    # For speed we only read: natoms, metadata, and the natoms coordinate lines.
    with path.open("r", encoding="utf-8", errors="replace") as f:
        line0 = f.readline()
        if not line0:
            raise ValueError(f"{path}: empty file")

        try:
            natoms = int(line0.strip())
        except Exception as e:
            raise ValueError(f"{path}: invalid first line natoms: {line0!r}") from e

        line1 = f.readline()
        if not line1:
            raise ValueError(f"{path}: missing metadata line")

        meta = line1.strip().split()
        if len(meta) < 2:
            raise ValueError(f"{path}: invalid metadata line: {line1!r}")

        name = meta[0]
        floats = [_parse_float(x) for x in meta[1:]]
        if len(floats) < 14:
            raise ValueError(
                f"{path}: metadata has only {len(floats)} numeric fields; need >=14 to read energy as 14th value"
            )

        # User spec: energy is the 14th numeric value in the 2nd line (after the name token)
        e_total = float(np.float32(floats[13]))

        coords = np.empty((natoms, 3), dtype=np.float32)
        z = np.empty((natoms,), dtype=np.float32)

        for i in range(natoms):
            atom_line = f.readline()
            if not atom_line:
                raise ValueError(f"{path}: unexpected EOF while reading atom lines (expected {natoms})")
            parts = atom_line.split()
            if len(parts) < 4:
                raise ValueError(f"{path}: invalid atom line {i}: {atom_line!r}")

            sym = parts[0]
            if sym not in _ATOMIC_NUMBERS:
                raise ValueError(f"{path}: unknown element symbol {sym!r} (extend _ATOMIC_NUMBERS if needed)")

            z[i] = float(_ATOMIC_NUMBERS[sym])
            coords[i, 0] = np.float32(_parse_float(parts[1]))
            coords[i, 1] = np.float32(_parse_float(parts[2]))
            coords[i, 2] = np.float32(_parse_float(parts[3]))

    return ParsedXYZ(name=name, pos=coords, atomic_numbers=z, e_total=e_total)


def _iter_xyz_files(input_dir: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(input_dir.glob(pattern))


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_filename(stem: str) -> str:
    # Keep filenames portable and avoid accidental subpaths.
    stem = stem.replace(os.sep, "_")
    stem = _SAFE_NAME_RE.sub("_", stem).strip("._-")
    return stem or "sample"


def _convert_one(
    xyz_path_str: str,
    output_dir_str: str,
    overwrite: bool,
    output_format: str,
    output_name: str,
) -> tuple[str, str]:
    """Worker-safe single-file conversion.

    Returns: (status, path)
      - status: 'converted' | 'skipped'
      - path: output file path
    """
    xyz_path = Path(xyz_path_str)
    output_dir = Path(output_dir_str)
    parsed = _parse_xyz(xyz_path)

    if output_name == "filename":
        out_name = xyz_path.stem
    elif output_name == "metadata":
        out_name = parsed.name if parsed.name else xyz_path.stem
    elif output_name == "metadata+filename":
        # Put the unique filename first so lexicographic order matches the raw dataset.
        out_name = f"{xyz_path.stem}__{parsed.name}" if parsed.name else xyz_path.stem
    else:
        raise ValueError(f"Unknown output_name={output_name!r}")

    out_name = _sanitize_filename(out_name)
    out_path = output_dir / f"{out_name}.pkl"

    if out_path.exists():
        if not overwrite:
            return "skipped", str(out_path)
        status = "overwritten"
    else:
        status = "converted"

    obj = _to_omol25_like(parsed, output_format=output_format)
    with out_path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return status, str(out_path)


def _to_omol25_like(p: ParsedXYZ, *, output_format: str) -> object:
    """Return an OMol25-compatible sample.

    Format options:
      - 'dict': fastest; stores numpy arrays + scalars under OMol25 key names.
      - 'pyg' : torch_geometric.data.Data object (closer to OMol25 PKLs).
    """
    if output_format not in {"dict", "pyg"}:
        raise ValueError(f"Unknown output_format={output_format!r}")

    forces = np.zeros_like(p.pos, dtype=np.float32)

    if output_format == "dict" or Data is None:
        return {
            "pos": p.pos,
            "atomic_numbers": p.atomic_numbers,
            "e_total": np.float32(p.e_total),
            "forces": forces,
            "natoms": int(p.pos.shape[0]),
        }

    # PyG format
    pos = torch.from_numpy(p.pos)
    z = torch.from_numpy(p.atomic_numbers)
    forces_t = torch.from_numpy(forces)
    e_total = torch.tensor(p.e_total, dtype=torch.float32)
    return Data(
        pos=pos,
        atomic_numbers=z,
        e_total=e_total,
        forces=forces_t,
        natoms=int(pos.shape[0]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Convert QM9-curated .xyz files into OMol25-style .pkl files (PyG Data). "
            "Forces are not present in QM9-curated xyz; they are written as zeros with shape (N,3). "
            "Energy is read as the 14th numeric value on the 2nd line (after the name token)."
        )
    )
    ap.add_argument(
        "--input_dir",
        type=str,
        default="/media/data/qm_dataset/data/Pipeline/xyz_raw/qm9_curated",
        help="Directory containing .xyz files",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="/media/data/qm_dataset/data/Pipeline/pkl/qm9_curated",
        help="Directory to write .pkl files",
    )
    ap.add_argument("--pattern", type=str, default="*.xyz", help="Glob pattern for xyz files")
    ap.add_argument("--limit", type=int, default=0, help="Optional max number of files to convert (0 = all)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .pkl files")
    ap.add_argument(
        "--num_workers",
        type=int,
        default=-1,
        help="Parallel workers (-1 = auto, 0 = single-process).",
    )
    ap.add_argument(
        "--output_format",
        type=str,
        default="dict",
        choices=["dict", "pyg"],
        help="Output PKL payload type. 'dict' is much faster; 'pyg' writes torch_geometric Data.",
    )
    ap.add_argument(
        "--output_name",
        type=str,
        default="filename",
        choices=["filename", "metadata", "metadata+filename"],
        help=(
            "How to name output .pkl files. 'filename' uses the input .xyz basename (unique for QM9). "
            "Using 'metadata' may cause many files to overwrite each other if the metadata name is not unique."
        ),
    )
    ap.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Skip files that fail to parse instead of aborting.",
    )

    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xyz_files = list(_iter_xyz_files(input_dir, args.pattern))
    if not xyz_files:
        raise FileNotFoundError(f"No .xyz files found in {input_dir} matching {args.pattern!r}")

    if args.limit and args.limit > 0:
        xyz_files = xyz_files[: args.limit]

    n_ok = 0
    n_skip = 0
    n_err = 0
    n_over = 0

    num_workers = int(args.num_workers)
    if num_workers < 0:
        # Auto: small cap to avoid oversubscribing on shared machines.
        num_workers = min(8, os.cpu_count() or 1)

    if num_workers == 0:
        iterator: Any = xyz_files
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(xyz_files), desc="Converting", unit="file")

        for xyz_path in iterator:
            try:
                status, _ = _convert_one(
                    str(xyz_path),
                    str(output_dir),
                    bool(args.overwrite),
                    str(args.output_format),
                    str(args.output_name),
                )
                if status == "converted":
                    n_ok += 1
                elif status == "overwritten":
                    n_over += 1
                else:
                    n_skip += 1
            except Exception:
                if not args.continue_on_error:
                    raise
                n_err += 1

            if tqdm is not None and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(ok=n_ok, overwritten=n_over, skipped=n_skip, err=n_err)
        print(f"Converted: {n_ok}, overwritten: {n_over}, skipped: {n_skip}, errors: {n_err}, output_dir: {output_dir}")
        return

    if num_workers > 0 and num_workers > (os.cpu_count() or num_workers):
        num_workers = os.cpu_count() or num_workers

    # Map with chunksize to reduce scheduling overhead for many tiny files.
    chunksize = 256
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        it: Any = ex.map(
            _convert_one,
            (str(x) for x in xyz_files),
            repeat(str(output_dir)),
            repeat(bool(args.overwrite)),
            repeat(str(args.output_format)),
            repeat(str(args.output_name)),
            chunksize=chunksize,
        )

        if tqdm is not None:
            it = tqdm(it, total=len(xyz_files), desc=f"Converting ({num_workers}w)", unit="file")

        # NOTE: to support --continue_on_error with ProcessPoolExecutor, we must
        # catch exceptions thrown by next(it), not by unpacking a result.
        it_iter = iter(it)
        for _ in range(len(xyz_files)):
            try:
                status, _ = next(it_iter)
            except StopIteration:
                break
            except Exception:
                if not args.continue_on_error:
                    raise
                n_err += 1
                if tqdm is not None and hasattr(it, "set_postfix"):
                    it.set_postfix(ok=n_ok, overwritten=n_over, skipped=n_skip, err=n_err)
                continue

            if status == "converted":
                n_ok += 1
            elif status == "overwritten":
                n_over += 1
            else:
                n_skip += 1

            if tqdm is not None and hasattr(it, "set_postfix"):
                it.set_postfix(ok=n_ok, overwritten=n_over, skipped=n_skip, err=n_err)

    print(
        f"Converted: {n_ok}, overwritten: {n_over}, skipped: {n_skip}, errors: {n_err}, workers: {num_workers}, chunksize: {chunksize}, output_dir: {output_dir}"
    )


if __name__ == "__main__":
    main()
