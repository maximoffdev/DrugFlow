import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def _split_by_batch_mask(data: torch.Tensor, batch_mask: torch.Tensor) -> list[torch.Tensor]:
    # Preserve order within each sample.
    uniques = torch.unique(batch_mask, sorted=True)
    return [data[batch_mask == i] for i in uniques]


def _parse_sizes(n_samples: int, num_nodes: int | None, sizes_csv: str | None) -> int | torch.Tensor:
    if sizes_csv is not None:
        sizes = [int(x.strip()) for x in sizes_csv.split(",") if x.strip()]
        if not sizes:
            raise ValueError("--sizes was provided but empty")
        if len(sizes) == 1:
            return int(sizes[0])
        if len(sizes) != n_samples:
            raise ValueError(f"--sizes must have length 1 or n_samples={n_samples}; got {len(sizes)}")
        return torch.tensor(sizes, dtype=torch.long)

    if num_nodes is None:
        raise ValueError("Must provide either --num-nodes or --sizes")
    if num_nodes <= 0:
        raise ValueError("--num-nodes must be > 0")
    return int(num_nodes)


def _write_xyz(out_path: Path, atomic_numbers: np.ndarray, coords: np.ndarray, comment: str = "") -> None:
    if atomic_numbers.shape[0] != coords.shape[0]:
        raise ValueError("atomic_numbers and coords must have same length")

    # XYZ format: N, comment, then "Element x y z".
    try:
        from rdkit import Chem

        periodic_table = Chem.GetPeriodicTable()
        symbols = [periodic_table.GetElementSymbol(int(z)) for z in atomic_numbers]
    except Exception:
        # Fallback: write atomic numbers as symbols.
        symbols = [str(int(z)) for z in atomic_numbers]

    lines: list[str] = [str(len(symbols)), str(comment or "")]
    for sym, (x, y, z) in zip(symbols, coords):
        lines.append(f"{sym} {x:.6f} {y:.6f} {z:.6f}")
    out_path.write_text("\n".join(lines) + "\n")


def _coords_to_2d(coords3d: np.ndarray) -> np.ndarray:
    # Project 3D coords to 2D via PCA (SVD).
    if coords3d.shape[0] == 0:
        return coords3d[:, :2]
    centered = coords3d - coords3d.mean(axis=0, keepdims=True)
    if coords3d.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float64)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T  # (3,2)
    coords2d = centered @ basis
    return coords2d.astype(np.float64)


def _write_png_with_rdkit(out_path: Path, atomic_numbers: np.ndarray, coords3d: np.ndarray, size_px: int = 400) -> bool:
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Geometry import Point3D
        from rdkit.Chem.Draw import rdMolDraw2D

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        return False

    mol = Chem.RWMol()
    for z in atomic_numbers:
        mol.AddAtom(Chem.Atom(int(z)))
    mol = mol.GetMol()

    coords2d = _coords_to_2d(coords3d)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y) in enumerate(coords2d):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)

    drawer = rdMolDraw2D.MolDraw2DCairo(int(size_px), int(size_px))
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    out_path.write_bytes(drawer.GetDrawingText())
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Sample molecules from an EnergyForceDiffusion checkpoint and export XYZ/PNG.")
    p.add_argument("--ckpt", type=Path, required=True, help="Path to a .ckpt file")
    p.add_argument("--outdir", type=Path, required=True, help="Output directory")
    p.add_argument("--n-samples", type=int, default=8, help="Number of molecules to sample")
    p.add_argument("--num-nodes", type=int, default=None, help="Fixed number of atoms per molecule")
    p.add_argument("--sizes", type=str, default=None, help="Comma-separated sizes; length 1 or n-samples")
    p.add_argument("--timesteps", type=int, default=None, help="Euler steps from t=0 -> t=1 (default: from checkpoint)")
    p.add_argument("--temperature", type=float, default=None, help="Temperature conditioning value (default: from checkpoint)")
    p.add_argument("--x-sampler", type=str, default="ode", choices=["ode", "sde"], help="Coordinate sampler")
    p.add_argument(
        "--h-sampler",
        type=str,
        default="markov_bridge",
        choices=["markov_bridge", "score"],
        help="Atom-type sampler",
    )
    p.add_argument("--stochastic-x", action="store_true", help="Enable stochastic reverse steps for x when using SDE")
    p.add_argument("--stochastic-h", action="store_true", help="Enable stochastic reverse steps for h when using score sampler")
    p.add_argument(
        "--atom-type-prior",
        type=str,
        default="uniform",
        choices=["uniform"],
        help="Atom-type prior used when the checkpoint has diffuse_h: false",
    )
    p.add_argument("--device", type=str, default="cuda", help="Device (e.g. cuda, cuda:0, cpu)")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--no-png", action="store_true", help="Skip PNG rendering")
    p.add_argument("--png-size", type=int, default=400, help="PNG size in pixels")
    p.add_argument("--print-atom-counts", action="store_true", help="Print atomic number counts per sample")
    args = p.parse_args()

    basedir = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(basedir))

    from src.model.energy_force_diffusion import EnergyForceDiffusion

    args.outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    model = EnergyForceDiffusion.load_from_checkpoint(str(args.ckpt), map_location=device)
    model = model.to(device)
    model.eval()

    num_nodes = _parse_sizes(n_samples=int(args.n_samples), num_nodes=args.num_nodes, sizes_csv=args.sizes)
    if isinstance(num_nodes, torch.Tensor):
        num_nodes = num_nodes.to(device)

    with torch.no_grad():
        batch = model.sample_ligand(
            n_samples=int(args.n_samples),
            num_nodes=num_nodes,
            timesteps=args.timesteps,
            temperature=args.temperature,
            x_sampler=str(args.x_sampler),
            h_sampler=str(args.h_sampler),
            atom_type_prior=str(args.atom_type_prior),
            stochastic_x=bool(args.stochastic_x),
            stochastic_h=bool(args.stochastic_h),
        )

    coords_list = _split_by_batch_mask(batch["x"].detach().cpu(), batch["mask"].detach().cpu())
    one_hot_list = _split_by_batch_mask(batch["one_hot"].detach().cpu(), batch["mask"].detach().cpu())
    allowed_z = torch.tensor(model.allowed_atomic_numbers, dtype=torch.long)

    wrote_png_any = False
    for i, (coords_i, one_hot_i) in enumerate(zip(coords_list, one_hot_list)):
        idx = torch.argmax(one_hot_i, dim=-1).to(torch.long)
        atomic_numbers = allowed_z[idx].numpy()
        coords = coords_i.numpy()

        if args.print_atom_counts:
            uniq, cnt = np.unique(atomic_numbers, return_counts=True)
            counts_str = ", ".join([f"{int(z)}:{int(c)}" for z, c in zip(uniq, cnt)])
            print(f"sample_{i:04d} atom_counts: {counts_str}")

        stem = f"sample_{i:04d}_n{coords.shape[0]}"
        xyz_path = args.outdir / f"{stem}.xyz"
        _write_xyz(xyz_path, atomic_numbers=atomic_numbers, coords=coords, comment=stem)

        if not args.no_png:
            png_path = args.outdir / f"{stem}.png"
            wrote = _write_png_with_rdkit(
                png_path,
                atomic_numbers=atomic_numbers,
                coords3d=coords,
                size_px=int(args.png_size),
            )
            wrote_png_any = wrote_png_any or wrote

    if (not args.no_png) and (not wrote_png_any):
        print("RDKit not available (or failed to draw); wrote XYZ only.")

    print(f"Wrote {len(coords_list)} XYZ files to: {args.outdir}")
    if not args.no_png:
        print(f"PNGs: {'enabled' if wrote_png_any else 'skipped'}")


if __name__ == "__main__":
    main()
