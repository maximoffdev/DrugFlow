import os
import pickle
import bisect
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from src.data.data_utils import TensorDict, collate_entity


@dataclass(frozen=True)
class PKLKeys:
    pos: str = "pos"
    atom_types: str = "atomic_numbers"  # OMol25 PKLs store this; can also be one-hot/labels
    energy: str = "e_total"
    forces: str = "forces"


class PKLEnergyForceDataset(Dataset):
    """Loads per-molecule .pkl files with (positions, atom types) inputs and (energy, forces) labels.

    Expected per-sample schema (dict-like):
      - keys.pos: (N, 3) float
      - keys.atom_types: either (N, K) one-hot float/int OR (N,) int class labels
      - keys.energy: scalar float
      - keys.forces: (N, 3) float

    Output per-sample:
      {
        'ligand': {'x': (N,3), 'one_hot': (N,K), 'size': int, 'name': str},
        'pocket': {'x': (0,3), 'one_hot': (0,0), 'size': 0, 'name': str},
        'energy': (1,),
        'force': (N,3),
      }

    Notes:
      - We include an empty 'pocket' dict to keep compatibility with the existing
        training_step signature used by DrugFlow.
      - Pocket conditioning can be introduced later without changing the top-level
        batch structure.
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        file_paths: Optional[Sequence[str | Path]] = None,
        keys: PKLKeys = PKLKeys(),
        atom_types_format: str = "atomic_numbers",  # 'atomic_numbers' | 'one_hot' | 'labels'
        num_atom_types: Optional[int] = None,
        allowed_atomic_numbers: Optional[Sequence[int]] = None,
        scan_limit: int = 2048,
        device: str | torch.device = "cpu",
        file_limit: Optional[int] = None,
        bundle_cache_size: int = 2,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.keys = keys
        self.atom_types_format = atom_types_format
        self.num_atom_types = num_atom_types
        self.allowed_atomic_numbers = list(allowed_atomic_numbers) if allowed_atomic_numbers is not None else None
        self.scan_limit = int(scan_limit)
        self.device = device
        self.bundle_cache_size = int(bundle_cache_size)

        if file_paths is not None:
            fps = [Path(p) for p in file_paths]
        else:
            fps = [self.data_dir / f for f in os.listdir(self.data_dir) if f.endswith(".pkl")]
            fps = sorted(fps)
        if file_limit is not None:
            fps = fps[:file_limit]
        self.file_paths: List[Path] = list(fps)

        # Build a molecule-level index so we can handle pkls that contain multiple structures.
        # This scans each file once at initialization to determine how many molecules it contains.
        self._cum_counts: List[int] = [0]
        inferred_atomic_numbers: Optional[set[int]] = set() if (self.atom_types_format == "atomic_numbers" and self.allowed_atomic_numbers is None) else None
        scanned_mols = 0

        for fp in self.file_paths:
            items, _names = self._load_as_items(fp)
            n_items = len(items)
            if n_items <= 0:
                raise ValueError(f"Bundle {fp.name} contains zero items")

            # Optional atomic number inference (scan first `scan_limit` molecules across the dataset)
            if inferred_atomic_numbers is not None and scanned_mols < self.scan_limit:
                for obj in items:
                    if scanned_mols >= self.scan_limit:
                        break
                    try:
                        z = self._get(obj, self.keys.atom_types)
                    except Exception:
                        continue
                    zt = torch.as_tensor(z)
                    if zt.dtype.is_floating_point:
                        zt = zt.round().to(torch.long)
                    else:
                        zt = zt.to(torch.long)
                    inferred_atomic_numbers.update(int(x) for x in torch.unique(zt).tolist())
                    scanned_mols += 1

            self._cum_counts.append(self._cum_counts[-1] + n_items)

        # Cache of loaded bundles to avoid repeated unpickling when many samples come from the same file.
        # This is per-process (DataLoader worker) and gets cleared on pickling.
        self._bundle_cache: "OrderedDict[str, Tuple[list[Any], list[str]]]" = OrderedDict()

        self.atomic_number_to_index: Optional[dict[int, int]] = None
        if self.atom_types_format == "atomic_numbers":
            if self.allowed_atomic_numbers is None:
                if inferred_atomic_numbers is None or len(inferred_atomic_numbers) == 0:
                    raise ValueError("Could not infer any atomic numbers during scan")
                self.allowed_atomic_numbers = sorted(inferred_atomic_numbers)
            self.allowed_atomic_numbers = sorted(set(int(z) for z in self.allowed_atomic_numbers))
            self.atomic_number_to_index = {int(z): i for i, z in enumerate(self.allowed_atomic_numbers)}
            # Override num_atom_types if not provided
            if self.num_atom_types is None:
                self.num_atom_types = len(self.allowed_atomic_numbers)
            if self.num_atom_types != len(self.allowed_atomic_numbers):
                raise ValueError(
                    f"num_atom_types ({self.num_atom_types}) must match allowed_atomic_numbers length ({len(self.allowed_atomic_numbers)})"
                )

    def __len__(self) -> int:
        return int(self._cum_counts[-1])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        file_i = bisect.bisect_right(self._cum_counts, idx) - 1
        inner_i = int(idx - self._cum_counts[file_i])
        fp = self.file_paths[file_i]

        items, names = self._get_bundle_items_cached(fp)
        data = items[inner_i]
        name = names[inner_i] if inner_i < len(names) else f"{fp.name}:{inner_i}"

        pos = self._get(data, self.keys.pos)
        atom_types = self._get(data, self.keys.atom_types)
        energy = self._get(data, self.keys.energy)
        forces = self._get(data, self.keys.forces)

        # Optional sanity check if natoms is present
        natoms = None
        try:
            natoms = self._get(data, "natoms")
        except Exception:
            natoms = None

        pos = torch.as_tensor(pos, dtype=torch.float32, device=self.device)
        forces = torch.as_tensor(forces, dtype=torch.float32, device=self.device)
        energy = torch.as_tensor(energy, dtype=torch.float32, device=self.device)
        # OMol25 stores energy as a scalar tensor; normalize to shape (1,)
        if energy.ndim == 0:
            energy = energy.view(1)
        elif energy.ndim == 1 and energy.numel() == 1:
            energy = energy.view(1)
        else:
            raise ValueError(f"Expected energy to be scalar, got shape {tuple(energy.shape)} in {fp.name}")

        if self.atom_types_format == "atomic_numbers":
            if self.atomic_number_to_index is None or self.num_atom_types is None:
                raise RuntimeError("atomic_numbers_to_index/num_atom_types not initialized")
            z = torch.as_tensor(atom_types, device=self.device)
            # OMol25 stores atomic_numbers as float tensors; convert safely
            if z.dtype.is_floating_point:
                z = z.round().to(torch.long)
            else:
                z = z.to(torch.long)

            # Map to contiguous indices
            # Use vectorized mapping via a lookup table when possible
            max_z = int(max(self.atomic_number_to_index.keys()))
            lut = torch.full((max_z + 1,), -1, device=z.device, dtype=torch.long)
            for zz, ii in self.atomic_number_to_index.items():
                lut[int(zz)] = int(ii)
            if (z < 0).any() or (z > max_z).any() or (lut[z] < 0).any():
                unknown = torch.unique(z[(z < 0) | (z > max_z) | (lut[z] < 0)]).tolist()
                raise ValueError(f"Found unknown atomic numbers {unknown} in {fp.name}. Allowed: {self.allowed_atomic_numbers}")
            labels = lut[z]
            one_hot = torch.nn.functional.one_hot(labels, num_classes=self.num_atom_types).to(torch.float32)

        elif self.atom_types_format == "one_hot":
            one_hot = torch.as_tensor(atom_types, device=self.device)
            if one_hot.dim() == 1:
                if self.num_atom_types is None:
                    raise ValueError(
                        "atom_types_format='one_hot' but atom_types is 1D; provide num_atom_types or set atom_types_format='labels'"
                    )
                one_hot = torch.nn.functional.one_hot(one_hot.to(torch.long), num_classes=self.num_atom_types).to(torch.float32)
            else:
                one_hot = one_hot.to(torch.float32)
        elif self.atom_types_format == "labels":
            labels = torch.as_tensor(atom_types, device=self.device).to(torch.long)
            if self.num_atom_types is None:
                raise ValueError("atom_types_format='labels' requires num_atom_types")
            one_hot = torch.nn.functional.one_hot(labels, num_classes=self.num_atom_types).to(torch.float32)
        else:
            raise ValueError(f"Unknown atom_types_format: {self.atom_types_format}")

        if pos.ndim != 2 or pos.size(-1) != 3:
            raise ValueError(f"Expected pos to have shape (N,3), got {tuple(pos.shape)} in {fp.name}")
        if natoms is not None:
            try:
                natoms_int = int(torch.as_tensor(natoms).item())
                if natoms_int != int(pos.size(0)):
                    raise ValueError(f"natoms ({natoms_int}) != pos.shape[0] ({int(pos.size(0))}) in {fp.name}")
            except Exception:
                # ignore natoms if it isn't a scalar
                pass
        if forces.shape != pos.shape:
            raise ValueError(
                f"Expected forces to match pos shape (N,3), got forces={tuple(forces.shape)} pos={tuple(pos.shape)} in {fp.name}"
            )
        if one_hot.ndim != 2 or one_hot.size(0) != pos.size(0):
            raise ValueError(
                f"Expected one_hot shape (N,K) aligned with pos; got one_hot={tuple(one_hot.shape)} pos={tuple(pos.shape)} in {fp.name}"
            )

        ligand = {
            "name": name,
            "x": pos,
            "one_hot": one_hot,
            "size": pos.size(0),
        }

        # Keep a stable schema with an empty pocket (ligand-only training for now)
        pocket = {
            "name": name,
            "x": torch.zeros((0, 3), dtype=pos.dtype, device=pos.device),
            "one_hot": torch.zeros((0, 0), dtype=one_hot.dtype, device=one_hot.device),
            "size": 0,
        }

        return {
            "ligand": ligand,
            "pocket": pocket,
            "energy": energy,
            "force": forces,
        }

    def __getstate__(self):
        state = dict(self.__dict__)
        # Avoid pickling large cached bundles when DataLoader forks workers.
        state["_bundle_cache"] = OrderedDict()
        return state

    @staticmethod
    def _get(obj: Any, k: str):
        """Tolerate dict-like or attribute-like access."""
        if isinstance(obj, dict):
            return obj[k]
        return getattr(obj, k)

    def _load_as_items(self, fp: Path) -> Tuple[list[Any], list[str]]:
        """Load a pkl file and normalize into a list of per-molecule objects + names."""
        with open(fp, "rb") as f:
            obj = pickle.load(f)

        # Case 1: bundled dict with explicit items
        if isinstance(obj, dict) and "items" in obj and isinstance(obj["items"], (list, tuple)):
            items = list(obj["items"])
            names = obj.get("names")
            if isinstance(names, (list, tuple)) and len(names) == len(items):
                return items, [str(n) for n in names]
            return items, [f"{fp.name}:{i}" for i in range(len(items))]

        # Case 2: plain list/tuple (treated as a list of per-molecule records)
        if isinstance(obj, (list, tuple)):
            items = list(obj)
            return items, [f"{fp.name}:{i}" for i in range(len(items))]

        # Case 3: single-molecule pkl
        return [obj], [fp.name]

    def _get_bundle_items_cached(self, fp: Path) -> Tuple[list[Any], list[str]]:
        key = str(fp)
        hit = self._bundle_cache.get(key)
        if hit is not None:
            self._bundle_cache.move_to_end(key)
            return hit

        items, names = self._load_as_items(fp)
        self._bundle_cache[key] = (items, names)
        self._bundle_cache.move_to_end(key)

        # Enforce a small LRU cache
        max_size = max(0, int(self.bundle_cache_size))
        if max_size == 0:
            self._bundle_cache.clear()
        else:
            while len(self._bundle_cache) > max_size:
                self._bundle_cache.popitem(last=False)

        return items, names

    @staticmethod
    def collate_fn(batch_items: List[Dict[str, Any]]):
        # collate ligand/pocket using existing utilities to get 'mask'
        out: Dict[str, Any] = {}
        for entity in ["ligand", "pocket"]:
            batch = [x[entity] for x in batch_items]
            out[entity] = TensorDict(**collate_entity(batch))

        out["energy"] = torch.cat([x["energy"] for x in batch_items], dim=0)  # (B,)
        out["force"] = torch.cat([x["force"] for x in batch_items], dim=0)    # (sumN, 3)
        return out


def split_pkl_file_paths(
    file_paths: Sequence[str | Path],
    *,
    val_fraction: float,
    seed: int = 42,
    val_n: Optional[int] = None,
) -> tuple[list[Path], list[Path]]:
    """Deterministically split a list of pkl files into train/val.

    Uses a local RNG so results don't depend on global random state.
    """
    fps = [Path(p) for p in file_paths]
    fps = sorted(fps)
    g = torch.Generator()
    g.manual_seed(int(seed))

    if len(fps) == 0:
        return [], []

    perm = torch.randperm(len(fps), generator=g).tolist()
    fps = [fps[i] for i in perm]

    if val_n is None:
        if not (0.0 < val_fraction < 1.0):
            raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")
        val_n = max(1, int(round(val_fraction * len(fps))))
    else:
        val_n = int(val_n)
        val_n = max(1, min(val_n, len(fps) - 1))

    val = fps[:val_n]
    train = fps[val_n:]
    return train, val
