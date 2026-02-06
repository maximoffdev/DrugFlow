from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.data_utils import TensorDict, collate_entity


@dataclass(frozen=True)
class ANIH5Keys:
    atomic_numbers: str = "atomic_numbers"
    coordinates: str = "coordinates"
    energy: str = "wb97x_dz.energy"
    forces: str = "wb97x_dz.forces"


IndexItem = Tuple[str, int]  # (group_name, conformation_index)


class ANIH5EnergyForceDataset(Dataset):
    """Loads per-conformation samples from an ANI-style HDF5 file.

    The HDF5 layout is expected to match the common ANI datasets:
      /<group>/atomic_numbers: (Na,)
      /<group>/coordinates: (Nc, Na, 3)
      /<group>/<energy_key>: (Nc,) or (Nc,1)
      /<group>/<forces_key>: (Nc, Na, 3)

    We flatten all valid (group, conformation) pairs into a single index.

    Output per-sample matches :class:`src.data.pkl_dataset.PKLEnergyForceDataset`:
      {
        'ligand': {'x': (N,3), 'one_hot': (N,K), 'size': int, 'name': str},
        'pocket': {'x': (0,3), 'one_hot': (0,0), 'size': 0, 'name': str},
        'energy': (1,),
        'force': (N,3),
      }

    Notes:
      - For performance + safety with multi-worker DataLoaders, this dataset opens
        the HDF5 file lazily per-process.
      - By default, conformations containing NaNs in energy/forces are filtered out.
    """

    def __init__(
        self,
        h5_path: str | Path,
        *,
        index: Optional[Sequence[IndexItem]] = None,
        keys: ANIH5Keys = ANIH5Keys(),
        allowed_atomic_numbers: Optional[Sequence[int]] = None,
        num_atom_types: Optional[int] = None,
        scan_limit_groups: Optional[int] = None,
        filter_nan: bool = True,
        filter_nan_keys: Optional[Sequence[str]] = None,
        scan_chunk_size: int = 128,
        device: str | torch.device = "cpu",
        conformation_limit: Optional[int] = None,
    ):
        super().__init__()
        self.h5_path = Path(h5_path)
        self.keys = keys
        self.allowed_atomic_numbers: list[int] | None = list(allowed_atomic_numbers) if allowed_atomic_numbers is not None else None
        num_atom_types_in: int | None = int(num_atom_types) if num_atom_types is not None else None
        self.scan_limit_groups = scan_limit_groups
        self.filter_nan = bool(filter_nan)
        self.filter_nan_keys = list(filter_nan_keys) if filter_nan_keys is not None else None
        self.scan_chunk_size = int(scan_chunk_size)
        self.device = device
        self.conformation_limit = conformation_limit

        self._h5 = None  # lazy h5py.File

        if self.allowed_atomic_numbers is None:
            self.allowed_atomic_numbers = self._infer_allowed_atomic_numbers(limit_groups=self.scan_limit_groups)
        allowed_z = sorted(set(int(z) for z in self.allowed_atomic_numbers))
        if num_atom_types_in is None:
            num_atom_types_val = len(allowed_z)
        else:
            num_atom_types_val = int(num_atom_types_in)
        if num_atom_types_val != len(allowed_z):
            raise ValueError(
                f"num_atom_types ({num_atom_types_val}) must match allowed_atomic_numbers length ({len(allowed_z)})"
            )

        # Freeze to non-optional concrete types after validation.
        self.allowed_atomic_numbers = allowed_z
        self.num_atom_types: int = num_atom_types_val
        self.atomic_number_to_index: dict[int, int] = {int(z): i for i, z in enumerate(self.allowed_atomic_numbers)}

        if index is None:
            self.index: List[IndexItem] = self._build_index()
        else:
            self.index = [(str(g), int(i)) for (g, i) in index]

        if self.conformation_limit is not None:
            self.index = self.index[: int(self.conformation_limit)]

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_h5"] = None
        return state

    def close(self) -> None:
        h5 = self._h5
        self._h5 = None
        if h5 is not None:
            try:
                h5.close()
            except Exception:
                pass

    def __del__(self):
        self.close()

    def _get_h5(self):
        if self._h5 is None:
            try:
                import h5py  # type: ignore
            except Exception as e:
                raise ImportError(
                    "ANIH5EnergyForceDataset requires 'h5py'. Install it via conda/pip (e.g. `conda install -c conda-forge h5py`)."
                ) from e
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        group_name, conf_idx = self.index[idx]
        f = self._get_h5()
        grp = f[group_name]

        z = np.asarray(grp[self.keys.atomic_numbers][()])
        x = np.asarray(grp[self.keys.coordinates][conf_idx])

        e_ds = grp[self.keys.energy]
        e = np.asarray(e_ds[conf_idx])

        f_ds = grp[self.keys.forces]
        forces = np.asarray(f_ds[conf_idx])

        # Normalize shapes
        if x.ndim != 2 or x.shape[-1] != 3:
            raise ValueError(f"Expected coordinates shape (N,3), got {tuple(x.shape)} in group={group_name}")
        if forces.shape != x.shape:
            raise ValueError(
                f"Expected forces shape to match coordinates (N,3), got forces={tuple(forces.shape)} coords={tuple(x.shape)} in group={group_name}"
            )

        # energy: scalar -> (1,)
        e = np.asarray(e, dtype=np.float32)
        if e.ndim == 0:
            e = e.reshape(1)
        elif e.ndim == 1 and e.size == 1:
            e = e.reshape(1)
        else:
            raise ValueError(f"Expected energy to be scalar for one conformation, got shape {tuple(e.shape)} in group={group_name}")

        # atomic numbers: (N,)
        zt = torch.as_tensor(z, device=self.device)
        if zt.dtype.is_floating_point:
            zt = zt.round().to(torch.long)
        else:
            zt = zt.to(torch.long)

        max_z = int(max(self.atomic_number_to_index.keys()))
        lut = torch.full((max_z + 1,), -1, device=zt.device, dtype=torch.long)
        for zz, ii in self.atomic_number_to_index.items():
            lut[int(zz)] = int(ii)
        if (zt < 0).any() or (zt > max_z).any() or (lut[zt] < 0).any():
            unknown = torch.unique(zt[(zt < 0) | (zt > max_z) | (lut[zt] < 0)]).tolist()
            raise ValueError(f"Found unknown atomic numbers {unknown} in group={group_name}. Allowed: {self.allowed_atomic_numbers}")

        labels = lut[zt]
        one_hot = torch.nn.functional.one_hot(labels, num_classes=int(self.num_atom_types)).to(torch.float32)

        pos = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        force_t = torch.as_tensor(forces, dtype=torch.float32, device=self.device)
        energy_t = torch.as_tensor(e, dtype=torch.float32, device=self.device)

        name = f"{group_name}:{conf_idx}"
        ligand = {
            "name": name,
            "x": pos,
            "one_hot": one_hot,
            "size": pos.size(0),
        }
        pocket = {
            "name": name,
            "x": torch.zeros((0, 3), dtype=pos.dtype, device=pos.device),
            "one_hot": torch.zeros((0, 0), dtype=one_hot.dtype, device=one_hot.device),
            "size": 0,
        }
        return {
            "ligand": ligand,
            "pocket": pocket,
            "energy": energy_t,
            "force": force_t,
        }

    def _infer_allowed_atomic_numbers(self, *, limit_groups: Optional[int] = None) -> List[int]:
        f = self._get_h5()
        uniq: set[int] = set()
        group_names = list(f.keys())
        if limit_groups is not None:
            group_names = group_names[: int(limit_groups)]
        if len(group_names) == 0:
            raise ValueError(f"No groups found in HDF5 file: {self.h5_path}")

        for g in group_names:
            grp = f[g]
            z = np.asarray(grp[self.keys.atomic_numbers][()])
            zt = torch.as_tensor(z)
            if zt.dtype.is_floating_point:
                zt = zt.round().to(torch.long)
            else:
                zt = zt.to(torch.long)
            uniq.update(int(v) for v in torch.unique(zt).tolist())

        if len(uniq) == 0:
            raise ValueError("Could not infer any atomic numbers during scan")
        return sorted(uniq)

    def _build_index(self) -> List[IndexItem]:
        if not self.h5_path.exists():
            raise FileNotFoundError(self.h5_path)

        f = self._get_h5()
        group_names = list(f.keys())
        if self.scan_limit_groups is not None:
            group_names = group_names[: int(self.scan_limit_groups)]

        if self.filter_nan_keys is None:
            filter_keys = [self.keys.energy, self.keys.forces]
        else:
            filter_keys = list(self.filter_nan_keys)

        out: List[IndexItem] = []

        for g in group_names:
            grp = f[g]
            coords_ds = grp[self.keys.coordinates]
            if coords_ds.ndim != 3 or coords_ds.shape[-1] != 3:
                raise ValueError(
                    f"Expected /{g}/{self.keys.coordinates} to have shape (Nc, Na, 3), got {tuple(coords_ds.shape)}"
                )
            nc = int(coords_ds.shape[0])

            if not self.filter_nan:
                out.extend((str(g), int(i)) for i in range(nc))
                continue

            valid = np.ones((nc,), dtype=bool)
            for k in filter_keys:
                if k in {self.keys.atomic_numbers, self.keys.coordinates}:
                    continue
                if k not in grp:
                    raise KeyError(f"Key {k!r} not found in group {g!r} of file {self.h5_path}")

                ds = grp[k]
                # Chunked scan to avoid loading huge arrays at once (especially forces).
                chunk = max(1, int(self.scan_chunk_size))
                for start in range(0, nc, chunk):
                    end = min(nc, start + chunk)
                    arr = np.asarray(ds[start:end])
                    arr2 = arr.reshape(arr.shape[0], -1)
                    bad = np.isnan(arr2).any(axis=1)
                    if bad.any():
                        valid[start:end] &= ~bad

            if int(valid.sum()) == 0:
                continue
            idxs = np.nonzero(valid)[0].astype(int).tolist()
            out.extend((str(g), int(i)) for i in idxs)

            if self.conformation_limit is not None and len(out) >= int(self.conformation_limit):
                out = out[: int(self.conformation_limit)]
                break

        if len(out) == 0:
            raise ValueError(
                "No valid conformations found while scanning HDF5. "
                "Check keys, or disable NaN filtering via filter_nan=False."
            )
        return out

    @staticmethod
    def collate_fn(batch_items: List[Dict[str, Any]]):
        out: Dict[str, Any] = {}
        for entity in ["ligand", "pocket"]:
            batch = [x[entity] for x in batch_items]
            out[entity] = TensorDict(**collate_entity(batch))

        out["energy"] = torch.cat([x["energy"] for x in batch_items], dim=0)  # (B,)
        out["force"] = torch.cat([x["force"] for x in batch_items], dim=0)    # (sumN, 3)
        return out


def split_ani_h5_index(
    index: Sequence[IndexItem],
    *,
    val_fraction: float,
    seed: int = 42,
    val_n: Optional[int] = None,
) -> tuple[list[IndexItem], list[IndexItem]]:
    """Deterministically split a flattened ANI-H5 index into train/val."""

    n = len(index)
    if n == 0:
        return [], []

    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).tolist()
    shuffled = [index[i] for i in perm]

    if val_n is None:
        if not (0.0 < float(val_fraction) < 1.0):
            raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")
        val_n = max(1, int(round(float(val_fraction) * n)))
    else:
        val_n = int(val_n)
        val_n = max(1, min(val_n, n - 1))

    val = list(shuffled[:val_n])
    train = list(shuffled[val_n:])
    return train, val


@lru_cache(maxsize=8)
def _cached_full_index(
    h5_path: str,
    *,
    atomic_numbers_key: str,
    coordinates_key: str,
    energy_key: str,
    forces_key: str,
    allowed_atomic_numbers: tuple[int, ...] | None,
    num_atom_types: int | None,
    scan_limit_groups: int | None,
    filter_nan: bool,
    scan_chunk_size: int,
    conformation_limit: int | None,
) -> tuple[IndexItem, ...]:
    """Build and cache the flattened (group, conformation) index.

    This keeps the expensive HDF5 NaN scan out of the model and avoids
    rescanning when train/val/test datasets are constructed.
    """

    keys = ANIH5Keys(
        atomic_numbers=atomic_numbers_key,
        coordinates=coordinates_key,
        energy=energy_key,
        forces=forces_key,
    )

    ds = ANIH5EnergyForceDataset(
        h5_path,
        keys=keys,
        allowed_atomic_numbers=list(allowed_atomic_numbers) if allowed_atomic_numbers is not None else None,
        num_atom_types=num_atom_types,
        scan_limit_groups=scan_limit_groups,
        filter_nan=filter_nan,
        scan_chunk_size=scan_chunk_size,
        conformation_limit=conformation_limit,
        device="cpu",
    )
    try:
        return tuple(ds.index)
    finally:
        ds.close()


def make_ani_h5_dataset_for_stage(
    h5_path: str | Path,
    *,
    stage: str,
    allowed_atomic_numbers: Sequence[int],
    num_atom_types: int,
    val_fraction: float,
    seed: int = 42,
    val_n: Optional[int] = None,
    atomic_numbers_key: str = "atomic_numbers",
    coordinates_key: str = "coordinates",
    energy_key: str = "wb97x_dz.energy",
    forces_key: str = "wb97x_dz.forces",
    filter_nan: bool = True,
    scan_limit_groups: Optional[int] = None,
    scan_chunk_size: int = 128,
    conformation_limit: Optional[int] = None,
    device: str | torch.device = "cpu",
) -> ANIH5EnergyForceDataset:
    """Create a train/val/test dataset from an ANI-style HDF5 file.

    Splits deterministically over conformations using the same semantics as the PKL loader.
    """

    h5_path = Path(h5_path)
    if not h5_path.exists() or not h5_path.is_file():
        raise FileNotFoundError(h5_path)

    full_index = _cached_full_index(
        str(h5_path),
        atomic_numbers_key=str(atomic_numbers_key),
        coordinates_key=str(coordinates_key),
        energy_key=str(energy_key),
        forces_key=str(forces_key),
        allowed_atomic_numbers=tuple(int(z) for z in allowed_atomic_numbers) if allowed_atomic_numbers is not None else None,
        num_atom_types=int(num_atom_types) if num_atom_types is not None else None,
        scan_limit_groups=int(scan_limit_groups) if scan_limit_groups is not None else None,
        filter_nan=bool(filter_nan),
        scan_chunk_size=int(scan_chunk_size),
        conformation_limit=int(conformation_limit) if conformation_limit is not None else None,
    )

    train_index, val_index = split_ani_h5_index(
        list(full_index),
        val_fraction=float(val_fraction),
        seed=int(seed),
        val_n=val_n,
    )

    if stage == "train":
        chosen = train_index
    elif stage in {"val", "test"}:
        chosen = val_index
    else:
        raise ValueError(f"Unknown stage: {stage}")

    keys = ANIH5Keys(
        atomic_numbers=str(atomic_numbers_key),
        coordinates=str(coordinates_key),
        energy=str(energy_key),
        forces=str(forces_key),
    )

    return ANIH5EnergyForceDataset(
        h5_path,
        index=chosen,
        keys=keys,
        allowed_atomic_numbers=list(allowed_atomic_numbers) if allowed_atomic_numbers is not None else None,
        num_atom_types=int(num_atom_types) if num_atom_types is not None else None,
        device=device,
    )
