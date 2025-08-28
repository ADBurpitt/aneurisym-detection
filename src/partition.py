from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set

import numpy as np
import warnings
from pydicom import dcmread
from pydicom.dataset import FileDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

from src.dataloader import AneurysmVolumeDataset


def _slice_normal(ds0: FileDataset) -> Optional[np.ndarray]:
    """Unit normal from ImageOrientationPatient (row x col)."""
    iop = getattr(ds0, "ImageOrientationPatient", None)
    if not (iop and len(iop) == 6):
        return None
    r = np.asarray(iop[:3], float)
    c = np.asarray(iop[3:], float)
    r /= max(np.linalg.norm(r), 1e-8)
    c /= max(np.linalg.norm(c), 1e-8)
    n = np.cross(r, c)
    n /= max(np.linalg.norm(n), 1e-8)
    return n


def _robust_z_spacing_from_headers(dsets: List[FileDataset]) -> float:
    """Median inter-slice distance via IPP·normal (headers only)."""
    n = _slice_normal(dsets[0])
    if n is None:
        return float(
            getattr(
                dsets[0],
                "SpacingBetweenSlices",
                getattr(dsets[0], "SliceThickness", 1.0),
            )
        )
    coords: List[float] = []
    for ds in dsets:
        ipp = getattr(ds, "ImagePositionPatient", None)
        if not (isinstance(ipp, (list, tuple)) and len(ipp) == 3):
            return float(
                getattr(
                    dsets[0],
                    "SpacingBetweenSlices",
                    getattr(dsets[0], "SliceThickness", 1.0),
                )
            )
        coords.append(float(np.dot(np.asarray(ipp, float), n)))
    coords = np.asarray(coords)
    diffs = np.diff(np.sort(coords))
    if diffs.size == 0:
        return float(
            getattr(
                dsets[0],
                "SpacingBetweenSlices",
                getattr(dsets[0], "SliceThickness", 1.0),
            )
        )
    return float(np.median(np.abs(diffs)))


def split_series_by_z(
    series_root: Path, z_thresh_mm: float = 1.5
) -> Tuple[List[Path], List[Path]]:
    """Return (thin, thick) series lists using only DICOM headers (fast).

    thin: z-spacing <= z_thresh_mm
    thick: z-spacing  > z_thresh_mm
    """
    thin: List[Path] = []
    thick: List[Path] = []

    for p in sorted(d for d in series_root.iterdir() if d.is_dir()):
        files = sorted(p.glob("*.dcm"))
        if not files:
            continue
        dsets = [dcmread(f, stop_before_pixels=True) for f in files]
        z = _robust_z_spacing_from_headers(dsets)
        (thin if z <= z_thresh_mm else thick).append(p)
    return thin, thick


def make_loaders(
    series_root: Path,
    labels_map: Dict[str, np.ndarray],
    *,
    z_thresh_mm: float = 1.5,
    thin_out: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    thick_xy: float = 1.0,
    thin_patch: Tuple[int, int, int] = (96, 192, 192),
    thick_patch: Tuple[int, int, int] = (16, 192, 192),
    batch_size_thin: int = 1,
    batch_size_thick: int = 1,
    num_workers: int = 0,
    shuffle: bool = True,
    uid_filter: Optional[Set[str]] = None,
) -> Tuple[DataLoader, DataLoader, AneurysmVolumeDataset, AneurysmVolumeDataset]:
    """Build two DataLoaders (thin, thick) with sensible spacings/patches.

    - Thin: resample to isotropic ``thin_out``.
    - Thick: keep native Z, resample XY to ``thick_xy`` via ``out_spacing=(None,thick_xy,thick_xy)``.
    """
    thin_dirs, thick_dirs = split_series_by_z(series_root, z_thresh_mm=z_thresh_mm)

    if uid_filter is not None:
        thin_dirs = [p for p in thin_dirs if p.name in uid_filter]
        thick_dirs = [p for p in thick_dirs if p.name in uid_filter]

    ds_thin = AneurysmVolumeDataset(
        thin_dirs,
        labels_map=labels_map,
        out_spacing=thin_out,
        patch_size=thin_patch,
        patch_mode="random",
        resampler="scipy",
        cache_root=Path("cache/volumes"),
    )

    ds_thick = AneurysmVolumeDataset(
        thick_dirs,
        labels_map=labels_map,
        out_spacing=(None, thick_xy, thick_xy),  # keep Z native
        patch_size=thick_patch,
        patch_mode="random",
        resampler="scipy",
        cache_root=Path("cache/volumes"),
    )

    # If a dataset is empty, force shuffle=False to avoid RandomSampler(num_samples=0)
    shuffle_thin = shuffle and (len(ds_thin) > 0)
    shuffle_thick = shuffle and (len(ds_thick) > 0)

    if len(ds_thin) == 0:
        warnings.warn("make_loaders: thin split is empty after filtering.")
    if len(ds_thick) == 0:
        warnings.warn("make_loaders: thick split is empty after filtering.")

    dl_thin = DataLoader(
        ds_thin,
        batch_size=batch_size_thin,
        shuffle=shuffle_thin,
        num_workers=num_workers,
    )
    dl_thick = DataLoader(
        ds_thick,
        batch_size=batch_size_thick,
        shuffle=shuffle_thick,
        num_workers=num_workers,
    )

    return dl_thin, dl_thick, ds_thin, ds_thick


def scan_spacing_stats(series_root: Path) -> List[Tuple[str, float]]:
    """Quick per-series z-spacing summary [(uid, z_mm), ...]."""
    report: List[Tuple[str, float]] = []
    for p in sorted(d for d in series_root.iterdir() if d.is_dir()):
        files = sorted(p.glob("*.dcm"))
        if not files:
            continue
        dsets = [dcmread(f, stop_before_pixels=True) for f in files]
        z = _robust_z_spacing_from_headers(dsets)
        report.append((p.name, float(z)))
    return report


def safe_stratified_split(uids, mods, aps, val_frac=0.25, seed=1337):
    uids = np.asarray(uids)
    mods = np.asarray(mods).astype(str)
    aps = np.asarray(aps).astype(int)

    rng = np.random.default_rng(seed)

    def try_split(strata):
        # check counts
        _, counts = np.unique(strata, return_counts=True)
        if counts.min() < 2:
            return None
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        try:
            tr_idx, va_idx = next(sss.split(uids, strata))
            return tr_idx, va_idx
        except Exception:
            return None

    # 1) Modality × AP
    strata_combo = np.array([f"{m}_{a}" for m, a in zip(mods, aps)])
    res = try_split(strata_combo)

    # 2) AP only
    if res is None:
        res = try_split(aps)

    # 3) random split
    if res is None:
        n = len(uids)
        idx = rng.permutation(n)
        cut = max(1, int(round(n * val_frac)))
        va_idx = idx[:cut]
        tr_idx = idx[cut:]
        res = (tr_idx, va_idx)

    tr_idx, va_idx = res
    train_uids, val_uids = uids[tr_idx].tolist(), uids[va_idx].tolist()

    # 4) ensure both have at least one AP=1 if possible
    def ap_count(idxs):
        return int(np.sum(aps[idxs] == 1))

    if ap_count(tr_idx) == 0 and ap_count(va_idx) > 1:
        # move one positive from val->train
        pos_va = np.where(aps[va_idx] == 1)[0][0]
        tr_idx = np.concatenate([tr_idx, [va_idx[pos_va]]])
        va_idx = np.delete(va_idx, pos_va)
    if ap_count(va_idx) == 0 and ap_count(tr_idx) > 1:
        # move one positive from train->val
        pos_tr = np.where(aps[tr_idx] == 1)[0][0]
        va_idx = np.concatenate([va_idx, [tr_idx[pos_tr]]])
        tr_idx = np.delete(tr_idx, pos_tr)

    train_uids, val_uids = uids[tr_idx].tolist(), uids[va_idx].tolist()
    return train_uids, val_uids
