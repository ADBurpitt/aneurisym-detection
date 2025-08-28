from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydicom import dcmread
from pydicom.dataset import FileDataset
import torch
from torch.utils.data import Dataset
import scipy.ndimage as ndi


# --- Cache utils ---
def _safe_uid(series_dir: Path) -> str:
    return series_dir.name


def _cache_key(
    uid: str,
    out_spacing: Optional[Tuple[Optional[float], Optional[float], Optional[float]]],
    hu_window: Tuple[float, float],
    normalize: bool,
) -> str:
    os_key = tuple(
        x if (x is None or isinstance(x, float)) else float(x)
        for x in (out_spacing or (None, None, None))
    )
    return f"{uid}__spc-{os_key}__hu-{hu_window}__norm-{int(normalize)}"


def _cache_paths(cache_root: Optional[Path], key: str) -> Optional[Path]:
    if cache_root is None:
        return None
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / f"{key}.npz"


# ----------------------------
# DICOM geometry helpers
# ----------------------------


def _slice_axes(
    ds0: FileDataset,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (row, col, normal) unit vectors from ImageOrientationPatient if present."""
    iop = getattr(ds0, "ImageOrientationPatient", None)
    if iop and len(iop) == 6:
        r = np.asarray(iop[:3], float)
        c = np.asarray(iop[3:], float)
        r /= max(np.linalg.norm(r), 1e-8)
        c /= max(np.linalg.norm(c), 1e-8)
        n = np.cross(r, c)
        n /= max(np.linalg.norm(n), 1e-8)
        return r, c, n
    return None, None, None


def _project_positions(
    datasets: List[FileDataset], n: np.ndarray
) -> Optional[np.ndarray]:
    """Project each slice IPP onto normal n -> 1D physical coordinate per slice."""
    coords = []
    for ds in datasets:
        ipp = getattr(ds, "ImagePositionPatient", None)
        if not (isinstance(ipp, (list, tuple)) and len(ipp) == 3):
            return None
        coords.append(float(np.dot(np.asarray(ipp, float), n)))
    return np.asarray(coords, float)


def _obliqueness_deg(n: Optional[np.ndarray]) -> float:
    """Angle (deg) between slice normal and nearest cardinal axis. 0 => axial-ish."""
    if n is None:
        return 0.0
    n = np.asarray(n, float)
    n /= max(np.linalg.norm(n), 1e-8)
    axes = np.eye(3)
    cos_max = float(np.max(np.abs(axes @ n)))
    angle = float(np.degrees(np.arccos(np.clip(cos_max, -1.0, 1.0))))
    return angle


# ----------------------------
# Intensity utilities
# ----------------------------


def _apply_rescale_and_photometric(ds: FileDataset, arr: np.ndarray) -> np.ndarray:
    """Apply RescaleSlope/Intercept (to HU-like) and fix MONOCHROME1 inversion."""
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    out = arr.astype(np.float32) * slope + intercept

    photo = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper()
    if photo == "MONOCHROME1":
        # Invert so that higher value = brighter
        out = out.max() - out + out.min()
    return out


def normalize_hu(
    vol: np.ndarray, min_hu: float = -1000.0, max_hu: float = 2000.0
) -> np.ndarray:
    """Clip and min-max scale to [0, 1]."""
    vol = np.clip(vol, min_hu, max_hu)
    vol = (vol - min_hu) / (max_hu - min_hu)
    return vol.astype(np.float32)


# ----------------------------
# Core loading + spacing
# ----------------------------


@dataclass
class DicomVolume:
    volume: np.ndarray  # (Z, Y, X), float32 HU-like
    spacing: Tuple[float, float, float]  # (z, y, x) in mm
    sop_uids: List[str]
    series_path: Path
    # Diagnostics (optional)
    obliqueness_deg: float = 0.0
    z_variability: float = 0.0  # std(dz)/median(dz)
    z_source: str = (
        "estimated"  # 'estimated' | 'SpacingBetweenSlices' | 'SliceThickness'
    )


def _sort_and_spacing(
    datasets: List[FileDataset], series_name: str = ""
) -> Tuple[List[FileDataset], Tuple[float, float, float], float, float, str]:
    """Robust sort by physical position and compute spacing.

    Returns: (sorted_datasets, (z,y,x) spacing, obliqueness_deg, z_variability, z_source)
    """
    r, c, n = _slice_axes(datasets[0])
    obliq = _obliqueness_deg(n)

    z_source = "estimated"
    z_var = 0.0

    if n is not None:
        positions = _project_positions(datasets, n)
    else:
        positions = None

    if positions is not None:
        order = np.argsort(positions)
        datasets = [datasets[i] for i in order]
        positions = positions[order]
        diffs = np.diff(positions)
        if diffs.size:
            z = float(np.median(np.abs(diffs)))
            z_var = float(np.std(diffs) / (z + 1e-6))
            z_source = "estimated"
        else:
            # Single slice: fall back to tag
            z = float(
                getattr(
                    datasets[0],
                    "SpacingBetweenSlices",
                    getattr(datasets[0], "SliceThickness", 1.0),
                )
            )
            z_source = (
                "SpacingBetweenSlices"
                if hasattr(datasets[0], "SpacingBetweenSlices")
                else "SliceThickness"
            )
    else:
        # Fallback: tag-based sort and spacing
        datasets.sort(key=lambda ds: float(getattr(ds, "InstanceNumber", 0)))
        first = datasets[0]
        if hasattr(first, "SpacingBetweenSlices"):
            z = float(getattr(first, "SpacingBetweenSlices"))
            z_source = "SpacingBetweenSlices"
        else:
            z = float(getattr(first, "SliceThickness", 1.0))
            z_source = "SliceThickness"

    # y/x from PixelSpacing
    first = datasets[0]
    px = getattr(first, "PixelSpacing", [1.0, 1.0])
    yx = (
        (float(px[0]), float(px[1]))
        if isinstance(px, (list, tuple)) and len(px) == 2
        else (1.0, 1.0)
    )

    # Warn on irregular spacing
    if z_var > 0.05:
        warnings.warn(
            f"[{series_name}] Irregular z-spacing detected (std/median={z_var:.3f}). Consider physical-space resampling."
        )

    return datasets, (z, yx[0], yx[1]), obliq, z_var, z_source


def load_dicom_series(series_path: Union[str, Path]) -> DicomVolume:
    """Load a DICOM series folder -> stacked 3D volume with spacing + diagnostics.

    Robust to single-frame (2D) and multi-frame (3D: [F,Y,X]) DICOMs:
    each frame is treated as one Z-slice so the final volume is always (Z,Y,X).
    """
    series_path = Path(series_path)
    files = sorted(series_path.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No DICOM files found in {series_path}")

    # Read headers first so we can sort/space before touching pixels
    datasets: List[FileDataset] = [dcmread(f, stop_before_pixels=True) for f in files]
    datasets, spacing, obliq, z_var, z_src = _sort_and_spacing(
        datasets, series_name=series_path.name
    )

    # Re-read with pixel data in sorted order
    datasets = [dcmread(Path(ds.filename), stop_before_pixels=False) for ds in datasets]

    imgs: List[np.ndarray] = []
    sop_uids: List[str] = []

    for ds in datasets:
        arr = ds.pixel_array  # could be 2D (Y,X) or 3D (F,Y,X)
        arr = _apply_rescale_and_photometric(ds, arr)

        if arr.ndim == 2:
            imgs.append(arr.astype(np.float32))
            sop_uids.append(str(getattr(ds, "SOPInstanceUID", "")))

        elif arr.ndim == 3:
            # Multi-frame: push each frame as a Z slice
            F = arr.shape[0]
            base_uid = str(getattr(ds, "SOPInstanceUID", ""))
            for i in range(F):
                imgs.append(arr[i].astype(np.float32))
                sop_uids.append(f"{base_uid}:{i}")

        else:
            raise ValueError(
                f"Unsupported pixel array ndim={arr.ndim} for {getattr(ds, 'SOPInstanceUID', 'unknown')}"
            )

    if len(imgs) == 0:
        raise ValueError(f"No frames found after reading pixels in {series_path}")

    vol = np.stack(imgs, axis=0).astype(np.float32)  # (Z,Y,X)

    # Safety: ensure 3D
    if vol.ndim == 2:
        vol = vol[None, ...]
    elif vol.ndim != 3:
        vol = np.squeeze(vol)
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D volume after squeeze, got {vol.shape}")

    return DicomVolume(
        volume=vol,
        spacing=spacing,
        sop_uids=sop_uids,
        series_path=series_path,
        obliqueness_deg=obliq,
        z_variability=z_var,
        z_source=z_src,
    )


# ----------------------------
# Resampling
# ----------------------------


def _resample_scipy(
    vol: np.ndarray,
    spacing: Tuple[float, float, float],
    new_spacing: Tuple[float, float, float],
    order: int = 1,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    # Guard: only 3D supported here
    vol = np.ascontiguousarray(vol)
    if vol.ndim != 3:
        raise ValueError(f"resample expects 3D (Z,Y,X), got {vol.shape}")

    z, y, x = spacing
    nz, ny, nx = new_spacing
    zooms = [z / nz, y / ny, x / nx]

    # Guard against zero/negative target dims
    pred = [int(round(s * zm)) for s, zm in zip(vol.shape, zooms)]
    for i in range(3):
        if pred[i] < 1 or not np.isfinite(zooms[i]):
            zooms[i] = 1.0
            if i == 0:
                nz = z
            elif i == 1:
                ny = y
            else:
                nx = x
            pred[i] = vol.shape[i]

    if any(abs(zm - 1.0) > 1e-3 for zm in zooms):
        out = ndi.zoom(vol, zoom=tuple(zooms), order=order)
        return out.astype(np.float32), (nz, ny, nx)
    return vol.astype(np.float32), spacing


def resample_volume(
    vol: np.ndarray,
    spacing: Tuple[float, float, float],
    new_spacing: Tuple[float, float, float],
    order: int = 1,
    method: str = "scipy",
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Resample using SciPy only (method arg is ignored to keep API stable)."""
    return _resample_scipy(vol, spacing, new_spacing, order=order)


# ----------------------------
# Dataset
# ----------------------------

TARGET_COLS_DEFAULT: List[str] = [
    "Left Infraclinoid Internal Carotid Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery",
    "Right Middle Cerebral Artery",
    "Anterior Communicating Artery",
    "Left Anterior Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Posterior Communicating Artery",
    "Right Posterior Communicating Artery",
    "Basilar Tip",
    "Other Posterior Circulation",
    "Aneurysm Present",
]


class AneurysmVolumeDataset(Dataset):
    """Model-ready dataset yielding full volumes or patches.

    Sample dict keys:
      - volume: FloatTensor [C,Z,Y,X] (C=1 if add_channel)
      - spacing: Tuple[float,float,float] (z,y,x) after resample
      - uid, label (optional), crop_origin, raw_spacing/raw_shape, resample_engine, zooms, diagnostics.
    """

    def __init__(
        self,
        series_dirs: Sequence[Union[str, Path]],
        labels_map: Optional[Dict[str, np.ndarray]] = None,
        target_cols: Sequence[str] = TARGET_COLS_DEFAULT,
        out_spacing: Optional[
            Tuple[Optional[float], Optional[float], Optional[float]]
        ] = (1.0, 1.0, 1.0),
        add_channel: bool = True,
        patch_size: Optional[Tuple[int, int, int]] = None,
        patch_mode: str = "center",
        transform: Optional[Callable] = None,
        normalize: bool = True,
        hu_window: Tuple[float, float] = (-1000.0, 2000.0),
        resampler: str = "auto",  # 'auto' | 'scipy' | 'physical'
        irregularity_threshold: float = 0.05,
        oblique_threshold_deg: float = 10.0,
        warn_large_upsample_factor: float = 4.0,
        cache_root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.series_dirs = [Path(p) for p in series_dirs]
        self.labels_map = labels_map or {}
        self.target_cols = list(target_cols)
        self.out_spacing = out_spacing
        self.add_channel = add_channel
        self.patch_size = patch_size
        self.patch_mode = patch_mode
        self.transform = transform
        self.normalize = normalize
        self.hu_window = hu_window
        self.resampler = resampler
        self.irregularity_threshold = irregularity_threshold
        self.oblique_threshold_deg = oblique_threshold_deg
        self.warn_large_upsample_factor = warn_large_upsample_factor
        self.cache_root = Path(cache_root) if cache_root is not None else None

    def __len__(self) -> int:
        return len(self.series_dirs)

    def _extract_patch(
        self, vol: np.ndarray, size: Tuple[int, int, int], mode: str = "center"
    ) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        Z, Y, X = vol.shape
        dz, dy, dx = size
        dz, dy, dx = min(dz, Z), min(dy, Y), min(dx, X)

        if mode == "center":
            cz, cy, cx = Z // 2, Y // 2, X // 2
        else:
            rng = np.random.default_rng()
            cz = int(rng.integers(dz // 2, max(Z - (dz - dz // 2), dz // 2 + 1)))
            cy = int(rng.integers(dy // 2, max(Y - (dy - dy // 2), dy // 2 + 1)))
            cx = int(rng.integers(dx // 2, max(X - (dx - dx // 2), dx // 2 + 1)))

        z0 = max(0, cz - dz // 2)
        y0 = max(0, cy - dy // 2)
        x0 = max(0, cx - dx // 2)
        z1, y1, x1 = min(Z, z0 + dz), min(Y, y0 + dy), min(X, x0 + dx)
        return vol[z0:z1, y0:y1, x0:x1], (z0, y0, x0)

    def __getitem__(
        self, idx: int
    ) -> Dict[
        str,
        Union[
            str,
            torch.Tensor,
            Tuple[int, int, int],
            Tuple[float, float, float],
            Dict[str, float],
        ],
    ]:
        series_dir = self.series_dirs[idx]
        uid = series_dir.name
        cache_key = _cache_key(uid, self.out_spacing, self.hu_window, self.normalize)
        cache_path = _cache_paths(self.cache_root, cache_key)

        # Try cache
        if cache_path is not None and cache_path.exists():
            data = np.load(cache_path)
            vol = data["vol"].astype(np.float32)  # (Z,Y,X) in [0,1] if normalize=True
            spacing = tuple(float(x) for x in data["spacing"])
            raw_spacing = tuple(float(x) for x in data["raw_spacing"])
            raw_shape = tuple(int(x) for x in data["raw_shape"])
            engine = "cache"
        else:
            # Load + normalize + resample (your existing logic)
            dv = load_dicom_series(series_dir)
            vol, raw_spacing = dv.volume, dv.spacing
            raw_shape = tuple(vol.shape)
            if vol.ndim == 2:
                vol = vol[None, ...]
            elif vol.ndim != 3:
                vol = np.squeeze(vol)
                if vol.ndim != 3:
                    raise ValueError(f"{uid}: expected 3D (Z,Y,X), got {vol.shape}")

            if self.normalize:
                vol = normalize_hu(vol, *self.hu_window)

            engine = "scipy"
            spacing = raw_spacing
            if self.out_spacing is not None:
                target_spacing = tuple(
                    rs if ts is None else float(ts)
                    for rs, ts in zip(raw_spacing, self.out_spacing)
                )
                vol, spacing = resample_volume(
                    vol, raw_spacing, target_spacing, order=1, method=engine
                )

            # Save cache
            if cache_path is not None:
                np.savez_compressed(
                    cache_path,
                    vol=vol.astype(np.float32),
                    spacing=np.asarray(spacing, np.float32),
                    raw_spacing=np.asarray(raw_spacing, np.float32),
                    raw_shape=np.asarray(raw_shape, np.int32),
                )

        # Optional patch
        crop_origin: Optional[Tuple[int, int, int]] = None
        if self.patch_size is not None:
            vol, crop_origin = self._extract_patch(
                vol, self.patch_size, self.patch_mode
            )

        # to torch tensor [Z,Y,X] -> optionally [1,Z,Y,X]
        t = torch.from_numpy(vol)  # float32
        if self.add_channel:
            t = t.unsqueeze(0)
        if self.transform is not None:
            t = self.transform(t)

        zooms = tuple(rs / ts for rs, ts in zip(raw_spacing, spacing))
        diagnostics = {
            "obliqueness_deg": float(dv.obliqueness_deg)
            if dv.obliqueness_deg is not None
            else 0.0,
            "z_variability": float(dv.z_variability)
            if dv.z_variability is not None
            else 0.0,
            "z_source": dv.z_source,
        }

        sample = {
            "volume": t.float(),
            "spacing": spacing,
            "uid": uid,
            "raw_spacing": raw_spacing,
            "raw_shape": raw_shape,
            "resample_engine": engine,
            "zooms": zooms,
            "diagnostics": diagnostics,
        }
        if crop_origin is not None:
            sample["crop_origin"] = crop_origin
        if uid in self.labels_map:
            sample["label"] = torch.as_tensor(self.labels_map[uid], dtype=torch.float32)
        return sample


# ----------------------------
# Labels map helper
# ----------------------------


def build_labels_map_from_csv(
    train_csv_path: Union[str, Path], target_cols: Sequence[str] = TARGET_COLS_DEFAULT
) -> Dict[str, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(train_csv_path)
    need = ["SeriesInstanceUID", *target_cols]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in train.csv: {missing}")

    labels_map: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        uid = str(row["SeriesInstanceUID"])  # Kaggle UID == folder name
        vec = row[target_cols].astype(float).to_numpy(dtype=np.float32)
        labels_map[uid] = vec
    return labels_map
