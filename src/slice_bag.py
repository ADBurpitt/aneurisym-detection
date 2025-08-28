from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def make_slice_bag(
    sample: Dict,
    k: int = 5,
    stride: int = 2,
    resize: Tuple[int, int] = (224, 224),
    center: bool = True,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Return bag [N, k, H, W] by clamping z-indices at edges (no F.pad).
    """
    vol = sample["volume"]  # [1,Z,Y,X] or [Z,Y,X]
    if vol.ndim == 4:
        vol = vol.squeeze(0)  # -> [Z,Y,X]
    if vol.ndim != 3:
        raise ValueError(
            f"Expected [Z,Y,X] or [1,Z,Y,X], got {tuple(sample['volume'].shape)}"
        )

    Z, Y, X = vol.shape
    half = k // 2

    # choose centers
    if center:
        start = half
        centers = list(range(start, Z, stride))
        if centers and centers[-1] > Z - 1 - half:
            centers[-1] = max(centers[-1], Z - 1 - half)
        if Z <= k:
            centers = [min(max(half, Z // 2), Z - 1 - half)]
    else:
        centers = [i + half for i in range(0, max(1, Z - k + 1), stride)]
        if not centers:
            centers = [min(half, Z - 1)]

    bag: List[torch.Tensor] = []
    true_centers: List[int] = []

    for c in centers:
        # build clamped z-index window
        z_idx = torch.arange(c - half, c + half + 1)
        z_idx = z_idx.clamp(0, Z - 1)

        # gather k slices -> [k,Y,X]
        slc = vol[z_idx, :, :]  # [k, Y, X]

        # resize to (H,W) using 2D bilinear: expect [N,C,H,W]
        slc = slc.unsqueeze(0)  # [1,k,Y,X]
        slc = F.interpolate(slc, size=resize, mode="bilinear", align_corners=False)
        bag.append(slc.squeeze(0))  # [k,H,W]
        true_centers.append(int(c))

    # fallback if something went weird
    if not bag:
        mid = Z // 2
        slc = vol[mid : mid + 1].repeat(k, 1, 1).unsqueeze(0)  # [1,k,Y,X]
        slc = F.interpolate(slc, size=resize, mode="bilinear", align_corners=False)
        bag = [slc.squeeze(0)]
        true_centers = [mid]

    return torch.stack(bag, dim=0), true_centers  # [N, k, H, W]


def batch_make_bags(
    batch: Dict,
    k: int = 5,
    stride: int = 2,
    resize: Tuple[int, int] = (224, 224),
) -> Tuple[List[torch.Tensor], List[List[int]]]:
    """Vectorized-ish helper: make a bag per item in a batch (B may be 1).
    Returns a list of bags (each [N_i, k, H, W]) and a list of centers per bag.
    """
    bags: List[torch.Tensor] = []
    centers_all: List[List[int]] = []
    vols = batch["volume"]  # [B,1,Z,Y,X]
    B = vols.shape[0]
    for b in range(B):
        sample = {
            k2: (v[b] if isinstance(v, torch.Tensor) else v) for k2, v in batch.items()
        }
        bag, centers = make_slice_bag(sample, k=k, stride=stride, resize=resize)
        bags.append(bag)
        centers_all.append(centers)
    return bags, centers_all
