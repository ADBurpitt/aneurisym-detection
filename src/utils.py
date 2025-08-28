import torch
import numpy as np


def set_seed(seed: int = 1337):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_targets(sample, target_cols):
    if "label" in sample:
        return sample["label"]
    return torch.zeros(len(target_cols), dtype=torch.float32)


def alt_iter(dl_a, dl_b):
    from itertools import zip_longest

    for a, b in zip_longest(dl_a, dl_b, fillvalue=None):
        if a is not None:
            yield a, "thin"
        if b is not None:
            yield b, "thick"


def unpack_sample_from_batch(batch):
    sample = {
        k: (
            v[0]
            if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.size(0) == 1
            else v
        )
        for k, v in batch.items()
    }
    if isinstance(sample.get("volume"), torch.Tensor) and sample["volume"].ndim == 5:
        sample["volume"] = sample["volume"][0]
    for k, v in list(sample.items()):
        if (
            not isinstance(v, torch.Tensor)
            and isinstance(v, (list, tuple))
            and len(v) == 1
        ):
            sample[k] = v[0]
    return sample


def compute_pos_weight_from_labels(
    labels_map, uids, cap: float = 50.0, laplace: float = 1.0, device="cpu"
):
    labels = np.stack([labels_map[u] for u in uids if u in labels_map], 0).astype(
        np.float32
    )
    pos = labels.sum(0)
    neg = labels.shape[0] - pos
    pw = (neg + laplace) / (pos + laplace)
    pw = np.clip(pw, 1.0, cap)
    return torch.as_tensor(pw, dtype=torch.float32, device=device)
