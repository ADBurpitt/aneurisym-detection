import numpy as np
from sklearn.metrics import roc_auc_score


def weighted_auc_14(target_cols, y_true, y_pred):
    aucs = []
    for j, _ in enumerate(target_cols):
        tj, pj = y_true[:, j], y_pred[:, j]
        auc = np.nan
        if len(np.unique(tj)) > 1:
            try:
                auc = roc_auc_score(tj, pj)
            except Exception:
                pass
        aucs.append(auc)

    weights = np.ones(len(target_cols), dtype=float)
    ap_idx = target_cols.index("Aneurysm Present")
    weights[ap_idx] = 13.0

    aucs_arr = np.array(aucs, dtype=float)
    mask = ~np.isnan(aucs_arr)
    if not np.any(mask):
        return np.nan
    return float(np.sum(aucs_arr[mask] * weights[mask]) / np.sum(weights[mask]))
