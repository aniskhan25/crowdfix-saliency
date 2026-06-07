"""
Standard saliency prediction metrics.

All functions accept NumPy arrays. Input shapes are arbitrary (flattened internally).
Saliency maps are float arrays; fixation maps are binary (1 = fixated, 0 = not).
"""

import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


def _normalize_dist(x: np.ndarray) -> np.ndarray:
    s = x.sum()
    return x / (s + 1e-8)


def auc_judd(sal_map: np.ndarray, fix_map: np.ndarray, n_splits: int = 100) -> float:
    """AUC-Judd: ROC with fixated pixels as positives, all others as negatives."""
    sal = _normalize(sal_map).ravel()
    fix = fix_map.ravel() > 0
    if not fix.any():
        return float("nan")
    thresholds = np.linspace(0, 1, n_splits)
    tpr = np.array([(sal[fix] >= t).mean() for t in thresholds])
    fpr = np.array([(sal[~fix] >= t).mean() for t in thresholds])
    idx = np.argsort(fpr)
    return float(np.trapz(tpr[idx], fpr[idx]))


def auc_borji(
    sal_map: np.ndarray, fix_map: np.ndarray, n_splits: int = 100, n_samples: int = 100
) -> float:
    """AUC-Borji: negatives sampled uniformly from the image (not just non-fixated pixels)."""
    sal = _normalize(sal_map).ravel()
    fix = fix_map.ravel() > 0
    if not fix.any():
        return float("nan")
    rng = np.random.default_rng(0)
    aucs = []
    for _ in range(n_splits):
        neg_idx = rng.choice(len(sal), size=n_samples, replace=True)
        neg = sal[neg_idx]
        pos = sal[fix]
        thresholds = np.sort(np.unique(np.concatenate([pos, neg])))[::-1]
        tpr = np.array([(pos >= t).mean() for t in thresholds])
        fpr = np.array([(neg >= t).mean() for t in thresholds])
        tpr = np.r_[0.0, tpr, 1.0]
        fpr = np.r_[0.0, fpr, 1.0]
        aucs.append(float(np.trapz(tpr, fpr)))
    return float(np.mean(aucs))


def nss(sal_map: np.ndarray, fix_map: np.ndarray) -> float:
    """NSS: mean normalized saliency at fixation locations."""
    fix = fix_map.ravel() > 0
    if not fix.any():
        return float("nan")
    sal = _normalize(sal_map).ravel()  # min-max first so z-score is scale-invariant
    sal = (sal - sal.mean()) / (sal.std() + 1e-8)
    return float(sal[fix].mean())


def cc(sal_map: np.ndarray, gt_map: np.ndarray) -> float:
    """CC: Pearson correlation between predicted and ground-truth saliency distributions."""
    p = sal_map.astype(float).ravel()
    g = gt_map.astype(float).ravel()
    p = (p - p.mean()) / (p.std() + 1e-8)
    g = (g - g.mean()) / (g.std() + 1e-8)
    return float(np.corrcoef(p, g)[0, 1])


def kldiv(sal_map: np.ndarray, gt_map: np.ndarray) -> float:
    """KL divergence KL(GT || pred). Lower is better."""
    p = _normalize_dist(sal_map.astype(float).ravel())
    g = _normalize_dist(gt_map.astype(float).ravel())
    return float(np.sum(g * np.log((g + 1e-8) / (p + 1e-8))))


def sim(sal_map: np.ndarray, gt_map: np.ndarray) -> float:
    """SIM: histogram intersection similarity."""
    p = _normalize_dist(sal_map.astype(float).ravel())
    g = _normalize_dist(gt_map.astype(float).ravel())
    return float(np.minimum(p, g).sum())


def compute_all(
    sal_map: np.ndarray, gt_sal_map: np.ndarray, fix_map: np.ndarray
) -> dict:
    """Compute all 6 metrics. Returns a dict with float values."""
    return {
        "auc_judd": auc_judd(sal_map, fix_map),
        "auc_borji": auc_borji(sal_map, fix_map),
        "nss": nss(sal_map, fix_map),
        "cc": cc(sal_map, gt_sal_map),
        "kldiv": kldiv(sal_map, gt_sal_map),
        "sim": sim(sal_map, gt_sal_map),
    }
