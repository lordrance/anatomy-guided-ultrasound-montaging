"""Evaluation metrics.

The primary metric is mTRE, which is neutral between the two methods. Dice and
NCC are computed too, but each of them structurally favours one method -- Dice
is close to what ICP minimises, NCC is exactly what Method A maximises -- so
they are reported as diagnostics and never as the basis of a conclusion.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from data import apply


def mtre(estimated: np.ndarray, reference: np.ndarray, corners: np.ndarray) -> float:
    """Mean displacement of the moving volume's 8 bounding-box corners, mm.

    `corners` are in the moving volume's own physical frame; both transforms map
    them into the fixed volume's frame, and we measure how far apart they land.
    """
    return float(np.linalg.norm(
        apply(estimated, corners) - apply(reference, corners), axis=1).mean())


def overlap_dice(fixed: dict, moving: dict, transform: np.ndarray) -> float:
    """Dice of the two humerus masks with the moving one placed by `transform`.

    Evaluated on the 0.5 mm grids that the registration itself used.
    """
    fm = fixed["mask_fine"]
    mm = moving["mask_fine"]
    if not fm.any() or not mm.any():
        return float("nan")
    idx = np.argwhere(fm)
    phys = fixed["offset"] + idx * fixed["spacing_fine"]
    q = (apply(np.linalg.inv(transform), phys) - moving["offset"]) / moving["spacing_fine"]
    q = np.round(q).astype(int)
    inside = ((q >= 0) & (q < np.array(mm.shape))).all(axis=1)
    hit = np.zeros(len(q), bool)
    hit[inside] = mm[q[inside, 0], q[inside, 1], q[inside, 2]]
    return float(2.0 * hit.sum() / (fm.sum() + mm.sum()))


def overlap_ncc(fixed: dict, moving: dict, transform: np.ndarray,
                n: int = 30_000, seed: int = 0) -> float:
    """Intensity NCC over the overlapping foreground, on the 0.5 mm grids."""
    f, m = fixed["img_fine"], moving["img_fine"]
    idx = np.argwhere(f > 0)
    if len(idx) == 0:
        return float("nan")
    if len(idx) > n:
        idx = idx[np.random.default_rng(seed).choice(len(idx), n, replace=False)]
    phys = fixed["offset"] + idx * fixed["spacing_fine"]
    vals = f[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float64)
    q = (apply(np.linalg.inv(transform), phys) - moving["offset"]) / moving["spacing_fine"]
    inside = ((q >= 0) & (q <= np.array(m.shape) - 1)).all(axis=1)
    if inside.sum() < 50:
        return float("nan")
    got = ndimage.map_coordinates(m, q[inside].T, order=1, mode="constant", cval=0.0)
    keep = got > 0
    if keep.sum() < 50:
        return float("nan")
    a = vals[inside][keep] - vals[inside][keep].mean()
    b = got[keep].astype(np.float64)
    b -= b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 1e-12 else float("nan")


def bootstrap_diff(a: np.ndarray, b: np.ndarray, groups: np.ndarray,
                   n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Cluster bootstrap of mean(b) - mean(a), resampling whole trials.

    Registrations from the same trial share volumes and an operator, so they are
    not independent; resampling trials rather than registrations keeps the
    interval honest about that.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    diffs = np.empty(n)
    for i in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([np.flatnonzero(groups == g) for g in pick])
        diffs[i] = b[sel].mean() - a[sel].mean()
    return (float(b.mean() - a.mean()),
            float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
