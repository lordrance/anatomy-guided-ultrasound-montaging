"""The two registration methods.

Both estimate a 6-DoF rigid transform placing the moving volume in the fixed
volume's frame. They differ only in the signal that drives the alignment:

    Method A  image intensity  (normalised cross-correlation, Nelder-Mead)
    Method B  bone anatomy     (trimmed point-to-plane ICP on humerus surfaces)

Equal transformation capacity is the point of the comparison, so neither method
is allowed a scaling, affine or deformable term.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, optimize, spatial

from config import DEFAULT
from data import apply, rigid_from_vec


@dataclass
class Result:
    transform: np.ndarray                       # 4x4, moving -> fixed frame
    seconds: float
    method: str
    converged: bool = True
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Method A -- intensity


def _sample_points(img: np.ndarray, spacing: np.ndarray, offset: np.ndarray,
                   n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """`n` random foreground voxels of `img` as (physical positions, values)."""
    idx = np.argwhere(img > 0)
    if len(idx) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    if len(idx) > n:
        idx = idx[rng.choice(len(idx), n, replace=False)]
    vals = img[idx[:, 0], idx[:, 1], idx[:, 2]]
    return offset + idx * spacing, vals


def _ncc_objective(vec, pts, vals, centre, init, moving, m_off, m_sp, min_frac):
    """Negative NCC over the region where the two volumes actually overlap."""
    t = rigid_from_vec(vec, centre) @ init
    q = apply(np.linalg.inv(t), pts)
    idx = (q - m_off) / m_sp
    inside = ((idx >= 0) & (idx <= np.array(moving.shape) - 1)).all(axis=1)
    if inside.mean() < min_frac:
        return 2.0
    got = ndimage.map_coordinates(moving, idx[inside].T, order=1,
                                  mode="constant", cval=0.0)
    keep = got > 0
    if keep.sum() < min_frac * len(pts) or keep.sum() < 50:
        return 2.0
    a = vals[inside][keep].astype(np.float64)
    b = got[keep].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-12:
        return 2.0
    return -float((a * b).sum() / denom)


def register_intensity(fixed: dict, moving: dict, init: np.ndarray,
                       cfg=DEFAULT, seed: int = 0) -> Result:
    """Coarse-to-fine NCC rigid registration.

    Mirrors the automatic half of the source dataset's own pipeline, which used
    fast NCC optimised with the Nelder-Mead simplex (Sewify et al., S1 Text). We
    do not reproduce their expert manual adjustment stage.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    current = init.copy()
    trace = []

    levels = (("coarse", cfg.coarse_spacing, 4.0, 4.0),
              ("fine", cfg.fine_spacing, 2.0, 2.0))
    for name, _spacing, t_step, r_step in levels:
        f_img = fixed[f"img_{name}"]
        m_img = moving[f"img_{name}"]
        f_sp = fixed[f"spacing_{name}"]
        m_sp = moving[f"spacing_{name}"]
        pts, vals = _sample_points(f_img, f_sp, fixed["offset"], cfg.ncc_samples, rng)
        if len(pts) == 0:
            return Result(current, time.time() - t0, "intensity", False,
                          {"error": "fixed volume has no foreground"})

        centre = apply(current, moving["offset"] + m_sp * np.array(m_img.shape) / 2)
        steps = np.array([t_step] * 3 + [r_step] * 3)
        simplex = np.vstack([np.zeros(6), np.diag(steps)])

        res = optimize.minimize(
            _ncc_objective, np.zeros(6),
            args=(pts, vals, centre, current, m_img, moving["offset"], m_sp,
                  cfg.ncc_min_overlap),
            method="Nelder-Mead",
            options=dict(initial_simplex=simplex, maxiter=cfg.ncc_maxiter,
                         maxfev=cfg.ncc_maxiter * 2, xatol=1e-3, fatol=1e-6),
        )
        current = rigid_from_vec(res.x, centre) @ current
        trace.append(dict(level=name, ncc=round(-float(res.fun), 5),
                          nfev=int(res.nfev), success=bool(res.success)))

    return Result(current, time.time() - t0, "intensity",
                  all(t["success"] for t in trace), {"levels": trace})


# --------------------------------------------------------------------------- #
# Method B -- anatomy


def _point_to_plane_step(src: np.ndarray, dst: np.ndarray,
                         nrm: np.ndarray) -> np.ndarray:
    """One linearised point-to-plane update, returned as a 4x4.

    Minimises sum_i (n_i . (R a_i + t - b_i))^2 for small rotations, which gives
    the usual 6x6 normal equations with rows [a_i x n_i, n_i].
    """
    a = np.hstack([np.cross(src, nrm), nrm])
    b = -np.einsum("ij,ij->i", nrm, src - dst)
    x, *_ = np.linalg.lstsq(a, b, rcond=None)
    alpha, t = x[:3], x[3:]
    kx = np.array([[0, -alpha[2], alpha[1]],
                   [alpha[2], 0, -alpha[0]],
                   [-alpha[1], alpha[0], 0]])
    r = np.eye(3) + kx
    u, _, vt = np.linalg.svd(r)          # re-orthonormalise the linearised R
    r = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = t
    return m


def register_anatomy(fixed: dict, moving: dict, init: np.ndarray,
                     cfg=DEFAULT, seed: int = 0) -> Result:
    """Trimmed point-to-plane ICP on the two humerus surfaces.

    Trimming matters: the two volumes see different, partly disjoint pieces of
    the humerus, so a large share of the points in one surface have no true
    counterpart in the other and must be allowed to drop out.
    """
    t0 = time.time()
    src0 = moving["surf_pts"].astype(np.float64)
    dst = fixed["surf_pts"].astype(np.float64)
    dnr = fixed["surf_nrm"].astype(np.float64)
    if len(src0) < 100 or len(dst) < 100:
        return Result(init.copy(), time.time() - t0, "anatomy", False,
                      {"error": f"surface too small ({len(src0)}, {len(dst)})"})

    tree = spatial.cKDTree(dst)
    current = init.copy()
    keep_n = max(int(cfg.icp_trim * len(src0)), 50)
    iters, rms, moved = 0, float("nan"), float("inf")

    for iters in range(1, cfg.icp_max_iter + 1):
        src = apply(current, src0)
        dist, idx = tree.query(src, k=1, workers=1)
        order = np.argsort(dist)[:keep_n]
        s, d, n = src[order], dst[idx[order]], dnr[idx[order]]
        rms = float(np.sqrt((dist[order] ** 2).mean()))
        step = _point_to_plane_step(s, d, n)
        current = step @ current
        moved = float(np.linalg.norm(apply(step, s) - s, axis=1).mean())
        if moved < cfg.icp_tol:
            break

    return Result(current, time.time() - t0, "anatomy", moved < cfg.icp_tol,
                  {"iterations": iters, "final_step_mm": round(moved, 6),
                   "trimmed_rms_mm": round(rms, 4),
                   "n_source": int(len(src0)), "n_target": int(len(dst))})


METHODS = {"intensity": register_intensity, "anatomy": register_anatomy}
