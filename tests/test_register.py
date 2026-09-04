"""Both registration methods, on synthetic data with a known answer.

These are the tests that would catch a sign error or a frame mix-up: a phantom
is built, displaced by a transform we chose, and each method has to recover it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DEFAULT
from data import apply, rigid_from_vec
from metrics import mtre, overlap_dice, overlap_ncc
from preprocess import bone_surface, resample_isotropic
from register import register_anatomy, register_intensity

SPACING = np.full(3, 1.0)


def phantom(seed: int = 0, n: int = 44) -> np.ndarray:
    """A blobby volume with enough structure to register, plus speckle."""
    rng = np.random.default_rng(seed)
    g = np.zeros((n, n, n), np.float32)
    for _ in range(12):
        c = rng.integers(8, n - 8, 3)
        r = rng.integers(3, 7)
        z, y, x = np.ogrid[:n, :n, :n]
        g[((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2) < r * r] = rng.uniform(.4, 1)
    return np.clip(g + rng.normal(0, 0.02, g.shape), 0, 1).astype(np.float32)


def make_volume(img: np.ndarray, offset=np.zeros(3), mask=None) -> dict:
    return {
        "img_coarse": img, "img_fine": img,
        "spacing_coarse": SPACING, "spacing_fine": SPACING,
        "offset": offset, "src_spacing": SPACING,
        "src_dim": np.array(img.shape),
        "mask_fine": np.zeros(img.shape, bool) if mask is None else mask,
        "surf_pts": np.zeros((0, 3), np.float32), "surf_nrm": np.zeros((0, 3), np.float32),
    }


def test_intensity_stays_put_when_it_is_already_right():
    img = phantom()
    v = make_volume(img)
    res = register_intensity(v, v, np.eye(4), DEFAULT, seed=0)
    corners = np.array([[0, 0, 0], [44, 44, 44]], float)
    assert mtre(res.transform, np.eye(4), corners) < 1.0


def test_intensity_recovers_a_known_shift():
    """Same voxels, different physical origin, so the answer is a known shift.

    Registering a volume against itself would make identity correct and prove
    nothing; here the moving volume carries the same array at a different
    offset, so the transform that aligns the content is exactly `d`.
    """
    img = phantom(1)
    d = np.array([3.0, -2.0, 1.0])
    fixed = make_volume(img, offset=np.zeros(3))
    moving = make_volume(img, offset=-d)
    truth = rigid_from_vec(list(d) + [0, 0, 0], np.zeros(3))

    start = rigid_from_vec([-2.0, 1.5, -1.0, 0, 0, 0], np.zeros(3)) @ truth
    corners = np.array([[0, 0, 0], [44, 44, 44]], float) - d
    assert mtre(start, truth, corners) > 2.0          # the start really is wrong
    res = register_intensity(fixed, moving, start, DEFAULT, seed=0)
    assert mtre(res.transform, truth, corners) < 1.0


def sphere_surface(centre, radius=10.0, n=40):
    """A shell mask and the surface extracted from it, on a 1 mm grid."""
    g = np.zeros((n, n, n), np.float32)
    z, y, x = np.ogrid[:n, :n, :n]
    d = np.sqrt((z - centre[0]) ** 2 + (y - centre[1]) ** 2 + (x - centre[2]) ** 2)
    g[d < radius] = 1.0
    pts, nrm = bone_surface(g > 0.5, SPACING, np.zeros(3), 4000, seed=0)
    return g > 0.5, pts, nrm


def test_anatomy_recovers_a_known_rigid_transform():
    mask, pts, nrm = sphere_surface((20, 20, 20))
    # a sphere is rotationally degenerate, so add a bump to pin the rotation
    n = 40
    z, y, x = np.ogrid[:n, :n, :n]
    mask = mask | (np.sqrt((z - 28) ** 2 + (y - 20) ** 2 + (x - 20) ** 2) < 6)
    pts, nrm = bone_surface(mask, SPACING, np.zeros(3), 4000, seed=0)

    truth = rigid_from_vec([2.0, -1.5, 1.0, 4.0, -3.0, 2.0], np.full(3, 20.0))
    fixed = make_volume(np.zeros((n, n, n), np.float32))
    fixed["surf_pts"], fixed["surf_nrm"] = pts.astype(np.float32), nrm.astype(np.float32)
    moving = make_volume(np.zeros((n, n, n), np.float32))
    src = apply(np.linalg.inv(truth), pts)
    moving["surf_pts"] = src.astype(np.float32)
    moving["surf_nrm"] = nrm.astype(np.float32)

    res = register_anatomy(fixed, moving, np.eye(4), DEFAULT, seed=0)
    corners = np.array([[0, 0, 0], [40, 40, 40]], float)
    assert mtre(res.transform, truth, corners) < 1.0


def test_anatomy_reports_failure_when_there_is_no_surface():
    v = make_volume(np.zeros((10, 10, 10), np.float32))
    res = register_anatomy(v, v, np.eye(4), DEFAULT, seed=0)
    assert not res.converged
    assert "surface too small" in res.info["error"]


def test_both_methods_keep_the_transform_rigid():
    img = phantom(2)
    v = make_volume(img)
    for fn in (register_intensity, register_anatomy):
        res = fn(v, v, rigid_from_vec([1, 1, 1, 2, 2, 2], np.full(3, 22.0)),
                 DEFAULT, seed=0)
        r = res.transform[:3, :3]
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-8), fn.__name__
        assert abs(np.linalg.det(r) - 1) < 1e-8, fn.__name__


def test_resample_preserves_physical_position():
    """A blob's centre of mass must not move when the grid changes."""
    n = 40
    img = np.zeros((n, n, n), np.float32)
    img[18:23, 18:23, 18:23] = 1.0
    src_spacing = np.array([0.5, 0.25, 1.0])
    offset = np.array([-3.0, 7.0, 0.5])

    def com(a, spacing):
        idx = np.array(np.nonzero(a > 0.25))
        w = a[a > 0.25]
        return offset + (idx * spacing[:, None] * w).sum(1) / w.sum()

    out, sp = resample_isotropic(img, src_spacing, offset, 0.5)
    assert np.allclose(com(img, src_spacing), com(out, sp), atol=0.4)


def test_metrics_are_perfect_on_identical_input():
    img = phantom(3)
    mask = img > 0.5
    v = make_volume(img, mask=mask)
    assert overlap_dice(v, v, np.eye(4)) == pytest.approx(1.0, abs=1e-9)
    assert overlap_ncc(v, v, np.eye(4)) == pytest.approx(1.0, abs=1e-6)
