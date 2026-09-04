"""Geometry and transform maths. No dataset needed."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import (affine_params_to_matrix, apply, euler_to_mat, relative,
                  rigid_from_vec, rotation_angle)
from metrics import mtre
from perturb import random_rigid, stable_seed


def is_rigid(m: np.ndarray) -> bool:
    r = m[:3, :3]
    return (np.allclose(r @ r.T, np.eye(3), atol=1e-10)
            and abs(np.linalg.det(r) - 1) < 1e-10
            and np.allclose(m[3], [0, 0, 0, 1]))


def test_euler_to_mat_is_a_rotation():
    for angles in [(0, 0, 0), (10, -20, 30), (90, 0, 0), (0, 0, 180)]:
        r = euler_to_mat(*angles)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(r) - 1) < 1e-12


def test_euler_zero_is_identity():
    assert np.allclose(euler_to_mat(0, 0, 0), np.eye(3))


def test_affine_params_rejects_scale_and_shear():
    ok = [1, 2, 3, 10, 20, 30] + [1, 1, 1] + [0, 0, 0]
    assert is_rigid(affine_params_to_matrix(ok))
    with pytest.raises(ValueError, match="unit-scale"):
        affine_params_to_matrix([0] * 6 + [1, 1, 1.2] + [0, 0, 0])
    with pytest.raises(ValueError, match="sheared"):
        affine_params_to_matrix([0] * 6 + [1, 1, 1] + [0, 0.5, 0])
    with pytest.raises(ValueError, match="12 parameters"):
        affine_params_to_matrix([0] * 11)


def test_rigid_from_vec_fixes_its_centre():
    centre = np.array([3.0, -4.0, 5.0])
    m = rigid_from_vec([0, 0, 0, 12, -7, 3], centre)
    assert is_rigid(m)
    assert np.allclose(apply(m, centre[None])[0], centre, atol=1e-9)


def test_rigid_from_vec_translation_is_exact():
    m = rigid_from_vec([1.5, -2.5, 0.25, 0, 0, 0], np.zeros(3))
    assert np.allclose(m[:3, 3], [1.5, -2.5, 0.25])
    assert np.allclose(m[:3, :3], np.eye(3))


def test_rigid_from_vec_angle_matches_the_request():
    for deg in (0.5, 5.0, 30.0):
        m = rigid_from_vec([0, 0, 0, deg, 0, 0], np.zeros(3))
        assert rotation_angle(m) == pytest.approx(deg, abs=1e-9)


def test_relative_is_frame_invariant():
    rng = np.random.default_rng(0)
    a = rigid_from_vec(rng.normal(size=6) * 5, np.zeros(3))
    b = rigid_from_vec(rng.normal(size=6) * 5, np.zeros(3))
    g = rigid_from_vec(rng.normal(size=6) * 50, np.zeros(3))
    assert np.allclose(relative(a, b), relative(g @ a, g @ b), atol=1e-9)


def test_mtre_is_zero_for_the_same_transform():
    rng = np.random.default_rng(1)
    t = rigid_from_vec(rng.normal(size=6) * 3, np.zeros(3))
    corners = rng.normal(size=(8, 3)) * 20
    assert mtre(t, t, corners) == pytest.approx(0.0, abs=1e-9)


def test_mtre_equals_the_translation_for_a_pure_shift():
    corners = np.random.default_rng(2).normal(size=(8, 3)) * 20
    shift = rigid_from_vec([3, 4, 0, 0, 0, 0], np.zeros(3))
    assert mtre(shift, np.eye(4), corners) == pytest.approx(5.0, abs=1e-9)


def test_perturbation_respects_its_bounds():
    centre = np.array([1.0, 2.0, 3.0])
    for t_max, r_max in ((1, 1), (5, 5), (10, 10)):
        for s in range(40):
            m = random_rigid(t_max, r_max, centre, seed=s)
            assert is_rigid(m)
            assert rotation_angle(m) <= r_max + 1e-9
            # translation is defined at the centre of rotation
            assert np.linalg.norm(apply(m, centre[None])[0] - centre) <= t_max + 1e-9


def test_perturbation_is_deterministic_and_seed_dependent():
    c = np.zeros(3)
    assert np.allclose(random_rigid(5, 5, c, 7), random_rigid(5, 5, c, 7))
    assert not np.allclose(random_rigid(5, 5, c, 7), random_rigid(5, 5, c, 8))


def test_stable_seed_does_not_depend_on_the_process():
    # a literal, so a change to the hashing would fail the test rather than
    # silently reshuffle every perturbation in the experiment
    assert stable_seed("PA001_trial_A_V9-V10", "HARD", 0) == 512119223
    assert stable_seed("a", "b") != stable_seed("b", "a")
