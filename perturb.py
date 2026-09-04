"""Deterministic rigid perturbations of the reference transform.

The experiment is a recovery task: we take the published reference relative
transform, break it by a known amount, and ask each method to find its way back.
That makes the target exact by construction and removes any dependence on the
robot poses, whose translation convention could not be resolved.

Seeds are derived from a stable hash of the identifiers, not from Python's
`hash()`, so the same pair, condition and repeat give the same perturbation on
any machine and in any process.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np


def stable_seed(*parts: object) -> int:
    key = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") % (2**31)


def random_rigid(max_translation: float, max_rotation: float,
                 centre: np.ndarray, seed: int) -> np.ndarray:
    """A rigid transform with |t| <= max_translation mm and angle <= max_rotation deg.

    Direction and axis are uniform on the sphere; magnitudes are uniform in
    [0, max]. Rotation is about `centre` so that a small angle does not smuggle
    in a large displacement.
    """
    rng = np.random.default_rng(seed)

    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = math.radians(rng.uniform(0.0, max_rotation))
    kx = np.array([[0, -axis[2], axis[1]],
                   [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])
    r = np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * (kx @ kx)

    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    t = direction * rng.uniform(0.0, max_translation)

    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = centre - r @ centre + t
    return m
