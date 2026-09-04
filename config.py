"""Single source of truth for paths and pre-registered experiment constants.

Nothing else in the project is allowed to hard-code a threshold, a seed or a
spacing. If a number matters to the result, it lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
DATA = CACHE / "dataset"          # extracted Zenodo members, original layout
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "figures"
DOCS = ROOT / "docs"

for _d in (CACHE, DATA, ARTIFACTS, FIGURES, DOCS):
    _d.mkdir(parents=True, exist_ok=True)

# --- data source ------------------------------------------------------------
ZENODO_RECORD = "20283247"
ZENODO_DOI = "10.5281/zenodo.20283247"
ZIP_NAME = "shoulder_3d_us_mosaicking_dataset.zip"
ZIP_URL = (
    f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{ZIP_NAME}/content"
)
ZIP_MD5 = "98fa88573b63b3b7ed4c78fe00ac354f"
ZIP_SIZE = 8_503_198_147
ARCHIVE_PREFIX = "De-identified US Trials/"

# --- pre-registered experiment constants (see EXPERIMENT_PLAN.md) -----------


@dataclass(frozen=True)
class ExperimentConfig:
    # pair selection
    pair_distance: int = 1               # adjacent acquisitions only
    min_overlap: float = 0.15            # >= 15% image overlap at the reference
    overlap_samples: int = 60_000        # Monte-Carlo points for the overlap test
    overlap_seed: int = 0

    # preprocessing
    coarse_spacing: float = 1.0          # mm, isotropic
    fine_spacing: float = 0.5            # mm, isotropic

    # perturbation levels: (label, max |t| mm, max angle deg)
    conditions: tuple[tuple[str, float, float], ...] = (
        ("EASY", 1.0, 1.0),
        ("MEDIUM", 5.0, 5.0),
        ("HARD", 10.0, 10.0),
    )
    n_seeds: int = 3

    # method A -- intensity NCC
    ncc_samples: int = 30_000            # foreground points sampled per level
    ncc_maxiter: int = 600               # Nelder-Mead iterations per level
    ncc_min_overlap: float = 0.05        # reject transforms that shrink overlap

    # method B -- anatomy-guided ICP
    icp_points: int = 5_000              # surface points after decimation
    icp_max_iter: int = 60
    icp_trim: float = 0.7                # keep the closest 70% of correspondences
    icp_tol: float = 1e-4                # mm, convergence on mean point motion

    # evaluation
    success_mtre: float = 5.0            # mm -- FIXED BEFORE ANY RESULT WAS SEEN


DEFAULT = ExperimentConfig()

# Volume geometry, verified against the MHD headers by `audit.py`.
EXPECTED_DIM = (512, 403, 256)
EXPECTED_SPACING = (0.11037, 0.10695, 0.23967)
