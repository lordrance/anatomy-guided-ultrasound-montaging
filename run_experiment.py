"""Run the perturbation-recovery experiment.

For every eligible pair, every initialisation condition and every repeat, both
methods are handed the *same* corrupted starting transform and asked to recover
the published reference relative transform.

    python run_experiment.py                 # everything
    python run_experiment.py --smoke 4       # four pairs, for a quick check
    python run_experiment.py --workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from config import ARTIFACTS, DEFAULT
from data import (read_reference_transforms, relative, rotation_angle, trial_dirs,
                  trial_label)
from metrics import mtre, overlap_dice, overlap_ncc
from pairs import load as load_pairs
from perturb import random_rigid, stable_seed
from preprocess import load as load_prep
from register import METHODS

RESULTS_JSON = ARTIFACTS / "results.json"
RESULTS_CSV = ARTIFACTS / "results.csv"


def moving_corners(prep: dict) -> np.ndarray:
    """The 8 corners of the moving volume's physical box, from its own header."""
    lo = prep["offset"]
    hi = lo + prep["src_spacing"] * prep["src_dim"]
    return np.array([[lo[0] if a else hi[0], lo[1] if b else hi[1], lo[2] if c else hi[2]]
                     for a in (1, 0) for b in (1, 0) for c in (1, 0)])


def run_pair(pair: dict, cfg=DEFAULT) -> list[dict]:
    trial = {trial_label(t): t for t in trial_dirs()}[pair["trial"]]
    ref = read_reference_transforms(str(trial))
    reference = relative(ref[pair["moving"]], ref[pair["fixed"]])

    moving = load_prep(pair["trial"], pair["moving"])
    fixed = load_prep(pair["trial"], pair["fixed"])
    corners = moving_corners(moving)
    centre = moving["offset"] + moving["src_spacing"] * moving["src_dim"] / 2
    centre_in_fixed = reference[:3, :3] @ centre + reference[:3, 3]

    rows: list[dict] = []
    for label, t_max, r_max in cfg.conditions:
        for rep in range(cfg.n_seeds):
            seed = stable_seed(pair["pair_id"], label, rep)
            noise = random_rigid(t_max, r_max, centre_in_fixed, seed)
            init = noise @ reference
            init_mtre = mtre(init, reference, corners)

            for name, fn in METHODS.items():
                res = fn(fixed, moving, init, cfg, seed=seed)
                err = mtre(res.transform, reference, corners)
                delta = np.linalg.inv(reference) @ res.transform
                rows.append(dict(
                    pair_id=pair["pair_id"], trial=pair["trial"],
                    participant=pair["participant"],
                    moving=pair["moving"], fixed=pair["fixed"],
                    overlap=pair["overlap"],
                    bone_dice_at_reference=pair.get("bone_dice_at_reference"),
                    condition=label, repeat=rep, seed=seed, method=name,
                    init_mtre_mm=round(init_mtre, 4),
                    mtre_mm=round(err, 4),
                    success=bool(err < cfg.success_mtre),
                    improved=bool(err < init_mtre),
                    residual_translation_mm=round(float(np.linalg.norm(delta[:3, 3])), 4),
                    residual_rotation_deg=round(rotation_angle(delta), 4),
                    dice=round(overlap_dice(fixed, moving, res.transform), 4),
                    ncc=round(overlap_ncc(fixed, moving, res.transform), 4),
                    seconds=round(res.seconds, 3),
                    converged=res.converged, info=res.info,
                    # kept so figures and audits can replay a case without
                    # re-running the optimiser and risking a different answer
                    transform=[round(v, 8) for v in res.transform.ravel().tolist()],
                    init_transform=[round(v, 8) for v in init.ravel().tolist()],
                    reference_transform=[round(v, 8) for v in reference.ravel().tolist()],
                ))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="run only the first N pairs")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    pairs = load_pairs()
    if args.smoke:
        pairs = pairs[:args.smoke]
    cfg = DEFAULT
    per_pair = len(cfg.conditions) * cfg.n_seeds * len(METHODS)
    print(f"{len(pairs)} pairs x {len(cfg.conditions)} conditions x {cfg.n_seeds} "
          f"seeds x {len(METHODS)} methods = {len(pairs)*per_pair} registrations")
    print(f"success threshold: mTRE < {cfg.success_mtre} mm (pre-registered)")

    t0 = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, out in enumerate(pool.map(run_pair, pairs), 1):
            rows.extend(out)
            done = [r for r in out if r["method"] == "anatomy"]
            print(f"  [{i:2d}/{len(pairs)}] {out[0]['pair_id']:<24s} "
                  f"anatomy success {sum(r['success'] for r in done)}/{len(done)}  "
                  f"({time.time()-t0:.0f}s elapsed)", flush=True)

    elapsed = time.time() - t0
    RESULTS_JSON.write_text(json.dumps(dict(
        config={k: (list(v) if isinstance(v, tuple) else v)
                for k, v in vars(cfg).items()},
        n_pairs=len(pairs), n_registrations=len(rows),
        wall_clock_seconds=round(elapsed, 1),
        platform=f"{platform.platform()} python {platform.python_version()}",
        rows=rows), indent=2))

    drop = {"info", "transform", "init_transform", "reference_transform"}
    flat = [{k: v for k, v in r.items() if k not in drop} for r in rows]
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)

    print(f"\n{len(rows)} registrations in {elapsed/60:.1f} min "
          f"-> {RESULTS_JSON.name}, {RESULTS_CSV.name}")


if __name__ == "__main__":
    main()
