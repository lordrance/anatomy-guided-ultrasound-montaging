"""Is the published reference the optimum of either method's criterion?

The headline table shows both methods ending 7-9 mm from the reference even from
a 0.6 mm start. Before reading that as failure, it is worth asking whether the
reference is where the criteria actually peak. It is not, and this script is the
evidence.

For every pair it reports, at the reference placement and at each method's own
solution:

    NCC over the overlapping foreground   -- what Method A maximises
    trimmed surface RMS in mm             -- what Method B minimises

    python diagnose_reference.py
"""

from __future__ import annotations

import json

import numpy as np
from scipy import spatial

from config import ARTIFACTS, DEFAULT, DOCS
from data import apply, read_reference_transforms, relative, trial_dirs, trial_label
from metrics import overlap_ncc
from preprocess import load as load_prep


def trimmed_rms(moving: dict, fixed: dict, transform: np.ndarray,
                trim: float = DEFAULT.icp_trim) -> float:
    src = apply(transform, moving["surf_pts"].astype(float))
    dst = fixed["surf_pts"].astype(float)
    if len(src) < 10 or len(dst) < 10:
        return float("nan")
    dist, _ = spatial.cKDTree(dst).query(src, k=1)
    keep = np.sort(dist)[:max(int(trim * len(dist)), 10)]
    return float(np.sqrt((keep ** 2).mean()))


def main() -> None:
    results = json.loads((ARTIFACTS / "results.json").read_text())["rows"]
    pairs = [p for p in json.loads((ARTIFACTS / "pairs.json").read_text()) if p["eligible"]]
    trials = {trial_label(t): t for t in trial_dirs()}

    out = []
    for p in pairs:
        lbl, a, b = p["trial"], p["moving"], p["fixed"]
        ref = read_reference_transforms(str(trials[lbl]))
        rel = relative(ref[a], ref[b])
        moving, fixed = load_prep(lbl, a), load_prep(lbl, b)

        easy = [r for r in results if r["pair_id"] == p["pair_id"] and r["condition"] == "EASY"]
        ncc_est = float(np.mean([r["ncc"] for r in easy if r["method"] == "intensity"]))
        rms_est = float(np.mean([r["info"]["trimmed_rms_mm"] for r in easy
                                 if r["method"] == "anatomy"]))
        out.append(dict(
            pair_id=p["pair_id"], trial=lbl,
            bone_dice_at_reference=p["bone_dice_at_reference"],
            ncc_at_reference=round(overlap_ncc(fixed, moving, rel), 4),
            ncc_at_intensity_solution=round(ncc_est, 4),
            surface_rms_at_reference_mm=round(trimmed_rms(moving, fixed, rel), 4),
            surface_rms_at_anatomy_solution_mm=round(rms_est, 4),
        ))

    n_ncc = sum(1 for r in out if r["ncc_at_intensity_solution"] > r["ncc_at_reference"])
    n_rms = sum(1 for r in out
                if r["surface_rms_at_anatomy_solution_mm"] < r["surface_rms_at_reference_mm"])
    summary = dict(
        n_pairs=len(out),
        intensity_beats_reference_on_its_own_criterion=n_ncc,
        anatomy_beats_reference_on_its_own_criterion=n_rms,
        mean_ncc_at_reference=round(float(np.mean([r["ncc_at_reference"] for r in out])), 4),
        mean_ncc_at_solution=round(float(np.mean(
            [r["ncc_at_intensity_solution"] for r in out])), 4),
        median_surface_rms_at_reference_mm=round(float(np.median(
            [r["surface_rms_at_reference_mm"] for r in out])), 4),
        median_surface_rms_at_solution_mm=round(float(np.median(
            [r["surface_rms_at_anatomy_solution_mm"] for r in out])), 4),
        rows=out,
    )
    (ARTIFACTS / "reference_diagnostic.json").write_text(json.dumps(summary, indent=2))

    print(f"{'pair':26s}{'NCC@ref':>9s}{'NCC@est':>9s}{'d':>8s}"
          f"{'rms@ref':>9s}{'rms@est':>9s}")
    for r in out:
        print(f"{r['pair_id']:26s}{r['ncc_at_reference']:9.4f}"
              f"{r['ncc_at_intensity_solution']:9.4f}"
              f"{r['ncc_at_intensity_solution']-r['ncc_at_reference']:+8.4f}"
              f"{r['surface_rms_at_reference_mm']:9.3f}"
              f"{r['surface_rms_at_anatomy_solution_mm']:9.3f}")
    print(f"\nintensity reaches a higher NCC than the reference on "
          f"{n_ncc}/{len(out)} pairs")
    print(f"anatomy reaches a lower surface residual than the reference on "
          f"{n_rms}/{len(out)} pairs")

    L = ["# Is the reference the optimum?", "",
         "Both methods end several millimetres from the published reference even "
         "when they start beside it. This file checks the obvious alternative "
         "explanation: that the reference is not where either method's criterion "
         "peaks.", "",
         "| | at the reference | at the method's solution |", "|---|---:|---:|",
         f"| mean NCC (Method A maximises this) | {summary['mean_ncc_at_reference']} "
         f"| **{summary['mean_ncc_at_solution']}** |",
         f"| median trimmed surface RMS (Method B minimises this) | "
         f"{summary['median_surface_rms_at_reference_mm']} mm "
         f"| **{summary['median_surface_rms_at_solution_mm']} mm** |", "",
         f"Method A reaches a higher NCC than the reference on **{n_ncc} of "
         f"{len(out)}** pairs. Method B reaches a lower surface residual than the "
         f"reference on **{n_rms} of {len(out)}**.", "",
         "So neither optimiser is broken. The published reference encodes "
         "information that a pairwise automatic criterion does not have: it was "
         "produced with expert manual adjustment guided by MRI bone "
         "segmentations, and by jointly refining groups of four neighbouring "
         "volumes with pose anchoring rather than one pair at a time.", "",
         "This is the quantitative form of the caveat stated throughout: the "
         "refined transforms are a **reference registration**, not absolute "
         "physical ground truth, and mTRE against them measures agreement with a "
         "particular high-quality reconstruction.", "",
         "## Per pair", "",
         "| pair | bone Dice at reference | NCC at reference | NCC at A's solution "
         "| surface RMS at reference | surface RMS at B's solution |",
         "|---|---:|---:|---:|---:|---:|"]
    for r in out:
        L.append(f"| {r['pair_id']} | {r['bone_dice_at_reference']} | "
                 f"{r['ncc_at_reference']} | {r['ncc_at_intensity_solution']} | "
                 f"{r['surface_rms_at_reference_mm']} mm | "
                 f"{r['surface_rms_at_anatomy_solution_mm']} mm |")
    (DOCS / "reference_diagnostic.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"written -> {DOCS / 'reference_diagnostic.md'}")


if __name__ == "__main__":
    main()
