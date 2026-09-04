"""Enumerate the volume pairs the experiment runs on.

The rule is fixed in EXPERIMENT_PLAN.md before any registration was run:
same trial, both volumes segmented, volume numbers differing by exactly 1, and
at least 15% image overlap under the published reference transforms.

The overlap test uses the reference transforms, which is legitimate because it
describes the *acquisition* -- it is the same number for both methods and is
computed before either of them runs. Bone-mask overlap is recorded too, but is
deliberately not used to filter: excluding pairs where the two masks barely meet
would remove exactly the cases anatomical guidance finds hardest, and would tilt
the comparison towards Method B.

    python pairs.py
"""

from __future__ import annotations

import csv
import json

import numpy as np

from config import ARTIFACTS, DATA, DEFAULT
from data import (apply, mask_dialect, mask_path, read_mhd,
                  read_reference_transforms, relative, trial_dirs, trial_label,
                  volume_path)
from metrics import overlap_dice
from preprocess import load as load_prep

PAIRS_FILE = ARTIFACTS / "pairs.json"


def overlap_fraction(rel: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                     samples: int, seed: int) -> float:
    """Fraction of the moving volume's box that lands inside the fixed one.

    Both volumes share the same grid geometry in this dataset, so one box serves
    for both; `rel` places the moving volume in the fixed volume's frame.
    """
    rng = np.random.default_rng(seed)
    pts = lo + rng.random((samples, 3)) * (hi - lo)
    q = apply(rel, pts)
    return float(((q >= lo) & (q <= hi)).all(axis=1).mean())


def mask_dice_at_reference(label: str, a: int, b: int, rel: np.ndarray) -> float:
    """Dice of the two humerus masks placed by the reference transforms.

    Descriptive only -- it is never used to include or exclude a pair. It says
    how much bone the two volumes actually share, which is the covariate that
    matters when reading Method B's failures.
    """
    return overlap_dice(load_prep(label, b), load_prep(label, a), rel)


def build(cfg=DEFAULT) -> list[dict]:
    seg_rows = list(csv.DictReader(
        (DATA / "metadata" / "humerus_segmentations.csv").open(encoding="utf-8")))
    by_label: dict[str, list[int]] = {}
    for r in seg_rows:
        by_label.setdefault(r["trial_label"], []).append(int(r["volume_number"]))
    trials = {trial_label(t): t for t in trial_dirs()}

    out: list[dict] = []
    for label in sorted(by_label):
        trial = trials[label]
        numbers = sorted(by_label[label])
        ref = read_reference_transforms(str(trial))
        vol = read_mhd(volume_path(trial, numbers[0]))
        lo, hi = vol.offset, vol.offset + vol.extent

        for a in numbers:
            b = a + cfg.pair_distance
            if b not in numbers:
                continue
            rel = relative(ref[a], ref[b])          # moving = a, fixed = b
            frac = overlap_fraction(rel, lo, hi, cfg.overlap_samples, cfg.overlap_seed)
            dial = (mask_dialect(mask_path(trial, a)), mask_dialect(mask_path(trial, b)))

            reasons = []
            if frac < cfg.min_overlap:
                reasons.append(f"overlap {frac:.1%} < {cfg.min_overlap:.0%}")
            if "ImFusion" in dial:
                reasons.append("ImFusion-dialect mask")
            if np.allclose(ref[a], ref[b]):
                reasons.append("identical reference transforms")

            rec = dict(
                pair_id=f"{label}_V{a}-V{b}", trial=label,
                participant=label.split("_")[0], moving=a, fixed=b,
                overlap=round(frac, 4), mask_dialects=list(dial),
                eligible=not reasons, excluded_because=reasons,
                ref_translation_mm=round(float(np.linalg.norm(rel[:3, 3])), 3),
            )
            if rec["eligible"]:
                rec["bone_dice_at_reference"] = round(
                    mask_dice_at_reference(label, a, b, rel), 4)
            out.append(rec)
            print(f"  {rec['pair_id']:<24s} overlap {frac*100:5.1f}%  "
                  + (f"keep   bone dice {rec['bone_dice_at_reference']:.3f}"
                     if rec["eligible"] else "DROP   " + "; ".join(reasons)),
                  flush=True)
    return out


def load() -> list[dict]:
    return [p for p in json.loads(PAIRS_FILE.read_text()) if p["eligible"]]


def main() -> None:
    pairs = build()
    PAIRS_FILE.write_text(json.dumps(pairs, indent=2))
    kept = [p for p in pairs if p["eligible"]]
    print(f"\n{len(kept)} of {len(pairs)} adjacent pairs are eligible")
    counts: dict[str, int] = {}
    for p in pairs:
        for r in p["excluded_because"]:
            counts[r.split(" <")[0] if r.startswith("overlap") else r] = \
                counts.get(r.split(" <")[0] if r.startswith("overlap") else r, 0) + 1
    for reason, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    excluded by {reason}: {n}")
    lo = min(p["overlap"] for p in pairs)
    print(f"    (the >= {DEFAULT.min_overlap:.0%} overlap rule excluded none: the "
          f"smallest adjacent overlap is {lo:.1%})")
    for t in sorted({p['trial'] for p in pairs}):
        n = sum(1 for p in kept if p['trial'] == t)
        print(f"    {t}: {n}")
    print(f"written -> {PAIRS_FILE}")


if __name__ == "__main__":
    main()
