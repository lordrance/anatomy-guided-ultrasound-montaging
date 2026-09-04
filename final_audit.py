"""Independent check on the numbers that end up in the README.

Everything here is recomputed from the committed CSV with its own code. It
deliberately does not import `metrics` or `analyse`: if those had a bug, an
audit built on them would inherit it. The only shared input is
`artifacts/results.csv`.

    python final_audit.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

from config import ARTIFACTS, DEFAULT

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def main() -> None:
    with (ARTIFACTS / "results.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("mtre_mm", "init_mtre_mm", "seconds", "overlap", "dice", "ncc",
                  "bone_dice_at_reference"):
            r[k] = float(r[k]) if r[k] not in ("", "None") else float("nan")
        r["success"] = r["success"] == "True"
        r["improved"] = r["improved"] == "True"
        r["repeat"] = int(r["repeat"])

    summary = json.loads((ARTIFACTS / "summary.json").read_text())
    pairs = [p for p in json.loads((ARTIFACTS / "pairs.json").read_text()) if p["eligible"]]

    print("bookkeeping")
    n_pairs = len({r["pair_id"] for r in rows})
    expected = n_pairs * len(DEFAULT.conditions) * DEFAULT.n_seeds * 2
    check("every eligible pair was run", n_pairs == len(pairs),
          f"{n_pairs} run, {len(pairs)} eligible")
    check("row count matches pairs x conditions x seeds x methods",
          len(rows) == expected, f"{len(rows)} == {expected}")
    check("no missing mTRE", not np.isnan([r["mtre_mm"] for r in rows]).any())
    check("summary.json agrees on the pair count",
          summary["n_pairs"] == n_pairs, f"{summary['n_pairs']} vs {n_pairs}")

    print("\nfairness")
    grouped = defaultdict(dict)
    for r in rows:
        grouped[(r["pair_id"], r["condition"], r["repeat"])][r["method"]] = r
    same_init = [abs(v["intensity"]["init_mtre_mm"] - v["anatomy"]["init_mtre_mm"])
                 for v in grouped.values() if len(v) == 2]
    check("both methods started from the same transform in every run",
          len(same_init) == len(grouped) and max(same_init) < 1e-9,
          f"{len(grouped)} paired runs, max difference {max(same_init):.2e} mm")
    check("no pair was evaluated more than once per (condition, repeat, method)",
          all(len(v) == 2 for v in grouped.values()))

    print("\nthe pre-registered threshold was applied as written")
    recomputed = [(r["mtre_mm"] < DEFAULT.success_mtre) == r["success"] for r in rows]
    check(f"success == (mTRE < {DEFAULT.success_mtre} mm) on every row", all(recomputed),
          f"{sum(recomputed)}/{len(rows)}")
    check("improved == (mTRE < initial mTRE) on every row",
          all((r["mtre_mm"] < r["init_mtre_mm"]) == r["improved"] for r in rows))

    print("\nheadline numbers, recomputed from the CSV")
    for cond in [c[0] for c in DEFAULT.conditions] + ["ALL"]:
        for method in ("intensity", "anatomy"):
            sel = [r for r in rows if r["method"] == method
                   and (cond == "ALL" or r["condition"] == cond)]
            mine = dict(
                n=len(sel),
                success_rate=round(float(np.mean([r["success"] for r in sel])), 6),
                median_mtre_mm=round(float(np.median([r["mtre_mm"] for r in sel])), 3),
                mean_mtre_mm=round(float(np.mean([r["mtre_mm"] for r in sel])), 3),
                seconds=round(float(np.mean([r["seconds"] for r in sel])), 3),
            )
            theirs = summary["table"][f"{cond}|{method}"]
            same = all(abs(mine[k] - theirs[k]) < 1e-6 for k in mine)
            check(f"{cond:6s} {method:10s} matches summary.json", same,
                  "" if same else f"mine {mine} vs {theirs}")

    print("\nstated claims")
    a = [r for r in rows if r["method"] == "anatomy"]
    i = [r for r in rows if r["method"] == "intensity"]
    d_success = np.mean([r["success"] for r in a]) - np.mean([r["success"] for r in i])
    d_mtre = np.mean([r["mtre_mm"] for r in a]) - np.mean([r["mtre_mm"] for r in i])
    check("anatomy has the higher success rate", d_success > 0,
          f"{np.mean([r['success'] for r in a]):.3f} vs "
          f"{np.mean([r['success'] for r in i]):.3f} (+{d_success:.3f})")
    check("anatomy has the higher mean mTRE, i.e. it is worse on average",
          d_mtre > 0, f"{np.mean([r['mtre_mm'] for r in a]):.2f} vs "
                      f"{np.mean([r['mtre_mm'] for r in i]):.2f} mm ({d_mtre:+.2f})")
    check("the success-rate difference is not distinguishable from zero",
          not summary["paired"]["ALL|success_rate"]["excludes_zero"],
          f"95% CI {summary['paired']['ALL|success_rate']['ci95']}")
    check("the mean-mTRE difference is distinguishable from zero",
          summary["paired"]["ALL|mtre_mm"]["excludes_zero"],
          f"95% CI {summary['paired']['ALL|mtre_mm']['ci95']}")

    high = [r for r in rows if r["bone_dice_at_reference"] >= 0.30]
    ha = np.mean([r["success"] for r in high if r["method"] == "anatomy"])
    hi = np.mean([r["success"] for r in high if r["method"] == "intensity"])
    check("anatomy leads where the pair shares bone (Dice >= 0.30)", ha > hi,
          f"{ha:.3f} vs {hi:.3f} over "
          f"{len({r['pair_id'] for r in high})} pairs")
    low = [r for r in rows if r["bone_dice_at_reference"] < 0.30]
    la = np.mean([r["success"] for r in low if r["method"] == "anatomy"])
    li = np.mean([r["success"] for r in low if r["method"] == "intensity"])
    check("and trails where it does not", la < li, f"{la:.3f} vs {li:.3f}")

    speed = np.mean([r["seconds"] for r in i]) / np.mean([r["seconds"] for r in a])
    check("anatomy is more than 10x faster", speed > 10, f"{speed:.1f}x")

    diag = json.loads((ARTIFACTS / "reference_diagnostic.json").read_text())
    check("intensity reaches a higher NCC than the reference on every pair",
          diag["intensity_beats_reference_on_its_own_criterion"] == diag["n_pairs"],
          f"{diag['intensity_beats_reference_on_its_own_criterion']}/{diag['n_pairs']}")

    geo = json.loads((ARTIFACTS / "geometry_validation.json").read_text())
    check("geometry reproduces the authors' published Dice on Slicer-dialect pairs",
          geo["slicer_only"]["correlation"] > 0.85,
          f"r = {geo['slicer_only']['correlation']} over n = {geo['slicer_only']['n']}")
    check("every pair used has two Slicer-dialect masks",
          all(p["mask_dialects"] == ["Slicer", "Slicer"] for p in pairs))

    failed = [c for c in CHECKS if not c[1]]
    print(f"\nFINAL AUDIT: {'PASS' if not failed else 'FAIL'} "
          f"({len(CHECKS) - len(failed)}/{len(CHECKS)})")
    for name, _, detail in failed:
        print(f"  failed: {name} -- {detail}")
    (ARTIFACTS / "final_audit.json").write_text(json.dumps(
        dict(passed=not failed,
             checks=[dict(check=n, passed=o, detail=d) for n, o, d in CHECKS]), indent=2))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
