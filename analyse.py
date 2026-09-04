"""Turn `artifacts/results.json` into the summary table and `docs/results.md`.

Nothing here re-runs a registration; it only reads what the experiment wrote.
Confidence intervals resample whole trials, because registrations from one trial
share volumes and an operator and are not independent.

    python analyse.py
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from config import ARTIFACTS, DOCS
from metrics import bootstrap_diff

RESULTS = ARTIFACTS / "results.json"
SUMMARY = ARTIFACTS / "summary.json"
METHOD_LABEL = {"intensity": "A intensity (NCC)", "anatomy": "B anatomy (ICP)"}


def cell(rows: list[dict]) -> dict:
    m = np.array([r["mtre_mm"] for r in rows])
    i = np.array([r["init_mtre_mm"] for r in rows])
    return dict(
        n=len(rows),
        init_mtre_mm=round(float(i.mean()), 3),
        median_mtre_mm=round(float(np.median(m)), 3),
        mean_mtre_mm=round(float(m.mean()), 3),
        # Did the optimiser move towards the reference or away from it? Positive
        # means it ended further away than it started. Reported because the
        # absolute error is dominated by how far the reference sits from either
        # method's own optimum, which is the same for both methods.
        delta_mtre_mm=round(float((m - i).mean()), 3),
        median_delta_mtre_mm=round(float(np.median(m - i)), 3),
        # kept at six decimals: rounding to four first turns 0.31349 into
        # 0.3135, which then formats as 31.4% and disagrees with the
        # independently recomputed 31.3% in final_audit.py
        success_rate=round(float(np.mean([r["success"] for r in rows])), 6),
        improved_rate=round(float(np.mean([r["improved"] for r in rows])), 6),
        seconds=round(float(np.mean([r["seconds"] for r in rows])), 3),
    )


def main() -> None:
    data = json.loads(RESULTS.read_text())
    rows = data["rows"]
    cfg = data["config"]
    conditions = [c[0] for c in cfg["conditions"]]
    methods = ["intensity", "anatomy"]

    by = defaultdict(list)
    for r in rows:
        by[(r["condition"], r["method"])].append(r)
        by[("ALL", r["method"])].append(r)

    table = {f"{c}|{m}": cell(by[(c, m)]) for c in conditions + ["ALL"] for m in methods}

    # paired comparison, resampling trials
    paired = {}
    for c in conditions + ["ALL"]:
        a = sorted(by[(c, "intensity")], key=lambda r: (r["pair_id"], r["condition"], r["repeat"]))
        b = sorted(by[(c, "anatomy")], key=lambda r: (r["pair_id"], r["condition"], r["repeat"]))
        assert [x["pair_id"] for x in a] == [x["pair_id"] for x in b]
        groups = np.array([x["trial"] for x in a])
        for name, key in (("success_rate", "success"), ("mtre_mm", "mtre_mm")):
            va = np.array([float(x[key]) for x in a])
            vb = np.array([float(x[key]) for x in b])
            d, lo, hi = bootstrap_diff(va, vb, groups)
            paired[f"{c}|{name}"] = dict(anatomy_minus_intensity=round(d, 4),
                                         ci95=[round(lo, 4), round(hi, 4)],
                                         excludes_zero=bool(lo > 0 or hi < 0))

    # stratify by how much bone the two volumes actually share
    edges = [(0.0, 0.001, "0 (no shared bone)"), (0.001, 0.10, "0-0.10"),
             (0.10, 0.30, "0.10-0.30"), (0.30, 1.01, ">= 0.30")]
    strata = {}
    for lo_e, hi_e, label in edges:
        sel = [r for r in rows if lo_e <= (r["bone_dice_at_reference"] or 0.0) < hi_e]
        if not sel:
            continue
        strata[label] = {
            "n_pairs": len({r["pair_id"] for r in sel}),
            **{m: dict(success_rate=round(float(np.mean(
                [r["success"] for r in sel if r["method"] == m])), 4),
                median_mtre_mm=round(float(np.median(
                    [r["mtre_mm"] for r in sel if r["method"] == m])), 3))
               for m in methods},
        }

    summary = dict(
        n_pairs=data["n_pairs"], n_registrations=data["n_registrations"],
        wall_clock_minutes=round(data["wall_clock_seconds"] / 60, 2),
        platform=data["platform"], success_threshold_mm=cfg["success_mtre"],
        table=table, paired=paired, by_bone_overlap=strata,
        trials=sorted({r["trial"] for r in rows}),
    )
    SUMMARY.write_text(json.dumps(summary, indent=2))
    write_md(summary, conditions, methods)

    print(f"{'condition':9s}{'method':20s}{'n':>4s}{'init':>7s}{'med mTRE':>10s}"
          f"{'success':>9s}{'sec':>7s}")
    for c in conditions + ["ALL"]:
        for m in methods:
            v = table[f"{c}|{m}"]
            print(f"{c:9s}{METHOD_LABEL[m]:20s}{v['n']:4d}{v['init_mtre_mm']:7.2f}"
                  f"{v['median_mtre_mm']:10.2f}{v['success_rate']:9.1%}{v['seconds']:7.2f}")
    print(f"\nwritten -> {SUMMARY.name}, docs/results.md")


def write_md(s: dict, conditions: list[str], methods: list[str]) -> None:
    t = s["table"]
    L = [
        "# Results", "",
        f"{s['n_registrations']} registrations over {s['n_pairs']} volume pairs "
        f"from {len(s['trials'])} trials, {s['wall_clock_minutes']} minutes of "
        f"wall clock on CPU.", "",
        f"Success is `mTRE < {s['success_threshold_mm']} mm` against the published "
        "refined reference registration. That threshold was fixed in "
        "`EXPERIMENT_PLAN.md` before any comparative result was looked at.", "",
        "## Main table", "",
        "| initialisation | method | n | mean initial mTRE | median mTRE | mean mTRE "
        "| mean change | success | ended closer than it started | seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in conditions + ["ALL"]:
        for m in methods:
            v = t[f"{c}|{m}"]
            L.append(f"| {c} | {METHOD_LABEL[m]} | {v['n']} | {v['init_mtre_mm']} mm "
                     f"| **{v['median_mtre_mm']} mm** | {v['mean_mtre_mm']} mm "
                     f"| {v['delta_mtre_mm']:+.2f} mm "
                     f"| **{v['success_rate']:.1%}** | {v['improved_rate']:.1%} "
                     f"| {v['seconds']} |")

    L += ["", "## Anatomy minus intensity, 95% CI from a bootstrap over trials", "",
          "| initialisation | success-rate difference | 95% CI | excludes zero |",
          "|---|---:|---|---|"]
    for c in conditions + ["ALL"]:
        p = s["paired"][f"{c}|success_rate"]
        L.append(f"| {c} | {p['anatomy_minus_intensity']:+.4f} | "
                 f"[{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] | "
                 f"{'yes' if p['excludes_zero'] else 'no'} |")

    L += ["", "## Split by how much bone the two volumes actually share", "",
          "Bone Dice at the reference is a property of the *pair*, computed before "
          "either method ran. It is not used to include or exclude anything.", "",
          "| bone Dice at reference | pairs | intensity success | anatomy success "
          "| intensity median mTRE | anatomy median mTRE |",
          "|---|---:|---:|---:|---:|---:|"]
    for label, v in s["by_bone_overlap"].items():
        L.append(f"| {label} | {v['n_pairs']} | {v['intensity']['success_rate']:.1%} "
                 f"| **{v['anatomy']['success_rate']:.1%}** "
                 f"| {v['intensity']['median_mtre_mm']} mm "
                 f"| {v['anatomy']['median_mtre_mm']} mm |")

    L += ["", "## Reading this honestly", "",
          "Both methods sit 7-9 mm from the reference on the median pair, even "
          "when they start 0.6 mm away from it. That is not an optimiser failure. "
          "`diagnose_reference.py` shows that the intensity method reaches a "
          "**higher** NCC than the reference does on **28 of 28 pairs**, and that "
          "ICP reaches a lower trimmed surface residual than the reference does on "
          "essentially all of them. Each method is optimising its own criterion "
          "successfully; the published reference is simply not where either "
          "criterion peaks.", "",
          "That is consistent with how the reference was made. The authors "
          "refined it with expert manual adjustment guided by MRI bone "
          "segmentations, and refined groups of four neighbouring volumes jointly "
          "with pose anchoring rather than one pair at a time. A pairwise "
          "automatic method has neither of those, so it cannot be expected to land "
          "on the same answer.", "",
          "The right reading of the table is therefore comparative, not absolute: "
          "**how often does each method land near the published reconstruction**, "
          "under the same perturbations, with the same 6-DoF freedom.", ""]
    (DOCS / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
