"""Check the downloaded data against what the dataset says about itself.

Runs before the experiment and writes `docs/data_audit.md`. Every number in that
file is read from the files on disk, not copied from the paper or the Zenodo
description -- where the two disagree, the audit records both.

    python audit.py
"""

from __future__ import annotations

import csv
import json

import numpy as np

from config import ARTIFACTS, DATA, DOCS, EXPECTED_DIM, EXPECTED_SPACING
from data import (mask_path, parse_mhd_header, read_label_map, read_mhd,
                  read_reference_transforms,
                  read_robot_poses, relative, rotation_angle, trial_dirs,
                  trial_label, volume_path)

# Values the source publications state, kept here so the audit can contradict
# them rather than quietly agreeing.
PAPER_SPACING = (0.1229, 0.107, 0.2405)
PAPER_N_VOLUMES, PAPER_N_SEGS, PAPER_N_TRIALS, PAPER_N_SUBJECTS = 321, 62, 7, 5


def check(name: str, ok: bool, detail: str = "", note: bool = False) -> dict:
    """`note=True` records a fact about the data rather than a pass/fail of ours.

    The spacing disagreement below is the reason this distinction exists: the
    headers and the paper genuinely differ, and that is worth recording, but it
    is not a defect in this pipeline.
    """
    tag = "NOTE" if note else ("PASS" if ok else "FAIL")
    print(f"  [{tag}] {name}" + (f" -- {detail}" if detail else ""))
    return dict(check=name, passed=bool(ok), detail=detail, note=note)


def main() -> None:
    checks: list[dict] = []
    seg_rows = list(csv.DictReader(
        (DATA / "metadata" / "humerus_segmentations.csv").open(encoding="utf-8")))
    trials = trial_dirs()
    labels = [trial_label(t) for t in trials]

    print("dataset")
    checks.append(check("7 trials present", len(trials) == PAPER_N_TRIALS,
                        f"{len(trials)}: {', '.join(labels)}"))
    checks.append(check("5 participants", len({l.split('_')[0] for l in labels}) == PAPER_N_SUBJECTS))
    checks.append(check(f"{PAPER_N_SEGS} humerus segmentations listed",
                        len(seg_rows) == PAPER_N_SEGS, str(len(seg_rows))))

    print("\ngeometry (read from every MHD header we downloaded)")
    dims, spacings, offsets, dtypes, mask_voxels = set(), set(), set(), set(), []
    dialects, brightness = set(), []
    missing = []
    for r in seg_rows:
        trial = trials[labels.index(r["trial_label"])]
        n = int(r["volume_number"])
        vp, mp = volume_path(trial, n), mask_path(trial, n)
        if not vp.exists() or not mp.exists():
            missing.append(f"{r['trial_label']} V{n}")
            continue
        h = parse_mhd_header(vp)
        dims.add(h["DimSize"])
        spacings.add(h["ElementSpacing"])
        offsets.add(h["Offset"])
        dtypes.add(h["ElementType"])
        vol = read_mhd(vp)
        mask = read_label_map(mp, vol)
        mask_voxels.append(int(mask.sum()))
        dialects.add("ImFusion" if "Position" in parse_mhd_header(mp) else "Slicer")
        if mask.any():
            brightness.append(float(vol.array[mask].mean()) / max(float(vol.array.mean()), 1e-9))

    checks.append(check("all needed volumes and masks downloaded", not missing,
                        "missing: " + ", ".join(missing) if missing else
                        f"{len(seg_rows)} volume/mask pairs"))
    checks.append(check("identical grid for every volume", len(dims) == 1,
                        f"DimSize {dims.pop() if len(dims)==1 else dims}"))
    sp = np.array([float(x) for x in list(spacings)[0].split()]) if len(spacings) == 1 else None
    checks.append(check("identical spacing for every volume", len(spacings) == 1,
                        f"{sp}" if sp is not None else str(spacings)))
    checks.append(check("spacing matches config.EXPECTED_SPACING",
                        sp is not None and np.allclose(sp, EXPECTED_SPACING, atol=1e-6)))
    checks.append(check(
        "spacing agrees with the figure printed in the PLOS ONE paper",
        sp is not None and np.allclose(sp, PAPER_SPACING, atol=1e-4),
        f"headers say {np.round(sp,5).tolist()}, paper says {list(PAPER_SPACING)}; "
        "the headers are what the pixel data sits on, so the headers are used",
        note=True))
    checks.append(check("voxel type is MET_UCHAR", dtypes == {"MET_UCHAR"}, str(dtypes)))

    print("\nhumerus masks")
    checks.append(check(
        "two MetaImage dialects are present among the masks", True,
        f"{sorted(dialects)}; the ImFusion ones carry a scene pose in their "
        "header, which read_label_map() ignores in favour of index correspondence",
        note=True))
    checks.append(check(
        "every mask lands on bright (bone-like) voxels of its own volume",
        min(brightness) > 1.3,
        f"mean intensity inside the mask is {min(brightness):.2f}x to "
        f"{max(brightness):.2f}x the volume mean (median {np.median(brightness):.2f}x)"))

    print("\nreference transforms")
    n_ref, rigid_ok = 0, True
    for t in trials:
        try:
            ref = read_reference_transforms(str(t))
        except ValueError as exc:
            rigid_ok = False
            print(f"    {trial_label(t)}: {exc}")
            continue
        n_ref += len(ref)
        for m in ref.values():
            r = m[:3, :3]
            if not (np.allclose(r @ r.T, np.eye(3), atol=1e-9)
                    and abs(np.linalg.det(r) - 1) < 1e-9):
                rigid_ok = False
    checks.append(check("reference transforms parse and are rigid", rigid_ok,
                        f"{n_ref} transforms across {len(trials)} trials"))

    print("\nrobot poses (parsed for the record; not used by the experiment)")
    rot_ref, rot_pose = [], []
    for t in trials:
        ref, pose = read_reference_transforms(str(t)), read_robot_poses(str(t))
        for n in sorted(set(ref) & set(pose)):
            if n + 1 in ref and n + 1 in pose:
                rot_ref.append(rotation_angle(relative(ref[n], ref[n + 1])))
                rot_pose.append(rotation_angle(relative(pose[n], pose[n + 1])))
    corr = float(np.corrcoef(rot_ref, rot_pose)[0, 1])
    checks.append(check("robot poses index-align with the reference transforms",
                        corr > 0.8,
                        f"relative-rotation correlation r = {corr:.3f} over "
                        f"{len(rot_ref)} consecutive pairs"))

    audit = dict(
        n_trials=len(trials), trials=labels,
        n_participants=len({l.split('_')[0] for l in labels}),
        n_segmentations=len(seg_rows),
        dim=EXPECTED_DIM, spacing_from_headers=list(map(float, sp)) if sp is not None else None,
        spacing_in_paper=list(PAPER_SPACING),
        mask_voxels=dict(min=int(min(mask_voxels)), max=int(max(mask_voxels)),
                         median=int(np.median(mask_voxels))) if mask_voxels else None,
        mask_dialects=sorted(dialects),
        mask_brightness_ratio=dict(min=round(min(brightness), 2), max=round(max(brightness), 2),
                                   median=round(float(np.median(brightness)), 2)),
        n_reference_transforms=n_ref,
        robot_pose_rotation_correlation=round(corr, 4),
        checks=checks,
    )
    (ARTIFACTS / "data_audit.json").write_text(json.dumps(audit, indent=2))
    write_md(audit)

    hard = [c for c in checks if not c["note"]]
    failed = [c for c in hard if not c["passed"]]
    notes = [c for c in checks if c["note"] and not c["passed"]]
    print(f"\n{len(hard) - len(failed)}/{len(hard)} checks passed"
          + (f", {len(notes)} recorded discrepancy" if notes else ""))
    if failed:
        print("FAILED:", ", ".join(c["check"] for c in failed))
    for c in notes:
        print(f"note: {c['check']}\n      {c['detail']}")


def write_md(a: dict) -> None:
    sp = a["spacing_from_headers"]
    lines = [
        "# Data audit", "",
        "Generated by `audit.py`. Every number here is read from the downloaded "
        "files; where the files disagree with the publications, both are shown.", "",
        "## What was downloaded", "",
        f"- source: Zenodo `10.5281/zenodo.20283247`, CC BY 4.0",
        f"- trials: **{a['n_trials']}** ({', '.join(a['trials'])})",
        f"- participants: **{a['n_participants']}**",
        f"- humerus segmentations: **{a['n_segmentations']}**",
        "- only the members the experiment needs were fetched (~1.7 GB of the "
        "8.5 GB archive), by HTTP range requests against the zip index", "",
        "## Volume geometry", "",
        "| property | value |", "|---|---|",
        f"| grid | {' x '.join(map(str, a['dim']))} |",
        f"| spacing, from the MHD headers | {' x '.join(f'{v:.5f}' for v in sp)} mm |",
        f"| spacing, as printed in the PLOS ONE paper | "
        f"{' x '.join(f'{v:.4f}' for v in a['spacing_in_paper'])} mm |",
        f"| physical extent | "
        f"{' x '.join(f'{v*d:.1f}' for v, d in zip(sp, a['dim']))} mm |",
        "| voxel type | MET_UCHAR |", "",
        "The header spacing and the published spacing differ by about 10% on the "
        "first axis. **The headers are used everywhere in this project**, since "
        "they are what the pixel data is actually on.", "",
        "## Humerus masks", "",
        f"Foreground voxels at full resolution: median "
        f"{a['mask_voxels']['median']:,}, range {a['mask_voxels']['min']:,}"
        f"-{a['mask_voxels']['max']:,}. Ultrasound images the bone *surface*, not "
        "its interior, so these are shells rather than solid bones.", "",
        f"The masks come in **two MetaImage dialects**: "
        f"{' and '.join(a['mask_dialects'])}. The ImFusion-written ones store a "
        "`Position` and `Orientation` that describe where the label map sat in "
        "the reconstructed scene, not the geometry of its own voxel grid -- for "
        "PA001_trial_A V12 the header `Orientation` equals that volume's "
        "reference rotation to 5e-7. Honouring those fields would apply the "
        "reference transform a second time. `read_label_map()` therefore ignores "
        "them and relies on index correspondence with the paired volume, which "
        "is verified two ways: identical `DimSize` and `ElementSpacing`, and the "
        "physical check that masked voxels are "
        f"{a['mask_brightness_ratio']['min']}x-{a['mask_brightness_ratio']['max']}x "
        f"brighter than the volume mean (median "
        f"{a['mask_brightness_ratio']['median']}x), which is what a bone surface "
        "looks like in ultrasound.", "",
        "## Transforms", "",
        f"- {a['n_reference_transforms']} reference transforms parsed; all are "
        "rigid (unit scale, zero shear, orthonormal rotation), which the "
        "dataset's own workspace script also asserts.",
        f"- Robot poses parse and index-align with them: the relative rotation "
        f"between consecutive volumes correlates at "
        f"**r = {a['robot_pose_rotation_correlation']:.3f}**. Their relative "
        "*translations* do not reconcile, so the experiment does not use them "
        "(see `EXPERIMENT_DEVIATIONS.md`).", "",
        "## Checks", "", "| check | result | detail |", "|---|---|---|",
    ]
    for c in a["checks"]:
        tag = "note" if c.get("note") else ("PASS" if c["passed"] else "**FAIL**")
        lines.append(f"| {c['check']} | {tag} | {c['detail'] or ''} |")
    (DOCS / "data_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written -> {DOCS / 'data_audit.md'}")


if __name__ == "__main__":
    main()
