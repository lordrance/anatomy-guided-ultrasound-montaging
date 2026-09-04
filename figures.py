"""Figures. Reads only the committed artifacts; runs no registration.

    python figures.py
"""

from __future__ import annotations

import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from config import ARTIFACTS, DEFAULT, FIGURES
from data import apply
from preprocess import load as load_prep

COLOUR = {"intensity": "#3b6ea5", "anatomy": "#c1502e"}
LABEL = {"intensity": "A  intensity (NCC)", "anatomy": "B  anatomy (ICP)"}


def load() -> tuple[list[dict], dict]:
    rows = json.loads((ARTIFACTS / "results.json").read_text())["rows"]
    summary = json.loads((ARTIFACTS / "summary.json").read_text())
    return rows, summary


def save(fig, name: str) -> None:
    path = FIGURES / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path.name}")


# --------------------------------------------------------------------------- #


def fig_success_vs_initialisation(rows, summary) -> None:
    """The headline: does either method hold up as the start gets worse?"""
    conditions = ["EASY", "MEDIUM", "HARD"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    x = np.arange(len(conditions))
    for m in ("intensity", "anatomy"):
        y, lo, hi = [], [], []
        for c in conditions:
            sel = [r["success"] for r in rows if r["condition"] == c and r["method"] == m]
            p = np.mean(sel)
            se = np.sqrt(p * (1 - p) / len(sel))
            y.append(p * 100)
            lo.append(max(p - 1.96 * se, 0) * 100)
            hi.append(min(p + 1.96 * se, 1) * 100)
        ax.errorbar(x, y, yerr=[np.array(y) - lo, np.array(hi) - np.array(y)],
                    marker="o", capsize=4, lw=2, color=COLOUR[m], label=LABEL[m])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{summary['table'][c+'|anatomy']['init_mtre_mm']:.1f} mm"
                        for c in conditions])
    ax.set_xlabel("initialisation condition (mean initial mTRE)")
    ax.set_ylabel(f"registrations within {DEFAULT.success_mtre:.0f} mm of the reference (%)")
    ax.set_title("Success rate vs initialisation error")
    ax.set_ylim(0, 60)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    # actual initial error on the x axis rather than the nominal label
    for m in ("intensity", "anatomy"):
        sel = [r for r in rows if r["method"] == m]
        init = np.array([r["init_mtre_mm"] for r in sel])
        suc = np.array([r["success"] for r in sel], float)
        bins = np.array([0, 1, 2, 4, 6, 8, 12])
        cx, cy, cn = [], [], []
        for i in range(len(bins) - 1):
            k = (init >= bins[i]) & (init < bins[i + 1])
            if k.sum() >= 8:
                cx.append(init[k].mean())
                cy.append(suc[k].mean() * 100)
                cn.append(k.sum())
        ax2.plot(cx, cy, marker="o", lw=2, color=COLOUR[m], label=LABEL[m])
        for a, b, n in zip(cx, cy, cn):
            ax2.annotate(str(n), (a, b), textcoords="offset points", xytext=(0, 6),
                         ha="center", fontsize=7, color=COLOUR[m])
    ax2.set_xlabel("actual initial mTRE (mm), binned")
    ax2.set_ylabel("success (%)")
    ax2.set_title("Same thing against the perturbation actually applied")
    ax2.set_ylim(0, 60)
    ax2.grid(alpha=0.3)
    ax2.legend(frameon=False)
    fig.suptitle("Both methods degrade gently; neither is rescued by an easy start",
                 y=1.02, fontsize=11)
    save(fig, "fig1_success_vs_initialisation.png")


def fig_bone_overlap(rows, summary) -> None:
    """Where the anatomy advantage lives: pairs that actually share bone."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    order = list(summary["by_bone_overlap"])
    x = np.arange(len(order))
    w = 0.38
    for i, m in enumerate(("intensity", "anatomy")):
        y = [summary["by_bone_overlap"][k][m]["success_rate"] * 100 for k in order]
        ax.bar(x + (i - 0.5) * w, y, w, color=COLOUR[m], label=LABEL[m])
        for a, b in zip(x + (i - 0.5) * w, y):
            ax.annotate(f"{b:.0f}%", (a, b), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}\n({summary['by_bone_overlap'][k]['n_pairs']} pairs)"
                        for k in order], fontsize=8)
    ax.set_xlabel("humerus Dice between the two volumes at the reference")
    ax.set_ylabel("success (%)")
    ax.set_title("Anatomy guidance only helps where there is shared bone")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(frameon=False)

    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per[r["pair_id"]][r["method"]].append(r["success"])
        per[r["pair_id"]]["bone"] = r["bone_dice_at_reference"] or 0.0
    for m in ("intensity", "anatomy"):
        bx = [v["bone"] for v in per.values()]
        by = [np.mean(v[m]) * 100 for v in per.values()]
        ax2.scatter(bx, by, s=34, alpha=0.75, color=COLOUR[m], label=LABEL[m])
    ax2.axvline(0.30, ls="--", c="grey", lw=1)
    ax2.set_xlabel("humerus Dice at the reference (per pair)")
    ax2.set_ylabel("success over the 9 runs of that pair (%)")
    ax2.set_title("Per pair")
    ax2.grid(alpha=0.3)
    ax2.legend(frameon=False)
    save(fig, "fig2_bone_overlap.png")


def fig_error_and_runtime(rows, summary) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                  gridspec_kw={"width_ratios": [1.7, 1]})
    conditions = ["EASY", "MEDIUM", "HARD"]
    pos, data, colours, ticks = [], [], [], []
    for i, c in enumerate(conditions):
        for j, m in enumerate(("intensity", "anatomy")):
            data.append([r["mtre_mm"] for r in rows
                         if r["condition"] == c and r["method"] == m])
            pos.append(i * 3 + j)
            colours.append(COLOUR[m])
        ticks.append(i * 3 + 0.5)
    bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black"))
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    for p, d in zip(pos, data):
        ax.scatter(np.full(len(d), p) + np.random.default_rng(0).normal(0, 0.08, len(d)),
                   d, s=4, alpha=0.25, color="black", zorder=3)
    ax.axhline(DEFAULT.success_mtre, ls="--", c="green", lw=1.2,
               label=f"success threshold {DEFAULT.success_mtre:.0f} mm")
    ax.set_xticks(ticks)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("mTRE against the reference (mm)")
    ax.set_yscale("log")
    ax.set_title("Final error. Anatomy is bimodal: it nails it or it slides off")
    ax.grid(alpha=0.3, axis="y")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLOUR[m], alpha=0.65) for m in COLOUR]
    ax.legend(handles + [ax.lines[-1]], list(LABEL.values())
              + [f"threshold {DEFAULT.success_mtre:.0f} mm"], frameon=False, fontsize=8)

    secs = [summary["table"]["ALL|" + m]["seconds"] for m in ("intensity", "anatomy")]
    ax2.bar([0, 1], secs, color=[COLOUR["intensity"], COLOUR["anatomy"]], width=0.55)
    for i, v in enumerate(secs):
        ax2.annotate(f"{v:.2f} s", (i, v), textcoords="offset points", xytext=(0, 3),
                     ha="center")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([LABEL[m] for m in ("intensity", "anatomy")], fontsize=8)
    ax2.set_ylabel("mean seconds per registration (CPU)")
    ax2.set_title(f"Runtime: {secs[0]/secs[1]:.0f}x apart")
    ax2.grid(alpha=0.3, axis="y")
    save(fig, "fig3_error_and_runtime.png")


def fig_reference_diagnostic() -> None:
    d = json.loads((ARTIFACTS / "reference_diagnostic.json").read_text())
    rows = d["rows"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    a = np.array([r["ncc_at_reference"] for r in rows])
    b = np.array([r["ncc_at_intensity_solution"] for r in rows])
    ax.scatter(a, b, s=38, color=COLOUR["intensity"])
    lim = [min(a.min(), b.min()) - 0.03, max(a.max(), b.max()) + 0.03]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("NCC at the published reference")
    ax.set_ylabel("NCC at Method A's solution")
    ax.set_title(f"Method A beats the reference on its own criterion\n"
                 f"{d['intensity_beats_reference_on_its_own_criterion']} of "
                 f"{d['n_pairs']} pairs above the diagonal")
    ax.grid(alpha=0.3)

    a = np.array([r["surface_rms_at_reference_mm"] for r in rows])
    b = np.array([r["surface_rms_at_anatomy_solution_mm"] for r in rows])
    ax2.scatter(a, b, s=38, color=COLOUR["anatomy"])
    lim = [0.1, max(a.max(), b.max()) * 1.3]
    ax2.plot(lim, lim, "k--", lw=1)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.set_xlabel("trimmed surface RMS at the reference (mm)")
    ax2.set_ylabel("at Method B's solution (mm)")
    ax2.set_title(f"Method B beats it too\n"
                  f"{d['anatomy_beats_reference_on_its_own_criterion']} of "
                  f"{d['n_pairs']} pairs below the diagonal")
    ax2.grid(alpha=0.3)
    fig.suptitle("The reference registration is not the optimum of either criterion, "
                 "so mTRE against it measures agreement, not accuracy", y=1.03, fontsize=10)
    save(fig, "fig4_reference_is_not_the_optimum.png")


# --------------------------------------------------------------------------- #
# qualitative


def _slice_pair(fixed: dict, moving: dict, transform: np.ndarray, axis: int = 2):
    """Fixed slice through the middle, and the moving volume resampled onto it."""
    f = fixed["img_fine"]
    k = f.shape[axis] // 2
    idx = [np.arange(f.shape[0]), np.arange(f.shape[1]), np.arange(f.shape[2])]
    idx[axis] = np.array([k])
    grid = np.stack(np.meshgrid(*idx, indexing="ij"), axis=-1).reshape(-1, 3)
    phys = fixed["offset"] + grid * fixed["spacing_fine"]
    q = (apply(np.linalg.inv(transform), phys) - moving["offset"]) / moving["spacing_fine"]
    m = ndimage.map_coordinates(moving["img_fine"], q.T, order=1, mode="constant", cval=0.0)
    shape = [s for i, s in enumerate(f.shape) if i != axis]
    fslice = np.take(f, k, axis=axis)
    return fslice, m.reshape(shape)


def fig_qualitative(rows) -> None:
    """One case each: anatomy recovers, and anatomy slides off."""
    per = defaultdict(list)
    for r in rows:
        if r["method"] == "anatomy" and r["condition"] == "HARD":
            per[r["pair_id"]].append(r)
    scored = sorted(per.items(), key=lambda kv: np.mean([x["mtre_mm"] for x in kv[1]]))
    picks = [("recovered", scored[0][1][0]), ("failed", scored[-1][1][0])]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4))
    for row, (tag, rec) in enumerate(picks):
        moving = load_prep(rec["trial"], rec["moving"])
        fixed = load_prep(rec["trial"], rec["fixed"])
        pair_rows = [r for r in rows if r["pair_id"] == rec["pair_id"]
                     and r["condition"] == rec["condition"] and r["repeat"] == rec["repeat"]]
        inten = next(r for r in pair_rows if r["method"] == "intensity")
        panels = [
            ("published reference", np.array(rec["reference_transform"]).reshape(4, 4), None),
            (f"A intensity, mTRE {inten['mtre_mm']:.1f} mm",
             np.array(inten["transform"]).reshape(4, 4), inten["mtre_mm"]),
            (f"B anatomy, mTRE {rec['mtre_mm']:.1f} mm",
             np.array(rec["transform"]).reshape(4, 4), rec["mtre_mm"]),
        ]
        for col, (title, T, err) in enumerate(panels):
            ax = axes[row, col]
            f, m = _slice_pair(fixed, moving, T)
            rgb = np.zeros(f.shape + (3,))
            rgb[..., 1] = np.clip(f * 2.2, 0, 1)          # fixed volume, green
            rgb[..., 0] = np.clip(m * 2.2, 0, 1)          # moving volume, red
            ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower")
            ok = "" if err is None else ("  PASS" if err < DEFAULT.success_mtre else "  FAIL")
            ax.set_title(title + ok, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(
            f"{rec['pair_id']}\n{tag}\nbone Dice at ref "
            f"{rec['bone_dice_at_reference']:.2f}", fontsize=8)
    fig.suptitle("Central slice through the fixed volume (green) with the moving "
                 "volume resampled onto it (red).\nYellow is agreement. HARD "
                 "initialisation, first repeat.", fontsize=10, y=0.99)
    save(fig, "fig5_qualitative.png")


def _find_chain(rows, length: int = 3) -> tuple[str, list[int]] | None:
    """A run of consecutive volumes whose every adjacent pair was evaluated."""
    have = {(r["trial"], r["moving"]) for r in rows}
    edges = {(r["trial"], r["moving"], r["fixed"]) for r in rows}
    for trial, start in sorted(have):
        run = [start]
        while len(run) < length and (trial, run[-1], run[-1] + 1) in edges:
            run.append(run[-1] + 1)
        if len(run) == length:
            return trial, run
    return None


def fig_mosaic(rows) -> None:
    """A three-volume strip, compounded under each set of transforms.

    This is the montage step the source paper is about, in miniature: chain the
    pairwise transforms and compound the volumes into one wider view.
    """
    found = _find_chain(rows)
    if found is None:
        print("  (no three-volume chain available, skipping mosaic figure)")
        return
    trial, run = found

    def rel_for(a: int, b: int, key: str, method: str) -> np.ndarray:
        rec = next(r for r in rows if r["trial"] == trial and r["moving"] == a
                   and r["fixed"] == b and r["condition"] == "HARD"
                   and r["repeat"] == 0 and r["method"] == method)
        return np.array(rec[key]).reshape(4, 4)

    vols = {n: load_prep(trial, n) for n in run}
    modes = [("published reference", "reference_transform", "anatomy"),
             ("A  intensity (NCC)", "transform", "intensity"),
             ("B  anatomy (ICP)", "transform", "anatomy")]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    for ax, (title, key, method) in zip(axes, modes):
        # place every volume in the frame of the first one
        place = {run[0]: np.eye(4)}
        for a, b in zip(run, run[1:]):
            place[b] = place[a] @ np.linalg.inv(rel_for(a, b, key, method))

        corners = np.vstack([apply(place[n], np.array(
            [[x, y, z] for x in (v["offset"][0], v["offset"][0] + v["src_spacing"][0] * v["src_dim"][0])
             for y in (v["offset"][1], v["offset"][1] + v["src_spacing"][1] * v["src_dim"][1])
             for z in (v["offset"][2], v["offset"][2] + v["src_spacing"][2] * v["src_dim"][2])]))
            for n, v in vols.items()])
        lo, hi = corners.min(0), corners.max(0)
        step = 0.6
        gx = np.arange(lo[0], hi[0], step)
        gy = np.arange(lo[1], hi[1], step)
        z = (lo[2] + hi[2]) / 2
        px, py = np.meshgrid(gx, gy, indexing="ij")
        pts = np.stack([px.ravel(), py.ravel(), np.full(px.size, z)], axis=1)

        canvas = np.zeros(px.size)
        for n, v in vols.items():
            q = (apply(np.linalg.inv(place[n]), pts) - v["offset"]) / v["spacing_fine"]
            got = ndimage.map_coordinates(v["img_fine"], q.T, order=1,
                                          mode="constant", cval=0.0)
            canvas = np.maximum(canvas, got)
        ax.imshow(np.clip(canvas.reshape(px.shape) * 2.2, 0, 1).T, cmap="gray",
                  origin="lower")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{trial}: V{run[0]}-V{run[-1]} compounded into V{run[0]}'s frame "
                 "(maximum intensity, mid-depth slice), pairwise transforms chained "
                 "from the HARD initialisation", fontsize=10)
    save(fig, "fig6_mosaic.png")


def main() -> None:
    rows, summary = load()
    print("writing figures:")
    fig_success_vs_initialisation(rows, summary)
    fig_bone_overlap(rows, summary)
    fig_error_and_runtime(rows, summary)
    fig_reference_diagnostic()
    fig_qualitative(rows)
    fig_mosaic(rows)


if __name__ == "__main__":
    main()
