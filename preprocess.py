"""Resample the segmented volumes to isotropic spacing and extract bone surfaces.

Done once, cached, and shared by both methods so that neither can be accused of
seeing better data than the other. Two resolutions are produced: 1.0 mm for the
coarse stage of the intensity registration and 0.5 mm for the fine stage and for
the surface extraction.

Resampling to an isotropic *physical* spacing rather than to a fixed cubic array
matters here: the source voxels are anisotropic (0.110 x 0.107 x 0.240 mm) on a
non-cubic grid (512 x 403 x 256), so forcing a cube would distort the geometry
the whole experiment is about.

    python preprocess.py
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy import ndimage
from skimage import measure

from config import CACHE, DEFAULT
from perturb import stable_seed
from data import (mask_path, read_label_map, read_mhd, trial_dirs, trial_label,
                  volume_path)

PREP = CACHE / "prep"
PREP.mkdir(parents=True, exist_ok=True)


def resample_isotropic(arr: np.ndarray, spacing: np.ndarray, offset: np.ndarray,
                       target: float, order: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Resample onto an isotropic grid of `target` mm, keeping physical position.

    Low-pass filters first along any axis that is being decimated; skipping that
    would alias ultrasound speckle straight into the coarse levels.
    """
    dim = np.array(arr.shape)
    extent = spacing * dim
    new_dim = np.maximum(np.round(extent / target).astype(int), 1)

    work = arr.astype(np.float32)
    sigma = np.maximum(0.5 * (target / spacing) - 0.5, 0.0)
    if sigma.max() > 0:
        work = ndimage.gaussian_filter(work, sigma=sigma, mode="nearest")

    # MetaImage places voxel i at `offset + i * spacing`, and every other file in
    # this project reads positions that way, so the resampled grid uses the same
    # convention: new voxel k sits at `offset + k * target`, i.e. source index
    # k * target / spacing. Using a half-voxel-centred grid here instead would
    # shift the resampled volumes by ~0.2 mm relative to their own masks.
    grids = [(target * np.arange(n)) / s for n, s in zip(new_dim, spacing)]
    coords = np.stack(np.meshgrid(*grids, indexing="ij"))
    out = ndimage.map_coordinates(work, coords, order=order, mode="constant", cval=0.0)
    return out.astype(np.float32), np.full(3, float(target))


def bone_surface(mask: np.ndarray, spacing: np.ndarray, offset: np.ndarray,
                 n_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Marching-cubes surface of a binary mask, in physical mm, decimated."""
    if mask.max() < 0.5:
        return np.zeros((0, 3)), np.zeros((0, 3))
    verts, faces, normals, _ = measure.marching_cubes(
        mask.astype(np.float32), level=0.5, spacing=tuple(spacing))
    verts = verts + offset
    if len(verts) > n_points:
        idx = np.random.default_rng(seed).choice(len(verts), n_points, replace=False)
        verts, normals = verts[idx], normals[idx]
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    return verts.astype(np.float32), normals.astype(np.float32)


def prepare_one(trial, number: int, cfg=DEFAULT) -> dict:
    label = trial_label(trial)
    out = PREP / f"{label}_V{number}.npz"
    if out.exists():
        with np.load(out) as z:
            return dict(trial=label, volume=number, path=str(out),
                        n_surface=int(z["surf_pts"].shape[0]), cached=True)

    vol = read_mhd(volume_path(trial, number))
    msk = read_label_map(mask_path(trial, number), vol)

    img_c, sp_c = resample_isotropic(vol.array, vol.spacing, vol.offset, cfg.coarse_spacing)
    img_f, sp_f = resample_isotropic(vol.array, vol.spacing, vol.offset, cfg.fine_spacing)
    msk_f, _ = resample_isotropic(msk.astype(np.float32), vol.spacing,
                                  vol.offset, cfg.fine_spacing)
    msk_f = msk_f > 0.5

    pts, nrm = bone_surface(msk_f, sp_f, vol.offset, cfg.icp_points,
                            seed=stable_seed(label, number))

    np.savez_compressed(
        out,
        img_coarse=img_c / 255.0, img_fine=img_f / 255.0, mask_fine=msk_f,
        spacing_coarse=sp_c, spacing_fine=sp_f, offset=vol.offset,
        src_spacing=vol.spacing, src_dim=np.array(vol.array.shape),
        surf_pts=pts, surf_nrm=nrm,
    )
    return dict(trial=label, volume=number, path=str(out),
                dim_coarse=list(map(int, img_c.shape)),
                dim_fine=list(map(int, img_f.shape)),
                mask_voxels=int(msk_f.sum()), n_surface=int(len(pts)),
                fg_fraction=round(float((img_f > 0).mean()), 4), cached=False)


def load(label: str, number: int) -> dict:
    with np.load(PREP / f"{label}_V{number}.npz") as z:
        return {k: z[k] for k in z.files}


def main() -> None:
    import csv

    from config import DATA
    rows = list(csv.DictReader(
        (DATA / "metadata" / "humerus_segmentations.csv").open(encoding="utf-8")))
    by_label = {trial_label(t): t for t in trial_dirs()}

    t0, info = time.time(), []
    for i, r in enumerate(rows, 1):
        trial = by_label[r["trial_label"]]
        d = prepare_one(trial, int(r["volume_number"]))
        info.append(d)
        state = "cached" if d["cached"] else f"{d['n_surface']} surface points"
        print(f"[{i:2d}/{len(rows)}] {d['trial']} V{d['volume']:<3d} {state}", flush=True)
    (CACHE / "prep_index.json").write_text(json.dumps(info, indent=2))
    print(f"prepared {len(info)} volumes in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
