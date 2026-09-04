"""Reading the dataset: MetaImage volumes, the two transform formats, geometry.

Deliberately dependency-light. The MHD reader is ~40 lines and the transform
parsers follow the dataset's own `scripts/` verbatim, so every step from bytes
on disk to a 4x4 matrix is visible in this file rather than hidden behind ITK.
"""

from __future__ import annotations

import math
import re
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from config import DATA

MET_DTYPE = {
    "MET_UCHAR": np.uint8,
    "MET_CHAR": np.int8,
    "MET_USHORT": np.uint16,
    "MET_SHORT": np.int16,
    "MET_INT": np.int32,
    "MET_FLOAT": np.float32,
    "MET_DOUBLE": np.float64,
}


# --------------------------------------------------------------------------- #
# MetaImage


@dataclass(frozen=True)
class Volume:
    """A MetaImage volume with its physical geometry.

    `array` is indexed [x, y, z] to match MetaImage's own axis order, so
    physical position = offset + spacing * index. Keeping that convention makes
    the transform maths below read the same way as the file format.
    """

    array: np.ndarray
    spacing: np.ndarray  # (3,) mm
    offset: np.ndarray   # (3,) mm, physical position of voxel [0, 0, 0]
    path: Path

    @property
    def dim(self) -> np.ndarray:
        return np.array(self.array.shape)

    @property
    def extent(self) -> np.ndarray:
        """Physical size of the sampled box, mm."""
        return self.spacing * self.dim

    @property
    def corners(self) -> np.ndarray:
        """The 8 corners of the physical bounding box, (8, 3) mm."""
        lo, hi = self.offset, self.offset + self.extent
        return np.array([[lo[0] if a else hi[0],
                          lo[1] if b else hi[1],
                          lo[2] if c else hi[2]]
                         for a in (1, 0) for b in (1, 0) for c in (1, 0)])

    @property
    def centre(self) -> np.ndarray:
        return self.offset + self.extent / 2


def parse_mhd_header(path: Path) -> dict[str, str]:
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        header[k.strip()] = v.strip()
    return header


def _read_meta_array(path: Path) -> tuple[np.ndarray, dict[str, str]]:
    """Voxels of a MetaImage file, as (x, y, z), plus its header.

    Two dialects appear in this dataset. The volumes and 43 of the masks are
    written by 3D Slicer (`Offset`, `TransformMatrix`, `BinaryDataByteOrderMSB`);
    19 masks are written by ImFusion (`Position`, `Orientation`,
    `ElementByteOrderMSB`, `ImFusionMetaImageVersion`). This function only reads
    the voxels, so it does not care which; the geometry fields are the caller's
    problem, and `read_mhd` and `read_label_map` handle them differently on
    purpose.
    """
    h = parse_mhd_header(path)
    if h.get("NDims") != "3":
        raise ValueError(f"{path}: only 3D volumes are supported")
    dim = np.array([int(x) for x in h["DimSize"].split()])
    dtype = MET_DTYPE[h["ElementType"]]
    msb = h.get("BinaryDataByteOrderMSB", h.get("ElementByteOrderMSB", "False"))
    if msb.lower() == "true":
        dtype = np.dtype(dtype).newbyteorder(">")

    blob = (path.parent / h["ElementDataFile"]).read_bytes()
    if h.get("CompressedData", "False").lower() == "true":
        blob = zipfile.zlib.decompress(blob)
    arr = np.frombuffer(blob, dtype=dtype, count=int(np.prod(dim)))
    # MetaImage is x-fastest; reshape to (z, y, x) then move to (x, y, z).
    return np.ascontiguousarray(arr.reshape(dim[2], dim[1], dim[0]).transpose(2, 1, 0)), h


def read_mhd(path: Path) -> Volume:
    """An image volume with its own physical geometry."""
    arr, h = _read_meta_array(path)
    if "Position" in h or "Orientation" in h:
        raise ValueError(
            f"{path}: ImFusion-dialect header. Its Position/Orientation record a "
            "scene pose, not the voxel grid's own geometry, so reading it as a "
            "standalone volume would place it wrongly. Use read_label_map().")
    tm = np.array([float(x) for x in h.get("TransformMatrix", "1 0 0 0 1 0 0 0 1").split()])
    if not np.allclose(tm.reshape(3, 3), np.eye(3)):
        raise ValueError(f"{path}: non-identity TransformMatrix {tm}")
    spacing = np.array([float(x) for x in h["ElementSpacing"].split()])
    offset = np.array([float(x) for x in h["Offset"].split()])
    return Volume(arr, spacing, offset, path)


def mask_dialect(path: Path) -> str:
    """Which tool wrote a label map: "ImFusion" or "Slicer".

    This matters for more than tidiness. Placed by the reference transforms, the
    Slicer-written masks reproduce the authors' own published per-pair Dice at
    r = 0.91; the ImFusion-written ones only reach r = 0.60 and come out
    systematically low, which says their voxel grid could not be reconciled with
    the coordinate conventions this pipeline uses. `pairs.py` therefore excludes
    pairs that involve one. See `validate_geometry.py` and
    EXPERIMENT_DEVIATIONS.md.
    """
    return "ImFusion" if "Position" in parse_mhd_header(path) else "Slicer"


def read_label_map(path: Path, volume: Volume) -> np.ndarray:
    """A humerus mask as a boolean array on `volume`'s grid.

    The mask's *own* world pose is deliberately ignored. For the 19
    ImFusion-exported masks that pose is the placement the reconstruction had
    when the label map was saved -- for example the `Orientation` of
    PA001_trial_A V12 equals that volume's reference rotation to 5e-7 -- so
    honouring it would apply the reference transform twice. What is actually
    guaranteed is index correspondence: the label map is written on the same
    512 x 403 x 256 grid as the volume it segments, which is checked here and
    confirmed empirically in `docs/data_audit.md` (across all 62 masks the
    masked voxels are 1.82x to 8.53x brighter than the volume mean, median
    3.99x, as a bone surface in ultrasound should be).
    """
    arr, h = _read_meta_array(path)
    if arr.shape != volume.array.shape:
        raise ValueError(f"{path}: grid {arr.shape} != volume grid {volume.array.shape}")
    spacing = np.array([float(x) for x in h["ElementSpacing"].split()])
    if not np.allclose(spacing, volume.spacing, atol=1e-6):
        raise ValueError(f"{path}: spacing {spacing} != volume spacing {volume.spacing}")
    return arr > 0


# --------------------------------------------------------------------------- #
# transforms


def euler_to_mat(rx: float, ry: float, rz: float) -> np.ndarray:
    """ImFusion's `Pose::eulerToMat`, degrees. Taken from the dataset's own
    `scripts/generate_hybrid_refined_registration_workspaces.py`."""
    rx, ry, rz = (math.radians(a) for a in (rx, ry, rz))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array([
        [cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy],
        [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - cz * sx],
        [-sy, cy * sx, cx * cy],
    ])


def affine_params_to_matrix(params: list[float]) -> np.ndarray:
    """12 workbook parameters -> 4x4. Asserts the transform really is rigid."""
    if len(params) != 12:
        raise ValueError(f"expected 12 parameters, got {len(params)}")
    t, rot, scale, shear = params[0:3], params[3:6], params[6:9], params[9:12]
    if any(abs(s - 1.0) > 1e-9 for s in scale):
        raise ValueError(f"reference transform is not unit-scale: {scale}")
    if any(abs(s) > 1e-9 for s in shear):
        raise ValueError(f"reference transform is sheared: {shear}")
    m = np.eye(4)
    m[:3, :3] = euler_to_mat(*rot)
    m[:3, 3] = t
    return m


@lru_cache(maxsize=None)
def read_reference_transforms(trial_dir: str) -> dict[int, np.ndarray]:
    """Reference placement of each volume, as **local -> world** 4x4 matrices.

    Source: `hybrid_refined_registration.xlsx`, column A, row N = volume N, each
    cell holding 12 affine parameters.

    The returned matrix is the **inverse** of the one those parameters spell out.
    Under the coordinate convention used in this project, the parameters read as
    though they were local -> world, but taking them that way does not reproduce
    the geometry implied by the authors' reported overlap and Dice values. The
    direction was settled empirically against their published per-pair Dice table
    (`results/evaluation/`), by `validate_geometry.py`:

        placement            correlation with their Dice     mean Dice
        as written                    -0.03                    0.001
        inverted                      +0.69                    0.240
        inverted, Slicer-only         +0.91                    0.278  (theirs 0.347)

    Scanning all 45 possible row offsets on PA001_trial_A V31-V32 puts a single
    unambiguous peak at the correct row under the inverted reading (Dice 0.464
    against their 0.431) and no peak at all under the literal one.

    Read without openpyxl so the parser stays visible; the sheet is a plain
    shared-string table.
    """
    path = Path(trial_dir) / "transforms" / "hybrid_refined_registration.xlsx"
    with zipfile.ZipFile(path) as z:
        shared = [re.sub("<[^>]+>", "", m).strip() for m in re.findall(
            r"<si>(.*?)</si>", z.read("xl/sharedStrings.xml").decode(), re.S)]
        sheet = z.read("xl/worksheets/sheet1.xml").decode()

    out: dict[int, np.ndarray] = {}
    for row in re.finditer(r'<row[^>]*r="(\d+)"[^>]*>(.*?)</row>', sheet, re.S):
        n = int(row.group(1))
        cell = re.search(r'<c r="A\d+"([^>]*)>\s*<v>(\d+)</v>', row.group(2), re.S)
        if not cell:
            continue
        text = shared[int(cell.group(2))] if 't="s"' in cell.group(1) else cell.group(2)
        parts = text.split()
        if len(parts) != 12:
            continue  # region label rows such as "Ant-Inf Vol 1-12"
        out[n] = np.linalg.inv(affine_params_to_matrix([float(p) for p in parts]))
    if not out:
        raise ValueError(f"{path}: no transform rows parsed")

    # PA002_trial_A carries three stray matrices in rows 49-51 after a blank row,
    # for a trial that only has 45 volumes. Rows past the volume count are not
    # transforms for any volume, so they are dropped rather than silently
    # answering a lookup for a volume that does not exist.
    n_volumes = trial_volume_count(Path(trial_dir))
    return {n: m for n, m in out.items() if n <= n_volumes}


@lru_cache(maxsize=None)
def trial_volume_count(trial: Path) -> int:
    """Volumes acquired in a trial, from `metadata/trials.csv`.

    Read from the metadata rather than counted on disk: only the 62 segmented
    volumes were downloaded, so the directory listing would undercount.
    """
    import csv

    label = trial_label(trial)
    with (DATA / "metadata" / "trials.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["trial_label"] == label:
                return int(row["volume_count"])
    raise KeyError(f"{label} not in trials.csv")


@lru_cache(maxsize=None)
def read_robot_poses(trial_dir: str) -> dict[int, np.ndarray]:
    """`pose_estimation_calibrated_robot.txt`: `A_estimate[:,:,N] =` blocks.

    Parsed for completeness and for the audit only. The experiment does not use
    these: their relative rotations agree with the reference transforms
    (r = 0.92) but their relative translations do not, and the convention could
    not be resolved from the released metadata. See EXPERIMENT_DEVIATIONS.md.
    """
    path = Path(trial_dir) / "transforms" / "pose_estimation_calibrated_robot.txt"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    block = re.compile(r"A_estimate\s*\[:,\s*:\s*,\s*(\d+)\]\s*=")
    out: dict[int, np.ndarray] = {}
    i = 0
    while i < len(lines):
        m = block.search(lines[i])
        if not m:
            i += 1
            continue
        rows, c = [], i + 1
        while c < len(lines) and len(rows) < 4:
            parts = lines[c].split()
            if parts:
                rows.append([float(p) for p in parts])
            c += 1
        out[int(m.group(1))] = np.array(rows)
        i = c
    return out


# --------------------------------------------------------------------------- #
# rigid transform helpers


def apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 to (N, 3) points."""
    return pts @ m[:3, :3].T + m[:3, 3]


def relative(m_moving: np.ndarray, m_fixed: np.ndarray) -> np.ndarray:
    """Placement of the moving volume expressed in the fixed volume's frame."""
    return np.linalg.inv(m_fixed) @ m_moving


def rotation_angle(r: np.ndarray) -> float:
    """Geodesic angle of a rotation matrix, degrees."""
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(r[:3, :3]) - 1) / 2))))


def rigid_from_vec(vec: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """6-vector -> 4x4, rotating about `centre`.

    `vec` is [tx, ty, tz, rx, ry, rz]: translation in mm and a rotation vector
    (axis * angle) in degrees. Rotating about the volume centroid rather than
    the origin keeps the two halves of the parameter vector on comparable
    scales, which matters for a derivative-free optimiser.
    """
    t, rv = np.asarray(vec[:3], float), np.asarray(vec[3:], float)
    theta = np.linalg.norm(rv)
    if theta < 1e-12:
        r = np.eye(3)
    else:
        k = rv / theta
        a = math.radians(theta)
        kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        r = np.eye(3) + math.sin(a) * kx + (1 - math.cos(a)) * (kx @ kx)
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = centre - r @ centre + t
    return m


# --------------------------------------------------------------------------- #
# dataset index


def trial_dirs() -> list[Path]:
    root = DATA / "data" / "participants"
    return sorted(p for p in root.glob("*/trials/*") if p.is_dir())


def volume_path(trial: Path, number: int) -> Path:
    vid = f"USAnon_{number:04d}"
    return trial / "ultrasound" / "volumes" / vid / f"{vid}_frame1.mhd"


def mask_path(trial: Path, number: int) -> Path:
    return trial / "ultrasound" / "humerus_segmentations" / f"V{number}.mhd"


def trial_label(trial: Path) -> str:
    return f"{trial.parents[1].name}_trial_{trial.name}"
