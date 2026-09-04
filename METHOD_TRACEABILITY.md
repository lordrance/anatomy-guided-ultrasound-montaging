# Method traceability

Every component, and where it came from. The point is that a reader can tell
which parts are Zhang et al.'s idea, which are the shoulder dataset's, and which
are mine.

- **A — from Zhang et al. 2026** (the ocular montaging preprint)
- **S — from the shoulder dataset / its PLOS ONE paper**
- **D — domain adaptation**, kept in spirit but changed in form for a new organ
- **N — new here**, mine, not in either source
- **I — implementation choice**, could reasonably have been done otherwise

| # | component | class | note |
|---|---|---|---|
| 1 | limited-FOV ultrasound, partially overlapping acquisitions | A | the setting of the source paper |
| 2 | find an anatomical structure, use it to constrain alignment | A | the central idea being transferred |
| 3 | merge aligned acquisitions into a wider view | A | the montage step, `figures.py::fig_mosaic` |
| 4 | replacing subjective manual merging with a geometric pipeline | A | the motivation |
| 5 | SAM for the boundary | — | **not transferred.** Used interactively in the source (prompted inference, no training), so it is a means of getting a boundary, not the contribution |
| 6 | translation-only constraint | D | the source constrains alignment to translation for an eye; here the constraint is 6-DoF rigid, which is what the reference transforms are |
| 7 | reproducibility against manual merging as the metric | D | not reproducible here — there is one published reconstruction, not repeated manual merges. Replaced by robustness to a controlled perturbation |
| 8 | eye wall boundary -> humerus surface | D | the anatomical structure |
| 9 | 2D B-scan montage -> 3D volume mosaic | D | dimensionality |
| 10 | shoulder 3D ultrasound dataset | S | Zenodo 10.5281/zenodo.20283247 |
| 11 | humerus label maps | S | expert semi-automatic segmentation in 3D Slicer, per their S3 Text |
| 12 | reference registrations | S | their hybrid-refined transforms, used as a reference, not as ground truth |
| 13 | NCC + Nelder-Mead as the intensity baseline | S | their S1 Text specifies exactly this for the automatic half of their pipeline |
| 14 | their expert manual adjustment stage | — | **not reproduced** |
| 15 | their joint refinement of four neighbouring volumes with pose anchoring | — | **not reproduced**; this project is strictly pairwise |
| 16 | their published per-pair Dice table | S | used by `validate_geometry.py` as an external check on our geometry |
| 17 | perturbation-recovery protocol | N | take the reference, break it by a known rigid amount, ask each method to recover it |
| 18 | three initialisation levels (1/5/10 mm and degrees) | N | |
| 19 | trimmed point-to-plane ICP on the bone surface | N | the anatomy-guided arm; the source uses a different mechanism for a different geometry |
| 20 | capacity-matched comparison (both methods 6-DoF rigid, same start) | N | so the comparison is about the alignment signal, not model freedom |
| 21 | mTRE against the reference as the primary metric | N | |
| 22 | pre-registered 5 mm success threshold | N | fixed in `EXPERIMENT_PLAN.md` before any comparative result |
| 23 | stratifying by bone Dice at the reference | N | the analysis that explains the bimodality |
| 24 | the "is the reference the optimum?" diagnostic | N | `diagnose_reference.py`; the quantitative form of the main caveat |
| 25 | establishing that the reference transforms need inverting under our coordinate convention | N | the direction is not documented in the release; settled against the authors' own Dice table |
| 26 | excluding ImFusion-dialect masks | N | forced by the evidence in `docs/geometry_validation.md` |
| 27 | selective range download from the Zenodo zip | I | avoids fetching 6.8 GB that is never opened |
| 28 | own MetaImage reader instead of SimpleITK | I | ~40 lines, one fewer dependency, and the two header dialects had to be handled by hand anyway |
| 29 | own xlsx parser instead of openpyxl in `data.py` | I | the sheet is a plain shared-string table |
| 30 | isotropic 1.0 mm and 0.5 mm resampling levels | I | 0.5 mm is below the authors' own 1.3 +- 0.42 mm hand-eye calibration error, so it is not the accuracy bottleneck |
| 31 | marching cubes for the surface, 5000-point decimation | I | |
| 32 | cluster bootstrap over trials | I | registrations within a trial share volumes and an operator |

## How faithful is this to Zhang et al. 2026?

Rough and deliberately unflattering:

| dimension | fidelity |
|---|---|
| problem structure (limited FOV, overlap, anatomy, align, merge) | ~80% |
| modality (ultrasound) | 100% |
| organ | 0% — different organ, on purpose |
| algorithm | ~15% — no SAM, different constraint, different anatomy representation |
| evaluation | ~10% — reproducibility replaced by perturbation recovery |
| **overall** | **~35-40%** |

This is a transfer of the *principle*, not a reimplementation. The number is low
because it should be: reimplementing an ocular pipeline on a shoulder is not what
was attempted.
