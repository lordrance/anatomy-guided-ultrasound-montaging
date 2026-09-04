# Experiment plan

**Written first.** This file was written before `register.py` existed and before
any registration was run, and the success threshold, pair-selection rule and
perturbation levels have not been touched since. Everything that did change is
in `EXPERIMENT_DEVIATIONS.md` with its reason.

Be precise about what that does and does not prove. The repository has a single
initial commit, so **the git history is not evidence of the ordering** — you are
taking my word for it. What the history does show is that no threshold was ever
edited: `config.py` has carried `success_mtre = 5.0` since the first commit, and
`final_audit.py` re-derives every success flag from it independently.

## 1. Research question

> Does explicit anatomical guidance make 3D ultrasound registration more robust
> to initialisation error than intensity-only registration?

Plainly: when stitching limited-field-of-view ultrasound, does knowing where the
bone is make the alignment harder to break than looking at grey levels alone?

## 2. Where the question comes from

Zhang et al. (2026), *Automated Montaging of Ultrasound Scans of the Eye*
(Research Square preprint, CC BY 4.0), take limited-FOV ocular B-scans acquired
at three gaze angles, obtain an anatomical boundary, align the scans under a
geometric constraint, and merge them into a wider view — replacing subjective
manual merging.

We keep the principle and change the anatomy:

| Zhang et al. 2026 | This project |
|---|---|
| ocular ultrasound | shoulder 3D ultrasound |
| limited FOV, three gaze angles | limited FOV, ~45 partially overlapping sweeps |
| anatomical boundary (retinal wall) | anatomical surface (humerus) |
| constrained alignment (translation only) | constrained alignment (6-DoF rigid) |
| montage of the posterior eye | mosaic of the shoulder |
| reproducibility vs manual merging | robustness vs initialisation error |

**Not transferred:** SAM. The source paper uses SAM *interactively* — pretrained
inference with human point/box prompts, no training and no fine-tuning — so it
is a convenient way to obtain a boundary, not the contribution. Our dataset
ships expert humerus segmentations, so we use those and say so.

**We do not claim** to reproduce Zhang et al., to validate their ocular findings,
or to improve on their method. A shoulder result says nothing about an eye.

## 3. Data

Shoulder 3D Ultrasound Mosaicking Dataset, Zenodo `10.5281/zenodo.20283247`,
CC BY 4.0, published 2026-05-19. Companion paper: Sewify et al., *PLOS ONE*
`10.1371/journal.pone.0347231`.

- 5 volunteers, 7 trials, 321 MHD volumes, 62 humerus label maps
- each volume 512 x 403 x 256, spacing 0.11037 x 0.10695 x 0.23967 mm,
  MET_UCHAR, physical extent 56.5 x 43.1 x 61.4 mm
- reference registrations: rigid (unit scale, zero shear), one per volume
- We download **only the members the experiment needs** (~1.7 GB of the 8.5 GB
  archive) by HTTP range requests against the zip central directory.

**The published refined registrations are reference registrations, not absolute
physical ground truth.** The authors produced them with semi-automatic NCC
registration plus expert manual adjustment, guided visually by MRI bone
segmentations, iterated "until the operators deemed US volume pairwise alignment
visually satisfactory". Everything below measures *agreement with that published
reference*, never physical accuracy.

## 4. Pair selection (pre-registered)

A pair is eligible when all of the following hold:

1. both volumes are in the same trial;
2. both have a humerus segmentation;
3. their volume numbers differ by exactly 1 (adjacent acquisitions);
4. geometric image overlap under the reference transforms is **>= 15%**,
   measured by Monte-Carlo sampling with a fixed seed.

Criterion 4 is a property of the acquisition and is neutral between the two
methods. We deliberately do **not** filter on bone-mask overlap: that would
discard exactly the cases where anatomical guidance has least to work with and
would bias the comparison towards Method B. Bone-mask overlap at the reference
is instead reported as a descriptive covariate.

Expected N before filtering: 46. The final N is reported.

## 5. Methods

Both methods estimate the same thing — a 6-DoF rigid transform — so the
comparison is about the *alignment signal*, not about model capacity.

**Method A — intensity baseline.** Normalised cross-correlation driven rigid
registration, optimised with Nelder–Mead, coarse-to-fine at 1.0 mm then 0.5 mm.
This mirrors the automatic half of the source dataset's own pipeline (S1 Text:
"Fast NCC ... optimised with the Nelder–Mead simplex method"). We do not
reproduce their expert manual adjustment step.

**Method B — anatomy-guided.** Humerus surface extracted from the provided
label maps by marching cubes in physical coordinates, decimated, then trimmed
point-to-plane ICP.

## 6. Perturbation-recovery protocol

Robot poses are **not** used. Their translation convention could not be
reconciled with the reference transforms from the released metadata (relative
rotations agree, r = 0.92, but relative translations do not), and resolving it is
not needed for this question.

Instead, for each pair we take the published reference relative transform,
corrupt it by a known random rigid perturbation about the moving volume's
centroid, and ask each method to recover it.

| condition | max translation | max rotation |
|---|---|---|
| EASY | 1 mm | 1 deg |
| MEDIUM | 5 mm | 5 deg |
| HARD | 10 mm | 10 deg |

3 deterministic seeds per (pair, condition). Both methods receive the
**byte-identical** perturbed starting transform.

Scale: ~46 pairs x 3 conditions x 3 seeds x 2 methods ~= 828 registrations.

## 7. Metrics

**Primary**

1. **mTRE** — mean displacement of the 8 corners of the moving volume between
   the estimated placement and the reference placement, in mm.
2. **Success rate** — fraction of registrations with **mTRE < 5 mm**. This
   threshold is fixed here, before any comparative result is seen.
3. **Runtime** — wall-clock seconds per registration.

**Secondary, diagnostic only.** Dice of the two humerus masks in the overlap
region, and NCC of the intensities in the overlap region. These are reported but
are **not** the basis of any conclusion, because each structurally favours one
method: Dice is close to what ICP optimises, NCC is exactly what Method A
optimises.

The headline figure plots success rate against *actual* initial mTRE, not
against the nominal condition label.

## 8. What would falsify the hypothesis

If Method B's success rate is not higher than Method A's at MEDIUM and HARD, the
answer to the research question is no, and that is what gets reported. A null
result is an acceptable outcome of this experiment.

## 9. Known bias, stated up front

The reference registrations were themselves refined using MRI bone geometry as
the visual guide. An anatomy-driven method is therefore being scored against a
reference that was built with anatomical guidance. This favours Method B. We
report it rather than correct for it, and we temper any positive result
accordingly.

## 10. Deliverables

One summary table; one success-rate-vs-initialisation-error plot; mTRE and
runtime comparisons; 2–3 registration/mosaic visualisations; at least one
failure case; tests; a data audit generated from the downloaded files; a README
that states the limitations above.

No hyperparameter search, no GPU, no deep learning, no publication-scale
statistics.
