# Deviations from the plan, and mistakes made along the way

`EXPERIMENT_PLAN.md` was written before any registration ran. This file records
everything that departed from it, including the errors I made and had to fix.
Nothing here was discovered by looking at the comparative result and working
backwards.

---

## DEV-1 — the reference transforms needed inverting for our coordinate convention

**Plan said:** read the 12 workbook parameters per volume as the volume's
placement, following the dataset's own `generate_hybrid_refined_registration_workspaces.py`.

**What happened:** read that way, adjacent humerus masks never touch. Our Dice
came out at 0.001 on average where the authors' own published table says 0.379.

**How it was settled:** not by tuning until it looked right. The dataset ships
`results/evaluation/dice_coefficient_extract_full_data.xlsx`, the authors' own
per-pair Dice. Two independent checks:

1. Scanning all 45 possible row offsets on PA001_trial_A V31-V32 gives a single
   unambiguous peak at the correct row under the inverted reading (Dice 0.464 vs
   their 0.431), and no peak at all under the literal one.
2. Across all 30 comparable pairs: correlation with their table is **-0.03** as
   written and **+0.69** inverted.

`read_reference_transforms()` therefore returns the inverse. Full table in
`docs/geometry_validation.md`.

**This was caught before the experiment ran.** Had it not been, every number in
this repository would have been wrong.

---

## DEV-2 — pairs involving an ImFusion-dialect mask are excluded

**Plan said:** all adjacent segmented pairs with >= 15% overlap.

**What happened:** the 62 masks come in two MetaImage dialects — 43 written by
3D Slicer, 19 by ImFusion. The ImFusion headers carry `Position` and
`Orientation` describing the reconstructed scene rather than their own voxel
grid; for PA001_trial_A V12 the header `Orientation` equals that volume's
reference rotation to 5e-7.

Under the corrected transform, Slicer-dialect pairs reproduce the authors'
published Dice at **r = 0.91**; ImFusion-dialect pairs only reach **r = 0.60**
and come out systematically low. Neither honouring the header pose nor ignoring
it reproduced their numbers consistently for those masks.

**Change:** pairs where either mask is ImFusion-dialect are excluded. This is why
N is 28 rather than 46. It is a data-integrity exclusion decided on external
evidence, before any registration, not a filter on outcome.

Note that the masks themselves are fine: our mask volumes match the authors'
listed volumes exactly (0.0900 / 0.6236 / 0.3954 / 0.8810 cm3 for the four spot
checks), and masked voxels are 1.82x-8.53x brighter than the volume mean in every
one of the 62, as a bone surface in ultrasound should be. It is the *frame* that
could not be recovered.

---

## DEV-3 — the >= 15% overlap rule turned out to exclude nothing

Pre-registered and applied as written, but after DEV-1 the smallest adjacent
overlap is 51%, so it removed no pairs. Reported rather than quietly dropped: a
pre-registered criterion that does nothing is still worth stating.

---

## DEV-4 — one pair has a degenerate reference

PA004_trial_A V9 and V10 have byte-identical reference transforms, which would
place two different volumes at exactly the same pose. Excluded, with the check
added to `pairs.py`. It was already excluded by DEV-2, so this changes no count.

---

## DEV-5 — robot poses were dropped from the design

**Plan said:** don't use them, and explained why. Recording the evidence here.

The robot poses parse cleanly and index-align with the reference transforms —
the relative rotation between consecutive volumes correlates at **r = 0.82 over
314 pairs**. Their relative *translations* do not reconcile under any of the four
obvious conventions (as-is, inverse, transpose, inverse-transpose): pairwise
distance structure correlates at 0.20 at best. The likely missing piece is a
fixed probe-to-end-effector offset that is not in the release.

Rather than guess, the experiment uses controlled perturbations, which makes the
recovery target exact by construction and removes the dependence entirely.

---

## DEV-6 — resampling convention, my mistake

The first version of `resample_isotropic` placed new voxel *k* at
`offset + target*(k+0.5)` while every other file read positions as
`offset + k*spacing`. That is a systematic ~0.2 mm shift between a volume and its
own mask. Caught while reconciling geometry for DEV-1, fixed to the corner
convention, and pinned by `test_resample_preserves_physical_position`. The
preprocessing cache was rebuilt from scratch afterwards.

---

## DEV-7 — non-reproducible seed, my mistake

Surface decimation seeded with Python's `hash()`, which is randomised per process
for strings, so the bone surfaces would have differed between runs. Replaced with
`perturb.stable_seed` (blake2b). Pinned by
`test_stable_seed_does_not_depend_on_the_process`, which asserts a literal value
so that a change to the hashing fails a test rather than silently reshuffling
every perturbation in the experiment.

---

## DEV-8 — a test that proved nothing, my mistake

`test_intensity_recovers_a_known_shift` originally registered a volume against
*itself* while asserting recovery of an invented transform. Identity is the
correct answer there, and the optimiser found it, so the test failed for the
right reason. Rewritten to use the same voxels at a different physical offset, so
the true transform is a known shift.

---

## DEV-9 — the header spacing disagrees with the paper

The MHD headers say 0.11037 x 0.10695 x 0.23967 mm; the PLOS ONE paper prints
0.1229 x 0.107 x 0.2405 mm, about 10% apart on the first axis. The headers are
what the pixel data sits on, so the headers are used everywhere. `audit.py`
records this as a NOTE rather than a failure, because our pipeline is not the
thing that is wrong.

---

## DEV-10 — ICP convergence reporting

The first version reported `converged = iterations < max`, which was always False
because the tolerance is tight. Changed to report whether the final step was
below tolerance, and to record `final_step_mm`. No effect on any result — the
solver already ran to the iteration cap either way.

---

## Things the plan promised and got

- Success threshold fixed at 5 mm before any comparative result: **yes** — it was
  written into the plan before `register.py` existed and never edited. The
  repository has one initial commit, so the history does not independently prove
  the ordering; `EXPERIMENT_PLAN.md` says so too rather than implying otherwise.
- Both methods handed byte-identical starting transforms: **yes**, verified by
  `final_audit.py` (max difference 0.00e+00 mm over 252 paired runs).
- No filtering on bone overlap: **yes** — it is recorded as a covariate and
  reported, never used to include or exclude.
- A null result is acceptable: the headline success-rate difference does not
  clear zero, and that is what the README says.
- No hyperparameter search, no GPU, no deep learning: **yes**.
