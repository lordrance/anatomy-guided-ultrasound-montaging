# Results

504 registrations over 28 volume pairs from 6 trials, 2.56 minutes of wall clock on CPU.

Success is `mTRE < 5.0 mm` against the published refined reference registration. That threshold was fixed in `EXPERIMENT_PLAN.md` before any comparative result was looked at.

## Main table

| initialisation | method | n | mean initial mTRE | median mTRE | mean mTRE | mean change | success | ended closer than it started | seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EASY | A intensity (NCC) | 84 | 0.628 mm | **7.425 mm** | 8.3 mm | +7.67 mm | **25.0%** | 0.0% | 4.098 |
| EASY | B anatomy (ICP) | 84 | 0.628 mm | **6.949 mm** | 8.991 mm | +8.36 mm | **35.7%** | 0.0% | 0.269 |
| MEDIUM | A intensity (NCC) | 84 | 3.09 mm | **7.568 mm** | 8.531 mm | +5.44 mm | **23.8%** | 9.5% | 3.914 |
| MEDIUM | B anatomy (ICP) | 84 | 3.09 mm | **7.992 mm** | 10.353 mm | +7.26 mm | **32.1%** | 8.3% | 0.28 |
| HARD | A intensity (NCC) | 84 | 6.667 mm | **8.584 mm** | 9.209 mm | +2.54 mm | **17.9%** | 33.3% | 3.871 |
| HARD | B anatomy (ICP) | 84 | 6.667 mm | **9.383 mm** | 12.434 mm | +5.77 mm | **26.2%** | 35.7% | 0.292 |
| ALL | A intensity (NCC) | 252 | 3.462 mm | **8.25 mm** | 8.68 mm | +5.22 mm | **22.2%** | 14.3% | 3.961 |
| ALL | B anatomy (ICP) | 252 | 3.462 mm | **8.004 mm** | 10.593 mm | +7.13 mm | **31.3%** | 14.7% | 0.28 |

## Anatomy minus intensity, 95% CI from a bootstrap over trials

| initialisation | success-rate difference | 95% CI | excludes zero |
|---|---:|---|---|
| EASY | +0.1071 | [-0.0899, +0.2989] | no |
| MEDIUM | +0.0833 | [-0.0920, +0.2530] | no |
| HARD | +0.0833 | [-0.0417, +0.2500] | no |
| ALL | +0.0913 | [-0.0617, +0.2645] | no |

## Split by how much bone the two volumes actually share

Bone Dice at the reference is a property of the *pair*, computed before either method ran. It is not used to include or exclude anything.

| bone Dice at reference | pairs | intensity success | anatomy success | intensity median mTRE | anatomy median mTRE |
|---|---:|---:|---:|---:|---:|
| 0 (no shared bone) | 3 | 0.0% | **0.0%** | 12.221 mm | 25.311 mm |
| 0-0.10 | 8 | 18.1% | **11.1%** | 9.691 mm | 8.181 mm |
| 0.10-0.30 | 6 | 38.9% | **9.3%** | 6.172 mm | 11.378 mm |
| >= 0.30 | 11 | 22.2% | **66.7%** | 6.477 mm | 4.42 mm |

## Reading this honestly

Both methods sit 7-9 mm from the reference on the median pair, even when they start 0.6 mm away from it. That is not an optimiser failure. `diagnose_reference.py` shows that the intensity method reaches a **higher** NCC than the reference does on **28 of 28 pairs**, and that ICP reaches a lower trimmed surface residual than the reference does on essentially all of them. Each method is optimising its own criterion successfully; the published reference is simply not where either criterion peaks.

That is consistent with how the reference was made. The authors refined it with expert manual adjustment guided by MRI bone segmentations, and refined groups of four neighbouring volumes jointly with pose anchoring rather than one pair at a time. A pairwise automatic method has neither of those, so it cannot be expected to land on the same answer.

The right reading of the table is therefore comparative, not absolute: **how often does each method land near the published reconstruction**, under the same perturbations, with the same 6-DoF freedom.

