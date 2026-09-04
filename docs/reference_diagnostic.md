# Is the reference the optimum?

Both methods end several millimetres from the published reference even when they start beside it. This file checks the obvious alternative explanation: that the reference is not where either method's criterion peaks.

| | at the reference | at the method's solution |
|---|---:|---:|
| mean NCC (Method A maximises this) | 0.6404 | **0.7977** |
| median trimmed surface RMS (Method B minimises this) | 0.785 mm | **0.3425 mm** |

Method A reaches a higher NCC than the reference on **28 of 28** pairs. Method B reaches a lower surface residual than the reference on **28 of 28**.

So neither optimiser is broken. The published reference encodes information that a pairwise automatic criterion does not have: it was produced with expert manual adjustment guided by MRI bone segmentations, and by jointly refining groups of four neighbouring volumes with pose anchoring rather than one pair at a time.

This is the quantitative form of the caveat stated throughout: the refined transforms are a **reference registration**, not absolute physical ground truth, and mTRE against them measures agreement with a particular high-quality reconstruction.

## Per pair

| pair | bone Dice at reference | NCC at reference | NCC at A's solution | surface RMS at reference | surface RMS at B's solution |
|---|---:|---:|---:|---:|---:|
| PA001_trial_A_V9-V10 | 0.017 | 0.8261 | 0.8591 | 2.0207 mm | 0.3381 mm |
| PA001_trial_A_V10-V11 | 0.3409 | 0.8186 | 0.883 | 0.7116 mm | 0.3946 mm |
| PA001_trial_A_V31-V32 | 0.461 | 0.8528 | 0.8802 | 0.457 mm | 0.3132 mm |
| PA001_trial_B_V9-V10 | 0.3138 | 0.7233 | 0.7895 | 0.8023 mm | 0.4096 mm |
| PA001_trial_B_V10-V11 | 0.4798 | 0.7654 | 0.8086 | 0.5513 mm | 0.4352 mm |
| PA001_trial_B_V11-V12 | 0.1429 | 0.7091 | 0.733 | 3.5009 mm | 0.8945 mm |
| PA001_trial_B_V30-V31 | 0.3343 | 0.5455 | 0.7165 | 0.5764 mm | 0.3152 mm |
| PA001_trial_B_V31-V32 | 0.066 | 0.6032 | 0.7578 | 1.7494 mm | 0.9705 mm |
| PA001_trial_B_V46-V47 | 0.2312 | 0.5063 | 0.8217 | 0.5603 mm | 0.3094 mm |
| PA002_trial_B_V11-V12 | 0.4205 | 0.7164 | 0.8061 | 0.4375 mm | 0.2792 mm |
| PA002_trial_B_V28-V29 | 0.0534 | 0.6876 | 0.7756 | 2.2549 mm | 0.4473 mm |
| PA002_trial_B_V29-V30 | 0.0828 | 0.5581 | 0.7772 | 1.1372 mm | 0.4196 mm |
| PA002_trial_B_V30-V31 | 0.1367 | 0.6595 | 0.8477 | 1.6929 mm | 0.9773 mm |
| PA002_trial_B_V31-V32 | 0.093 | 0.7268 | 0.7993 | 0.989 mm | 0.3731 mm |
| PA003_trial_A_V11-V12 | 0.3675 | 0.7434 | 0.7596 | 0.3663 mm | 0.2541 mm |
| PA003_trial_A_V32-V33 | 0.3833 | 0.7113 | 0.7234 | 0.4342 mm | 0.2309 mm |
| PA003_trial_A_V33-V34 | 0.5429 | 0.7487 | 0.8499 | 0.5035 mm | 0.3469 mm |
| PA004_trial_A_V10-V11 | 0.0 | 0.6436 | 0.8387 | 12.0707 mm | 0.2879 mm |
| PA004_trial_A_V30-V31 | 0.2845 | 0.6368 | 0.7414 | 0.6794 mm | 0.2821 mm |
| PA004_trial_A_V31-V32 | 0.0937 | 0.6114 | 0.7639 | 1.2055 mm | 0.3855 mm |
| PA005_trial_A_V8-V9 | 0.519 | 0.583 | 0.7957 | 0.5784 mm | 0.4173 mm |
| PA005_trial_A_V9-V10 | 0.5155 | 0.6206 | 0.8055 | 0.352 mm | 0.2841 mm |
| PA005_trial_A_V10-V11 | 0.2656 | 0.6503 | 0.8408 | 0.6217 mm | 0.3175 mm |
| PA005_trial_A_V11-V12 | 0.0281 | 0.5845 | 0.8125 | 0.7677 mm | 0.4007 mm |
| PA005_trial_A_V28-V29 | 0.0 | 0.4539 | 0.7487 | 4.5218 mm | 1.0696 mm |
| PA005_trial_A_V29-V30 | 0.0371 | 0.3976 | 0.7743 | 1.9011 mm | 0.2995 mm |
| PA005_trial_A_V30-V31 | 0.0 | 0.3982 | 0.8662 | 1.6935 mm | 0.291 mm |
| PA005_trial_A_V31-V32 | 0.1197 | 0.4503 | 0.7598 | 2.0402 mm | 0.291 mm |
