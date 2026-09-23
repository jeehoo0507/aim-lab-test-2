# Random / Full → MaskedKD · seed 0 · scratch

Four new 100-epoch runs; teacher and original baselines are reused. Optimizer and learning-rate schedule do not reset at the switch.
All runs share dataset, teacher, and initial student weights. Random keeps 98 teacher patches throughout; Full changes from 196 to 98.

| Method | Best epoch | Val best (%) | Val last (%) | Test best (%) | Test last (%) | Δ last vs MaskedKD (pp) |
|---|---:|---:|---:|---:|---:|---:|
| student | 35 | 62.00 | 60.70 | 58.56 | 59.07 | +0.00 |
| random | 74 | 60.60 | 59.40 | 60.11 | 59.73 | +0.66 |
| full | 51 | 60.40 | 58.90 | 57.86 | 57.73 | -1.33 |
| random_to_student_20 | 89 | 61.40 | 61.20 | 58.10 | 58.83 | -0.24 |
| random_to_student_50 | 93 | 61.30 | 60.90 | 60.45 | 59.93 | +0.87 |
| full_to_student_20 | 76 | 60.70 | 59.80 | 59.10 | 59.36 | +0.29 |
| full_to_student_50 | 98 | 61.70 | 61.50 | 59.66 | 59.21 | +0.14 |

## Prefix validation check

The new run should agree with its existing unswitched baseline at the switch epoch.

| Scheduled run | Switch after epoch | New val (%) | Existing prefix val (%) | Difference (pp) |
|---|---:|---:|---:|---:|
| random_to_student_20 | 20 | 57.60 | 57.60 | +0.00 |
| random_to_student_50 | 50 | 58.50 | 58.50 | +0.00 |
| full_to_student_20 | 20 | 56.70 | 56.70 | +0.00 |
| full_to_student_50 | 50 | 58.70 | 58.70 | +0.00 |

This is an exploratory single-seed comparison on a previously inspected test set. Confirm any apparent gain with fresh seeds.
