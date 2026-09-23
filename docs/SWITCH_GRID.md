# Four-way fixed switch pilot

Experiment 2 follow-up: seed 0, scratch DeiT-Tiny, 100 epochs. The frozen DeiT-S teacher and the existing seed-0 `student` (MaskedKD), `random` (Random Mask KD), and `full` (Full KD) baselines are reused. Four new training schedules are compared:

| Run | Epochs 1–20 | Epochs 21–50 | Epochs 51–100 |
|---|---|---|---|
| `random_to_student_20` | Random 98 | MaskedKD top-98 | MaskedKD top-98 |
| `random_to_student_50` | Random 98 | Random 98 | MaskedKD top-98 |
| `full_to_student_20` | Full 196 | MaskedKD top-98 | MaskedKD top-98 |
| `full_to_student_50` | Full 196 | Full 196 | MaskedKD top-98 |

Only the **teacher's** token selection changes; the student always sees all 196 patches. All four runs use the same teacher, student initialization, train/validation split, data order, optimizer, KD loss, and 100-epoch learning-rate schedule. The switch does not reset optimizer state or learning rate. The teacher is not retrained. Existing intermediate Full/Random checkpoints have weights but no optimizer state, so each new run repeats its prefix. The comparison report checks its validation score at the switch point against the original unswitched run.

The earlier `random_to_student_10` and `random_rescue_to_student_20` experiments are **not** among these four jobs. In this grid, `random` means Random Mask KD: all 98 teacher patches are drawn uniformly each step, not ten random replacements of the student selection.

## Run on the GPU server

If the two Full→MaskedKD runs from the earlier instructions are **currently running**, let them finish before pulling the new code. Completed runs are reused automatically; the launcher only starts unfinished conditions.

From the `aim-lab-test-2` repository:

```bash
git pull --ff-only origin main
mkdir -p logs
nohup .venv/bin/python -u scripts/run_switch_grid.py --jobs 4 > logs/switch_grid_launcher.log 2>&1 < /dev/null &
```

The A5000 benchmark previously passed four concurrent jobs and estimated 10.60 GiB free VRAM as a conservative requirement. The launcher checks current free VRAM against that measurement before starting jobs. DataLoader workers are fixed to zero per job to avoid the earlier multiprocessing abort. Existing method output folders remain separate; the four new folders are `outputs/experiment2/seed_0/scratch/<method>/`.

Monitor with:

```bash
tail -f logs/switch_grid_launcher.log
tail -n 2 outputs/experiment2/_jobs/seed_0_scratch_*_to_student_*.log
```

`COMPLETE` in the launcher log means all four runs finished, their best/last checkpoints were tested, and a comparison report was written. A failed job leaves its log and last completed-epoch checkpoint; fix the cause and rerun the launcher. If an old unfinished run was created under a different code version, do not delete its files: inspect its log before resuming.

```bash
cat outputs/experiment2/analysis/switch_grid_seed_0_scratch.md
```

The report includes validation best/last, test best/last macro accuracy, each scheduled run's difference from MaskedKD, and a prefix check against Full/Random. Epoch-wise probe files still contain student attention, raw top-98 selection, actual teacher input and teacher/student predictions. A positive result is exploratory because this test set has already been inspected; follow-up seeds are needed for a stable claim.
