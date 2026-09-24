# Random10 → Low10 schedule follow-up

This opt-in experiment trains **six new scratch DeiT-Tiny students**: seeds 1 and 2,
each with Random10→Low10 after epoch 50, after epoch 70, and the mixed schedule.
The old 100-epoch Random10, Low10 and MaskedKD runs are reused as baselines.
All runs reuse the original frozen teacher and the same dataset, initial student
weights, optimizer, learning-rate schedule and 100-epoch budget. Student input
is full (196 patches); teacher input stays at 98 patches and one teacher forward
per training batch. No foreground annotation enters these three training masks.

| Epoch | Fixed @50 | Fixed @70 | Mixed outgoing 10 |
|---:|---|---|---|
| 1–50 | Random10 | Random10 | 10 random |
| 51–70 | Low10 | Random10 | 5 lowest-attention + 5 random |
| 71–100 | Low10 | Low10 | 8 lowest-attention + 2 random |

Each method fills all ten vacancies with random patches outside the student's
top-98, so the teacher budget remains fixed. The mixed method keeps exploring
through epoch 100. These schedules were chosen after inspecting seed-0 results;
they are exploratory, not independently selected on seed 1 or 2.

From the GPU server's `aim-lab-test-2` directory, fetch and switch to the
experiment branch (the shared `main` branch is unchanged):

```bash
git fetch origin codex/random-low-sweep
git switch --track origin/codex/random-low-sweep
```

Then launch:

```bash
mkdir -p logs
nohup bash setup.sh adaptive --config configs/random_low_sweep.json \
  --seeds 1 2 --inits scratch \
  --methods random_rescue_to_low_50 random_rescue_to_low_70 random_low_mixed_10 \
  --jobs 6 > logs/random_low_sweep_launcher.log 2>&1 < /dev/null &
```

The launcher checks the existing baselines, dataset, teacher checkpoint and
initial student fingerprints before training. It checks the measured A5000 GPU
budget; six jobs require at least **14.72 GiB free** according to the previous
benchmark. If the check fails, training has not started. Free GPU memory before
retrying. With six simultaneous jobs, allow roughly **4–5 hours**, contingent on
similar server load; parallel jobs still compete for GPU and I/O. DataLoader
workers are zero to avoid the earlier multiprocessing abort. A rerun resumes
incomplete runs; completed matching runs are reused.

Monitor without changing training:

```bash
tail -f logs/random_low_sweep_launcher.log
tail -n 2 outputs/random_low_sweep/_jobs/seed_1_scratch_random_low_mixed_10.log
```

Each run writes validation and fixed-probe records every epoch, plus weights at
epoch 50, validation-best and epoch 100. Once the launcher prints `COMPLETE 6
students`, evaluate and compare the **epoch-100 test macro accuracy**:

```bash
bash setup.sh adaptive-evaluate --config configs/random_low_sweep.json \
  --seeds 1 2 --inits scratch \
  --methods random_rescue_to_low_50 random_rescue_to_low_70 random_low_mixed_10
cat outputs/random_low_sweep/analysis/quick_comparison.csv
```

The comparison includes the already trained MaskedKD, Random10 and Low10 for
each seed. Do not select a schedule by repeatedly inspecting the test set;
the earlier seed-0 test set has already informed these schedules.
