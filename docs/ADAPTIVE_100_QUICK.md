# Fast 100-epoch adaptive-mask pilot

This pilot reuses the completed experiment-2 MaskedKD, Random Rescue 10, and
Low-score Rescue 10 students and each seed's frozen teacher. It trains only
Random10→Low10 and Random10→MaskedKD for seed 0, separately for scratch and
ImageNet student initialization: **four new 100-epoch students total**. The
four processes run concurrently when the measured A5000 memory check passes.
The underlying student/teacher training batches still use one teacher forward
with 98 image patches; the fixed 200-image gate adds evaluation work at ten-epoch
intervals. It does not increase the number of teacher passes per training batch.

The 100-epoch cosine schedule, training set, frozen teacher, initialization,
optimizer, and KD settings must match the existing baselines. The launcher checks
the old result metadata and the freshly constructed initial model before GPU
training. The primary comparison is **epoch-100 last checkpoint macro accuracy**.
That avoids mixing checkpoint-selection rules. Validation uses the same full
1,000-image set as the old runs, so its 200 gate images are also in checkpoint
validation; do not use validation-best differences for the causal claim. The test
set has already been inspected, so even a seed-0 gain is exploratory.

On the GPU server, after pulling this commit and confirming the original
`outputs/experiment2` results are present:

```bash
git pull --ff-only origin main
mkdir -p logs
nohup bash setup.sh adaptive --config configs/adaptive100_quick.json \
  --seeds 0 --inits scratch imagenet \
  --methods adaptive_random_to_low_10 adaptive_random_to_student \
  --jobs 4 > logs/adaptive100_quick_launcher.log 2>&1 < /dev/null &
```

If the launcher stops at a provenance or GPU-memory check, no new training has
started. Check `logs/adaptive100_quick_launcher.log` and fix that specific issue.
Monitoring:

```bash
tail -f logs/adaptive100_quick_launcher.log
tail -n 3 outputs/adaptive100_quick/_jobs/seed_0_scratch_adaptive_random_to_student.log
```

The previous seed-0 100-epoch runs took about three hours per student. With
four concurrent jobs, expect roughly **3–4 hours wall time** for this pilot,
depending on I/O contention and gate-probe overhead. This is an estimate, not
a benchmark for the new method. Each run retains per-epoch history and fixed
probe outputs, periodic checkpoints at epoch 50, and best/last checkpoints.

After all four runs finish:

```bash
bash setup.sh adaptive-evaluate --config configs/adaptive100_quick.json \
  --seeds 0 --inits scratch imagenet \
  --methods adaptive_random_to_low_10 adaptive_random_to_student
```

Read `outputs/adaptive100_quick/analysis/quick_comparison.csv`. It includes the
three existing baselines and two new conditions for each initialization,
epoch-100 test macro accuracy, actual switch epoch, and differences in percentage
points from MaskedKD and Random10. A blank switch epoch means the rule never
found two consecutive favorable gate checks. Before writing this table, the
evaluator rejects mismatched teacher, dataset, or initial model fingerprints.
Use fresh seeds 1 and 2 in a separate follow-up once the decision rule is frozen.

## Held-out test attention examples

The compact Git export contains test logits but not the test photos, attention
arrays, or model weights. To draw test examples, use the saved 100-epoch
checkpoints and prepared COCO data on the GPU server after pulling the latest
code. This only performs inference; it does not train or change checkpoints.

```bash
.venv/bin/python scripts/plot_test_attention.py --init scratch \
  --sample-ids 189828 56545 --output-dir reports/test_attention/seed0_scratch
```

ID 189828 (airplane) was corrected by the adaptive Low10 student relative to
Random10; ID 56545 (bird) changed from correct to incorrect. These were picked
**after inspecting test predictions** and are explanatory examples, not new
evidence of a population-level gain. Each image shows the full teacher's and
three students' last-layer CLS→patch attention, the student's raw top-98, and a
fixed illustrative 98-patch teacher input. Random replacement is deterministic
for this figure but is not a record of any particular training-batch draw.
Attention brightness is not a causal attribution score.

To share the two PNGs:

```bash
git add reports/test_attention/seed0_scratch
git commit -m "results: add test attention examples"
git push origin main
```

For the original five-method, 200-epoch within-study protocol, see
[ADAPTIVE_200.md](ADAPTIVE_200.md).
