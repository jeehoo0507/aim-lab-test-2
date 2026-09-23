# 200-epoch reliability-gated masking pilot

This is an **exploratory** extension of experiment 2, not a claim that teacher
prediction preservation guarantees better student learning. The existing COCO
train/val/test split and each seed's frozen DeiT-S teacher are reused. The
ImageNet-pretrained and scratch DeiT-Tiny students each see all 196 image
patches. Every training teacher pass sees 98 image patches plus CLS.

The five 200-epoch conditions per student initialization are MaskedKD,
Random Rescue 10, Low-score Rescue 10, Random10→Low10, and
Random10→MaskedKD. All use the same 200-epoch warmup/cosine schedule and
batch/optimizer settings. They **cannot** be compared directly to the old
100-epoch runs as a controlled schedule comparison. The first run is seed 0;
fresh seeds 1 and 2 can be added with the same command and output root.

The adaptive conditions start with Random10. On the fixed 200-image validation
probe, every 10 epochs they compare the current student's Random10 mask to a
candidate mask. Random10 and Low10 use the **same ten incoming patches** for
each image/repeat; they only differ in which selected ten patches are removed.
Random draws are keyed by image ID and repeat, so they do not change across
epochs or seeds. For MaskedKD the candidate is raw top-98 with no swap.
Teacher full-image predictions and each masked prediction are saved, along with
patch indices and the student attention score. The gate reports paired
full→masked KL, ground-truth log probability on examples the full teacher gets
right, and full-correct→masked-wrong frequency. A candidate is favorable when
KL decreases, log probability increases, and flips do not increase. After
epoch 30, two consecutive favorable probes switch the **next** training epoch
to the candidate permanently. If this does not occur, the run stays Random10.
This fixed rule is a pilot heuristic and should not be revised using test scores.

The 200 probe images are excluded from the 800-image validation subset used to
select student checkpoints. This prevents direct reuse of gate images for
student checkpoint selection. The existing frozen teacher was selected under
the earlier experiment-2 protocol. The test set has already been inspected in
earlier work, so a gain here remains exploratory until replicated on fresh
seeds/data.

Only one teacher forward is used per **training batch**. Gate comparisons are
evaluation-only and occur every ten epochs; they add computation but never
increase the 98-token training budget. An online multi-branch lookahead is not
part of this method. The launcher checks the measured A5000 free-memory budget
for its parallel job count. The output root is separate from experiment 2.

## GPU-server run

The original experiment-2 teacher checkpoints, prepared COCO dataset, and
`outputs/experiment2/benchmark.json` must already exist on the server. Pull
this code, then start the two-initialization seed-0 pilot with four concurrent
student processes:

```bash
git pull --ff-only origin main
mkdir -p logs
nohup bash setup.sh adaptive --seeds 0 --inits scratch imagenet --jobs 4 > logs/adaptive200_launcher.log 2>&1 < /dev/null &
```

For a shorter first pass, use `--inits scratch --output-root
outputs/adaptive200_scratch` and pass the same two arguments to
`adaptive-evaluate`. To add replication seeds, start a fresh output root with
`--seeds 0 1 2`. The protocol file pins selected seeds and initializations;
rerunning the **same** selection safely resumes incomplete jobs, while a new
selection needs a separate output root.

Monitor the launcher and per-condition logs:

```bash
tail -f logs/adaptive200_launcher.log
tail -n 3 outputs/adaptive200/_jobs/seed_0_scratch_adaptive_random_to_low_10.log
```

After all requested training conditions complete, evaluate best and last
checkpoints once, then export compact results:

```bash
bash setup.sh adaptive-evaluate --seeds 0 --inits scratch imagenet
bash setup.sh export --config configs/adaptive200.json --push
```

`outputs/adaptive200/analysis/gate_history.csv` records probe decisions;
`history.json` records each epoch's active mask, loss, validation score and
diagnostic metrics. `gate/epoch_*.npz` retains the paired masks and logits for
explanatory figures on the server. The main comparison is per-seed test macro
accuracy at epoch 200 and validation-best checkpoints, alongside the actual
switch epoch. A no-switch outcome is reported as such, not omitted.
