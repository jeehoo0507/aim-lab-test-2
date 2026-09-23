# Quick pilot: Full KD → MaskedKD

This is a small follow-up to Experiment 2, not part of the original seven-method protocol. The teacher uses all 196 patches for epochs 1–20 or 1–50, then the student's attention top-98 from epoch 21 or 51 through epoch 100. The student always sees all 196 patches. The teacher checkpoint, student initialization, data order, optimizer, KD loss and 100-epoch learning-rate schedule are held fixed. There is no optimizer or scheduler reset at the switch.

Compare with the **existing** seed-0 scratch `student` (MaskedKD throughout) and `full` (Full KD throughout) runs. Only the two switch schedules need training. The old Full KD checkpoints contain weights but no optimizer state, so starting from one would confound the switch with an optimizer reset. Each new run therefore repeats its Full KD prefix. The prefix validation score at the switch epoch is reported against the original Full KD curve.

On the server, from the `aim-lab-test-2` repository:

```bash
git pull --ff-only origin main
mkdir -p logs
nohup .venv/bin/python -u run.py train --config configs/full_switch_pilot.json --seed 0 --init scratch --method full_to_student_20 > logs/full_to_student_20.log 2>&1 < /dev/null &
nohup .venv/bin/python -u run.py train --config configs/full_switch_pilot.json --seed 0 --init scratch --method full_to_student_50 > logs/full_to_student_50.log 2>&1 < /dev/null &
```

Both jobs use zero DataLoader workers. The output directories are new children of `outputs/experiment2/seed_0/scratch/`; previous results remain intact. If GPU memory is busy, launch the second command after the first completes. Check progress with `tail -n 3 logs/full_to_student_*.log`. Each run prints `100/100` when training has finished.

After **both** runs finish:

```bash
.venv/bin/python run.py evaluate --config configs/full_switch_pilot.json --seeds 0 --inits scratch --methods full_to_student_20 full_to_student_50
.venv/bin/python scripts/summarize_full_switch.py
cat outputs/experiment2/analysis/full_switch_seed_0_scratch.md
```

The table reports validation best/last, test best/last macro accuracy and the epoch-100 test difference in percentage points from MaskedKD. `run.py analyze --config configs/full_switch_pilot.json` can subsequently generate the learning and probe curves for all runs. The epoch-wise probe files already record raw student selection and actual teacher input, including the immediate switch. These two new runs are an exploratory seed-0 check: prior analyses have already inspected this test set, and a positive result needs confirmation on additional seeds.
