# Quick random → MaskedKD pilots

"Random" in Experiment 2 has two distinct meanings. Both schedules below are available so the intended comparison can be selected without retraining the existing baselines.

| Pilot | Through switch epoch | After switch | Existing controls |
|---|---|---|---|
| `random_to_student_10` | Random Mask KD: teacher gets 98 uniformly random patches through epoch 10 | MaskedKD top-98 from epoch 11 | `random`, `student` |
| `random_rescue_to_student_20` | Random Rescue 10: teacher gets student top-98 after replacing 10 patches at random through epoch 20 | MaskedKD top-98 from epoch 21 | `random_rescue_10`, `student` |

The student always sees all 196 patches, the teacher always sees 98, and the optimizer and 100-epoch LR schedule continue through the switch. The original `random`/`random_rescue_10` checkpoints contain no optimizer state, so each scheduled run repeats its prefix. Each schedule is a **separate** question; the random-mask prefix removes student guidance entirely, while the random-rescue prefix only perturbs ten student-selected patches. The switch epochs are exploratory choices from already inspected validation curves, not validated optimal points.

On the server in `aim-lab-test-2`, choose either command or run both if GPU memory allows:

```bash
git pull --ff-only origin main
mkdir -p logs
nohup .venv/bin/python -u run.py train --config configs/full_switch_pilot.json --seed 0 --init scratch --method random_to_student_10 > logs/random_to_student_10.log 2>&1 < /dev/null &
nohup .venv/bin/python -u run.py train --config configs/full_switch_pilot.json --seed 0 --init scratch --method random_rescue_to_student_20 > logs/random_rescue_to_student_20.log 2>&1 < /dev/null &
```

The config has zero DataLoader workers for safe parallel execution. Each run writes into its own `outputs/experiment2/seed_0/scratch/<method>/` folder. To check progress, use `tail -n 3 logs/random_to_student_10.log logs/random_rescue_to_student_20.log`.

After the chosen run finishes at `100/100`, evaluate and summarize **only that method**. For the random-mask schedule:

```bash
.venv/bin/python run.py evaluate --config configs/full_switch_pilot.json --seeds 0 --inits scratch --methods random_to_student_10
.venv/bin/python scripts/summarize_random_switch.py --variant mask
cat outputs/experiment2/analysis/random_mask_switch_seed_0_scratch.md
```

For the random-rescue schedule, replace the method with `random_rescue_to_student_20`, pass `--variant rescue`, and read `random_rescue_switch_seed_0_scratch.md`. The reports compare existing MaskedKD, the unswitched random baseline, and the switch run, including an epoch-10/20 prefix check and best/last test macro accuracy. This is an exploratory seed-0 result; new seeds are needed before claiming an effect.
