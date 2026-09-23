"""Summarize the four fixed switch schedules against existing Experiment 2 runs."""
import argparse
import json
from pathlib import Path


SCHEDULES = (("random_to_student_20", "random", 20),
             ("random_to_student_50", "random", 50),
             ("full_to_student_20", "full", 20),
             ("full_to_student_50", "full", 50))
METHODS = ("student", "random", "full", *(item[0] for item in SCHEDULES))


def read(path):
    return json.loads(path.read_text())


def pct(value):
    return f"{100 * value:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/experiment2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init", choices=("scratch", "imagenet"), default="scratch")
    args = parser.parse_args()
    root = Path(args.output_root) / f"seed_{args.seed}" / args.init
    runs = {}
    for method in METHODS:
        path = root / method
        runs[method] = (read(path / "result.json"), read(path / "history.json"),
                        read(path / "test_metrics.json"))
    provenance = {(result["metadata_sha256"], result["teacher_sha256"], result["initial_model_sha256"])
                  for result, _, _ in runs.values()}
    if len(provenance) != 1:
        raise ValueError("Dataset, teacher, or initial student model differs between compared runs")
    masked_last = runs["student"][2]["last"]["macro_accuracy"]
    lines = [f"# Random / Full → MaskedKD · seed {args.seed} · {args.init}", "",
             "Four new 100-epoch runs; teacher and original baselines are reused. Optimizer and learning-rate schedule do not reset at the switch.",
             "All runs share dataset, teacher, and initial student weights. Random keeps 98 teacher patches throughout; Full changes from 196 to 98.", "",
             "| Method | Best epoch | Val best (%) | Val last (%) | Test best (%) | Test last (%) | Δ last vs MaskedKD (pp) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method, (result, history, test) in runs.items():
        if len(history) != 100:
            raise ValueError(f"Expected 100 epochs for {method}, found {len(history)}")
        lines.append(f"| {method} | {result['best_epoch']} | {pct(result['best_validation_macro'])} | "
                     f"{pct(history[-1]['validation']['macro_accuracy'])} | {pct(test['best']['macro_accuracy'])} | "
                     f"{pct(test['last']['macro_accuracy'])} | "
                     f"{100 * (test['last']['macro_accuracy'] - masked_last):+.2f} |")
    lines += ["", "## Prefix validation check", "",
              "The new run should agree with its existing unswitched baseline at the switch epoch.", "",
              "| Scheduled run | Switch after epoch | New val (%) | Existing prefix val (%) | Difference (pp) |",
              "|---|---:|---:|---:|---:|"]
    for method, prefix, switch in SCHEDULES:
        value = runs[method][1][switch - 1]["validation"]["macro_accuracy"]
        reference = runs[prefix][1][switch - 1]["validation"]["macro_accuracy"]
        lines.append(f"| {method} | {switch} | {pct(value)} | {pct(reference)} | "
                     f"{100 * (value - reference):+.2f} |")
    lines += ["", "This is an exploratory single-seed comparison on a previously inspected test set. Confirm any apparent gain with fresh seeds."]
    destination = Path(args.output_root) / "analysis" / f"switch_grid_seed_{args.seed}_{args.init}.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
