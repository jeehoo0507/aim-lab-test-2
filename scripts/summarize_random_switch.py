"""Compare one random-to-MaskedKD schedule with its existing controls."""
import argparse
import json
from pathlib import Path


SCHEDULES = {"mask": ("random", "random_to_student_10", 10),
             "rescue": ("random_rescue_10", "random_rescue_to_student_20", 20)}


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=SCHEDULES, required=True)
    parser.add_argument("--output-root", default="outputs/experiment2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init", choices=("scratch", "imagenet"), default="scratch")
    args = parser.parse_args()
    prefix, scheduled, switch = SCHEDULES[args.variant]
    root = Path(args.output_root) / f"seed_{args.seed}" / args.init
    methods = ("student", prefix, scheduled)
    runs = {}
    for method in methods:
        directory = root / method
        runs[method] = (read(directory / "result.json"), read(directory / "history.json"),
                        read(directory / "test_metrics.json"))
    hashes = {(result["metadata_sha256"], result["teacher_sha256"], result["initial_model_sha256"])
              for result, _, _ in runs.values()}
    if len(hashes) != 1:
        raise ValueError("Dataset, teacher, or student initialization differs among compared runs")

    def pct(value):
        return f"{100 * value:.2f}"

    baseline = runs["student"][2]
    lines = [f"# {prefix} → MaskedKD · seed {args.seed} · {args.init}", "",
             f"Teacher sees 98 patches throughout. Switch after epoch {switch}, without resetting optimizer or learning-rate schedule.",
             "All three runs share dataset, teacher checkpoint, and student initialization hashes.", "",
             "| Method | Best epoch | Val best (%) | Val last (%) | Test best (%) | Test last (%) | Δ test last vs MaskedKD (pp) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method, (result, history, metrics) in runs.items():
        lines.append(f"| {method} | {result['best_epoch']} | {pct(result['best_validation_macro'])} | "
                     f"{pct(history[-1]['validation']['macro_accuracy'])} | "
                     f"{pct(metrics['best']['macro_accuracy'])} | {pct(metrics['last']['macro_accuracy'])} | "
                     f"{100 * (metrics['last']['macro_accuracy'] - baseline['last']['macro_accuracy']):+.2f} |")
    value = runs[scheduled][1][switch - 1]["validation"]["macro_accuracy"]
    reference = runs[prefix][1][switch - 1]["validation"]["macro_accuracy"]
    lines += ["", f"Prefix check at epoch {switch}: scheduled val {pct(value)}%, existing {prefix} val {pct(reference)}% "
              f"(difference {100 * (value - reference):+.2f} pp).", "",
              "Single-seed exploratory comparison on a previously inspected test set. Confirm any gain on new seeds."]
    destination = Path(args.output_root) / "analysis" / f"random_{args.variant}_switch_seed_{args.seed}_{args.init}.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
