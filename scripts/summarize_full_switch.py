"""Small, read-only comparison of the full-to-masked pilot with existing runs."""
import argparse
import json
from pathlib import Path


METHODS = ("student", "full", "full_to_student_20", "full_to_student_50")


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/experiment2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init", choices=("scratch", "imagenet"), default="scratch")
    args = parser.parse_args()
    root = Path(args.output_root) / f"seed_{args.seed}" / args.init
    runs = {}
    for method in METHODS:
        directory = root / method
        runs[method] = (read(directory / "result.json"), read(directory / "history.json"),
                        read(directory / "test_metrics.json"))
    hashes = {(result["metadata_sha256"], result["teacher_sha256"], result["initial_model_sha256"])
              for result, _, _ in runs.values()}
    if len(hashes) != 1:
        raise ValueError("Dataset, teacher, or student initialization differs among compared runs")

    def percent(value):
        return f"{100 * value:.2f}"

    baseline = runs["student"][2]
    lines = [f"# Full KD → MaskedKD pilot · seed {args.seed} · {args.init}", "",
             "All four runs share the dataset, teacher checkpoint, and student initialization hashes.",
             "The two new runs use Full KD through epoch 20 or 50, then MaskedKD from the next epoch.",
             "Student input stays at 196 patches; only the teacher input changes from 196 to 98.", "",
             "| Method | Best epoch | Val best (%) | Val last (%) | Test best (%) | Test last (%) | Δ test last vs MaskedKD (pp) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method, (result, history, metrics) in runs.items():
        lines.append(f"| {method} | {result['best_epoch']} | {percent(result['best_validation_macro'])} | "
                     f"{percent(history[-1]['validation']['macro_accuracy'])} | "
                     f"{percent(metrics['best']['macro_accuracy'])} | {percent(metrics['last']['macro_accuracy'])} | "
                     f"{100 * (metrics['last']['macro_accuracy'] - baseline['last']['macro_accuracy']):+.2f} |")
    lines += ["", "Prefix check against the existing Full KD validation curve:", "",
              "| New run | Epoch | New val (%) | Full KD val (%) | Difference (pp) |",
              "|---|---:|---:|---:|---:|"]
    full_history = runs["full"][1]
    for method, epoch in (("full_to_student_20", 20), ("full_to_student_50", 50)):
        value = runs[method][1][epoch - 1]["validation"]["macro_accuracy"]
        reference = full_history[epoch - 1]["validation"]["macro_accuracy"]
        lines.append(f"| {method} | {epoch} | {percent(value)} | {percent(reference)} | {100 * (value - reference):+.2f} |")
    lines += ["", "This is an exploratory single-seed comparison. The test set has already been inspected in earlier experiments; use fresh seeds to confirm any apparent gain."]
    destination = Path(args.output_root) / "analysis" / f"full_switch_seed_{args.seed}_{args.init}.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
