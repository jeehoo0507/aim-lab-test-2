"""DeiT-Base teacher pilot: one teacher, three paired scratch students per seed."""
import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from .config import Config
from .data import validate_manifest
from .models import build_model
from .train import evaluate_test, run_path
from .utils import (RunLock, compatible_config, metadata_hash, model_fingerprint,
                    seed_all, setup_device, sha256, source_fingerprint, write_json)

ROOT = Path(__file__).resolve().parent.parent
METHODS = ("full", "student", "random_rescue_10")
SMALL_REPORT = "reports/20260923_152152_732466"


def check_small_baselines(cfg, seeds, small_root):
    """Pair with frozen Small reports without needing their server-side weights."""
    digest = metadata_hash(cfg)
    records = {}
    for seed in seeds:
        current = replace(cfg, seed=seed)
        seed_all(seed)
        model = build_model("student", current)
        initial = model_fingerprint(model)
        del model
        teachers = set()
        for method in METHODS:
            path = Path(small_root) / f"seed_{seed}/scratch/{method}/result.json"
            result = json.loads(path.read_text())
            saved = Config.load(**result["config"])
            if saved.teacher_variant != "small":
                raise ValueError(f"Expected a Small teacher baseline: {path}")
            compatible_config(replace(saved, teacher_variant="base").to_dict(), current.to_dict())
            if (result["seed"] != seed or result["method"] != method or result["student_init"] != "scratch"
                    or result["last_epoch"] != cfg.epochs or result["partial_training"]):
                raise ValueError(f"Baseline is incomplete or has the wrong identity: {path}")
            if result["metadata_sha256"] != digest or result["initial_model_sha256"] != initial:
                raise ValueError(f"Baseline data/student initialization mismatch: {path}")
            teachers.add(result["teacher_sha256"])
            records[(seed, method)] = result
        if len(teachers) != 1:
            raise ValueError(f"Small baselines do not share one teacher: seed={seed}")
    return records


def prepare_protocol(cfg, seeds, small_root):
    if cfg.teacher_variant != "base" or cfg.student_init != "scratch":
        raise ValueError("This pilot requires teacher_variant=base and student_init=scratch")
    if cfg.num_workers != 0 or cfg.max_train_batches is not None:
        raise ValueError("Use num_workers=0 and complete epochs for the three-process pilot")
    if cfg.model_scale == "deit" and (cfg.epochs != 100 or cfg.teacher_epochs != 30 or not cfg.teacher_pretrained):
        raise ValueError("The real-data pilot uses an ImageNet teacher, 30 teacher and 100 student epochs")
    manifest = validate_manifest(cfg.data_root)
    if len(manifest["classes"]) != cfg.num_classes:
        raise ValueError("Dataset class count differs from config")
    baseline = check_small_baselines(cfg, seeds, small_root)
    root = Path(cfg.output_root)
    path = root / "comparison_protocol.json"
    protocol = {"schema": 1, "study": "teacher_base_pilot", "config": cfg.to_dict(),
                "seeds": seeds, "methods": list(METHODS), "parallel_jobs": 3,
                "small_report": str(Path(small_root).resolve()), "metadata_sha256": metadata_hash(cfg),
                "code_sha256": source_fingerprint(),
                "small_baselines": [{"seed": seed, "method": method,
                    "teacher_sha256": record["teacher_sha256"],
                    "initial_model_sha256": record["initial_model_sha256"]}
                    for (seed, method), record in baseline.items()],
                "comparison": "epoch-100 test macro; report best separately; exploratory existing test set",
                "teacher_inputs": {"full": 196, "student": 98, "random_rescue_10": 98}}
    if path.exists():
        saved = json.loads(path.read_text())
        compatible_config(saved["config"], protocol["config"])
        for key in ("study", "seeds", "methods", "metadata_sha256", "code_sha256", "small_baselines"):
            if saved[key] != protocol[key]:
                raise ValueError(f"Existing pilot {key} changed; choose a new output root")
    elif any(root.glob("seed_*")):
        raise ValueError("Output root already contains another study; choose a new output root")
    write_json(path, protocol)
    return baseline


def required_memory(rows, total_gib):
    # Separate processes each own their teacher and CUDA context.
    return sum(row["peak_reserved_gib"] + .75 for row in rows) + max(2, total_gib * .1)


def require_memory(cfg, required):
    if cfg.device.startswith("cuda"):
        import torch
        free, _ = torch.cuda.mem_get_info(setup_device(cfg))
        print(f"GPU free={free / 1024**3:.2f} GiB; required with headroom={required:.2f} GiB", flush=True)
        if free / 1024**3 < required:
            raise RuntimeError("Insufficient free VRAM; the next pilot stage has not started")


def task_seconds(cfg, row, train_count):
    epochs = cfg.teacher_epochs if row["method"] == "teacher" else cfg.epochs
    ordinary = row["seconds"] * train_count / row["samples"]
    ordinary += row["validation_seconds"] + row["checkpoint_seconds"] + row.get("probe_seconds", 0)
    details = len(set(e for e in cfg.diagnostic_epochs if 0 <= e <= cfg.epochs) | {0, cfg.epochs}) + 1
    return epochs * ordinary + details * max(0, row.get("diagnostic_seconds", 0) - row.get("probe_seconds", 0))


def benchmark_pilot(cfg, seeds, warmup=2, steps=8):
    """Measure the Base architecture, including a mixed Full/Masked/Random wave."""
    import torch
    from .benchmark import _hardware, run_trial
    device = setup_device(cfg)
    single = run_trial(cfg, 1, warmup=warmup, steps=steps, modes=("teacher", *METHODS), measure_all=True)
    teacher = next(r for r in single["rows"] if r["method"] == "teacher")
    students = [r for r in single["rows"] if r["method"] in METHODS]
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3 if device.type == "cuda" else 0
    if device.type == "cuda":
        require_memory(cfg, required_memory(students, total))
    parallel = run_trial(cfg, 3, warmup=warmup, steps=steps, parallel_methods=METHODS)
    teacher_memory = required_memory([teacher], total) if device.type == "cuda" else None
    student_memory = max(required_memory(students, total), required_memory(parallel["rows"], total)) if device.type == "cuda" else None
    train_count = sum(r["split"] == "train" for r in json.loads((Path(cfg.data_root) / "manifest.json").read_text())["images"])
    teacher_hours = task_seconds(cfg, teacher, train_count) / 3600
    student_hours = max(task_seconds(cfg, row, train_count) for row in parallel["rows"]) / 3600
    total_hours = len(seeds) * (teacher_hours + student_hours)
    report = {"schema": 1, "study": "teacher_base_pilot", "config": cfg.to_dict(),
              "hardware": _hardware(device), "seeds": seeds,
              "metadata_sha256": metadata_hash(cfg), "code_sha256": source_fingerprint(),
              "warmup_updates": warmup, "measured_updates": steps, "trials": [single, parallel],
              "required_free_gib": {"teacher": teacher_memory, "three_students": student_memory},
              "estimate": {"teacher_hours_per_seed": teacher_hours,
                           "three_students_wall_hours_per_seed": student_hours,
                           "full_plan_hours": total_hours, "planning_range_hours": [total_hours, total_hours * 1.5],
                           "note": "Short measured extrapolation, not a confidence interval; excludes download, final test/export; reruns may have less work remaining"},
              "gpu_estimate": device.type == "cuda"}
    write_json(Path(cfg.output_root) / "benchmark.json", report)
    print(json.dumps({k: report[k] for k in ("hardware", "required_free_gib", "estimate", "gpu_estimate")}, indent=2), flush=True)
    return report


def train_task(cfg, seed, role, method, config_path):
    log = Path(cfg.output_root) / "_jobs" / f"seed_{seed}_{role}_{method}.log"
    print(f"START seed={seed} {role} {method}; log={log}", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(cfg.num_threads), MKL_NUM_THREADS=str(cfg.num_threads))
    with log.open("a", buffering=1) as stream:
        subprocess.run([sys.executable, "-u", str(ROOT / "run.py"), "train", "--config", str(config_path),
                        "--seed", str(seed), "--role", role, "--method", method,
                        "--init", "imagenet" if role == "teacher" else "scratch"],
                       check=True, stdout=stream, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    print(f"DONE seed={seed} {role} {method}", flush=True)


def require_completed(cfg, seeds, baseline):
    """Validate all runs before opening test, including the shared Base teacher."""
    for seed in seeds:
        current = replace(cfg, seed=seed)
        teacher_path = run_path(current, "teacher")
        teacher = json.loads((teacher_path / "result.json").read_text())
        compatible_config(teacher["config"], replace(current, student_init="imagenet").to_dict())
        if (teacher["role"] != "teacher" or teacher["last_epoch"] != cfg.teacher_epochs
                or teacher["partial_training"] or teacher["metadata_sha256"] != metadata_hash(cfg)):
            raise ValueError(f"Incomplete or mismatched teacher: {teacher_path}")
        teacher_hash = sha256(teacher_path / "best.pt")
        for method in METHODS:
            path = run_path(current, method=method)
            result = json.loads((path / "result.json").read_text())
            compatible_config(result["config"], current.to_dict())
            if (result["method"] != method or result["last_epoch"] != cfg.epochs or result["partial_training"]
                    or result["teacher_sha256"] != teacher_hash
                    or result["initial_model_sha256"] != baseline[(seed, method)]["initial_model_sha256"]
                    or result["metadata_sha256"] != metadata_hash(cfg)):
                raise ValueError(f"Incomplete or mismatched student: {path}")


def compare_sizes(cfg, seeds, small_root):
    rows = []
    interactions = []
    for seed in seeds:
        scores = {}
        for size, root in (("small", Path(small_root)), ("base", Path(cfg.output_root))):
            for method in METHODS:
                metrics = json.loads((root / f"seed_{seed}/scratch/{method}/test_metrics.json").read_text())
                scores[size, method] = metrics["last"]["macro_accuracy"]
                rows.append({"seed": seed, "teacher": size, "method": method,
                             "last_macro_accuracy": scores[size, method],
                             "best_macro_accuracy": metrics["best"]["macro_accuracy"]})
        for row in rows[-6:]:
            row["delta_vs_masked_pp"] = 100 * (row["last_macro_accuracy"] - scores[row["teacher"], "student"])
            row["base_minus_small_pp"] = 100 * (scores["base", row["method"]] - scores["small", row["method"]])
        small_gain = 100 * (scores["small", "random_rescue_10"] - scores["small", "student"])
        base_gain = 100 * (scores["base", "random_rescue_10"] - scores["base", "student"])
        interactions.append({"seed": seed, "random10_gain_small_pp": small_gain,
                             "random10_gain_base_pp": base_gain, "change_in_random10_gain_pp": base_gain - small_gain})
    destination = Path(cfg.output_root) / "analysis"
    destination.mkdir(exist_ok=True, parents=True)
    for name, data in (("teacher_size_comparison.csv", rows), ("teacher_size_interaction.csv", interactions)):
        with (destination / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    return destination / "teacher_size_comparison.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "benchmark", "evaluate", "export"))
    parser.add_argument("--config", default="configs/teacher_base_pilot.json")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--device")
    parser.add_argument("--small-root", default=SMALL_REPORT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--push", action="store_true", help="Only with export: commit/push the new report")
    args = parser.parse_args()
    if args.warmup < 0 or args.steps < 1 or any(s < 0 for s in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use nonnegative unique seeds/warmup and positive measured steps")
    if args.push and args.action != "export":
        parser.error("--push is only supported by export")
    os.chdir(ROOT)
    cfg = Config.load(args.config, data_root=args.data_root, output_root=args.output_root, device=args.device)
    with RunLock(Path(cfg.output_root) / ".teacher_base.lock"):
        baseline = prepare_protocol(cfg, args.seeds, args.small_root)
        if args.action in ("benchmark", "run"):
            report = benchmark_pilot(cfg, args.seeds, args.warmup, args.steps)
            if args.action == "benchmark":
                return
            if cfg.teacher_pretrained:
                print("Preparing cached official ImageNet DeiT-Base weights", flush=True)
                model = build_model("teacher", cfg, pretrained=True)
                del model
            config_path = Path(cfg.output_root).resolve() / "_jobs/config.json"
            write_json(config_path, cfg.to_dict())
            # Teachers are trained one at a time, then each seed's three students
            # share the frozen checkpoint through independent model instances.
            for seed in args.seeds:
                require_memory(cfg, report["required_free_gib"]["teacher"])
                train_task(cfg, seed, "teacher", "student", config_path)
            for seed in args.seeds:
                require_memory(cfg, report["required_free_gib"]["three_students"])
                with ThreadPoolExecutor(max_workers=3) as pool:
                    futures = [pool.submit(train_task, cfg, seed, "student", method, config_path) for method in METHODS]
                    for future in futures:
                        future.result()
            require_completed(cfg, args.seeds, baseline)
            from .analysis import analyze
            print(f"COMPLETE {3 * len(args.seeds)} students; validation analysis: {analyze(cfg.output_root)}", flush=True)
            print("Next: bash setup.sh teacher-base-evaluate (repeat the same path/seed overrides)", flush=True)
        else:
            require_completed(cfg, args.seeds, baseline)
            if args.action == "evaluate":
                # Read baseline scores only after all training is complete.
                for seed in args.seeds:
                    evaluate_test(replace(cfg, seed=seed, student_init="imagenet"), role="teacher")
                    for method in METHODS:
                        evaluate_test(replace(cfg, seed=seed), method=method)
            from .analysis import analyze
            analyze(cfg.output_root)
            print(f"Teacher-size comparison: {compare_sizes(cfg, args.seeds, args.small_root)}", flush=True)
            if args.action == "export":
                from .export import export_results
                export_results(cfg, push=args.push)


if __name__ == "__main__":
    main()
