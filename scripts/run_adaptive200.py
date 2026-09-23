#!/usr/bin/env python3
"""Parallel adaptive masking follow-up; reuse trained experiment-2 teachers."""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coco_kd.adaptive import ADAPTIVE_TARGETS
from coco_kd.anneal import stage_teacher
from coco_kd.config import Config
from coco_kd.data import validate_manifest
from coco_kd.system import storage_report
from coco_kd.utils import write_json

METHODS = ("student", "random_rescue_10", "low_score_rescue_10", *ADAPTIVE_TARGETS)
BASELINES = ("student", "random_rescue_10", "low_score_rescue_10")
MATCHED_TRAINING_KEYS = ("num_classes", "batch_size", "eval_batch_size", "accumulation_steps",
                         "epochs", "teacher_epochs", "scratch_lr", "pretrained_lr", "teacher_lr",
                         "min_lr", "warmup_epochs", "weight_decay", "drop_path", "label_smoothing",
                         "kd_alpha", "temperature", "keep_tokens", "foreground_threshold",
                         "teacher_pretrained", "amp", "grad_clip", "model_scale", "max_train_batches")


def cli():
    p = argparse.ArgumentParser(description="Random10 reliability-gate pilot")
    p.add_argument("action", choices=("run", "evaluate", "analyze"))
    p.add_argument("--config", default="configs/adaptive200.json")
    p.add_argument("--output-root")
    p.add_argument("--teacher-root", default="outputs/experiment2")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--inits", nargs="+", choices=("scratch", "imagenet"), default=["scratch", "imagenet"])
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    return p.parse_args()


def require_completed(cfg, seeds, inits, methods):
    for seed in seeds:
        for init in inits:
            for method in methods:
                path = Path(cfg.output_root) / f"seed_{seed}" / init / method / "result.json"
                if not path.exists():
                    raise ValueError(f"Incomplete study; no test evaluation yet: {path}")
                result = json.loads(path.read_text())
                if result["last_epoch"] != cfg.epochs or result["partial_training"]:
                    raise ValueError(f"Partial study; no test evaluation yet: {path}")


def check_existing_baselines(cfg, baseline_root, seeds, inits):
    """Check the old 100-epoch comparison before launching new GPU work."""
    from coco_kd.utils import metadata_hash
    checked = {}
    for seed in seeds:
        for init in inits:
            current = replace(cfg, seed=seed, student_init=init).to_dict()
            records = {}
            for method in BASELINES:
                path = Path(baseline_root) / f"seed_{seed}" / init / method / "result.json"
                if not path.is_file():
                    raise FileNotFoundError(f"Existing 100-epoch baseline required: {path}")
                record = json.loads(path.read_text())
                if record["last_epoch"] != cfg.epochs or record["partial_training"]:
                    raise ValueError(f"Baseline must be complete at epoch {cfg.epochs}: {path}")
                if record["seed"] != seed or record["student_init"] != init or record["method"] != method:
                    raise ValueError(f"Baseline identity mismatch: {path}")
                for key in MATCHED_TRAINING_KEYS:
                    if record["config"][key] != current[key]:
                        raise ValueError(f"Baseline {key} mismatch: {path}")
                if record["metadata_sha256"] != metadata_hash(replace(cfg, seed=seed)):
                    raise ValueError(f"Baseline dataset mismatch: {path}")
                records[method] = record
            for key in ("teacher_sha256", "initial_model_sha256", "metadata_sha256"):
                if len({record[key] for record in records.values()}) != 1:
                    raise ValueError(f"Baseline {key} differs across methods: seed={seed} init={init}")
            checked[(seed, init)] = records
    return checked


def compare_quick(cfg, baseline_root, seeds, inits, methods):
    """Last-checkpoint-only comparison; reject unmatched teacher/init provenance."""
    baseline = check_existing_baselines(cfg, baseline_root, seeds, inits)
    rows = []
    for seed in seeds:
        for init in inits:
            old = baseline[(seed, init)]
            original_scores = {}
            for method in BASELINES:
                path = Path(baseline_root) / f"seed_{seed}" / init / method / "test_metrics.json"
                if not path.is_file():
                    raise FileNotFoundError(f"Baseline test metrics required: {path}")
                original_scores[method] = json.loads(path.read_text())["last"]["macro_accuracy"]
            for method in BASELINES:
                rows.append({"seed": seed, "initialization": init, "method": method,
                             "last_macro_accuracy": original_scores[method], "switch_after_epoch": None,
                             "delta_vs_masked_pp": 100 * (original_scores[method] - original_scores["student"]),
                             "delta_vs_random10_pp": 100 * (original_scores[method] - original_scores["random_rescue_10"])})
            for method in methods:
                path = Path(cfg.output_root) / f"seed_{seed}" / init / method
                result = json.loads((path / "result.json").read_text())
                for key in ("teacher_sha256", "initial_model_sha256", "metadata_sha256"):
                    if result[key] != old["student"][key]:
                        raise ValueError(f"New run {key} does not match old baselines: {path}")
                score = json.loads((path / "test_metrics.json").read_text())["last"]["macro_accuracy"]
                rows.append({"seed": seed, "initialization": init, "method": method,
                             "last_macro_accuracy": score,
                             "switch_after_epoch": (result.get("gate_state") or {}).get("switched_after_epoch"),
                             "delta_vs_masked_pp": 100 * (score - original_scores["student"]),
                             "delta_vs_random10_pp": 100 * (score - original_scores["random_rescue_10"])})
    import csv
    destination = Path(cfg.output_root) / "analysis" / "quick_comparison.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def check_gpu_budget(cfg, teacher_root, jobs):
    if not cfg.device.startswith("cuda"):
        return
    import torch
    benchmark_path = Path(teacher_root) / "benchmark.json"
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Need measured GPU parallel budget: {benchmark_path}")
    benchmark = json.loads(benchmark_path.read_text())
    device = torch.device(cfg.device)
    name = torch.cuda.get_device_name(device)
    measured = benchmark["hardware"]["name"]
    if name != measured:
        raise ValueError(f"GPU differs from benchmark ({name} versus {measured}); benchmark this GPU first")
    required = benchmark["required_free_gib_by_jobs"].get(str(jobs))
    if required is None:
        raise ValueError(f"No measured parallel budget for {jobs} jobs")
    free, _ = torch.cuda.mem_get_info(device)
    print(f"GPU free={free / 1024**3:.2f} GiB; required={required:.2f} GiB for {jobs} jobs", flush=True)
    if free / 1024**3 < required:
        raise RuntimeError("Not enough measured free GPU memory for parallel run")


def main():
    os.chdir(ROOT)
    args = cli()
    if args.jobs not in range(1, 8):
        raise ValueError("--jobs must be 1 through 7")
    cfg = Config.load(args.config, output_root=args.output_root)
    if cfg.epochs == 200 and not cfg.validation_exclude_probe:
        raise ValueError("The 200-epoch protocol requires probe images excluded from checkpoint validation")
    if cfg.epochs not in (100, 200):
        raise ValueError("Use a fixed 100- or 200-epoch pilot config")
    methods = tuple(dict.fromkeys(args.methods))
    if cfg.epochs == 100:
        check_existing_baselines(cfg, args.teacher_root, args.seeds, args.inits)
    if args.action == "analyze":
        from coco_kd.analysis import analyze
        print(analyze(cfg.output_root))
        return
    if args.action == "evaluate":
        from coco_kd.train import evaluate_test
        require_completed(cfg, args.seeds, args.inits, methods)
        for seed in args.seeds:
            for init in args.inits:
                for method in methods:
                    evaluate_test(replace(cfg, seed=seed, student_init=init), method=method)
        from coco_kd.analysis import analyze
        print(analyze(cfg.output_root))
        if cfg.epochs == 100:
            print(f"Matched last-checkpoint comparison: {compare_quick(cfg, args.teacher_root, args.seeds, args.inits, methods)}")
        return

    manifest = validate_manifest(cfg.data_root)
    if len(manifest["classes"]) != cfg.num_classes:
        raise ValueError("Wrong dataset for configured model")
    check_gpu_budget(cfg, args.teacher_root, args.jobs)
    from coco_kd.models import build_model
    from coco_kd.utils import model_fingerprint, seed_all
    if "imagenet" in args.inits:
        pretrained_student = build_model("student", cfg, pretrained=True)
        del pretrained_student
    sources = []
    for seed in args.seeds:
        sources.append(stage_teacher(replace(cfg, seed=seed), args.teacher_root))
    if cfg.epochs == 100:
        baseline = check_existing_baselines(cfg, args.teacher_root, args.seeds, args.inits)
        for source in sources:
            for init in args.inits:
                if source["sha256"] != baseline[(source["seed"], init)]["student"]["teacher_sha256"]:
                    raise ValueError("Staged teacher does not match existing 100-epoch baselines")
                seed_all(source["seed"])
                fresh = build_model("student", replace(cfg, seed=source["seed"], student_init=init),
                                    pretrained=init == "imagenet")
                if model_fingerprint(fresh) != baseline[(source["seed"], init)]["student"]["initial_model_sha256"]:
                    raise ValueError(f"Student initialization differs from baselines: seed={source['seed']} init={init}")
                del fresh
    protocol = {"schema": 1, "config": cfg.to_dict(), "seeds": args.seeds, "inits": args.inits,
                "methods": list(methods), "teacher_sources": sources,
                "training_budget": "one student full-196 and one frozen teacher 98-patch pass per training batch",
                "gate": "fixed 200-image validation probe every 10 epochs; two consecutive favorable probes "
                        "after epoch 30; switch takes effect next epoch",
                "evaluation": "test only after all requested runs complete; previously inspected test is exploratory",
                "baseline_root": args.teacher_root if cfg.epochs == 100 else None,
                "comparison": "epoch-100 last checkpoints only" if cfg.epochs == 100 else "epoch-200 within-study"}
    protocol_path = Path(cfg.output_root) / "adaptive_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Protocol changed; choose a new output root")
    write_json(protocol_path, protocol)
    jobs_dir = Path(cfg.output_root) / "_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    task_config = jobs_dir / "config.json"
    write_json(task_config, cfg.to_dict())
    tasks = [(seed, init, method) for seed in args.seeds for init in args.inits for method in methods]
    student = build_model("student", cfg)
    snapshots = (cfg.epochs - 1) // cfg.checkpoint_every + 2
    estimated_weights_gb = sum(p.numel() for p in student.parameters()) * 4 * snapshots * len(tasks) / 1e9
    del student
    write_json(Path(cfg.output_root) / "preflight.json", {
        "student_tasks": len(tasks), "parallel_jobs": args.jobs,
        "estimated_completed_student_weight_gb": estimated_weights_gb,
        "excludes": "teacher copies, resume optimizer state, probe arrays, logs, environment and dataset",
        "storage": storage_report({"outputs": cfg.output_root, "data": cfg.data_root})})
    print(f"Estimated completed student weights: {estimated_weights_gb:.2f} GB across {len(tasks)} runs", flush=True)

    def run_one(task):
        seed, init, method = task
        log = jobs_dir / f"seed_{seed}_{init}_{method}.log"
        print(f"START seed={seed} init={init} method={method}; log={log}", flush=True)
        env = dict(os.environ, OMP_NUM_THREADS=str(cfg.num_threads), MKL_NUM_THREADS=str(cfg.num_threads))
        with log.open("a", buffering=1) as stream:
            subprocess.run([sys.executable, "-u", "run.py", "train", "--config", str(task_config),
                            "--seed", str(seed), "--init", init, "--method", method],
                           check=True, stdout=stream, stderr=subprocess.STDOUT, env=env)
        print(f"DONE seed={seed} init={init} method={method}", flush=True)

    with ThreadPoolExecutor(max_workers=min(args.jobs, len(tasks))) as pool:
        list(pool.map(run_one, tasks))
    require_completed(cfg, args.seeds, args.inits, methods)
    from coco_kd.analysis import analyze
    print(f"COMPLETE {len(tasks)} students; validation analysis: {analyze(cfg.output_root)}", flush=True)
    print("Run evaluate only after the protocol is frozen.", flush=True)


if __name__ == "__main__":
    main()
