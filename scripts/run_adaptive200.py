#!/usr/bin/env python3
"""Parallel 200-epoch follow-up; stages the already trained frozen teachers."""
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


def cli():
    p = argparse.ArgumentParser(description="Equal-200-epoch Random10 reliability-gate pilot")
    p.add_argument("action", choices=("run", "evaluate", "analyze"))
    p.add_argument("--config", default="configs/adaptive200.json")
    p.add_argument("--output-root")
    p.add_argument("--teacher-root", default="outputs/experiment2")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--inits", nargs="+", choices=("scratch", "imagenet"), default=["scratch", "imagenet"])
    p.add_argument("--jobs", type=int, default=4)
    return p.parse_args()


def require_completed(cfg, seeds, inits):
    for seed in seeds:
        for init in inits:
            for method in METHODS:
                path = Path(cfg.output_root) / f"seed_{seed}" / init / method / "result.json"
                if not path.exists():
                    raise ValueError(f"Incomplete study; no test evaluation yet: {path}")
                result = json.loads(path.read_text())
                if result["last_epoch"] != cfg.epochs or result["partial_training"]:
                    raise ValueError(f"Partial study; no test evaluation yet: {path}")


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
    if cfg.epochs != 200 or not cfg.validation_exclude_probe:
        raise ValueError("Use a 200-epoch config with gate images excluded from checkpoint validation")
    if args.action == "analyze":
        from coco_kd.analysis import analyze
        print(analyze(cfg.output_root))
        return
    if args.action == "evaluate":
        from coco_kd.train import evaluate_test
        require_completed(cfg, args.seeds, args.inits)
        for seed in args.seeds:
            for init in args.inits:
                for method in METHODS:
                    evaluate_test(replace(cfg, seed=seed, student_init=init), method=method)
        from coco_kd.analysis import analyze
        print(analyze(cfg.output_root))
        return

    manifest = validate_manifest(cfg.data_root)
    if len(manifest["classes"]) != cfg.num_classes:
        raise ValueError("Wrong dataset for configured model")
    check_gpu_budget(cfg, args.teacher_root, args.jobs)
    from coco_kd.models import build_model
    if "imagenet" in args.inits:
        pretrained_student = build_model("student", cfg, pretrained=True)
        del pretrained_student
    sources = []
    for seed in args.seeds:
        sources.append(stage_teacher(replace(cfg, seed=seed), args.teacher_root))
    protocol = {"schema": 1, "config": cfg.to_dict(), "seeds": args.seeds, "inits": args.inits,
                "methods": list(METHODS), "teacher_sources": sources,
                "training_budget": "one student full-196 and one frozen teacher 98-patch pass per training batch",
                "gate": "fixed 200-image validation probe every 10 epochs; two consecutive favorable probes "
                        "after epoch 30; switch takes effect next epoch",
                "evaluation": "test only after all requested runs complete; previously inspected test is exploratory"}
    protocol_path = Path(cfg.output_root) / "adaptive_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Protocol changed; choose a new output root")
    write_json(protocol_path, protocol)
    jobs_dir = Path(cfg.output_root) / "_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    task_config = jobs_dir / "config.json"
    write_json(task_config, cfg.to_dict())
    tasks = [(seed, init, method) for seed in args.seeds for init in args.inits for method in METHODS]
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
    require_completed(cfg, args.seeds, args.inits)
    from coco_kd.analysis import analyze
    print(f"COMPLETE {len(tasks)} students; validation analysis: {analyze(cfg.output_root)}", flush=True)
    print("Run evaluate only after the protocol is frozen.", flush=True)


if __name__ == "__main__":
    main()
