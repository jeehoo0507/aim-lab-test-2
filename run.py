#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from coco_kd.config import Config, METHODS


def parser():
    p = argparse.ArgumentParser(description="COCO experiment 2; see README for staged setup/pilot/run/evaluate/export")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("prepare", "check", "benchmark", "train", "pipeline", "evaluate", "analyze", "export", "smoke"):
        q = sub.add_parser(name)
        q.add_argument("--config", default="configs/experiment2.json")
        q.add_argument("--data-root")
        q.add_argument("--output-root")
        q.add_argument("--device")
        if name in ("pipeline", "evaluate"):
            q.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
            q.add_argument("--inits", nargs="+", choices=["scratch", "imagenet"], default=["scratch", "imagenet"])
            q.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
        if name == "pipeline":
            q.add_argument("--jobs", choices=["1", "2", "auto"], default="1")
        if name == "benchmark":
            q.add_argument("--steps", type=int, default=8, help="Measured optimizer updates after warmup; accumulation included")
            q.add_argument("--warmup", type=int, default=2)
            q.add_argument("--max-jobs", type=int, choices=[1, 2], default=2)
        if name == "train":
            q.add_argument("--role", choices=["teacher", "student"], default="student")
            q.add_argument("--method", choices=METHODS, default="student")
            q.add_argument("--seed", type=int, default=0)
            q.add_argument("--init", choices=["scratch", "imagenet"], default="imagenet")
            q.add_argument("--stop-after", type=int)
        if name == "export":
            q.add_argument("--push", action="store_true", help="Commit only this report directory and git push; no auth changes")
    return p


def pipeline(cfg, args):
    from coco_kd.data import validate_manifest
    from coco_kd.system import preflight
    from coco_kd.train import train
    from coco_kd.utils import write_json
    manifest = validate_manifest(cfg.data_root)
    if len(manifest["classes"]) != cfg.num_classes:
        raise ValueError("Config class count does not match dataset")
    preflight(cfg, Path(cfg.output_root) / "preflight.json")
    if args.jobs == "auto":
        from coco_kd.benchmark import automatic_jobs
        args.jobs = automatic_jobs(cfg)
    else:
        args.jobs = int(args.jobs)
    # One job owns one seed, so teacher is never trained twice for different student initializations.
    def run_seed(seed):
        commands = [("teacher", "student", "imagenet")]
        commands += [("student", method, initialization) for initialization in args.inits for method in args.methods]
        for role, method, initialization in commands:
            if args.jobs == 1:
                train(replace(cfg, seed=seed, student_init=initialization), role=role, method=method)
            else:
                directory = Path(cfg.output_root) / "_jobs"
                directory.mkdir(parents=True, exist_ok=True)
                config_path = directory / f"seed_{seed}.json"
                write_json(config_path, cfg.to_dict())
                env = dict(os.environ, OMP_NUM_THREADS=str(cfg.num_threads), MKL_NUM_THREADS=str(cfg.num_threads))
                log = Path("logs") / f"seed_{seed}_{initialization}_{role}_{method}.log"
                log.parent.mkdir(exist_ok=True)
                with open(log, "a", buffering=1) as file:
                    subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "train", "--config", str(config_path),
                                    "--role", role, "--method", method, "--seed", str(seed), "--init", initialization],
                                   check=True, stdout=file, stderr=subprocess.STDOUT, env=env)
                print(f"DONE seed={seed} {role} {initialization} {method}; log={log}", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(run_seed, args.seeds))
    from coco_kd.analysis import analyze
    analyze(cfg.output_root)
    print("Training complete. Run evaluate after freezing the protocol, then export --push.")


def main():
    args = parser().parse_args()
    cfg = Config.load(args.config, data_root=args.data_root, output_root=args.output_root, device=args.device)
    if args.command == "prepare":
        from coco_kd.prepare import prepare
        prepare(cfg.data_root)
    elif args.command == "check":
        from coco_kd.data import validate_manifest
        from coco_kd.system import preflight
        preflight(cfg, Path(cfg.output_root) / "preflight.json")
        if (Path(cfg.data_root) / "manifest.json").exists():
            validate_manifest(cfg.data_root)
            print("PASSED dataset hashes/splits/probe validation")
    elif args.command == "benchmark":
        from coco_kd.benchmark import benchmark
        benchmark(cfg, Path(cfg.output_root) / "benchmark.json", steps=args.steps, warmup=args.warmup, max_jobs=args.max_jobs)
    elif args.command == "train":
        from coco_kd.train import train
        train(replace(cfg, seed=args.seed, student_init=args.init), role=args.role, method=args.method, stop_after=args.stop_after)
    elif args.command == "pipeline":
        pipeline(cfg, args)
    elif args.command == "evaluate":
        from coco_kd.train import evaluate_test
        for seed in args.seeds:
            evaluate_test(replace(cfg, seed=seed, student_init="imagenet"), role="teacher")
            for initialization in args.inits:
                for method in args.methods:
                    evaluate_test(replace(cfg, seed=seed, student_init=initialization), method=method)
    elif args.command == "analyze":
        from coco_kd.analysis import analyze
        print(analyze(cfg.output_root))
    elif args.command == "export":
        from coco_kd.export import export_results
        export_results(cfg, push=args.push)
    elif args.command == "smoke":
        from coco_kd.synthetic import smoke
        smoke(args.output_root or "outputs/smoke", device=args.device or "cpu")


if __name__ == "__main__":
    main()
