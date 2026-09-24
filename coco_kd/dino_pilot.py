"""DINO saliency controls for the COCO teacher-Base pilot; no SSL retraining."""
import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from .anneal import stage_teacher
from .config import Config
from .data import validate_manifest
from .dino_attention import DINO_METHODS, DINO_SHA256, DINO_URL, build_selector
from .models import build_model
from .teacher_base import required_memory, require_memory, task_seconds, train_task
from .train import evaluate_test, run_path
from .utils import (RunLock, compatible_config, load_checkpoint, metadata_hash,
                    model_fingerprint, seed_all, sha256, source_fingerprint, write_json)

ROOT = Path(__file__).resolve().parent.parent
BASELINE_METHODS = ("full", "student", "random_rescue_10")
DEFAULT_BASELINE = "reports/20260924_204055_692914"


def prepare_protocol(cfg, teacher_root, baseline_root, seeds):
    root, source = Path(cfg.output_root).resolve(), Path(teacher_root).resolve()
    if root == source or root in source.parents or source in root.parents:
        raise ValueError("Use a separate DINO output directory")
    if cfg.student_init != "scratch" or cfg.num_workers != 0 or cfg.max_train_batches is not None:
        raise ValueError("Use scratch students, num_workers=0 and complete epochs")
    if cfg.model_scale != "debug" and (cfg.epochs != 100 or cfg.teacher_epochs != 30):
        raise ValueError("This comparison uses 100 student epochs and the existing 30-epoch teacher")
    manifest = validate_manifest(cfg.data_root)
    if len(manifest["classes"]) != cfg.num_classes:
        raise ValueError("Dataset class count differs from config")
    teachers, baselines = [], []
    for seed in seeds:
        current = replace(cfg, seed=seed)
        record = stage_teacher(current, source, copy=False)
        state = load_checkpoint(record["source"])
        if state["config"].get("teacher_variant", "small") != cfg.teacher_variant:
            raise ValueError("Teacher size differs from the comparison config")
        result = json.loads((Path(record["source"]).parent / "result.json").read_text())
        if result["partial_training"] or result["last_epoch"] != cfg.teacher_epochs:
            raise ValueError("Source teacher training is incomplete")
        teachers.append(record)
        seed_all(seed)
        initial = model_fingerprint(build_model("student", current))
        for method in BASELINE_METHODS:
            directory = Path(baseline_root) / f"seed_{seed}/scratch/{method}"
            saved = json.loads((directory / "result.json").read_text())
            compatible_config(Config.load(**saved["config"]).to_dict(), current.to_dict())
            if (saved["method"] != method or saved["seed"] != seed or saved["student_init"] != "scratch"
                    or saved["partial_training"] or saved["last_epoch"] != cfg.epochs
                    or saved["metadata_sha256"] != metadata_hash(cfg)
                    or saved["teacher_sha256"] != record["sha256"] or saved["initial_model_sha256"] != initial):
                raise ValueError(f"Unpaired or incomplete baseline: {directory}")
            # Register hashes without reading test scores before new training ends.
            baselines.append({"seed": seed, "method": method, "initial_model_sha256": initial,
                              "result_sha256": sha256(directory / "result.json"),
                              "test_sha256": sha256(directory / "test_metrics.json")})
    selector = build_selector(cfg, DINO_METHODS[0])
    selector_hash = model_fingerprint(selector)
    del selector
    protocol = {"schema": 1, "study": "dino_attention_pilot", "config": cfg.to_dict(),
                "seeds": seeds, "methods": list(DINO_METHODS), "teachers": teachers,
                "baselines": baselines, "baseline_root": str(Path(baseline_root).resolve()),
                "metadata_sha256": metadata_hash(cfg), "code_sha256": source_fingerprint(),
                "selector_sha256": selector_hash, "selector": "frozen_DINO_v1_ViT-S16",
                "selector_weights_url": DINO_URL, "selector_file_sha256": DINO_SHA256,
                "selector_is_synthetic": cfg.model_scale == "debug",
                "student_patches": 196, "teacher_patches": 98,
                "dino_random_rescue_10": "All epochs: remove random 10 from top98; add random10 from outside top98",
                "dino_paper_late": "Epochs 1-50 top98; 51-100 top78 plus random20 from other118 (40%+10% rounded)",
                "primary_endpoint": "Epoch100 test macro; validation-best reported separately",
                "interpretation": "COCO adaptation of MaskedKD section5.1/Table5, not DINO SSL or ImageNet reproduction. "
                                  "One-seed pilot and previously observed test set are exploratory."}
    path = root / "comparison_protocol.json"
    if path.exists():
        previous = json.loads(path.read_text())
        compatible_config(previous["config"], protocol["config"])
        for key in ("study", "seeds", "methods", "metadata_sha256", "code_sha256", "selector_sha256", "baselines"):
            if previous[key] != protocol[key]:
                raise ValueError(f"Existing DINO study {key} changed; use a new output directory")
        if [t["sha256"] for t in previous["teachers"]] != [t["sha256"] for t in teachers]:
            raise ValueError("Teacher changed since protocol registration")
    elif any(root.glob("seed_*")):
        raise ValueError("Output already contains another experiment")
    write_json(path, protocol)
    for seed in seeds:
        stage_teacher(replace(cfg, seed=seed), source)
    return protocol


def benchmark(cfg, seeds, jobs, warmup, steps):
    from .benchmark import _hardware, run_trial
    from .utils import setup_device
    import torch
    device = setup_device(cfg)
    single = run_trial(cfg, 1, warmup, steps, modes=DINO_METHODS, measure_all=True)
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3 if device.type == "cuda" else 0
    worst = max(single["rows"], key=lambda r: r["peak_reserved_gib"])
    # Conservative requirement for any wave, including jobs=2 (two waves).
    required = required_memory([worst] * jobs, total) if device.type == "cuda" else None
    require_memory(cfg, required)
    trials = [single]
    for offset in range(0, len(DINO_METHODS), jobs):
        modes = DINO_METHODS[offset:offset + jobs]
        if jobs > 1:
            trials.append(run_trial(cfg, len(modes), warmup, steps, parallel_methods=modes))
    count = sum(r["split"] == "train" for r in json.loads((Path(cfg.data_root) / "manifest.json").read_text())["images"])
    if jobs == 1:
        seconds = sum(task_seconds(cfg, row, count) for row in single["rows"])
    else:
        seconds = sum(max(task_seconds(cfg, row, count) for row in trial["rows"]) for trial in trials[1:])
    if device.type == "cuda":
        required = max(required, *(required_memory(t["rows"], total) for t in trials[1:])) if jobs > 1 else required
    report = {"study": "dino_attention_pilot", "hardware": _hardware(device), "jobs": jobs,
              "trials": trials, "required_free_gib": required, "gpu_estimate": device.type == "cuda",
              "estimated_hours_per_seed": seconds / 3600, "estimated_plan_hours": seconds * len(seeds) / 3600,
              "note": "Measured frozen DINO forward is included. Includes validation/probe/checkpoint; excludes download, test/export. No teacher retraining."}
    write_json(Path(cfg.output_root) / "benchmark.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "trials"}, indent=2), flush=True)
    return report


def require_completed(cfg, protocol):
    for seed in protocol["seeds"]:
        initial = next(r["initial_model_sha256"] for r in protocol["baselines"] if r["seed"] == seed)
        teacher = next(r["sha256"] for r in protocol["teachers"] if r["seed"] == seed)
        for method in DINO_METHODS:
            directory = run_path(replace(cfg, seed=seed), method=method)
            result = json.loads((directory / "result.json").read_text())
            compatible_config(result["config"], replace(cfg, seed=seed).to_dict())
            for key, expected in {"seed": seed, "role": "student", "method": method,
                                  "last_epoch": cfg.epochs, "partial_training": False,
                                  "initial_model_sha256": initial, "teacher_sha256": teacher,
                                  "metadata_sha256": protocol["metadata_sha256"],
                                  "selector_sha256": protocol["selector_sha256"],
                                  "code_sha256": protocol["code_sha256"]}.items():
                if result.get(key) != expected:
                    raise ValueError(f"Run {key} does not match protocol: {directory}")
            history = json.loads((directory / "history.json").read_text())
            if [r["epoch"] for r in history] != list(range(1, cfg.epochs + 1)):
                raise ValueError(f"Incomplete history: {directory}")
            for name in ("best", "last"):
                if not (directory / f"{name}.pt").exists():
                    raise FileNotFoundError(directory / f"{name}.pt")


def summarize(cfg, protocol):
    from .analysis import analyze
    destination = analyze(cfg.output_root)
    rows = []
    for seed in protocol["seeds"]:
        for source, root, methods in (("student", Path(protocol["baseline_root"]), BASELINE_METHODS),
                                       ("dino", Path(cfg.output_root), DINO_METHODS)):
            for method in methods:
                directory = root / f"seed_{seed}/scratch/{method}"
                result = json.loads((directory / "result.json").read_text())
                tests = json.loads((directory / "test_metrics.json").read_text())
                for name, epoch in (("last", cfg.epochs), ("best", result["best_epoch"])):
                    if tests[name]["epoch"] != epoch:
                        raise ValueError(f"Test epoch mismatch: {directory}")
                    rows.append({"seed": seed, "selection_source": source, "method": method,
                                 "checkpoint": name, "epoch": epoch,
                                 "test_macro_pct": tests[name]["macro_accuracy"] * 100})
    differences = []
    for seed in protocol["seeds"]:
        for checkpoint in ("last", "best"):
            scores = {r["method"]: r["test_macro_pct"] for r in rows if r["seed"] == seed and r["checkpoint"] == checkpoint}
            student_gain = scores["random_rescue_10"] - scores["student"]
            dino_gain = scores["dino_random_rescue_10"] - scores["dino"]
            differences.append({"seed": seed, "checkpoint": checkpoint,
                                "random10_gain_student_pp": student_gain, "random10_gain_dino_pp": dino_gain,
                                "dino_minus_student_random10_gain_pp": dino_gain - student_gain,
                                "paper_late_gain_dino_pp": scores["dino_paper_late"] - scores["dino"]})
    for name, table in (("dino_comparison.csv", rows), ("dino_random10_effect.csv", differences)):
        with (destination / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    write_json(destination / "dino_interpretation.json", {"seeds": protocol["seeds"],
               "scope": protocol["interpretation"], "comparisons": differences,
               "warning": "Do not count probe images or diagnostic repeats as independent training seeds."})
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "benchmark", "evaluate", "export"))
    parser.add_argument("--config", default="configs/dino_attention_pilot.json")
    parser.add_argument("--teacher-root", default="outputs/teacher_base_pilot")
    parser.add_argument("--baseline-report", default=DEFAULT_BASELINE)
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--device")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--jobs", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.steps < 1 or any(s < 0 for s in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use unique nonnegative seeds/warmup and positive steps")
    if args.push and args.action != "export":
        parser.error("--push is only for export")
    os.chdir(ROOT)
    cfg = Config.load(args.config, data_root=args.data_root, output_root=args.output_root, device=args.device)
    with RunLock(Path(cfg.output_root) / ".dino.lock"):
        protocol = prepare_protocol(cfg, args.teacher_root, args.baseline_report, args.seeds)
        if args.action in ("run", "benchmark"):
            report = benchmark(cfg, args.seeds, args.jobs, args.warmup, args.steps)
            if args.action == "benchmark":
                return
            config_path = Path(cfg.output_root).resolve() / "_jobs/config.json"
            write_json(config_path, cfg.to_dict())
            for seed in args.seeds:
                for offset in range(0, len(DINO_METHODS), args.jobs):
                    require_memory(cfg, report["required_free_gib"])
                    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                        tasks = [pool.submit(train_task, cfg, seed, "student", method, config_path)
                                 for method in DINO_METHODS[offset:offset + args.jobs]]
                        for task in tasks:
                            task.result()
            require_completed(cfg, protocol)
            from .analysis import analyze
            print(f"COMPLETE DINO students; validation: {analyze(cfg.output_root)}", flush=True)
            print("Next: bash setup.sh dino-evaluate (repeat custom paths/seeds)", flush=True)
        else:
            require_completed(cfg, protocol)
            if args.action == "evaluate":
                for seed in args.seeds:
                    for method in DINO_METHODS:
                        evaluate_test(replace(cfg, seed=seed), method=method)
            print(f"DINO comparison: {summarize(cfg, protocol)}", flush=True)
            if args.action == "export":
                from .export import export_results
                export_results(cfg, push=args.push)


if __name__ == "__main__":
    main()
