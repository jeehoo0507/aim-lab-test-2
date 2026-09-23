"""Opt-in two-condition follow-up using the existing experiment-2 teachers."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from .config import Config
from .masking import annealed_swaps
from .utils import (RunLock, compatible_config, load_checkpoint, metadata_hash,
                    setup_device, sha256, write_json)

PAIR = ("student", "random_anneal_10")


def stage_teacher(cfg, teacher_root, copy=True):
    """Copy a frozen teacher only; never resume/retrain the old experiment."""
    source = Path(teacher_root) / f"seed_{cfg.seed}" / "teacher" / "best.pt"
    if not source.is_file():
        raise FileNotFoundError(f"Existing trained teacher required: {source}")
    state = load_checkpoint(source)
    if state.get("role") != "teacher" or state.get("metadata_sha256") != metadata_hash(cfg):
        raise ValueError("Teacher role/dataset mismatch")
    for key in ("seed", "num_classes", "model_scale"):
        if state["config"][key] != getattr(cfg, key):
            raise ValueError(f"Teacher {key} mismatch")
    fingerprint = sha256(source)
    target = Path(cfg.output_root) / f"seed_{cfg.seed}" / "teacher" / "best.pt"
    if target.exists():
        if sha256(target) != fingerprint:
            raise ValueError("Copied teacher changed; use a new comparison output root")
    elif copy:
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".tmp")
        shutil.copyfile(source, temp)
        if sha256(temp) != fingerprint:
            raise ValueError("Teacher changed during copy")
        temp.replace(target)
    return {"seed": cfg.seed, "source": str(source.resolve()), "sha256": fingerprint,
            "epoch": state["epoch"], "training_code_sha256": state.get("code_sha256")}


def prepare_protocol(cfg, teacher_root, seeds, inits):
    root, source = Path(cfg.output_root).resolve(), Path(teacher_root).resolve()
    if root == source or root in source.parents or source in root.parents:
        raise ValueError("Comparison output and original experiment must be separate directories")
    path = root / "comparison_protocol.json"
    old = json.loads(path.read_text()) if path.exists() else None
    if old:
        compatible_config(old["config"], cfg.to_dict())
    elif any(root.glob("seed_*")):
        raise ValueError("Unmarked output contains runs; choose an empty comparison root")
    teachers = {r["seed"]: r for r in old["teachers"]} if old else {}
    # Validate every source first, then register the protocol before copying.
    # An interrupted copy can be safely retried without an unmarked output tree.
    for seed in seeds:
        teachers[seed] = stage_teacher(replace(cfg, seed=seed), source, copy=False)
    pairs = {tuple(row) for row in old["pairs"]} if old else set()
    pairs.update((seed, init) for seed in seeds for init in inits)
    protocol = {"schema": 1, "config": cfg.to_dict(), "pairs": sorted(pairs),
                "methods": list(PAIR), "teachers": list(teachers.values()),
                "schedule": [{"epoch": e, "random_swaps": annealed_swaps(e)} for e in range(cfg.epochs + 1)],
                "teacher_patch_budget": 98, "student_patch_budget": 196,
                "primary_endpoint": "test macro accuracy at epoch 100; validation-best also reported",
                "interpretation": "Two-arm schedule comparison, not proof that decay beats constant Random 10. "
                                  "Seed 0 is a pilot; the existing test set has already been examined."}
    write_json(path, protocol)
    for seed in seeds:
        stage_teacher(replace(cfg, seed=seed), source)
    return protocol


def completed_rows(root, protocol):
    """Require complete, genuinely paired runs before reporting success/exporting."""
    rows = []
    for seed, init in protocol["pairs"]:
        results = []
        for method in PAIR:
            run = root / f"seed_{seed}" / init / method
            result = json.loads((run / "result.json").read_text())
            compatible_config({**protocol["config"], "seed": seed, "student_init": init}, result["config"])
            expected_teacher = next(t["sha256"] for t in protocol["teachers"] if t["seed"] == seed)
            if result["method"] != method or result["teacher_sha256"] != expected_teacher:
                raise ValueError(f"Run/protocol provenance mismatch: {run}")
            history = json.loads((run / "history.json").read_text())
            tests = json.loads((run / "test_metrics.json").read_text())
            expected = result["config"]["epochs"]
            if len(history) != expected or [h["epoch"] for h in history] != list(range(1, expected + 1)):
                raise ValueError(f"Incomplete history: {run}")
            if result["partial_training"] or result["last_epoch"] != expected:
                raise ValueError(f"Partial training cannot be reported as complete: {run}")
            for h in history:
                swaps = annealed_swaps(h["epoch"]) if method == "random_anneal_10" else 0
                if abs(h["train"]["swaps"] - swaps) > 1e-8:
                    raise ValueError(f"Wrong actual swap schedule: {run}, epoch {h['epoch']}")
            for name, epoch in (("best", result["best_epoch"]), ("last", expected)):
                if tests[name]["epoch"] != epoch:
                    raise ValueError(f"Test checkpoint mismatch: {run}/{name}")
            results.append(result)
            rows.append({"seed": seed, "initialization": init, "method": method,
                         "best_epoch": result["best_epoch"],
                         "validation_best_pct": 100 * result["best_validation_macro"],
                         "validation_last_pct": 100 * history[-1]["validation"]["macro_accuracy"],
                         "test_best_pct": 100 * tests["best"]["macro_accuracy"],
                         "test_last_pct": 100 * tests["last"]["macro_accuracy"]})
        for key in ("initial_model_sha256", "teacher_sha256", "metadata_sha256", "code_sha256"):
            if results[0][key] != results[1][key]:
                raise ValueError(f"Unpaired {key}: seed={seed}, init={init}")
        compatible_config(results[0]["config"], results[1]["config"])
    return rows


def summarize(root, protocol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from .analysis import NAMES
    rows = completed_rows(root, protocol)
    destination = root / "analysis"
    destination.mkdir(exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(destination / "comparison.csv", index=False)
    lines = ["# Random 10→0 vs MaskedKD", "", protocol["interpretation"], "",
             "Student sees 196 patches; teacher always sees 98. Random replacements: 10 through epoch 20, "
             "integer linear decay to 0 at epoch 80, then 0. No LR/optimizer restart.", "",
             "| Init | Seed | Method | Best epoch | Val best % | Val last % | Test best % | Test last % |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['initialization']} | {r['seed']} | {NAMES[r['method']]} | {r['best_epoch']} | "
                     f"{r['validation_best_pct']:.3f} | {r['validation_last_pct']:.3f} | "
                     f"{r['test_best_pct']:.3f} | {r['test_last_pct']:.3f} |")
    lines += ["", "![Comparison](anneal_comparison.png)", "",
              "`main_scratch.png` / `main_imagenet.png`: raw student selection and teacher KL curves. "
              "`paired_seed_differences.csv`: schedule minus MaskedKD, both best/last. "
              "`learning_curves.csv`: actual swaps per epoch. Repeated probe images are not extra seeds."]
    (destination / "comparison.md").write_text("\n".join(lines) + "\n")
    curves = pd.read_csv(destination / "learning_curves.csv")
    inits = list(table.initialization.unique())
    fig, axes = plt.subplots(3, len(inits), figsize=(6 * len(inits), 10), squeeze=False, constrained_layout=True)
    colors = {"student": "#405269", "random_anneal_10": "#138b83"}
    for col, init in enumerate(inits):
        for method in PAIR:
            c = curves[(curves.initialization == init) & (curves.method == method)]
            for row, key, factor, label in ((0, "swaps", 1, "Random swaps (of 98 teacher patches)"),
                                           (1, "val_macro_accuracy", 100, "Validation macro accuracy (%)")):
                groups = c.groupby("epoch")[key]
                mean, sd = groups.mean() * factor, groups.std().fillna(0) * factor
                axes[row, col].plot(mean.index, mean, color=colors[method], label=NAMES[method])
                axes[row, col].fill_between(mean.index, mean - sd, mean + sd, color=colors[method], alpha=.12)
                axes[row, col].set(xlabel="Epoch", ylabel=label, xlim=(0, protocol["config"]["epochs"]))
                for boundary in (20, 80):
                    if boundary <= protocol["config"]["epochs"]:
                        axes[row, col].axvline(boundary, color="grey", linestyle=":", alpha=.5)
            sub = table[(table.initialization == init) & (table.method == method)]
            for j, key in enumerate(("test_best_pct", "test_last_pct")):
                x = j + (-.12 if method == "student" else .12)
                values = sub[key].to_numpy()
                jitter = np.linspace(-.035, .035, len(values)) if len(values) > 1 else np.zeros(1)
                axes[2, col].scatter(x + jitter, values, color=colors[method], alpha=.5)
                axes[2, col].scatter(x, values.mean(), color=colors[method], marker="D", label=NAMES[method] if j == 0 else None)
        axes[0, col].set_title(init)
        axes[0, col].legend(fontsize=9)
        axes[2, col].set(xticks=[0, 1], xticklabels=["Validation-best", f"Epoch {protocol['config']['epochs']} (primary)"], ylabel="Test macro accuracy (%)")
        for ax in axes[:, col]:
            ax.grid(alpha=.15)
            ax.spines[["top", "right"]].set_visible(False)
    prefix = "SYNTHETIC SMOKE | " if protocol["config"]["model_scale"] == "debug" else ""
    fig.suptitle(prefix + "Random 10 to 0 vs MaskedKD\nBands: seed SD when n > 1; test points: individual seeds", fontsize=11)
    for extension in ("png", "pdf"):
        fig.savefig(destination / f"anneal_comparison.{extension}", dpi=180)
    plt.close(fig)
    print(table.to_string(index=False), flush=True)
    print(f"Complete comparison: {destination / 'comparison.md'}", flush=True)


def run_comparison(cfg, teacher_root, seeds, inits, jobs):
    from .analysis import analyze
    from .data import validate_manifest
    from .models import build_model
    from .train import evaluate_test
    root = Path(cfg.output_root)
    with RunLock(root / ".comparison.lock"):
        manifest = validate_manifest(cfg.data_root)
        if len(manifest["classes"]) != cfg.num_classes:
            raise ValueError("Dataset class count mismatch")
        device = setup_device(cfg)
        protocol = prepare_protocol(cfg, teacher_root, seeds, inits)
        if "imagenet" in inits:
            model = build_model("student", cfg, pretrained=True)
            del model
        if device.type == "cuda":
            import torch
            free, total = torch.cuda.mem_get_info(device)
            print(f"GPU free={free / 1024**3:.2f}/{total / 1024**3:.2f} GiB; jobs={jobs}", flush=True)
        directory = root / "_jobs"
        directory.mkdir(exist_ok=True)
        config_path = directory / "config.json"
        write_json(config_path, cfg.to_dict())
        tasks = [(seed, init, method) for seed in seeds for init in inits for method in PAIR]
        def task(item):
            seed, init, method = item
            log = directory / f"seed_{seed}_{init}_{method}.log"
            print(f"START seed={seed} {init} {method}; log={log}", flush=True)
            with log.open("a", buffering=1) as file:
                subprocess.run([sys.executable, "-u", str(Path(__file__).resolve().parent.parent / "run.py"),
                                "train", "--config", str(config_path), "--seed", str(seed), "--init", init,
                                "--method", method], check=True, stdout=file, stderr=subprocess.STDOUT,
                               env=dict(os.environ, OMP_NUM_THREADS=str(cfg.num_threads), MKL_NUM_THREADS=str(cfg.num_threads)))
            print(f"DONE seed={seed} {init} {method}", flush=True)
        with ThreadPoolExecutor(max_workers=min(jobs, len(tasks))) as pool:
            list(pool.map(task, tasks))
        # The fixed protocol is registered before training; no per-epoch test peeking.
        for seed, init, method in tasks:
            evaluate_test(replace(cfg, seed=seed, student_init=init), method=method)
        analyze(root)
        summarize(root, protocol)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "export"))
    p.add_argument("--config", default="configs/experiment2.json")
    p.add_argument("--output-root", default="outputs/random_anneal")
    p.add_argument("--teacher-root", default="outputs/experiment2")
    p.add_argument("--data-root")
    p.add_argument("--device")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--inits", nargs="+", choices=("scratch", "imagenet"), default=["scratch", "imagenet"])
    p.add_argument("--jobs", type=int, choices=range(1, 5), default=4)
    p.add_argument("--push", action="store_true", help="Only with export; upload compact report, never weights")
    args = p.parse_args()
    if args.action == "run":
        if args.push:
            p.error("Use export --push after reviewing the completed comparison")
        cfg = Config.load(args.config, output_root=args.output_root, data_root=args.data_root, device=args.device,
                          num_workers=0, checkpoint_every=10)
        if cfg.epochs != 100 and cfg.model_scale != "debug":
            p.error("The comparison protocol requires 100 epochs")
        run_comparison(cfg, args.teacher_root, sorted(set(args.seeds)), sorted(set(args.inits)), args.jobs)
    else:
        root = Path(args.output_root)
        with RunLock(root / ".comparison.lock"):
            protocol = json.loads((root / "comparison_protocol.json").read_text())
            cfg = Config(**protocol["config"])
            cfg = replace(cfg, output_root=str(root))
            completed_rows(root, protocol)
            from .export import export_results
            export_results(cfg, push=args.push)


if __name__ == "__main__":
    main()
