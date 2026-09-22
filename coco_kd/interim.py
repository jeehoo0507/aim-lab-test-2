"""Validation-only snapshot while training continues; never read model weights."""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def export_interim(cfg, push=False, seeds=(0, 1, 2)):
    from .analysis import analyze
    from .config import METHODS
    from .utils import sha256, write_json

    if push and cfg.model_scale == "debug":
        raise ValueError("Synthetic interim reports must not be pushed")
    root = Path(cfg.output_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Training output directory does not exist: {root}")
    tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    base = Path("reports/smoke/interim") if cfg.model_scale == "debug" else Path("reports/interim")
    destination = base / tag
    destination.mkdir(parents=True)
    rows, completed = [], {}
    for seed in seeds:
        runs = [("teacher", "imagenet", "teacher", root / f"seed_{seed}" / "teacher")]
        runs += [("student", init, method, root / f"seed_{seed}" / init / method)
                 for init in ("scratch", "imagenet") for method in METHODS]
        for role, initialization, method, run in runs:
            # Read completion first. A run finishing during export remains incomplete in this snapshot.
            result = json.loads((run / "result.json").read_text()) if (run / "result.json").exists() else None
            config = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else None
            history = json.loads((run / "history.json").read_text()) if (run / "history.json").exists() else []
            if push and any(c and c.get("model_scale") == "debug"
                            for c in (config, result.get("config") if result else None)):
                raise ValueError("Synthetic interim reports must not be pushed")
            for name, value in (("config", config), ("history", history), ("result", result)):
                if value is not None and (name != "history" or history):
                    write_json(destination / run.relative_to(root) / f"{name}.json", value)
            best = max(history, key=lambda h: h["validation"]["macro_accuracy"], default=None)
            rows.append({"seed": seed, "role": role, "initialization": initialization, "method": method,
                         "state": "complete" if result else ("incomplete" if config or history else "not_started"),
                         "last_epoch": history[-1]["epoch"] if history else 0,
                         "best_epoch": best["epoch"] if best else None,
                         "best_val_macro": best["validation"]["macro_accuracy"] if best else None,
                         "last_val_macro": history[-1]["validation"]["macro_accuracy"] if history else None,
                         "recent_epoch_seconds_with_probe": [h["epoch_seconds_with_probe"] for h in history[-5:]]})
            if role == "student" and result:
                completed[(seed, initialization, method)] = run

    # Only complete seven-method blocks enter plots, so methods share the same seed cohort.
    groups = [(seed, init) for seed in seeds for init in ("scratch", "imagenet")
              if all((seed, init, method) in completed for method in METHODS)]
    with tempfile.TemporaryDirectory(prefix="interim_") as temporary:
        snapshot = Path(temporary)
        for seed, init in groups:
            for method in METHODS:
                run = completed[(seed, init, method)]
                target = snapshot / run.relative_to(root)
                target.mkdir(parents=True)
                for name in ("history.json", "result.json"):
                    shutil.copy2(destination / run.relative_to(root) / name, target / name)
                # Completed epoch probes are immutable; reference them without duplicating large arrays.
                (target / "probe").symlink_to(run / "probe", target_is_directory=True)
        if groups:
            print(f"Analyzing {len(groups)} completed seed/initialization blocks (CPU only)", flush=True)
            shutil.copytree(analyze(snapshot), destination / "analysis")
    write_json(destination / "progress.json", {
        "snapshot_utc": tag, "source_root": str(root), "planned_student_runs": len(seeds) * 2 * len(METHODS),
        "completed_student_runs": len(completed), "analysis_groups": groups, "runs": rows,
        "note": "Validation only. Incomplete means no completion marker, not proof of a running process."
    })
    (destination / "README.md").write_text(
        "# Interim validation snapshot\n\n"
        "Training continues. This report reads saved JSON and completed epoch probes only; "
        "it does not evaluate test data, load/hash weights, or modify training outputs.\n"
        "progress.json and run histories include all planned runs. Incomplete is not a process health check.\n"
        "Analysis uses only seed/initialization blocks with all seven methods complete. "
        "Partial blocks remain in histories, not cross-method mean curves.\n"
        "Raw selection is before swaps; actual selection is teacher input. "
        "These validation results are preliminary, not final test performance.\n")
    files = [{"path": str(p.relative_to(destination)), "bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(destination.rglob("*")) if p.is_file()]
    write_json(destination / "manifest.json", files)
    if any(f["bytes"] > 50 * 1024**2 for f in files):
        raise ValueError(f"File exceeds 50 MiB; report kept locally at {destination}")
    print(f"Students complete: {len(completed)}/{len(seeds) * 2 * len(METHODS)}; "
          f"exported {sum(f['bytes'] for f in files) / 1024**2:.2f} MiB to {destination.resolve()}", flush=True)
    if push:
        subprocess.run(["git", "add", "--", str(destination)], check=True)
        subprocess.run(["git", "commit", "--only", "-m", f"results: interim validation {tag}", "--", str(destination)], check=True)
        subprocess.run(["git", "push"], check=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment2.json")
    parser.add_argument("--output-root")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    # No CUDA calls; limit analysis CPU contention with the ongoing training jobs.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    from .config import Config
    export_interim(Config.load(args.config, output_root=args.output_root), args.push, tuple(dict.fromkeys(args.seeds)))


if __name__ == "__main__":
    main()
