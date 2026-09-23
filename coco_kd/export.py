import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm

from .analysis import analyze
from .system import storage_report
from .utils import sha256, write_json


def export_results(cfg, push=False):
    root = Path(cfg.output_root).resolve()
    analysis = analyze(root)
    tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    destination = (Path("reports/smoke") if cfg.model_scale == "debug" else Path("reports")) / tag
    destination.mkdir(parents=True)
    shutil.copytree(analysis, destination / "analysis")
    for name in ("preflight.json", "benchmark.json", "comparison_protocol.json", "adaptive_protocol.json"):
        if (root / name).exists():
            shutil.copy2(root / name, destination / name)
    # Snapshot only compact data: full probe tensors and weights remain on the server.
    for source in sorted(root.glob("seed_*/**/*")):
        if source.is_file() and source.name in {"config.json", "history.json", "result.json", "test_metrics.json", "test_best.npz", "test_last.npz"}:
            target = destination / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for name in ("manifest.json", "quality_report.json"):
        source = Path(cfg.data_root) / name
        if source.exists():
            shutil.copy2(source, destination / f"dataset_{name}")
    weights = [{"server_path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)}
               for p in tqdm(sorted(root.glob("seed_*/**/*.pt")), desc="Checkpoint checksums (weights stay on server)")]
    write_json(destination / "checkpoint_manifest.json", weights)
    write_json(destination / "storage.json", storage_report({"project": Path.cwd(), "data": cfg.data_root, "outputs": root}))
    (destination / "README.md").write_text("# Experiment 2 export\n\nWeights and full attention/probe NPZ files remain on the server.\n"
        "analysis/per_image/*.csv.gz contains every saved epoch's per-image predictions, selection metrics, KL and correction flags.\n"
        "Checkpoint paths and SHA256 are in checkpoint_manifest.json. Test results are present only after explicit evaluate.\n"
        "Intervals over 3 seeds are exploratory; probe images or repeated masks are not extra training seeds.\n")
    files = [{"path": str(p.relative_to(destination)), "bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(destination.rglob("*")) if p.is_file()]
    write_json(destination / "manifest.json", files)
    if any(r["bytes"] > 50 * 1024**2 for r in files):
        raise ValueError(f"Export contains a file >50 MiB; kept locally at {destination}, not pushed")
    print(f"Exported {sum(r['bytes'] for r in files) / 1024**2:.2f} MiB to {destination.resolve()}")
    if push:
        if cfg.model_scale == "debug":
            raise ValueError("Synthetic smoke reports are not published as experiment results")
        # Never stage unrelated code or outputs, never rewrite history, never auto-merge.
        subprocess.run(["git", "add", "--", str(destination)], check=True)
        subprocess.run(["git", "commit", "--only", "-m", f"results: export experiment 2 {tag}", "--", str(destination)], check=True)
        subprocess.run(["git", "push"], check=True)
    return destination
