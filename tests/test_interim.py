import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from coco_kd.config import Config, METHODS
from coco_kd.interim import export_interim
from coco_kd.utils import write_json


def make_run(root, cfg, seed, initialization, method, complete=True):
    run = root / f"seed_{seed}" / initialization / method
    write_json(run / "config.json", cfg.to_dict())
    history = [{"epoch": epoch, "lr": .001,
                "validation": {"macro_accuracy": score, "accuracy": score},
                "train": {"loss": .2, "optimizer_updates": 1, "seconds": 1.,
                          "swaps": 0., "added_foreground": 0., "removed_foreground": 0.},
                "epoch_seconds_with_probe": 2.} for epoch, score in [(1, .8), (2, .7)]]
    write_json(run / "history.json", history)
    (run / "last.pt").write_bytes(b"deliberately invalid weights: must not be loaded")
    (run / "probe").mkdir()
    if complete:
        write_json(run / "result.json", {"config": cfg.to_dict(), "seed": seed,
                                        "student_init": initialization, "method": method})
        # Test values must never enter the intermediate validation report.
        (run / "test_metrics.json").write_text("not valid JSON: do not read test metrics")
        for epoch in range(3):
            foreground = np.tile(np.arange(196) < 50, (2, 1))
            logits = np.array([[2., 0.], [0., 2.]])
            np.savez_compressed(run / "probe" / f"epoch_{epoch:03d}.npz",
                                sample_id=[1, 2], label=[0, 1], epoch=epoch,
                                raw_indices=np.tile(np.arange(98), (2, 1)),
                                actual_indices=np.tile(np.arange(98), (2, 1)),
                                foreground=foreground, coverage=foreground.astype(float),
                                student_logits=logits, teacher_full_logits=logits,
                                teacher_raw_logits=logits, teacher_actual_logits=logits, swaps=[0, 0])
    else:
        (run / "probe" / "epoch_002.npz").write_bytes(b"active probe: must not be read")
    return run


def test_interim_reads_only_completed_probes_and_keeps_sources_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "outputs"
    cfg = Config(output_root=str(root), model_scale="debug", epochs=2)
    for method in METHODS:
        make_run(root, cfg, 0, "scratch", method)
    make_run(root, cfg, 0, "imagenet", "ce", complete=False)
    # A partially completed seed must not enter otherwise matched seed-0 figures.
    make_run(root, cfg, 1, "scratch", "ce")
    write_json(root / "analysis" / "sentinel.json", {"existing": True})
    before = {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
              for p in root.rglob("*") if p.is_file()}
    destination = export_interim(cfg)
    after = {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
             for p in root.rglob("*") if p.is_file()}
    assert before == after
    progress = json.loads((destination / "progress.json").read_text())
    assert progress["planned_student_runs"] == 42 and progress["completed_student_runs"] == 8
    assert progress["analysis_groups"] == [[0, "scratch"]]
    partial = next(r for r in progress["runs"] if r["seed"] == 0 and r["initialization"] == "imagenet" and r["method"] == "ce")
    assert partial["state"] == "incomplete" and partial["last_epoch"] == 2
    assert partial["best_val_macro"] == .8 and partial["last_val_macro"] == .7
    assert (destination / "seed_0/imagenet/ce/history.json").exists()
    curves = pd.read_csv(destination / "analysis/learning_curves.csv")
    assert set(curves.seed) == {0} and set(curves.method) == set(METHODS)
    assert len(list((destination / "analysis/per_image").glob("*.csv.gz"))) == 7
    assert (destination / "analysis/main_scratch.png").is_file()
    assert not (destination / "analysis/test_results.csv").exists()
    assert not any(p.is_symlink() or p.suffix in {".pt", ".npz"} or p.name == "test_metrics.json"
                   for p in destination.rglob("*"))
    for entry in json.loads((destination / "manifest.json").read_text()):
        assert hashlib.sha256((destination / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]


def test_interim_with_no_completed_block_and_debug_push_guard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "outputs"
    cfg = Config(output_root=str(root), model_scale="debug")
    make_run(root, cfg, 0, "scratch", "ce", complete=False)
    destination = export_interim(cfg, seeds=(0,))
    assert not (destination / "analysis").exists()
    assert json.loads((destination / "progress.json").read_text())["completed_student_runs"] == 0
    with pytest.raises(ValueError, match="must not be pushed"):
        export_interim(cfg, push=True)
