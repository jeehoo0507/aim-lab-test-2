import csv
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from coco_kd.config import Config
from coco_kd.models import build_model
from coco_kd.synthetic import synthetic_data
from coco_kd.teacher_base import METHODS, check_small_baselines, prepare_protocol, require_completed, required_memory
from coco_kd.train import evaluate_test, run_path, train
from coco_kd.utils import compatible_config, model_fingerprint, seed_all, write_json


def test_base_architecture_and_student_initialization_are_correct():
    cfg = Config(teacher_variant="base")
    teacher = build_model("teacher", cfg).eval()
    assert len(teacher.blocks) == 12
    assert teacher.pos_embed.shape == (1, 197, 768)
    assert teacher.blocks[0].attn.num_heads == 12
    assert sum(p.numel() for p in teacher.parameters()) == 85806346
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        full, attention = teacher(x, return_attention=True)
        masked = teacher(x, attention.topk(98, 1).indices)
    assert full.shape == masked.shape == (1, 10)
    assert attention.shape == (1, 196)
    hashes = []
    for variant in ("small", "base"):
        seed_all(0)
        hashes.append(model_fingerprint(build_model("student", replace(cfg, teacher_variant=variant))))
    assert hashes[0] == hashes[1]


def test_old_configs_mean_small_and_cannot_load_as_base():
    saved = Config().to_dict()
    saved.pop("teacher_variant")
    compatible_config(saved, Config().to_dict())
    with pytest.raises(ValueError, match="teacher_variant"):
        compatible_config(saved, Config(teacher_variant="base").to_dict())
    with pytest.raises(ValueError, match="teacher_variant"):
        Config.load(teacher_variant="large")
    assert required_memory([{"peak_reserved_gib": 2}, {"peak_reserved_gib": 3}, {"peak_reserved_gib": 2}], 24) == pytest.approx(11.65)


def test_pilot_parallel_training_resume_evaluation_and_export(tmp_path):
    synthetic_data(tmp_path / "data")
    small = Config.load(data_root=str(tmp_path / "data"), output_root=str(tmp_path / "small"),
                        model_scale="debug", num_classes=2, device="cpu", teacher_pretrained=False,
                        student_init="scratch", num_workers=0, batch_size=2, eval_batch_size=2,
                        accumulation_steps=2, epochs=2, teacher_epochs=2, warmup_epochs=0,
                        diagnostic_epochs=[0, 2], diagnostic_repeats=1, checkpoint_every=1)
    train(small, role="teacher")
    for method in METHODS:
        train(small, method=method)
        evaluate_test(small, method=method)
    cfg = replace(small, output_root=str(tmp_path / "base"), teacher_variant="base")
    baseline = prepare_protocol(cfg, [0], small.output_root)
    with pytest.raises(FileNotFoundError):
        require_completed(cfg, [0], baseline)
    # Leave a real interrupted teacher; the launcher must resume it before students.
    teacher_cfg = replace(cfg, student_init="imagenet")
    train(teacher_cfg, role="teacher", stop_after=1)
    config = tmp_path / "config.json"
    write_json(config, cfg.to_dict())
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, MPLCONFIGDIR=str(tmp_path / "mpl"), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")

    def command(action):
        result = subprocess.run([sys.executable, "-m", "coco_kd.teacher_base", action,
                                 "--config", str(config), "--small-root", small.output_root,
                                 "--warmup", "0", "--steps", "1"], cwd=root, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
        assert result.returncode == 0, result.stdout
        return result.stdout

    command("run")
    assert "RESUME" in (Path(cfg.output_root) / "_jobs/seed_0_teacher_student.log").read_text()
    require_completed(cfg, [0], baseline)
    report = json.loads((Path(cfg.output_root) / "benchmark.json").read_text())
    assert not report["gpu_estimate"]
    parallel = report["trials"][1]
    assert parallel["jobs"] == 3 and {r["method"] for r in parallel["rows"]} == set(METHODS)
    assert all(r["probe_seconds"] > 0 and r["validation_seconds"] > 0 for r in parallel["rows"])
    for method in METHODS:
        path = run_path(cfg, method=method)
        assert not (path / "test_metrics.json").exists()
        before = (path / "last.pt").stat().st_mtime_ns
        train(cfg, method=method)
        assert (path / "last.pt").stat().st_mtime_ns == before
    command("evaluate")
    rows = list(csv.DictReader((Path(cfg.output_root) / "analysis/teacher_size_comparison.csv").open()))
    assert len(rows) == 6 and {r["teacher"] for r in rows} == {"small", "base"}
    base_random = next(r for r in rows if r["teacher"] == "base" and r["method"] == "random_rescue_10")
    base_masked = next(r for r in rows if r["teacher"] == "base" and r["method"] == "student")
    assert float(base_random["delta_vs_masked_pp"]) == pytest.approx(100 * (float(base_random["last_macro_accuracy"]) - float(base_masked["last_macro_accuracy"])))
    assert "Exported" in command("export")
    with pytest.raises(ValueError, match="different settings"):
        check_small_baselines(replace(cfg, scratch_lr=1e-3), [0], small.output_root)
    with pytest.raises(ValueError, match="different settings"):
        prepare_protocol(replace(cfg, kd_alpha=.7), [0], small.output_root)
