import json
import shutil
from dataclasses import replace

import numpy as np
import pytest
import torch

from coco_kd.adaptive import paired_indices, update_gate
from coco_kd.adaptive import GateProbe
from coco_kd.analysis import analyze
from coco_kd.config import Config
from coco_kd.data import CocoSubset
from coco_kd.synthetic import synthetic_data
from coco_kd.train import run_path, train
from coco_kd.utils import load_checkpoint, write_json


def test_paired_policy_only_changes_outgoing_selection():
    attention = torch.arange(196, dtype=torch.float32)[None].repeat(2, 1)
    ids = torch.tensor([13, 17])
    raw, random, low = paired_indices(attention, ids, 0, torch.device("cpu"))
    again = paired_indices(attention, ids, 0, torch.device("cpu"))
    assert all(torch.equal(a, b) for a, b in zip((raw, random, low), again))
    for i in range(2):
        raw_set, random_set, low_set = (set(x[i].tolist()) for x in (raw, random, low))
        assert len(random_set) == len(low_set) == 98
        assert random_set - raw_set == low_set - raw_set
        assert raw_set - low_set == set(range(98, 108))


def test_gate_requires_two_successive_eligible_checks():
    cfg = Config(gate_min_epoch=30)
    state = {"favorable_streak": 0, "switched_after_epoch": None}
    for epoch, favorable in ((20, True), (30, True), (40, False), (50, True), (60, True)):
        state = update_gate(state, {"epoch": epoch, "favorable": favorable}, cfg)
    assert state["switched_after_epoch"] == 60
    assert update_gate(state, {"epoch": 70, "favorable": False}, cfg) == state


@pytest.mark.parametrize("method", ["adaptive_random_to_low_10", "adaptive_random_to_student"])
def test_adaptive_resume_and_validation_probe_separation(tmp_path, method):
    root = tmp_path / "data"
    synthetic_data(root)
    manifest = json.loads((root / "manifest.json").read_text())
    for row in manifest["images"]:
        if row["split"] == "val" and row["id"] % 2 == 0:
            row["probe"] = False
    write_json(root / "manifest.json", manifest)
    cfg = Config.load(data_root=str(root), output_root=str(tmp_path / "full"), num_classes=2,
                      device="cpu", model_scale="debug", student_init="scratch", teacher_pretrained=False,
                      num_workers=0, batch_size=2, eval_batch_size=2, accumulation_steps=2,
                      teacher_epochs=2, epochs=3, warmup_epochs=0, diagnostic_epochs=[0, 3],
                      diagnostic_repeats=1, gate_interval=1, gate_min_epoch=1, gate_repeats=1,
                      validation_exclude_probe=True)
    assert len(CocoSubset(root, "probe")) == 2
    assert len(CocoSubset(root, "val_fit")) == 2
    teacher = train(cfg, "teacher")
    train(cfg, method=method)
    original = load_checkpoint(run_path(cfg, method=method) / "last.pt")
    resumed_cfg = replace(cfg, output_root=str(tmp_path / "resumed"))
    shutil.copytree(teacher.parent, run_path(resumed_cfg, "teacher"))
    train(resumed_cfg, method=method, stop_after=1)
    train(resumed_cfg, method=method)
    run = run_path(resumed_cfg, method=method)
    resumed = load_checkpoint(run / "last.pt")
    for name in original["model"]:
        torch.testing.assert_close(original["model"][name], resumed["model"][name], rtol=0, atol=0)
    history = json.loads((run / "history.json").read_text())
    assert all(row["active_method"] in ("random_rescue_10", "low_score_rescue_10", "student") for row in history)
    assert (run / "gate/epoch_001.npz").exists()
    with np.load(run / "gate/epoch_001.npz") as data:
        assert data["random_indices"].shape == data["candidate_indices"].shape == (1, 2, 98)
    assert (analyze(resumed_cfg.output_root) / "gate_history.csv").exists()


def test_training_switch_starts_after_second_favorable_probe(tmp_path, monkeypatch):
    root = tmp_path / "data"
    synthetic_data(root)
    cfg = Config.load(data_root=str(root), output_root=str(tmp_path / "out"), num_classes=2,
                      device="cpu", model_scale="debug", student_init="scratch", teacher_pretrained=False,
                      num_workers=0, batch_size=2, eval_batch_size=2, accumulation_steps=2,
                      teacher_epochs=1, epochs=3, warmup_epochs=0, diagnostic_epochs=[0, 3],
                      diagnostic_repeats=1, gate_interval=1, gate_min_epoch=1, gate_repeats=1)
    train(cfg, "teacher")
    monkeypatch.setattr(GateProbe, "measure", lambda self, model, epoch, target: {
        "epoch": epoch, "favorable": True, "diagnostic_seconds": 0.0,
        "teacher_full_image_passes": 0, "teacher_98_image_passes": 0,
        "kl_gain": 1.0, "true_logp_gain": 1.0, "flip_gain": 0.0})
    train(cfg, method="adaptive_random_to_low_10")
    run = run_path(cfg, method="adaptive_random_to_low_10")
    history = json.loads((run / "history.json").read_text())
    assert [row["active_method"] for row in history] == ["random_rescue_10", "random_rescue_10", "low_score_rescue_10"]
    assert json.loads((run / "result.json").read_text())["gate_state"]["switched_after_epoch"] == 2
