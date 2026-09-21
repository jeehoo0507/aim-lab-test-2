import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from coco_kd.config import Config
from coco_kd.probe import Probe
from coco_kd.synthetic import synthetic_data
from coco_kd.train import train, run_path, evaluate_test, learning_rate
from coco_kd.utils import load_checkpoint


def config(tmp_path):
    synthetic_data(tmp_path / "data")
    return Config.load(data_root=str(tmp_path / "data"), output_root=str(tmp_path / "out"),
                       num_classes=2, device="cpu", model_scale="debug", student_init="scratch",
                       teacher_pretrained=False, num_workers=0, batch_size=2, eval_batch_size=2,
                       accumulation_steps=2, teacher_epochs=2, epochs=3, diagnostic_repeats=1,
                       diagnostic_epochs=[0, 3], checkpoint_every=2, warmup_epochs=0)


def test_actual_learning_rate_has_no_hidden_scaling():
    cfg = Config()
    assert learning_rate(cfg, 5, 100, "student") == pytest.approx(5e-5)
    assert learning_rate(replace(cfg, student_init="scratch"), 5, 100, "student") == pytest.approx(5e-4)


def test_resume_equivalent_to_uninterrupted_and_exports_probes(tmp_path):
    cfg = config(tmp_path)
    teacher = train(cfg, "teacher")
    train(cfg, method="low_score_rescue_10")
    run = run_path(cfg, method="low_score_rescue_10")
    expected = load_checkpoint(run / "last.pt")
    other = replace(cfg, output_root=str(tmp_path / "resumed"))
    import shutil
    other_teacher = run_path(other, "teacher")
    shutil.copytree(teacher.parent, other_teacher)
    # Moving output path is explicitly supported.
    train(other, method="low_score_rescue_10", stop_after=1)
    train(other, method="low_score_rescue_10")
    actual = load_checkpoint(run_path(other, method="low_score_rescue_10") / "last.pt")
    for key in expected["model"]:
        torch.testing.assert_close(actual["model"][key], expected["model"][key], rtol=0, atol=0)
    assert expected["best_epoch"] == actual["best_epoch"]
    assert "optimizer" not in actual and "best_model" not in actual
    assert (run / "epoch_002.pt").exists() and not (run / "epoch_001.pt").exists()
    assert not (run / "epoch_003.pt").exists()  # compact last.pt is the final snapshot
    with np.load(run / "probe/epoch_003.npz") as a:
        assert a["raw_indices"].shape == a["actual_indices"].shape == (4, 98)
        assert a["teacher_full_logits"].shape == (4, 2)
        assert a["attention"].shape == (4, 196)
    assert not (run / "test_metrics.json").exists()  # no test peeking during training
    evaluation = evaluate_test(cfg, method="low_score_rescue_10")
    assert set(evaluation) == {"best", "last"}
    before = (run / "last.pt").stat().st_mtime_ns
    train(cfg, method="low_score_rescue_10")
    assert (run / "last.pt").stat().st_mtime_ns == before
    with pytest.raises(ValueError, match="different settings"):
        train(replace(cfg, pretrained_lr=1e-4), method="low_score_rescue_10")


def test_probe_randomness_not_affected_by_training_seed_or_batch_size(tmp_path):
    cfg = config(tmp_path)
    p = Probe(cfg, tmp_path / "one", None, torch.device("cpu"))
    q = Probe(replace(cfg, seed=2, eval_batch_size=1), tmp_path / "two", None, torch.device("cpu"))
    attention = torch.rand(4, 196)
    fg = torch.rand(4, 196) > .75
    ids = torch.tensor([101, 103, 107, 109])
    whole, _ = p.choose(attention, "random_rescue_10", fg, ids)
    pieces = [q.choose(attention[i:i+1], "random_rescue_10", fg[i:i+1], ids[i:i+1])[0] for i in range(4)]
    assert torch.equal(whole, torch.cat(pieces))
