import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from coco_kd.anneal import PAIR, completed_rows, prepare_protocol, run_comparison
from coco_kd.config import Config, METHODS
from coco_kd.masking import annealed_swaps, binary_mask, select_tokens
from coco_kd.models import build_model
from coco_kd.probe import Probe
from coco_kd.synthetic import synthetic_data
from coco_kd.train import train
from coco_kd.utils import sha256


def test_schedule_budget_and_endpoints():
    assert len(METHODS) == 7  # follow-up is opt-in
    counts = [annealed_swaps(e) for e in range(101)]
    assert counts[:21] == [10] * 21
    assert counts[50] == 5 and counts[80:] == [0] * 21
    assert all(a >= b for a, b in zip(counts, counts[1:]))
    assert set(counts) == set(range(11))
    a = torch.rand(3, 196)
    raw = binary_mask(a.topk(98, 1).indices)
    for epoch in range(1, 101):
        idx, swaps = select_tokens(a, "random_anneal_10", epoch=epoch, generator=torch.Generator().manual_seed(3))
        assert idx.shape == (3, 98) and all(len(row.unique()) == 98 for row in idx)
        selected = binary_mask(idx)
        assert (swaps == counts[epoch]).all()
        assert torch.equal((raw & ~selected).sum(1), swaps)
        assert torch.equal((selected & ~raw).sum(1), swaps)
    for epoch, reference in ((1, "random_rescue_10"), (20, "random_rescue_10"), (80, "student"), (100, "student")):
        expected, _ = select_tokens(a, reference, generator=torch.Generator().manual_seed(7))
        actual, _ = select_tokens(a, "random_anneal_10", epoch=epoch, generator=torch.Generator().manual_seed(7))
        assert torch.equal(actual, expected)
    with pytest.raises(ValueError, match="epoch"):
        select_tokens(a, "random_anneal_10")
    with pytest.raises(ValueError, match="100-epoch"):
        train(Config(epochs=50), method="random_anneal_10")


def debug_config(tmp_path):
    synthetic_data(tmp_path / "data")
    return Config.load(data_root=str(tmp_path / "data"), output_root=str(tmp_path / "original"),
                       device="cpu", model_scale="debug", num_classes=2, teacher_pretrained=False,
                       student_init="scratch", num_workers=0, num_threads=2, batch_size=2, eval_batch_size=2,
                       accumulation_steps=2, teacher_epochs=1, epochs=2, warmup_epochs=0,
                       diagnostic_epochs=[], diagnostic_repeats=1, checkpoint_every=1)


def test_probe_uses_checkpoint_epoch_including_best(tmp_path):
    cfg = debug_config(tmp_path)
    teacher = build_model("teacher", cfg).eval()
    probe = Probe(cfg, tmp_path / "probe_run", teacher, torch.device("cpu"))
    model = build_model("student", cfg)
    for epoch in (50, 80):
        probe.log(model, epoch, "random_anneal_10", name="best")
        with np.load(probe.path / "best.npz") as a:
            assert (a["swaps"] == annealed_swaps(epoch)).all()
            assert a["epoch"] == epoch
            if epoch == 80:
                np.testing.assert_array_equal(a["actual_indices"], a["raw_indices"])
                np.testing.assert_array_equal(a["teacher_actual_logits"], a["teacher_raw_logits"])
    attention, fg, ids = torch.rand(4, 196), torch.zeros(4, 196, dtype=torch.bool), torch.arange(4)
    whole, _ = probe.choose(attention, "random_anneal_10", fg, ids, epoch=50)
    pieces = [probe.choose(attention[i:i+1], "random_anneal_10", ~fg[i:i+1], ids[i:i+1], epoch=50)[0] for i in range(4)]
    assert torch.equal(whole, torch.cat(pieces))  # batch independent; no FG labels used


def test_parallel_comparison_evaluation_and_export(tmp_path, monkeypatch):
    from coco_kd.export import export_results
    cfg = debug_config(tmp_path)
    teacher = train(cfg, role="teacher")
    original_hash = sha256(teacher)
    comparison = replace(cfg, output_root=str(tmp_path / "followup"))
    root = tmp_path / "followup"
    monkeypatch.chdir(tmp_path)
    run_comparison(comparison, cfg.output_root, [0], ["scratch"], jobs=2)
    assert sha256(teacher) == original_hash
    protocol = json.loads((root / "comparison_protocol.json").read_text())
    rows = completed_rows(root, protocol)
    assert len(rows) == 2 and {r["method"] for r in rows} == set(PAIR)
    assert (root / "analysis/anneal_comparison.png").exists()
    paired = pd.read_csv(root / "analysis/paired_seed_differences.csv")
    assert set(paired.method) == {"random_anneal_10"}
    assert set(paired.checkpoint) == {"best", "last"}
    exported = export_results(comparison)
    assert (exported / "comparison_protocol.json").exists()
    assert (exported / "analysis/comparison.csv").exists()
    assert not list(exported.rglob("*.pt"))
    assert (exported / "seed_0/scratch/random_anneal_10/test_last.npz").exists()
    manifest = json.loads((exported / "manifest.json").read_text())
    assert all(sha256(exported / row["path"]) == row["sha256"] for row in manifest)
    # Identical invocation is resumable/reusable and does not change completed weights.
    last = root / "seed_0/scratch/random_anneal_10/last.pt"
    before = last.stat().st_mtime_ns
    run_comparison(comparison, cfg.output_root, [0], ["scratch"], jobs=2)
    assert before == last.stat().st_mtime_ns
    with pytest.raises(ValueError, match="different settings"):
        prepare_protocol(replace(comparison, scratch_lr=1e-4), cfg.output_root, [0], ["scratch"])
    with pytest.raises(ValueError, match="separate"):
        prepare_protocol(cfg, cfg.output_root, [0], ["scratch"])
    history_path = root / "seed_0/scratch/random_anneal_10/history.json"
    history = json.loads(history_path.read_text())
    history[0]["train"]["swaps"] = 0
    history_path.write_text(json.dumps(history))
    with pytest.raises(ValueError, match="swap schedule"):
        completed_rows(root, protocol)
