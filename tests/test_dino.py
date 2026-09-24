import ast
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from coco_kd.config import Config
from coco_kd.dino_attention import DINO_METHODS, build_selector
from coco_kd.dino_pilot import prepare_protocol, require_completed
from coco_kd.masking import binary_mask, select_tokens
from coco_kd.models import DeiT
from coco_kd.probe import Probe
from coco_kd.synthetic import synthetic_data
from coco_kd.train import train, evaluate_test, run_path
from coco_kd.utils import load_checkpoint, model_fingerprint, write_json


def official_dino_class():
    source = Path(__file__).resolve().parents[1] / "vendor/DINO/vision_transformer.py"
    tree = ast.parse(source.read_text())
    names = {"drop_path", "DropPath", "Mlp", "Attention", "Block", "PatchEmbed", "VisionTransformer"}
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    namespace = {"torch": torch, "nn": nn, "math": math, "partial": partial, "trunc_normal_": nn.init.trunc_normal_}
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["VisionTransformer"]


def test_dino_features_attention_and_top98_match_unmodified_official():
    ours = DeiT(24, 2, 3, drop_path=0)
    ours.head = nn.Identity()
    ours.eval()
    reference = official_dino_class()(embed_dim=24, depth=2, num_heads=3, qkv_bias=True,
                                     norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0).eval()
    reference.load_state_dict(ours.state_dict(), strict=True)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        features, attention = ours(x, return_attention=True)
        expected = reference.get_last_selfattention(x).mean(1)[:, 0, 1:]
        torch.testing.assert_close(features, reference(x), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(attention, expected, rtol=1e-5, atol=1e-6)
        assert torch.equal(attention.topk(98, 1).indices, expected.topk(98, 1).indices)


def test_auxiliary_dino_is_frozen_and_does_not_consume_student_rng():
    before = torch.get_rng_state().clone()
    selector = build_selector(Config(model_scale="debug"), "dino")
    assert torch.equal(before, torch.get_rng_state())
    assert not selector.training and all(not p.requires_grad for p in selector.parameters())
    assert model_fingerprint(selector) == model_fingerprint(build_selector(Config(model_scale="debug"), "dino"))
    assert build_selector(Config(), "student") is None


def test_random10_tokens_and_paper_percentage_are_distinct():
    attention = torch.arange(196).float().expand(4, -1)
    raw, _ = select_tokens(attention, "dino")
    random, counts = select_tokens(attention, "dino_random_rescue_10", generator=torch.Generator().manual_seed(3))
    assert torch.all((binary_mask(raw) & binary_mask(random)).sum(1) == 88)
    assert counts.tolist() == [10] * 4
    early, early_counts = select_tokens(attention, "dino_paper_late", epoch=50)
    assert torch.equal(raw, early) and not early_counts.any()
    late, late_counts = select_tokens(attention, "dino_paper_late", epoch=51, generator=torch.Generator().manual_seed(3))
    assert torch.all(binary_mask(late).sum(1) == 98)
    assert torch.all(binary_mask(late).gather(1, attention.topk(78, 1).indices))
    assert torch.equal(late_counts, (binary_mask(late) & ~binary_mask(raw)).sum(1))
    changed_fg, _ = select_tokens(attention, "dino_random_rescue_10", foreground=torch.ones_like(attention, dtype=torch.bool),
                                 generator=torch.Generator().manual_seed(3))
    assert torch.equal(changed_fg, random)
    with pytest.raises(ValueError, match="epoch"):
        select_tokens(attention, "dino_paper_late")


def small_config(tmp_path, output):
    return Config.load(data_root=str(tmp_path / "data"), output_root=str(output), model_scale="debug",
                       teacher_variant="base", num_classes=2, device="cpu", teacher_pretrained=False,
                       student_init="scratch", num_workers=0, batch_size=2, eval_batch_size=2,
                       accumulation_steps=2, epochs=2, teacher_epochs=2, warmup_epochs=0,
                       diagnostic_epochs=[0, 2], diagnostic_repeats=1, checkpoint_every=1)


def test_dino_resume_and_probe_preserve_frozen_selection(tmp_path):
    synthetic_data(tmp_path / "data")
    cfg = small_config(tmp_path, tmp_path / "one")
    teacher = train(cfg, role="teacher")
    method = "dino_random_rescue_10"
    train(cfg, method=method)
    original = load_checkpoint(run_path(cfg, method=method) / "last.pt")
    resumed_cfg = replace(cfg, output_root=str(tmp_path / "two"))
    shutil.copytree(teacher.parent, run_path(resumed_cfg, "teacher"))
    train(resumed_cfg, method=method, stop_after=1)
    train(resumed_cfg, method=method)
    resumed = load_checkpoint(run_path(resumed_cfg, method=method) / "last.pt")
    for key in original["model"]:
        torch.testing.assert_close(original["model"][key], resumed["model"][key], rtol=0, atol=0)
    assert original["selector_sha256"] == resumed["selector_sha256"]
    directory = run_path(cfg, method=method)
    with np.load(directory / "probe/epoch_000.npz") as first, np.load(directory / "probe/epoch_002.npz") as last:
        np.testing.assert_array_equal(first["attention"], last["attention"])
        np.testing.assert_array_equal(first["actual_indices"], last["actual_indices"])
        assert not np.array_equal(first["student_attention"], last["student_attention"])
    selector = build_selector(cfg, method)
    p = Probe(cfg, tmp_path / "probe1", None, torch.device("cpu"), selector=selector)
    q = Probe(replace(cfg, seed=2), tmp_path / "probe2", None, torch.device("cpu"), selector=selector)
    a, fg, ids = torch.rand(4, 196), torch.zeros(4, 196, dtype=torch.bool), torch.arange(4)
    whole = p.choose(a, "dino_paper_late", fg, ids, epoch=51)[0]
    pieces = torch.cat([q.choose(a[i:i+1], "dino_paper_late", fg[i:i+1], ids[i:i+1], epoch=51)[0] for i in range(4)])
    assert torch.equal(whole, pieces)


def test_dino_launcher_benchmark_pairing_evaluate_export(tmp_path):
    synthetic_data(tmp_path / "data")
    source = small_config(tmp_path, tmp_path / "source")
    train(source, role="teacher")
    for method in ("full", "student", "random_rescue_10"):
        train(source, method=method)
        evaluate_test(source, method=method)
    cfg = replace(source, output_root=str(tmp_path / "dino"))
    with pytest.raises(ValueError, match="size"):
        prepare_protocol(replace(cfg, teacher_variant="small"), source.output_root, source.output_root, [0])
    protocol = prepare_protocol(cfg, source.output_root, source.output_root, [0])
    with pytest.raises(FileNotFoundError):
        require_completed(cfg, protocol)
    configuration = tmp_path / "dino.json"
    write_json(configuration, cfg.to_dict())
    env = dict(os.environ, MPLCONFIGDIR=str(tmp_path / "mpl"), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")

    def command(action):
        completed = subprocess.run([sys.executable, "-m", "coco_kd.dino_pilot", action, "--config", str(configuration),
                                   "--teacher-root", source.output_root, "--baseline-report", source.output_root,
                                   "--warmup", "0", "--steps", "1"], cwd=Path(__file__).resolve().parents[1],
                                  env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
        assert completed.returncode == 0, completed.stdout
        return completed.stdout

    command("run")
    require_completed(cfg, protocol)
    assert not (run_path(cfg, method="dino") / "test_metrics.json").exists()
    bench = json.loads((Path(cfg.output_root) / "benchmark.json").read_text())
    assert not bench["gpu_estimate"]
    assert bench["trials"][1]["jobs"] == 3
    assert {r["method"] for r in bench["trials"][1]["rows"]} == set(DINO_METHODS)
    command("evaluate")
    rows = list(csv.DictReader((Path(cfg.output_root) / "analysis/dino_random10_effect.csv").open()))
    assert len(rows) == 2 and {r["checkpoint"] for r in rows} == {"best", "last"}
    assert "Exported" in command("export")
    with pytest.raises(ValueError, match="different settings"):
        prepare_protocol(replace(cfg, kd_alpha=.7), source.output_root, source.output_root, [0])
