import csv
import json
from dataclasses import replace

import pytest

from coco_kd.config import Config
from coco_kd.synthetic import synthetic_data
from coco_kd.utils import metadata_hash, write_json
from scripts.run_adaptive200 import BASELINES, check_existing_baselines, compare_quick, require_completed


def make_results(tmp_path):
    data = tmp_path / "data"
    synthetic_data(data)
    cfg = Config.load(data_root=str(data), output_root=str(tmp_path / "quick"), num_classes=2,
                      model_scale="debug", epochs=100, seed=0, student_init="scratch")
    original = tmp_path / "original"
    base = {"config": replace(cfg, output_root=str(original)).to_dict(), "last_epoch": 100,
            "partial_training": False, "seed": 0, "student_init": "scratch",
            "teacher_sha256": "same_teacher", "initial_model_sha256": "same_initial",
            "metadata_sha256": metadata_hash(cfg)}
    for method, score in zip(BASELINES, (.60, .62, .58)):
        run = original / "seed_0" / "scratch" / method
        write_json(run / "result.json", {**base, "method": method})
        write_json(run / "test_metrics.json", {"last": {"macro_accuracy": score}})
    method = "adaptive_random_to_student"
    run = tmp_path / "quick" / "seed_0" / "scratch" / method
    write_json(run / "result.json", {**base, "method": method,
                                       "gate_state": {"switched_after_epoch": 40}})
    write_json(run / "test_metrics.json", {"last": {"macro_accuracy": .64}})
    return cfg, original, method, run


def test_quick_comparison_uses_matched_100_epoch_last_checkpoints(tmp_path):
    cfg, original, method, _ = make_results(tmp_path)
    require_completed(cfg, [0], ["scratch"], [method])
    path = compare_quick(cfg, original, [0], ["scratch"], [method])
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    adaptive = rows[-1]
    assert adaptive["method"] == method
    assert adaptive["switch_after_epoch"] == "40"
    assert float(adaptive["delta_vs_masked_pp"]) == pytest.approx(4.0)
    assert float(adaptive["delta_vs_random10_pp"]) == pytest.approx(2.0)


def test_quick_comparison_rejects_initialization_mismatch(tmp_path):
    cfg, original, method, run = make_results(tmp_path)
    result = json.loads((run / "result.json").read_text())
    result["initial_model_sha256"] = "different"
    write_json(run / "result.json", result)
    with pytest.raises(ValueError, match="initial_model_sha256"):
        compare_quick(cfg, original, [0], ["scratch"], [method])


def test_quick_preflight_rejects_baseline_schedule_mismatch(tmp_path):
    cfg, original, _, _ = make_results(tmp_path)
    bad = replace(cfg, epochs=200)
    with pytest.raises(ValueError, match="epoch 200"):
        check_existing_baselines(bad, original, [0], ["scratch"])
