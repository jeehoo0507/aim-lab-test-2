import copy
import json
import time

import pytest

from coco_kd.benchmark import automatic_jobs, benchmark, memory_requirement, recommend, run_trial
from coco_kd.config import Config, METHODS
from coco_kd.synthetic import synthetic_data
from coco_kd.system import checkpoint_estimate


def trial(factor=1):
    rows = []
    for method in ("teacher", *METHODS):
        row = {"method": method, "seconds": 10 * factor, "samples": 100, "peak_reserved_gib": 2}
        if method in ("teacher", "student"):
            row.update(validation_seconds=2 * factor, checkpoint_seconds=factor)
        if method == "student":
            row.update(probe_seconds=3 * factor, diagnostic_seconds=20 * factor)
        rows.append(row)
    return {"jobs": 1, "status": "passed", "rows": rows}


def test_parallel_recommendation_accounts_for_three_seed_tail_and_contention():
    cfg = Config()
    fast = recommend(trial(), trial(1.2), cfg, 6000)
    assert fast["recommended_jobs"] == 2
    assert fast["full_plan_speedup"] == pytest.approx(3 / 2.2)
    assert fast["aggregate_throughput_speedup"] == pytest.approx(2 / 1.2)
    # Enough VRAM alone must not trigger concurrent training.
    slow = recommend(trial(), trial(2.1), cfg, 6000)
    assert slow["recommended_jobs"] == 1
    assert slow["parallel_hours"] > slow["serial_hours"]
    assert recommend(trial(), {"status": "failed"}, cfg, 6000)["recommended_jobs"] == 1
    assert memory_requirement(trial(), 24) == pytest.approx(7.9)


def test_storage_estimate_includes_resume_teacher_and_midpoint():
    result = checkpoint_estimate(Config(), 5526346, 21669514)
    expected = (5526346 * 4 * 16 * 42 + 21669514 * 4 * 9 * 3 + 5526346 * 4 * 4 * 6)
    assert result["total_decimal_gb"] == pytest.approx(expected / 1e9)


def test_real_spawned_single_and_dual_benchmark_and_auto_guards(tmp_path):
    synthetic_data(tmp_path / "data")
    cfg = Config.load(data_root=str(tmp_path / "data"), output_root=str(tmp_path / "out"),
                      num_classes=2, device="cpu", model_scale="debug", student_init="scratch",
                      teacher_pretrained=False, num_workers=0, batch_size=2, eval_batch_size=2,
                      accumulation_steps=2, epochs=2, teacher_epochs=2, diagnostic_repeats=1,
                      diagnostic_epochs=[0, 2])
    destination = tmp_path / "out/benchmark.json"
    report = benchmark(cfg, destination, steps=1, warmup=1, max_jobs=1)
    assert len(report["trials"][0]["rows"]) == 8
    assert automatic_jobs(cfg) == 1
    assert report["estimated_probe_validation_gib"] > 0
    # Exercise multiprocessing/barriers and per-worker scratch files on CPU.
    dual = run_trial(cfg, 2, warmup=1, steps=1)
    assert len(dual["rows"]) == 16
    assert {r["rank"] for r in dual["rows"]} == {0, 1}
    assert all(r["seconds"] > 0 for r in dual["rows"])
    assert not list((tmp_path / "out/_benchmark").iterdir())
    assert not list((tmp_path / "out").glob("seed_*"))
    stale = copy.deepcopy(report)
    stale["measured_at_unix"] = time.time() - 90000
    destination.write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="older than 24h"):
        automatic_jobs(cfg)
    incomplete = copy.deepcopy(report)
    incomplete.pop("recommendation")
    destination.write_text(json.dumps(incomplete))
    with pytest.raises(ValueError, match="did not finish"):
        automatic_jobs(cfg)
