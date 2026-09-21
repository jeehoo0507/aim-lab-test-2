import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coco_kd.analysis import correction_summary, convergence_summary, paired_differences
from coco_kd.metrics import kl
from coco_kd.utils import RunLock


def test_correction_keeps_censored_failures():
    rows = []
    for sample_id, correct in [(1, [False, True, False, True, True, True]), (2, [False] * 6)]:
        for epoch, value in enumerate(correct):
            rows.append({"sample_id": sample_id, "epoch": epoch, "teacher_full_correct": True, "student_correct": value})
    result = correction_summary(pd.DataFrame(rows))
    assert result["epoch0_correction_opportunities"] == 2
    assert result["corrected_by_end"] == .5
    assert result["mean_wait_capped_at_horizon_plus1"] == (3 + 6) / 2


def test_convergence_threshold_requires_three_epochs_and_counts_updates():
    frame = pd.DataFrame({"seed": [0] * 5, "initialization": ["scratch"] * 5, "method": ["student"] * 5,
                          "epoch": list(range(1, 6)), "val_macro_accuracy": [.7, .85, .83, .82, .9],
                          "optimizer_updates": [4] * 5, "train_seconds": [2] * 5, "epoch_seconds_with_probe": [3] * 5})
    row = convergence_summary(frame).iloc[0]
    assert row.first_epoch == 2 and row.updates_to_first == 8
    assert row.training_seconds_to_first == 4


def test_seed_comparison_pairs_same_seed_and_not_images():
    rows = [{"seed": seed, "initialization": "imagenet", "checkpoint": "last", "method": method,
             "macro_accuracy": accuracy, "accuracy": accuracy}
            for seed in range(3) for method, accuracy in [("student", .7 + seed * .01), ("low_score_rescue_10", .73 + seed * .01)]]
    result = paired_differences(pd.DataFrame(rows))
    assert len(result) == 2 and (result.n_seeds == 3).all()
    assert np.allclose(result.mean_difference_pp, 3)


def test_kl_direction_and_run_lock(tmp_path):
    assert np.allclose(kl([[1., 2.]], [[1., 2.]]), 0)
    assert kl([[1., 2.]], [[3., -2.]])[0] > 0
    with RunLock(tmp_path / ".lock"):
        with pytest.raises(RuntimeError, match="Another process"):
            with RunLock(tmp_path / ".lock"):
                pass
