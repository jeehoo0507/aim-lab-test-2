import numpy as np
import pytest
import torch

from coco_kd.masking import binary_mask, effective_method, select_tokens
from coco_kd.metrics import classification, selection
from coco_kd.probe import Probe


@pytest.mark.parametrize("mode", ["student", "random", "random_rescue_10", "low_score_rescue_10", "foreground_rescue_10", "random_matched"])
def test_budget_and_unique_tokens(mode):
    attn = torch.arange(196, dtype=torch.float32).repeat(2, 1)
    fg = torch.zeros(2, 196, dtype=torch.bool)
    fg[:, :20] = True
    idx, swaps = select_tokens(attn, mode, foreground=fg, generator=torch.Generator().manual_seed(7))
    assert idx.shape == (2, 98)
    assert all(len(row.unique()) == 98 for row in idx)
    raw = binary_mask(attn.topk(98, 1).indices)
    chosen = binary_mask(idx)
    if mode in ("random_rescue_10", "low_score_rescue_10", "foreground_rescue_10", "random_matched"):
        assert torch.equal((raw & ~chosen).sum(1), swaps)
        assert torch.equal((chosen & ~raw).sum(1), swaps)
    if mode == "low_score_rescue_10":
        assert chosen[:, 108:].all()  # strongest 88 preserved
        assert not chosen[:, 98:108].any()
    if mode == "foreground_rescue_10":
        assert ((chosen & ~raw) & fg).sum() == 20
        assert not ((raw & ~chosen) & fg).any()


def test_fixed_random_does_not_use_ground_truth_and_matched_handles_no_fg():
    attn = torch.rand(3, 196)
    outputs = []
    for foreground in (None, torch.zeros(3, 196, dtype=torch.bool), torch.ones(3, 196, dtype=torch.bool)):
        outputs.append(select_tokens(attn, "random_rescue_10", foreground=foreground, generator=torch.Generator().manual_seed(7))[0])
    assert torch.equal(outputs[0], outputs[1]) and torch.equal(outputs[0], outputs[2])
    for mode in ("random_matched", "foreground_rescue_10"):
        idx, swaps = select_tokens(attn, mode, foreground=torch.zeros(3, 196, dtype=torch.bool))
        assert not swaps.any()
        assert torch.equal(idx, attn.topk(98, 1).indices)


@pytest.mark.parametrize("method,switch", [("full_to_student_20", 20), ("full_to_student_50", 50)])
def test_full_to_student_switch_boundary(method, switch):
    attn = torch.rand(2, 196)
    assert effective_method(method, switch) == "full"
    assert effective_method(method, switch + 1) == "student"
    before, before_swaps = select_tokens(attn, method, epoch=switch)
    after, after_swaps = select_tokens(attn, method, epoch=switch + 1)
    assert before is None and not before_swaps.any()  # teacher sees all 196
    assert torch.equal(after, attn.topk(98, 1).indices)
    assert not after_swaps.any()
    probe = Probe.__new__(Probe)
    probe.device = torch.device("cpu")
    ids = torch.arange(2)
    foreground = torch.zeros_like(attn, dtype=torch.bool)
    full_indices, _ = probe.choose(attn, method, foreground, ids, epoch=switch)
    masked_indices, _ = probe.choose(attn, method, foreground, ids, epoch=switch + 1)
    assert torch.equal(full_indices, torch.arange(196).expand(2, -1))
    assert torch.equal(masked_indices, after)
    with pytest.raises(ValueError, match="epoch"):
        select_tokens(attn, method)


@pytest.mark.parametrize("method,first,switch", [
    ("random_to_student_10", "random", 10),
    ("random_to_student_20", "random", 20),
    ("random_to_student_50", "random", 50),
    ("random_rescue_to_student_20", "random_rescue_10", 20),
])
def test_random_to_student_switch_matches_original_methods(method, first, switch):
    attn = torch.arange(196, dtype=torch.float32).repeat(2, 1)
    foreground = torch.zeros_like(attn, dtype=torch.bool)
    assert effective_method(method, switch) == first
    assert effective_method(method, switch + 1) == "student"
    expected, expected_swaps = select_tokens(attn, first, foreground=foreground,
                                              generator=torch.Generator().manual_seed(7), epoch=switch)
    actual, actual_swaps = select_tokens(attn, method, foreground=foreground,
                                          generator=torch.Generator().manual_seed(7), epoch=switch)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_swaps, expected_swaps)
    after, after_swaps = select_tokens(attn, method, foreground=foreground, epoch=switch + 1)
    assert torch.equal(after, attn.topk(98, 1).indices)
    assert not after_swaps.any()
    probe = Probe.__new__(Probe)
    probe.device = torch.device("cpu")
    ids = torch.arange(2)
    probe_after, _ = probe.choose(attn, method, foreground, ids, epoch=switch + 1)
    assert torch.equal(probe_after, after)


def test_metric_zero_fg_and_accuracy():
    fg = np.zeros((2, 196), bool)
    fg[0, :20] = True
    indices = np.tile(np.arange(98), (2, 1))
    values = selection(indices, fg, fg.astype(float))
    assert values["foreground_missing"][0] == 0
    assert np.isnan(values["foreground_missing"][1])
    assert values["background_ratio"][0] == pytest.approx(78 / 98)
    result = classification([[1, 0], [1, 0], [0, 1]], [0, 1, 1], 2)
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_accuracy"] == .75
