from dataclasses import replace
import json

import pytest
import torch

from coco_kd.config import Config
from coco_kd.deletion import AuditSettings, audit_candidates, deletion_targets, student_forward
from coco_kd.deletion_resources import Limits, ResourceGuard, ResourceStop, check_resources
from coco_kd.deletion_run import train_one, run_directory
from coco_kd.masking import select_tokens
from coco_kd.models import build_model
from coco_kd.synthetic import synthetic_data
from coco_kd.utils import load_checkpoint


def test_virtual_audit_depends_on_student_not_just_teacher_distance():
    cfg = Config(label_smoothing=0)
    policy = AuditSettings(etas=(1.0,))
    parent = torch.tensor([.8, .2]).log()
    candidates = torch.tensor([[.75, .25], [.85, .15]]).log()
    _, low = audit_candidates(torch.tensor([.7, .3]).log(), 0, parent, candidates, cfg, policy)
    _, high = audit_candidates(torch.tensor([.95, .05]).log(), 0, parent, candidates, cfg, policy)
    assert low[1] < low[0]
    assert high[0] < high[1]
    # A zero reference gradient never licenses aggressive deletion.
    passed, _ = audit_candidates(torch.tensor([.9, .1]).double().log(), 0,
                                torch.tensor([.8, .2]).double().log(), candidates, cfg, policy)
    assert not passed.any()


def test_joint_deletion_is_checked_and_reference_target_is_fixed():
    cfg, policy = Config(label_smoothing=0), AuditSettings()
    parent = torch.tensor([.9, .1]).log()
    # Redundant object patches: either one suffices, losing both changes target.
    candidates = torch.tensor([[.9, .1], [.9, .1], [.1, .9]]).log()
    passed, _ = audit_candidates(torch.tensor([.3, .7]).log(), 0, parent, candidates, cfg, policy)
    assert passed.tolist() == [True, True, False]


class ConstantTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.masks = []

    def forward(self, images, indices=None):
        if indices is not None:
            self.masks.extend(indices.detach().cpu().tolist())
        return torch.tensor([1., -1.], device=images.device).expand(len(images), -1)


def policy_inputs():
    return (torch.zeros(1, 3, 224, 224), torch.tensor([0]), torch.tensor([[0., 0.]]),
            torch.linspace(0, 1, 196).reshape(1, -1), torch.randn(1, 196, 5))


def test_random_proposal_replays_original_rng_order_and_budget():
    cfg, policy = Config(num_classes=2, amp=False), AuditSettings()
    x, y, z, a, f = policy_inputs()
    rng = torch.Generator().manual_seed(18)
    expected, _ = select_tokens(a, 'random_rescue_10', generator=rng)
    expected_state = rng.get_state()
    teacher = ConstantTeacher()
    actual_rng = torch.Generator().manual_seed(18)
    _, stats = deletion_targets(teacher, x, y, z, a, f, cfg, policy, 'rescue_audit', actual_rng)
    assert torch.equal(actual_rng.get_state(), expected_state)
    assert expected[0].tolist() in teacher.masks  # Existing Random10 is an actual candidate.
    assert stats['kept'] == 98 and stats['swaps'] == 10
    assert stats['teacher_views'] <= policy.max_views_per_image
    assert all(len(m) == len(set(m)) for m in teacher.masks)


def test_deterministic_pruning_stops_at_compute_budget_without_rng_consumption():
    cfg, policy = Config(num_classes=2, amp=False), AuditSettings(max_views_per_image=9)
    rng = torch.Generator().manual_seed(15)
    before = rng.get_state()
    _, stats = deletion_targets(ConstantTeacher(), *policy_inputs(), cfg, policy,
                                 'deterministic_audit', rng)
    assert torch.equal(before, rng.get_state())
    assert 98 <= stats['kept'] < 196
    assert stats['teacher_views'] <= 9 and stats['budget_stops'] == 1


def test_full_forward_hook_does_not_change_outputs_or_keep_graph():
    cfg = Config(num_classes=2, model_scale='debug', drop_path=0)
    model = build_model('student', cfg)
    x = torch.randn(2, 3, 224, 224)
    logits, attention, features = student_forward(model, x)
    reference, raw = model(x, return_attention=True)
    torch.testing.assert_close(logits, reference)
    torch.testing.assert_close(attention, raw)
    assert features.shape == (2, 196, 24) and not features.requires_grad
    assert not model.norm._forward_hooks


def test_resource_thresholds_fail_closed():
    info = {'ram': {'available_gib': 30}, 'disk_free_gib': 40,
            'gpu': {'free_gib': 9, 'temperature_c': 50}}
    limits = Limits()
    assert check_resources(info, limits) == []
    assert check_resources({**info, 'gpu': {'free_gib': 2, 'temperature_c': 79}}, limits)
    assert check_resources({**info, 'gpu': None}, limits)
    guard = ResourceGuard(limits, '/tmp')
    guard.stop_requested = True
    with pytest.raises(ResourceStop, match='Stop requested'):
        guard.tick()


def debug_config(tmp_path):
    synthetic_data(tmp_path/'data')
    cfg = Config.load(data_root=str(tmp_path/'data'), output_root=str(tmp_path/'out'),
        device='cpu', model_scale='debug', num_classes=2, teacher_pretrained=False,
        student_init='scratch', batch_size=2, eval_batch_size=2, accumulation_steps=2,
        num_workers=0, epochs=2, teacher_epochs=1, warmup_epochs=0)
    limits = Limits(min_ram_available_gib=.001, min_disk_free_gib=.001, duty_cycle=1, nice=0)
    return cfg, AuditSettings(max_views_per_image=9), limits


def test_audit_training_resume_matches_uninterrupted_and_no_test_peeking(tmp_path):
    cfg, policy, limits = debug_config(tmp_path)
    train_one(cfg, policy, limits, 'teacher')
    teacher = run_directory(cfg, 'teacher')/'best.pt'
    train_one(cfg, policy, limits, 'rescue_audit', teacher)
    expected = load_checkpoint(run_directory(cfg, 'rescue_audit')/'last.pt')
    other = replace(cfg, output_root=str(tmp_path/'resume'))
    train_one(other, policy, limits, 'rescue_audit', teacher, until_epoch=1)
    train_one(other, policy, limits, 'rescue_audit', teacher)
    path = run_directory(other, 'rescue_audit')
    actual = load_checkpoint(path/'last.pt')
    for key in expected['model']:
        torch.testing.assert_close(actual['model'][key], expected['model'][key], rtol=0, atol=0)
    result = json.loads((path/'result.json').read_text())
    assert not result['test_evaluated'] and not (path/'test_metrics.json').exists()
    with pytest.raises(ValueError, match='identity changed'):
        train_one(other, replace(policy, epsilon=.1), limits, 'rescue_audit', teacher)


def test_nonrandom_method_completes_with_variable_lengths(tmp_path):
    cfg, policy, limits = debug_config(tmp_path)
    train_one(cfg, policy, limits, 'teacher')
    teacher = run_directory(cfg, 'teacher')/'best.pt'
    train_one(cfg, policy, limits, 'deterministic_audit', teacher, until_epoch=1)
    history = json.loads((run_directory(cfg, 'deterministic_audit')/'history.json').read_text())
    assert 98 <= history[0]['train']['kept'] <= 196
    assert history[0]['train']['teacher_views'] <= policy.max_views_per_image


def test_resource_abort_rolls_back_uncommitted_epoch(tmp_path, monkeypatch):
    import coco_kd.deletion_run as runner
    cfg, policy, limits = debug_config(tmp_path)
    train_one(cfg, policy, limits, 'teacher')
    teacher = run_directory(cfg, 'teacher')/'best.pt'
    original = runner.training_epoch
    def interrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[10] == 2:  # epoch; updates have occurred but are not committed.
            raise ResourceStop('simulated temperature stop')
        return result
    monkeypatch.setattr(runner, 'training_epoch', interrupt)
    with pytest.raises(ResourceStop, match='temperature'):
        train_one(cfg, policy, limits, 'rescue_audit', teacher)
    path = run_directory(cfg, 'rescue_audit')
    assert json.loads((path/'STOPPED.json').read_text())['completed_epoch'] == 1
    assert load_checkpoint(path/'last.pt')['epoch'] == 1
    monkeypatch.setattr(runner, 'training_epoch', original)
    train_one(cfg, policy, limits, 'rescue_audit', teacher)
    assert not (path/'STOPPED.json').exists()
    assert load_checkpoint(path/'last.pt')['epoch'] == 2


def test_benchmark_is_temporary_and_changed_settings_require_remeasurement(tmp_path):
    from coco_kd.deletion_run import benchmark, require_benchmark
    cfg, policy, limits = debug_config(tmp_path)
    report = benchmark(cfg, policy, limits, steps=1)
    assert len(report['rows']) == 5
    assert all(row['seconds_per_image'] > 0 for row in report['rows'])
    assert not list((tmp_path/'out').rglob('*.pt'))
    require_benchmark(cfg, policy, limits)
    with pytest.raises(ValueError, match='Benchmark settings'):
        require_benchmark(replace(cfg, batch_size=4), policy, limits)
