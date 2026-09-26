from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from coco_kd import deletion_parallel as parallel
from coco_kd import deletion_run as run
from coco_kd.config import Config
from coco_kd.deletion import AuditSettings, removal_proposals
from coco_kd.deletion_resources import Limits, ResourceGuard, ResourceStop
from coco_kd.synthetic import synthetic_data


def scalar_redundancy(current, removable, k, attention, features):
    f = F.normalize(features.float(), dim=-1)
    similarity = f @ f.T
    left, allowed, removed = set(current.tolist()), set(removable.tolist()), []
    for _ in range(k):
        def key(i):
            neighbors = sorted(left-{i})
            score = float(similarity[i, neighbors].max()) if neighbors else -float('inf')
            return (-score, float(attention[i]), i)
        chosen = min(allowed, key=key)
        left.remove(chosen)
        allowed.remove(chosen)
        removed.append(chosen)
    return removed


@pytest.mark.parametrize('feature_kind', ['random', 'tied', 'zero'])
def test_vectorized_proposals_match_scalar_search_exactly(feature_kind):
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(196, 24, generator=generator)
    if feature_kind == 'tied':
        features[:] = features[0].clone()
    elif feature_kind == 'zero':
        features.zero_()
    attention = torch.randint(0, 3, (196,), generator=generator).float()
    top = attention.topk(98).indices
    rest = torch.tensor(sorted(set(range(196))-set(top.tolist())))
    for current, removable in [(torch.arange(196), torch.arange(196)), (torch.cat((top, rest[:10])), top)]:
        for k in (1, 5, 10):
            expected = scalar_redundancy(current, removable, k, attention, features)
            actual = removal_proposals(current, removable, k, attention, features)
            assert any(set(ids) == set(expected) for ids in actual)
            # When it duplicates low-attention proposal, preserving its first
            # occurrence is the existing candidate-family convention.
            unique_expected = []
            low = sorted(removable.tolist(), key=lambda i: (float(attention[i]), i))[:k]
            for ids in (low, expected):
                if not any(set(ids) == set(other) for other in unique_expected):
                    unique_expected.append(ids)
            assert actual == unique_expected


def info():
    return {'logical_cpus': 16, 'ram': {'available_gib': 40}, 'disk_free_gib': 100,
            'gpu': {'free_gib': 18, 'temperature_c': 50}}


def test_capacity_reserves_all_process_caps_and_external_memory():
    cfg, limits = Config(), Limits(gpu_memory_gib=12, min_ram_available_gib=8)
    available = info()
    plan = parallel.capacity(available, cfg, limits, 1.6, 3, 3)
    assert plan['capacity_jobs'] == 3
    assert parallel.admission(available, cfg, limits, plan, 3) == []
    available['gpu']['free_gib'] = 10
    assert parallel.admission(available, cfg, limits, plan, 3)
    assert parallel.capacity(available, cfg, limits, 1.6, 3, 3)['capacity_jobs'] == 1
    available['ram']['available_gib'] = 9
    assert parallel.capacity(available, cfg, limits, 1.6, 3, 3)['capacity_jobs'] == 0


def test_more_memory_does_not_select_slower_parallelism():
    trials = [{'jobs': 1, 'status': 'passed', 'estimated_train_only_wall_hours': 30},
              {'jobs': 2, 'status': 'passed', 'estimated_train_only_wall_hours': 20},
              {'jobs': 3, 'status': 'passed', 'estimated_train_only_wall_hours': 23}]
    assert parallel.choose_trial(trials)['jobs'] == 2
    trials[2]['estimated_train_only_wall_hours'] = 17
    assert parallel.choose_trial(trials)['jobs'] == 3


def test_wall_estimate_accounts_for_queued_seed_wave():
    report = {'rows': [{'seconds_per_image': 0.01, 'peak_reserved_gib': 1.5}]}
    result = parallel.estimate_trial([report, report], 6000, 100, 3)
    assert result['estimated_train_only_wall_hours'] == pytest.approx(10/3)


def tiny_setup(tmp_path):
    synthetic_data(tmp_path/'data')
    cfg = Config.load(data_root=str(tmp_path/'data'), output_root=str(tmp_path/'out'),
                      device='cpu', model_scale='debug', num_classes=2, teacher_pretrained=False,
                      student_init='scratch', batch_size=2, eval_batch_size=2, accumulation_steps=2,
                      num_workers=0, num_threads=1, epochs=1, teacher_epochs=1, warmup_epochs=0)
    limits = Limits(min_ram_available_gib=.001, min_disk_free_gib=.001, duty_cycle=1, nice=0, max_session_minutes=None)
    return cfg, AuditSettings(max_views_per_image=9), limits


def test_synchronized_spawn_probe_and_fixed_teacher_seed_training(tmp_path):
    cfg, policy, limits = tiny_setup(tmp_path)
    reports = parallel.probe_trial(cfg, policy, limits, ['student', 'rescue_audit'], 1, 2)
    assert len(reports) == 2
    assert all(p['warmup_microbatches'] == 1 and len(p['rows']) == 2 for p in reports)
    run.train_one(cfg, policy, limits, 'teacher')
    checkpoint = run.run_directory(cfg, 'teacher')/'best.pt'
    context = parallel.mp.get_context('spawn')
    processes = []
    for seed in (1, 2):
        local = replace(cfg, seed=seed)
        process = context.Process(target=parallel._train_worker,
            args=(local, policy, limits, ['rescue_audit'], checkpoint, 0, None))
        process.start()
        processes.append(process)
    assert parallel.wait_workers(processes, ResourceGuard(limits, tmp_path))
    records = [json.loads((run.run_directory(replace(cfg, seed=s), 'rescue_audit')/'result.json').read_text()) for s in (1, 2)]
    assert records[0]['teacher_sha256'] == records[1]['teacher_sha256']
    assert records[0]['initial_model_sha256'] != records[1]['initial_model_sha256']
    assert all(r['provenance']['teacher_seed'] == 0 and not r['test_evaluated'] for r in records)
    with pytest.raises(ValueError, match='identity changed'):
        run.train_one(replace(cfg, seed=1), policy, limits, 'rescue_audit', checkpoint, teacher_seed=1)


def test_worker_failure_stops_only_owned_siblings(tmp_path):
    class Process:
        def __init__(self, failed=False):
            self.exitcode = 1 if failed else None
            self.terminated = False
        def is_alive(self):
            return self.exitcode is None
        def terminate(self):
            self.terminated, self.exitcode = True, -15
        def join(self, timeout=None):
            pass
    failed, sibling = Process(True), Process()
    limits = Limits(min_ram_available_gib=.001, min_disk_free_gib=.001, duty_cycle=1, nice=0)
    assert not parallel.wait_workers([failed, sibling], ResourceGuard(limits, tmp_path))
    assert sibling.terminated and not failed.terminated
