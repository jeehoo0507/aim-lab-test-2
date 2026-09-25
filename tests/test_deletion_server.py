from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess

import pytest
import torch

import coco_kd.deletion_run as runner
from coco_kd.deletion_resources import Limits, ResourceGuard, ResourceStop

ROOT = Path(__file__).resolve().parent.parent


def available_hardware(name='NVIDIA RTX A5000'):
    return {'ram': {'available_gib': 40}, 'disk_free_gib': 100,
            'gpu': {'name': name, 'total_gib': 24, 'free_gib': 18, 'temperature_c': 50}}


def test_server_profile_preserves_effective_batch_and_desktop_keeps_deadline():
    cfg, _, limits, _ = runner.load_profile(ROOT/'configs/deletion_server.json')
    desktop, _, local_limits, _ = runner.load_profile(ROOT/'configs/deletion_5080.json')
    assert cfg.batch_size * cfg.accumulation_steps == desktop.batch_size * desktop.accumulation_steps == 128
    assert cfg.batch_size == 32 and limits.max_session_minutes is None and limits.duty_cycle == 1
    assert cfg.output_root != desktop.output_root and local_limits.max_session_minutes == 120


def test_unlimited_session_still_stops_for_temperature(monkeypatch, tmp_path):
    import coco_kd.deletion_resources as resources
    current = available_hardware()
    monkeypatch.setattr(resources, 'hardware', lambda *_: current)
    guard = ResourceGuard(Limits(max_session_minutes=None), tmp_path, torch.device('cuda:0'))
    guard.started -= 100000000
    guard.tick(force=True)
    current['gpu']['temperature_c'] = 90
    with pytest.raises(ResourceStop, match='temperature'):
        guard.tick(force=True)
    with pytest.raises(ValueError, match='Session limit'):
        Limits(max_session_minutes=0)


def test_ampere_a5000_does_not_require_blackwell_runtime(monkeypatch, tmp_path):
    cfg, policy, limits, source = runner.load_profile(ROOT/'configs/deletion_server.json')
    cfg = replace(cfg, data_root=str(tmp_path/'missing'), output_root=str(tmp_path/'out'))
    monkeypatch.setattr(runner, 'versions', lambda: {'torch': '2.5.1+cu124'})
    monkeypatch.setattr(torch.version, 'cuda', '12.4')
    monkeypatch.setattr(runner, 'hardware', lambda *_: available_hardware())
    result = runner.check(cfg, policy, limits, source, tmp_path/'missing.pt')
    assert not any('CUDA 12.8' in b for b in result['blockers_before_student_training'])
    monkeypatch.setattr(runner, 'hardware', lambda *_: available_hardware('NVIDIA GeForce RTX 5080'))
    result = runner.check(cfg, policy, limits, source, tmp_path/'missing.pt')
    assert any('CUDA 12.8' in b for b in result['blockers_before_student_training'])


def test_strict_check_exits_before_training(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(runner, 'check', lambda *_: {'blockers_before_student_training': ['missing teacher']})
    with pytest.raises(SystemExit) as error:
        runner.main(['check', '--strict', '--config', 'configs/deletion_server.json'])
    assert error.value.code == 2


def test_background_start_does_not_continue_after_failed_check(tmp_path):
    shutil.copyfile(ROOT/'setup_deletion_server.sh', tmp_path/'setup_deletion_server.sh')
    runtime = tmp_path/'.venv/bin/python'
    runtime.parent.mkdir(parents=True)
    runtime.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$3" >> stages\n[[ "$3" != check ]]\n')
    runtime.chmod(0o755)
    result = subprocess.run(['bash', str(tmp_path/'setup_deletion_server.sh'), 'start'],
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert (tmp_path/'stages').read_text().splitlines() == ['check']


def test_server_unlimited_run_dispatches_all_methods(monkeypatch, tmp_path):
    cfg, policy, limits, source = runner.load_profile(ROOT/'configs/deletion_server.json')
    cfg = replace(cfg, output_root=str(tmp_path/'out'))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'load_profile', lambda *_: (cfg, policy, limits, source))
    monkeypatch.setattr(runner, 'validate_manifest', lambda *_: {})
    monkeypatch.setattr(runner, 'sha256', lambda *_: 'same-manifest')
    monkeypatch.setattr(runner, 'require_benchmark', lambda *_: None)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    calls = []
    monkeypatch.setattr(runner, 'train_one', lambda c, p, lim, method, *_: calls.append((method, lim.max_session_minutes)))
    runner.main(['run', '--teacher-checkpoint', '/trusted/existing/teacher.pt'])
    assert calls == [(method, None) for method in runner.DEFAULT_METHODS]


def test_reused_teacher_must_match_data_seed_size_and_class_count(monkeypatch):
    cfg, _, _, _ = runner.load_profile(ROOT/'configs/deletion_server.json')
    monkeypatch.setattr(runner, 'metadata_hash', lambda *_: 'dataset')
    state = {'role': 'teacher', 'metadata_sha256': 'dataset', 'epoch': 18, 'config': cfg.to_dict()}
    runner.validate_teacher_state(cfg, state)
    for key, value in [('seed', 2), ('teacher_variant', 'small'), ('num_classes', 2)]:
        bad = {**state, 'config': {**state['config'], key: value}}
        with pytest.raises(ValueError):
            runner.validate_teacher_state(cfg, bad)


def test_benchmark_from_other_gpu_is_rejected(monkeypatch, tmp_path):
    from dataclasses import asdict
    cfg, policy, limits, _ = runner.load_profile(ROOT/'configs/deletion_server.json')
    cfg = replace(cfg, output_root=str(tmp_path))
    report = {'experiment': cfg.to_dict(), 'audit': asdict(policy), 'resources': asdict(limits),
              'versions': runner.versions(), 'code_sha256': runner.code_hash(),
              'hardware': available_hardware('NVIDIA GeForce RTX 5080')}
    (tmp_path/'benchmark.json').write_text(json.dumps(report))
    monkeypatch.setattr(runner, 'hardware', lambda *_: available_hardware())
    with pytest.raises(ValueError, match='Benchmark GPU changed'):
        runner.require_benchmark(cfg, policy, limits)
