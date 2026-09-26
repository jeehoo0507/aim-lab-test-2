"""Measure capacity, then run independent student seeds with one fixed teacher."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict, replace
import gc
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import signal
import statistics
import time
import traceback

import torch

from . import deletion_run as run
from .data import validate_manifest
from .deletion import METHODS
from .deletion_resources import ResourceGuard, ResourceStop, hardware, check_resources
from .utils import RunLock, sha256, write_json


def capacity(info, cfg, limits, peak_gib, max_jobs, seed_count):
    """Reserve the full allocator cap of every worker plus a CUDA-context margin."""
    cap = min(limits.gpu_memory_gib, max(3.0, math.ceil((1.5*peak_gib+0.5)*2)/2))
    slot_gib = cap+1.0
    gpu_slots = math.floor((info['gpu']['free_gib']-limits.min_gpu_free_gib)/slot_gib)
    ram_slots = math.floor((info['ram']['available_gib']-limits.min_ram_available_gib)/3.0)
    cpu_slots = max(1, (info['logical_cpus']-2)//cfg.num_threads)
    jobs = max(0, min(max_jobs, seed_count, gpu_slots, ram_slots, cpu_slots))
    return {'capacity_jobs': jobs, 'allocator_cap_gib_per_worker': cap,
            'reserved_gib_per_worker_with_context': slot_gib,
            'ram_budget_gib_per_worker': 3.0, 'gpu_slots': gpu_slots,
            'ram_slots': ram_slots, 'cpu_slots': cpu_slots}


def admission(info, cfg, limits, reservation, jobs):
    errors = check_resources(info, limits)
    required = jobs*reservation['reserved_gib_per_worker_with_context']+limits.min_gpu_free_gib
    if info.get('gpu', {}).get('free_gib', 0) < required:
        errors.append(f'Need {required:.2f} GiB free including all worker caps and reserve')
    if info['ram']['available_gib'] < limits.min_ram_available_gib+jobs*reservation['ram_budget_gib_per_worker']:
        errors.append('Not enough RAM for all worker budgets')
    if jobs > max(1, (info['logical_cpus']-2)//cfg.num_threads):
        errors.append('Not enough CPU threads for this worker count')
    return errors


def estimate_trial(reports, train_count, epochs, seed_count):
    # One worker runs all methods for one seed; waves account for queued seeds.
    hours = [sum(r['seconds_per_image'] for r in report['rows'])*train_count*epochs/3600
             for report in reports]
    wall = max(hours)*math.ceil(seed_count/len(reports))
    return {'estimated_train_only_wall_hours': wall, 'planning_wall_hours': [1.25*wall, 1.75*wall],
            'slowest_seed_train_only_hours': max(hours),
            'peak_reserved_gib_per_worker': max(r['peak_reserved_gib'] for p in reports for r in p['rows'])}


def choose_trial(trials):
    passed = sorted((t for t in trials if t['status'] == 'passed'), key=lambda t: t['jobs'])
    if not passed:
        raise RuntimeError('No parallel calibration trial passed')
    best = passed[0]
    for trial in passed[1:]:
        # Extra processes must improve whole-study throughput by at least 5%.
        if trial['estimated_train_only_wall_hours'] < best['estimated_train_only_wall_hours']*0.95:
            best = trial
    return best


def _probe_worker(cfg, policy, limits, methods, steps, barrier):
    root = Path(cfg.output_root)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'worker.log').open('w', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        try:
            run.benchmark(cfg, policy, limits, steps=steps, methods=methods, barrier=barrier, warmup=1)
        except BaseException:
            barrier.abort()
            traceback.print_exc()
            raise


def _train_worker(cfg, policy, limits, methods, checkpoint, teacher_seed, until_epoch):
    root = Path(cfg.output_root)/f'seed_{cfg.seed}'
    root.mkdir(parents=True, exist_ok=True)
    with (root/'worker.log').open('a', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        try:
            with RunLock(root/'.worker.lock'):
                for method in methods:
                    run.train_one(cfg, policy, limits, method, checkpoint, until_epoch, teacher_seed)
                    gc.collect()
                    if cfg.device.startswith('cuda'):
                        torch.cuda.empty_cache()
        except BaseException:
            traceback.print_exc()
            raise


def stop_workers(processes):
    # Only children created by this invocation; never touch unrelated GPU jobs.
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)


def wait_workers(processes, guard):
    handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        handlers[sig] = signal.signal(sig, lambda *_: setattr(guard, 'stop_requested', True))
    try:
        while any(p.is_alive() for p in processes):
            guard.tick()
            if any(p.exitcode not in (None, 0) for p in processes):
                stop_workers(processes)
                break
            time.sleep(.5)
        for process in processes:
            process.join()
        return all(p.exitcode == 0 for p in processes)
    except BaseException:
        stop_workers(processes)
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def probe_trial(cfg, policy, limits, methods, steps, jobs):
    trial_root = Path(cfg.output_root)/'_calibration'/f'jobs_{jobs}'
    context = mp.get_context('spawn')  # No inherited CUDA context.
    barrier = context.Barrier(jobs)
    processes, directories = [], []
    guard = ResourceGuard(limits, cfg.output_root, torch.device(cfg.device))
    try:
        for rank in range(jobs):
            directory = trial_root/f'worker_{rank}'
            directory.mkdir(parents=True, exist_ok=True)
            (directory/'benchmark.json').unlink(missing_ok=True)
            local = replace(cfg, output_root=str(directory))
            process = context.Process(target=_probe_worker, args=(local, policy, limits, methods, steps, barrier))
            process.start()
            processes.append(process)
            directories.append(directory)
        if not wait_workers(processes, guard):
            return None
        return [json.loads((p/'benchmark.json').read_text()) for p in directories]
    except BaseException:
        stop_workers(processes)
        raise


def calibrate(cfg, policy, limits, methods, seeds, max_jobs, steps, train_count):
    info = hardware(cfg.output_root, torch.device(cfg.device).index or 0)
    errors = check_resources(info, limits)
    if errors:
        raise ResourceStop('; '.join(errors))
    print('CALIBRATE single worker (temporary models, no research checkpoints)', flush=True)
    first = probe_trial(cfg, policy, limits, methods, steps, 1)
    if first is None:
        raise RuntimeError('Single-worker benchmark failed; inspect _calibration/jobs_1/worker_0/worker.log')
    trial = {'jobs': 1, 'status': 'passed', **estimate_trial(first, train_count, cfg.epochs, len(seeds))}
    peak = trial['peak_reserved_gib_per_worker']
    info = hardware(cfg.output_root, torch.device(cfg.device).index or 0)
    reservation = capacity(info, cfg, limits, peak, max_jobs, len(seeds))
    if reservation['capacity_jobs'] < 1:
        raise ResourceStop('Not enough free GPU/RAM/CPU capacity after reserving measured worker budgets')
    worker_limits = replace(limits, gpu_memory_gib=reservation['allocator_cap_gib_per_worker'])
    trials = [trial]
    for jobs in range(2, reservation['capacity_jobs']+1):
        info = hardware(cfg.output_root, torch.device(cfg.device).index or 0)
        errors = admission(info, cfg, worker_limits, reservation, jobs)
        if errors:
            trials.append({'jobs': jobs, 'status': 'skipped', 'reason': '; '.join(errors)})
            break
        print(f'CALIBRATE {jobs} simultaneous workers; synchronized timing', flush=True)
        reports = probe_trial(cfg, policy, worker_limits, methods, steps, jobs)
        if reports is None:
            trials.append({'jobs': jobs, 'status': 'failed', 'logs': f'_calibration/jobs_{jobs}/'})
            break
        trials.append({'jobs': jobs, 'status': 'passed', **estimate_trial(reports, train_count, cfg.epochs, len(seeds))})
    selected = choose_trial(trials)
    plan = {'selected_jobs': selected['jobs'], 'student_seeds': seeds, 'methods': methods,
            'reservation': reservation, 'trials': trials, 'selected_estimate': selected,
            'worker_resources': asdict(worker_limits), 'hardware': info,
            'code_sha256': run.code_hash(), 'versions': run.versions(),
            'note': 'Synthetic joint timing, not a confidence interval. Real IO/validation/deletion acceptance differ. Estimates cover full runs, not remaining work.'}
    write_json(Path(cfg.output_root)/'parallel_plan.json', plan)
    print(json.dumps(plan, indent=2), flush=True)
    return plan, worker_limits


def ensure_protocol(cfg, policy, methods, seeds, checkpoint, teacher_seed):
    _, shared = run.identity(cfg, policy, 'parallel-study', checkpoint, teacher_seed)
    protocol = {'study': 'audited_deletion_fixed_teacher', 'student_seeds': seeds,
                'teacher_seed': teacher_seed, 'methods': methods, 'provenance': shared,
                'interpretation': 'Student initialization/training variability conditional on one fixed teacher, not independent teacher retraining.'}
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    path = Path(cfg.output_root)/'parallel_protocol.json'
    if path.exists() and json.loads(path.read_text())['signature'] != signature:
        raise ValueError('Parallel study settings/code/runtime/data/teacher changed; choose a new output root')
    write_json(path, {**protocol, 'signature': signature})


def train_seeds(cfg, policy, limits, methods, seeds, checkpoint, teacher_seed, plan, until_epoch):
    context = mp.get_context('spawn')
    guard = ResourceGuard(limits, cfg.output_root, torch.device(cfg.device))
    jobs = plan['selected_jobs']
    for start in range(0, len(seeds), jobs):
        wave = seeds[start:start+jobs]
        info = hardware(cfg.output_root, torch.device(cfg.device).index or 0)
        errors = admission(info, cfg, limits, plan['reservation'], len(wave))
        if errors:
            raise ResourceStop('; '.join(errors))
        print(f'START students seeds={wave}; fixed teacher seed={teacher_seed}', flush=True)
        processes = []
        try:
            for seed in wave:
                local = replace(cfg, seed=seed)
                process = context.Process(target=_train_worker,
                    args=(local, policy, limits, methods, checkpoint, teacher_seed, until_epoch))
                process.start()
                processes.append(process)
            if not wait_workers(processes, guard):
                raise RuntimeError('A worker stopped; other workers were stopped. Inspect seed_*/worker.log and STOPPED.json, then rerun explicitly.')
        except BaseException:
            stop_workers(processes)
            raise
    print(f'COMPLETE student seeds={seeds}; test not evaluated. Next: bash setup_deletion_server.sh parallel-evaluate', flush=True)


def evaluate_seeds(cfg, policy, limits, methods, seeds, checkpoint, teacher_seed):
    all_results = []
    for seed in seeds:
        local = replace(cfg, seed=seed)
        run.evaluate_runs(local, policy, limits, methods, checkpoint, teacher_seed)
        for method in methods:
            results = json.loads((run.run_directory(local, method)/'test_metrics.json').read_text())
            all_results.append({'seed': seed, 'method': method, 'test': results})
    summary = []
    for method in methods:
        rows = [r for r in all_results if r['method'] == method]
        for checkpoint_name in ('last', 'best'):
            scores = [r['test'][checkpoint_name]['macro_accuracy'] for r in rows]
            summary.append({'method': method, 'checkpoint': checkpoint_name, 'n_student_seeds': len(scores),
                'test_macro_mean': statistics.mean(scores),
                'test_macro_sample_std': statistics.stdev(scores) if len(scores) > 1 else None})
    write_json(Path(cfg.output_root)/'multi_seed_results.json', {'teacher_seed': teacher_seed,
               'student_seeds': seeds, 'summary': summary, 'per_seed': all_results})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'calibrate', 'evaluate'))
    parser.add_argument('--config', default='configs/deletion_server.json')
    parser.add_argument('--data-root')
    parser.add_argument('--output-root', default='outputs/deletion_server_multi')
    parser.add_argument('--teacher-checkpoint', default='outputs/teacher_base_pilot/seed_0/teacher/best.pt')
    parser.add_argument('--teacher-seed', type=int, default=0)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(run.DEFAULT_METHODS))
    parser.add_argument('--max-jobs', type=int, default=3)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--until-epoch', type=int)
    args = parser.parse_args(argv)
    if not 1 <= args.max_jobs <= 3 or not 1 <= args.steps <= 8:
        parser.error('max-jobs must be 1..3 and steps 1..8')
    if (len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0
            or len(set(args.methods)) != len(args.methods) or args.teacher_seed < 0
            or args.until_epoch is not None and args.until_epoch < 1):
        parser.error('Seeds/methods must be unique; seeds nonnegative and until-epoch positive')
    cfg, policy, limits, source = run.load_profile(args.config)
    cfg = replace(cfg, data_root=args.data_root or cfg.data_root, output_root=args.output_root, seed=0)
    if not cfg.device.startswith('cuda'):
        parser.error('Real parallel experiments require CUDA')
    root = Path(cfg.output_root)
    with RunLock(root/'.deletion.lock'):
        result = run.check(cfg, policy, limits, source, args.teacher_checkpoint, args.teacher_seed)
        if result['blockers_before_student_training']:
            raise RuntimeError('Resolve preflight blockers before starting parallel workers')
        manifest = validate_manifest(cfg.data_root)
        if sha256(Path(cfg.data_root)/'manifest.json') != sha256(source):
            raise ValueError('Use the frozen experiment-2 dataset')
        ensure_protocol(cfg, policy, args.methods, args.seeds, args.teacher_checkpoint, args.teacher_seed)
        if args.action == 'evaluate':
            evaluate_seeds(cfg, policy, limits, args.methods, args.seeds, args.teacher_checkpoint, args.teacher_seed)
            return
        plan, worker_limits = calibrate(cfg, policy, limits, args.methods, args.seeds,
                                       args.max_jobs, args.steps, sum(r['split'] == 'train' for r in manifest['images']))
        if args.action == 'start':
            train_seeds(cfg, policy, worker_limits, args.methods, args.seeds,
                        args.teacher_checkpoint, args.teacher_seed, plan, args.until_epoch)


if __name__ == '__main__':
    try:
        main()
    except ResourceStop as error:
        print(f'STOPPED: {error}. Resume explicitly after resolving the cause.', flush=True)
        raise SystemExit(75)
