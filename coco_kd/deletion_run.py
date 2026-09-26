"""Opt-in sequential deletion experiments. No training is started by check/env/prepare."""
import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

from .config import Config, CLASSES
from .data import CocoSubset, loader, validate_manifest
from .deletion import AuditSettings, METHODS, deletion_targets, student_forward
from .deletion_resources import Limits, ResourceGuard, ResourceStop, hardware, check_resources
from .metrics import classification
from .models import build_model
from .train import distillation_loss, learning_rate, optimizer_for
from .utils import (RunLock, autocast, cpu_state, load_checkpoint, metadata_hash,
                    model_fingerprint, save_checkpoint, save_npz, seed_all, sha256, write_json)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METHODS = ('student', 'random_rescue_10', 'rescue_audit', 'deterministic_audit')


def versions():
    return {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'numpy', 'Pillow')}


def load_profile(path):
    raw = json.loads(Path(path).read_text())
    if set(raw)-{'experiment', 'audit', 'resources', 'dataset_manifest'}:
        raise ValueError('Unknown profile section')
    cfg = Config.load(**raw['experiment'])
    if cfg.num_workers != 0 or cfg.num_threads > 2:
        raise ValueError('Deletion profiles require workers=0 and at most 2 CPU threads')
    return cfg, AuditSettings(**raw['audit']), Limits(**raw['resources']), raw['dataset_manifest']


def code_hash():
    digest = hashlib.sha256()
    for name in ('deletion.py', 'deletion_run.py', 'deletion_resources.py', 'models.py', 'train.py',
                 'data.py', 'config.py', 'masking.py', 'utils.py', 'metrics.py', 'deletion_parallel.py'):
        digest.update(name.encode())
        digest.update((ROOT/'coco_kd'/name).read_bytes())
    return digest.hexdigest()


def identity(cfg, policy, method, teacher_file=None, teacher_seed=None):
    values = cfg.to_dict()
    for key in ('data_root', 'output_root', 'device'):
        values.pop(key)
    record = {'experiment': values, 'audit': asdict(policy), 'method': method,
              'data_sha256': metadata_hash(cfg), 'teacher_sha256': sha256(teacher_file) if teacher_file else None,
              'teacher_seed': (cfg.seed if teacher_seed is None else teacher_seed) if teacher_file else None,
              'code_sha256': code_hash(), 'versions': versions()}
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(), record


def teacher_path(cfg, override=None):
    return Path(override) if override else Path(cfg.output_root)/f'seed_{cfg.seed}/teacher/best.pt'


def run_directory(cfg, method):
    root = Path(cfg.output_root)/f'seed_{cfg.seed}'
    return root/'teacher' if method == 'teacher' else root/cfg.student_init/method


def validate_teacher_state(cfg, state, teacher_seed=None):
    saved = state.get('config', {})
    if state.get('role') != 'teacher' or state.get('metadata_sha256') != metadata_hash(cfg):
        raise ValueError('Teacher is not an experiment-2 checkpoint for this exact dataset')
    expected_seed = cfg.seed if teacher_seed is None else teacher_seed
    if saved.get('teacher_variant', 'small') != cfg.teacher_variant or saved.get('seed') != expected_seed:
        raise ValueError('Teacher size/seed mismatch')
    if int(state.get('epoch', 0)) < 1:
        raise ValueError('An untrained teacher checkpoint cannot supervise students')
    if saved.get('num_classes') != cfg.num_classes or saved.get('model_scale', 'deit') != cfg.model_scale:
        raise ValueError('Teacher class count/model scale mismatch')


def get_teacher(cfg, path, device, teacher_seed=None):
    state = load_checkpoint(path)
    validate_teacher_state(cfg, state, teacher_seed)
    teacher = build_model('teacher', cfg).to(device)
    teacher.load_state_dict(state['model'])
    return teacher.requires_grad_(False).eval()


def check(cfg, policy, limits, source_manifest, checkpoint, teacher_seed=None):
    info = hardware(cfg.output_root)
    blockers = check_resources(info, limits)
    installed = versions()
    major_minor = tuple(int(v) for v in installed['torch'].split('+')[0].split('.')[:2])
    # RTX A5000 is Ampere; only GeForce RTX 50xx requires the Blackwell runtime.
    is_blackwell = re.search(r'\bRTX\s+50\d0\b', info.get('gpu', {}).get('name', ''), re.I)
    if is_blackwell and (major_minor < (2, 7) or torch.version.cuda is None or float(torch.version.cuda) < 12.8):
        blockers.append('RTX 5080 requires the separate CUDA 12.8 environment: bash setup_deletion.sh env')
    manifest = Path(cfg.data_root)/'manifest.json'
    if not manifest.exists():
        blockers.append('COCO data absent: bash setup_deletion.sh prepare')
    else:
        try:
            data = validate_manifest(cfg.data_root, verify_files=False)
            missing = sum(not (Path(cfg.data_root)/row[key]).is_file() for row in data['images'] for key in ('image', 'mask'))
            if missing:
                blockers.append(f'{missing} image/mask files missing; run prepare')
            if sha256(manifest) != sha256(source_manifest):
                blockers.append('Dataset manifest differs from the frozen experiment-2 report')
        except (ValueError, OSError) as error:
            blockers.append(str(error))
    if not Path(checkpoint).is_file():
        blockers.append('Trained teacher absent: supply --teacher-checkpoint or explicitly run teacher')
    elif manifest.is_file():
        try:
            validate_teacher_state(cfg, load_checkpoint(checkpoint), teacher_seed)
        except (ValueError, OSError, RuntimeError, KeyError) as error:
            blockers.append(f'Teacher checkpoint incompatible: {error}')
    result = {'hardware': info, 'versions': installed, 'cuda_build': torch.version.cuda,
              'experiment': cfg.to_dict(), 'audit': asdict(policy), 'resources': asdict(limits),
              'sequential_methods': list(DEFAULT_METHODS), 'effective_batch': cfg.batch_size*cfg.accumulation_steps,
              'teacher_checkpoint': str(checkpoint), 'blockers_before_student_training': blockers,
              'teacher_seed': cfg.seed if teacher_seed is None else teacher_seed,
              'training_started': False, 'note': 'Duty cycle inserts idle time; it is not a GPU utilization or power cap.'}
    write_json(Path(cfg.output_root)/'preflight.json', result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def restore_data(cfg, limits, source):
    """Download only recorded IDs; reproduce and verify the original manifest."""
    from concurrent.futures import ThreadPoolExecutor
    from .prepare import ANNOTATIONS, strict_candidates, quality_record
    source = Path(source)
    data = json.loads(source.read_text())
    if data.get('classes') != list(CLASSES):
        raise ValueError('Expected experiment-2 COCO classes')
    for row in data['images']:
        for key in ('image', 'mask'):
            path = Path(row[key])
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe dataset path')
    root = Path(cfg.data_root)
    root.mkdir(parents=True, exist_ok=True)
    if (root/'manifest.json').exists():
        if sha256(root/'manifest.json') != sha256(source):
            raise ValueError('Refusing to replace a different existing dataset manifest')
        try:
            validate_manifest(root)
            print('Dataset already verified')
            return
        except (FileNotFoundError, ValueError):
            pass
    guard = ResourceGuard(limits, root)
    guard.tick(force=True)
    archive = root/'annotations_trainval2017.zip'
    if not archive.exists():
        part = archive.with_suffix('.zip.part')
        subprocess.run(['curl', '--fail', '--location', '--retry', '3', '--limit-rate', '20M',
                        '--continue-at', '-', '--output', str(part), ANNOTATIONS], check=True)
        if sha256(part) != data['annotations_sha256']:
            raise ValueError('Downloaded annotations differ from the frozen experiment')
        os.replace(part, archive)
    if sha256(archive) != data['annotations_sha256']:
        raise ValueError('Annotation checksum mismatch')
    wanted = {row['id']: row for row in data['images']}
    restored = set()
    with zipfile.ZipFile(archive) as z:
        for split in ('train2017', 'val2017'):
            info = json.loads(z.read(f'annotations/instances_{split}.json'))
            items = [(split, *row) for row in strict_candidates(info) if row[0]['id'] in wanted]
            del info
            pool = ThreadPoolExecutor(max_workers=2)
            try:
                for actual in pool.map(lambda item: quality_record(root, item), items):
                    guard.tick()
                    expected = wanted[actual['id']]
                    for key in ('label', 'image_sha256', 'mask_sha256', 'pixel_sha256'):
                        if actual[key] != expected[key]:
                            raise ValueError(f'Dataset reconstruction mismatch: {actual["id"]} {key}')
                    restored.add(actual['id'])
                    if len(restored) % 100 == 0:
                        print(f'DATA {len(restored)}/{len(wanted)} verified', flush=True)
            except BaseException:
                guard.stop_requested = True
                raise
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
    if restored != set(wanted):
        raise ValueError('Some manifest images were not reconstructed')
    temporary = root/'manifest.json.tmp'
    shutil.copyfile(source, temporary)
    os.replace(temporary, root/'manifest.json')
    validate_manifest(root)
    print(f'Dataset ready: {len(wanted)} original image IDs; no re-splitting')


@torch.no_grad()
def evaluate_guarded(model, dataset, cfg, device, guard, destination=None):
    model.eval()
    outputs, labels, ids = [], [], []
    for batch in loader(dataset, cfg):
        guard.tick()
        with autocast(cfg, device):
            result = model(batch['image'].to(device))
        outputs.append(result.float().cpu().numpy())
        labels.append(batch['label'].numpy())
        ids.append(batch['sample_id'].numpy())
        guard.pace()
    values, targets = np.concatenate(outputs), np.concatenate(labels)
    if destination:
        save_npz(destination, logits=values, label=targets, sample_id=np.concatenate(ids))
    return classification(values, targets, cfg.num_classes)


def training_epoch(model, teacher, dataset, optimizer, scaler, cfg, policy, device, guard, method, epoch, trace=None):
    model.train()
    seed_all(cfg.seed+epoch*100003)
    generator = torch.Generator(device=device).manual_seed(cfg.seed+epoch*200003)
    batches = loader(dataset, cfg, train=True, epoch=epoch)
    total = min(len(batches), cfg.max_train_batches or len(batches))
    sums, seen, updates, skipped = {}, 0, 0, 0
    begun = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(batches):
        if step >= total:
            break
        guard.tick()
        x, y = batch['image'].to(device), batch['label'].to(device)
        details = []
        with autocast(cfg, device):
            if method == 'teacher':
                logits, targets, metrics = model(x), None, {}
            else:
                logits, attention, features = student_forward(model, x)
                targets, metrics = deletion_targets(teacher, x, y, logits, attention, features, cfg,
                                                   policy, method, generator, guard, details.append if trace else None)
            loss, ce, kd = distillation_loss(logits, y, targets, cfg)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Non-finite training loss')
        start = step//cfg.accumulation_steps*cfg.accumulation_steps
        end = min(start+cfg.accumulation_steps, total)
        samples = min(end*cfg.batch_size, len(dataset))-start*cfg.batch_size
        scaler.scale(loss*len(x)/samples).backward()
        if step+1 == end:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=not scaler.is_enabled())
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < scale)
            updates += int(scaler.get_scale() >= scale)
            optimizer.zero_grad(set_to_none=True)
        metrics.update(loss=float(loss.detach()), ce=float(ce), kd=float(kd), accuracy=float((logits.argmax(1)==y).float().mean()))
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0)+value*len(x)
        seen += len(x)
        if trace:
            for record in details:
                row = record.pop('batch_row')
                trace.write(json.dumps({'seed': cfg.seed, 'epoch': epoch, 'step': step,
                    'sample_id': int(batch['sample_id'][row]), **record})+'\n')
        guard.pace()
        if step % 25 == 0:
            print(f'{method} epoch={epoch} batch={step+1}/{total} loss={float(loss.detach()):.4f}', flush=True)
    return {**{k: v/seen for k, v in sums.items()}, 'samples': seen, 'updates': updates,
            'amp_skipped_updates': skipped, 'seconds': time.monotonic()-begun}


def train_one(cfg, policy, limits, method, checkpoint=None, until_epoch=None, teacher_seed=None):
    role = 'teacher' if method == 'teacher' else 'student'
    if role == 'student' and not Path(checkpoint).is_file():
        raise FileNotFoundError('Trained teacher missing; use teacher action or --teacher-checkpoint')
    directory = run_directory(cfg, method)
    signature, provenance = identity(cfg, policy, method, checkpoint if role == 'student' else None, teacher_seed)
    directory.mkdir(parents=True, exist_ok=True)
    done = directory/'result.json'
    if done.exists():
        if json.loads(done.read_text())['signature'] != signature:
            raise ValueError('Completed run identity changed; choose a new output root')
        print(f'REUSED {directory}')
        return
    previous = load_checkpoint(directory/'last.pt') if (directory/'last.pt').exists() else None
    if previous and previous.get('signature') != signature:
        raise ValueError('Resume settings/data/teacher/code/runtime changed; choose a new output root')
    from .utils import setup_device
    device = setup_device(cfg)
    guard = ResourceGuard(limits, directory, device)
    guard.configure()
    seed_all(cfg.seed)
    model = build_model(role, cfg, pretrained=previous is None and (cfg.teacher_pretrained if role == 'teacher' else cfg.student_init == 'imagenet')).to(device)
    initial_hash = previous['initial_model_sha256'] if previous else model_fingerprint(model)
    teacher = get_teacher(cfg, checkpoint, device, teacher_seed) if role == 'student' else None
    optimizer = optimizer_for(model, cfg)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp and device.type == 'cuda')
    best_weights, best_score, best_epoch, history, start = cpu_state(model), -1.0, 0, [], 0
    if previous:
        model.load_state_dict(previous['model'])
        optimizer.load_state_dict(previous['optimizer'])
        scaler.load_state_dict(previous['scaler'])
        best_weights, best_score, best_epoch = previous['best_model'], previous['best_score'], previous['best_epoch']
        history, start = previous['history'], previous['epoch']
    base = {'signature': signature, 'provenance': provenance, 'config': cfg.to_dict(), 'audit': asdict(policy),
            'role': role, 'method': method, 'metadata_sha256': metadata_hash(cfg),
            'teacher_sha256': provenance['teacher_sha256'], 'initial_model_sha256': initial_hash}
    write_json(directory/'config.json', {'experiment': cfg.to_dict(), 'audit': asdict(policy), 'resources': asdict(limits)})
    def commit(epoch):
        save_checkpoint(directory/'last.pt', {**base, 'epoch': epoch, 'model': cpu_state(model),
                         'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'history': history,
                         'best_model': best_weights, 'best_score': best_score, 'best_epoch': best_epoch})
        save_checkpoint(directory/'best.pt', {**base, 'epoch': best_epoch, 'model': best_weights})
        write_json(directory/'history.json', history)
    if previous:
        # last.pt is authoritative after a crash between individual file writes.
        save_checkpoint(directory/'best.pt', {**base, 'epoch': best_epoch, 'model': best_weights})
        write_json(directory/'history.json', history)
    else:
        commit(0)
    del previous
    train_data = CocoSubset(cfg.data_root, 'train', train=True, threshold=cfg.foreground_threshold)
    validation = CocoSubset(cfg.data_root, 'val', threshold=cfg.foreground_threshold)
    epochs = cfg.teacher_epochs if role == 'teacher' else cfg.epochs
    end = min(epochs, until_epoch) if until_epoch is not None else epochs
    committed_epoch = start
    handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        handlers[sig] = signal.signal(sig, lambda *_: setattr(guard, 'stop_requested', True))
    try:
        for epoch in range(start+1, end+1):
            guard.tick(force=True)
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(cfg, epoch, epochs, role)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            begun = time.monotonic()
            trace_path = directory/'audit'/f'epoch_{epoch:03d}.jsonl'
            if method in ('rescue_audit', 'deterministic_audit'):
                trace_path.parent.mkdir(exist_ok=True)
                with trace_path.with_suffix('.part').open('w') as trace:
                    metrics = training_epoch(model, teacher, train_data, optimizer, scaler, cfg, policy, device, guard, method, epoch, trace)
            else:
                metrics = training_epoch(model, teacher, train_data, optimizer, scaler, cfg, policy, device, guard, method, epoch)
            val = evaluate_guarded(model, validation, cfg, device, guard)
            if val['macro_accuracy'] is None:
                raise ValueError('Validation must contain every class')
            if val['macro_accuracy'] > best_score:
                best_weights, best_score, best_epoch = cpu_state(model), val['macro_accuracy'], epoch
            history.append({'epoch': epoch, 'train': metrics, 'validation': val,
                            'seconds': time.monotonic()-begun,
                            'peak_reserved_gib': torch.cuda.max_memory_reserved(device)/1024**3 if device.type == 'cuda' else 0})
            if trace_path.with_suffix('.part').exists():
                os.replace(trace_path.with_suffix('.part'), trace_path)
            commit(epoch)
            committed_epoch = epoch
            print(f'COMMITTED {method} epoch {epoch}/{epochs}: val_macro={val["macro_accuracy"]:.4f}', flush=True)
        if end >= epochs:
            write_json(done, {**base, 'last_epoch': epochs, 'best_epoch': best_epoch,
                       'best_validation_macro': best_score, 'partial_training': cfg.max_train_batches is not None,
                       'test_evaluated': False, 'seconds_total': sum(h['seconds'] for h in history)})
            save_checkpoint(directory/'last.pt', {**base, 'epoch': epochs, 'model': cpu_state(model)})
        (directory/'STOPPED.json').unlink(missing_ok=True)
    except (ResourceStop, KeyboardInterrupt, torch.cuda.OutOfMemoryError) as error:
        write_json(directory/'STOPPED.json', {'reason': str(error) or 'Keyboard interrupt',
                   'resume': 'Restart the same command. Uncommitted epoch is replayed.',
                   'completed_epoch': committed_epoch})
        raise ResourceStop(str(error) or 'Interrupted') from None
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def benchmark(cfg, policy, limits, steps=2, methods=None, barrier=None, warmup=0):
    """Temporary synthetic full-size models; no experiment checkpoints updated."""
    from .utils import setup_device
    device = setup_device(cfg)
    guard = ResourceGuard(limits, cfg.output_root, device)
    guard.configure()
    rows = []
    for method in (('teacher', *DEFAULT_METHODS) if methods is None else methods):
        seed_all(812)
        role = 'teacher' if method == 'teacher' else 'student'
        model = build_model(role, cfg).to(device).train()
        teacher = build_model('teacher', cfg).to(device).requires_grad_(False).eval() if role == 'student' else None
        optimizer = optimizer_for(model, cfg)
        scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp and device.type == 'cuda')
        generator = torch.Generator(device=device).manual_seed(771)
        images = torch.randn(cfg.batch_size, 3, 224, 224, device=device)
        labels = torch.arange(cfg.batch_size, device=device) % cfg.num_classes
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        begun = time.monotonic()
        for step in range(warmup+steps):
            if step == warmup:
                if barrier is not None:
                    barrier.wait(timeout=180)
                guard.last_work = time.monotonic()
                begun = time.monotonic()
            guard.tick()
            optimizer.zero_grad(set_to_none=True)
            with autocast(cfg, device):
                if teacher is None:
                    logits, targets = model(images), None
                else:
                    logits, attention, features = student_forward(model, images)
                    targets, _ = deletion_targets(teacher, images, labels, logits, attention, features,
                                                  cfg, policy, method, generator, guard)
                loss, _, _ = distillation_loss(logits, labels, targets, cfg)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            guard.pace()
        elapsed = time.monotonic()-begun
        seconds_per_image = elapsed/(steps*cfg.batch_size)
        rows.append({'method': method, 'seconds_per_image': seconds_per_image,
                     'train_only_hours_per_6000_image_epoch': seconds_per_image*6000/3600,
                     'peak_reserved_gib': torch.cuda.max_memory_reserved(device)/1024**3 if device.type == 'cuda' else 0})
        print(json.dumps(rows[-1]), flush=True)
        if role == 'student':
            del attention, features
        del model, teacher, optimizer, scaler, images, labels, logits, targets, loss
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    report = {'hardware': hardware(cfg.output_root), 'experiment': cfg.to_dict(), 'audit': asdict(policy),
              'resources': asdict(limits), 'versions': versions(), 'code_sha256': code_hash(), 'rows': rows,
              'measured_microbatches': steps, 'warmup_microbatches': warmup,
              'note': 'Synthetic timing only. Early/late audit acceptance and data IO differ. Includes idle duty, excludes validation/download. Optimizer every microbatch makes optimizer cost conservative.'}
    write_json(Path(cfg.output_root)/'benchmark.json', report)
    return report


def require_benchmark(cfg, policy, limits):
    path = Path(cfg.output_root)/'benchmark.json'
    if not path.exists():
        raise ValueError('Run benchmark first and review its measured memory/time report')
    report = json.loads(path.read_text())
    saved, current = report['experiment'].copy(), cfg.to_dict()
    for key in ('data_root', 'output_root'):
        saved.pop(key, None)
        current.pop(key, None)
    if (saved != current or report['audit'] != json.loads(json.dumps(asdict(policy)))
            or report['versions'] != versions() or report.get('code_sha256') != code_hash()
            or any(report['resources'][k] != getattr(limits, k) for k in ('gpu_memory_gib', 'min_gpu_free_gib', 'duty_cycle'))):
        raise ValueError('Benchmark settings/code/runtime changed; run benchmark again')
    if cfg.device.startswith('cuda'):
        current_gpu = hardware(cfg.output_root, torch.device(cfg.device).index or 0).get('gpu', {})
        saved_gpu = report.get('hardware', {}).get('gpu', {})
        if not current_gpu or any(current_gpu.get(k) != saved_gpu.get(k) for k in ('name', 'total_gib')):
            raise ValueError('Benchmark GPU changed or unavailable; benchmark on this server first')


def evaluate_runs(cfg, policy, limits, methods, checkpoint, teacher_seed=None):
    from .utils import setup_device
    device = setup_device(cfg)
    guard = ResourceGuard(limits, cfg.output_root, device)
    guard.configure()
    test = CocoSubset(cfg.data_root, 'test')
    for method in methods:
        path = run_directory(cfg, method)
        done = json.loads((path/'result.json').read_text())
        signature, _ = identity(cfg, policy, method, checkpoint, teacher_seed)
        if done['signature'] != signature or done['partial_training']:
            raise ValueError('Complete matching runs are required before final test')
        model = build_model('student', cfg).to(device)
        results = {}
        for name in ('best', 'last'):
            state = load_checkpoint(path/f'{name}.pt')
            model.load_state_dict(state['model'])
            results[name] = {'epoch': state['epoch'], **evaluate_guarded(model, test, cfg, device, guard, path/f'test_{name}.npz')}
        write_json(path/'test_metrics.json', results)
        print(method, json.dumps(results), flush=True)
        del model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'prepare', 'benchmark', 'teacher', 'pilot', 'run', 'evaluate'))
    parser.add_argument('--config', default='configs/deletion_5080.json')
    parser.add_argument('--data-root')
    parser.add_argument('--output-root')
    parser.add_argument('--teacher-checkpoint')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--teacher-seed', type=int, help='Explicit frozen teacher seed; defaults to student seed')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument('--until-epoch', type=int)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--strict', action='store_true', help='check exits nonzero when student training has blockers')
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    cfg, policy, limits, source = load_profile(args.config)
    cfg = replace(cfg, data_root=args.data_root or cfg.data_root, output_root=args.output_root or cfg.output_root,
                  seed=cfg.seed if args.seed is None else args.seed)
    checkpoint = teacher_path(cfg, args.teacher_checkpoint)
    if args.until_epoch is not None and args.until_epoch < 1 or not 1 <= args.steps <= 8:
        parser.error('until-epoch must be positive; steps must be 1..8')
    if args.action == 'check':
        result = check(cfg, policy, limits, source, checkpoint, args.teacher_seed)
        if args.strict and result['blockers_before_student_training']:
            raise SystemExit(2)
        return
    with RunLock(Path(cfg.output_root)/'.deletion.lock'):
        if args.action == 'prepare':
            with RunLock(Path(cfg.data_root)/'.prepare.lock'):
                restore_data(cfg, limits, source)
            return
        if not cfg.device.startswith('cuda'):
            parser.error('Real experiment launcher requires CUDA; CPU is only used by unit tests')
        if args.action == 'benchmark':
            benchmark(cfg, policy, limits, args.steps)
            return
        validate_manifest(cfg.data_root)
        if sha256(Path(cfg.data_root)/'manifest.json') != sha256(source):
            raise ValueError('Use the frozen COCO experiment-2 manifest')
        if args.action == 'teacher':
            require_benchmark(cfg, policy, limits)
            train_one(cfg, policy, limits, 'teacher', until_epoch=args.until_epoch)
        elif args.action == 'evaluate':
            evaluate_runs(cfg, policy, limits, args.methods, checkpoint, args.teacher_seed)
        else:
            require_benchmark(cfg, policy, limits)
            if args.teacher_checkpoint is None and not (run_directory(cfg, 'teacher')/'result.json').exists():
                raise ValueError('Finish the teacher action before starting students, or supply a trusted --teacher-checkpoint')
            until = args.until_epoch or (2 if args.action == 'pilot' else None)
            session_start = time.monotonic()
            for method in args.methods:
                remaining = (None if limits.max_session_minutes is None else
                             limits.max_session_minutes-(time.monotonic()-session_start)/60)
                if remaining is not None and remaining <= 0:
                    raise ResourceStop('Session time limit reached before next method')
                train_one(cfg, policy, replace(limits, max_session_minutes=remaining), method, checkpoint, until, args.teacher_seed)
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    try:
        main()
    except ResourceStop as error:
        print(f'STOPPED: {error}. Last completed epoch is preserved.', file=sys.stderr)
        sys.exit(75)
