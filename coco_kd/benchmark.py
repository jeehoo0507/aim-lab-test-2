"""Measure one/two independent processes on the same GPU with real prepared data."""
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
from tempfile import TemporaryDirectory
import time
import traceback

import torch

from .config import Config, METHODS
from .data import CocoSubset, loader, validate_manifest
from .masking import select_tokens
from .models import build_model
from .probe import Probe
from .train import distillation_loss, evaluate, optimizer_for
from .utils import (autocast, cpu_state, metadata_hash, save_checkpoint, seed_all,
                    setup_device, source_fingerprint, write_json)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _hardware(device):
    if device.type != "cuda":
        return {"device": str(device)}
    p = torch.cuda.get_device_properties(device)
    return {"device": str(device), "name": p.name, "total_bytes": p.total_memory,
            "uuid": str(getattr(p, "uuid", "unavailable"))}


def _worker(values, rank, barrier, queue, directory, warmup, steps):
    try:
        cfg = Config.load(**values)
        seed_all(9123 + rank)
        device = setup_device(cfg)
        train_data = CocoSubset(cfg.data_root, "train", train=True, threshold=cfg.foreground_threshold)
        val_data = CocoSubset(cfg.data_root, "val", threshold=cfg.foreground_threshold)
        work = Path(directory) / f"worker_{rank}"
        work.mkdir()
        rows = []
        for method in ("teacher", *METHODS):
            role = "teacher" if method == "teacher" else "student"
            model = build_model(role, cfg).to(device)
            teacher = build_model("teacher", cfg).to(device).requires_grad_(False).eval() if role == "student" else None
            optimizer = optimizer_for(model, cfg)
            scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
            generator = torch.Generator(device=device).manual_seed(7221 + rank)
            batches = loader(train_data, cfg, train=True, epoch=1)
            iterator = iter(batches)

            def update():
                nonlocal iterator
                optimizer.zero_grad(set_to_none=True)
                count = 0
                for _ in range(cfg.accumulation_steps):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(batches)
                        batch = next(iterator)
                    x, y = batch["image"].to(device), batch["label"].to(device)
                    fg = batch["foreground"].to(device)
                    with autocast(cfg, device):
                        logits, attention = model(x, return_attention=True)
                        targets = None
                        if method not in ("ce", "teacher"):
                            indices, _ = select_tokens(attention, method, foreground=fg, generator=generator)
                            targets = teacher(x, indices)
                        loss, _, _ = distillation_loss(logits, y, targets, cfg)
                    scaler.scale(loss / cfg.accumulation_steps).backward()
                    count += len(x)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=not scaler.is_enabled())
                scaler.step(optimizer)
                scaler.update()
                return count

            for _ in range(warmup):
                update()
            _sync(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            barrier.wait(timeout=180)
            begun = time.perf_counter()
            samples = sum(update() for _ in range(steps))
            _sync(device)
            seconds = time.perf_counter() - begun
            barrier.wait(timeout=180)
            row = {"method": method, "rank": rank, "samples": samples,
                   "updates": steps, "seconds": seconds, "images_per_second": samples / seconds}
            if method in ("teacher", "student"):
                barrier.wait(timeout=180)
                begun = time.perf_counter()
                evaluate(model, val_data, cfg, device, work / "validation.npz")
                _sync(device)
                row["validation_seconds"] = time.perf_counter() - begun
                row["validation_bytes"] = (work / "validation.npz").stat().st_size
                barrier.wait(timeout=180)
                begun = time.perf_counter()
                save_checkpoint(work / "last.pt", {"model": cpu_state(model), "best_model": cpu_state(model),
                                                    "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()})
                _sync(device)
                row["checkpoint_seconds"] = time.perf_counter() - begun
                row["checkpoint_bytes"] = (work / "last.pt").stat().st_size
                (work / "last.pt").unlink()
                barrier.wait(timeout=180)
            if method == "student":
                from dataclasses import replace
                probe = Probe(replace(cfg, epochs=1000000, diagnostic_epochs=[]), work, teacher, device)
                probe.log(model, 1, "student")  # Populate full-teacher cache before timing.
                _sync(device)
                barrier.wait(timeout=180)
                begun = time.perf_counter()
                probe.log(model, 2, "student")
                _sync(device)
                row["probe_seconds"] = time.perf_counter() - begun
                row["probe_bytes"] = (work / "probe/epoch_002.npz").stat().st_size
                barrier.wait(timeout=180)
                probe.cfg.diagnostic_epochs = [3]
                begun = time.perf_counter()
                probe.log(model, 3, "student")
                _sync(device)
                row["diagnostic_seconds"] = time.perf_counter() - begun
                row["diagnostic_bytes"] = (work / "probe/epoch_003_counterfactual.npz").stat().st_size
                del probe
                barrier.wait(timeout=180)
            row["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else 0.0
            row["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
            rows.append(row)
            queue.put({"event": "progress", "rank": rank, "method": method})
            del model, teacher, optimizer, scaler, batches, iterator, update
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            barrier.wait(timeout=180)
        queue.put({"event": "done", "rank": rank, "rows": rows})
    except Exception:
        queue.put({"event": "error", "rank": rank, "error": traceback.format_exc()})
        try:
            barrier.abort()
        except Exception:
            pass


def run_trial(cfg, jobs, warmup=2, steps=8):
    context = mp.get_context("spawn")
    barrier, queue = context.Barrier(jobs), context.Queue()
    base = Path(cfg.output_root) / "_benchmark"
    base.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f"jobs{jobs}_", dir=base) as directory:
        workers = [context.Process(target=_worker, args=(cfg.to_dict(), rank, barrier, queue, directory, warmup, steps))
                   for rank in range(jobs)]
        rows, done = [], set()
        try:
            for worker in workers:
                worker.start()
            last_progress = time.monotonic()
            while len(done) < jobs:
                try:
                    message = queue.get(timeout=5)
                except Empty:
                    if any(p.exitcode is not None and p.exitcode != 0 for p in workers):
                        raise RuntimeError(f"Benchmark process failed: {[p.exitcode for p in workers]}")
                    if time.monotonic() - last_progress > 240:
                        raise TimeoutError("Benchmark made no progress for 240s; stopped only its own workers")
                    continue
                last_progress = time.monotonic()
                if message["event"] == "error":
                    raise RuntimeError(message["error"])
                if message["event"] == "done":
                    rows.extend(message["rows"])
                    done.add(message["rank"])
                else:
                    print(f"BENCH jobs={jobs}: {message['method']} worker={message['rank']} complete", flush=True)
            for worker in workers:
                worker.join(timeout=10)
            return {"jobs": jobs, "status": "passed", "rows": rows}
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                if worker.pid is not None:
                    worker.join(timeout=10)
            queue.close()


def _worst(rows, method, key):
    return max(r[key] for r in rows if r["method"] == method)


def time_per_seed(trial, cfg, train_count):
    rows = trial["rows"]
    train_time = {mode: train_count * max(r["seconds"] / r["samples"] for r in rows if r["method"] == mode)
                  for mode in ("teacher", *METHODS)}
    tv = _worst(rows, "teacher", "validation_seconds")
    tc = _worst(rows, "teacher", "checkpoint_seconds")
    sv = _worst(rows, "student", "validation_seconds")
    sc = _worst(rows, "student", "checkpoint_seconds")
    normal = _worst(rows, "student", "probe_seconds")
    detailed = _worst(rows, "student", "diagnostic_seconds")
    n_detail = len(set(e for e in cfg.diagnostic_epochs if 0 <= e <= cfg.epochs) | {0, cfg.epochs}) + 1
    teacher_seconds = cfg.teacher_epochs * (train_time["teacher"] + tv + tc)
    student_seconds = sum(cfg.epochs * (train_time[m] + sv + sc + normal)
                          + 2 * normal + n_detail * max(0, detailed - normal) for m in METHODS) * 2
    return {"teacher_seconds": teacher_seconds, "student_seconds": student_seconds,
            "total_seconds": teacher_seconds + student_seconds,
            "train_epoch_seconds_by_method": train_time}


def recommend(single, parallel, cfg, train_count, seeds=3):
    one = time_per_seed(single, cfg, train_count)
    result = {"recommended_jobs": 1, "serial_hours": seeds * one["total_seconds"] / 3600,
              "one_seed": one, "reason": "A valid, materially faster parallel trial is required"}
    if parallel is not None and parallel["status"] == "passed":
        two = time_per_seed(parallel, cfg, train_count)
        # Whole-seed scheduler: three seeds on two workers leave one single-seed tail.
        duration = (seeds // 2) * two["total_seconds"] + (seeds % 2) * one["total_seconds"]
        result.update(parallel_hours=duration / 3600,
                      aggregate_throughput_speedup=2 * one["total_seconds"] / two["total_seconds"],
                      full_plan_speedup=seeds * one["total_seconds"] / duration)
        if result["full_plan_speedup"] >= 1.10:
            result.update(recommended_jobs=2, reason="Projected full-plan time improves >=10%, with VRAM headroom")
        else:
            result["reason"] = "Parallel contention/tail saves <10%; prefer one worker"
    chosen = result.get("parallel_hours") if result["recommended_jobs"] == 2 else result["serial_hours"]
    result["planning_range_hours"] = [chosen, chosen * 1.5]
    result["excludes"] = "download, initial environment setup, final test/export; full 42-student/3-teacher plan, not remaining work"
    return result


def memory_requirement(single, total_gib):
    per_worker = max(r["peak_reserved_gib"] for r in single["rows"]) + .75
    return 2 * per_worker + max(2.0, .1 * total_gib)


def benchmark(cfg, destination, steps=8, warmup=2, max_jobs=2):
    if steps < 1 or warmup < 1 or max_jobs not in (1, 2):
        raise ValueError("Positive warmup/steps and max_jobs 1 or 2 required")
    manifest = validate_manifest(cfg.data_root)
    device = setup_device(cfg)
    train_count = sum(r["split"] == "train" for r in manifest["images"])
    report = {"schema": 2, "config": cfg.to_dict(), "metadata_sha256": metadata_hash(cfg),
              "code_sha256": source_fingerprint(), "measured_at_unix": time.time(),
              "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "hardware": _hardware(device),
              "warmup_updates": warmup, "measured_updates": steps,
              "training_images": train_count, "trials": [],
              "note": "Random weights, real prepared images. Temporary checkpoints. Data loading, accumulation and representative validation/probe/I/O included."}
    print("Benchmark: real images + model sizes; experiment checkpoints are untouched", flush=True)
    single = run_trial(cfg, 1, warmup, steps)
    report["trials"].append(single)
    write_json(destination, report)
    parallel = None
    if device.type == "cuda" and max_jobs == 2:
        free, total = torch.cuda.mem_get_info(device)
        needed = memory_requirement(single, total / 1024**3)
        report["parallel_required_free_gib"] = needed
        report["free_gib_before_parallel"] = free / 1024**3
        if free / 1024**3 >= needed:
            print("Enough VRAM headroom; comparing two concurrent workers", flush=True)
            try:
                parallel = run_trial(cfg, 2, warmup, steps)
                if memory_requirement(parallel, total / 1024**3) > free / 1024**3:
                    parallel["status"] = "insufficient_headroom"
            except (RuntimeError, TimeoutError) as error:
                parallel = {"jobs": 2, "status": "failed", "error": str(error)}
        else:
            parallel = {"jobs": 2, "status": "skipped_memory"}
        report["trials"].append(parallel)
    report["recommendation"] = recommend(single, parallel, cfg, train_count)
    if device.type != "cuda":
        report["recommendation"]["reason"] = "CPU benchmark: not an A5000 speed/concurrency estimate"
    from .system import checkpoint_estimate
    student_count = sum(p.numel() for p in build_model("student", cfg).parameters())
    teacher_count = sum(p.numel() for p in build_model("teacher", cfg).parameters())
    report["checkpoint_storage"] = checkpoint_estimate(cfg, student_count, teacher_count)
    probe_bytes = _worst(single["rows"], "student", "probe_bytes")
    diagnostic_bytes = _worst(single["rows"], "student", "diagnostic_bytes")
    n_detail = len(set(e for e in cfg.diagnostic_epochs if 0 <= e <= cfg.epochs) | {0, cfg.epochs}) + 1
    report["estimated_probe_validation_gib"] = (42 * ((cfg.epochs + 2) * probe_bytes + n_detail * diagnostic_bytes
            + cfg.epochs * _worst(single["rows"], "student", "validation_bytes"))
            + 3 * cfg.teacher_epochs * _worst(single["rows"], "teacher", "validation_bytes")) / 1024**3
    write_json(destination, report)
    print(json.dumps({"recommendation": report["recommendation"], "checkpoint_storage": report["checkpoint_storage"],
                      "probe_validation_gib": report["estimated_probe_validation_gib"]}, indent=2), flush=True)
    print("After pilot review: bash setup.sh run --jobs auto", flush=True)
    return report


def automatic_jobs(cfg):
    path = Path(cfg.output_root) / "benchmark.json"
    if not path.exists():
        raise ValueError("Run bash setup.sh benchmark before --jobs auto")
    report = json.loads(path.read_text())
    if report.get("schema") != 2 or report.get("metadata_sha256") != metadata_hash(cfg) or report.get("code_sha256") != source_fingerprint():
        raise ValueError("Benchmark data/code changed; run benchmark again")
    if "recommendation" not in report:
        raise ValueError("Benchmark did not finish; run benchmark again")
    saved, current = report["config"], cfg.to_dict()
    ignored = {"seed", "student_init", "output_root"}
    if any(saved.get(k) != v for k, v in current.items() if k not in ignored):
        raise ValueError("Benchmark settings changed; run benchmark again")
    if report.get("cuda_visible_devices") != os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("CUDA visibility changed; run benchmark again")
    if time.time() - report["measured_at_unix"] > 24 * 3600:
        raise ValueError("Benchmark is older than 24h; rerun because shared GPU load may have changed")
    jobs = report["recommendation"]["recommended_jobs"]
    device = setup_device(cfg)
    if report.get("hardware") != _hardware(device):
        raise ValueError("Benchmark hardware changed; run benchmark again")
    if jobs == 2 and device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        passed = [t for t in report["trials"] if t["status"] == "passed"]
        needed = max(memory_requirement(t, total / 1024**3) for t in passed)
        if free / 1024**3 < needed:
            print("Free VRAM decreased: reducing automatic jobs to 1", flush=True)
            return 1
    print(f"Using benchmark recommendation: jobs={jobs}", flush=True)
    return jobs
