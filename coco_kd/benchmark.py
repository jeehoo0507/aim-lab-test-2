"""Measure one/two independent processes on the same GPU with real prepared data."""
import gc
from dataclasses import replace
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
from .dino_attention import DINO_METHODS, build_selector, selection_attention
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


def _worker(values, rank, barrier, queue, directory, warmup, steps, modes, measure_all=False):
    try:
        # A benchmark worker is already a spawned process. Nested DataLoader
        # subprocesses are unstable under multi-job CUDA runs and multiply shared
        # memory use, so each benchmark process reads its own batches directly.
        cfg = replace(Config.load(**values), num_workers=0)
        seed_all(9123 + rank)
        device = setup_device(cfg)
        train_data = CocoSubset(cfg.data_root, "train", train=True, threshold=cfg.foreground_threshold)
        val_data = CocoSubset(cfg.data_root, "val", threshold=cfg.foreground_threshold)
        work = Path(directory) / f"worker_{rank}"
        work.mkdir()
        rows = []
        for method in modes:
            role = "teacher" if method == "teacher" else "student"
            model = build_model(role, cfg).to(device)
            selector = build_selector(cfg, method)
            if selector is not None:
                selector = selector.to(device)
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
                        attention = selection_attention(selector, x, attention)
                        targets = None
                        if method not in ("ce", "teacher"):
                            indices, _ = select_tokens(attention, method, foreground=fg, generator=generator, epoch=100)
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
            if method in ("teacher", "student") or measure_all:
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
            if role == "student" and (method == "student" or measure_all):
                probe = Probe(replace(cfg, epochs=1000000, diagnostic_epochs=[]), work, teacher, device, selector=selector)
                probe.log(model, 1, method)  # Populate full-teacher cache before timing.
                _sync(device)
                barrier.wait(timeout=180)
                begun = time.perf_counter()
                probe.log(model, 2, method)
                _sync(device)
                row["probe_seconds"] = time.perf_counter() - begun
                row["probe_bytes"] = (work / "probe/epoch_002.npz").stat().st_size
                barrier.wait(timeout=180)
                probe.cfg.diagnostic_epochs = [3]
                begun = time.perf_counter()
                probe.log(model, 3, method)
                _sync(device)
                row["diagnostic_seconds"] = time.perf_counter() - begun
                row["diagnostic_bytes"] = (work / "probe/epoch_003_counterfactual.npz").stat().st_size
                del probe
                barrier.wait(timeout=180)
            row["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else 0.0
            row["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
            rows.append(row)
            queue.put({"event": "progress", "rank": rank, "method": method})
            del model, teacher, selector, optimizer, scaler, batches, iterator, update
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


def run_trial(cfg, jobs, warmup=2, steps=8, modes=("teacher", *METHODS),
              parallel_methods=None, measure_all=False):
    # Heterogeneous student jobs must execute the same validation/checkpoint/probe
    # phases so that their synchronization barriers cannot become misaligned.
    if parallel_methods is not None:
        if len(parallel_methods) != jobs or any(m not in (*METHODS, *DINO_METHODS) for m in parallel_methods):
            raise ValueError("parallel_methods must contain one baseline student method per worker")
        measure_all = True
    context = mp.get_context("spawn")
    barrier, queue = context.Barrier(jobs), context.Queue()
    base = Path(cfg.output_root) / "_benchmark"
    base.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f"jobs{jobs}_", dir=base) as directory:
        workers = [context.Process(target=_worker, args=(cfg.to_dict(), rank, barrier, queue, directory, warmup, steps,
                   (parallel_methods[rank],) if parallel_methods is not None else modes, measure_all))
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
            return {"jobs": jobs, "profile": "mixed" if parallel_methods is not None else "all" if len(modes) > 1 else modes[0],
                    "dataloader_workers_per_process": 0,
                    "status": "passed", "rows": rows}
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


def student_task_seconds(trial, cfg, train_count):
    rows = trial["rows"]
    train = train_count * max(r["seconds"] / r["samples"] for r in rows if r["method"] == "student")
    validation = _worst(rows, "student", "validation_seconds")
    checkpoint = _worst(rows, "student", "checkpoint_seconds")
    normal = _worst(rows, "student", "probe_seconds")
    detailed = _worst(rows, "student", "diagnostic_seconds")
    n_detail = len(set(e for e in cfg.diagnostic_epochs if 0 <= e <= cfg.epochs) | {0, cfg.epochs}) + 1
    return (cfg.epochs * (train + validation + checkpoint + normal) + 2 * normal
            + n_detail * max(0, detailed - normal))


def recommend(single, parallel_trials, cfg, train_count, seeds=3):
    one = time_per_seed(single, cfg, train_count)
    result = {"recommended_jobs": 1, "serial_hours": seeds * one["total_seconds"] / 3600,
              "one_seed": one, "reason": "A valid, materially faster parallel trial is required",
              "candidates": {}}
    times = {1: one}
    for trial in parallel_trials:
        if trial["status"] == "passed":
            times[trial["jobs"]] = trial
    best_hours = result["serial_hours"]
    single_task = student_task_seconds(single, cfg, train_count)
    average_task = one["student_seconds"] / (len(METHODS) * 2)
    total_student_tasks = seeds * len(METHODS) * 2
    for jobs, trial in sorted(times.items()):
        if jobs == 1:
            continue
        slowdown = student_task_seconds(trial, cfg, train_count) / single_task
        teacher_workers = min(jobs, seeds)
        duration = ((seeds + teacher_workers - 1) // teacher_workers) * one["teacher_seconds"]
        duration += ((total_student_tasks + jobs - 1) // jobs) * average_task * slowdown
        hours = duration / 3600
        candidate = {"hours": hours,
                     "student_task_slowdown": slowdown,
                     "aggregate_student_throughput_speedup": jobs / slowdown,
                     "full_plan_speedup": result["serial_hours"] / hours}
        result["candidates"][str(jobs)] = candidate
        if candidate["full_plan_speedup"] >= 1.10 and hours < best_hours:
            best_hours = hours
            result.update(recommended_jobs=jobs, parallel_hours=hours,
                          aggregate_student_throughput_speedup=candidate["aggregate_student_throughput_speedup"],
                          full_plan_speedup=candidate["full_plan_speedup"],
                          reason="Fastest measured full-plan time with >=10% improvement and VRAM headroom")
    if result["recommended_jobs"] == 1 and times.keys() != {1}:
        result["reason"] = "Parallel contention/tail saves <10%; prefer one worker"
    chosen = best_hours
    result["planning_range_hours"] = [chosen, chosen * 1.5]
    result["excludes"] = "download, initial environment setup, final test/export; full 42-student/3-teacher plan, not remaining work"
    return result


def memory_requirement(trial, total_gib):
    per_worker = max(r["peak_reserved_gib"] for r in trial["rows"]) + .75
    return trial["jobs"] * per_worker + max(2.0, .1 * total_gib)


def benchmark(cfg, destination, steps=8, warmup=2, max_jobs=7):
    if steps < 1 or warmup < 1 or max_jobs not in range(1, 8):
        raise ValueError("Positive warmup/steps and max_jobs from 1 through 7 required")
    manifest = validate_manifest(cfg.data_root)
    device = setup_device(cfg)
    train_count = sum(r["split"] == "train" for r in manifest["images"])
    report = {"schema": 3, "config": cfg.to_dict(), "metadata_sha256": metadata_hash(cfg),
              "code_sha256": source_fingerprint(), "measured_at_unix": time.time(),
              "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "hardware": _hardware(device),
              "warmup_updates": warmup, "measured_updates": steps,
              "training_images": train_count, "trials": [],
              "note": "Random weights, real prepared images. Temporary checkpoints. Data loading, accumulation and representative validation/probe/I/O included. Each spawned benchmark process loads synchronously (num_workers=0) to avoid nested multiprocessing."}
    print("Benchmark: real images + model sizes; experiment checkpoints are untouched", flush=True)
    single = run_trial(cfg, 1, warmup, steps)
    report["trials"].append(single)
    write_json(destination, report)
    parallel = []
    if device.type == "cuda" and max_jobs > 1:
        free, total = torch.cuda.mem_get_info(device)
        report["required_free_gib_by_jobs"] = {}
        report["free_gib_before_parallel"] = free / 1024**3
        student_peak = max(r["peak_reserved_gib"] for r in single["rows"] if r["method"] != "teacher")
        per_worker = student_peak + .75
        margin = max(2.0, .1 * total / 1024**3)
        for jobs in range(2, max_jobs + 1):
            needed = jobs * per_worker + margin
            report["required_free_gib_by_jobs"][str(jobs)] = needed
            if free / 1024**3 >= needed:
                print(f"Enough VRAM headroom; comparing {jobs} concurrent workers", flush=True)
                try:
                    trial = run_trial(cfg, jobs, warmup, steps, modes=("student",))
                    if memory_requirement(trial, total / 1024**3) > free / 1024**3:
                        trial["status"] = "insufficient_headroom"
                except (RuntimeError, TimeoutError) as error:
                    trial = {"jobs": jobs, "status": "failed", "error": str(error)}
            else:
                trial = {"jobs": jobs, "status": "skipped_memory"}
            parallel.append(trial)
            report["trials"].append(trial)
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
    estimated_output_gb = (report["checkpoint_storage"]["total_decimal_gb"]
                           + report["estimated_probe_validation_gib"] * 1024**3 / 1e9 + .25)
    report["storage_budget"] = {"estimated_output_decimal_gb": estimated_output_gb,
                                "recommended_decimal_gb": cfg.checkpoint_target_gb,
                                "warning_decimal_gb": cfg.output_warning_gb,
                                "above_warning": estimated_output_gb > cfg.output_warning_gb,
                                "note": "Includes checkpoints, measured probe/validation projection and 0.25GB metadata/analysis allowance; excludes data and environment/cache."}
    write_json(destination, report)
    shared_report = None
    if cfg.model_scale == "deit":
        shared_report = Path("reports/benchmark/latest.json")
        write_json(shared_report, report)
    print(json.dumps({"recommendation": report["recommendation"], "checkpoint_storage": report["checkpoint_storage"],
                      "probe_validation_gib": report["estimated_probe_validation_gib"]}, indent=2), flush=True)
    print(json.dumps({"storage_budget": report["storage_budget"]}, indent=2), flush=True)
    if report["storage_budget"]["above_warning"]:
        print(f"WARNING: projected outputs exceed the {cfg.output_warning_gb:.2f}GB planning threshold", flush=True)
    if shared_report:
        print(f"Shareable benchmark report: {shared_report.resolve()}", flush=True)
    print("After pilot review: bash setup.sh run --jobs auto", flush=True)
    return report


def automatic_jobs(cfg):
    path = Path(cfg.output_root) / "benchmark.json"
    if not path.exists():
        raise ValueError("Run bash setup.sh benchmark before --jobs auto")
    report = json.loads(path.read_text())
    if report.get("schema") != 3 or report.get("metadata_sha256") != metadata_hash(cfg) or report.get("code_sha256") != source_fingerprint():
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
    if jobs > 1 and device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        passed = {t["jobs"]: t for t in report["trials"] if t["status"] == "passed"}
        required = report.get("required_free_gib_by_jobs", {})
        def needed(candidate):
            if str(candidate) in required:
                return required[str(candidate)]
            return memory_requirement(passed[candidate], total / 1024**3)
        while jobs > 1 and free / 1024**3 < needed(jobs):
            jobs -= 1
        if jobs != report["recommendation"]["recommended_jobs"]:
            print(f"Free VRAM decreased: reducing automatic jobs to {jobs}", flush=True)
    print(f"Using benchmark recommendation: jobs={jobs}", flush=True)
    return jobs
