import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from .config import TRAIN_METHODS
from .data import CocoSubset, loader
from .masking import binary_mask, select_tokens
from .metrics import classification
from .models import build_model
from .probe import Probe
from .utils import (RunLock, autocast, compatible_config, cpu_state, load_checkpoint, metadata_hash,
                    model_fingerprint, source_fingerprint, save_checkpoint, save_npz,
                    seed_all, setup_device, sha256, write_json)


def run_path(cfg, role="student", method="student"):
    root = Path(cfg.output_root) / f"seed_{cfg.seed}"
    return root / "teacher" if role == "teacher" else root / cfg.student_init / method


def learning_rate(cfg, epoch, total_epochs, role):
    # These are ACTUAL optimizer peak learning rates: no hidden batch-size multiplier.
    base = cfg.teacher_lr if role == "teacher" else (cfg.scratch_lr if cfg.student_init == "scratch" else cfg.pretrained_lr)
    warmup = min(cfg.warmup_epochs, max(0, total_epochs - 1))
    if epoch <= warmup:
        return base * epoch / max(1, warmup)
    fraction = (epoch - warmup - 1) / max(1, total_epochs - warmup - 1)
    return min(cfg.min_lr, base) + (base - min(cfg.min_lr, base)) * (1 + math.cos(math.pi * fraction)) / 2


def optimizer_for(model, cfg):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        target = no_decay if p.ndim == 1 or name.endswith(".bias") or name in ("pos_embed", "cls_token") else decay
        target.append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                              {"params": no_decay, "weight_decay": 0}], lr=cfg.scratch_lr)


def distillation_loss(logits, labels, teacher_logits, cfg):
    ce = F.cross_entropy(logits.float(), labels, label_smoothing=cfg.label_smoothing)
    if teacher_logits is None:
        return ce, ce.detach(), logits.new_zeros(())
    t = cfg.temperature
    kd = F.kl_div(F.log_softmax(logits.float() / t, 1), F.softmax(teacher_logits.float() / t, 1),
                  reduction="batchmean") * t**2
    return (1 - cfg.kd_alpha) * ce + cfg.kd_alpha * kd, ce.detach(), kd.detach()


@torch.inference_mode()
def evaluate(model, dataset, cfg, device, destination=None):
    model.eval()
    outputs = {"logits": [], "label": [], "sample_id": []}
    for batch in loader(dataset, cfg):
        with autocast(cfg, device):
            logits = model(batch["image"].to(device))
        outputs["logits"].append(logits.float().cpu().numpy())
        outputs["label"].append(batch["label"].numpy())
        outputs["sample_id"].append(batch["sample_id"].numpy())
    arrays = {k: np.concatenate(v) for k, v in outputs.items()}
    if destination:
        save_npz(destination, **arrays)
    return classification(arrays["logits"], arrays["label"], cfg.num_classes)


def train_epoch(model, teacher, dataset, optimizer, scaler, cfg, device, method, epoch):
    model.train()
    seed_all(cfg.seed + epoch * 100003)
    mask_rng = torch.Generator(device=device).manual_seed(cfg.seed + epoch * 200003)
    batches = loader(dataset, cfg, train=True, epoch=epoch)
    total = min(len(batches), cfg.max_train_batches or len(batches))
    sums = {k: 0.0 for k in ("loss", "ce", "kd", "swaps", "added_foreground", "removed_foreground", "correct")}
    seen = updates = skipped = 0
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(batches, total=total, desc=f"{method} epoch {epoch}")):
        if step >= total:
            break
        x, y, foreground = batch["image"].to(device), batch["label"].to(device), batch["foreground"].to(device)
        with autocast(cfg, device):
            logits, attention = model(x, return_attention=True)
            targets, indices, swaps = None, None, None
            if teacher is not None and method not in ("ce", "teacher"):
                indices, swaps = select_tokens(attention, method, foreground=foreground, generator=mask_rng, epoch=epoch)
                with torch.no_grad():
                    targets = teacher(x, indices)
            loss, ce, kd = distillation_loss(logits, y, targets, cfg)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss epoch {epoch}, batch {step}")
        start = step // cfg.accumulation_steps * cfg.accumulation_steps
        end = min(start + cfg.accumulation_steps, total)
        window_samples = min(end * cfg.batch_size, len(dataset)) - start * cfg.batch_size
        scaler.scale(loss * len(x) / window_samples).backward()
        if step + 1 == end:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=not scaler.is_enabled())
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale:
                skipped += 1
            else:
                updates += 1
            optimizer.zero_grad(set_to_none=True)
        for key, value in (("loss", loss), ("ce", ce), ("kd", kd)):
            sums[key] += value.item() * len(x)
        sums["correct"] += (logits.argmax(1) == y).sum().item()
        if indices is not None:
            raw = binary_mask(attention.detach().topk(98, 1).indices)
            selected = binary_mask(indices)
            sums["added_foreground"] += ((selected & ~raw) & foreground).sum().item()
            sums["removed_foreground"] += ((raw & ~selected) & foreground).sum().item()
        if swaps is not None:
            sums["swaps"] += swaps.sum().item()
        seen += len(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {**{k: v / seen for k, v in sums.items()}, "samples": seen, "optimizer_updates": updates,
            "amp_skipped_updates": skipped, "seconds": time.monotonic() - started}


def train(cfg, role="student", method="student", stop_after=None):
    if role not in ("teacher", "student") or method not in (*TRAIN_METHODS, "teacher"):
        raise ValueError("Unknown role or method")
    if method == "random_anneal_10" and cfg.model_scale != "debug" and cfg.epochs != 100:
        raise ValueError("random_anneal_10 is a fixed 100-epoch protocol; use a separate design for other lengths")
    directory = run_path(cfg, role, method)
    with RunLock(directory / ".run.lock"):
        return _train(cfg, role, method, directory, stop_after)


def _train(cfg, role, method, directory, stop_after):
    method = "teacher" if role == "teacher" else method
    device = setup_device(cfg)
    digest = metadata_hash(cfg)
    code_hash = source_fingerprint()
    teacher_file = run_path(cfg, "teacher") / "best.pt" if role == "student" else None
    teacher_hash = sha256(teacher_file) if teacher_file else None
    saved_config = directory / "config.json"
    if saved_config.exists():
        compatible_config(json.loads(saved_config.read_text()), cfg.to_dict())
    result_path = directory / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["metadata_sha256"] != digest or result["teacher_sha256"] != teacher_hash:
            raise ValueError("Completed run data/teacher changed; use a new output root")
        if result.get("code_sha256") != code_hash:
            raise ValueError("Training code changed since completion; use a new output root")
        print(f"REUSED complete {directory}", flush=True)
        return directory / "best.pt"
    write_json(saved_config, cfg.to_dict())
    state = load_checkpoint(directory / "last.pt") if (directory / "last.pt").exists() else None
    if state and (state["metadata_sha256"] != digest or state["teacher_sha256"] != teacher_hash):
        raise ValueError("Resume data/teacher mismatch")
    if state and state.get("code_sha256") != code_hash:
        raise ValueError("Training code changed since checkpoint; use a new output root")
    seed_all(cfg.seed)
    pretrained = (cfg.teacher_pretrained if role == "teacher" else cfg.student_init == "imagenet") and state is None
    model = build_model(role, cfg, pretrained=pretrained).to(device)
    initial_hash = state["initial_model_sha256"] if state else model_fingerprint(model)
    teacher = None
    if teacher_file:
        source = load_checkpoint(teacher_file)
        if source["role"] != "teacher" or source["metadata_sha256"] != digest:
            raise ValueError("Teacher provenance mismatch")
        teacher = build_model("teacher", cfg).to(device)
        teacher.load_state_dict(source["model"])
        teacher.requires_grad_(False).eval()
        del source
    optimizer, scaler = optimizer_for(model, cfg), torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    start, best_score, best_epoch, best_weights, history = 0, -1.0, 0, None, []
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start, best_score, best_epoch = state["epoch"], state["best_score"], state["best_epoch"]
        history, best_weights = state["history"], state["best_model"]
        print(f"RESUME {directory} after epoch {start}", flush=True)
    train_data = CocoSubset(cfg.data_root, "train", train=True, threshold=cfg.foreground_threshold)
    validation = CocoSubset(cfg.data_root, "val", threshold=cfg.foreground_threshold)
    epochs = cfg.teacher_epochs if role == "teacher" else cfg.epochs
    probe = Probe(cfg, directory, teacher, device) if teacher is not None else None
    base = {"config": cfg.to_dict(), "role": role, "method": method,
            "metadata_sha256": digest, "teacher_sha256": teacher_hash, "code_sha256": code_hash,
            "initial_model_sha256": initial_hash}
    if state:
        # last.pt is the committed epoch; undo a best/history write from an interrupted later epoch.
        save_checkpoint(directory / "best.pt", {**base, "epoch": best_epoch, "model": best_weights})
        write_json(directory / "history.json", history)
        del state
    elif probe:
        # Epoch-0 raw diagnostics are sufficient; the deterministic initialization can be reconstructed.
        probe.log(model, 0, method)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start + 1, epochs + 1):
        if cfg.model_scale == "deit":
            if shutil.disk_usage(directory).free < 2 * 1024**3:
                raise OSError("Less than 2 GiB free at output path; stopped before next epoch, resume after freeing space")
        begun = time.monotonic()
        lr = learning_rate(cfg, epoch, epochs, role)
        for group in optimizer.param_groups:
            group["lr"] = lr
        training = train_epoch(model, teacher, train_data, optimizer, scaler, cfg, device, method, epoch)
        val = evaluate(model, validation, cfg, device, directory / "validation" / f"epoch_{epoch:03d}.npz")
        if probe:
            probe.log(model, epoch, method)
        if val["macro_accuracy"] is None:
            raise ValueError("All classes required in validation")
        score = val["macro_accuracy"]
        if score > best_score:
            best_score, best_epoch, best_weights = score, epoch, cpu_state(model)
            save_checkpoint(directory / "best.pt", {**base, "epoch": epoch, "model": best_weights})
        peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0
        history.append({"epoch": epoch, "lr": lr, "train": training, "validation": val,
                        "epoch_seconds_with_probe": time.monotonic() - begun, "peak_allocated_gib": peak})
        payload = {**base, "model": cpu_state(model), "epoch": epoch}
        if epoch % cfg.checkpoint_every == 0 and epoch < epochs:
            save_checkpoint(directory / f"epoch_{epoch:03d}.pt", payload)
        # Full resume state only for latest epoch; periodic snapshots are weights only.
        resume_payload = {**payload, "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(), "best_model": best_weights, "best_epoch": best_epoch,
                        "best_score": best_score, "history": history}
        save_checkpoint(directory / "last.pt", resume_payload)
        if role == "student" and method == "student" and epoch == 50:
            save_checkpoint(directory / "resume_050.pt", resume_payload)
        write_json(directory / "history.json", history)
        print(f"{directory.name} {epoch}/{epochs}: val macro={score:.4f}, lr={lr:.3g}, swaps={training['swaps']:.1f}, train={training['seconds']:.1f}s, peak={peak:.2f}GiB", flush=True)
        if stop_after is not None and epoch >= stop_after and epoch < epochs:
            return directory / "last.pt"
    if probe:
        model.load_state_dict(best_weights)
        probe.log(model, best_epoch, method, name="best")
    result = {**base, "best_epoch": best_epoch, "best_validation_macro": best_score, "last_epoch": epochs,
              "seed": cfg.seed, "student_init": cfg.student_init, "partial_training": cfg.max_train_batches is not None,
              "checkpoint_selection": "maximum validation macro accuracy; earliest epoch breaks ties",
              "test_evaluated": False, "seconds_total": sum(r["epoch_seconds_with_probe"] for r in history)}
    write_json(result_path, result)
    # Completed runs no longer need optimizer/scaler/history inside last.pt. Keeping a compact
    # best and final pair preserves both test evaluations while releasing most checkpoint space.
    save_checkpoint(directory / "last.pt", {**payload, "best_epoch": best_epoch, "best_score": best_score})
    return directory / "best.pt"


def evaluate_test(cfg, role="student", method="student"):
    directory = run_path(cfg, role, method)
    with RunLock(directory / ".run.lock"):
        result = json.loads((directory / "result.json").read_text())
        compatible_config(result["config"], cfg.to_dict())
        if result["metadata_sha256"] != metadata_hash(cfg):
            raise ValueError("Dataset changed since training")
        device = setup_device(cfg)
        test = CocoSubset(cfg.data_root, "test", threshold=cfg.foreground_threshold)
        model = build_model(role, cfg).to(device)
        metrics = {}
        for name in ("best", "last"):
            state = load_checkpoint(directory / f"{name}.pt")
            model.load_state_dict(state["model"])
            metrics[name] = {"epoch": state["epoch"], **evaluate(model, test, cfg, device, directory / f"test_{name}.npz")}
        write_json(directory / "test_metrics.json", metrics)
        return metrics
