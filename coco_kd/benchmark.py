import gc
import time

import torch

from .config import METHODS
from .masking import select_tokens
from .models import build_model
from .train import distillation_loss, optimizer_for
from .utils import autocast, setup_device, write_json


def benchmark(cfg, destination, steps=3):
    device = setup_device(cfg)
    rows = []
    for method in METHODS:
        student, teacher = build_model("student", cfg).to(device), build_model("teacher", cfg).to(device)
        teacher.requires_grad_(False).eval()
        optimizer = optimizer_for(student, cfg)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
        x = torch.randn(cfg.batch_size, 3, 224, 224, device=device)
        y = torch.arange(cfg.batch_size, device=device) % cfg.num_classes
        fg = torch.rand(cfg.batch_size, 196, device=device) > .75
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.monotonic()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            with autocast(cfg, device):
                logits, attention = student(x, return_attention=True)
                chosen, _ = select_tokens(attention, method, foreground=fg)
                target = teacher(x, chosen) if method != "ce" else None
                loss, _, _ = distillation_loss(logits, y, target, cfg)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        rows.append({"method": method, "batch_size": cfg.batch_size, "device": str(device),
                     "seconds_per_step": (time.monotonic() - start) / steps,
                     "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
                     "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else None,
                     "note": "random weights, real architecture, Adam state; excludes validation and probe overhead"})
        del student, teacher, optimizer, scaler, x, y, fg, logits, attention, target, loss, chosen
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(destination, rows)
    print(rows)
    return rows
