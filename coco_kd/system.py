import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from .config import METHODS
from .models import build_model
from .utils import setup_device, write_json


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return p.stdout.strip() if p.returncode == 0 else p.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)


def storage_report(paths):
    rows = []
    for name, value in paths.items():
        p = Path(value).expanduser().resolve()
        ancestor = p
        while not ancestor.exists():
            ancestor = ancestor.parent
        usage = shutil.disk_usage(ancestor)
        rows.append({"name": name, "configured": str(value), "resolved": str(p),
                     "free_gib": usage.free / 1024**3, "total_gib": usage.total / 1024**3,
                     "filesystem": command(["df", "-h", str(ancestor)]),
                     "mount": command(["findmnt", "-T", str(ancestor), "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"]) if sys.platform == "linux" else None})
    return rows


def checkpoint_estimate(cfg, student_parameters, teacher_parameters):
    def snapshots(epochs):
        return 1 + epochs // cfg.checkpoint_every + int(epochs % cfg.checkpoint_every != 0)
    students = student_parameters * 4 * (snapshots(cfg.epochs) + 5) * len(METHODS) * 2 * 3
    teachers = teacher_parameters * 4 * (snapshots(cfg.teacher_epochs) + 5) * 3
    midpoint = student_parameters * 4 * 4 * 2 * 3 if cfg.epochs >= 50 else 0
    return {"student_gib": students / 1024**3, "teacher_gib": teachers / 1024**3,
            "maskedkd_midpoint_gib": midpoint / 1024**3,
            "total_gib": (students + teachers + midpoint) / 1024**3,
            "total_decimal_gb": (students + teachers + midpoint) / 10**9,
            "excludes": "data, environments/caches, probes/validation, analysis/export; approximate tensor bytes"}


def preflight(cfg, destination):
    device = setup_device(cfg)
    student = build_model("student", cfg)
    teacher = build_model("teacher", cfg)
    count = sum(p.numel() for p in student.parameters())
    storage = checkpoint_estimate(cfg, count, sum(p.numel() for p in teacher.parameters()))
    report = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
              "git_commit": command(["git", "rev-parse", "HEAD"]),
              "git_status": command(["git", "status", "--short"]),
              "device": str(device), "student_parameters": count,
              "teacher_parameters": sum(p.numel() for p in teacher.parameters()),
              "checkpoint_storage": storage,
              "estimate_excludes": "raw probes, data, uv environment/caches, analysis/export",
              "storage": storage_report({"project": Path.cwd(), "data": cfg.data_root, "outputs": cfg.output_root,
                  "torch_cache": os.environ.get("TORCH_HOME", str(Path.home() / ".cache/torch")),
                  "uv_cache": os.environ.get("UV_CACHE_DIR", str(Path.home() / ".cache/uv"))}),
              "gpu": command(["nvidia-smi"]),
              "block_devices": command(["lsblk", "-o", "NAME,SIZE,ROTA,TYPE,FSTYPE,MOUNTPOINTS,MODEL"]) if sys.platform == "linux" else None}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        report["cuda"] = {"name": torch.cuda.get_device_name(device), "free_gib": free / 1024**3, "total_gib": total / 1024**3}
    write_json(destination, report)
    print(json.dumps(report, indent=2))
    return report
