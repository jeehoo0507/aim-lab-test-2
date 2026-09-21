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


def preflight(cfg, destination):
    device = setup_device(cfg)
    student = build_model("student", cfg)
    teacher = build_model("teacher", cfg)
    count = sum(p.numel() for p in student.parameters())
    snapshots = 1 + cfg.epochs // cfg.checkpoint_every + int(cfg.epochs % cfg.checkpoint_every != 0)
    # periodic weights + best + latest (weights, best weights, Adam moments)
    weight_gib = count * 4 * (snapshots + 5) * 42 / 1024**3
    report = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
              "git_commit": command(["git", "rev-parse", "HEAD"]),
              "git_status": command(["git", "status", "--short"]),
              "device": str(device), "student_parameters": count,
              "teacher_parameters": sum(p.numel() for p in teacher.parameters()),
              "student_checkpoint_estimate_42_runs_gib": weight_gib,
              "estimate_excludes": "teacher weights, raw probes, data, uv environment/caches; measure after smoke",
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
