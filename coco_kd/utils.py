import hashlib
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_device(cfg):
    torch.set_num_threads(cfg.num_threads)
    device = torch.device(cfg.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported execution devices: cpu, cuda, cuda:N")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Install CUDA PyTorch on the server or use --device cpu for smoke tests.")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        # Deterministic kernels where possible; independent seeds still required.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
    return device


def autocast(cfg, device):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if cfg.amp and device.type == "cuda" else nullcontext()


def sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(sanitize(obj), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def save_checkpoint(path, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(checkpoint, temp)
    os.replace(temp, path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with open(temp, "wb") as file:
        np.savez_compressed(file, **arrays)
    os.replace(temp, path)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def source_fingerprint():
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for name in ("adaptive.py", "config.py", "data.py", "models.py", "masking.py", "train.py", "probe.py", "utils.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    lock = root.parent / "uv.lock"
    if lock.exists():
        digest.update(lock.read_bytes())
    return digest.hexdigest()


def model_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class RunLock:
    """OS advisory lock, released automatically after a crash; no stale PID deletion."""
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "a+")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.file.close()
            raise RuntimeError(f"Another process is writing this run: {self.path.parent}") from None
        return self

    def __exit__(self, *args):
        self.file.close()


def load_checkpoint(path):
    # Only load checkpoints produced by this project, or trusted upstream weights.
    return torch.load(path, map_location="cpu", weights_only=False)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_hash(cfg):
    return sha256(Path(cfg.data_root) / "manifest.json")


def compatible_config(saved, current):
    # Paths and machine-specific settings may change when moving to the server.
    ignored = {"data_root", "output_root", "device", "num_workers", "num_threads"}
    differences = {k: (saved.get(k), v) for k, v in current.items()
                   if k not in ignored and saved.get(k) != v}
    if differences:
        raise ValueError(f"Existing run has different settings. Choose a new output_root: {differences}")
