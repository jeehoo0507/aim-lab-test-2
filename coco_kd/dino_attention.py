"""Frozen DINO v1 saliency for supervised KD (MaskedKD section 5.1).

This is not DINO self-supervised training or a DINO classification teacher.
The official backbone is architecture-compatible with our 224px DeiT adapter.
"""
from pathlib import Path

import torch
from torch import nn

from .models import DeiT
from .utils import sha256

DINO_METHODS = ("dino", "dino_random_rescue_10", "dino_paper_late")
DINO_URL = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
# Filled from the official downloaded file, also checked before every load.
DINO_SHA256 = "1566d50496f27f52f07fea6094fa29b2fdd6fae89da65bdd3ebc3b24ef6b7eb7"


def build_selector(cfg, method):
    if method not in DINO_METHODS:
        return None
    # Constructing an auxiliary model must not change student init/dropout RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1729)
        dim, depth, heads = (48, 2, 3) if cfg.model_scale == "debug" else (384, 12, 6)
        model = DeiT(dim, depth, heads, drop_path=0, num_classes=cfg.num_classes)
        model.head = nn.Identity()
        if cfg.model_scale != "debug":
            filename = DINO_URL.rsplit("/", 1)[1]
            path = Path(torch.hub.get_dir()) / "checkpoints" / filename
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                # The launcher downloads once before starting worker processes.
                temporary = path.with_suffix(".download")
                torch.hub.download_url_to_file(DINO_URL, str(temporary), hash_prefix=DINO_SHA256)
                temporary.replace(path)
            if sha256(path) != DINO_SHA256:
                raise ValueError(f"Official DINO checkpoint checksum mismatch: {path}")
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.requires_grad_(False).eval()


@torch.no_grad()
def selection_attention(selector, images, student_attention):
    return student_attention if selector is None else selector(images, return_attention=True)[1]
