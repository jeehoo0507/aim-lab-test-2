import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class CocoSubset(Dataset):
    def __init__(self, root, split, train=False, threshold=0.5):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if split == "probe":
            matches = lambda row: row["probe"]
        elif split == "val_fit":
            matches = lambda row: row["split"] == "val" and not row["probe"]
        else:
            matches = lambda row: row["split"] == split
        self.records = [row for row in self.manifest["images"] if matches(row)]
        if not self.records:
            raise ValueError(f"Empty {split} split")
        self.train, self.threshold = train, threshold

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        for key in ("image", "mask"):
            p = Path(row[key])
            if p.is_absolute() or ".." in p.parts:
                raise ValueError("Dataset manifest paths must be relative")
        with Image.open(self.root / row["image"]) as f:
            image = f.convert("RGB")
        with Image.open(self.root / row["mask"]) as f:
            mask = f.convert("L")
        if image.size != mask.size:
            raise ValueError(f"Image/mask dimensions disagree: {row['id']}")
        # Keep the complete image: cropping could silently remove the label object.
        image = TF.resize(image, [224, 224], InterpolationMode.BICUBIC, antialias=True)
        mask = TF.resize(mask, [224, 224], InterpolationMode.NEAREST)
        if self.train and torch.rand(()).item() < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        image = TF.normalize(TF.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        coverage = torch.nn.functional.avg_pool2d(TF.to_tensor(mask), 16, 16).flatten()
        return {"image": image, "label": row["label"], "sample_id": row["id"],
                "coverage": coverage, "foreground": coverage >= self.threshold}


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def loader(dataset, cfg, train=False, epoch=0):
    generator = torch.Generator().manual_seed(cfg.seed + epoch * 100003)
    return DataLoader(dataset, batch_size=cfg.batch_size if train else cfg.eval_batch_size,
                      shuffle=train, num_workers=cfg.num_workers, drop_last=False,
                      pin_memory=cfg.device.startswith("cuda"), generator=generator,
                      worker_init_fn=seed_worker, persistent_workers=False)


def validate_manifest(root, verify_files=True):
    from .utils import sha256
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text())
    ids = [r["id"] for r in m["images"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Image IDs overlap across splits")
    digests = [r["pixel_sha256"] for r in m["images"]]
    if len(digests) != len(set(digests)):
        raise ValueError("Duplicate decoded images in manifest")
    for split in ("train", "val", "test"):
        labels = {r["label"] for r in m["images"] if r["split"] == split}
        if labels != set(range(len(m["classes"]))):
            raise ValueError(f"Missing class in {split}")
    for r in m["images"]:
        if r["probe"] and r["split"] != "val":
            raise ValueError("Probe must be a validation subset")
        for key in ("image", "mask"):
            p = Path(r[key])
            if p.is_absolute() or ".." in p.parts:
                raise ValueError("Unsafe manifest path")
            if verify_files and sha256(root / p) != r[key + "_sha256"]:
                raise ValueError(f"Changed or corrupt {key}: {r['id']}")
    return m
