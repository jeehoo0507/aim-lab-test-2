"""Download only strict single-category COCO candidates; no full image archive."""
import hashlib
import json
import os
import subprocess
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_api
from tqdm import tqdm

from .config import CLASSES
from .data import validate_manifest
from .utils import sha256, write_json

ANNOTATIONS = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"


def strict_candidates(info, classes=CLASSES):
    category = {x["id"]: x["name"] for x in info["categories"]}
    annotations = defaultdict(list)
    for a in info["annotations"]:
        annotations[a["image_id"]].append(a)
    for image in sorted(info["images"], key=lambda x: x["id"]):
        anns = annotations[image["id"]]
        kinds = {a["category_id"] for a in anns}
        if len(kinds) != 1:
            continue
        name = category[next(iter(kinds))]
        if name not in classes or any(a.get("iscrowd", 0) or a["area"] <= 0 or not a.get("segmentation") for a in anns):
            continue
        yield image, classes.index(name), anns


def segmentation(anns, height, width):
    union = np.zeros((height, width), dtype=bool)
    for a in anns:
        seg = a["segmentation"]
        if isinstance(seg, list):
            rle = mask_api.merge(mask_api.frPyObjects(seg, height, width))
        elif isinstance(seg["counts"], list):
            rle = mask_api.frPyObjects(seg, height, width)
        else:
            rle = seg
        decoded = mask_api.decode(rle)
        if decoded.ndim == 3:
            decoded = decoded.any(2)
        union |= decoded.astype(bool)
    return union


def fetch_image(url, path):
    if path.exists():
        try:
            with Image.open(path) as f:
                f.load()
            return
        except OSError:
            pass  # Replace a corrupt cached download only after a valid replacement is ready.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    last_error = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(temporary, "wb") as file:
                while chunk := response.read(1024 * 1024):
                    file.write(chunk)
            with Image.open(temporary) as f:
                f.load()
            os.replace(temporary, path)
            return
        except Exception as error:
            last_error = error
    raise RuntimeError(f"Could not download {url}; rerun preparation to resume") from last_error


def quality_record(root, item):
    split, image, label, anns = item
    name = Path(image["file_name"]).name
    image_rel, mask_rel = f"images/{split}/{name}", f"masks/{split}/{Path(name).stem}.png"
    path = root / image_rel
    fetch_image(f"https://s3.amazonaws.com/images.cocodataset.org/{split}/{name}", path)
    with Image.open(path) as file:
        rgb = file.convert("RGB")
        rgb.load()
    if rgb.size != (image["width"], image["height"]):
        raise ValueError(f"Dimensions disagree with COCO annotations: {image['id']}")
    mask = segmentation(anns, image["height"], image["width"])
    if not mask.any():
        return {"rejected_id": image["id"], "reason": "empty_segmentation"}
    m = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = np.asarray(m.resize((224, 224), Image.Resampling.NEAREST)) > 0
    # Very tiny foregrounds are kept for classification and reported, not removed after looking at accuracy.
    if not resized.any():
        return {"rejected_id": image["id"], "reason": "empty_after_resize"}
    (root / mask_rel).parent.mkdir(parents=True, exist_ok=True)
    m.save(root / mask_rel)
    pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
    small = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    dhash = sum(int(bit) << j for j, bit in enumerate(bits))
    return {"id": image["id"], "label": label, "source_split": split,
            "image": image_rel, "mask": mask_rel, "image_sha256": sha256(path),
            "mask_sha256": sha256(root / mask_rel), "pixel_sha256": pixels,
            "dhash": dhash, "object_fraction": float(resized.mean()), "instances": len(anns)}


def deduplicate(records):
    """Keep validation-source priority; conservatively remove perceptual neighbors.

    Five disjoint hash bands guarantee candidate retrieval for <=4 differing bits.
    This is a reproducible near-duplicate heuristic, not a guarantee of no semantic overlap.
    """
    bands, pixels, kept, rejected = defaultdict(list), set(), [], []
    for row in sorted(records, key=lambda r: (r["source_split"] != "val2017", r["id"])):
        h = row["dhash"]
        keys = [(b, (h >> (b * 13)) & ((1 << 13) - 1)) for b in range(5)]
        neighbors = {idx for key in keys for idx in bands[key]}
        duplicate = row["pixel_sha256"] in pixels or any((h ^ kept[i]["dhash"]).bit_count() <= 4 for i in neighbors)
        if duplicate:
            rejected.append({"id": row["id"], "reason": "decoded_or_perceptual_duplicate"})
            continue
        idx = len(kept)
        kept.append(row)
        pixels.add(row["pixel_sha256"])
        for key in keys:
            bands[key].append(idx)
    return kept, rejected


def make_splits(records, train_per_class=600, val_per_class=100, probe_per_class=20, seed=20260922):
    count = len(CLASSES)
    available = [sum(r["source_split"] == "train2017" and r["label"] == c for r in records) for c in range(count)]
    common_total = min(train_per_class + val_per_class, min(available))
    actual_val = min(val_per_class, common_total // 7)
    actual_train = min(train_per_class, common_total - actual_val)
    if actual_val < probe_per_class or actual_train < 1:
        raise ValueError(f"Not enough clean candidates: {available}")
    chosen = []
    rng = np.random.default_rng(seed)
    for c in range(count):
        rows = sorted([r for r in records if r["source_split"] == "train2017" and r["label"] == c], key=lambda r: r["id"])
        order = rng.permutation(len(rows))
        for j, idx in enumerate(order[:actual_train + actual_val]):
            split = "val" if j < actual_val else "train"
            chosen.append({**rows[idx], "split": split, "probe": j < probe_per_class})
        test = [r for r in records if r["source_split"] == "val2017" and r["label"] == c]
        if not test:
            raise ValueError(f"Class {CLASSES[c]} has no held-out test examples")
        chosen.extend({**r, "split": "test", "probe": False} for r in test)
    return sorted(chosen, key=lambda r: r["id"]), actual_train, actual_val


def prepare(root, workers=4):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        validate_manifest(root)
        print(f"REUSED verified immutable dataset: {root}")
        return
    archive = root / "annotations_trainval2017.zip"
    if not archive.exists():
        part = archive.with_suffix(".zip.part")
        subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--continue-at", "-",
                        "--output", str(part), ANNOTATIONS], check=True)
        with zipfile.ZipFile(part) as z:
            if z.testzip() is not None:
                raise ValueError("Corrupt COCO annotation archive")
        os.replace(part, archive)
    items = []
    with zipfile.ZipFile(archive) as z:
        for split in ("train2017", "val2017"):
            info = json.loads(z.read(f"annotations/instances_{split}.json"))
            items.extend((split, *row) for row in strict_candidates(info))
    cache = root / "quality_records.json"
    if cache.exists():
        quality = json.loads(cache.read_text())
    else:
        with ThreadPoolExecutor(max_workers=min(4, max(1, workers))) as pool:
            quality = list(tqdm(pool.map(lambda item: quality_record(root, item), items), total=len(items), desc="COCO image/mask QC"))
        write_json(cache, quality)
    records, duplicates = deduplicate([r for r in quality if "id" in r])
    images, ntrain, nval = make_splits(records)
    write_json(root / "quality_report.json", {"candidates": len(items), "duplicates": duplicates,
               "invalid": [r for r in quality if "rejected_id" in r], "train_per_class": ntrain,
               "val_per_class": nval, "test": sum(r["split"] == "test" for r in images),
               "near_duplicate_rule": "64-bit dHash Hamming <=4, plus decoded RGB SHA256"})
    write_json(root / "manifest.json", {"schema": 1, "classes": list(CLASSES), "split_seed": 20260922,
               "transform": "full_image_resize224_flip_train_only", "annotations_sha256": sha256(archive),
               "probe_per_class": 20, "images": images})
    validate_manifest(root)
    print(f"Prepared {len(images)} images; train={ntrain * 10}, val={nval * 10}; {root}")
