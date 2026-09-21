import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from .utils import sha256, write_json


def synthetic_data(root, classes=2):
    root = Path(root)
    images = []
    rng = np.random.default_rng(19)
    sample_id = 0
    for split, number in (("train", 3), ("val", 2), ("test", 2)):
        for label in range(classes):
            for _ in range(number):
                sample_id += 1
                pixels = rng.integers(0, 64, (64, 64, 3), dtype=np.uint8)
                pixels[16:48, 16:48, label % 3] = 220
                mask = np.zeros((64, 64), dtype=np.uint8)
                mask[16:48, 16:48] = 255
                relative_image, relative_mask = f"images/{sample_id}.png", f"masks/{sample_id}.png"
                for relative, values in ((relative_image, pixels), (relative_mask, mask)):
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(values).save(path)
                images.append({"id": sample_id, "label": label, "split": split, "probe": split == "val",
                               "source_split": "synthetic", "image": relative_image, "mask": relative_mask,
                               "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                               "image_sha256": sha256(root / relative_image), "mask_sha256": sha256(root / relative_mask)})
    write_json(root / "manifest.json", {"schema": 1, "classes": [f"class_{i}" for i in range(classes)],
               "split_seed": 19, "transform": "full_image_resize224_flip_train_only", "images": images})


def smoke(root, device="cpu"):
    from dataclasses import replace
    from .analysis import analyze
    from .config import Config, METHODS
    from .data import validate_manifest
    from .train import train, evaluate_test
    from .utils import source_fingerprint
    root = Path(root) / source_fingerprint()[:12]
    synthetic_data(root / "data")
    validate_manifest(root / "data")
    cfg = Config.load(data_root=str(root / "data"), output_root=str(root / "outputs"), device=device,
                      num_classes=2, batch_size=2, eval_batch_size=2, accumulation_steps=2, num_workers=0,
                      num_threads=2, epochs=2, teacher_epochs=2, warmup_epochs=0, checkpoint_every=1,
                      student_init="scratch", teacher_pretrained=False, model_scale="debug",
                      diagnostic_epochs=[0, 2], diagnostic_repeats=1)
    train(cfg, role="teacher")
    for method in METHODS:
        train(cfg, method=method)
        evaluate_test(cfg, method=method)
    analyze(cfg.output_root)
    write_json(root / "smoke_config.json", cfg.to_dict())
    print(f"PASSED synthetic end-to-end: {root.resolve()}")
