#!/usr/bin/env python3
"""Render held-out test images from completed checkpoints; no training or test-set tuning."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image

from coco_kd.config import Config
from coco_kd.data import CocoSubset
from coco_kd.masking import select_tokens
from coco_kd.models import build_model
from coco_kd.utils import autocast, load_checkpoint, metadata_hash, setup_device, sha256


RUNS = (
    ("MaskedKD", "student", "student"),
    ("Random10", "random_rescue_10", "random_rescue_10"),
    ("Random10→Low10", "adaptive_random_to_low_10", "low_score_rescue_10"),
)
STREAM_INDEX = {"random_rescue_10": 2, "low_score_rescue_10": 4}


def run_directory(args, method):
    root = args.adaptive_root if method.startswith("adaptive_") else args.baseline_root
    return Path(root) / f"seed_{args.seed}" / args.init / method


def image_and_mask(data_root, row):
    arrays = []
    for key, mode, resampling in (("image", "RGB", Image.Resampling.BICUBIC),
                                  ("mask", "L", Image.Resampling.NEAREST)):
        path = Path(row[key])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe dataset path: {path}")
        with Image.open(data_root / path) as source:
            arrays.append(np.asarray(source.convert(mode).resize((224, 224), resampling)).copy())
    return arrays[0].astype(np.float32) / 255.0, arrays[1]


def show_selected(image, indices):
    chosen = np.zeros(196, dtype=bool)
    chosen[indices] = True
    mask = np.repeat(np.repeat(chosen.reshape(14, 14), 16, 0), 16, 1)
    return image * np.where(mask[..., None], 1.0, .18), chosen


def choose(attention, sample_id, method, foreground):
    if method == "student":
        return select_tokens(attention, method)[0]
    stream = STREAM_INDEX[method]
    generator = torch.Generator(device=attention.device).manual_seed(sample_id * 1009 + 10000019 * stream)
    return select_tokens(attention, method, foreground=foreground, generator=generator)[0]


def active_at_last(history, method):
    # Experiment-2 baseline histories predate the active_method field.
    if method.startswith("adaptive_"):
        return history[-1]["active_method"]
    return history[-1].get("active_method", method)


@torch.inference_mode()
def render_one(args, dataset, row, teacher, teacher_hash, device, output):
    index = next(i for i, item in enumerate(dataset.records) if item["id"] == row["id"])
    batch = dataset[index]
    x = batch["image"].unsqueeze(0).to(device)
    fg = batch["foreground"].unsqueeze(0).to(device)
    image, mask = image_and_mask(Path(dataset.root), row)
    original_cfg = Config.load(run_directory(args, "student") / "config.json",
                               data_root=str(dataset.root), device=str(device))
    with autocast(original_cfg, device):
        full_logits, teacher_attention = teacher(x, return_attention=True)
    full_probability = full_logits.float().softmax(1)[0]
    records = []
    for label, method, expected_active in RUNS:
        path = run_directory(args, method)
        result = json.loads((path / "result.json").read_text())
        cfg = Config.load(path / "config.json", data_root=str(dataset.root), device=str(device))
        if result["last_epoch"] != 100 or result["partial_training"]:
            raise ValueError(f"Expected a completed 100-epoch run: {path}")
        if result["metadata_sha256"] != metadata_hash(cfg) or result["teacher_sha256"] != teacher_hash:
            raise ValueError(f"Dataset or teacher provenance mismatch: {path}")
        history = json.loads((path / "history.json").read_text())
        active = active_at_last(history, method)
        if method != "adaptive_random_to_low_10" and active != expected_active:
            raise ValueError(f"Unexpected final masking method: {path}")
        state = load_checkpoint(path / "last.pt")
        if state["epoch"] != 100:
            raise ValueError(f"Expected epoch-100 checkpoint: {path}")
        model = build_model("student", cfg).to(device).eval()
        model.load_state_dict(state["model"])
        with autocast(cfg, device):
            logits, attention = model(x, return_attention=True)
            raw = attention.topk(98, dim=1).indices
            actual = choose(attention, row["id"], active, fg)
            masked_logits = teacher(x, actual)
        records.append({"label": label, "active": active,
                        "attention": attention.float().cpu().numpy()[0],
                        "raw": raw.cpu().numpy()[0], "actual": actual.cpu().numpy()[0],
                        "student_prob": logits.float().softmax(1)[0].cpu().numpy(),
                        "teacher_prob": masked_logits.float().softmax(1)[0].cpu().numpy()})
        del model

    fig, axes = plt.subplots(len(records), 6, figsize=(18, 8.7), constrained_layout=True)
    vmax = max(float(teacher_attention.max()), *(float(r["attention"].max()) for r in records))
    true = int(row["label"])
    classes = dataset.manifest["classes"]
    for j, record in enumerate(records):
        ax = axes[j]
        ax[0].imshow(image)
        ax[1].imshow(mask, cmap="gray", vmin=0, vmax=255)
        ax[2].imshow(teacher_attention.float().cpu().numpy()[0].reshape(14, 14),
                     cmap="magma", vmin=0, vmax=vmax)
        ax[3].imshow(record["attention"].reshape(14, 14), cmap="magma", vmin=0, vmax=vmax)
        raw_image, raw = show_selected(image, record["raw"])
        actual_image, actual = show_selected(image, record["actual"])
        ax[4].imshow(raw_image)
        ax[5].imshow(actual_image)
        for patch in np.flatnonzero(raw != actual):
            y, x_patch = divmod(int(patch), 14)
            ax[5].add_patch(Rectangle((x_patch * 16 - .5, y * 16 - .5), 16, 16,
                                      fill=False, edgecolor="lime" if actual[patch] else "red", lw=1.2))
        student = record["student_prob"]
        masked = record["teacher_prob"]
        ax[0].set_ylabel(f"{record['label']}\nstudent: {classes[int(student.argmax())]} "
                         f"({student[true]:.0%} true-class)", fontsize=10)
        ax[5].text(.02, .02, f"masked teacher: {classes[int(masked.argmax())]} "
                   f"({masked[true]:.0%} true-class)\n{record['active']}",
                   transform=ax[5].transAxes, fontsize=8, color="white",
                   bbox={"facecolor": "black", "alpha": .72, "edgecolor": "none"})
        for panel in ax:
            panel.set_xticks([])
            panel.set_yticks([])
    for panel, title in zip(axes[0], ("Test image", "Object annotation", "Full teacher attention",
                                      "Student attention", "Student top-98", "Actual teacher input")):
        panel.set_title(title, fontsize=10)
    fig.colorbar(axes[0, 2].images[0], ax=list(axes[:, 2:4].ravel()), shrink=.6,
                 label="Last-layer mean-head CLS→patch attention; common color scale")
    fig.suptitle(f"Held-out test image {row['id']} · true class {classes[true]} · "
                 f"full teacher {classes[int(full_probability.argmax())]} "
                 f"({full_probability[true]:.0%} true-class)\n"
                 "Fixed test example; illustrative only. Green=added, red=removed patch.", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-root", default="outputs/experiment2")
    p.add_argument("--adaptive-root", default="outputs/adaptive100_quick")
    p.add_argument("--data-root", default="data/coco_single")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init", choices=("scratch", "imagenet"), default="scratch")
    p.add_argument("--sample-ids", nargs="+", type=int, required=True)
    p.add_argument("--output-dir", default="reports/test_attention")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = setup_device(Config.load(run_directory(args, "student") / "config.json", device=args.device))
    dataset = CocoSubset(args.data_root, "test")
    records = {row["id"]: row for row in dataset.records}
    for sample_id in args.sample_ids:
        if sample_id not in records:
            raise ValueError(f"Sample {sample_id} is not in the held-out test split")
    teacher_file = Path(args.baseline_root) / f"seed_{args.seed}" / "teacher" / "best.pt"
    teacher_hash = sha256(teacher_file)
    cfg = Config.load(run_directory(args, "student") / "config.json", data_root=args.data_root, device=args.device)
    teacher = build_model("teacher", cfg).to(device).eval()
    teacher.load_state_dict(load_checkpoint(teacher_file)["model"])
    for sample_id in args.sample_ids:
        output = Path(args.output_dir) / f"seed_{args.seed}_{args.init}_test_{sample_id}.png"
        print(render_one(args, dataset, records[sample_id], teacher, teacher_hash, device, output).resolve())


if __name__ == "__main__":
    main()
