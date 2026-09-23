"""Plot a fixed COCO probe image from saved Experiment 2 attention arrays."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


NAMES = {
    "student": "MaskedKD",
    "random_rescue_10": "Random Rescue 10",
    "foreground_rescue_10": "FG Rescue 10",
    "low_score_rescue_10": "Low-score Rescue 10",
    "random_anneal_10": "Random 10 to 0",
}


def safe_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe dataset path: {relative}")
    return root / path


def selected_image(image, indices):
    chosen = np.zeros(196, dtype=bool)
    chosen[indices] = True
    keep = np.repeat(np.repeat(chosen.reshape(14, 14), 16, axis=0), 16, axis=1)
    return image * np.where(keep[..., None], 1.0, 0.18), chosen


def plot(root, data_root, seed, initialization, methods, epoch, class_name, sample_id, output):
    root = Path(root)
    first_run = root / f"seed_{seed}" / initialization / methods[0]
    config = json.loads((first_run / "config.json").read_text())
    data_root = Path(data_root or config["data_root"])
    manifest = json.loads((data_root / "manifest.json").read_text())
    classes = manifest["classes"]
    if class_name not in classes:
        raise ValueError(f"Unknown class {class_name!r}; available: {classes}")
    candidates = sorted((r for r in manifest["images"]
                         if r["probe"] and r["label"] == classes.index(class_name)),
                        key=lambda r: r["id"])
    if not candidates:
        raise ValueError(f"No probe image in class {class_name}")
    if sample_id is None:
        row = candidates[0]  # Fixed rule: no result-dependent example selection.
    else:
        row = next((r for r in candidates if r["id"] == sample_id), None)
        if row is None:
            raise ValueError(f"ID {sample_id} is not a {class_name} probe image")

    image_path = safe_path(data_root, row["image"])
    mask_path = safe_path(data_root, row["mask"])
    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)) / 255.0
    with Image.open(mask_path) as source:
        mask = np.asarray(source.convert("L").resize((224, 224), Image.Resampling.NEAREST))

    records = []
    for method in methods:
        file = root / f"seed_{seed}" / initialization / method / "probe" / f"epoch_{epoch:03d}.npz"
        with np.load(file, allow_pickle=False) as arrays:
            matches = np.flatnonzero(arrays["sample_id"] == row["id"])
            if len(matches) != 1:
                raise ValueError(f"Expected sample {row['id']} exactly once in {file}")
            i = int(matches[0])
            records.append({key: arrays[key][i].copy() for key in
                            ("attention", "teacher_attention", "raw_indices", "actual_indices")})

    vmax = max(float(r[key].max()) for r in records for key in ("attention", "teacher_attention"))
    fig, axes = plt.subplots(len(methods), 6, figsize=(16, 3.2 * len(methods)), squeeze=False,
                             constrained_layout=True)
    attention_axes = []
    for j, (method, record) in enumerate(zip(methods, records)):
        ax = axes[j]
        ax[0].imshow(image)
        ax[1].imshow(mask, cmap="gray", vmin=0, vmax=255)
        ax[2].imshow(record["teacher_attention"].reshape(14, 14), cmap="magma", vmin=0, vmax=vmax)
        ax[3].imshow(record["attention"].reshape(14, 14), cmap="magma", vmin=0, vmax=vmax)
        attention_axes.extend(ax[2:4])
        raw_image, raw = selected_image(image, record["raw_indices"].astype(int))
        actual_image, actual = selected_image(image, record["actual_indices"].astype(int))
        ax[4].imshow(raw_image)
        ax[5].imshow(actual_image)
        for patch in np.flatnonzero(raw != actual):
            y, x = divmod(int(patch), 14)
            color = "lime" if actual[patch] else "red"
            ax[5].add_patch(Rectangle((x * 16 - 0.5, y * 16 - 0.5), 16, 16,
                                      fill=False, edgecolor=color, linewidth=1.4))
        ax[0].set_ylabel(NAMES.get(method, method), fontsize=11)
        ax[3].text(0.03, 0.03, f"max {record['attention'].max():.3f}", color="white",
                   transform=ax[3].transAxes, fontsize=9,
                   bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"})
        ax[5].text(0.03, 0.03, f"{int((~raw & actual).sum())} added", color="white",
                   transform=ax[5].transAxes, fontsize=9,
                   bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"})
        for panel in ax:
            panel.set_xticks([])
            panel.set_yticks([])
    for ax, title in zip(axes[0],
                         ("Input", "Object mask", "Teacher attention", "Student attention",
                          "Student top-98", "Actual teacher input")):
        ax.set_title(title)
    bar = fig.colorbar(axes[0, 2].images[0], ax=attention_axes, shrink=0.65, pad=0.01)
    bar.set_label("CLS to patch attention (same scale across panels)")
    fig.suptitle(f"COCO probe: {class_name}, image {row['id']} | seed {seed}, {initialization}, epoch {epoch}\n"
                 "First probe image by ID; green = added patch, red = removed patch", fontsize=13)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/experiment2")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init", choices=("scratch", "imagenet"), default="imagenet")
    parser.add_argument("--methods", nargs="+", default=("student", "random_rescue_10", "foreground_rescue_10"))
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--class-name", default="bird")
    parser.add_argument("--sample-id", type=int, default=None)
    parser.add_argument("--output", default="reports/attention_maps/seed0_imagenet_bird.png")
    args = parser.parse_args()
    plot(args.root, args.data_root, args.seed, args.init, args.methods, args.epoch,
         args.class_name, args.sample_id, args.output)
