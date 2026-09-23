"""Paired, fixed-probe counterfactuals for an opt-in masking switch.

The gate never runs in a training minibatch. Random10 and its candidate receive
the same ten incoming patches, so only the outgoing policy differs for Low10.
"""
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import CocoSubset, loader
from .masking import binary_mask
from .utils import autocast, save_npz, write_json

ADAPTIVE_TARGETS = {"adaptive_random_to_low_10": "low_score_rescue_10",
                    "adaptive_random_to_student": "student"}


def paired_indices(attention, sample_ids, repeat, device):
    """Return Random10, Low10, and raw top-98 indices with a shared random draw."""
    raw = attention.detach().topk(98, dim=1).indices
    random_masks, low_masks = [], []
    for i, sample_id in enumerate(sample_ids):
        generator = torch.Generator(device=device).manual_seed(int(sample_id) * 1009 + 65537 * repeat + 48151623)
        present = binary_mask(raw[i:i + 1])[0]
        outside = torch.where(~present)[0]
        incoming = outside[torch.randperm(len(outside), device=device, generator=generator)[:10]]
        random_positions = torch.randperm(98, device=device, generator=generator)[:10]
        low_positions = attention[i, raw[i]].argsort()[:10]
        random_choice, low_choice = raw[i].clone(), raw[i].clone()
        random_choice[random_positions] = incoming
        low_choice[low_positions] = incoming
        random_masks.append(random_choice)
        low_masks.append(low_choice)
    return raw, torch.stack(random_masks), torch.stack(low_masks)


class GateProbe:
    def __init__(self, cfg, directory, teacher, device):
        self.cfg, self.teacher, self.device = cfg, teacher, device
        self.dataset = CocoSubset(cfg.data_root, "probe", threshold=cfg.foreground_threshold)
        self.path = Path(directory) / "gate"
        self.full_cache = None
        previous = sorted(self.path.glob("epoch_*.npz"))
        if previous:
            with np.load(previous[0], allow_pickle=False) as saved:
                ids = [r["id"] for r in self.dataset.records]
                if saved["sample_id"].tolist() != ids:
                    raise ValueError("Gate cache probe IDs changed; use a new output root")
                self.full_cache = saved["full_logits"].copy()
        write_json(self.path / "manifest.json", {
            "ids": [r["id"] for r in self.dataset.records],
            "rule": "two consecutive favorable checks, at/after min_epoch; switch applies next epoch",
            "favorable": "candidate has higher ground-truth log probability, lower KL from full teacher, "
                         "and no more full-correct-to-wrong flips than paired Random10",
            "incoming": "same ten unselected patches for Random10 and Low10, fixed by image ID and repeat",
            "interval": cfg.gate_interval, "min_epoch": cfg.gate_min_epoch, "repeats": cfg.gate_repeats,
            "checkpoint_selection": "validation excludes these probe images when validation_exclude_probe=true"})

    @torch.inference_mode()
    def measure(self, model, epoch, target):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        model.eval()
        batches = []
        offset = 0
        uncached_full = self.full_cache is None
        for batch in loader(self.dataset, self.cfg):
            x, labels = batch["image"].to(self.device), batch["label"].to(self.device)
            with autocast(self.cfg, self.device):
                _, attention = model(x, return_attention=True)
                full = (self.teacher(x).float() if self.full_cache is None else
                        torch.from_numpy(self.full_cache[offset:offset + len(x)]).to(self.device))
                raw, _, _ = paired_indices(attention, batch["sample_id"], 0, self.device)
                raw_logits = self.teacher(x, raw).float() if target == "student" else None
            full_correct = full.argmax(1) == labels
            random_logits, candidate_logits = [], []
            random_indices, candidate_indices = [], []
            for repeat in range(self.cfg.gate_repeats):
                raw, random_choice, low_choice = paired_indices(attention, batch["sample_id"], repeat, self.device)
                candidate_choice = raw if target == "student" else low_choice
                with autocast(self.cfg, self.device):
                    r = self.teacher(x, random_choice).float()
                    c = raw_logits if raw_logits is not None else self.teacher(x, candidate_choice).float()
                random_logits.append(r.cpu().numpy())
                candidate_logits.append(c.cpu().numpy())
                random_indices.append(random_choice.cpu().numpy().astype(np.int16))
                candidate_indices.append(candidate_choice.cpu().numpy().astype(np.int16))
            batches.append({"sample_id": batch["sample_id"].numpy(), "label": labels.cpu().numpy(),
                            "foreground": batch["foreground"].numpy(), "attention": attention.float().cpu().numpy(),
                            "full_correct": full_correct.cpu().numpy(), "full_logits": full.cpu().numpy(),
                            "random_logits": np.stack(random_logits), "candidate_logits": np.stack(candidate_logits),
                            "random_indices": np.stack(random_indices), "candidate_indices": np.stack(candidate_indices)})
            offset += len(x)
        arrays = {key: np.concatenate([row[key] for row in batches], axis=1 if key in
                  ("random_logits", "candidate_logits", "random_indices", "candidate_indices") else 0)
                  for key in batches[0]}
        save_npz(self.path / f"epoch_{epoch:03d}.npz", epoch=np.array(epoch), **arrays)
        if self.full_cache is None:
            self.full_cache = arrays["full_logits"]
        full = torch.from_numpy(arrays["full_logits"])
        labels = torch.from_numpy(arrays["label"]).long()
        correct = torch.from_numpy(arrays["full_correct"])
        p_full = F.softmax(full, dim=1)
        observations = {}
        for key in ("random", "candidate"):
            logits = torch.from_numpy(arrays[f"{key}_logits"])  # repeat, image, class
            logp = F.log_softmax(logits, dim=-1)
            kl = (p_full[None] * (F.log_softmax(full, dim=-1)[None] - logp)).sum(-1)
            true_logp = logp.gather(-1, labels[None, :, None].expand(logits.shape[0], -1, 1)).squeeze(-1)
            flips = correct[None] & (logits.argmax(-1) != labels[None])
            observations[key] = {"kl": float(kl.mean()),
                                 "true_logp_on_full_correct": float(true_logp[:, correct].mean()) if correct.any() else 0.0,
                                 "full_correct_to_wrong": float(flips.float().mean()),
                                 "accuracy": float((logits.argmax(-1) == labels[None]).float().mean())}
        random, candidate = observations["random"], observations["candidate"]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        metrics = {"epoch": epoch, "n_images": len(labels), "full_teacher_correct": int(correct.sum()),
                   "diagnostic_seconds": time.perf_counter() - started,
                   "teacher_full_image_passes": len(labels) if uncached_full else 0,
                   "teacher_98_image_passes": len(labels) * (1 + self.cfg.gate_repeats)
                                              if target == "student" else len(labels) * 2 * self.cfg.gate_repeats,
                   "random": random, "candidate": candidate,
                   "kl_gain": random["kl"] - candidate["kl"],
                   "true_logp_gain": candidate["true_logp_on_full_correct"] - random["true_logp_on_full_correct"],
                   "flip_gain": random["full_correct_to_wrong"] - candidate["full_correct_to_wrong"]}
        metrics["favorable"] = (metrics["kl_gain"] > 0 and metrics["true_logp_gain"] > 0
                                and metrics["flip_gain"] >= 0)
        write_json(self.path / f"epoch_{epoch:03d}.json", metrics)
        return metrics


def update_gate(state, metrics, cfg):
    """Pure decision function; persisted in the full last.pt resume checkpoint."""
    if state["switched_after_epoch"] is not None:
        return state
    eligible = metrics["epoch"] >= cfg.gate_min_epoch and metrics["favorable"]
    state = dict(state)
    state["favorable_streak"] = state["favorable_streak"] + 1 if eligible else 0
    if state["favorable_streak"] >= 2:
        state["switched_after_epoch"] = metrics["epoch"]
    return state
