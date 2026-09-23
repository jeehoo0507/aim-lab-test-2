from pathlib import Path

import numpy as np
import torch

from .data import CocoSubset, loader
from .masking import binary_mask, effective_method, select_tokens
from .utils import autocast, save_npz, write_json

DIAGNOSTICS = ("student", "random", "random_rescue_10", "foreground_rescue_10",
               "low_score_rescue_10", "random_matched")


class Probe:
    def __init__(self, cfg, directory, teacher, device):
        self.cfg, self.teacher, self.device = cfg, teacher, device
        self.dataset = CocoSubset(cfg.data_root, "probe", threshold=cfg.foreground_threshold)
        self.path = Path(directory) / "probe"
        self.full_cache = None
        write_json(self.path / "manifest.json", {"ids": [r["id"] for r in self.dataset.records],
                   "split": "val", "transform": "full_image_resize224_no_augmentation",
                   "selection": "last_layer_head_mean_CLS_to_patch_before_rescue",
                   "diagnostic_repeats": cfg.diagnostic_repeats,
                   "randomness": "independent image/method/repeat streams, fixed across epochs and training seeds"})

    def choose(self, attention, method, foreground, ids, repeat=0, epoch=None):
        method = effective_method(method, epoch)
        if method in ("ce", "full"):
            return torch.arange(196, device=self.device).expand(len(ids), -1), torch.zeros(len(ids), device=self.device, dtype=torch.long)
        chosen, counts = [], []
        # Match the original Random-10 stream at the start of the schedule.
        index = DIAGNOSTICS.index("random_rescue_10" if method == "random_anneal_10" else method)
        for i, sample_id in enumerate(ids):
            # Stable across batch size, epoch, initialization and training seed.
            seed = int(sample_id) * 1009 + 10000019 * index + 65537 * repeat
            generator = torch.Generator(device=self.device).manual_seed(seed)
            idx, swaps = select_tokens(attention[i:i + 1], method, foreground=foreground[i:i + 1], generator=generator, epoch=epoch)
            chosen.append(idx)
            counts.append(swaps)
        return torch.cat(chosen), torch.cat(counts)

    @torch.inference_mode()
    def log(self, model, epoch, method, name=None):
        model.eval()
        records, extra = [], []
        detailed = epoch in self.cfg.diagnostic_epochs or epoch == self.cfg.epochs or name == "best"
        offset = 0
        for batch in loader(self.dataset, self.cfg):
            x, fg = batch["image"].to(self.device), batch["foreground"].to(self.device)
            with autocast(self.cfg, self.device):
                logits, attention = model(x, return_attention=True)
                if self.full_cache is None:
                    full, teacher_attention = self.teacher(x, return_attention=True)
                    full, teacher_attention = full.float().cpu().numpy(), teacher_attention.float().cpu().numpy()
                else:
                    full = self.full_cache[0][offset:offset + len(x)]
                    teacher_attention = self.full_cache[1][offset:offset + len(x)]
                raw, _ = select_tokens(attention, "student")
                actual, swaps = self.choose(attention, method, fg, batch["sample_id"], epoch=epoch)
                raw_logits = self.teacher(x, raw).float().cpu().numpy()
                actual_logits = full if effective_method(method, epoch) in ("ce", "full") else self.teacher(x, actual).float().cpu().numpy()
            row = {key: batch[key].numpy() for key in ("sample_id", "label", "foreground", "coverage")}
            row.update(attention=attention.float().cpu().numpy(), raw_indices=raw.cpu().numpy().astype(np.int16),
                       actual_indices=actual.cpu().numpy().astype(np.int16), swaps=swaps.cpu().numpy(),
                       student_logits=logits.float().cpu().numpy(), teacher_full_logits=full,
                       teacher_attention=teacher_attention, teacher_raw_logits=raw_logits,
                       teacher_actual_logits=actual_logits)
            records.append(row)
            if detailed:
                variants = {}
                for mode in DIAGNOSTICS:
                    all_indices, all_logits, all_swaps = [], [], []
                    for repeat in range(self.cfg.diagnostic_repeats):
                        indices, swaps = self.choose(attention, mode, fg, batch["sample_id"], repeat)
                        with autocast(self.cfg, self.device):
                            output = self.teacher(x, indices)
                        all_indices.append(indices.cpu().numpy().astype(np.int16))
                        all_logits.append(output.float().cpu().numpy())
                        all_swaps.append(swaps.cpu().numpy())
                    variants[f"indices__{mode}"] = np.stack(all_indices)
                    variants[f"logits__{mode}"] = np.stack(all_logits)
                    variants[f"swaps__{mode}"] = np.stack(all_swaps)
                extra.append(variants)
            offset += len(x)
        arrays = {k: np.concatenate([r[k] for r in records]) for k in records[0]}
        if self.full_cache is None:
            self.full_cache = arrays["teacher_full_logits"], arrays["teacher_attention"]
        stem = name or f"epoch_{epoch:03d}"
        save_npz(self.path / f"{stem}.npz", epoch=np.array(epoch), **arrays)
        if detailed:
            detail = {k: np.concatenate([r[k] for r in extra], axis=1) for k in extra[0]}
            save_npz(self.path / f"{stem}_counterfactual.npz", epoch=np.array(epoch),
                     sample_id=arrays["sample_id"], label=arrays["label"], foreground=arrays["foreground"],
                     coverage=arrays["coverage"], teacher_full_logits=arrays["teacher_full_logits"], **detail)
