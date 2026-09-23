import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

METHODS = ("ce", "full", "random", "student", "random_rescue_10",
           "foreground_rescue_10", "low_score_rescue_10")
# Opt-in follow-up; keep the original seven-method pipeline unchanged.
TRAIN_METHODS = (*METHODS, "random_anneal_10")
CLASSES = ("giraffe", "airplane", "clock", "zebra", "train", "bird",
           "elephant", "toilet", "cow", "bear")


@dataclass
class Config:
    data_root: str = "data/coco_single"
    output_root: str = "outputs/experiment2"
    device: str = "cuda"
    num_classes: int = 10
    batch_size: int = 32
    eval_batch_size: int = 32
    accumulation_steps: int = 4
    num_workers: int = 2
    num_threads: int = 2
    epochs: int = 100
    teacher_epochs: int = 30
    seed: int = 0
    scratch_lr: float = 5e-4
    pretrained_lr: float = 5e-5
    teacher_lr: float = 5e-5
    min_lr: float = 1e-6
    warmup_epochs: int = 5
    weight_decay: float = 0.05
    drop_path: float = 0.1
    label_smoothing: float = 0.1
    kd_alpha: float = 0.5
    temperature: float = 1.0
    keep_tokens: int = 98
    foreground_threshold: float = 0.5
    checkpoint_every: int = 50
    checkpoint_target_gb: float = 5.0
    output_warning_gb: float = 10.0
    diagnostic_epochs: tuple = (0, 10, 25, 50, 75, 100)
    diagnostic_repeats: int = 5
    student_init: str = "imagenet"
    teacher_pretrained: bool = True
    amp: bool = True
    grad_clip: float = 1.0
    model_scale: str = "deit"
    max_train_batches: int | None = None

    @classmethod
    def load(cls, path=None, **overrides):
        values = json.loads(Path(path).read_text()) if path else {}
        values.update({k: v for k, v in overrides.items() if v is not None})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg = cls(**values)
        for key in ("num_classes", "batch_size", "eval_batch_size", "accumulation_steps",
                    "epochs", "teacher_epochs", "num_threads", "checkpoint_every", "diagnostic_repeats"):
            if getattr(cfg, key) < 1:
                raise ValueError(f"{key} must be positive")
        if cfg.num_classes < 2 or cfg.keep_tokens != 98:
            raise ValueError("Use at least 2 classes and 98 kept patches for this experiment")
        if cfg.num_workers < 0 or cfg.warmup_epochs < 0:
            raise ValueError("Invalid worker/warmup count")
        if cfg.student_init not in ("scratch", "imagenet") or cfg.model_scale not in ("debug", "deit"):
            raise ValueError("Invalid initialization/model scale")
        if not 0 < cfg.foreground_threshold <= 1 or not 0 <= cfg.kd_alpha <= 1 or cfg.temperature <= 0:
            raise ValueError("Invalid masking/KD parameters")
        if not 0 <= cfg.label_smoothing < 1 or not 0 <= cfg.drop_path < 1:
            raise ValueError("Invalid smoothing/drop path")
        if min(cfg.scratch_lr, cfg.pretrained_lr, cfg.teacher_lr, cfg.grad_clip) <= 0 or cfg.min_lr < 0 or cfg.weight_decay < 0:
            raise ValueError("Invalid optimization parameters")
        if cfg.checkpoint_target_gb <= 0 or cfg.output_warning_gb <= cfg.checkpoint_target_gb:
            raise ValueError("output_warning_gb must exceed the positive checkpoint_target_gb")
        if cfg.max_train_batches is not None and cfg.max_train_batches < 1:
            raise ValueError("max_train_batches must be positive")
        return cfg

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))
