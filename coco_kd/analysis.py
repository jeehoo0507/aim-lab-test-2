"""CPU-only analysis; no model loading and no access to training images required."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from .metrics import kl, selection
from .utils import write_json

NAMES = {"ce": "CE", "full": "Full KD", "random": "Random mask", "student": "MaskedKD",
         "random_rescue_10": "Random rescue 10", "foreground_rescue_10": "FG rescue 10",
         "low_score_rescue_10": "Low-score rescue 10", "random_anneal_10": "Random 10 to 0"}


def correction_summary(frame):
    """Fixed epoch-0 cohort. Report failures as censored, not discarded."""
    initial = frame[frame.epoch == 0]
    ids = initial.loc[initial.teacher_full_correct & ~initial.student_correct, "sample_id"]
    horizon = int(frame.epoch.max())
    times = []
    for sample_id in ids:
        sample = frame[frame.sample_id == sample_id].sort_values("epoch")
        correct = sample.student_correct.to_numpy(bool)
        epochs = sample.epoch.to_numpy(int)
        first = None
        for i in range(len(sample) - 2):
            if epochs[i] >= 1 and epochs[i + 2] - epochs[i] == 2 and correct[i:i + 3].all():
                first = int(epochs[i])
                break
        times.append(horizon + 1 if first is None else first)
    values = np.array(times)
    return {"epoch0_correction_opportunities": len(ids), "horizon": horizon,
            "corrected_by_25": float((values <= 25).mean()) if len(values) and horizon >= 27 else None,
            "corrected_by_50": float((values <= 50).mean()) if len(values) and horizon >= 52 else None,
            "corrected_by_end": float((values <= horizon).mean()) if len(values) else None,
            "mean_wait_capped_at_horizon_plus1": float(values.mean()) if len(values) else None,
            "criterion": "first of 3 consecutive correct probe epochs; non-corrected retained as censored"}


def convergence_summary(curves, threshold=.8):
    rows = []
    if curves.empty:
        return pd.DataFrame()
    for keys, group in curves.groupby(["seed", "initialization", "method"]):
        group = group.sort_values("epoch").reset_index(drop=True)
        above = group.val_macro_accuracy.to_numpy() >= threshold
        first = next((i for i in range(len(group) - 2) if above[i:i + 3].all()), None)
        rows.append({"seed": keys[0], "initialization": keys[1], "method": keys[2],
                     "threshold": threshold, "reached_three_epochs": first is not None,
                     "first_epoch": int(group.epoch.iloc[first]) if first is not None else None,
                     "updates_to_first": int(group.optimizer_updates.iloc[:first + 1].sum()) if first is not None else None,
                     "training_seconds_to_first": float(group.train_seconds.iloc[:first + 1].sum()) if first is not None else None,
                     "wall_seconds_to_first_with_probe": float(group.epoch_seconds_with_probe.iloc[:first + 1].sum()) if first is not None else None,
                     "last10_val_mean": group.val_macro_accuracy.tail(10).mean(),
                     "previous10_val_mean": group.val_macro_accuracy.iloc[-20:-10].mean() if len(group) >= 20 else None})
    return pd.DataFrame(rows)


def probe_frame(file, seed, initialization, method):
    with np.load(file, allow_pickle=False) as a:
        raw = selection(a["raw_indices"], a["foreground"], a["coverage"])
        actual = selection(a["actual_indices"], a["foreground"], a["coverage"])
        student, full, masked = a["student_logits"], a["teacher_full_logits"], a["teacher_actual_logits"]
        labels = a["label"]
        s_correct, t_correct = student.argmax(1) == labels, full.argmax(1) == labels
        m_correct = masked.argmax(1) == labels
        d = {"sample_id": a["sample_id"], "label": labels, "epoch": int(a["epoch"]),
             "seed": seed, "initialization": initialization, "method": method,
             "student_correct": s_correct, "teacher_full_correct": t_correct,
             "teacher_masked_correct": m_correct, "teacher_raw_correct": a["teacher_raw_logits"].argmax(1) == labels,
             "student_prediction": student.argmax(1), "teacher_prediction": full.argmax(1),
             "teacher_masked_prediction": masked.argmax(1),
             "teacher_mask_kl": kl(full, masked), "teacher_raw_kl": kl(full, a["teacher_raw_logits"]),
             "teacher_student_kl": kl(full, student), "prediction_agreement": student.argmax(1) == full.argmax(1),
             "teacher_flip": t_correct & ~m_correct,
             "correction_opportunity": t_correct & ~s_correct,
             "correction_signal_preserved": t_correct & ~s_correct & m_correct,
             "swaps": a["swaps"]}
        for prefix, metrics in (("raw", raw), ("actual", actual)):
            d.update({f"{prefix}_{key}": value for key, value in metrics.items()})
        before = np.zeros_like(a["foreground"], dtype=bool)
        after = before.copy()
        np.put_along_axis(before, a["raw_indices"].astype(int), True, 1)
        np.put_along_axis(after, a["actual_indices"].astype(int), True, 1)
        d["added_foreground"] = ((after & ~before) & a["foreground"]).sum(1)
        d["removed_foreground"] = ((before & ~after) & a["foreground"]).sum(1)
        return pd.DataFrame(d)


def paired_differences(results):
    rows = []
    if results.empty:
        return pd.DataFrame()
    for initialization in results.initialization.unique():
        for checkpoint in ("best", "last"):
            subset = results[(results.initialization == initialization) & (results.checkpoint == checkpoint)]
            for metric in ("macro_accuracy", "accuracy"):
                pivot = subset.pivot(index="seed", columns="method", values=metric)
                for method in NAMES:
                    if method == "student" or method not in pivot or "student" not in pivot:
                        continue
                    difference = (pivot[method] - pivot.student).dropna() * 100
                    n = len(difference)
                    if not n:
                        continue
                    half = t.ppf(.975, n - 1) * difference.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
                    rows.append({"initialization": initialization, "checkpoint": checkpoint, "method": method,
                                 "reference": "student", "metric": metric, "n_seeds": n,
                                 "mean_difference_pp": difference.mean(), "ci_low_pp": difference.mean() - half,
                                 "ci_high_pp": difference.mean() + half,
                                 "seed_differences_pp": json.dumps(difference.to_dict()),
                                 "interpretation": "exploratory paired t interval; n=3 insufficient for strong claims"})
    return pd.DataFrame(rows)


def analyze(output_root):
    root = Path(output_root)
    destination = root / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "per_image").mkdir(exist_ok=True)
    (destination / "per_image_counterfactual").mkdir(exist_ok=True)
    debug = False
    curves, probes, results, counterfactual, corrections = [], [], [], [], []
    for run in sorted(root.glob("seed_*/*/*")):
        if not (run / "result.json").exists():
            continue
        info = json.loads((run / "result.json").read_text())
        debug |= info["config"]["model_scale"] == "debug"
        seed, initialization, method = info["seed"], info["student_init"], info["method"]
        run_key = {"seed": seed, "initialization": initialization, "method": method}
        for h in json.loads((run / "history.json").read_text()):
            curves.append({**run_key, "epoch": h["epoch"], "lr": h["lr"],
                           "val_macro_accuracy": h["validation"]["macro_accuracy"],
                           "val_accuracy": h["validation"]["accuracy"], "train_loss": h["train"]["loss"],
                           "optimizer_updates": h["train"]["optimizer_updates"],
                           "train_seconds": h["train"]["seconds"], "epoch_seconds_with_probe": h["epoch_seconds_with_probe"],
                           "swaps": h["train"]["swaps"], "added_foreground": h["train"]["added_foreground"],
                           "removed_foreground": h["train"]["removed_foreground"]})
        image_frames, counter_frames = [], []
        for file in sorted((run / "probe").glob("epoch_*.npz")):
            if "counterfactual" in file.name:
                with np.load(file, allow_pickle=False) as a:
                    for key in a.files:
                        if not key.startswith("logits__"):
                            continue
                        mode = key.split("__", 1)[1]
                        for repeat, logits in enumerate(a[key]):
                            metrics = selection(a[f"indices__{mode}"][repeat], a["foreground"], a["coverage"])
                            full_correct = a["teacher_full_logits"].argmax(1) == a["label"]
                            masked_correct = logits.argmax(1) == a["label"]
                            counter_frames.append(pd.DataFrame({**run_key, "epoch": int(a["epoch"]), "mask": mode,
                                "repeat": repeat, "sample_id": a["sample_id"], "label": a["label"],
                                "teacher_prediction": logits.argmax(1), "teacher_correct": masked_correct,
                                "full_correct": full_correct, "kl": kl(a["teacher_full_logits"], logits),
                                "background_ratio": metrics["background_ratio"], "foreground_missing": metrics["foreground_missing"],
                                "swaps": a[f"swaps__{mode}"][repeat]}))
                            counterfactual.append({**run_key, "epoch": int(a["epoch"]), "mask": mode, "repeat": repeat,
                                                  "teacher_accuracy": masked_correct.mean(), "teacher_full_accuracy": full_correct.mean(),
                                                  "kl": kl(a["teacher_full_logits"], logits).mean(),
                                                  "full_correct_to_wrong": (full_correct & ~masked_correct).mean(),
                                                  "background_ratio": metrics["background_ratio"].mean(),
                                                  "foreground_missing": np.nanmean(metrics["foreground_missing"]),
                                                  "swaps": a[f"swaps__{mode}"][repeat].mean()})
                continue
            frame = probe_frame(file, seed, initialization, method)
            image_frames.append(frame)
            summary = frame.select_dtypes(include=[np.number, bool]).mean().to_dict()
            summary.pop("sample_id", None)
            summary.pop("label", None)
            probes.append({**summary, **run_key, "epoch": int(frame.epoch.iloc[0]), "n_probe": len(frame)})
        if image_frames:
            combined = pd.concat(image_frames)
            combined.to_csv(destination / "per_image" / f"seed_{seed}_{initialization}_{method}.csv.gz", index=False)
            corrections.append({**run_key, **correction_summary(combined)})
        if counter_frames:
            pd.concat(counter_frames).to_csv(destination / "per_image_counterfactual" / f"seed_{seed}_{initialization}_{method}.csv.gz", index=False)
        if (run / "test_metrics.json").exists():
            metrics = json.loads((run / "test_metrics.json").read_text())
            for checkpoint, value in metrics.items():
                results.append({**run_key, "checkpoint": checkpoint, "epoch": value["epoch"],
                                "accuracy": value["accuracy"], "macro_accuracy": value["macro_accuracy"],
                                **{f"class_{i}": a for i, a in enumerate(value["per_class_accuracy"])}})
    frames = {"learning_curves": pd.DataFrame(curves), "probe_curves": pd.DataFrame(probes),
              "test_results": pd.DataFrame(results), "counterfactual": pd.DataFrame(counterfactual),
              "error_correction": pd.DataFrame(corrections)}
    frames["paired_seed_differences"] = paired_differences(frames["test_results"])
    frames["convergence"] = convergence_summary(frames["learning_curves"])
    for name, frame in frames.items():
        if not frame.empty:
            frame.to_csv(destination / f"{name}.csv", index=False)
    figures(frames, destination, debug)
    write_json(destination / "analysis_info.json", {"completed_student_runs": len({(r['seed'], r['initialization'], r['method']) for r in curves}),
               "test_available": bool(results), "synthetic_debug": debug,
               "note": "Raw selection is before rescue; actual selection is teacher input. Repeated masks are not independent training seeds."})
    return destination


def figures(frames, destination, debug=False):
    curves, probes = frames["learning_curves"], frames["probe_curves"]
    if curves.empty:
        return
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    for initialization in curves.initialization.unique():
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for method, label in NAMES.items():
            c = curves[(curves.initialization == initialization) & (curves.method == method)]
            p = probes[(probes.initialization == initialization) & (probes.method == method)] if not probes.empty else pd.DataFrame()
            for ax, frame, value, ylabel, factor in (
                (axes[0, 0], c, "val_macro_accuracy", "Validation macro accuracy (%)", 100),
                (axes[0, 1], p, "teacher_mask_kl", "Teacher KL (full || actual input)", 1),
                (axes[1, 0], p, "raw_background_ratio", "Student's own background selection (%)", 100),
                (axes[1, 1], p, "raw_foreground_missing", "Student's own foreground missing (%)", 100)):
                if frame.empty:
                    continue
                groups = frame.groupby("epoch")[value]
                mean, std = groups.mean() * factor, groups.std().fillna(0) * factor
                line, = ax.plot(mean.index, mean, label=label, linewidth=1.6)
                ax.fill_between(mean.index, mean - std, mean + std, color=line.get_color(), alpha=.1)
                ax.set(xlabel="Epoch", ylabel=ylabel)
                ax.grid(alpha=.15)
        axes[0, 0].legend(fontsize=8)
        title = "SYNTHETIC SMOKE TEST" if debug else "COCO single-label classification"
        count = curves[curves.initialization == initialization].seed.nunique()
        fig.suptitle(f"{title} | {initialization} | {count} seed(s), mean ± seed SD")
        fig.savefig(destination / f"main_{initialization}.png", dpi=180)
        fig.savefig(destination / f"main_{initialization}.svg")
        plt.close(fig)
