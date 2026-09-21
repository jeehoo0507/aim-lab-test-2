import numpy as np
from scipy.special import log_softmax


def classification(logits, labels, num_classes):
    prediction = np.asarray(logits).argmax(1)
    labels = np.asarray(labels)
    per_class = [float((prediction[labels == c] == c).mean()) if (labels == c).any() else None
                 for c in range(num_classes)]
    return {"accuracy": float((prediction == labels).mean()),
            "macro_accuracy": float(np.mean(per_class)) if all(x is not None for x in per_class) else None,
            "per_class_accuracy": per_class, "samples": len(labels)}


def kl(full_logits, other_logits):
    log_p, log_q = log_softmax(full_logits, axis=-1), log_softmax(other_logits, axis=-1)
    return (np.exp(log_p) * (log_p - log_q)).sum(-1)


def selection(indices, foreground, coverage):
    selected = np.zeros_like(foreground, dtype=bool)
    np.put_along_axis(selected, indices.astype(int), True, axis=1)
    retained = (selected & foreground).sum(1)
    total = foreground.sum(1)
    missing = np.full(len(total), np.nan)
    np.divide(total - retained, total, out=missing, where=total > 0)
    bg = 1 - retained / indices.shape[1]
    bg_prior = 1 - foreground.mean(1)
    area_total = coverage.sum(1)
    area_missing = np.full(len(total), np.nan)
    np.divide(((~selected) * coverage).sum(1), area_total, out=area_missing, where=area_total > 0)
    return {"background_ratio": bg, "foreground_missing": missing,
            "background_excess_vs_area": bg - bg_prior,
            "foreground_area_missing": area_missing,
            "zero_foreground": total == 0}
