import torch


def effective_method(method, epoch):
    """Resolve opt-in fixed switches to MaskedKD without resetting training."""
    switches = {"full_to_student_20": ("full", 20),
                "full_to_student_50": ("full", 50),
                "random_to_student_10": ("random", 10),
                "random_rescue_to_student_20": ("random_rescue_10", 20)}
    if method in switches:
        if epoch is None:
            raise ValueError("Switch schedule requires an epoch")
        first, switch = switches[method]
        return first if epoch <= switch else "student"
    return method


def annealed_swaps(epoch):
    """Fixed 100-epoch protocol: 10 through 20, linear integer decay, 0 from 80.

    Epoch 0 uses the initial budget for diagnostics. Integer ceiling avoids
    rounding to zero before the specified end of exploration.
    """
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError("A non-negative integer epoch is required")
    return max(0, min(10, (10 * (80 - epoch) + 59) // 60))


def select_tokens(attention, method, keep=98, foreground=None, generator=None, epoch=None):
    """Return unique spatial patch indices and actual swap counts; CLS is separate.

    Fixed-10 random/low-score rescues do NOT consult foreground labels.
    random_matched is a probe-only control with the same feasible count as FG.
    """
    method = effective_method(method, epoch)
    budget = annealed_swaps(epoch) if method == "random_anneal_10" else 10
    if method == "random_anneal_10":
        method = "random_rescue_10"
    attention = attention.detach()
    batch, n = attention.shape
    if not 1 <= keep <= n:
        raise ValueError("Invalid token budget")
    swaps = torch.zeros(batch, device=attention.device, dtype=torch.long)
    if method in ("ce", "full", "teacher"):
        return None, swaps
    if method == "random":
        scores = torch.rand(attention.shape, device=attention.device, generator=generator)
        return scores.topk(keep, 1).indices, swaps
    selected = attention.topk(keep, 1).indices
    if method == "student" or (method == "random_rescue_10" and budget == 0):
        return selected, swaps
    allowed = ("foreground_rescue_10", "random_rescue_10", "low_score_rescue_10", "random_matched")
    if method not in allowed:
        raise ValueError(method)
    if method in ("foreground_rescue_10", "random_matched") and foreground is None:
        raise ValueError("Foreground ground truth required for oracle/matched control")
    for i in range(batch):
        present = torch.zeros(n, dtype=torch.bool, device=attention.device)
        present[selected[i]] = True
        outside = torch.where(~present)[0]
        count = min(budget, keep, n - keep)
        outgoing = torch.arange(keep, device=attention.device)
        incoming = outside
        if method in ("foreground_rescue_10", "random_matched"):
            out_bg = torch.where(~foreground[i, selected[i]])[0]
            in_fg = torch.where(foreground[i] & ~present)[0]
            count = min(count, len(out_bg), len(in_fg))
            if method == "foreground_rescue_10":
                outgoing, incoming = out_bg, in_fg
        swaps[i] = count
        if not count:
            continue
        if method == "low_score_rescue_10":
            out = attention[i, selected[i]].argsort()[:count]
        else:
            out = outgoing[torch.randperm(len(outgoing), device=attention.device, generator=generator)[:count]]
        new = incoming[torch.randperm(len(incoming), device=attention.device, generator=generator)[:count]]
        selected[i, out] = new
    return selected, swaps


def binary_mask(indices, n=196):
    return torch.zeros(len(indices), n, dtype=torch.bool, device=indices.device).scatter_(1, indices, True)
