import torch


def select_tokens(attention, method, keep=98, foreground=None, generator=None):
    """Return unique spatial patch indices and actual swap counts; CLS is separate.

    Fixed-10 random/low-score rescues do NOT consult foreground labels.
    random_matched is a probe-only control with the same feasible count as FG.
    """
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
    if method == "student":
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
        count = min(10, keep, n - keep)
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
