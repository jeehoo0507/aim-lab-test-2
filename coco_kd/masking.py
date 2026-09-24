import torch


def effective_method(method, epoch):
    """Resolve opt-in fixed switches without resetting training."""
    switches = {"full_to_student_20": ("full", 20),
                "full_to_student_50": ("full", 50),
                "random_to_student_10": ("random", 10),
                "random_to_student_20": ("random", 20),
                "random_to_student_50": ("random", 50),
                "random_rescue_to_student_20": ("random_rescue_10", 20)}
    low_switches = {"random_rescue_to_low_50": 50,
                    "random_rescue_to_low_70": 70}
    if method in low_switches:
        if epoch is None:
            raise ValueError("Switch schedule requires an epoch")
        return "random_rescue_10" if epoch <= low_switches[method] else "low_score_rescue_10"
    if method == "random_low_mixed_10":
        if epoch is None:
            raise ValueError("Mixed schedule requires an epoch")
        return "random_rescue_10" if epoch <= 50 else method
    if method in switches:
        if epoch is None:
            raise ValueError("Switch schedule requires an epoch")
        first, switch = switches[method]
        return first if epoch <= switch else "student"
    return method


def mixed_low_slots(epoch):
    """Number of low-attention outgoing patches in the fixed 10-swap budget."""
    if epoch is None or epoch < 0:
        raise ValueError("Mixed schedule requires a non-negative epoch")
    return 0 if epoch <= 50 else 5 if epoch <= 70 else 8


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
    if method == "dino":
        method = "student"
    elif method == "dino_random_rescue_10":
        method = "random_rescue_10"
    elif method == "dino_paper_late":
        if epoch is None or epoch < 0:
            raise ValueError("DINO paper schedule requires a nonnegative epoch")
        if epoch <= 50:
            method = "student"
        else:
            if keep != 98 or attention.shape[1] != 196:
                raise ValueError("DINO paper control requires 196 patches and keep=98")
            # Table 5: 40% high-attention + 10% random, in the later half.
            # Round 196*.4 to 78; draw the remaining 20 from all other 118.
            ranked = attention.detach().argsort(dim=1, descending=True)
            top, pool = ranked[:, :78], ranked[:, 78:]
            draws = torch.rand(pool.shape, device=attention.device, generator=generator).topk(20, 1).indices
            chosen = torch.cat((top, pool.gather(1, draws)), dim=1)
            original = torch.zeros_like(attention, dtype=torch.bool).scatter_(1, ranked[:, :98], True)
            return chosen, (~original.gather(1, chosen)).sum(1)
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
    allowed = ("foreground_rescue_10", "random_rescue_10", "low_score_rescue_10",
               "random_low_mixed_10", "random_matched")
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
        elif method == "random_low_mixed_10":
            ordered = attention[i, selected[i]].argsort()
            low_count = min(mixed_low_slots(epoch), count)
            low = ordered[:low_count]
            remaining = ordered[low_count:]
            random = remaining[torch.randperm(len(remaining), device=attention.device,
                                               generator=generator)[:count - low_count]]
            out = torch.cat((low, random))
        else:
            out = outgoing[torch.randperm(len(outgoing), device=attention.device, generator=generator)[:count]]
        new = incoming[torch.randperm(len(incoming), device=attention.device, generator=generator)[:count]]
        selected[i, out] = new
    return selected, swaps


def binary_mask(indices, n=196):
    return torch.zeros(len(indices), n, dtype=torch.bool, device=indices.device).scatter_(1, indices, True)
