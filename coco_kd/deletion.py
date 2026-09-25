"""Teacher-input deletion with an exact, local virtual-logit learning audit.

This certifies only the evaluated virtual update, not future student accuracy.
The existing experiment's model, loss and mask implementations remain intact.
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .masking import select_tokens
from .utils import autocast

METHODS = ("student", "random_rescue_10", "rescue_audit", "deterministic_audit", "full")


@dataclass
class AuditSettings:
    epsilon: float = 0.05
    etas: tuple = (0.1, 0.5, 1.0)
    swap_budgets: tuple = (10, 5)
    min_gain: float = 1e-8
    tolerance: float = 1e-10
    candidate_batch: int = 3
    max_views_per_image: int = 24

    def __post_init__(self):
        if not 0 <= self.epsilon < 1 or not self.etas or any(not 0 < e <= 1 for e in self.etas):
            raise ValueError("Invalid audit epsilon/etas")
        if tuple(self.swap_budgets) != (10, 5):
            raise ValueError("This pilot pre-registers swap budgets [10, 5]")
        if self.min_gain <= 0 or self.tolerance < 0 or self.candidate_batch < 1 or self.max_views_per_image < 9:
            raise ValueError("Invalid audit numeric/compute limits")


def student_forward(model, images):
    """Reuse the normal full-input forward; no extra stochastic student pass."""
    captured = []
    handle = model.norm.register_forward_hook(lambda _m, _i, out: captured.append(out[:, 1:].detach()))
    try:
        logits, attention = model(images, return_attention=True)
    finally:
        handle.remove()
    return logits, attention, captured[0]


@torch.no_grad()
def audit_candidates(z, label, parent_logits, candidate_logits, cfg, policy):
    """Return pass flags and worst-step relative lost progress for one example."""
    z = z.detach().double().reshape(1, -1)
    q = (parent_logits.detach().double().reshape(1, -1) / cfg.temperature).softmax(-1)
    qc = (candidate_logits.detach().double() / cfg.temperature).softmax(-1)
    y = torch.full_like(z, cfg.label_smoothing / z.shape[-1])
    y[0, int(label)] += 1 - cfg.label_smoothing
    a, t = cfg.kd_alpha, cfg.temperature
    p1, pt = z.softmax(-1), (z / t).softmax(-1)
    gb = (1 - a) * (p1 - y) + a * t * (pt - q)
    gc = (1 - a) * (p1 - y) + a * t * (pt - qc)

    def objective(values):
        # Teacher entropy is constant for every candidate; omit it for stability.
        return -(1 - a) * (y * values.log_softmax(-1)).sum(-1) - a * t*t * (q * (values/t).log_softmax(-1)).sum(-1)

    origin = objective(z)
    passed = torch.ones(len(qc), dtype=torch.bool, device=z.device)
    risk = torch.full((len(qc),), -torch.inf, dtype=torch.float64, device=z.device)
    for eta in policy.etas:
        reference_gain = origin - objective(z - eta * gb)
        candidate_gain = origin - objective(z - eta * gc)
        valid = torch.isfinite(candidate_gain) & torch.isfinite(reference_gain) & (reference_gain >= policy.min_gain)
        passed &= valid & (candidate_gain + policy.tolerance >= (1-policy.epsilon) * reference_gain)
        relative = (reference_gain - candidate_gain) / reference_gain.clamp_min(policy.min_gain)
        risk = torch.maximum(risk, relative)
    return passed, risk


def removal_proposals(current, removable, k, attention, features, random_out=None):
    """Bounded candidate family: random, low attention, conditional redundancy."""
    attention = attention.detach().cpu()
    features = features.detach().float().cpu()
    proposals = []
    if random_out is not None:
        proposals.append(random_out[:k].tolist())
    ranked = sorted(removable.tolist(), key=lambda i: (float(attention[i]), i))
    proposals.append(ranked[:k])
    f = F.normalize(features.float(), dim=-1)
    similarity = (f @ f.T).cpu()
    left, allowed, removed = set(current.tolist()), set(removable.tolist()), []
    for _ in range(k):
        def key(i):
            neighbors = sorted(left - {i})
            redundancy = float(similarity[i, neighbors].max()) if neighbors else -float("inf")
            return (-redundancy, float(attention[i]), i)
        chosen = min(allowed, key=key)
        removed.append(chosen)
        allowed.remove(chosen)
        left.remove(chosen)
    proposals.append(removed)
    unique = {}
    for ids in proposals:
        unique.setdefault(tuple(sorted(ids)), ids)
    return list(unique.values())


class TeacherViews:
    def __init__(self, teacher, image, cfg, policy, guard=None):
        self.teacher, self.image, self.cfg, self.policy, self.guard = teacher, image, cfg, policy, guard
        self.views = self.tokens = self.calls = 0

    @torch.no_grad()
    def __call__(self, masks):
        if self.views + len(masks) > self.policy.max_views_per_image:
            raise ValueError("Teacher candidate-view budget exceeded")
        result = []
        # Callers supply same-length masks. Batch candidates, not full models.
        for start in range(0, len(masks), self.policy.candidate_batch):
            chunk = masks[start:start+self.policy.candidate_batch]
            if self.guard:
                self.guard.tick()
            indices = torch.stack(chunk)
            with autocast(self.cfg, self.image.device):
                out = self.teacher(self.image.expand(len(chunk), -1, -1, -1), indices)
            result.append(out.float())
            self.views += len(chunk)
            self.tokens += sum(len(m) for m in chunk)
            self.calls += 1
            if self.guard:
                self.guard.pace()
        return torch.cat(result)


@torch.no_grad()
def deletion_targets(teacher, images, labels, logits, attention, features, cfg, policy, method, generator, guard=None, detail_sink=None):
    if method not in METHODS:
        raise ValueError(method)
    if method in ("student", "random_rescue_10", "full"):
        masks, swaps = select_tokens(attention, method, generator=generator)
        if guard:
            guard.tick()
        with autocast(cfg, images.device):
            targets = teacher(images, masks)
        if guard:
            guard.pace()
        count = 196 if masks is None else 98
        return targets.detach(), {"kept": float(count), "swaps": float(swaps.float().mean()),
                "teacher_views": 1.0, "teacher_tokens": float(count), "accepted": 0.0, "fallback": 0.0,
                "budget_stops": 0.0, "audit_risk": 0.0}
    outputs, totals = [], {key: 0.0 for key in ("kept", "swaps", "teacher_views", "teacher_tokens", "accepted", "fallback", "budget_stops", "audit_risk")}
    for row in range(len(images)):
        a = attention[row].detach()
        f = features[row].detach()
        top = a.topk(98).indices
        views = TeacherViews(teacher, images[row:row+1], cfg, policy, guard)
        selected, chosen_logits, risk_value = top, None, 0.0
        accepted = fallback = budget_stop = swaps = 0
        if method == "rescue_audit":
            # Match existing Random10: outgoing RNG draw before incoming RNG draw.
            out = top[torch.randperm(98, device=a.device, generator=generator)[:10]]
            present = torch.zeros(196, dtype=torch.bool, device=a.device)
            present[top] = True
            outside = torch.where(~present)[0]
            incoming = outside[torch.randperm(98, device=a.device, generator=generator)[:10]]
            for k in policy.swap_budgets:
                parent = torch.cat((top, incoming[:k]))
                parent_logits = views([parent])[0]
                proposals = removal_proposals(parent, top, k, a, f, out)
                children = []
                for deleted in proposals:
                    # Preserve baseline's replacement-slot order for its random proposal.
                    child = top.clone()
                    for slot, token in enumerate(deleted):
                        child[(top == token).nonzero().item()] = incoming[slot]
                    children.append(child)
                targets = views(children)
                passed, risks = audit_candidates(logits[row], labels[row], parent_logits, targets, cfg, policy)
                valid = [j for j in range(len(children)) if bool(passed[j])]
                if valid:
                    j = min(valid, key=lambda n: (float(risks[n]), tuple(sorted(children[n].tolist()))))
                    selected, chosen_logits, risk_value = children[j], targets[j], float(risks[j])
                    accepted, swaps = 1, k
                    break
            if chosen_logits is None:
                chosen_logits = views([top])[0]
                fallback = 1  # Rejected rescue, not an audited compression of 108.
        else:
            original = torch.arange(196, device=a.device)
            parent_logits = views([original])[0]
            selected, chosen_logits = original, parent_logits
            while len(selected) > 98:
                changed = False
                for k in dict.fromkeys(min(n, len(selected)-98) for n in (10, 5, 1)):
                    proposals = removal_proposals(selected, selected, k, a, f)
                    if views.views + len(proposals) > policy.max_views_per_image:
                        budget_stop = 1
                        break
                    children = [selected[~torch.isin(selected, torch.tensor(d, device=a.device))] for d in proposals]
                    targets = views(children)
                    passed, risks = audit_candidates(logits[row], labels[row], parent_logits, targets, cfg, policy)
                    valid = [j for j in range(len(children)) if bool(passed[j])]
                    if valid:
                        j = min(valid, key=lambda n: (float(risks[n]), tuple(children[n].tolist())))
                        selected, chosen_logits, risk_value = children[j], targets[j], float(risks[j])
                        accepted, changed = 1, True
                        break
                if budget_stop or not changed:
                    break
        outputs.append(chosen_logits)
        values = dict(kept=len(selected), swaps=swaps, teacher_views=views.views, teacher_tokens=views.tokens,
                      accepted=accepted, fallback=fallback, budget_stops=budget_stop, audit_risk=risk_value)
        if detail_sink is not None:
            detail_sink({'batch_row': row, **values})
        for key, value in values.items():
            totals[key] += value
    return torch.stack(outputs), {key: value/len(images) for key, value in totals.items()}
