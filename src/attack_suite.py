"""WP2 attack package (deliverable D2): prediction-targeted and confidence-targeted attacks.

The distinction is the scientific spine of the whole project. A moderation pipeline can fail
in two independent ways, and they need different tests:

  PREDICTION-TARGETED   flip the label. Accuracy collapses; AURC also collapses, but as a
                        SIDE EFFECT of there being nothing left to rank. FGSM, PGD, AutoAttack.
  CONFIDENCE-TARGETED   leave the label alone, move the confidence. Accuracy is bit-identical;
                        AURC collapses because the ORDERING is destroyed. ACE, and the
                        label-free over-confidence attack of Ledda et al. 2025.

Reporting one without the other is what makes the central claim unfalsifiable, which is why
every table in this project prints accuracy beside AURC.

DIRECTION TAXONOMY (the axis WP3/WP4 pivot on):
  under_confidence   push confidence DOWN on correct predictions  -> the model abstains on
                     work it could have done. Cost: review rate explodes.
  over_confidence    push confidence UP on wrong predictions       -> the model auto-decides
                     work it gets wrong. Cost: residual risk explodes. This is the dangerous
                     one, and the one adversarial training is least likely to fix, because
                     the entropy-minimizing region does not lie on the decision boundary.
  both               ACE does both at once (down where correct, up where wrong), which is
                     why it is the strongest single attack on a risk-coverage curve and also
                     why it needs ground-truth labels -- its practical weakness.

Every attack here operates on [0,1] pixels against a models.Detector whose forward()
normalizes internally, so an epsilon is always an epsilon in pixel space.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# ------------------------------------------------------------- prediction-targeted

def fgsm(model, x, y, eps: float, **_):
    """Goodfellow et al. 2015: one signed-gradient step of size eps. The cheapest possible
    baseline; if a defense does not beat FGSM it does not beat anything."""
    x = x.clone().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    g, = torch.autograd.grad(loss, x)
    return (x + eps * g.sign()).clamp(0, 1).detach()


def pgd_linf(model, x, y, eps: float, alpha: float = None, steps: int = 10,
             random_start: bool = True, **_):
    """Madry et al. 2018, Linf. Random start matters: without it PGD is a multi-step FGSM
    that can stall in a flat region and overstate robustness."""
    alpha = alpha if alpha is not None else max(eps / 4, 1 / 255)
    delta = (torch.empty_like(x).uniform_(-eps, eps) if random_start else torch.zeros_like(x))
    delta = ((x + delta).clamp(0, 1) - x).detach()
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(model(x + delta), y)
        g, = torch.autograd.grad(loss, delta)
        delta = (delta.detach() + alpha * g.sign()).clamp(-eps, eps)
        delta = (x + delta).clamp(0, 1) - x
    return (x + delta).detach()


def pgd_l2(model, x, y, eps: float, alpha: float = None, steps: int = 10, **_):
    """L2 PGD. Reported separately from Linf because a detector can be robust in one norm
    and not the other, and quoting a single 'robustness' number hides that."""
    alpha = alpha if alpha is not None else eps / 4
    b = x.shape[0]
    delta = torch.randn_like(x)
    n = delta.view(b, -1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
    delta = (delta / n * eps * torch.rand(b, 1, 1, 1, device=x.device))
    delta = ((x + delta).clamp(0, 1) - x).detach()
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(model(x + delta), y)
        g, = torch.autograd.grad(loss, delta)
        gn = g.view(b, -1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
        delta = delta.detach() + alpha * g / gn
        dn = delta.view(b, -1).norm(dim=1).view(-1, 1, 1, 1)
        delta = delta * torch.clamp(eps / dn.clamp_min(1e-12), max=1.0)
        delta = (x + delta).clamp(0, 1) - x
    return (x + delta).detach()


def autoattack_safe(model, x, y, eps: float, n_queries: int = 500, **_):
    """AutoAttack, composed so it can actually run on a 2-logit model.

    MEASURED on this box with the official fra31 package: 'apgd-t' raises
    IndexError: index -3 is out of bounds for dimension 1 with size 2, because targeted DLR
    reads x_sorted[:,-3] and [:,-4] and needs >= 4 logits. checks.py warns for n_cls<=2 and
    runs the attack anyway. version='standard' only crashes once points SURVIVE apgd-ce, so
    it appears to work against a weak detector and dies against a robust one -- i.e. it will
    break precisely when WP3 succeeds. Setting n_classes=3 on a 2-logit model does not help.

    The safe composition is apgd-ce + fab-t (with n_target_classes=1) + square, verified to
    run here. Square is query-bounded because it is by far the slowest component (~2.8 img/s
    at default settings).
    """
    from autoattack import AutoAttack

    aa = AutoAttack(model, norm="Linf", eps=eps, version="custom",
                    attacks_to_run=["apgd-ce", "fab-t", "square"], verbose=False)
    aa.fab.n_target_classes = 1          # REQUIRED: default reaches for logits that do not exist
    aa.square.n_queries = n_queries
    with torch.enable_grad():
        return aa.run_standard_evaluation(x, y, bs=x.shape[0])


# ------------------------------------------------------------ confidence-targeted

def ace(model, x, y, eps: float, iters: int = 15, **_):
    """Galil & El-Yaniv Algorithm 1. Direction computed ONCE, outside the loop; epsilon
    halved per-sample until the argmax is preserved. Returns x_adv only (the runner
    re-derives logits from the accept pass via ace_full)."""
    return ace_full(model, x, y, eps, iters)[0]


def ace_uint8(model, x, y, eps: float, iters: int = 15, **_):
    """ACE constrained to the 8-bit pixel lattice -- the realisable threat model."""
    return ace_full(model, x, y, eps, iters, quantize=True)[0]


def ace_full(model, x, y, eps: float, iters: int = 15, quantize: bool = False):
    """As ace(), returning (x_adv, effective_eps, clean_pred, accepted_logits).

    QUANTIZE decides the threat model, and it is not a detail. Measured effective epsilons for
    unquantised ACE on this detector are 0.102 / 0.261 / 0.447 grey levels at eps =
    0.0005 / 0.002 / 0.005 -- the mean is BELOW one 8-bit quantisation step (0.5/255 = 0.00196).
    The decode cache is uint8 and ToTensor puts x exactly on the k/255 grid, so rounding a
    sub-step perturbation back to uint8 erases it outright. Unquantised ACE therefore describes
    an attacker with post-decode TENSOR access; quantise=True describes one who can only upload
    a FILE, which is the moderation threat model that matters. Report which one a row is.

    This is the reason integer-constrained attacks exist in the forensics literature (Tondi,
    Electronics Letters 54(21), 2018): small perturbations are cancelled by rounding to pixels.

    The accepted logits come FROM THE ACCEPT-CHECK FORWARD and must be the reported ones:
    re-running the model on x_adv uses a different batch shape, and EfficientNet under cuDNN
    is not bit-identical across batch shapes, so a boundary sample flips and label
    preservation reads 0.998 instead of 1.0 for reasons unrelated to the attack.
    """
    with torch.enable_grad():
        xg = x.clone().requires_grad_(True)
        logits = model(xg)
        pred = logits.argmax(1)
        kappa = logits.softmax(1).gather(1, pred[:, None]).squeeze(1)
        grad, = torch.autograd.grad(kappa.sum(), xg)
    eta = grad.sign()
    direction = torch.where(pred == y, -1.0, 1.0).view(-1, 1, 1, 1)

    out, out_logits = x.clone(), logits.detach().clone()
    eps_i = torch.full((x.shape[0],), float(eps), device=x.device)
    pending = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    for _ in range(iters):
        idx = pending.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            break
        cand = (x[idx] + direction[idx] * eps_i[idx].view(-1, 1, 1, 1) * eta[idx]).clamp(0, 1)
        if quantize:
            cand = torch.round(cand * 255.0) / 255.0     # snap to what a saved file can hold
        with torch.no_grad():
            cl = model(cand)
        ok = cl.argmax(1) == pred[idx]
        out[idx[ok]] = cand[ok]
        out_logits[idx[ok]] = cl[ok]
        pending[idx[ok]] = False
        eps_i[idx[~ok]] *= 0.5
    return out, (out - x).abs().amax(dim=(1, 2, 3)), pred, out_logits


def overconfidence(model, x, y=None, eps: float = 4 / 255, alpha: float = None,
                   steps: int = 20, **_):
    """Label-free OVER-confidence attack (Ledda et al. 2025).

    Minimize H(f(x+delta), onehot(yhat)) with yhat FROZEN at the clean prediction. Two
    properties matter and both come from that freezing:
      - LABEL-FREE: no ground truth is used anywhere, so it is deployable by an attacker who
        cannot see labels. This is exactly ACE's practical weakness.
      - LABEL-PRESERVING BY CONSTRUCTION: the objective pushes probability mass TOWARD the
        already-predicted class, so the argmax can only be reinforced. Accuracy is
        unchanged not by an accept test but by the shape of the loss.

    Effect: the model becomes confidently wrong wherever it was quietly wrong. AUROC(failure)
    degrades, residual risk under an abstention policy explodes, and accuracy does not move --
    so an accuracy-based monitor sees a perfectly healthy system.

    Adversarial training is not expected to fix this: the entropy-minimizing region lies in
    the interior of the decision cell, not on the boundary that AT hardens. That is the WP3
    hypothesis this attack exists to test.
    """
    alpha = alpha if alpha is not None else max(eps / 8, 0.5 / 255)
    with torch.no_grad():
        yhat = model(x).argmax(1)
    delta = torch.zeros_like(x)
    for _ in range(steps):
        delta.requires_grad_(True)
        logp = F.log_softmax(model(x + delta), dim=1)
        loss = F.nll_loss(logp, yhat)          # == cross-entropy to the frozen prediction
        g, = torch.autograd.grad(loss, delta)
        delta = (delta.detach() - alpha * g.sign()).clamp(-eps, eps)   # DESCEND: minimize H
        delta = (x + delta).clamp(0, 1) - x
    return (x + delta).detach()


def underconfidence(model, x, y=None, eps: float = 4 / 255, alpha: float = None,
                    steps: int = 20, **_):
    """Label-free UNDER-confidence attack: the mirror of overconfidence().

    Maximize H(f(x+delta), onehot(yhat)) -- push probability mass AWAY from the predicted
    class without crossing the boundary. Unlike the over-confidence direction this is NOT
    label-preserving by construction (enough entropy eventually flips the argmax), so the
    runner must measure label preservation and report it rather than assume it.

    Cost model: this one inflates the REVIEW RATE (the system abstains on work it could do)
    where over-confidence inflates RESIDUAL RISK (the system auto-decides work it gets
    wrong). WP4 prices both, and they are not symmetric: one wastes moderator hours, the
    other ships wrong decisions.
    """
    alpha = alpha if alpha is not None else max(eps / 8, 0.5 / 255)
    with torch.no_grad():
        yhat = model(x).argmax(1)
    delta = torch.zeros_like(x)
    for _ in range(steps):
        delta.requires_grad_(True)
        logp = F.log_softmax(model(x + delta), dim=1)
        loss = F.nll_loss(logp, yhat)
        g, = torch.autograd.grad(loss, delta)
        delta = (delta.detach() + alpha * g.sign()).clamp(-eps, eps)   # ASCEND: maximize H
        delta = (x + delta).clamp(0, 1) - x
    return (x + delta).detach()


# ------------------------------------------------------------------------ registry

# family: what it targets. direction: which way confidence is pushed. uses_labels: whether
# ground truth is required (an attacker in the wild usually has none).
ATTACKS = {
    "fgsm":       dict(fn=fgsm,             family="prediction", direction="n/a",
                       uses_labels=True,  defaults=dict(eps=2 / 255)),
    "pgd_linf":   dict(fn=pgd_linf,         family="prediction", direction="n/a",
                       uses_labels=True,  defaults=dict(eps=2 / 255, steps=10)),
    "pgd_l2":     dict(fn=pgd_l2,           family="prediction", direction="n/a",
                       uses_labels=True,  defaults=dict(eps=0.5, steps=10)),
    "autoattack": dict(fn=autoattack_safe,  family="prediction", direction="n/a",
                       uses_labels=True,  defaults=dict(eps=2 / 255)),
    "ace":        dict(fn=ace,              family="confidence", direction="both",
                       uses_labels=True,  defaults=dict(eps=0.005)),
    # realisable variant: every perturbed pixel is an integer number of grey levels
    "ace_uint8":  dict(fn=ace_uint8,        family="confidence", direction="both",
                       uses_labels=True,  defaults=dict(eps=0.005)),
    "overconf":   dict(fn=overconfidence,   family="confidence", direction="over_confidence",
                       uses_labels=False, defaults=dict(eps=4 / 255, steps=20)),
    "underconf":  dict(fn=underconfidence,  family="confidence", direction="under_confidence",
                       uses_labels=False, defaults=dict(eps=4 / 255, steps=20)),
}

# What SHOULD move, per family. A wrong implementation is caught by these signs, not by a
# reviewer noticing later. Checked by tests/test_attack_signs.py.
EXPECTED_SIGNS = {
    "prediction": {"accuracy": "down", "aurc": "up", "auroc_failure": "any"},
    "confidence": {"accuracy": "unchanged", "aurc": "up", "auroc_failure": "down",
                   "ece": "up"},
}


def _eps_str(eps: float) -> str:
    """Linf budgets are written k/255 in this literature, and that form is EXACT.

    Formatting 8/255 as a decimal ("eps0.0156863") loses precision, so parsing the filename
    back gives a different float than the one that was run -- and the condition string is the
    only join key between an attack run and its row in a table. k_255 round-trips exactly and
    is what a reviewer expects to read.
    """
    q = eps * 255
    if abs(q - round(q)) < 1e-9 and 1 <= round(q) <= 255:
        return f"{round(q)}_255"
    return f"{eps:g}"


def _eps_parse(s: str) -> float:
    return int(s[:-4]) / 255 if s.endswith("_255") else float(s)


def condition_name(attack: str, **kw) -> str:
    """Stable condition string: attack + its distinguishing hyperparameters.

    Encoded so score.py can pivot a whole matrix without a lookup table, and so two runs
    that differ in any attack parameter can never land in the same file.
    """
    if attack == "clean":
        return "clean"
    spec = ATTACKS[attack]
    p = {**spec["defaults"], **{k: v for k, v in kw.items() if v is not None}}
    bits = [attack, f"eps{_eps_str(p['eps'])}"]
    if "steps" in p:
        bits.append(f"s{p['steps']}")
    return "_".join(bits)


# Realistic (non-adversarial) conditions are an EXPLICIT list. Without one, any typo'd
# attack name -- "pgd_linfx_eps2_255_s10" -- fell through to family="realistic" with a
# fabricated eps=0.0, i.e. it landed in the wrong pivot group carrying a wrong epsilon, and
# the table looked fine. Unknown heads now raise.
REALISTIC_PREFIXES = ("jpeg_q", "downscale_", "webp", "squarecrop")


def parse_condition(condition: str, strict: bool = True) -> dict:
    """Inverse of condition_name, for grouping a table by family/direction."""
    if condition == "clean":
        return dict(attack="clean", family="clean", direction="n/a", eps=0.0)
    head = condition.split("_eps")[0]
    if head not in ATTACKS:
        if condition.startswith(REALISTIC_PREFIXES):
            return dict(attack=condition, family="realistic", direction="n/a", eps=0.0)
        if strict:
            raise ValueError(
                f"unknown condition {condition!r}: not an attack in ATTACKS and not one of "
                f"{REALISTIC_PREFIXES}. Refusing to guess -- a mis-parsed condition silently "
                f"joins the wrong pivot group with eps=0.")
        return dict(attack=condition, family="unknown", direction="n/a", eps=float("nan"))
    rest = condition.split("_eps", 1)[1]
    eps_s, _, steps_s = rest.partition("_s")
    out = dict(attack=head, family=ATTACKS[head]["family"],
               direction=ATTACKS[head]["direction"], eps=_eps_parse(eps_s),
               uses_labels=ATTACKS[head]["uses_labels"])
    if steps_s:
        out["steps"] = int(steps_s)
    return out
