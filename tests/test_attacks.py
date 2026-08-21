"""Attack correctness tests. CPU-only, seconds, no data and no pretrained weights.

Each test pins a property that, if broken, produces a plausible-looking wrong number rather
than an error. That is the whole risk profile of adversarial-robustness code: a weak attack
reads as a robust model, and a mis-scoped epsilon reads as a strong one.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.attack_suite import (ATTACKS, ace_full, condition_name, fgsm, overconfidence,
                              parse_condition, pgd_l2, pgd_linf, underconfidence)


class TinyDetector(nn.Module):
    """Normalization INSIDE forward, exactly like models.NormalizedModel, so the tests
    exercise the same [0,1] pixel-space contract the real detectors use."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.register_buffer("mean", torch.full((1, 3, 1, 1), 0.5))
        self.register_buffer("std", torch.full((1, 3, 1, 1), 0.25))
        self.net = nn.Sequential(nn.Conv2d(3, 4, 5, stride=4), nn.ReLU(),
                                 nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(16, 2))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


@pytest.fixture
def setup():
    torch.manual_seed(0)
    m = TinyDetector().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    x = torch.rand(24, 3, 32, 32)
    y = torch.randint(0, 2, (24,))
    return m, x, y


def _conf(m, x):
    with torch.no_grad():
        return m(x).softmax(1).max(1).values


def test_all_attacks_stay_in_pixel_range(setup):
    """Every attack must return valid images. An attack that leaves [0,1] is attacking a
    point the deployed system can never receive, and its epsilon means nothing."""
    m, x, y = setup
    for name, spec in ATTACKS.items():
        if name == "autoattack":
            continue          # the fra31 package moves tensors to CUDA internally; covered
                              # by the measured binary-gotcha probe instead, not by a CPU test
        out = ace_full(m, x, y, 0.01)[0] if name == "ace" else spec["fn"](m, x, y, **spec["defaults"])
        assert out.min() >= 0.0 and out.max() <= 1.0, f"{name} left [0,1]"
        assert out.shape == x.shape


def test_epsilon_balls_are_respected(setup):
    """Linf and L2 budgets must actually bind. A perturbation larger than the reported
    epsilon makes every robustness number in the report an overstatement."""
    m, x, y = setup
    eps = 4 / 255
    adv = pgd_linf(m, x, y, eps=eps, steps=5)
    assert (adv - x).abs().amax() <= eps + 1e-6

    adv1 = fgsm(m, x, y, eps=eps)
    assert (adv1 - x).abs().amax() <= eps + 1e-6

    adv2 = pgd_l2(m, x, y, eps=0.5, steps=5)
    l2 = (adv2 - x).flatten(1).norm(dim=1)
    assert l2.max() <= 0.5 + 1e-4


def test_prediction_attacks_reduce_accuracy(setup):
    """A prediction-targeted attack that does not move accuracy is silently broken -- most
    often because the gradient was taken through a detached tensor."""
    m, x, y = setup
    base = (m(x).argmax(1) == y).float().mean()
    for fn in (fgsm, pgd_linf):
        adv = fn(m, x, y, eps=16 / 255, steps=10)
        with torch.no_grad():
            acc = (m(adv).argmax(1) == y).float().mean()
        assert acc <= base, f"{fn.__name__} did not reduce accuracy ({acc:.3f} vs {base:.3f})"


def test_overconfidence_preserves_labels_and_raises_confidence(setup):
    """Ledda et al.: minimizing H against the FROZEN clean prediction can only reinforce the
    argmax, so label preservation is exact BY CONSTRUCTION -- not by an accept test. If a
    label moves, the sign of the update is wrong (it is ascending instead of descending)."""
    m, x, y = setup
    with torch.no_grad():
        pred0 = m(x).argmax(1)
    adv = overconfidence(m, x, eps=8 / 255, steps=20)
    with torch.no_grad():
        pred1 = m(adv).argmax(1)
    assert (pred1 == pred0).all(), "over-confidence attack changed a prediction"
    assert _conf(m, adv).mean() > _conf(m, x).mean()


def test_underconfidence_lowers_belief_in_the_clean_prediction(setup):
    """The mirror direction, measured on the right quantity.

    Max-softmax is the WRONG probe here: pushing p(yhat) below 0.5 flips the argmax, after
    which max-softmax climbs back up (measured: 0.5403 -> 0.5529 on this fixture). The
    quantity the attack actually controls is p(clean-predicted class), and that must fall.

    The objective is KL to the UNIFORM distribution (Ledda et al. Eq. 11), which is an
    ATTRACTOR at p = 1/K, not an unbounded descent. So the right assertions are that belief
    moves TOWARD 1/K, not that it decreases monotonically per sample: a point that starts at
    0.5112 can land at 0.5114 on the far side of the attractor, and that is correct behaviour.
    Per-sample monotonicity was a property of the earlier cross-entropy-ascent implementation,
    which had no stopping point and turned "under-confidence" into "label flip"."""
    m, x, y = setup
    K = 2
    with torch.no_grad():
        p0 = m(x).softmax(1)
        pred0 = p0.argmax(1)
    adv = underconfidence(m, x, eps=8 / 255, steps=20)
    with torch.no_grad():
        p1 = m(adv).softmax(1)
    belief0 = p0.gather(1, pred0[:, None]).squeeze(1)
    belief1 = p1.gather(1, pred0[:, None]).squeeze(1)
    assert belief1.mean() < belief0.mean(), "under-confidence did not reduce p(clean pred)"
    # EVERY sample is pulled into a tight band around the uniform point. A per-sample
    # monotone bound would be wrong: a signed step of size alpha cannot land exactly on the
    # attractor, so a point that starts within that jitter of 1/K (distance ~0.011 here)
    # cannot get closer. The band, and the shrinking mean distance, are the real properties.
    d0, d1 = (belief0 - 1.0 / K).abs(), (belief1 - 1.0 / K).abs()
    assert d1.max() < 0.02, f"a sample escaped the uniform band: {d1.max():.4f}"
    assert d1.mean() < 0.5 * d0.mean(), "distance to uniform did not at least halve"
    # and confidence collapses toward chance rather than overshooting into the other class
    assert p1.max(1).values.mean() < 0.55


def test_confidence_attacks_are_label_free(setup):
    """They must not read y at all: an attacker in the wild has no ground truth, and an
    attack that secretly needs it overstates what a real adversary can do."""
    m, x, y = setup
    a = overconfidence(m, x, y=y, eps=8 / 255, steps=10)
    b = overconfidence(m, x, y=None, eps=8 / 255, steps=10)
    assert torch.equal(a, b)
    assert ATTACKS["overconf"]["uses_labels"] is False
    assert ATTACKS["ace"]["uses_labels"] is True      # ACE does need labels; stated, not hidden


def test_ace_preserves_every_label(setup):
    """ACE accepts a step only when the argmax survives, and returns unperturbed samples
    otherwise, so preservation is exactly 1.0. Anything less means the reported logits came
    from a different forward pass than the accept check."""
    m, x, y = setup
    _, eff, pred, logits = ace_full(m, x, y, eps=0.01)
    assert (logits.argmax(1) == pred).all()
    assert eff.max() <= 0.01 + 1e-6


def test_condition_names_round_trip():
    """The condition string is the only join key between an attack run and its table row."""
    for name, spec in ATTACKS.items():
        c = condition_name(name)
        p = parse_condition(c)
        assert p["attack"] == name
        assert p["family"] == spec["family"]
        assert p["direction"] == spec["direction"]
        assert p["eps"] == spec["defaults"]["eps"], f"{name}: eps did not round-trip exactly"
    assert parse_condition("clean")["family"] == "clean"
    assert parse_condition("jpeg_q50")["family"] == "realistic"
    assert parse_condition("downscale_0.5")["family"] == "realistic"
    # k/255 form is exact and readable; the pre-existing ace artifacts stay parseable
    assert condition_name("pgd_linf") == "pgd_linf_eps2_255_s10"
    assert parse_condition("ace_eps0.005")["eps"] == 0.005


# ---------------------------------------------- 10. the two hardening properties

def test_quantised_ace_lands_only_on_the_8bit_lattice():
    """The realisable threat model. Unquantised ACE produces mean perturbations of 0.10-0.45
    grey levels -- below the 0.5/255 rounding threshold -- so it describes an attacker with
    post-decode tensor access, not one who uploads a file. The quantised variant must move
    every pixel by a whole number of grey levels or not at all."""
    torch.manual_seed(0)
    m = TinyDetector().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    x = torch.round(torch.rand(16, 3, 32, 32) * 255) / 255      # start ON the lattice
    y = torch.randint(0, 2, (16,))

    adv, eff, pred, logits = ace_full(m, x, y, eps=4 / 255, quantize=True)
    delta = (adv - x) * 255
    assert torch.allclose(delta, delta.round(), atol=1e-4), "quantised ACE left the 8-bit lattice"
    assert (logits.argmax(1) == pred).all()

    # and sub-step epsilon must be a no-op, not a small effect
    adv0, _, _, _ = ace_full(m, x, y, eps=0.0005, quantize=True)
    assert torch.equal(adv0, x), "sub-quantisation epsilon perturbed a uint8 image"


def test_squarecrop_removes_the_geometry_artifact():
    """SID-Set: label 1 and label 2 are 100% square, only ~5% of reals are, so
    `width==height -> fake` scores 0.9785 on our test split -- above the CLIP probe. The
    geometry control must make that rule constant, i.e. carry no label information at all."""
    from src.features import apply_condition
    from PIL import Image

    for w, h in ((1024, 1024), (1024, 681), (640, 960)):
        out = apply_condition(Image.new("RGB", (w, h)), "squarecrop")
        assert out.size[0] == out.size[1] == min(w, h), f"{w}x{h} did not become square"
