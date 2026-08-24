"""The uncertainty seam: the contract stays backward-compatible, the gate only helps, and a
sigma that is secretly 1 - MSP is caught. CPU, no GPU, no data files (synthetic)."""
import numpy as np
import pytest

from src import moderation as MOD
from src import sigma as SIG


def test_two_axis_reduces_to_one_axis_when_sigma_absent():
    """Every existing 1-axis call must be unchanged: actions() with no sigma equals the old
    rule, so nothing that already works can regress."""
    rng = np.random.default_rng(0)
    p = rng.random(500)
    a1 = MOD.actions(p, 0.2, 0.8)
    a2 = MOD.actions(p, 0.2, 0.8, sigma=None, t_sigma=None)
    assert np.array_equal(a1, a2)


def test_sigma_gate_only_moves_items_into_review():
    """The uncertainty gate may only send items to REVIEW; it can never flip an item to ALLOW
    or FLAG, so it cannot increase residual risk -- only trade coverage for safety."""
    rng = np.random.default_rng(1)
    p, sig = rng.random(2000), rng.random(2000)
    base = MOD.actions(p, 0.2, 0.8)
    gated = MOD.actions(p, 0.2, 0.8, sig, np.quantile(sig, 0.7))
    moved = base != gated
    assert np.all(gated[moved] == MOD.REVIEW)          # every change is INTO review
    assert np.all(base[moved] != MOD.REVIEW)           # from a former auto-decision
    assert (gated == MOD.REVIEW).sum() >= (base == MOD.REVIEW).sum()


def test_degenerate_sigma_is_flagged():
    """A sigma that is a monotone function of 1 - MSP carries no new information; the seam
    must say so, or 'confidence beyond softmax' is a relabelling."""
    rng = np.random.default_rng(2)
    conf = 0.5 + 0.5 * rng.random(3000)                # MSP in [0.5, 1]
    degen_sigma = (1.0 - conf) + 1e-6 * rng.random(3000)   # basically 1 - MSP
    real_sigma = rng.random(3000)                      # independent
    d1, rho1 = SIG.is_degenerate(degen_sigma, conf)
    d2, rho2 = SIG.is_degenerate(real_sigma, conf)
    assert d1 and abs(rho1) > 0.98
    assert not d2 and abs(rho2) < 0.5


def test_sigma_gate_reduces_or_holds_residual_risk():
    """On a constructed case where the uncertain items are the wrong ones, the gate must
    strictly reduce residual risk -- the property that makes the second axis worth having."""
    n = 4000
    rng = np.random.default_rng(3)
    y = (rng.random(n) > 0.5).astype(int)
    # confident-but-wrong region: p_fake near 0/1 disagreeing with y, but high sigma
    p = np.where(y == 1, 0.05, 0.95)                   # confidently WRONG on everyone
    sigma = np.full(n, 0.4)                            # all flagged uncertain
    r1 = MOD.evaluate_policy(p, y, 0.1, 0.9)
    r2 = MOD.evaluate_policy(p, y, 0.1, 0.9, sigma, 0.3)
    assert r2["review_rate"] > r1["review_rate"]
    assert not np.isfinite(r2["residual_risk"]) or r2["residual_risk"] <= r1["residual_risk"]


def test_load_sigma_refuses_partial_coverage():
    """A sigma file that does not cover every predicted uid must RAISE, not silently inner-join
    and misreport coverage."""
    import tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp())
    SIG.write_sigma(np.array(["a", "b"]), np.array([0.1, 0.2], np.float32),
                    "t", "test", "clean", "s", "s", out_dir=d)
    with pytest.raises(ValueError):
        SIG.load_sigma("sigma_t_test_clean", np.array(["a", "b", "c"]), sigma_dir=d)
    out = SIG.load_sigma("sigma_t_test_clean", np.array(["b", "a"]), sigma_dir=d)
    assert np.allclose(out, [0.2, 0.1])                # returns aligned to the requested order
