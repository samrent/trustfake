"""Eight tests for the WP1 metrics module. No GPU, no data, no network.

Each test is a defence against a specific way a selective-classification number lies.
Seeds and generators are the ones from notes/prior_verify_metrics.py, so the constants
quoted in BUILD.md are locked here by assert instead of remembered.
"""

import numpy as np
import pytest

import metrics as M


# ------------------------------------------------------------------ 1. perfect ranker

def test_perfect_ranker_has_zero_excess_aurc():
    """A ranker that puts every correct prediction above every mistake IS the empirical
    oracle, so E-AURC must be exactly 0 -- not 'small'. If this fails, the oracle
    reference is wrong and every E-AURC in the report is offset by a constant."""
    rng = np.random.default_rng(0)
    correct = (rng.random(1000) > 0.2).astype(float)
    conf = correct + rng.random(1000) * 1e-6      # correct-first, no cross-block ties

    assert M.eaurc(conf, correct) == pytest.approx(0.0, abs=1e-15)

    # The closed form r + (1-r)ln(1-r) is asymptotic: at n=1000 it is off by ~1e-4, the
    # same order as the effects this harness reports. This is why we never use it.
    r = 1.0 - correct.mean()
    closed = r + (1.0 - r) * np.log(1.0 - r)
    empirical = M.aurc_optimal_empirical(correct)
    assert abs(empirical - closed) > 1e-5
    assert abs(empirical - closed) < 1e-3


# ------------------------------------------------------------------- 2. random ranker

def test_random_ranker_anchors():
    """An uninformative confidence gives AURC -> r and AUGRC -> r/2, where r is the base
    error rate. These are the anchors every reported number is read against: an AURC near
    the base error rate means the confidence carries no ranking information at all.

    Asserted against the theoretical anchor, NOT against a recorded sample. For a random
    ranker E[R_sel(top-k)] = r at every k, so E[AURC] = r exactly; the Monte-Carlo standard
    error over 200 seeds at n=5000 is ~6e-4. Measured on this box: 0.200021 at r=0.2 and
    0.350112 at r=0.35 (AUGRC 0.100049 / 0.175026). BUILD.md records 0.3489 for the second
    case -- 1.7 SE below the anchor, i.e. a different draw, not a different quantity. A
    test that pinned it would have locked sampling noise into the suite.
    """
    for r in (0.2, 0.35):
        a, g = [], []
        for s in range(200):
            rg = np.random.default_rng(s)
            correct = (rg.random(5000) > r).astype(float)
            conf = rg.random(5000)
            a.append(M.aurc(conf, correct))
            g.append(M.augrc(conf, correct))
        assert np.mean(a) == pytest.approx(r, abs=2.5e-3)          # ~4 standard errors
        assert np.mean(g) == pytest.approx(r / 2, abs=2.5e-3)


# --------------------------------------------------------------------- 3. all correct

def test_all_correct_is_zero_risk_everywhere():
    """No errors means no risk at any coverage. Guards the empty-tail edge case where the
    cumulative sum is all zeros and a naive implementation divides 0/0."""
    rng = np.random.default_rng(1)
    correct = np.ones(500)
    conf = rng.random(500)
    assert M.aurc(conf, correct) == 0.0
    assert M.augrc(conf, correct) == 0.0
    assert M.eaurc(conf, correct) == 0.0
    assert np.isnan(M.auroc_failure(conf, correct))   # undefined, must not be 0.5 or 1.0


# ----------------------------------------------------------------- 4. reversed ranker

def test_reversed_ranker_is_worse_than_random():
    """Negating a good confidence must produce an AURC worse than chance. This is the
    regime a confidence attack drives the model into -- the abstention rule then rejects
    preferentially the predictions that were right."""
    rng = np.random.default_rng(2)
    correct = (rng.random(2000) > 0.2).astype(float)
    score = correct + rng.random(2000) * 0.01
    good = M.aurc(score, correct)
    reversed_ = M.aurc(-score, correct)
    random_ = M.aurc(rng.random(2000), correct)
    assert good < random_ < reversed_
    assert M.auroc_failure(-score, correct) < 0.5 < M.auroc_failure(score, correct)


# -------------------------------------------------------------- 5. O(n) == O(n^2)

def _brute_augrc(conf, correct):
    """The definition, written the slow obvious way: mean over k of (loss of top-k)/n."""
    conf = np.asarray(conf, float)
    loss = 1.0 - np.asarray(correct, float)
    n = len(conf)
    order = np.argsort(-conf, kind="stable")
    loss_s = loss[order]
    return float(np.mean([loss_s[:k].sum() / n for k in range(1, n + 1)]))


def test_augrc_linear_matches_quadratic_bruteforce():
    """The cumsum implementation must equal the definition. With distinct confidences the
    block-weighted average also equals the textbook mean over k=1..n, which is what keeps
    our AURC comparable to published ones."""
    rng = np.random.default_rng(3)
    correct = (rng.random(400) > 0.3).astype(float)
    conf = rng.random(400)                      # continuous: no ties
    assert M.augrc(conf, correct) == pytest.approx(_brute_augrc(conf, correct), abs=1e-12)
    # and with no ties the two weightings coincide
    assert M.augrc(conf, correct, weights="uniform") == pytest.approx(
        M.augrc(conf, correct, weights="block"), abs=1e-12)


# ------------------------------------------------------------------- 6. risk identity

def test_generalized_equals_coverage_times_selective():
    """R_gen == cov * R_sel elementwise. Cheap, and it catches the off-by-one in the
    operating-point index that otherwise shows up as a plausible-looking wrong curve."""
    rng = np.random.default_rng(4)
    correct = (rng.random(400) > 0.3).astype(float)
    conf = rng.random(400)
    cov, sel, gen, thr = M.rc_curve(conf, correct)
    assert np.abs(gen - cov * sel).max() < 1e-15
    assert cov[-1] == 1.0                        # full coverage is always an operating point
    assert sel[-1] == pytest.approx(1.0 - correct.mean())
    assert np.all(np.diff(thr) < 0)              # strictly decreasing: ties collapsed


# ---------------------------------------------------------- 7. permutation invariance

def test_permutation_invariance_under_heavy_ties():
    """THE tie test. With only 10 distinct confidence levels, naive cumulative AURC varies
    by ~5.6e-3 across permutations of the same predictions -- larger than the effects this
    harness reports. fp16 softmax saturating to exactly 1.0 manufactures those blocks
    silently. Collapsing tie blocks makes the curve a property of the model."""
    rng = np.random.default_rng(5)
    correct = (rng.random(600) > 0.25).astype(float)
    conf = np.round(rng.random(600), 1)          # 10 distinct levels

    vals, naive = [], []
    for s in range(8):
        p = np.random.default_rng(s).permutation(600)
        vals.append((M.aurc(conf[p], correct[p]), M.augrc(conf[p], correct[p])))
        # the naive convention: average over every k, ties not collapsed
        order = np.argsort(-conf[p], kind="stable")
        cum = np.cumsum(1.0 - correct[p][order])
        naive.append(float((cum / np.arange(1, 601)).mean()))

    assert max(v[0] for v in vals) - min(v[0] for v in vals) == 0.0
    assert max(v[1] for v in vals) - min(v[1] for v in vals) == 0.0
    assert max(naive) - min(naive) > 1e-3        # the bug we are defending against, measured


# ---------------------------------------------------------- 8. temperature invariance

def test_temperature_cannot_change_binary_selective_metrics():
    """THE headline. For a 2-class model MSP = sigmoid(|z1 - z0| / T) is strictly monotone
    in the logit margin, so temperature scaling cannot reorder samples: it slides the
    operating point along a fixed risk-coverage curve. AURC, AUGRC and AUROC(failure) are
    therefore bit-identical across T, while ECE and NLL move.

    Consequence, and it is an assert rather than a citation: calibration is not a defence
    for a confidence-based abstention rule. That is exactly the gap an evidential /
    Dirichlet confidence is built to fill.
    """
    rng = np.random.default_rng(6)
    logits = rng.normal(0, 2.5, size=(20000, 2))
    y = rng.integers(0, 2, size=20000)
    correct = (logits.argmax(1) == y).astype(float)

    temps = (0.5, 1.0, 2.0, 5.0, 10.0)
    rows = [(T, M.aurc(M.msp(logits, T), correct),
                M.augrc(M.msp(logits, T), correct),
                M.auroc_failure(M.msp(logits, T), correct),
                M.ece(M.msp(logits, T), correct, 15),
                M.nll(logits, y, T)) for T in temps]

    for col in (1, 2, 3):                        # AURC, AUGRC, AUROC(failure)
        assert max(r[col] for r in rows) - min(r[col] for r in rows) == 0.0
    assert max(r[4] for r in rows) - min(r[4] for r in rows) > 0.05    # ECE moves
    assert max(r[5] for r in rows) - min(r[5] for r in rows) > 0.05    # NLL moves

    # three classes: the margin is no longer sufficient, so the ordering CAN change
    logits3 = rng.normal(0, 2.5, size=(20000, 3))
    y3 = rng.integers(0, 3, size=20000)
    correct3 = (logits3.argmax(1) == y3).astype(float)
    a3 = [M.aurc(M.msp(logits3, T), correct3) for T in temps]
    assert max(a3) - min(a3) > 0.0

    # fitting T on calibration data must not move the binary selective metrics either
    T_hat = M.fit_temperature(logits[:5000], y[:5000])
    assert 0.01 < T_hat < 100.0
    assert M.aurc(M.msp(logits, T_hat), correct) == rows[1][1]


# --------------------------------------- 9. the boundary of the temperature claim

def test_temperature_invariance_breaks_only_by_float_saturation():
    """Bound on test 8, so the claim ships with its own limit rather than being pushed
    back on: 'cannot change binary AURC' is exact in real arithmetic, and exact in float64
    down to T = 0.5. Below T ~ 0.2 the softmax saturates MSP to exactly 1.0 and
    manufactures tie blocks -- measured here at T=0.05: 12060/20000 samples collapse onto
    one confidence value and AURC moves 4.4e-3, the same order as the tie-permutation bug.

    The ordering never changed; the representation did. So the honest statement is
    'temperature cannot reorder samples, therefore cannot change binary AURC -- provided
    the confidences remain representable', and any table must carry n_operating_points
    next to AURC to show that they did.
    """
    rng = np.random.default_rng(6)
    logits = rng.normal(0, 2.5, size=(20000, 2))
    y = rng.integers(0, 2, size=20000)
    correct = (logits.argmax(1) == y).astype(float)
    base = M.aurc(M.msp(logits, 1.0), correct)

    for T in (0.5, 1.0, 2.0, 5.0, 10.0):
        conf = M.msp(logits, T)
        assert (conf == 1.0).sum() == 0
        assert len(np.unique(conf)) == 20000
        assert M.aurc(conf, correct) == base

    conf_sat = M.msp(logits, 0.05)
    assert (conf_sat == 1.0).sum() > 10000
    assert len(np.unique(conf_sat)) < 7000
    assert abs(M.aurc(conf_sat, correct) - base) > 1e-3
