"""WP1 selective-classification metrics for TrustFake (TReC 2026).

Data-agnostic by design: every function here consumes plain arrays -- (scores, labels),
(conf, correct), (logits, y) -- and never a dataset, a model or a file. That is what lets
WP2/WP3 hand over an npz from an attacked or robustly-trained model and get it scored
without touching their code, and what lets Munition-ID reuse this module as a data adapter
rather than a rebuild.

Conventions fixed here, because in this literature they are not standard and the
differences are the same order as the effects being reported:

  selective risk   R_sel(tau)  = sum(loss_i * [g_i >= tau]) / sum([g_i >= tau])
  generalized risk R_gen(tau)  = sum(loss_i * [g_i >= tau]) / n
  coverage         cov(tau)    = sum([g_i >= tau]) / n
  identity:        R_gen == cov * R_sel, elementwise, asserted in the test suite.

  loss is 0/1: loss_i = 1 - correct_i.

  Operating points exist ONLY at distinct confidence values (tie blocks collapsed). A
  threshold that falls strictly inside a tie block selects a set that depends on the sort
  permutation, so its risk is not a property of the model. fp16 softmax saturating to
  exactly 1.0 manufactures such blocks silently; naive cumulative AURC then varies by
  ~1.6e-2 across 8 permutations of the same predictions (growing with permutations sampled;
  re-measured 2026-08-19 on the repo's own fixture).

  AURC/AUGRC are averages over operating points. Two weightings are implemented:
    weights='block'   (default) each distinct operating point carries the size of its tie
                      block. Reduces exactly to the textbook mean-over-k=1..n definition
                      when all confidences are distinct, so numbers stay comparable to
                      published AURCs, and a 5000-sample tie block cannot count the same
                      as a singleton.
    weights='uniform' unweighted mean over the achievable coverage points.
  Trapezoid-over-coverage is a third convention in the literature; it is not implemented,
  and any table produced here must state which of the above it used.

sklearn is used for roc_auc_score and f1_score only. fd-shifts (Traub et al., the reference
implementation of AUGRC) is a hydra+lightning monorepo and is not on PyPI: it is read and
cited, never imported.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score as _sk_f1
from sklearn.metrics import roc_auc_score as _sk_auroc

__all__ = [
    "softmax", "msp", "rc_curve", "aurc", "augrc", "aurc_optimal_empirical", "eaurc",
    "auroc_detection", "auroc_failure", "accuracy", "f1", "ece", "nll", "brier",
    "fit_temperature", "apply_temperature", "risk_at_coverage", "coverage_at_risk",
    "operating_point_at_coverage", "review_rate", "summary",
]


# --------------------------------------------------------------------------- primitives

def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax in float64 with max-subtraction. Accepts float32 logits.

    float64 is not decoration: fp16/fp32 softmax saturates to exactly 1.0 for confident
    rows, which manufactures the tie blocks that break the risk-coverage curve.
    """
    z = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    z = z / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def msp(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Maximum softmax probability -- the default confidence function g(x).

    For a 2-class model MSP = sigmoid(|z1 - z0| / T), strictly monotone in the logit
    margin, so temperature cannot reorder the samples. That is why binary AURC is
    temperature-invariant (test 8) and calibration is not a defence for a
    confidence-based abstention rule.

    Swap this for an entropy, energy or Dirichlet-evidence score wherever a
    confidence_fn is taken -- the rest of the module never assumes MSP.
    """
    return softmax(logits, temperature).max(axis=1)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Return logits / T (float64). Provided so callers never divide in fp32 by hand."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return np.asarray(logits, dtype=np.float64) / float(temperature)


def _as_conf_correct(conf, correct):
    g = np.asarray(conf, dtype=np.float64).ravel()
    c = np.asarray(correct).ravel().astype(np.float64)
    if g.shape != c.shape:
        raise ValueError(f"conf {g.shape} and correct {c.shape} must have the same length")
    if g.size == 0:
        raise ValueError("empty input")
    if not np.all(np.isfinite(g)):
        raise ValueError("conf contains non-finite values")
    if not np.all((c == 0.0) | (c == 1.0)):
        raise ValueError("correct must be 0/1 or boolean")
    return g, c


# ---------------------------------------------------------------------- risk-coverage

def rc_curve(conf, correct):
    """Risk-coverage curve with tie blocks collapsed.

    Returns (coverage, selective_risk, generalized_risk, thresholds), each of length m =
    the number of DISTINCT confidence values, ordered from highest threshold (smallest
    coverage) to lowest threshold (coverage 1.0).

    Operating point j accepts exactly {i : conf_i >= thresholds[j]}; no threshold inside a
    tie block is ever evaluated, so the curve is a property of the predictions and not of
    the sort order. Block sizes are recoverable from n * diff(coverage).
    """
    g, c = _as_conf_correct(conf, correct)
    n = g.size
    loss = 1.0 - c

    order = np.argsort(-g, kind="stable")
    g_sorted = g[order]
    loss_sorted = loss[order]

    cum_loss = np.cumsum(loss_sorted)
    # last index of each tie block, i.e. positions where the next value differs
    block_end = np.flatnonzero(np.diff(g_sorted)) if n > 1 else np.array([], dtype=int)
    idx = np.concatenate([block_end, [n - 1]])

    k = (idx + 1).astype(np.float64)          # number of accepted samples
    coverage = k / n
    selective = cum_loss[idx] / k
    generalized = cum_loss[idx] / n
    thresholds = g_sorted[idx]
    return coverage, selective, generalized, thresholds


def _block_weights(coverage: np.ndarray, n: int) -> np.ndarray:
    """Tie-block sizes recovered from the coverage grid; sums to n."""
    k = np.rint(coverage * n)
    return np.diff(np.concatenate([[0.0], k]))


def _weighted_mean(values: np.ndarray, coverage: np.ndarray, n: int, weights: str) -> float:
    if weights == "uniform":
        return float(values.mean())
    if weights == "block":
        w = _block_weights(coverage, n)
        return float((values * w).sum() / w.sum())
    raise ValueError("weights must be 'block' or 'uniform'")


def aurc(conf, correct, weights: str = "block") -> float:
    """Area under the risk-coverage curve = weighted mean of SELECTIVE risk.

    Lower is better. AURC is dominated by the low-coverage tail, where selective risk
    divides by a vanishing denominator and the estimate rests on a handful of samples --
    the Traub et al. critique, and the reason augrc() exists alongside it.
    """
    g, c = _as_conf_correct(conf, correct)
    cov, sel, _, _ = rc_curve(g, c)
    return _weighted_mean(sel, cov, g.size, weights)


def augrc(conf, correct, weights: str = "block") -> float:
    """Area under the GENERALIZED risk-coverage curve = weighted mean of generalized risk.

    Generalized risk divides by n rather than by the accepted count, so it does not blow
    up as coverage goes to zero. For a random ranker AUGRC ~ r/2 where r is the error
    rate, against AURC ~ r. O(n) by construction; the O(n^2) brute force is in the tests.
    """
    g, c = _as_conf_correct(conf, correct)
    cov, _, gen, _ = rc_curve(g, c)
    return _weighted_mean(gen, cov, g.size, weights)


def aurc_optimal_empirical(correct, weights: str = "block") -> float:
    """AURC of the oracle ranker on THIS sample: same n, same errors, correct ranked first.

    Empirical, not the asymptotic closed form r + (1-r)*ln(1-r), which is 9.65e-05 off at
    n=1000 -- the same order as the effect sizes this harness reports.
    """
    c = np.asarray(correct).ravel().astype(np.float64)
    if c.size == 0:
        raise ValueError("empty input")
    n = c.size
    oracle_correct = np.concatenate([np.ones(int(c.sum())), np.zeros(n - int(c.sum()))])
    oracle_conf = np.arange(n, 0, -1, dtype=np.float64)   # strictly decreasing: no ties
    return aurc(oracle_conf, oracle_correct, weights=weights)


def eaurc(conf, correct, weights: str = "block") -> float:
    """Excess AURC = aurc - aurc_optimal_empirical. Zero for a perfect ranker, by test 1.

    Isolates ranking quality from the error rate ONLY PARTIALLY, and the residual dependence
    is large enough to mislead. A pair from this repo's own artifacts: effb0_at_pgd clean has
    AUROC(failure) 0.8006 with E-AURC 44.6e-3, while effb0_trades clean has the LOWER
    AUROC(failure) 0.7761 with the HIGHER E-AURC 58.2e-3 -- consistent, but the E-AURC gap is
    driven as much by their different error rates as by ranking. Use E-AURC BESIDE AURC as a
    within-model diagnostic; carry cross-model claims on AUROC(failure), which is rate-free
    by construction.
    """
    return aurc(conf, correct, weights) - aurc_optimal_empirical(correct, weights)


# --------------------------------------------------------------------- operating points

def operating_point_at_coverage(conf, correct, target_coverage: float):
    """(coverage, selective_risk, generalized_risk, threshold) at the largest ACHIEVABLE
    coverage <= target. Returns the actual coverage, which may differ from the target when
    a tie block straddles it -- report it, never the target."""
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must be in (0, 1]")
    cov, sel, gen, thr = rc_curve(conf, correct)
    ok = np.flatnonzero(cov <= target_coverage + 1e-12)
    j = int(ok[-1]) if ok.size else 0
    return float(cov[j]), float(sel[j]), float(gen[j]), float(thr[j])


def risk_at_coverage(conf, correct, target_coverage: float) -> float:
    """Selective risk at the largest achievable coverage <= target. See the caveat in
    operating_point_at_coverage: with heavy ties the achieved coverage can sit well below
    the target, and a table that prints 'risk@0.8' without the achieved coverage lies."""
    return operating_point_at_coverage(conf, correct, target_coverage)[1]


def coverage_at_risk(conf, correct, max_risk: float) -> float:
    """Largest coverage whose selective risk <= max_risk; 0.0 if no operating point
    qualifies. This is the deployment question: how much traffic can be auto-decided while
    holding the error rate under the moderation SLA."""
    cov, sel, _, _ = rc_curve(conf, correct)
    ok = np.flatnonzero(sel <= max_risk + 1e-12)
    return float(cov[ok].max()) if ok.size else 0.0


def review_rate(conf, threshold: float) -> float:
    """Fraction routed to human review = 1 - coverage at this threshold. The moderation
    cost axis: WP4's selective layer trades this against selective risk."""
    g = np.asarray(conf, dtype=np.float64).ravel()
    return float((g < threshold).mean())


# --------------------------------------------------------- discrimination & calibration

def accuracy(y_true, y_pred) -> float:
    """Plain top-1 accuracy. Always print it BESIDE AURC: the whole point of ACE is that
    accuracy is unchanged while AURC collapses, and without it that claim is unfalsifiable
    (PGD also wrecks AURC, but by destroying accuracy)."""
    return float((np.asarray(y_true).ravel() == np.asarray(y_pred).ravel()).mean())


def f1(y_true, y_pred, average: str = "binary") -> float:
    """F1 as named in the WP1 metric list. Positive class is 1 = fake under our binary
    mapping (real=0; synthetic and tampered both fold to 1)."""
    return float(_sk_f1(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel(),
                        average=average, zero_division=0))


def auroc_detection(p_fake, y_true) -> float:
    """AUROC of the DETECTION task: does p(fake) rank fakes above reals.

    Distinct from auroc_failure. The proposal's bare 'AUROC' is ambiguous and conflating
    the two is the cheap embarrassing error -- every table must label which one it shows.
    """
    return float(_sk_auroc(np.asarray(y_true).ravel(), np.asarray(p_fake, dtype=np.float64).ravel()))


def auroc_failure(conf, correct) -> float:
    """AUROC of FAILURE PREDICTION: does confidence rank correct predictions above the
    model's own mistakes. 0.5 is chance; a confidence attack drives it toward 0, which is
    worse than useless -- the abstention rule then preferentially rejects correct
    predictions."""
    g, c = _as_conf_correct(conf, correct)
    if c.min() == c.max():
        return float("nan")      # undefined with no errors (or no correct predictions)
    return float(_sk_auroc(c, g))


def ece(conf, correct, bins: int = 15, scheme: str = "equal_width",
        domain: tuple[float, float] = (0.0, 1.0)) -> float:
    """Expected calibration error: sum over bins of (bin mass) * |accuracy - confidence|.

    Both knobs must be stated in any table that prints this number:
      scheme='equal_width' bins over `domain`; 'equal_mass' uses quantile edges.
      domain: binary MSP lives in [0.5, 1], so equal-width bins over [0,1] leave 7 of 15
              empty and the number is not comparable to a multi-class ECE.

    ECE is also biased upward by binning: a perfectly calibrated binary model measures
    ~0.0056 at 5 bins, ~0.0102 at 15, ~0.0573 at 200. The floor is ~0.01 at n~5000, so
    report nll() and brier() (unbiased, no binning) alongside it.
    """
    g, c = _as_conf_correct(conf, correct)
    n = g.size
    if scheme == "equal_width":
        edges = np.linspace(domain[0], domain[1], bins + 1)
    elif scheme == "equal_mass":
        edges = np.quantile(g, np.linspace(0.0, 1.0, bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
    else:
        raise ValueError("scheme must be 'equal_width' or 'equal_mass'")
    idx = np.clip(np.digitize(g, edges[1:-1], right=False), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        cnt = int(m.sum())
        if cnt:
            total += (cnt / n) * abs(c[m].mean() - g[m].mean())
    return float(total)


def nll(logits, y_true, temperature: float = 1.0) -> float:
    """Mean negative log-likelihood in float64. Unbiased, no binning, and it MOVES under
    temperature scaling -- which is exactly what makes it the right companion to the
    temperature-invariance result on AURC."""
    p = softmax(logits, temperature)
    y = np.asarray(y_true).ravel().astype(int)
    return float(-np.log(np.clip(p[np.arange(y.size), y], 1e-300, 1.0)).mean())


def brier(logits, y_true, temperature: float = 1.0) -> float:
    """Multi-class Brier score: mean over samples of sum_k (p_k - onehot_k)^2.

    Note the convention: for two classes this is twice the (p - y)^2 form some papers use.
    """
    p = softmax(logits, temperature)
    y = np.asarray(y_true).ravel().astype(int)
    onehot = np.zeros_like(p)
    onehot[np.arange(y.size), y] = 1.0
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def fit_temperature(logits_calib, y_calib, bounds: tuple[float, float] = (0.01, 100.0)) -> float:
    """Fit a single temperature T by minimizing NLL on the CALIBRATION split (1-D L-BFGS-B
    on log T, analytic gradient).

    Fit here, apply elsewhere. Fitting T on the split you report is the classic self-own
    and the first thing a reviewer checks; the manifest keeps calib and test on disjoint
    shards so this cannot happen by accident.

    It will move ECE and NLL. It provably cannot move binary AURC, AUGRC or
    AUROC(failure) -- see msp().
    """
    from scipy.optimize import minimize

    z = np.asarray(logits_calib, dtype=np.float64)
    y = np.asarray(y_calib).ravel().astype(int)
    rows = np.arange(y.size)

    def obj(log_t):
        t = float(np.exp(log_t[0]))
        s = z / t
        s = s - s.max(axis=1, keepdims=True)
        lse = np.log(np.exp(s).sum(axis=1))
        loss = float((lse - s[rows, y]).mean())
        p = np.exp(s - lse[:, None])
        onehot = np.zeros_like(p)
        onehot[rows, y] = 1.0
        # d loss / d log t = mean over samples of  -(1/t) * sum_k (p_k - onehot_k) * z_k
        grad = float((-(p - onehot) * z).sum(axis=1).mean() / t)
        return loss, np.array([grad])

    lo, hi = np.log(bounds[0]), np.log(bounds[1])
    res = minimize(obj, x0=np.array([0.0]), jac=True, method="L-BFGS-B", bounds=[(lo, hi)])
    return float(np.exp(res.x[0]))


# ------------------------------------------------------------------------------ summary

def summary(logits, y_true, temperature: float = 1.0, positive_class: int = 1,
            coverages=(0.8, 0.5), weights: str = "block") -> dict:
    """One row of the condition table, from raw logits.

    Every number a WP1 table needs, computed through one code path so the clean and
    attacked rows can never drift apart. Keys are stable; add, never rename.
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true).ravel().astype(int)
    p = softmax(z, temperature)
    yhat = p.argmax(axis=1)
    conf = p.max(axis=1)
    correct = (yhat == y).astype(np.float64)

    out = {
        "n": int(y.size),
        "temperature": float(temperature),
        "weights": weights,
        "accuracy": accuracy(y, yhat),
        "error_rate": 1.0 - accuracy(y, yhat),
        "f1": f1(y, yhat) if p.shape[1] == 2 else f1(y, yhat, average="macro"),
        "auroc_detection": auroc_detection(p[:, positive_class], (y == positive_class).astype(int)),
        "auroc_failure": auroc_failure(conf, correct),
        "aurc": aurc(conf, correct, weights),
        "augrc": augrc(conf, correct, weights),
        "eaurc": eaurc(conf, correct, weights),
        "ece_15_equal_width_01": ece(conf, correct, 15, "equal_width", (0.0, 1.0)),
        "ece_15_equal_mass": ece(conf, correct, 15, "equal_mass"),
        "nll": nll(z, y, temperature),
        "brier": brier(z, y, temperature),
        "n_operating_points": int(rc_curve(conf, correct)[0].size),
    }
    if p.shape[1] == 2:
        out["ece_15_equal_width_binary_domain"] = ece(conf, correct, 15, "equal_width", (0.5, 1.0))
    for c in coverages:
        cov, sel, gen, thr = operating_point_at_coverage(conf, correct, c)
        out[f"risk@cov{c:g}"] = sel
        out[f"achieved_cov@{c:g}"] = cov
    return out
