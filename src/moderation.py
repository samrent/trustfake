"""WP4 selective moderation layer (deliverable D4).

Turns a detector output into one of three moderation ACTIONS:

    ALLOW    auto-accept the image as real
    REVIEW   escalate to a human moderator
    FLAG     auto-flag the image as fake

The rule is TWO thresholds on p(fake), not one on confidence:

    p_fake <  t_low               -> ALLOW
    p_fake >  t_high              -> FLAG
    t_low <= p_fake <= t_high     -> REVIEW

Why two, and why on the risk axis rather than the confidence axis: a single confidence
threshold (conf = max(p, 1-p) < tau) is exactly the symmetric special case
t_low = 1 - tau, t_high = tau, so nothing is lost -- but moderation costs are NOT symmetric.
Auto-allowing a deepfake ships harm; auto-flagging a real image annoys a user; escalating
either one costs a moderator's minute. Two thresholds let those three prices be set
independently. A symmetric rule silently asserts they are equal.

THRESHOLDS ARE FITTED ON THE CLEAN CALIBRATION SPLIT AND FROZEN. They are then applied
unchanged to test and to every attacked condition. Re-fitting on attacked data would be an
oracle policy -- it assumes the moderator knows an attack is underway, which is precisely
the assumption the attack defeats -- and it would understate every collapse.

Deployment indicators, as the proposal names them:
    coverage       fraction auto-decided (1 - review_rate)
    residual_risk  error rate AMONG the auto-decided (what actually ships wrong)
    review_rate    fraction sent to a human (the moderation bill)
plus the two that matter asymmetrically:
    missed_fake_rate   fakes auto-ALLOWED, as a fraction of all fakes. The harm.
    false_flag_rate    reals auto-FLAGGED, as a fraction of all reals. The annoyance.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from . import metrics as M
from .sigma import load_sigma

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRED = ROOT / "runs" / "predictions"

ALLOW, REVIEW, FLAG = 0, 1, 2
ACTION_NAMES = {ALLOW: "allow", REVIEW: "review", FLAG: "flag"}


def actions(p_fake: np.ndarray, t_low: float, t_high: float,
            sigma: np.ndarray = None, t_sigma: float = None) -> np.ndarray:
    """The deck's decision rule. Risk axis p(fake): below t_low ALLOW, above t_high FLAG,
    between REVIEW. Uncertainty axis (optional): sigma above t_sigma forces REVIEW regardless
    of p(fake) -- ALLOW = confident real AND certain; FLAG = confident fake AND certain.

    With sigma=None this is exactly the one-axis rule, so every existing call is unchanged.
    The uncertainty gate only ever MOVES items into REVIEW; it never auto-decides, so it
    cannot increase residual risk, only trade coverage for safety."""
    p = np.asarray(p_fake, dtype=np.float64)
    out = np.full(p.shape, REVIEW, dtype=np.int8)
    out[p < t_low] = ALLOW
    out[p > t_high] = FLAG
    if sigma is not None and t_sigma is not None:
        out[np.asarray(sigma, dtype=np.float64) > t_sigma] = REVIEW
    return out


def evaluate_policy(p_fake, y, t_low: float, t_high: float,
                    sigma=None, t_sigma: float = None) -> dict:
    """Deployment indicators for one (policy, condition) pair. y: 1 = fake, 0 = real.
    Pass sigma + t_sigma for the two-axis rule."""
    p = np.asarray(p_fake, dtype=np.float64)
    y = np.asarray(y).astype(int)
    a = actions(p, t_low, t_high, sigma, t_sigma)
    auto = a != REVIEW
    n = y.size

    decided_pred = (a[auto] == FLAG).astype(int)          # FLAG asserts fake
    decided_y = y[auto]
    n_auto = int(auto.sum())

    fakes, reals = y == 1, y == 0
    return {
        "t_low": float(t_low), "t_high": float(t_high), "n": int(n),
        "coverage": n_auto / n,
        "review_rate": 1.0 - n_auto / n,
        # NaN, not 0.0. A policy that auto-decides nothing has no residual risk to measure,
        # and reporting 0.0 makes total refusal to work look like perfect safety -- the single
        # most flattering possible misreading of a collapsed system.
        "residual_risk": float((decided_pred != decided_y).mean()) if n_auto else float("nan"),
        "n_auto": n_auto,
        "missed_fake_rate": float((a[fakes] == ALLOW).mean()) if fakes.any() else 0.0,
        "false_flag_rate": float((a[reals] == FLAG).mean()) if reals.any() else 0.0,
        "review_of_fakes": float((a[fakes] == REVIEW).mean()) if fakes.any() else 0.0,
        "review_of_reals": float((a[reals] == REVIEW).mean()) if reals.any() else 0.0,
        "full_coverage_error": float(((p > 0.5).astype(int) != y).mean()),
    }


def fit_thresholds(p_fake, y, sla_residual_risk: float = 0.05,
                   sla_missed_fake: float | None = None, grid: int = 200,
                   min_auto: int | None = None) -> tuple[float, float]:
    """Minimize review rate subject to the residual-risk SLA (and optionally a cap on the
    missed-fake rate), on the CLEAN CALIBRATION split.

    The SLA is a CHOICE, not a fact. 5% residual risk means one in twenty auto-decisions is
    wrong; a real platform would set it from the cost of a wrong decision and the size of its
    moderation team. It is recorded in the artifact so a reader can disagree with it
    explicitly rather than inherit it silently.

    If no threshold pair meets the SLA, the policy degrades to REVIEW EVERYTHING (t_low=0,
    t_high=1, review_rate=1.0) rather than quietly returning the least-bad infeasible pair.
    An infeasible SLA must be visible as "this model cannot be deployed at this SLA", which
    is a finding, not a failure.
    """
    p = np.asarray(p_fake, dtype=np.float64)
    y = np.asarray(y).astype(int)
    # A residual risk of 0% over 3 auto-decided items is not evidence of a 5% SLA: the
    # tightest thresholds always "satisfy" any SLA by deciding almost nothing. Require enough
    # auto-decisions that the SLA could have been violated at all -- 1/sla items is the
    # coarsest resolution at which the measured rate can even represent the threshold.
    if min_auto is None:
        min_auto = int(np.ceil(1.0 / max(sla_residual_risk, 1e-9)))
    qs = np.unique(np.quantile(p, np.linspace(0, 1, grid)))
    best, best_review = (0.0, 1.0), 1.0
    for t_low in qs:
        hi = qs[qs >= t_low]
        for t_high in hi:
            r = evaluate_policy(p, y, t_low, t_high)
            if r["n_auto"] < min_auto:
                continue
            if not np.isfinite(r["residual_risk"]) or r["residual_risk"] > sla_residual_risk:
                continue
            if sla_missed_fake is not None and r["missed_fake_rate"] > sla_missed_fake:
                continue
            if r["review_rate"] < best_review:
                best_review, best = r["review_rate"], (float(t_low), float(t_high))
    return best


def fit_sigma_gate(sigma, clean_review_budget: float = 0.10) -> float:
    """Freeze the uncertainty gate at a CLEAN review budget: review the most-uncertain
    `clean_review_budget` fraction of clean-calib items. t_sigma = the (1 - budget) quantile
    of clean-calib sigma.

    This is the honest design. On clean the gate costs a little extra review for little gain.
    Its value is entirely UNDER ATTACK: a confidence attack that leaves accuracy untouched
    still drives model uncertainty up, so far more items cross the frozen t_sigma and are
    sent to a human instead of being wrongly auto-decided. The gate is fitted on clean and
    never sees attacked data, so the rescue is not an oracle."""
    q = float(np.quantile(np.asarray(sigma, np.float64), 1.0 - clean_review_budget))
    return q


def p_fake_from(logits: np.ndarray, temperature: float) -> np.ndarray:
    return M.softmax(logits, temperature)[:, 1]


def load(stem: str, pred: pathlib.Path = PRED):
    z = np.load(pred / f"{stem}.npz", allow_pickle=False)
    meta = json.loads((pred / f"{stem}.json").read_text())
    return z["logits"].astype(np.float64), z["y"].astype(int), z["label"].astype(int), meta, z["uid"]


COLS = [("condition", 26), ("acc", 7), ("coverage", 9), ("review%", 8), ("resid_risk", 11),
        ("missed_fake", 12), ("false_flag", 11)]


def fmt(rows: list[dict]) -> str:
    head = " ".join(f"{c:>{w}}" for c, w in COLS)
    lines = [head, "-" * len(head)]
    for r in rows:
        v = {"condition": r["condition"], "acc": f"{r['accuracy']:.4f}",
             "coverage": f"{r['coverage']*100:.1f}%", "review%": f"{r['review_rate']*100:.1f}%",
             "resid_risk": f"{r['residual_risk']*100:.2f}%",
             "missed_fake": f"{r['missed_fake_rate']*100:.2f}%",
             "false_flag": f"{r['false_flag_rate']*100:.2f}%"}
        lines.append(" ".join(f"{v[c]:>{w}}" for c, w in COLS))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="tf_efficientnet_b0")
    ap.add_argument("--conditions", nargs="*", default=["clean"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--sla", type=float, default=0.05, help="max residual risk on auto-decisions")
    ap.add_argument("--sigma-budget", type=float, default=0.10,
                    help="clean review budget for the uncertainty gate")
    ap.add_argument("--sigma-producer", default=None,
                    help="e.g. ens3 -> loads runs/sigma/sigma_{producer}_{split}_{cond}.npz and "
                         "adds the uncertainty gate (the deck's second axis)")
    ap.add_argument("--sla-missed-fake", type=float, default=None)
    ap.add_argument("--pred-dir", type=pathlib.Path, default=PRED)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "runs" / "moderation_report.md")
    a = ap.parse_args()

    blp = ROOT / "runs" / "trivial_baselines.json"
    if blp.exists():
        from .baselines import headline
        print(headline(json.loads(blp.read_text())) + "\n")

    clz = np.load(a.pred_dir / f"predictions_{a.model}_calib_clean.npz", allow_pickle=False)
    cl, cy = clz["logits"].astype(np.float64), clz["y"].astype(int)
    T = M.fit_temperature(cl, cy)
    t_low, t_high = fit_thresholds(p_fake_from(cl, T), cy, a.sla, a.sla_missed_fake)

    t_sigma = None
    if a.sigma_producer:
        from . import sigma as SIG
        csig = load_sigma(f"sigma_{a.sigma_producer}_calib_clean", clz["uid"])
        t_sigma = fit_sigma_gate(csig, a.sigma_budget)
        print(f"sigma gate ({a.sigma_producer}) fitted on clean calib: t_sigma = {t_sigma:.5f} "
              f"(reviews the {a.sigma_budget:.0%} most-uncertain clean items; frozen)")
    calib_result = evaluate_policy(p_fake_from(cl, T), cy, t_low, t_high)
    print(f"policy fitted on CLEAN CALIB (n={cy.size}): T={T:.4f}  "
          f"t_low={t_low:.4f}  t_high={t_high:.4f}   SLA residual risk <= {a.sla:.0%}")
    print(f"  on calib itself: review {calib_result['review_rate']*100:.1f}%  "
          f"residual risk {calib_result['residual_risk']*100:.2f}%  "
          f"missed fakes {calib_result['missed_fake_rate']*100:.2f}%")
    if t_low == 0.0 and t_high == 1.0:
        print("  !! SLA INFEASIBLE: no threshold pair met it; policy degrades to review-everything")

    rows = []
    for cond in a.conditions:
        try:
            lo, y, lab, meta, uid = load(f"predictions_{a.model}_{a.split}_{cond}", a.pred_dir)
        except FileNotFoundError:
            print(f"  (skipping {cond}: no predictions)")
            continue
        p = p_fake_from(lo, T)
        r = evaluate_policy(p, y, t_low, t_high)
        r["condition"] = cond
        r["accuracy"] = float(((p > 0.5).astype(int) == y).mean())
        if a.sigma_producer and t_sigma is not None:
            try:
                sg = load_sigma(f"sigma_{a.sigma_producer}_{a.split}_{cond}", uid)
                r2 = evaluate_policy(p, y, t_low, t_high, sg, t_sigma)
                r["residual_risk_2axis"] = r2["residual_risk"]
                r["review_rate_2axis"] = r2["review_rate"]
                r["coverage_2axis"] = r2["coverage"]
            except (FileNotFoundError, ValueError):
                r["residual_risk_2axis"] = None
        rows.append(r)

    table = fmt(rows)
    print("\n" + table)

    if a.sigma_producer and t_sigma is not None:
        cmp = ["", "1-axis (p_fake only)  vs  2-axis (+ uncertainty gate):",
               f"{'condition':<26}{'resid 1ax':>11}{'resid 2ax':>11}{'review 1ax':>12}{'review 2ax':>12}"]
        for r in rows:
            if r.get("residual_risk_2axis") is None:
                continue
            def pc(v):
                return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v*100:.2f}%"
            cmp.append(f"{r['condition']:<26}{pc(r['residual_risk']):>11}{pc(r['residual_risk_2axis']):>11}"
                       f"{pc(r['review_rate']):>12}{pc(r['review_rate_2axis']):>12}")
        print("\n".join(cmp))
    a.out.write_text(
        f"# WP4 selective moderation -- {a.model}\n\n"
        f"- policy fitted on CLEAN CALIB and frozen: `t_low={t_low:.4f}`, `t_high={t_high:.4f}`, "
        f"temperature `{T:.4f}`\n"
        f"- objective: minimize review rate subject to residual risk <= {a.sla:.0%} "
        f"(an SLA is a choice, not a fact)\n"
        f"- provenance: SID-Set validation split, held-out slice (official test split withheld)\n"
        f"- residual risk = error rate among AUTO-DECIDED items; missed_fake = fakes auto-allowed\n\n"
        f"```\n{table}\n```\n")
    a.out.with_suffix(".json").write_text(json.dumps(
        {"temperature": T, "t_low": t_low, "t_high": t_high, "sla": a.sla,
         "calib": calib_result, "rows": rows}, indent=2, default=float) + "\n")
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
