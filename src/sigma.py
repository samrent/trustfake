"""The uncertainty seam: carry a per-image sigma alongside the logits, and score it.

The PIs' decision diagram routes moderation on TWO axes -- a risk axis p(fake) AND an
uncertainty axis sigma -- where sigma comes from MC-Dropout, deep ensembles or evidential
heads that OTHER subgroups build. The prediction contract carries logits only, so those
methods have nowhere to land. This module is the slot.

A sigma file is decoupled from the prediction that it annotates, because in the real workflow
another subgroup produces it:

    runs/sigma/sigma_{producer}_{split}_{condition}.npz
        uid    str          join key, must be a subset of the prediction's uids
        sigma  float32[n]   higher = more uncertain, by convention
    + JSON sidecar {producer, semantics, scale, members, n, note}

`sigma_aware` scoring answers the question the deck implies but never states: is sigma a
BETTER failure predictor than the softmax confidence it is meant to surpass? If sigma is just
a monotone function of 1 - MSP it adds nothing, and `is_degenerate` catches exactly that --
"confidence beyond softmax" is meaningless when the new signal is the old one relabelled.

The reference producer here is a DEEP ENSEMBLE (named in the PIs' deck): sigma = standard
deviation across ensemble members of p(fake) at T=1. It needs no retraining and no dropout
(EfficientNet-B0 as configured has no dropout modules), and it is computed post-hoc from
prediction npz files already on disk, so it is CPU-only. It is a reference to validate the
seam; the camp brings its own sigma and drops it into the same slot.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from . import metrics as M

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRED = ROOT / "runs" / "predictions"
SIGMA = ROOT / "runs" / "sigma"


def _p_fake(stem: str, pred_dir: pathlib.Path, temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(pred_dir / f"{stem}.npz", allow_pickle=False)
    return z["uid"], M.softmax(z["logits"].astype(np.float64), temperature)[:, 1]


def ensemble_sigma(members: list[str], split: str, condition: str,
                   pred_dir: pathlib.Path = PRED, out_dir: pathlib.Path = SIGMA) -> pathlib.Path:
    """Deep-ensemble uncertainty: std of p(fake) across members, per image, at T=1.

    T=1 on purpose: the members' frozen calibration temperatures differ by 4x (standard 3.34,
    TRADES 0.76), and dividing each member's logits by a different T before taking their
    variance would measure calibration disagreement, not model disagreement. The ensemble
    signal is the raw predictive spread.
    """
    uids, cols = None, []
    for m in members:
        u, p = _p_fake(f"predictions_{m}_{split}_{condition}", pred_dir)
        idx = {x: i for i, x in enumerate(u.tolist())}
        if uids is None:
            uids = u
        keep = np.array([idx[x] for x in uids.tolist()])   # align every member to the first
        cols.append(p[keep])
    P = np.stack(cols, axis=0)                              # [members, n]
    sigma = P.std(axis=0, ddof=0).astype(np.float32)
    return write_sigma(uids, sigma, f"ens{len(members)}", split, condition,
                       semantics="higher = more uncertain", scale="std of p(fake) in [0, 0.5]",
                       members=members, note="deep ensemble, T=1, post-hoc from prediction npz",
                       out_dir=out_dir)


def write_sigma(uid, sigma, producer, split, condition, semantics, scale,
                members=None, note="", out_dir: pathlib.Path = SIGMA) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sigma_{producer}_{split}_{condition}"
    np.savez(out_dir / f"{stem}.npz", uid=np.asarray(uid), sigma=np.asarray(sigma, np.float32))
    (out_dir / f"{stem}.json").write_text(json.dumps({
        "producer": producer, "split": split, "condition": condition, "n": int(len(uid)),
        "semantics": semantics, "scale": scale, "members": members, "note": note,
    }, indent=2) + "\n")
    return out_dir / f"{stem}.npz"


def load_sigma(stem: str, pred_uid: np.ndarray, sigma_dir: pathlib.Path = SIGMA) -> np.ndarray:
    """Return sigma aligned to a prediction's uid order. Raises if the sigma file does not
    cover every predicted uid -- a silent inner-join would drop rows and misstate coverage."""
    z = np.load(sigma_dir / f"{stem}.npz", allow_pickle=False)
    idx = {u: i for i, u in enumerate(z["uid"].tolist())}
    missing = [u for u in pred_uid.tolist() if u not in idx]
    if missing:
        raise ValueError(f"{stem}: sigma missing for {len(missing)} predicted uids "
                         f"(e.g. {missing[:3]}); a sigma file must cover the prediction it annotates")
    keep = np.array([idx[u] for u in pred_uid.tolist()])
    return z["sigma"].astype(np.float64)[keep]


def is_degenerate(sigma: np.ndarray, conf: np.ndarray, thresh: float = 0.98) -> tuple[bool, float]:
    """Is this sigma just the softmax confidence relabelled? Compares sigma against (1 - conf)
    by rank correlation; |rho| >= thresh means the 'new' uncertainty carries no information the
    MSP did not already have, and the deck's 'confidence beyond softmax' promise is unmet."""
    a = np.asarray(sigma, np.float64)
    b = 1.0 - np.asarray(conf, np.float64)
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return True, 1.0
    rho = float(np.corrcoef(ra, rb)[0, 1])
    return abs(rho) >= thresh, rho


def sigma_report(model: str, split: str, condition: str, sigma_stem: str,
                 pred_dir: pathlib.Path = PRED, sigma_dir: pathlib.Path = SIGMA,
                 temperature: float = 1.0) -> dict:
    """Failure-prediction quality of sigma vs MSP on the same rows: does the uncertainty axis
    sort mistakes from correct answers better than the softmax confidence does?"""
    z = np.load(pred_dir / f"predictions_{model}_{split}_{condition}.npz", allow_pickle=False)
    logits, y = z["logits"].astype(np.float64), z["y"].astype(int)
    conf = M.msp(logits, temperature)
    correct = (logits.argmax(1) == y).astype(float)
    sigma = load_sigma(sigma_stem, z["uid"], sigma_dir)
    degen, rho = is_degenerate(sigma, conf)
    # a good uncertainty ranks CORRECT above wrong; sigma is "higher = more uncertain", so its
    # failure-AUROC is measured on -sigma to match the MSP convention (higher = more correct)
    return {
        "model": model, "condition": condition, "sigma": sigma_stem,
        "auroc_fail_msp": M.auroc_failure(conf, correct),
        "auroc_fail_sigma": M.auroc_failure(-sigma, correct),
        "aurc_msp": M.aurc(conf, correct),
        "aurc_sigma": M.aurc(-sigma, correct),
        "sigma_is_degenerate": degen, "rank_corr_with_1_minus_msp": rho,
        "n": int(y.size),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--members", nargs="*",
                    default=["effb0_standard_eps2_255", "effb0_at_pgd_eps2_255", "effb0_trades_eps2_255"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--conditions", nargs="*", default=["clean"])
    ap.add_argument("--report-model", default=None, help="if set, also print sigma-vs-MSP for this model")
    a = ap.parse_args()
    for cond in a.conditions:
        p = ensemble_sigma(a.members, a.split, cond)
        print(f"  wrote {p.name}")
        if a.report_model:
            stem = f"sigma_ens{len(a.members)}_{a.split}_{cond}"
            r = sigma_report(a.report_model, a.split, cond, stem)
            print(f"    {a.report_model} {cond}: AUROC(fail) MSP {r['auroc_fail_msp']:.4f} "
                  f"vs sigma {r['auroc_fail_sigma']:.4f}  "
                  f"(sigma {'DEGENERATE' if r['sigma_is_degenerate'] else 'independent'}, "
                  f"rho={r['rank_corr_with_1_minus_msp']:.3f})")


if __name__ == "__main__":
    main()
