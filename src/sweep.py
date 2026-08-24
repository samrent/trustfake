"""Combinatorial robustness-training sweep: train many defense configs on a small fit slice,
score each on the fresh selection-val slice, and rank by CONFIDENCE RESILIENCE under attack.

Why confidence resilience, not robust accuracy: robust accuracy under PGD/AutoAttack is the
crowded label axis. This project's thesis and the PIs' area is the CONFIDENCE axis -- an attack
that leaves accuracy identical but destroys the failure-prediction ranking the moderation layer
depends on. So "works best" = the config whose failure-AUROC stays highest under the confidence
attacks (ACE, over-confidence) while holding a clean-accuracy floor.

Discipline (see the approved plan + HOLDOUT.md):
- train on a SMALL fit slice (--limit-fit), each config to a UNIQUE --run-name so nothing
  collides, checkpoints into runs/sweep_ckpt/ (never the canonical runs/checkpoints);
- rank on the SELVAL slice (manifest_v4_sweep.parquet), predictions into runs/sweep_pred/
  (never the canonical runs/predictions), scored at T=1 -- valid because AURC / failure-AUROC
  are temperature-invariant for a binary model (proved in metrics.py / tests);
- NEVER touch the sealed holdout;
- the single winning config is confirmed once on the full test split, separately.

Strategy A (hyperparameter grid of the existing standard/at_pgd/trades objectives) needs no new
training code and runs today. Strategies B (stacked/hybrid) and C (confidence-targeted defense)
require new loss objectives in train.py and are filled in once the owner picks them from
SWEEP-STRATEGIES.md -- their grids are stubbed below.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

from . import metrics as M

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
CKPT = ROOT / "runs" / "sweep_ckpt"
PRED = ROOT / "runs" / "sweep_pred"
MANIFEST = ROOT / "runs" / "manifest_v4_sweep.parquet"

# The confidence attacks the ranking is judged on, plus clean for the accuracy floor.
EVAL_ATTACKS = [("clean", None), ("ace_uint8", 0.005), ("overconf", 0.0157)]


def grid_A(limit_fit: int, epochs: int) -> list[dict]:
    """Existing methods, pruned ~12-config grid. Each dict is a full train invocation."""
    base = dict(limit_fit=limit_fit, epochs=epochs, batch=32, manifest=str(MANIFEST))
    cfgs = []
    # a clean-trained reference
    cfgs.append({**base, "method": "standard", "eps": 2 / 255, "run_name": "sw_std"})
    # PGD-AT across epsilon x inner-steps
    for eps in (1, 2, 4):
        for steps in (3, 7):
            cfgs.append({**base, "method": "at_pgd", "eps": eps / 255, "inner_steps": steps,
                         "run_name": f"sw_atpgd_e{eps}_s{steps}"})
    # TRADES across epsilon x beta
    for eps in (1, 2, 4):
        for beta in (3, 6):
            cfgs.append({**base, "method": "trades", "eps": eps / 255, "beta": beta,
                         "run_name": f"sw_trades_e{eps}_b{beta}"})
    return cfgs


GRIDS = {
    "A": grid_A,
    # "B": grid_B,  # stacked/hybrid -- needs new objectives in train.py (see SWEEP-STRATEGIES.md)
    # "C": grid_C,  # confidence-targeted defense -- needs a new objective (see SWEEP-STRATEGIES.md)
}


def train_config(cfg: dict) -> pathlib.Path:
    """Train one config as a subprocess (fresh process => no GPU-memory accumulation)."""
    ck = CKPT / f"{cfg['run_name']}.pt"
    if ck.exists():
        print(f"  [skip train] {cfg['run_name']} (checkpoint exists)")
        return ck
    cmd = [PY, "-m", "src.train", "--methods", cfg["method"], "--run-name", cfg["run_name"],
           "--ckpt-dir", str(CKPT), "--limit-fit", str(cfg["limit_fit"]),
           "--epochs", str(cfg["epochs"]), "--batch", str(cfg["batch"]),
           "--eps", repr(cfg["eps"]), "--manifest", cfg["manifest"]]
    for k in ("inner_steps", "beta", "eps_warmup_epochs", "lr"):
        if k in cfg:
            cmd += [f"--{k.replace('_', '-')}", str(cfg[k])]
    if cfg.get("init_from"):
        cmd += ["--init-from", cfg["init_from"]]
    print(f"  [train] {cfg['run_name']}: {cfg['method']} eps={cfg['eps']*255:.0f}/255")
    subprocess.run(cmd, check=True, cwd=ROOT, env={**_env()})
    return ck


def eval_config(cfg: dict) -> None:
    """Score one checkpoint on selval for clean + the confidence attacks, into runs/sweep_pred."""
    ck = (CKPT / f"{cfg['run_name']}.pt").resolve()
    for attack, eps in EVAL_ATTACKS:
        stem = PRED / f"predictions_{cfg['run_name']}_selval_{_cond(attack, eps)}.npz"
        if stem.exists():
            continue
        cmd = [PY, "-m", "src.run_attacks", "--model", f"ckpt:{ck}",
               "--model-id", cfg["run_name"], "--split", "selval", "--attacks", attack,
               "--batch", "32", "--pred-dir", str(PRED), "--manifest", str(MANIFEST)]
        if eps is not None:
            cmd += ["--eps", str(eps)]
        subprocess.run(cmd, check=True, cwd=ROOT, env={**_env()})


def _cond(attack, eps):
    from .attack_suite import condition_name
    return "clean" if attack == "clean" else condition_name(attack, eps=eps)


def _env():
    import os
    return {**os.environ, "HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


def _score(run_name: str, cond: str) -> dict | None:
    p = PRED / f"predictions_{run_name}_selval_{cond}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    lo, y = z["logits"].astype(np.float64), z["y"].astype(int)
    conf = M.msp(lo, 1.0)                       # T=1: AURC/failure-AUROC are T-invariant here
    correct = (lo.argmax(1) == y).astype(float)
    return {"acc": M.accuracy(y, lo.argmax(1)),
            "fauroc": M.auroc_failure(conf, correct),
            "aurc": M.aurc(conf, correct)}


def rank(cfgs: list[dict], clean_floor: float, out: pathlib.Path) -> None:
    """Rank by confidence resilience (mean failure-AUROC under ACE + over-conf) subject to a
    clean-accuracy floor; report a composite beside it so the owner can re-weight."""
    rows = []
    ace, over = _cond("ace_uint8", 0.005), _cond("overconf", 0.0157)
    for cfg in cfgs:
        rn = cfg["run_name"]
        cl, a, o = _score(rn, "clean"), _score(rn, ace), _score(rn, over)
        if not (cl and a and o):
            print(f"  (incomplete: {rn})")
            continue
        # resilience = how well confidence still sorts mistakes under the two confidence attacks
        resil = float(np.nanmean([a["fauroc"], o["fauroc"]]))
        meets = cl["acc"] >= clean_floor
        rows.append({
            "run_name": rn, "method": cfg["method"], "eps_255": round(cfg["eps"] * 255, 1),
            "clean_acc": cl["acc"], "clean_fauroc": cl["fauroc"],
            "ace_fauroc": a["fauroc"], "overconf_fauroc": o["fauroc"],
            "ace_aurc_e3": a["aurc"] * 1e3, "overconf_aurc_e3": o["aurc"] * 1e3,
            "conf_resilience": resil, "meets_clean_floor": meets,
            # composite: half clean accuracy, half confidence resilience (both ~[0,1]); documented,
            # re-weightable from the JSON. Floor-failing configs keep their score but sort last.
            "composite": 0.5 * cl["acc"] + 0.5 * resil,
        })
    # sort: floor-meeting first, then by confidence resilience
    rows.sort(key=lambda r: (r["meets_clean_floor"], r["conf_resilience"]), reverse=True)

    hdr = (f"{'run_name':<20}{'method':<9}{'eps':>5}{'clean_acc':>10}{'ace_fAUROC':>11}"
           f"{'ovr_fAUROC':>11}{'RESILIENCE':>11}{'composite':>10}{'floor':>6}")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(f"{r['run_name']:<20}{r['method']:<9}{r['eps_255']:>5.0f}{r['clean_acc']:>10.4f}"
                     f"{r['ace_fauroc']:>11.4f}{r['overconf_fauroc']:>11.4f}{r['conf_resilience']:>11.4f}"
                     f"{r['composite']:>10.4f}{('yes' if r['meets_clean_floor'] else 'NO'):>6}")
    body = "\n".join(lines)
    print("\n" + body)
    out.write_text(
        f"# Robustness sweep ranking — by confidence resilience under attack\n\n"
        f"Trained on a small fit slice; scored on the fresh SELVAL slice (never trained on) at T=1.\n"
        f"RESILIENCE = mean failure-AUROC under ACE-uint8 + over-confidence (higher = confidence stays\n"
        f"trustworthy under attack). Clean floor = {clean_floor}. The winner is confirmed once on test.\n\n"
        f"```\n{body}\n```\n")
    out.with_suffix(".json").write_text(json.dumps({"clean_floor": clean_floor, "rows": rows},
                                                   indent=2, default=float) + "\n")
    print(f"\nwritten: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", choices=list(GRIDS), default="A")
    ap.add_argument("--limit-fit", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--clean-floor", type=float, default=0.75)
    ap.add_argument("--smoke", action="store_true", help="one tiny config end-to-end")
    ap.add_argument("--rank-only", action="store_true", help="skip train/eval, just rebuild the table")
    a = ap.parse_args()
    CKPT.mkdir(parents=True, exist_ok=True); PRED.mkdir(parents=True, exist_ok=True)

    if a.smoke:
        cfgs = [dict(method="at_pgd", eps=2 / 255, inner_steps=3, limit_fit=800, epochs=2,
                     batch=32, manifest=str(MANIFEST), run_name="sw_smoke")]
    else:
        cfgs = GRIDS[a.strategy](a.limit_fit, a.epochs)

    print(f"sweep strategy {a.strategy}: {len(cfgs)} config(s), limit_fit={cfgs[0]['limit_fit']}, "
          f"epochs={cfgs[0]['epochs']}")
    if not a.rank_only:
        for cfg in cfgs:
            train_config(cfg)
            eval_config(cfg)
    rank(cfgs, a.clean_floor, ROOT / "runs" / "sweep_ranking.md")


if __name__ == "__main__":
    main()
