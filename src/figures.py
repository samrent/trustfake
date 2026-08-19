"""The two figures. Both read prediction npz files and nothing else.

Fig 1 -- risk-coverage: selective AND generalized risk on the same axes, one line per
condition, AURC/AUGRC annotated. The x-axis is floored at 5% coverage: below that the
selective risk is an average over a handful of samples, and plotting it presents sampling
noise as a result.

Fig 2 -- confidence histograms split by correct/incorrect, clean vs attacked (ACE paper
Fig 5). This is the most legible slide in the deck: under a confidence attack the two
distributions swap places, which is what "the abstention rule collapsed" looks like.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import metrics as M

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRED = ROOT / "runs" / "predictions"
COV_FLOOR = 0.05


def load(stem: str, pred=None):
    z = np.load((pred or PRED) / f"{stem}.npz")
    return z["logits"].astype(np.float64), z["y"].astype(int)


def fig_rc(stems, labels, T: float, out: pathlib.Path, pred=None) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for stem, lab in zip(stems, labels):
        logits, y = load(stem, pred)
        p = M.softmax(logits, T)
        conf, correct = p.max(1), (p.argmax(1) == y).astype(float)
        cov, sel, gen, _ = M.rc_curve(conf, correct)
        m = cov >= COV_FLOOR
        a, g = M.aurc(conf, correct), M.augrc(conf, correct)
        ax1.plot(cov[m], sel[m] * 100, lw=1.6, label=f"{lab}  (AURC {a*1e3:.1f}e-3)")
        ax2.plot(cov[m], gen[m] * 100, lw=1.6, label=f"{lab}  (AUGRC {g*1e3:.1f}e-3)")
    for ax, name in ((ax1, "selective risk"), (ax2, "generalized risk")):
        ax.set_xlabel(f"coverage (floored at {COV_FLOOR:.0%})")
        ax.set_ylabel(f"{name}  (% error)")
        ax.set_title(name)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Risk-coverage, SID-Set validation slice -- threshold and temperature "
                 "frozen on clean calib", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  {out}")


def fig_conf(stems, labels, T: float, out: pathlib.Path, pred=None) -> None:
    fig, axes = plt.subplots(1, len(stems), figsize=(5.2 * len(stems), 4), sharey=True)
    axes = np.atleast_1d(axes)
    bins = np.linspace(0.5, 1.0, 40)
    for ax, stem, lab in zip(axes, stems, labels):
        logits, y = load(stem, pred)
        p = M.softmax(logits, T)
        conf, correct = p.max(1), (p.argmax(1) == y)
        ax.hist(conf[correct], bins=bins, alpha=0.65, label=f"correct (n={correct.sum()})",
                color="#2b7bba", density=True)
        ax.hist(conf[~correct], bins=bins, alpha=0.65, label=f"wrong (n={(~correct).sum()})",
                color="#d1495b", density=True)
        ax.set_title(f"{lab}\nAUROC(failure) {M.auroc_failure(conf, correct.astype(float)):.4f}",
                     fontsize=10)
        ax.set_xlabel("confidence (MSP)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle("Confidence by correctness -- an abstention rule works only while these "
                 "separate", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="tf_efficientnet_b0")
    ap.add_argument("--conditions", nargs="*", default=["clean", "ace_eps0.0005", "ace_eps0.002", "ace_eps0.005"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--pred-dir", type=pathlib.Path, default=PRED)
    a = ap.parse_args()

    calib_logits, calib_y = load(f"predictions_{a.model}_calib_clean", a.pred_dir)
    T = M.fit_temperature(calib_logits, calib_y)
    print(f"temperature {T:.4f} fitted on clean calib, frozen")

    stems = [f"predictions_{a.model}_{a.split}_{c}" for c in a.conditions]
    labels = [c.replace("ace_eps", "ACE eps=") for c in a.conditions]
    fig_rc(stems, labels, T, ROOT / "runs" / "fig1_risk_coverage.png", a.pred_dir)
    fig_conf([stems[0], stems[-1]], [labels[0], labels[-1]], T,
             ROOT / "runs" / "fig2_confidence_hist.png", a.pred_dir)


if __name__ == "__main__":
    main()
