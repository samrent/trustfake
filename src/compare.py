"""WP3 deliverable D3: the comparative table across defenses x conditions.

The proposal's question is not "is the robust model more accurate" -- it is whether robust
training buys robustness WITHOUT degrading confidence reliability and selective behaviour.
Those can move in opposite directions, so the table carries both axes side by side:

    accuracy / robust accuracy   the label axis  (what adversarial training targets)
    AUROC(failure), AURC         the confidence axis (what the moderation layer depends on)

Each model's temperature is fitted on ITS OWN clean calibration split and frozen before any
attacked row is scored. Fitting per model is required -- a temperature from another model is
meaningless -- and freezing before the attack is required, because recomputing it on attacked
data is an oracle that understates every collapse.

All three arms share one initialisation and one 12-epoch schedule, so a difference between
rows is a difference of METHOD, not of budget.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from . import metrics as M

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRED = ROOT / "runs" / "predictions"

CONDITIONS = [
    ("clean", "clean"),
    ("jpeg_q50", "realistic: JPEG q50"),
    ("downscale_0.5", "realistic: 0.5x resize"),
    ("pgd_linf_eps0.00784_s10", "adversarial: PGD 2/255 (label)"),
    ("ace_uint8_eps0.005", "adversarial: ACE uint8 (confidence)"),
    ("overconf_eps0.0157_s20", "adversarial: over-confidence (label-free)"),
]


def load(stem: str, pred: pathlib.Path):
    z = np.load(pred / f"{stem}.npz", allow_pickle=False)
    meta = json.loads((pred / f"{stem}.json").read_text())
    return z["logits"].astype(np.float64), z["y"].astype(int), meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*",
                    default=["effb0_standard_eps2_255", "effb0_at_pgd_eps2_255", "effb0_trades_eps2_255"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--pred-dir", type=pathlib.Path, default=PRED)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "runs" / "d3_comparative.md")
    a = ap.parse_args()

    hdr = f"{'condition':<34}" + "".join(f"{m.replace('effb0_',''):>26}" for m in a.models)
    lines, rows = [hdr, "-" * len(hdr)], {}

    temps = {}
    for m in a.models:
        cl, cy, _ = load(f"predictions_{m}_calib_clean", a.pred_dir)
        temps[m] = M.fit_temperature(cl, cy)

    for cond, label in CONDITIONS:
        cells = []
        for m in a.models:
            try:
                z, y, _ = load(f"predictions_{m}_{a.split}_{cond}", a.pred_dir)
            except FileNotFoundError:
                cells.append(f"{'--':>26}")
                continue
            s = M.summary(z, y, temperature=temps[m])
            rows[(m, cond)] = s
            cells.append(f"{s['accuracy']:>7.4f} {s['auroc_failure']:>8.4f} {s['aurc']*1e3:>8.1f}")
        lines.append(f"{label:<34}" + "".join(cells))

    body = ("\n".join(lines) +
            "\n\ncells: accuracy | AUROC(failure) | AURC x10^-3     "
            + "  ".join(f"T({m.replace('effb0_','')})={temps[m]:.3f}" for m in a.models))
    print(body)

    a.out.write_text(
        "# WP3 / D3 — comparative results\n\n"
        "All three arms: one shared initialisation (`phase1_init.pt`), 12 epochs, identical data\n"
        "protocol and identical evaluation. Differences are of METHOD, not of budget.\n"
        "Each model's temperature is fitted on its OWN clean calib split and frozen before any\n"
        "attacked row is scored.\n\n"
        f"```\n{body}\n```\n")
    (a.out.with_suffix(".json")).write_text(json.dumps(
        {"temperatures": temps,
         "rows": {f"{m}|{c}": v for (m, c), v in rows.items()}}, indent=2, default=float) + "\n")
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
