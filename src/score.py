"""Score prediction npz files into the WP1 condition table.

Temperature is FIT ON CALIB and FROZEN, then applied to test. Re-fitting anything on the
reported split -- temperature, threshold, or the coverage sweep -- is an oracle and
understates every collapse. The manifest makes calib and test shard-disjoint so this
cannot happen by accident.

Every row prints accuracy BESIDE AURC. Without it the central WP4 claim is unfalsifiable:
a confidence attack leaves accuracy untouched and wrecks AURC, but PGD also wrecks AURC --
by destroying accuracy. Only the pair distinguishes them.

n_op is the number of distinct confidence values. If it collapses far below n, tie blocks
are being manufactured (fp16 saturation) and AURC is no longer a property of the model.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

import metrics as M

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRED = ROOT / "runs" / "predictions"

COLS = [("condition", 22), ("n", 6), ("acc", 7), ("F1", 7), ("AUROC_det", 10),
        ("AUROC_fail", 11), ("AURC_e-3", 9), ("AUGRC_e-3", 10), ("E-AURC_e-3", 11),
        ("ECE", 7), ("risk@.8", 8), ("risk@.5", 8), ("n_op", 7)]


def load(stem: str, pred=None) -> dict:
    pred = pred or PRED
    z = np.load(pred / f"{stem}.npz", allow_pickle=False)
    meta = json.loads((pred / f"{stem}.json").read_text())
    return {"logits": z["logits"].astype(np.float64), "y": z["y"].astype(int),
            "label": z["label"].astype(int), "uid": z["uid"], "meta": meta}


def row(name: str, logits, y, T: float) -> dict:
    s = M.summary(logits, y, temperature=T)
    s["condition"] = name
    return s


def fmt(rows: list[dict]) -> str:
    head = " ".join(f"{c:>{w}}" for c, w in COLS)
    lines = [head, "-" * len(head)]
    for r in rows:
        vals = {
            "condition": r["condition"], "n": r["n"], "acc": f"{r['accuracy']:.4f}",
            "F1": f"{r['f1']:.4f}", "AUROC_det": f"{r['auroc_detection']:.4f}",
            "AUROC_fail": f"{r['auroc_failure']:.4f}",
            "AURC_e-3": f"{r['aurc']*1e3:.2f}", "AUGRC_e-3": f"{r['augrc']*1e3:.2f}",
            "E-AURC_e-3": f"{r['eaurc']*1e3:.2f}", "ECE": f"{r['ece_15_equal_mass']:.4f}",
            "risk@.8": f"{r['risk@cov0.8']*100:.2f}%", "risk@.5": f"{r['risk@cov0.5']*100:.2f}%",
            "n_op": r["n_operating_points"],
        }
        lines.append(" ".join(f"{str(vals[c]):>{w}}" for c, w in COLS))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="vit_large_patch14_clip_224")
    ap.add_argument("--calib", default=None, help="stem of the calibration npz (for temperature)")
    ap.add_argument("--conditions", nargs="*", default=["clean"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "runs" / "condition_table.md")
    ap.add_argument("--pred-dir", type=pathlib.Path, default=PRED)
    a = ap.parse_args()

    calib_stem = a.calib or f"predictions_{a.model}_calib_clean"
    calib = load(calib_stem, a.pred_dir)
    T = M.fit_temperature(calib["logits"], calib["y"])
    print(f"temperature fitted on {calib_stem} (n={calib['y'].size}): T = {T:.4f}   "
          f"[frozen; applied to every row below]")

    rows, extra = [], []
    for cond in a.conditions:
        d = load(f"predictions_{a.model}_{a.split}_{cond}", a.pred_dir)
        r0 = row(cond, d["logits"], d["y"], T)
        r0["batch"] = d["meta"].get("batch")
        rows.append(r0)
        rows.append(row(f"{cond} (T=1)", d["logits"], d["y"], 1.0))
        if cond == "clean":
            for k, nm in ((1, "real vs synthetic"), (2, "real vs tampered")):
                m = (d["label"] == 0) | (d["label"] == k)
                extra.append(row(nm, d["logits"][m], d["y"][m], T))

    batches = {r["condition"]: r.get("batch") for r in rows if r.get("batch") is not None}
    warn = ""
    if len(set(batches.values())) > 1:
        warn = ("!! rows were produced at DIFFERENT batch sizes " + str(batches) +
                " -- cuDNN is not bit-identical across batch shapes, so a one-sample drift "
                "can masquerade as an effect. Regenerate at a single batch size.")
        print("\n" + warn)

    table = fmt(rows)
    print("\n" + table)
    if extra:
        print("\nclean, broken out by fake class (each vs the same real pool):")
        print(fmt(extra))

    a.out.write_text(
        f"# WP1 condition table -- {a.model}\n\n"
        f"- split: `{a.split}`, n = {rows[0]['n']}\n"
        f"- provenance: SID-Set validation split, held-out slice (official test split withheld)\n"
        f"- temperature {T:.4f} fitted on `{calib_stem}` and frozen\n"
        f"- AURC/AUGRC convention: tie blocks collapsed, block-size-weighted mean over operating points\n"
        f"- ECE column: 15 bins, equal-mass\n\n"
        f"```\n{table}\n```\n\n" + (f"> {warn}\n\n" if warn else "")
        + (f"clean, broken out by fake class:\n\n```\n{fmt(extra)}\n```\n" if extra else ""))
    (a.out.with_suffix(".json")).write_text(json.dumps(
        {"temperature": T, "rows": rows, "by_class": extra}, indent=2, default=float) + "\n")
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
