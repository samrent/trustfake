"""Fit the linear probe on frozen features and emit the WP1 prediction contract.

    predictions_{model}_{split}_{condition}.npz
        uid     (str)   THE key: "<source_split>:<img_id>". Join on this, never img_id.
        img_id  (str)   dataset id, for traceability only -- not unique across splits
        y       (int8)  binary target: 0 real, 1 fake (synthetic or tampered)
        label   (int8)  original 3-class label, so results break out by difficulty
        logits  (float32 [n, 2])
    + JSON sidecar {model_id, manifest_sha, condition, seed, timestamp, ...}

`condition` is required from line one. With it, WP2 and WP3 hand over arrays from attacked
or robustly-trained models and WP1 scores them without touching their code. Any function
that assumes clean inputs has to be torn up mid-camp.

Binary logits are written as [0, d] where d is the probe's decision margin: softmax([0,d])
gives exactly sigmoid(d) = p(fake), so the 2-column form loses nothing and every consumer
sees one shape. C is chosen by cross-validation INSIDE the fit split only.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, DEFAULT_MODEL


def _load(model_id: str, split: str, condition: str, fdir=FEATURES):
    stem = f"{model_id.split('.')[0]}_{split}_{condition}"
    x = np.load(fdir / f"{stem}.npy")
    meta = json.loads((fdir / f"{stem}.json").read_text())
    return x, np.asarray(meta["y_binary"], dtype=np.int8), np.asarray(meta["label"], dtype=np.int8), meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    root = pathlib.Path(__file__).resolve().parents[1]
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--splits", nargs="*", default=["calib", "test"])
    ap.add_argument("--condition", default="clean")
    ap.add_argument("--fit-condition", default="clean",
                    help="condition the probe is FIT on; keep 'clean' so robustness is measured, not trained")
    ap.add_argument("--manifest", type=pathlib.Path, default=root / "runs" / "manifest_v1.parquet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--features-dir", type=pathlib.Path, default=FEATURES)
    ap.add_argument("--pred-dir", type=pathlib.Path, default=root / "runs" / "predictions")
    a = ap.parse_args()

    man = pq.read_table(a.manifest, columns=["uid", "img_id"])
    uid2img = dict(zip(man.column("uid").to_pylist(), man.column("img_id").to_pylist()))
    manifest_sha = json.loads(a.manifest.with_suffix(".json").read_text())["manifest_sha256"]

    xf, yf, _, mf = _load(a.model, "fit", a.fit_condition, a.features_dir)
    scaler = StandardScaler().fit(xf)
    xfs = scaler.transform(xf)

    grid = [0.001, 0.01, 0.1, 1.0, 10.0]
    cv = StratifiedKFold(5, shuffle=True, random_state=a.seed)
    scores = [cross_val_score(LogisticRegression(C=c, max_iter=2000), xfs, yf, cv=cv).mean()
              for c in grid]
    best = grid[int(np.argmax(scores))]
    print("  C grid (5-fold CV inside the fit split): " +
          "  ".join(f"{c}={s:.4f}" for c, s in zip(grid, scores)) + f"   -> C={best}")

    clf = LogisticRegression(C=best, max_iter=5000).fit(xfs, yf)
    print(f"  probe fit on {xf.shape[0]} images, {xf.shape[1]}-d, train acc {clf.score(xfs, yf):.4f}")

    outdir = a.pred_dir
    outdir.mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        x, y, lab, meta = _load(a.model, split, a.condition, a.features_dir)
        d = clf.decision_function(scaler.transform(x))
        logits = np.stack([np.zeros_like(d), d], axis=1).astype(np.float32)
        uids = meta["uids"]
        stem = f"predictions_{a.model.split('.')[0]}_{split}_{a.condition}"
        np.savez(outdir / f"{stem}.npz",
                 uid=np.array(uids), img_id=np.array([uid2img[u] for u in uids]),
                 y=y.astype(np.int8), label=lab.astype(np.int8), logits=logits)
        (outdir / f"{stem}.json").write_text(json.dumps({
            "model_id": f"{a.model}+linear_probe", "manifest_sha": manifest_sha,
            "condition": a.condition, "split": split, "seed": a.seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n": int(y.size), "probe_C": best, "fit_condition": a.fit_condition,
            "fit_n": int(xf.shape[0]), "feature_dim": int(xf.shape[1]),
            "key": "uid", "logit_convention": "[0, margin]; softmax -> p(fake) = sigmoid(margin)",
        }, indent=2) + "\n")
        acc = ((d > 0).astype(np.int8) == y).mean()
        print(f"  {stem}.npz  n={y.size}  accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
