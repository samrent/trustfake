"""Build the SID-Set split manifest: (uid, img_id, shard, label, y_binary, split).

THE KEY IS `uid`, NOT `img_id`. Measured on the shards on disk: 8,634 img_ids appear in
BOTH the train and the validation split -- 4,496 synthetic and 4,138 tampered -- because
those two classes are numbered sequentially and the counter restarts per split
(full_synthetic_000155 exists in each, with different bytes: verified byte-hash distinct,
0/6 identical on inspection). The real class is content-hash named and has ZERO
collisions, which is why the problem is invisible if you spot-check reals.

Nothing errors when you get this wrong. A flat feature cache keyed on img_id silently
overwrites 8,634 fit-split images with test-split images, and a join between two npz files
keyed on img_id silently merges the wrong rows. So the global key is
    uid = "<source_split>:<img_id>"    e.g. "train:full_synthetic_000155"
img_id is retained for traceability back to the dataset, and never used as a key.

The leakage firewall is STRUCTURAL, not disciplinary:

  fit    rows come only from train-*.parquet shards
  calib  rows come only from validation-* shards
  test   rows come only from validation-* shards, and never a shard used by calib

Fitting the probe, fitting the temperature and reporting the numbers therefore cannot
touch the same images even by mistake. Fitting the temperature on the reported split is
the classic self-own and the first thing a reviewer checks; here it is impossible without
editing this file.

Naming discipline: the official SID-Set test split is withheld by the authors (SIDA repo,
to prevent leakage). Everything called 'test' here is carved from the VALIDATION split.
The constant SPLIT_PROVENANCE below is what every report and slide must print. Never
write 'SID-Set test set'.

Class balance: SID-Set is one real class against two fake classes, so the natural binary
prior is 1:2. We keep ALL real rows and sample an equal number of fakes, half label 1
(fully synthetic) and half label 2 (tampered), giving a 50/50 binary prior. The policy is
recorded in the sidecar because the base rate sets what accuracy and F1 mean, and the
error rate sets the random-ranker AURC anchor.

The `label` column survives into the manifest on purpose: label 1 (fully synthetic) is the
easy case and label 2 (tampered, local edits) is the hard one, so the clean risk-coverage
curve must be breakable out by class. A flat curve on the pooled set is usually label 1
saturating.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATA = pathlib.Path("/home/samuel-renteria/Desktop/FILES/DATASETS/sid-set/data")
SPLIT_PROVENANCE = "SID-Set validation split, held-out slice (official test split withheld by authors)"
LABEL_NAMES = {0: "real", 1: "full_synthetic", 2: "tampered"}

PROFILES = {
    "smoke": dict(fit=2, calib=1, test=2),
    "full": dict(fit=12, calib=8, test=26),
    # WP3 trains weights and needs more than a linear probe does. Enlarging `fit` leaves
    # calib and test BYTE-IDENTICAL (same seed, same validation-shard permutation, which
    # does not depend on the train-shard count) -- verified by uid-hash on 2026-08-19.
    # That is what lets the defense comparison scale its training set without moving the
    # splits every earlier number was reported on.
    "train": dict(fit=30, calib=8, test=26),
}


def _shards(kind: str) -> list[pathlib.Path]:
    return sorted(DATA.glob(f"{kind}-*.parquet"))


def _read_index(paths: list[pathlib.Path]) -> dict:
    """Read (img_id, label) only -- parquet is columnar, so the image bytes never load."""
    ids, labels, shards = [], [], []
    for p in paths:
        t = pq.read_table(p, columns=["img_id", "label"])
        ids.extend(t.column("img_id").to_pylist())
        labels.extend(t.column("label").to_pylist())
        shards.extend([p.name] * t.num_rows)
    return dict(img_id=ids, label=np.asarray(labels, dtype=np.int64), shard=shards)


def _balance(idx: dict, rng: np.random.Generator, policy: str) -> np.ndarray:
    """Return row positions to keep. 'binary_5050': all real + equal fakes, half per fake class."""
    label = idx["label"]
    real = np.flatnonzero(label == 0)
    if policy == "none":
        return np.arange(label.size)
    if policy != "binary_5050":
        raise ValueError(f"unknown balance policy {policy}")
    n_real = real.size
    per_class = n_real // 2
    keep = [real]
    for fake_label in (1, 2):
        pool = np.flatnonzero(label == fake_label)
        take = min(per_class, pool.size)
        keep.append(rng.choice(pool, size=take, replace=False))
    return np.sort(np.concatenate(keep))


def _count_img_id_collisions() -> int:
    """img_id-column-only scan of every shard on disk; ~1-2 min, run once per manifest build."""
    train, val = set(), set()
    for f in sorted(DATA.glob("train-*.parquet")):
        train.update(pq.ParquetFile(f).read(columns=["img_id"]).column("img_id").to_pylist())
    for f in sorted(DATA.glob("validation-*.parquet")):
        val.update(pq.ParquetFile(f).read(columns=["img_id"]).column("img_id").to_pylist())
    return len(train & val)


def build(profile: str, seed: int, balance: str, out: pathlib.Path) -> dict:
    train_shards, val_shards = _shards("train"), _shards("validation")
    n = PROFILES[profile]
    if len(train_shards) < n["fit"] or len(val_shards) < n["calib"] + n["test"]:
        sys.exit(f"not enough shards on disk: have {len(train_shards)} train / "
                 f"{len(val_shards)} validation, need {n['fit']} / {n['calib'] + n['test']}")

    rng = np.random.default_rng(seed)
    val_order = rng.permutation(len(val_shards))
    assign = {
        "fit": train_shards[: n["fit"]],
        "calib": [val_shards[i] for i in val_order[: n["calib"]]],
        "test": [val_shards[i] for i in val_order[n["calib"]: n["calib"] + n["test"]]],
    }

    # ---- structural assertions: these are the firewall, not comments about it
    assert all(p.name.startswith("train-") for p in assign["fit"])
    assert all(p.name.startswith("validation-") for p in assign["calib"] + assign["test"])
    assert not (set(p.name for p in assign["calib"]) & set(p.name for p in assign["test"])), \
        "calib and test share a shard -- temperature would be fitted on reported data"

    rows = {k: [] for k in ("uid", "img_id", "shard", "label", "y_binary", "split")}
    counts = {}
    for si, (split, paths) in enumerate(assign.items()):
        idx = _read_index(paths)
        keep = _balance(idx, np.random.default_rng(seed + 1000 + si), balance)
        lab = idx["label"][keep]
        source = "train" if split == "fit" else "validation"
        rows["uid"].extend([f"{source}:{idx['img_id'][i]}" for i in keep])
        rows["img_id"].extend([idx["img_id"][i] for i in keep])
        rows["shard"].extend([idx["shard"][i] for i in keep])
        rows["label"].extend(lab.tolist())
        rows["y_binary"].extend((lab > 0).astype(np.int64).tolist())
        rows["split"].extend([split] * keep.size)
        counts[split] = {
            "n": int(keep.size),
            "shards": [p.name for p in paths],
            "by_label": {LABEL_NAMES[k]: int((lab == k).sum()) for k in (0, 1, 2)},
            "prior_fake": float((lab > 0).mean()),
        }

    assert len(set(rows["uid"])) == len(rows["uid"]), "duplicate uid -- the key is broken"
    n_colliding = len(rows["img_id"]) - len(set(rows["img_id"]))

    table = pa.table({
        "uid": pa.array(rows["uid"], pa.string()),
        "img_id": pa.array(rows["img_id"], pa.string()),
        "shard": pa.array(rows["shard"], pa.string()),
        "label": pa.array(rows["label"], pa.int8()),
        "y_binary": pa.array(rows["y_binary"], pa.int8()),
        "split": pa.array(rows["split"], pa.string()),
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()

    sidecar = {
        "manifest_sha256": sha,
        "rows": table.num_rows,
        "profile": profile,
        "seed": seed,
        "balance_policy": balance,
        "split_provenance": SPLIT_PROVENANCE,
        "leakage_rule": "fit from train shards only; calib and test from disjoint validation shards",
        "key": "uid = '<source_split>:<img_id>'; img_id alone is NOT unique across splits",
        "img_id_collisions_in_manifest": n_colliding,
        # Computed, never asserted: this figure was once a hard-coded 8634, counted while
        # only half the train shards were on disk, and nothing in the repo recomputed it.
        # True count over all 283 shards: 19,107 (10,000 synthetic + 9,107 tampered) --
        # every one of the 10,000 fully-synthetic validation img_ids collides with train.
        "img_id_collisions_in_full_dataset": _count_img_id_collisions(),
        "cache_policy": "short-side 256, JPEG q95 -- lossy, unsuitable for frequency-domain "
                        "detectors; raw parquet retained for that reason",
        "splits": counts,
        "regenerate": f"python -m src.manifest --profile {profile} --seed {seed} "
                      f"--balance {balance} --out {out}",
    }
    side = out.with_suffix(".json")
    side.write_text(json.dumps(sidecar, indent=2) + "\n")
    return sidecar


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=list(PROFILES), default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--balance", choices=["binary_5050", "none"], default="binary_5050")
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parents[1] / "runs" / "manifest_v1.parquet")
    a = ap.parse_args()
    s = build(a.profile, a.seed, a.balance, a.out)
    print(json.dumps({k: v for k, v in s.items() if k != "splits"}, indent=2))
    for split, c in s["splits"].items():
        print(f"  {split:<6} n={c['n']:>6}  {c['by_label']}  prior_fake={c['prior_fake']:.3f}  "
              f"{len(c['shards'])} shards")


if __name__ == "__main__":
    main()
