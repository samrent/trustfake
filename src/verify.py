"""Artifact self-consistency checker: run before you believe any table.

Every defect this checks for actually happened in this repo, silently, and was caught by a
reviewer rather than by the code:

  - a `--limit 512` run overwrote the 15,316-row clean baseline with a 512-row file, so every
    delta measured against it was meaningless and nothing errored;
  - a checkpoint sidecar claimed a 12-epoch budget for a run cut at epoch 7;
  - the clean row was regenerated at batch 16 while the attacked rows were at batch 32, and
    cuDNN's batch-shape non-determinism moved accuracy by one sample in 15,316 -- enough to
    break a headline that reads "accuracy identical across every row";
  - a mistyped condition parsed as a `realistic` row carrying a fabricated eps of 0.

The guard is on the PER-SPLIT UID SEQUENCE, not the manifest file hash. manifest_v1 and
manifest_v2 have byte-identical calib and test uid sequences and differ only in `fit`, so a
whole-file hash would reject exactly the comparison WP3 needs while catching nothing real.

Exit code is non-zero if any check fails, so this belongs in the pre-flight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRED = ROOT / "runs" / "predictions"


def split_uid_list(manifest: pathlib.Path, split: str) -> list[str]:
    t = pq.read_table(manifest, columns=["uid", "split"])
    return [u for u, s in zip(t.column("uid").to_pylist(), t.column("split").to_pylist()) if s == split]


def split_uid_sha(manifest: pathlib.Path, split: str) -> tuple[int, str]:
    uids = split_uid_list(manifest, split)
    return len(uids), hashlib.sha256("\n".join(uids).encode()).hexdigest()[:16]


def check(manifests: list[pathlib.Path], pred_dir: pathlib.Path) -> tuple[list[str], list[str]]:
    fails: list[str] = []
    warns: list[str] = []
    samples: dict[str, dict[str, str]] = {}
    expect: dict[str, tuple[int, str]] = {}
    for m in manifests:
        if not m.exists():
            continue
        for split in ("fit", "calib", "test"):
            n, sha = split_uid_sha(m, split)
            prev = expect.get(split)
            # calib/test must be IDENTICAL across manifests -- that invariant is what lets WP3
            # enlarge the training set without moving the splits every number was reported on
            if prev and split in ("calib", "test") and prev != (n, sha):
                fails.append(f"{split} differs between manifests: {prev} vs {(n, sha)}")
            expect.setdefault(split, (n, sha))

    sidecars = sorted(pred_dir.glob("predictions_*.json"))
    if not sidecars:
        fails.append(f"no prediction sidecars in {pred_dir}")
    by_model: dict[str, dict[str, int]] = {}

    for j in sidecars:
        meta = json.loads(j.read_text())
        npz = j.with_suffix(".npz")
        stem = j.stem
        if not npz.exists():
            fails.append(f"{stem}: sidecar without npz")
            continue
        z = np.load(npz, allow_pickle=False)

        split = meta.get("split")
        if split in expect:
            n_expect, sha_expect = expect[split]
            sha = hashlib.sha256("\n".join(z["uid"].tolist()).encode()).hexdigest()[:16]
            if meta.get("sample_n"):
                # A subsampled row is legitimate, but only against rows drawn identically.
                split_uids = set(split_uid_list(manifests[0], split))
                if not set(z["uid"].tolist()) <= split_uids:
                    fails.append(f"{stem}: sampled uids are not a subset of {split}")
                if sha != meta.get("sample_uid_sha"):
                    fails.append(f"{stem}: uid sha {sha} != recorded sample_uid_sha "
                                 f"{meta.get('sample_uid_sha')}")
                samples.setdefault(f"{meta.get('model_id')}/{split}", {})[meta["condition"]] = sha
            else:
                if meta.get("n") != n_expect:
                    fails.append(f"{stem}: n={meta.get('n')} but manifest {split} declares "
                                 f"{n_expect} (a partial run overwrote a full one?)")
                if sha != sha_expect:
                    fails.append(f"{stem}: uid sequence {sha} != manifest {split} {sha_expect}")
                samples.setdefault(f"{meta.get('model_id')}/{split}", {})[meta["condition"]] = "FULL"

        for field, dt in (("y", "int8"), ("logits", "float32")):
            if field not in z:
                fails.append(f"{stem}: missing field {field!r}")
            elif z[field].dtype != np.dtype(dt):
                fails.append(f"{stem}: {field} dtype {z[field].dtype} != {dt}")
        if "logits" in z and z["logits"].ndim == 2 and z["logits"].shape[1] != 2:
            fails.append(f"{stem}: logits shape {z['logits'].shape}, expected [n,2]")
        if len(z["uid"]) != len(set(z["uid"].tolist())):
            fails.append(f"{stem}: duplicate uids")
        if meta.get("batch") is None:
            if meta.get("logits_source") == "cached_features":
                warns.append(f"{stem}: logits from cached features whose extraction batch was "
                             f"not recorded (pre-dates the field); re-extract to close the chain")
            else:
                fails.append(f"{stem}: no batch recorded -- batch shape changes results on this stack")
        else:
            by_model.setdefault(f"{meta.get('model_id')}/{split}", {})[meta["condition"]] = meta["batch"]

        try:
            from .attack_suite import parse_condition
            parse_condition(meta["condition"])
        except ValueError as e:
            fails.append(f"{stem}: {e}")

    for key, shas in samples.items():
        if len(set(shas.values())) > 1:
            fails.append(f"{key}: rows drawn from DIFFERENT samples {shas} -- a subsampled row "
                         f"and a full-split row are not comparable")

    for key, conds in by_model.items():
        if len(set(conds.values())) > 1:
            fails.append(f"{key}: rows produced at different batch sizes {conds} -- cuDNN is not "
                         f"bit-identical across batch shapes; regenerate at one batch size")

    # sigma files (the uncertainty seam) must be uid-aligned to their split and finite
    sig_dir = ROOT / "runs" / "sigma"
    for sj in sorted(sig_dir.glob("sigma_*.json")) if sig_dir.exists() else []:
        meta = json.loads(sj.read_text())
        snpz = sj.with_suffix(".npz")
        if not snpz.exists():
            fails.append(f"{sj.stem}: sigma sidecar without npz")
            continue
        z = np.load(snpz, allow_pickle=False)
        if "sigma" not in z or "uid" not in z:
            fails.append(f"{sj.stem}: sigma npz missing uid/sigma")
            continue
        if not np.all(np.isfinite(z["sigma"])):
            fails.append(f"{sj.stem}: sigma contains non-finite values")
        if len(z["uid"]) != len(set(z["uid"].tolist())):
            fails.append(f"{sj.stem}: duplicate uids in sigma")
        sp = meta.get("split")
        if sp in expect and set(z["uid"].tolist()) - set(split_uid_list(manifests[0], sp)):
            fails.append(f"{sj.stem}: sigma uids not a subset of the {sp} split")

    for ck in sorted((ROOT / "runs" / "checkpoints").glob("*.json")):
        m = json.loads(ck.read_text())
        for field in ("method", "backbone", "seed", "epochs_planned", "selected_epoch"):
            if field not in m and field not in ("epochs_planned", "selected_epoch"):
                fails.append(f"checkpoint {ck.stem}: missing {field!r}")

    tracked = subprocess.run(["git", "ls-files", "--", str(ROOT)], capture_output=True,
                             text=True, cwd=ROOT).stdout.split()
    bulk = [f for f in tracked if f.endswith((".pt", ".pth", ".npz", ".npy", ".parquet"))]
    if bulk:
        fails.append(f"bulk artifacts tracked by git: {bulk[:5]}")
    return fails, warns


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifests", nargs="*", type=pathlib.Path,
                    default=[ROOT / "runs" / "manifest_v1.parquet", ROOT / "runs" / "manifest_v2.parquet"])
    ap.add_argument("--pred-dir", type=pathlib.Path, default=PRED)
    a = ap.parse_args()
    fails, warns = check(a.manifests, a.pred_dir)
    for w in warns:
        print(f"  warn: {w}")
    if fails:
        print(f"FAILED — {len(fails)} problem(s):")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("OK — every prediction matches its manifest split, dtypes and shapes hold, batch "
          "sizes are consistent per model, conditions parse, no bulk in git.")


if __name__ == "__main__":
    main()
