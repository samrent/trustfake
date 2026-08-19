#!/usr/bin/env python3
"""Fetch SID-Set shards from HuggingFace and verify them before boarding.

    saberzl/SID_Set — public, ungated, cc-by-4.0
      data/train-000NN-of-00249.parquet        249 shards, 123.2 GB
      data/validation-000NN-of-00034.parquet    34 shards,  16.8 GB
      test: NOT on HF. 60k images withheld to prevent leakage; request via
            github.com/hzlsaber/SIDA. Never label validation numbers "test set".

Why this file exists rather than a shell one-liner: a truncated parquet
discovered in Lausanne is unfixable, so every shard is opened and counted here,
on home wifi, and the counts are written to a manifest. Resumable — re-running
skips shards already present and verified.

    python fetch_sid_set.py validation           # 34 shards, 16.8 GB
    python fetch_sid_set.py train --shards 20    # first 20 train shards
    python fetch_sid_set.py verify               # re-verify what is on disk
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

REPO = "saberzl/SID_Set"
DEST = Path.home() / "Desktop/FILES/DATASETS/sid-set"
MANIFEST = DEST / "manifest.json"
LABELS = {0: "real", 1: "synthetic", 2: "tampered"}


def load_manifest() -> dict:
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text())
        except json.JSONDecodeError:
            pass
    return {"shards": {}}


def verify_shard(path: Path) -> dict:
    """Open the shard for real. Metadata alone would not catch a truncated file."""
    pf = pq.ParquetFile(path)
    rows = pf.metadata.num_rows
    labels: Counter = Counter()
    for batch in pf.iter_batches(columns=["label"], batch_size=4096):
        labels.update(batch.column("label").to_pylist())
    return {
        "bytes": path.stat().st_size,
        "rows": rows,
        "row_groups": pf.metadata.num_row_groups,
        "labels": {LABELS.get(k, str(k)): v for k, v in sorted(labels.items())},
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def fetch(split: str, n_shards: int | None) -> list[str]:
    total = {"train": 249, "validation": 34}[split]
    want = total if n_shards is None else min(n_shards, total)
    patterns = [f"data/{split}-{i:05d}-of-{total:05d}.parquet" for i in range(want)]
    print(f"[fetch] {split}: {want}/{total} shards -> {DEST}", flush=True)
    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=str(DEST),
        max_workers=4,
    )
    return patterns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["train", "validation", "verify"])
    ap.add_argument("--shards", type=int, default=None, help="first N shards (default: all)")
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    man = load_manifest()

    if args.split != "verify":
        t0 = time.time()
        fetch(args.split, args.shards)
        print(f"[fetch] download finished in {time.time() - t0:.0f}s", flush=True)

    # Verify every parquet on disk, not just the ones just fetched.
    files = sorted(DEST.rglob("*.parquet"))
    print(f"[verify] {len(files)} shards on disk", flush=True)
    bad, total_rows = [], 0
    agg: Counter = Counter()
    for i, f in enumerate(files, 1):
        key = f.name
        prior = man["shards"].get(key)
        if prior and prior.get("bytes") == f.stat().st_size and "rows" in prior:
            info = prior  # already verified at this exact size
        else:
            try:
                info = verify_shard(f)
            except Exception as e:  # truncated / corrupt — the whole point of this pass
                print(f"  [{i}/{len(files)}] {key}  CORRUPT: {e}", flush=True)
                bad.append(key)
                man["shards"][key] = {"bytes": f.stat().st_size, "error": str(e)}
                continue
            man["shards"][key] = info
            MANIFEST.write_text(json.dumps(man, indent=1))
        total_rows += info["rows"]
        agg.update(info["labels"])
        print(f"  [{i}/{len(files)}] {key}  {info['rows']} rows  {info['labels']}", flush=True)

    MANIFEST.write_text(json.dumps(man, indent=1))
    print(f"\n[done] {len(files) - len(bad)}/{len(files)} shards good, {total_rows} rows", flush=True)
    print(f"[done] labels: {dict(agg)}", flush=True)
    if bad:
        print(f"[done] CORRUPT ({len(bad)}): {bad}", flush=True)
        print("[done] delete those files and re-run to refetch.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
