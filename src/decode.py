"""Decode the manifest's rows out of the parquet shards into a flat JPEG cache.

Cache key is `uid` ("<source_split>:<img_id>"), never img_id -- see manifest.py for the
8,634-collision reason. On disk the ':' becomes '__'.

Why a cache at all: decode is the actual bottleneck of this pipeline (~128 img/s
single-thread against 263 img/s for the GPU forward), the shards are ~500 MB each, and
every condition (clean, JPEG q90/q50/q30, downscale, WebP) and every model re-reads the
same images. Decoding once turns a 22 GB read into a 1 GB one.

Cost of the cache, stated because it must appear in the README: short-side 256 + JPEG q95
is LOSSY. Frequency-domain detectors that key on compression artefacts cannot be evaluated
from it. The raw parquet is retained for exactly that reason.

Row groups are streamed one at a time so peak memory per worker stays ~60 MB rather than
the ~500 MB a whole shard would cost -- the box is already carrying 6 GB of swap.
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import pyarrow.parquet as pq
from PIL import Image

DATA = pathlib.Path("/home/samuel-renteria/Desktop/FILES/DATASETS/sid-set/data")
CACHE = pathlib.Path("/home/samuel-renteria/Desktop/FILES/DATASETS/sid-set/cache256")
SHORT_SIDE = 256
QUALITY = 95


def cache_path(uid: str, root: pathlib.Path = CACHE) -> pathlib.Path:
    return root / (uid.replace(":", "__") + ".jpg")


def _decode_shard(args) -> tuple[str, int, int, list[str]]:
    shard, wanted, root, force = args
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    want = dict(wanted)                      # img_id -> uid
    pf = pq.ParquetFile(DATA / shard)
    written, skipped, failed = 0, 0, []
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=["img_id", "image"])
        ids = t.column("img_id").to_pylist()
        for j, img_id in enumerate(ids):
            uid = want.get(img_id)
            if uid is None:
                continue
            out = cache_path(uid, root)
            if out.exists() and not force:
                skipped += 1
                continue
            try:
                raw = t.column("image")[j]["bytes"].as_py()
                im = Image.open(io.BytesIO(raw))
                im = im.convert("RGB")       # handles PNG, RGBA, palette, greyscale
                w, h = im.size
                s = SHORT_SIDE / min(w, h)
                if s < 1.0:                  # never upscale: it invents detail
                    im = im.resize((max(1, round(w * s)), max(1, round(h * s))),
                                   Image.LANCZOS)
                im.save(out, "JPEG", quality=QUALITY)
                written += 1
            except Exception as e:           # noqa: BLE001 - record, never abort the batch
                failed.append(f"{uid}: {type(e).__name__}: {e}")
    return shard, written, skipped, failed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    root = pathlib.Path(__file__).resolve().parents[1]
    ap.add_argument("--manifest", type=pathlib.Path, default=root / "runs" / "manifest_v1.parquet")
    ap.add_argument("--splits", nargs="*", default=["fit", "calib", "test"])
    ap.add_argument("--cache", type=pathlib.Path, default=CACHE)
    ap.add_argument("--workers", type=int, default=4)     # capped: the box is swapping
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    t = pq.read_table(a.manifest, columns=["uid", "img_id", "shard", "split"])
    by_shard: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for uid, img_id, shard, split in zip(t.column("uid").to_pylist(),
                                         t.column("img_id").to_pylist(),
                                         t.column("shard").to_pylist(),
                                         t.column("split").to_pylist()):
        if split in a.splits:
            by_shard[shard].append((img_id, uid))
    total = sum(len(v) for v in by_shard.values())
    print(f"{total} images across {len(by_shard)} shards -> {a.cache}")

    jobs = [(s, v, str(a.cache), a.force) for s, v in sorted(by_shard.items())]
    t0 = time.time()
    written = skipped = 0
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (shard, w, sk, fail) in enumerate(ex.map(_decode_shard, jobs), 1):
            written += w
            skipped += sk
            failures += fail
            el = time.time() - t0
            print(f"  [{i:>2}/{len(jobs)}] {shard:<34} +{w:<5} skip {sk:<5} "
                  f"{written / max(el, 1e-9):6.1f} img/s", flush=True)

    el = time.time() - t0
    report = {"written": written, "skipped": skipped, "failed": len(failures),
              "seconds": round(el, 1), "img_per_s": round(written / max(el, 1e-9), 1),
              "short_side": SHORT_SIDE, "quality": QUALITY, "workers": a.workers,
              "failures": failures[:20]}
    (root / "runs" / "decode_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    if failures:
        print(f"FAILURES ({len(failures)}), first few:")
        for f in failures[:5]:
            print("  ", f)


if __name__ == "__main__":
    main()
