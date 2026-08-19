"""Trivial baselines that require no model, and the reason every table must print them.

SID-Set's binary task carries a geometry artifact. Measured on the exact test split
(n=15,316, 50/50 prior):

    width == height  -> fake      accuracy 0.9785
    original was PNG -> fake      accuracy 0.7673
    majority class                accuracy 0.5000

Label 1 (fully synthetic) is 100% square 1024x1024 PNG and label 2 (tampered) is 100% square;
only ~5% of real images are square. So a one-line rule with no learning in it scores 0.9785 --
above this project's CLIP ViT-L/14 probe (0.9289) and far above its EfficientNet-B0 probe
(0.8311). This is the bias Grommelt et al. name in "Fake or JPEG? Revealing Common Biases in
Generated Image Detection Datasets".

Consequences, and they are not cosmetic:
  1. A detector accuracy on this dataset is NOT evidence of forensic capability until it is
     read against 0.9785, not against 0.5. Reporting 0.9289 as a success is reporting a
     number that a `width == height` check beats.
  2. Our own pipeline does not read geometry directly -- images reach the model as 224x224
     centre crops -- but the decode cache re-encodes every image to JPEG q95, which turns a
     PNG-vs-JPEG provenance difference into a single-vs-double compression difference that a
     CNN can learn. The shortcut is laundered, not removed.
  3. The correct fix is a geometry- and format-controlled evaluation subset, not a better
     model. That is a WP1 protocol change and it is worth more than any accuracy point.

This module computes the baselines from parquet metadata alone, so they cost no GPU and
cannot drift from the split they describe.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path("/home/samuel-renteria/Desktop/FILES/DATASETS/sid-set/data")


def compute(manifest: pathlib.Path, split: str = "test") -> dict:
    t = pq.read_table(manifest, columns=["uid", "img_id", "shard", "label", "y_binary", "split"])
    want: dict[str, dict] = collections.defaultdict(dict)
    for u, i, s, lab, y, sp in zip(t.column("uid").to_pylist(), t.column("img_id").to_pylist(),
                                   t.column("shard").to_pylist(), t.column("label").to_pylist(),
                                   t.column("y_binary").to_pylist(), t.column("split").to_pylist()):
        if sp == split:
            want[s][i] = (u, int(y), int(lab))

    sq, png, ys, labs = [], [], [], []
    for shard, ids in want.items():
        pf = pq.ParquetFile(DATA / shard)
        for rg in range(pf.metadata.num_row_groups):
            tt = pf.read_row_group(rg, columns=["img_id", "width", "height", "image"])
            for iid, w, h, im in zip(tt.column("img_id").to_pylist(), tt.column("width").to_pylist(),
                                     tt.column("height").to_pylist(), tt.column("image").to_pylist()):
                if iid in ids:
                    sq.append(int(w == h))
                    png.append(int(Image.open(io.BytesIO(im["bytes"])).format == "PNG"))
                    ys.append(ids[iid][1])
                    labs.append(ids[iid][2])
    sq, png, ys, labs = map(np.asarray, (sq, png, ys, labs))
    prior = max(ys.mean(), 1 - ys.mean())
    out = {
        "split": split, "n": int(ys.size), "fake_prior": float(ys.mean()),
        "B0_majority_class": float(prior),
        "B0_square_is_fake": float((sq == ys).mean()),
        "B0_png_is_fake": float((png == ys).mean()),
        "B0_square_or_png_is_fake": float(((sq | png) == ys).mean()),
        "square_rate_by_label": {str(k): float(sq[labs == k].mean()) for k in sorted(set(labs.tolist()))},
        "png_rate_by_label": {str(k): float(png[labs == k].mean()) for k in sorted(set(labs.tolist()))},
        "note": "A detector accuracy on SID-Set must be read against B0_square_is_fake, not "
                "against 0.5. See the module docstring.",
    }
    return out


def headline(b: dict, condition: str | None = None) -> str:
    """Protocol-aware. Under the `squarecrop` geometry control the artifact is removed BY
    CONSTRUCTION -- every image is square, so `width == height` is constant and its accuracy
    is the majority class. Printing 0.9785 next to a controlled row would misstate the
    protocol in the direction that flatters nobody."""
    if condition and "squarecrop" in condition:
        return (f"GEOMETRY-CONTROLLED protocol: every image centre-cropped to square, so "
                f"'width==height -> fake' is constant and scores the majority class "
                f"({b['B0_majority_class']:.4f}). On the RAW data the same rule scores "
                f"{b['B0_square_is_fake']:.4f} -- this row is not riding that artifact.")
    return (f"TRIVIAL BASELINE on this split: 'width==height -> fake' = "
            f"{b['B0_square_is_fake']:.4f} accuracy (majority class {b['B0_majority_class']:.4f}). "
            f"Read every model accuracy against that number.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=pathlib.Path, default=ROOT / "runs" / "manifest_v1.parquet")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "runs" / "trivial_baselines.json")
    a = ap.parse_args()
    b = compute(a.manifest, a.split)
    a.out.write_text(json.dumps(b, indent=2) + "\n")
    print(json.dumps(b, indent=2))
    print("\n" + headline(b))


if __name__ == "__main__":
    main()
