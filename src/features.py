"""Extract frozen backbone features for the manifest's images.

Frozen backbone + linear probe on cached features is the UniversalFakeDetect reference
recipe: it fits in seconds, it is the baseline reviewers expect, and it keeps the whole
experiment inside an afternoon. Fine-tuning is a camp experiment with a training loop
attached, and it is deliberately out of scope here.

Two details that fail silently if you get them wrong:

  1. The transform must come from timm's own resolve_model_data_config. CLIP's
     normalization is (0.4815, 0.4578, 0.4082) / (0.2686, 0.2613, 0.2758) with crop_pct
     1.0 -- NOT ImageNet's. Hardcoding ImageNet stats costs accuracy and raises nothing.
  2. Features are written in the manifest's uid order and the uid list is written beside
     them. Never re-derive the order from a directory listing.

fp16 for the forward, float32 on disk, float64 for anything downstream that touches a
softmax.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import pyarrow.parquet as pq
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .decode import CACHE, cache_path

FEATURES = pathlib.Path("/home/samuel-renteria/Desktop/FILES/DATASETS/sid-set/features")
DEFAULT_MODEL = "vit_large_patch14_clip_224.openai"


class CachedImages(Dataset):
    """Reads the flat JPEG cache. `condition` is applied here, in pixel space, before the
    model transform -- that is what makes clean / jpeg_q50 / downscale comparable."""

    def __init__(self, uids, transform, condition: str = "clean", root: pathlib.Path = CACHE):
        self.uids, self.transform, self.condition, self.root = list(uids), transform, condition, root

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i):
        im = Image.open(cache_path(self.uids[i], self.root)).convert("RGB")
        im = apply_condition(im, self.condition)
        return self.transform(im), i


def apply_condition(im: Image.Image, condition: str) -> Image.Image:
    """The perturbation ladder the proposal names verbatim: compression, resizing,
    re-encoding. Applied to the image, never to the tensor, so the epsilon story stays in
    pixel space."""
    import io as _io

    if condition == "clean":
        return im
    if condition == "squarecrop":
        # GEOMETRY CONTROL. SID-Set label 1 is 100% square 1024x1024 and label 2 is 100%
        # square, while only ~5% of real images are; a bare `width == height -> fake` rule
        # scores 0.9785 on our test split, ABOVE the CLIP probe's 0.9289. Centre-cropping
        # every image to square makes geometry carry zero label information by construction,
        # so a model evaluated under this condition cannot be riding the artifact.
        # Fit AND evaluate under the same condition or the control is just a domain shift.
        w, h = im.size
        k = min(w, h)
        return im.crop(((w - k) // 2, (h - k) // 2, (w - k) // 2 + k, (h - k) // 2 + k))
    if condition.startswith("jpeg_q"):
        buf = _io.BytesIO()
        im.save(buf, "JPEG", quality=int(condition[6:]))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if condition.startswith("downscale_"):
        f = float(condition.split("_")[1])
        w, h = im.size
        small = im.resize((max(1, int(w * f)), max(1, int(h * f))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)
    if condition == "webp":
        buf = _io.BytesIO()
        im.save(buf, "WEBP", quality=80)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    raise ValueError(f"unknown condition {condition!r}")


def extract(split: str, model_id: str, condition: str, manifest: pathlib.Path,
            batch: int = 64, workers: int = 4,
            outdir: pathlib.Path = FEATURES) -> pathlib.Path:
    t = pq.read_table(manifest, columns=["uid", "y_binary", "label", "split"])
    keep = [i for i, s in enumerate(t.column("split").to_pylist()) if s == split]
    uids = [t.column("uid")[i].as_py() for i in keep]
    y = np.array([t.column("y_binary")[i].as_py() for i in keep], dtype=np.int8)
    lab = np.array([t.column("label")[i].as_py() for i in keep], dtype=np.int8)

    model = timm.create_model(model_id, pretrained=True, num_classes=0).eval().cuda()
    cfg = timm.data.resolve_model_data_config(model)
    tf = timm.data.create_transform(**cfg, is_training=False)
    print(f"{model_id}: input {cfg['input_size']} mean {tuple(round(x,4) for x in cfg['mean'])} "
          f"crop_pct {cfg['crop_pct']}")

    ds = CachedImages(uids, tf, condition)
    dl = DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=False, shuffle=False)

    out = np.empty((len(uids), model.num_features), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for x, i in dl:
            f = model(x.cuda()).float()      # fp16 forward, float32 out: softmax never
            out[i.numpy()] = f.cpu().numpy() # sees fp16, which saturates MSP to exactly 1.0
    el = time.time() - t0

    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_id.split('.')[0]}_{split}_{condition}"
    np.save(outdir / f"{stem}.npy", out)
    (outdir / f"{stem}.json").write_text(json.dumps({
        "model_id": model_id, "split": split, "condition": condition,
        "n": len(uids), "dim": int(out.shape[1]), "uids": uids,
        "y_binary": y.tolist(), "label": lab.tolist(),
        "img_per_s": round(len(uids) / el, 1), "seconds": round(el, 1),
        "mean": [float(x) for x in cfg["mean"]], "std": [float(x) for x in cfg["std"]],
        "crop_pct": cfg["crop_pct"], "input_size": list(cfg["input_size"]),
    }, indent=2) + "\n")
    print(f"  {split}/{condition}: {out.shape} in {el:.1f}s ({len(uids)/el:.1f} img/s)"
          f" -> {outdir.name}/")
    return outdir / f"{stem}.npy"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    root = pathlib.Path(__file__).resolve().parents[1]
    ap.add_argument("--splits", nargs="*", default=["fit", "calib", "test"])
    ap.add_argument("--conditions", nargs="*", default=["clean"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--manifest", type=pathlib.Path, default=root / "runs" / "manifest_v1.parquet")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--features-dir", type=pathlib.Path, default=FEATURES,
                    help="namespace for the feature cache; --smoke runs MUST NOT share it "
                         "with a full run, the stems collide")
    a = ap.parse_args()
    for c in a.conditions:
        for s in a.splits:
            extract(s, a.model, c, a.manifest, a.batch, a.workers, a.features_dir)


if __name__ == "__main__":
    main()
