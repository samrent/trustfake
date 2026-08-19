"""ACE (Attack on Confidence Estimation) against a real detector, plus PGD for contrast.

Two things here are the number-one silent bugs in this area, so they are structural:

  1. NORMALIZATION LIVES INSIDE forward(). mean/std are registered buffers, so the module
     consumes [0,1] pixels and every epsilon in the report is an epsilon in pixel space.
     Normalizing outside means every reported epsilon is wrong by a factor of ~1/std, and
     nothing raises.

  2. ACE's perturbation direction is computed ONCE, outside the accept/halve loop
     (Galil & El-Yaniv, Algorithm 1). Recomputing the gradient each iteration turns it into
     PGD and destroys the effective-epsilon interpretation.

ACE moves confidence, not predictions: it pushes DOWN the confidence of correct predictions
and UP the confidence of wrong ones, then accepts a step per-sample only if the argmax is
unchanged, halving epsilon otherwise. So accuracy is preserved to 4 dp by construction --
and if the accuracy column moves, the accept test is broken and every number downstream is
meaningless. That is the regression check, not an accuracy threshold.

The attack target is EfficientNet-B0 (cheap backward, cached offline), wearing a linear
head fitted on its own frozen features. No backbone fine-tuning: the gradient flows through
frozen weights to the INPUT, which is all an input-space attack needs.
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
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms

from .decode import CACHE
from .features import FEATURES, CachedImages, apply_condition  # noqa: F401 (ladder lives there)

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRED = ROOT / "runs" / "predictions"
TARGET = "tf_efficientnet_b0.ns_jft_in1k"


class NormalizedModel(nn.Module):
    """backbone + linear head, with normalization INSIDE forward(). Input: [0,1] pixels."""

    def __init__(self, backbone: nn.Module, head: nn.Linear, mean, std):
        super().__init__()
        self.backbone, self.head = backbone, head
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone((x - self.mean) / self.std))


def build_target(seed: int = 0, fdir=FEATURES):
    """Fit the linear head on cached EfficientNet-B0 features, then fold the standardizer
    into the head so the deployed module is exactly backbone -> Linear."""
    stem = f"{TARGET.split('.')[0]}_fit_clean"
    x = np.load(fdir / f"{stem}.npy")
    meta = json.loads((fdir / f"{stem}.json").read_text())
    y = np.asarray(meta["y_binary"])

    sc = StandardScaler().fit(x)
    clf = LogisticRegression(C=0.01, max_iter=5000).fit(sc.transform(x), y)
    w = (clf.coef_[0] / sc.scale_).astype(np.float32)
    b = float(clf.intercept_[0] - (clf.coef_[0] * sc.mean_ / sc.scale_).sum())

    head = nn.Linear(x.shape[1], 2)
    with torch.no_grad():                       # logits = [0, margin]: contract-compatible
        head.weight.zero_(); head.bias.zero_()
        head.weight[1] = torch.from_numpy(w)
        head.bias[1] = b

    backbone = timm.create_model(TARGET, pretrained=True, num_classes=0)
    cfg = timm.data.resolve_model_data_config(backbone)
    model = NormalizedModel(backbone, head, cfg["mean"], cfg["std"]).eval().cuda()
    for p in model.parameters():
        p.requires_grad_(False)

    # the model transform with Normalize REMOVED: the module normalizes internally
    full = timm.data.create_transform(**cfg, is_training=False)
    tf = transforms.Compose([t for t in full.transforms
                             if not isinstance(t, transforms.Normalize)])
    print(f"target {TARGET}: head fitted on {x.shape[0]} imgs, train acc "
          f"{clf.score(sc.transform(x), y):.4f}; transform {[type(t).__name__ for t in tf.transforms]}")
    return model, tf


def ace(model, x, y, eps: float, iters: int = 15):
    """Galil & El-Yaniv Algorithm 1. Returns (x_adv, effective_eps_per_sample).

    Direction is computed once; the loop only halves epsilon for samples whose argmax
    flipped. A sample that never finds an accepting epsilon is returned UNPERTURBED, which
    is why label preservation is exactly 1.0.

    The accepted logits are returned FROM THE ACCEPT-CHECK FORWARD and must be reported as
    the model output. Re-running the model on x_adv afterwards is a different batch shape,
    and EfficientNet under cuDNN is not bit-identical across batch shapes: a sample sitting
    on the decision boundary then flips between the two forwards, and label preservation
    reads 0.9980 instead of 1.0000 for a reason that has nothing to do with the attack.
    """
    x = x.cuda(non_blocking=False)
    y = y.cuda()
    with torch.enable_grad():
        xg = x.clone().requires_grad_(True)
        logits = model(xg)
        pred = logits.argmax(1)
        kappa = logits.softmax(1).gather(1, pred[:, None]).squeeze(1)
        grad = torch.autograd.grad(kappa.sum(), xg)[0]
    eta = grad.sign()

    correct = pred == y
    # push confidence DOWN where correct, UP where wrong: both directions destroy the
    # ordering that an abstention rule depends on, while leaving the argmax alone
    direction = torch.where(correct, -1.0, 1.0).view(-1, 1, 1, 1)

    out = x.clone()
    out_logits = logits.detach().clone()          # unperturbed samples keep clean logits
    eps_i = torch.full((x.shape[0],), float(eps), device=x.device)
    pending = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    for _ in range(iters):
        idx = pending.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            break
        cand = (x[idx] + direction[idx] * eps_i[idx].view(-1, 1, 1, 1) * eta[idx]).clamp(0, 1)
        with torch.no_grad():
            cand_logits = model(cand)
        ok = cand_logits.argmax(1) == pred[idx]
        acc_idx = idx[ok]
        out[acc_idx] = cand[ok]
        out_logits[acc_idx] = cand_logits[ok]
        pending[acc_idx] = False
        eps_i[idx[~ok]] *= 0.5
    eff = (out - x).abs().amax(dim=(1, 2, 3))
    return out, eff, pred, out_logits


def run(split: str, condition: str, model, tf, manifest, batch: int = 32,
        limit: int | None = None, pred_dir: pathlib.Path = PRED):
    t = pq.read_table(manifest, columns=["uid", "img_id", "y_binary", "label", "split"])
    keep = [i for i, s in enumerate(t.column("split").to_pylist()) if s == split]
    if limit:
        keep = keep[:limit]
    uids = [t.column("uid")[i].as_py() for i in keep]
    imgs = [t.column("img_id")[i].as_py() for i in keep]
    y = np.array([t.column("y_binary")[i].as_py() for i in keep], dtype=np.int8)
    lab = np.array([t.column("label")[i].as_py() for i in keep], dtype=np.int8)

    # A condition is either an ADVERSARIAL one (ace_eps*), applied in tensor space by the
    # attack, or a REALISTIC one (jpeg_q*, downscale_*, webp), applied to the PIL image
    # before the model transform. The proposal names compression, resizing and re-encoding
    # explicitly, so the ladder is spec compliance, not garnish. Both kinds travel through
    # the same detector and land in the same npz contract, which is what makes the
    # clean / realistic / adversarial rows of the WP3 table comparable at all.
    adversarial = condition.startswith("ace_eps")
    pixel_condition = "clean" if adversarial else condition
    dl = DataLoader(CachedImages(uids, tf, pixel_condition, CACHE), batch_size=batch,
                    num_workers=4, pin_memory=False, shuffle=False)
    eps = float(condition.split("eps")[1]) if adversarial else 0.0

    logits = np.empty((len(uids), 2), dtype=np.float32)
    effs, preserved, t0 = [], [], time.time()
    for xb, ib in dl:
        yb = torch.from_numpy(y[ib.numpy()].astype(np.int64))
        if eps > 0:
            _, eff, clean_pred, lb = ace(model, xb, yb, eps)
            preserved.append((lb.argmax(1) == clean_pred).float().mean().item())
            effs.append(eff.cpu().numpy())
        else:
            with torch.no_grad():
                lb = model(xb.cuda())
        logits[ib.numpy()] = lb.float().cpu().numpy()
    el = time.time() - t0

    pred_dir.mkdir(parents=True, exist_ok=True)
    stem = f"predictions_{TARGET.split('.')[0]}_{split}_{condition}"
    np.savez(pred_dir / f"{stem}.npz", uid=np.array(uids), img_id=np.array(imgs),
             y=y, label=lab, logits=logits)
    side = {"model_id": f"{TARGET}+linear_probe", "condition": condition, "split": split,
            "batch": batch,
            "seed": 0, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "n": len(uids),
            "manifest_sha": json.loads(manifest.with_suffix(".json").read_text())["manifest_sha256"],
            "key": "uid", "attack": "ACE (Galil & El-Yaniv Alg.1)" if eps else "none",
            "condition_kind": "adversarial" if adversarial else
                              ("clean" if condition == "clean" else "realistic"),
            "eps_requested": eps, "seconds": round(el, 1)}
    if eps:
        e = np.concatenate(effs)
        side |= {"eps_effective_mean": float(e.mean()), "eps_effective_max": float(e.max()),
                 "label_preservation": float(np.mean(preserved))}
    (pred_dir / f"{stem}.json").write_text(json.dumps(side, indent=2) + "\n")
    msg = f"  {stem}: n={len(uids)} in {el:.0f}s"
    if eps:
        msg += (f"  eff_eps={side['eps_effective_mean']:.5f} "
                f"label_preserved={side['label_preservation']:.4f}")
    print(msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="*", default=["calib", "test"])
    ap.add_argument("--conditions", nargs="*",
                    default=["clean", "ace_eps0.0005", "ace_eps0.002", "ace_eps0.005"])
    ap.add_argument("--manifest", type=pathlib.Path, default=ROOT / "runs" / "manifest_v1.parquet")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--features-dir", type=pathlib.Path, default=FEATURES)
    ap.add_argument("--pred-dir", type=pathlib.Path, default=PRED)
    a = ap.parse_args()
    model, tf = build_target(fdir=a.features_dir)
    for split in a.splits:
        for cond in a.conditions:
            if split == "calib" and cond != "clean":
                continue                     # temperature is fitted on CLEAN calib, always
            run(split, cond, model, tf, a.manifest, a.batch, a.limit, a.pred_dir)


if __name__ == "__main__":
    main()
