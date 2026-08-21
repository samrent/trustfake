"""Run the WP2 attack matrix: {model spec} x {attack} x {epsilon} -> prediction npz files.

One runner for every attack and every detector, so the clean row and the attacked rows can
never come from different code paths. That is not tidiness: if the clean baseline is
produced by a different script than the attacked condition, a preprocessing difference
between them shows up as a robustness result.

Emits the standard contract, with the condition string carrying the attack and its
distinguishing hyperparameters (attack_suite.condition_name), so score.py and moderation.py
pivot the whole matrix without a lookup table.

For confidence-targeted attacks the sidecar also records label_preservation and the
effective epsilon. If label preservation is not ~1.0 for an attack that claims to be
label-preserving, the run is invalid and the number must be read, not assumed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from .attack_suite import ATTACKS, ace_full, condition_name, parse_condition
from .decode import CACHE
from .features import CachedImages, FEATURES
from .models import build

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRED = ROOT / "runs" / "predictions"


def run_one(det, split: str, attack: str, eps: float | None, manifest: pathlib.Path,
            batch: int, limit: int | None, pred_dir: pathlib.Path, steps: int | None = None,
            pixel_condition: str | None = None):
    """attack='clean' + pixel_condition='jpeg_q50' runs a REALISTIC condition; attack=<name>
    runs an adversarial one. Both land in the same contract through the same code path, which
    is what makes the clean / realistic / adversarial rows of the WP3 table comparable."""
    t = pq.read_table(manifest, columns=["uid", "img_id", "y_binary", "label", "split"])
    keep = [i for i, s in enumerate(t.column("split").to_pylist()) if s == split]
    if limit:
        keep = keep[:limit]
    uids = [t.column("uid")[i].as_py() for i in keep]
    imgs = [t.column("img_id")[i].as_py() for i in keep]
    y = np.array([t.column("y_binary")[i].as_py() for i in keep], dtype=np.int8)
    lab = np.array([t.column("label")[i].as_py() for i in keep], dtype=np.int8)

    cond = (pixel_condition or "clean") if attack == "clean" else condition_name(attack, eps=eps, steps=steps)
    spec = ATTACKS.get(attack)
    kw = {} if attack == "clean" else {**spec["defaults"],
                                       **{k: v for k, v in (("eps", eps), ("steps", steps)) if v is not None}}

    dl = DataLoader(CachedImages(uids, det.transform, pixel_condition or "clean", CACHE), batch_size=batch,
                    num_workers=4, pin_memory=False, shuffle=False)
    logits = np.empty((len(uids), 2), dtype=np.float32)
    effs, preserved, t0 = [], [], time.time()

    for xb, ib in dl:
        xb = xb.cuda()
        yb = torch.from_numpy(y[ib.numpy()].astype(np.int64)).cuda()
        if attack == "clean":
            with torch.no_grad():
                lb = det.module(xb)
        elif attack in ("ace", "ace_uint8"):
            # ACE reports the accept-pass logits: re-running on x_adv changes the batch shape
            # and cuDNN is not bit-identical across shapes, which flips boundary samples.
            _, eff, clean_pred, lb = ace_full(det.module, xb, yb, kw["eps"],
                                              quantize=(attack == "ace_uint8"))
            effs.append(eff.cpu().numpy())
            preserved.append((lb.argmax(1) == clean_pred).float().mean().item())
        else:
            with torch.no_grad():
                clean_pred = det.module(xb).argmax(1)
            xadv = spec["fn"](det.module, xb, yb, **kw)
            with torch.no_grad():
                lb = det.module(xadv)
            effs.append((xadv - xb).abs().amax(dim=(1, 2, 3)).cpu().numpy())
            preserved.append((lb.argmax(1) == clean_pred).float().mean().item())
        logits[ib.numpy()] = lb.float().cpu().numpy()

    el = time.time() - t0
    pred_dir.mkdir(parents=True, exist_ok=True)
    stem = f"predictions_{det.model_id}_{split}_{cond}"
    np.savez(pred_dir / f"{stem}.npz", uid=np.array(uids), img_id=np.array(imgs),
             y=y, label=lab, logits=logits)
    side = {"model_id": det.model_id, "backbone": det.backbone, "condition": cond,
            "split": split, "n": len(uids), "seconds": round(el, 1), "key": "uid",
            # BATCH SIZE IS PART OF THE CONDITION. EfficientNet under cuDNN is not
            # bit-identical across batch shapes: regenerating the clean baseline at batch 16
            # instead of 32 moved accuracy by one sample in 15,316, which is enough to break
            # a headline that reads "accuracy identical across every row".
            "batch": batch,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": 0,
            "manifest_sha": json.loads(manifest.with_suffix(".json").read_text())["manifest_sha256"],
            "provenance": det.provenance, **parse_condition(cond)}
    if effs:
        e = np.concatenate(effs)
        # frac_perturbed matters because AutoAttack and APGD write back ONLY the samples they
        # managed to flip (fra31 autoattack.py, torchattacks apgd.py), so x_adv is a MIXTURE of
        # clean and attacked images. Robust accuracy stays meaningful on a mixture; AURC and ECE
        # do not. Any table that pivots rows with different frac_perturbed together is lying.
        side |= {"eps_effective_mean": float(e.mean()), "eps_effective_max": float(e.max()),
                 "frac_perturbed": float((e > 0).mean()),
                 "label_preservation": float(np.mean(preserved)), "attack_kwargs": kw}
        if attack in ("pgd_linf", "pgd_l2"):
            from .attack_suite import default_alpha
            side["alpha"] = kw.get("alpha") or default_alpha(kw["eps"], kw.get("steps", 10))
            side["alpha_rule"] = "2.5*eps/steps (Madry; RobustBench convention)"
    (pred_dir / f"{stem}.json").write_text(json.dumps(side, indent=2, default=str) + "\n")
    msg = f"  {stem:<62} {len(uids):>6} imgs {el:>5.0f}s"
    if effs:
        msg += f"  eff_eps {side['eps_effective_mean']:.5f}  preserved {side['label_preservation']:.4f}"
    print(msg, flush=True)
    return stem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="probe:tf_efficientnet_b0.ns_jft_in1k")
    ap.add_argument("--attacks", nargs="*", default=["clean", "fgsm", "pgd_linf", "ace", "overconf", "underconf"])
    ap.add_argument("--conditions", nargs="*", default=[],
                    help="realistic (pixel-space) conditions: jpeg_q50, downscale_0.5, webp, squarecrop")
    ap.add_argument("--eps", type=float, default=None, help="override the attack default")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--manifest", type=pathlib.Path, default=ROOT / "runs" / "manifest_v1.parquet")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--features-dir", type=pathlib.Path, default=FEATURES)
    ap.add_argument("--pred-dir", type=pathlib.Path, default=PRED)
    a = ap.parse_args()

    # A limited run writes the SAME stem as a full run: predictions_{model}_{split}_{cond}.
    # Running with --limit against the default directory silently replaced the 15,316-row
    # clean baseline with a 512-row one, and every delta measured against it became
    # meaningless without erroring. A partial run must be namespaced.
    if a.limit and a.pred_dir == PRED:
        raise SystemExit(
            "--limit writes the same filenames as a full run and would overwrite the "
            "canonical predictions. Pass --pred-dir runs/partial (or another directory).")

    det = build(a.model, a.features_dir)
    print(f"detector {det.model_id} ({det.provenance['kind']})")
    for atk in a.attacks:
        run_one(det, a.split, atk, a.eps, a.manifest, a.batch, a.limit, a.pred_dir, a.steps)
    for cond in a.conditions:
        run_one(det, a.split, "clean", None, a.manifest, a.batch, a.limit, a.pred_dir,
                pixel_condition=cond)


if __name__ == "__main__":
    main()
