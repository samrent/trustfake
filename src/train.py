"""WP3 defense package: standard training, PGD adversarial training (Madry), and TRADES.

The proposal demands a CONTROLLED comparison -- fixed backbone, fixed data protocol, fixed
evaluation pipeline, comparable training budgets. Operationally that means:

  fixed backbone      tf_efficientnet_b0.ns_jft_in1k, identical init seed for all three runs
  fixed data          the same fit split, the same augmentation, the same batch order
  fixed evaluation    the WP1 harness, unchanged, on splits these runs never see
  matched budget      the SAME NUMBER OF OPTIMIZER STEPS, not the same wall clock. PGD-k
                      training costs k+1 forward/backward passes per step, so matching wall
                      clock would give the clean model ~8x more weight updates and the
                      comparison would measure budget, not method. Wall clock is reported
                      alongside so the cost is visible rather than hidden.

MODEL SELECTION NEVER TOUCHES calib OR test. The fit split is further divided into
fit_train / fit_val by a seeded permutation of uids; the epoch is chosen on fit_val. calib
still fits only the temperature, test is still reported once. Selecting the epoch on calib
would leak the calibration set into the model, which is the same self-own as fitting
temperature on test, one level up.

Attacks during training operate on [0,1] pixels because NormalizedModel normalizes inside
forward(). If normalization were applied outside, every epsilon here and in WP2 would be
wrong by ~1/std and nothing would raise.

EPSILON IS NOT 8/255 HERE, AND THAT IS A RESULT, NOT A SHORTCUT.
The first run used the standard ImageNet-robustness budget eps=8/255 and PGD-AT COLLAPSED:
loss pinned at 0.71 (= ln 2, uniform output), fit_val clean and robust both 0.4943 (= the
majority-class rate). Evidence: runs/train_wp3_eps8_255_collapsed.log.

The reason is specific to forensics. Deepfake evidence is small-amplitude, often
high-frequency residue -- generator fingerprints, resampling traces, blending seams. An
8/255 Linf ball is wide enough to erase that evidence outright, so the robust-optimal
classifier inside that ball really can be a constant. Robustness budgets transplanted from
object recognition, where the signal is large-amplitude semantic structure, do not transfer.

So the defenses train at forensic budgets (1-4/255), with a linear epsilon warm-up so early
epochs learn the task before the ball opens, and optionally initialized from the standard
checkpoint. The collapse at 8/255 stays in the report as the finding it is.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .decode import CACHE, cache_path
from .models import CKPT_DIR, new_trainable, save_checkpoint

ROOT = pathlib.Path(__file__).resolve().parents[1]


class TrainImages(Dataset):
    """(x in [0,1], y) from the decode cache. Mild augmentation only: the cache is already
    short-side 256, and heavy augmentation would confound the robustness comparison."""

    def __init__(self, uids, y, train: bool):
        self.uids, self.y = list(uids), np.asarray(y)
        self.tf = (transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]) if train else
            transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]))

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        im = Image.open(cache_path(self.uids[i], CACHE)).convert("RGB")
        return self.tf(im), int(self.y[i])


def split_fit(manifest: pathlib.Path, seed: int, val_frac: float = 0.1, limit_fit: int = 0):
    """fit -> (fit_train, fit_val) by seeded permutation. calib and test are not touched.

    limit_fit > 0 caps the TRAINING set to that many images (val stays proportional). This is
    the small-slice knob for a combinatorial sweep: it touches training only, so every reported
    eval split is byte-identical and verify.py still passes. The cap is applied AFTER the seeded
    permutation, so the slice is a fixed, reproducible subset for a given seed.
    """
    t = pq.read_table(manifest, columns=["uid", "y_binary", "split"])
    keep = [i for i, s in enumerate(t.column("split").to_pylist()) if s == "fit"]
    uids = np.array([t.column("uid")[i].as_py() for i in keep])
    y = np.array([t.column("y_binary")[i].as_py() for i in keep])
    order = np.random.default_rng(seed).permutation(len(uids))
    n_val = int(len(uids) * val_frac)
    va, tr = order[:n_val], order[n_val:]
    if limit_fit and limit_fit < len(tr):
        tr = tr[:limit_fit]
    return (uids[tr], y[tr]), (uids[va], y[va])


# --------------------------------------------------------------------------- attacks

def pgd_ce(model, x, y, eps, alpha, steps, scaler):
    """Madry inner maximization: maximize cross-entropy inside an Linf ball, in [0,1]."""
    delta = torch.empty_like(x).uniform_(-eps, eps)
    delta = ((x + delta).clamp(0, 1) - x).detach()
    for _ in range(steps):
        delta.requires_grad_(True)
        with torch.amp.autocast("cuda"):
            loss = F.cross_entropy(model(x + delta), y)
        g, = torch.autograd.grad(scaler.scale(loss), delta)
        delta = (delta.detach() + alpha * g.sign()).clamp(-eps, eps)
        delta = ((x + delta).clamp(0, 1) - x)
    return (x + delta).detach()


def pgd_kl(model, x, eps, alpha, steps, scaler):
    """TRADES inner maximization: maximize KL(f(x+delta) || f(x)), NOT cross-entropy.

    Using CE here is the classic TRADES misimplementation -- it silently degrades the method
    into adversarial training with an extra regularizer, and the result still looks
    plausible, which is why it survives review.
    """
    with torch.no_grad(), torch.amp.autocast("cuda"):
        p_clean = F.softmax(model(x).float(), dim=1)
    delta = (0.001 * torch.randn_like(x)).detach()
    delta = ((x + delta).clamp(0, 1) - x).detach()
    for _ in range(steps):
        delta.requires_grad_(True)
        with torch.amp.autocast("cuda"):
            logp = F.log_softmax(model(x + delta).float(), dim=1)
            loss = F.kl_div(logp, p_clean, reduction="batchmean")
        g, = torch.autograd.grad(scaler.scale(loss), delta)
        delta = (delta.detach() + alpha * g.sign()).clamp(-eps, eps)
        delta = ((x + delta).clamp(0, 1) - x)
    return (x + delta).detach()


def pgd_overconf(model, x, eps, alpha, steps, scaler):
    """Inner attack for the CONFIDENCE defense: worst-case confidence inflation (Ledda et al.).

    Freeze the clean prediction yhat, then DESCEND cross-entropy to yhat -- pushing probability
    mass toward the already-predicted class, i.e. inflating confidence without a label. The outer
    loss (method at_conf) then trains the model to still classify these confidence-inflated inputs
    CORRECTLY, so an attacker cannot manufacture confident-but-WRONG predictions -- which is exactly
    what the ACE / over-confidence attack does to the moderation layer. This targets the confidence
    axis, not the label axis that PGD-AT/TRADES defend.
    """
    with torch.no_grad(), torch.amp.autocast("cuda"):
        yhat = model(x).float().argmax(1)
    delta = (0.001 * torch.randn_like(x)).detach()
    delta = ((x + delta).clamp(0, 1) - x).detach()
    for _ in range(steps):
        delta.requires_grad_(True)
        with torch.amp.autocast("cuda"):
            loss = F.cross_entropy(model(x + delta), yhat)
        g, = torch.autograd.grad(scaler.scale(loss), delta)
        delta = (delta.detach() - alpha * g.sign()).clamp(-eps, eps)   # DESCEND: inflate confidence
        delta = ((x + delta).clamp(0, 1) - x)
    return (x + delta).detach()


# ---------------------------------------------------------------------------- training

def evaluate(model, dl, eps, alpha, steps, scaler, adv: bool):
    model.eval()
    n = correct = 0
    for x, y in dl:
        x, y = x.cuda(non_blocking=False), y.cuda()
        if adv:
            x = pgd_ce(model, x, y, eps, alpha, steps, scaler)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            correct += (model(x).argmax(1) == y).sum().item()
        n += y.numel()
    model.train()
    return correct / max(n, 1)


def train(method: str, a) -> pathlib.Path:
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True

    (tr_uid, tr_y), (va_uid, va_y) = split_fit(a.manifest, a.seed, a.val_frac,
                                               getattr(a, "limit_fit", 0))
    dl_tr = DataLoader(TrainImages(tr_uid, tr_y, True), batch_size=a.batch, shuffle=True,
                       num_workers=a.workers, pin_memory=False, drop_last=True)
    dl_va = DataLoader(TrainImages(va_uid, va_y, False), batch_size=a.batch, shuffle=False,
                       num_workers=a.workers, pin_memory=False)

    model, _, _ = new_trainable(a.backbone)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4,
                          nesterov=True)
    steps_per_epoch = len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                total_steps=a.epochs * steps_per_epoch,
                                                pct_start=0.25)
    scaler = torch.amp.GradScaler("cuda")

    if a.init_from:
        blob = torch.load(CKPT_DIR / a.init_from, map_location="cuda", weights_only=False)
        model.load_state_dict(blob["state_dict"])
        print(f"  [{method}] initialized from {a.init_from} "
              f"(epoch {blob['meta'].get('epoch')}, {blob['meta'].get('selection_value'):.4f})")

    # --run-name gives each sweep config a UNIQUE tag. Without it the tag is method+eps only,
    # so two configs differing in beta/steps/lr/init/seed/slice would overwrite each other's
    # checkpoint AND their prediction stems (model_id = effb0_{tag}). The sweep MUST pass one.
    if getattr(a, "run_name", None):
        tag = a.run_name
    else:
        tag = f"{method}_eps{round(a.eps*255)}_255" if a.eps * 255 == round(a.eps * 255) else method
    ckpt_dir = pathlib.Path(getattr(a, "ckpt_dir", None) or CKPT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    hist, best, best_path = [], -1.0, ckpt_dir / f"{tag}.pt"
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        # linear epsilon warm-up: open the ball only once the task is learned, otherwise the
        # first epochs optimize against noise and the model settles on a constant output
        w = min(1.0, ep / max(a.eps_warmup_epochs, 1)) if a.eps_warmup_epochs else 1.0
        eps, alpha = a.eps * w, a.alpha * w
        run_loss, seen = 0.0, 0
        for x, y in dl_tr:
            x, y = x.cuda(non_blocking=False), y.cuda()
            if method == "standard":
                with torch.amp.autocast("cuda"):
                    loss = F.cross_entropy(model(x), y)
            elif method == "at_pgd":
                xa = pgd_ce(model, x, y, eps, alpha, a.inner_steps, scaler)
                with torch.amp.autocast("cuda"):
                    loss = F.cross_entropy(model(xa), y)
            elif method == "trades":
                xa = pgd_kl(model, x, eps, alpha, a.inner_steps, scaler)
                with torch.amp.autocast("cuda"):
                    logits_c = model(x).float()
                    logits_a = model(xa).float()
                    loss = (F.cross_entropy(logits_c, y) + a.beta *
                            F.kl_div(F.log_softmax(logits_a, 1), F.softmax(logits_c, 1),
                                     reduction="batchmean"))
            elif method == "at_kl":
                # STRATEGY B (hybrid): adversarial CE + a consistency KL between the adversarial
                # and clean predictions. Unlike TRADES (CE on CLEAN + KL), the CE here is on the
                # ADVERSARIAL example -- a genuine AT + regularizer stack, not a rename of TRADES.
                xa = pgd_ce(model, x, y, eps, alpha, a.inner_steps, scaler)
                with torch.amp.autocast("cuda"):
                    logits_a = model(xa).float()
                    logits_c = model(x).float()
                    loss = (F.cross_entropy(logits_a, y) + a.beta *
                            F.kl_div(F.log_softmax(logits_a, 1), F.softmax(logits_c.detach(), 1),
                                     reduction="batchmean"))
            elif method == "at_conf":
                # STRATEGY C1: adversarial training against the CONFIDENCE attack. Inner max
                # inflates confidence (pgd_overconf); outer CE forces correctness on those inputs,
                # so confidence cannot be inflated on wrong predictions.
                xa = pgd_overconf(model, x, eps, alpha, a.inner_steps, scaler)
                with torch.amp.autocast("cuda"):
                    loss = F.cross_entropy(model(xa), y)
            elif method == "conf_reg":
                # STRATEGY C2: no inner attack -- a confidence PENALTY on errors. Penalize high
                # softmax confidence on the (detached) misclassified examples, directly optimizing
                # the failure-prediction property the moderation layer reads.
                with torch.amp.autocast("cuda"):
                    logits = model(x).float()
                    p = F.softmax(logits, 1)
                    wrong = (logits.argmax(1) != y).float().detach()
                    loss = F.cross_entropy(logits, y) + a.lambda_reg * (p.max(1).values * wrong).mean()
            else:
                raise ValueError(method)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += float(loss) * y.numel()
            seen += y.numel()

        clean_acc = evaluate(model, dl_va, eps, alpha, a.eval_steps, scaler, adv=False)
        rob_acc = evaluate(model, dl_va, eps, alpha, a.eval_steps, scaler, adv=True)
        # selection metric: robust accuracy for the defenses, clean for the standard model --
        # each method is selected on what it is trying to optimize, stated rather than assumed
        sel = clean_acc if method in ("standard", "conf_reg", "at_conf") else rob_acc
        hist.append({"epoch": ep, "loss": run_loss / seen, "eps": eps, "fit_val_clean": clean_acc,
                     "fit_val_robust_pgd": rob_acc, "selected_on": sel,
                     "lr": sched.get_last_lr()[0], "minutes": round((time.time() - t0) / 60, 2)})
        print(f"  [{tag}] ep{ep:>2}/{a.epochs} eps {eps*255:.1f}/255 loss {run_loss/seen:.4f}  "
              f"fit_val clean {clean_acc:.4f}  robust {rob_acc:.4f}  "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
        if sel > best:
            best = sel
            save_checkpoint(model, best_path, {
                "model_id": f"effb0_{tag}", "backbone": a.backbone, "method": method,
                "epoch": ep, "selected_on": "fit_val_clean" if method in ("standard", "conf_reg", "at_conf") else "fit_val_robust_pgd",
                "selection_value": sel, "eps": eps, "alpha": alpha,
                "inner_steps": a.inner_steps if method not in ("standard", "conf_reg") else 0,
                "beta": a.beta if method in ("trades", "at_kl") else None,
                "lambda_reg": a.lambda_reg if method == "conf_reg" else None,
                "epochs": a.epochs, "batch": a.batch, "lr": a.lr, "seed": a.seed,
                # Budget parity is judged on steps the RUN spends, which is
                # epochs_planned * steps_per_epoch once the schedule completes -- not on
                # steps to the selected epoch, which is a property of model selection and
                # differs between arms even under a perfectly matched budget. Recording only
                # one of these invites reading the wrong number as the budget.
                "optimizer_steps_to_selected_epoch": ep * steps_per_epoch,
                "optimizer_steps_planned": a.epochs * steps_per_epoch,
                "steps_per_epoch": steps_per_epoch,
                "selected_epoch": ep,
                "epochs_planned": a.epochs,
                "is_last_epoch": ep == a.epochs,
                "n_fit_train": int(len(tr_uid)), "n_fit_val": int(len(va_uid)),
                "limit_fit": int(getattr(a, "limit_fit", 0)), "run_name": getattr(a, "run_name", None),
                "eps_warmup_epochs": a.eps_warmup_epochs, "init_from": a.init_from,
                "manifest": str(a.manifest), "history": hist,
                "wall_clock_min": round((time.time() - t0) / 60, 2),
                "note": "model selection on a held-out slice of FIT; calib and test never seen",
            })
    print(f"  [{tag}] best {best:.4f} -> {best_path}  ({(time.time()-t0)/60:.1f} min)")
    return best_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", nargs="*", default=["standard", "at_pgd", "trades"])
    ap.add_argument("--backbone", default="tf_efficientnet_b0.ns_jft_in1k")
    ap.add_argument("--manifest", type=pathlib.Path, default=ROOT / "runs" / "manifest_v2.parquet")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--eps", type=float, default=2 / 255,
                    help="forensic budget; 8/255 collapses this task (see module docstring)")
    ap.add_argument("--eps-warmup-epochs", type=int, default=4)
    ap.add_argument("--init-from", default=None, help="checkpoint filename to warm-start from")
    ap.add_argument("--alpha", type=float, default=0.5 / 255)
    ap.add_argument("--inner-steps", type=int, default=7)
    ap.add_argument("--eval-steps", type=int, default=10)
    ap.add_argument("--beta", type=float, default=6.0, help="TRADES / at_kl beta (Zhang et al. 2019)")
    ap.add_argument("--lambda-reg", type=float, default=1.0, help="conf_reg confidence-penalty weight")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default=None,
                    help="unique tag for a sweep config; becomes the checkpoint name and model_id "
                         "(effb0_{run_name}). REQUIRED per config in a sweep to avoid collisions.")
    ap.add_argument("--limit-fit", type=int, default=0,
                    help="cap the training set to N images (small-slice sweep); 0 = use all fit")
    ap.add_argument("--ckpt-dir", default=None,
                    help="write checkpoints here instead of runs/checkpoints (keep sweeps out of "
                         "the canonical dir)")
    a = ap.parse_args()
    for m in a.methods:
        print(f"=== {m} ===", flush=True)
        train(m, a)


if __name__ == "__main__":
    main()
