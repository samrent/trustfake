"""One place that turns a model reference into (module, eval_transform, model_id).

WP2 attacks a detector, WP3 trains detectors, WP4 serves a detector. Without a single
constructor they drift apart, and the first symptom is an epsilon that means something
different in the attack script than in the demonstrator -- silently, because both run.

A model spec is a string:

    probe:<timm_backbone>       frozen backbone + a linear head folded from the sklearn
                                probe fitted on cached features (the UniversalFakeDetect
                                reference recipe; no weights are trained)
    ckpt:<path>                 a checkpoint written by WP3 training (standard / AT / TRADES)

Every module returned satisfies the same three invariants, which are what make numbers
comparable across work packages:

  1. It consumes [0,1] pixels. Normalization is registered buffers applied INSIDE forward(),
     so an epsilon is always an epsilon in pixel space. Normalizing outside is the
     number-one silent bug in adversarial robustness reporting: every epsilon in the report
     is then wrong by a factor of ~1/std, and nothing raises.
  2. It emits logits of shape [n, 2] under the project convention, so softmax gives
     p(fake) directly and one scoring path serves everything.
  3. It carries its own eval transform WITHOUT Normalize, resolved from timm's own
     model config -- CLIP stats are not ImageNet stats, and hardcoding the wrong ones
     costs accuracy without raising.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np
import timm
import torch
import torch.nn as nn
from torchvision import transforms

from .features import FEATURES

ROOT = pathlib.Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "runs" / "checkpoints"


class NormalizedModel(nn.Module):
    """backbone + head, normalization inside forward(). Input: [0,1] pixels, output [n,2]."""

    def __init__(self, backbone: nn.Module, head: nn.Module, mean, std):
        super().__init__()
        self.backbone, self.head = backbone, head
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone((x - self.mean) / self.std))


@dataclass
class Detector:
    module: NormalizedModel
    transform: transforms.Compose
    model_id: str          # goes into the npz filename and the sidecar
    backbone: str
    provenance: dict       # how it was built, for the sidecar


def eval_transform(cfg: dict) -> transforms.Compose:
    """timm's own eval transform with Normalize stripped -- the module normalizes itself."""
    full = timm.data.create_transform(**cfg, is_training=False)
    return transforms.Compose([t for t in full.transforms
                               if not isinstance(t, transforms.Normalize)])


def _fold_probe(backbone_dim: int, coef, intercept, scaler_mean, scaler_scale) -> nn.Linear:
    """Fold a StandardScaler + sklearn binary LogisticRegression into one Linear(d, 2).

    margin = w . ((x - mu)/sigma) + b  =  (w/sigma) . x + (b - sum(w*mu/sigma))
    Row 0 is left at zero so logits are [0, margin]: softmax -> sigmoid(margin) = p(fake).
    Folding rather than keeping the scaler separate means the attack differentiates through
    exactly the deployed function, with no preprocessing step outside the graph.
    """
    w = (np.asarray(coef) / scaler_scale).astype(np.float32)
    b = float(intercept - (np.asarray(coef) * scaler_mean / scaler_scale).sum())
    head = nn.Linear(backbone_dim, 2)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.weight[1] = torch.from_numpy(w)
        head.bias[1] = b
    return head


def build_probe_detector(backbone_id: str, features_dir: pathlib.Path = FEATURES,
                         C: float = 0.01, device: str = "cuda") -> Detector:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    stem = f"{backbone_id.split('.')[0]}_fit_clean"
    x = np.load(features_dir / f"{stem}.npy")
    meta = json.loads((features_dir / f"{stem}.json").read_text())
    y = np.asarray(meta["y_binary"])

    sc = StandardScaler().fit(x)
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(x), y)
    head = _fold_probe(x.shape[1], clf.coef_[0], float(clf.intercept_[0]), sc.mean_, sc.scale_)

    backbone = timm.create_model(backbone_id, pretrained=True, num_classes=0)
    cfg = timm.data.resolve_model_data_config(backbone)
    module = NormalizedModel(backbone, head, cfg["mean"], cfg["std"]).eval().to(device)
    for p in module.parameters():
        p.requires_grad_(False)

    return Detector(
        module=module, transform=eval_transform(cfg),
        model_id=backbone_id.split(".")[0], backbone=backbone_id,
        provenance={"kind": "probe", "backbone": backbone_id, "C": C,
                    "fit_n": int(x.shape[0]), "feature_dim": int(x.shape[1]),
                    "fit_train_acc": float(clf.score(sc.transform(x), y)),
                    "features_dir": str(features_dir)},
    )


def new_trainable(backbone_id: str, device: str = "cuda") -> tuple[NormalizedModel, transforms.Compose, dict]:
    """A fresh end-to-end trainable detector: pretrained backbone + a 2-way head, all
    parameters requiring grad. WP3 trains these; the module is byte-compatible with what
    WP2 attacks and WP4 serves, so a robust checkpoint drops in with no adapter."""
    backbone = timm.create_model(backbone_id, pretrained=True, num_classes=0)
    cfg = timm.data.resolve_model_data_config(backbone)
    head = nn.Linear(backbone.num_features, 2)
    module = NormalizedModel(backbone, head, cfg["mean"], cfg["std"]).to(device)
    for p in module.parameters():
        p.requires_grad_(True)
    return module, eval_transform(cfg), cfg


def save_checkpoint(module: NormalizedModel, path: pathlib.Path, meta: dict) -> pathlib.Path:
    """Weights plus the full training provenance. The sidecar is committed; the .pt is not
    (it is bulk), so the JSON must contain everything needed to reproduce the run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": module.state_dict(), "meta": meta}, path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return path


def build_checkpoint_detector(path: pathlib.Path, device: str = "cuda") -> Detector:
    blob = torch.load(path, map_location=device, weights_only=False)
    meta = blob["meta"]
    module, tf, _ = new_trainable(meta["backbone"], device)
    module.load_state_dict(blob["state_dict"])
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return Detector(module=module, transform=tf,
                    model_id=meta["model_id"], backbone=meta["backbone"],
                    provenance={"kind": "checkpoint", "path": str(path), **meta})


def build(spec: str, features_dir: pathlib.Path = FEATURES, device: str = "cuda") -> Detector:
    """`probe:<timm_id>` or `ckpt:<path>`. The single entry point for WP2, WP3 eval and WP4."""
    kind, _, rest = spec.partition(":")
    if kind == "probe":
        return build_probe_detector(rest, features_dir, device=device)
    if kind == "ckpt":
        p = pathlib.Path(rest)
        return build_checkpoint_detector(p if p.is_absolute() else CKPT_DIR / p, device=device)
    raise ValueError(f"unknown model spec {spec!r}; expected 'probe:<timm_id>' or 'ckpt:<path>'")
