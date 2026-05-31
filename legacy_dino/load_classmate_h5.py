"""Load classmate's Keras `.weights.h5` into a PyTorch (timm) model.

The classmate trained:
    timm.vit_small_patch14_dinov2  (frozen, num_classes=0, dynamic_img_size=True)
        |
    Dense(384 -> 512, ReLU) -> Dropout(0.4) -> Dense(512 -> 22, softmax)

Saved with Keras 3 + PyTorch backend, so:
  - Conv2d weights are already in PyTorch (out, in, H, W) layout (no transpose needed)
  - Dense weights are in Keras (in, out) layout — must be transposed for nn.Linear
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import h5py
import timm
import torch
from torch import nn


# Class list inferred from classmate's test_report.txt — alphabetical, 22 countries.
CLASSMATE_CLASSES: List[str] = [
    "argentina", "australia", "austria", "brazil", "canada", "chile",
    "colombia", "croatia", "france", "germany", "hungary", "india",
    "indonesia", "italy", "japan", "kenya", "malaysia", "mexico",
    "poland", "spain", "sweden", "usa",
]

BACKBONE_PREFIX = "layers/dino_feature_extractor/dino/vars/"


class GeoClassifier(nn.Module):
    """DINOv2 ViT-S/14 backbone + 2-layer MLP head matching the classmate's Keras model."""

    def __init__(self, num_classes: int = 22, hidden: int = 512, dropout: float = 0.4,
                 backbone_name: str = "vit_small_patch14_dinov2"):
        super().__init__()
        # NOTE: pretrained=False here — we'll overwrite weights from the .h5 in load_from_h5().
        # dynamic_img_size=True so pos_embed (saved at native 1370 tokens) interpolates per-input.
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, num_classes=0, dynamic_img_size=True
        )
        self.fc1 = nn.Linear(384, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)            # [B, 384] CLS embedding (num_classes=0 returns features)
        x = self.relu(self.fc1(feats))
        x = self.dropout(x)
        return self.fc2(x)


def load_from_h5(model: GeoClassifier, h5_path: Path, strict: bool = True) -> Tuple[List[str], List[str]]:
    """Copy weights from the Keras .h5 into the PyTorch model in-place.

    Returns (missing_keys, unexpected_h5_keys).
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        # ---- Backbone ----
        backbone_state = {}
        for key in f[BACKBONE_PREFIX].keys():
            arr = f[BACKBONE_PREFIX + key][...]
            backbone_state[key] = torch.from_numpy(arr)

        missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
        if strict and (missing or unexpected):
            # filter expected mismatches: head removed (num_classes=0) shouldn't have any head keys
            real_missing = [k for k in missing if not k.startswith("head")]
            if real_missing or unexpected:
                raise RuntimeError(
                    f"Backbone load mismatch.\nMissing in h5: {real_missing}\nUnexpected in h5: {unexpected}"
                )

        # ---- Head ----
        # Keras Dense kernel shape (in, out) -> PyTorch Linear weight (out, in): transpose.
        d0_w = torch.from_numpy(f["layers/dense/vars/0"][...]).T.contiguous()    # (512, 384)
        d0_b = torch.from_numpy(f["layers/dense/vars/1"][...])                   # (512,)
        d1_w = torch.from_numpy(f["layers/dense_1/vars/0"][...]).T.contiguous()  # (22, 512)
        d1_b = torch.from_numpy(f["layers/dense_1/vars/1"][...])                 # (22,)

        with torch.no_grad():
            assert model.fc1.weight.shape == d0_w.shape, f"fc1 shape mismatch: {model.fc1.weight.shape} vs {d0_w.shape}"
            assert model.fc2.weight.shape == d1_w.shape, f"fc2 shape mismatch: {model.fc2.weight.shape} vs {d1_w.shape}"
            model.fc1.weight.copy_(d0_w)
            model.fc1.bias.copy_(d0_b)
            model.fc2.weight.copy_(d1_w)
            model.fc2.bias.copy_(d1_b)

        return list(missing), list(unexpected)


def build_classmate_model(h5_path: Path, num_classes: int = 22) -> GeoClassifier:
    """Convenience: build model + load classmate weights in one call."""
    model = GeoClassifier(num_classes=num_classes)
    load_from_h5(model, h5_path)
    model.eval()
    return model


if __name__ == "__main__":
    # Smoke test
    import sys
    h5 = Path(__file__).parent / "dino_geo_28_countries_full.weights.h5"
    print(f"Loading {h5}")
    m = build_classmate_model(h5)
    print(f"Model OK. Total params: {sum(p.numel() for p in m.parameters())/1e6:.1f}M")
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = m(x)
    print(f"Forward OK. Output shape: {out.shape}  argmax class: {CLASSMATE_CLASSES[out.argmax(1).item()]}")
