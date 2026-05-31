"""Load Levi's DINOv2 Keras/PyTorch-backend `.weights.h5` exports.

Supports both architectures found in the submitted files:

- models 0-3: DINOv2 ViT-S/14 + Dense(384->512) + Dropout + Dense(512->22)
- models 4-5: DINOv2 ViT-S/14 + BatchNorm + Dense(384->1024) + Dropout
              + Dense(1024->512) + Dropout + Dense(512->22)

The DINO backbone is stored with PyTorch-shaped tensors inside the H5 file, so
the backbone can be loaded directly. Keras Dense kernels are transposed.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import h5py
import timm
import torch
from torch import nn


CLASSMATE_CLASSES: List[str] = [
    "argentina", "australia", "austria", "brazil", "canada", "chile",
    "colombia", "croatia", "france", "germany", "hungary", "india",
    "indonesia", "italy", "japan", "kenya", "malaysia", "mexico",
    "poland", "spain", "sweden", "usa",
]

BACKBONE_PREFIX = "layers/dino_feature_extractor/dino/vars/"


class OldDinoClassifier(nn.Module):
    def __init__(self, num_classes: int = 22, hidden: int = 512, dropout: float = 0.4):
        super().__init__()
        self.backbone = timm.create_model("vit_small_patch14_dinov2", pretrained=False, num_classes=0, dynamic_img_size=True)
        self.fc1 = nn.Linear(384, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class PyramidDinoClassifier(nn.Module):
    def __init__(self, num_classes: int = 22, dropout: float = 0.4):
        super().__init__()
        self.backbone = timm.create_model("vit_small_patch14_dinov2", pretrained=False, num_classes=0, dynamic_img_size=True)
        self.bn = nn.BatchNorm1d(384)
        self.fc1 = nn.Linear(384, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.bn(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


def detect_architecture(h5_path: Path) -> str:
    with h5py.File(h5_path, "r") as f:
        if "layers/dense_2/vars/0" in f:
            return "pyramid"
        return "old"


def load_backbone(model: nn.Module, f: h5py.File) -> Tuple[List[str], List[str]]:
    backbone_state = {}
    for key in f[BACKBONE_PREFIX].keys():
        backbone_state[key] = torch.from_numpy(f[BACKBONE_PREFIX + key][...])
    missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
    return list(missing), list(unexpected)


def load_dense(linear: nn.Linear, f: h5py.File, key: str) -> None:
    weight = torch.from_numpy(f[f"{key}/vars/0"][...]).T.contiguous()
    bias = torch.from_numpy(f[f"{key}/vars/1"][...])
    with torch.no_grad():
        if linear.weight.shape != weight.shape:
            raise RuntimeError(f"{key} weight shape mismatch: {linear.weight.shape} vs {weight.shape}")
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)


def load_batchnorm(bn: nn.BatchNorm1d, f: h5py.File, key: str = "layers/batch_normalization") -> None:
    with torch.no_grad():
        bn.weight.copy_(torch.from_numpy(f[f"{key}/vars/0"][...]))
        bn.bias.copy_(torch.from_numpy(f[f"{key}/vars/1"][...]))
        bn.running_mean.copy_(torch.from_numpy(f[f"{key}/vars/2"][...]))
        bn.running_var.copy_(torch.from_numpy(f[f"{key}/vars/3"][...]))


def build_model_from_h5(h5_path: Path, num_classes: int = 22) -> nn.Module:
    h5_path = Path(h5_path)
    arch = detect_architecture(h5_path)
    model: nn.Module
    if arch == "pyramid":
        model = PyramidDinoClassifier(num_classes=num_classes)
    else:
        model = OldDinoClassifier(num_classes=num_classes)

    with h5py.File(h5_path, "r") as f:
        load_backbone(model, f)
        if arch == "pyramid":
            assert isinstance(model, PyramidDinoClassifier)
            load_batchnorm(model.bn, f)
            load_dense(model.fc1, f, "layers/dense")
            load_dense(model.fc2, f, "layers/dense_1")
            load_dense(model.fc3, f, "layers/dense_2")
        else:
            assert isinstance(model, OldDinoClassifier)
            load_dense(model.fc1, f, "layers/dense")
            load_dense(model.fc2, f, "layers/dense_1")

    model.eval()
    return model


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("h5", type=Path)
    args = ap.parse_args()
    model = build_model_from_h5(args.h5)
    print(f"Loaded {args.h5} as {model.__class__.__name__}")
    print(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
