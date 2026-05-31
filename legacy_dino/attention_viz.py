"""Attention visualization for the classmate's DINOv2 ViT-S/14 country classifier.

Implements:
  * `enable_attention_capture(model)` — patches each block's attention so it
    stores its softmax-normalized attention map after each forward pass.
  * `attention_rollout(...)` — Abnar & Zuidema 2020 rollout combining attention
    across layers (with identity for residual connections).
  * `cls_to_patch_grid(...)` — extract CLS-row attention to patch tokens and
    reshape to a 2D grid.
  * `overlay_heatmap(...)` — alpha-blend a heatmap over the original image.
  * `draw_argmax_circle(...)` — annotate the peak attention location.
  * `make_figure(...)` — convenience: takes an image and the model, returns a
    paper-ready PIL.Image with the original image, the heatmap overlay, and the
    annotated overlay side-by-side.

Designed to run on CPU; one image takes ~300ms on a Ryzen 7 8840HS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import nn
import matplotlib
matplotlib.use("Agg")  # no display required
import matplotlib.cm as cm


# ---------------------------------------------------------------------------
# Attention capture: monkey-patch each timm Attention.forward to store maps
# ---------------------------------------------------------------------------

def _patched_attention_forward(self_attn, x: torch.Tensor, attn_mask=None, is_causal: bool = False) -> torch.Tensor:
    """Drop-in replacement for timm.layers.Attention.forward that stores the
    softmax-normalized attention map in self_attn.attn_map.

    Implements the same math as timm but skips F.scaled_dot_product_attention
    so we can read the attention weights. attn_mask / is_causal are accepted
    for signature parity with current timm but not used (no masking in ViT).
    """
    B, N, C = x.shape
    qkv = self_attn.qkv(x).reshape(B, N, 3, self_attn.num_heads, self_attn.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    # newer timm DINOv2 may carry q_norm / k_norm (no-op identity if not used)
    q = self_attn.q_norm(q) if hasattr(self_attn, "q_norm") else q
    k = self_attn.k_norm(k) if hasattr(self_attn, "k_norm") else k

    attn = (q @ k.transpose(-2, -1)) * self_attn.scale
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = attn.softmax(dim=-1)
    self_attn.attn_map = attn.detach()                 # [B, heads, N, N]
    attn = self_attn.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self_attn.proj(x)
    x = self_attn.proj_drop(x)
    return x


def enable_attention_capture(model: nn.Module) -> List[nn.Module]:
    """Patch every timm Attention block in `model` so that after each forward
    pass `block.attn.attn_map` holds its [B, heads, N, N] attention.

    Returns the list of patched attention modules in block order.
    """
    patched: List[nn.Module] = []
    backbone = getattr(model, "backbone", model)  # support our GeoClassifier wrapper
    for block in backbone.blocks:
        attn = block.attn
        # Bind the patched forward as a bound method on this specific instance.
        attn.forward = _patched_attention_forward.__get__(attn, attn.__class__)
        attn.attn_map = None
        patched.append(attn)
    return patched


def get_attention_maps(model: nn.Module) -> List[torch.Tensor]:
    """After a forward pass on a patched model, collect [heads, N, N] maps per layer (B=1 expected)."""
    backbone = getattr(model, "backbone", model)
    out = []
    for block in backbone.blocks:
        a = block.attn.attn_map
        if a is None:
            raise RuntimeError("Attention not captured. Call enable_attention_capture(model) first.")
        out.append(a)
    return out


# ---------------------------------------------------------------------------
# Attention rollout (Abnar & Zuidema 2020)
# ---------------------------------------------------------------------------

def attention_rollout(
    attentions: Sequence[torch.Tensor],
    head_fusion: str = "mean",
    discard_ratio: float = 0.0,
) -> torch.Tensor:
    """
    attentions: list of [B, heads, N, N] tensors (one per layer, in forward order).
    head_fusion: "mean" | "max" | "min" — how to combine heads.
    discard_ratio: drop this fraction of lowest-attention edges per layer (0 keeps all).

    Returns: [B, N, N] rolled-out attention.
    """
    assert head_fusion in {"mean", "max", "min"}
    rollout: Optional[torch.Tensor] = None
    for attn in attentions:
        if head_fusion == "mean":
            a = attn.mean(dim=1)
        elif head_fusion == "max":
            a = attn.max(dim=1).values
        else:
            a = attn.min(dim=1).values

        if discard_ratio > 0:
            B, N, _ = a.shape
            flat = a.view(B, -1)
            k = int(flat.size(1) * discard_ratio)
            if k > 0:
                _, idx = flat.topk(k, dim=-1, largest=False)
                flat.scatter_(-1, idx, 0.0)
            a = flat.view(B, N, N)

        # add identity (account for residual connection) and renormalize
        I = torch.eye(a.size(-1), device=a.device).unsqueeze(0).expand_as(a)
        a = a + I
        a = a / a.sum(dim=-1, keepdim=True)

        rollout = a if rollout is None else a @ rollout

    assert rollout is not None
    return rollout


def cls_to_patch_grid(rollout: torch.Tensor, grid_h: int, grid_w: int, num_prefix: int = 1) -> torch.Tensor:
    """Take the CLS row of a rollout and reshape it into a 2D patch grid.

    rollout: [B, N, N], CLS at index 0; patch tokens are tokens [num_prefix : num_prefix+grid_h*grid_w].
    Returns [B, grid_h, grid_w].
    """
    cls_to_patches = rollout[:, 0, num_prefix : num_prefix + grid_h * grid_w]
    return cls_to_patches.reshape(rollout.size(0), grid_h, grid_w)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _normalize(arr: np.ndarray) -> np.ndarray:
    a_min, a_max = float(arr.min()), float(arr.max())
    if a_max - a_min < 1e-8:
        return np.zeros_like(arr)
    return (arr - a_min) / (a_max - a_min)


def _normalize_for_display(
    arr: np.ndarray,
    mode: str = "minmax",
    vmax_multiplier: float = 4.0,
    percentile: float = 99.0,
) -> np.ndarray:
    """Normalize a heatmap for visualization without overstating tiny contrasts.

    ``minmax`` stretches each individual image to the full color range. That is
    useful for finding the local maximum, but visually exaggerates small
    differences. ``mass`` first normalizes the map to sum to 1 and maps colors
    against a fixed multiple of uniform patch mass, so colors are more comparable
    across examples. ``percentile`` is a compromise that clips only extreme local
    values.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if mode == "minmax":
        return _normalize(arr)
    if mode == "percentile":
        hi = float(np.percentile(arr, percentile))
        lo = float(arr.min())
        if hi - lo < 1e-8:
            return np.zeros_like(arr)
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    if mode == "mass":
        mass = arr / max(float(arr.sum()), 1e-12)
        uniform = 1.0 / max(1, mass.size)
        vmax = uniform * vmax_multiplier
        return np.clip(mass / max(vmax, 1e-12), 0.0, 1.0)
    raise ValueError(f"Unknown heatmap normalization mode: {mode}")


def upsample_map(attn_map: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Bilinearly upsample a 2D attention map to the target (H, W)."""
    t = torch.from_numpy(attn_map).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=size, mode="bilinear", align_corners=False)
    return t.squeeze().numpy()


def colorize(attn_norm: np.ndarray, cmap_name: str = "jet") -> np.ndarray:
    """Map a [0,1] 2D array to RGB uint8 via matplotlib colormap."""
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(np.clip(attn_norm, 0.0, 1.0))[..., :3]
    return (rgba * 255).astype(np.uint8)


def overlay_heatmap(
    image_pil: Image.Image,
    attn_map: np.ndarray,
    alpha: float = 0.38,
    cmap_name: str = "viridis",
    norm_mode: str = "mass",
    vmax_multiplier: float = 4.0,
) -> Image.Image:
    """Alpha-blend a colorized heatmap over the original image.

    `attn_map` must be 2D; it will be upsampled to image size and normalized to [0,1].
    """
    W, H = image_pil.size
    am = upsample_map(attn_map, (H, W))
    am = _normalize_for_display(am, mode=norm_mode, vmax_multiplier=vmax_multiplier)
    heat = colorize(am, cmap_name)
    heat_pil = Image.fromarray(heat).convert("RGB")
    base = image_pil.convert("RGB")
    return Image.blend(base, heat_pil, alpha=alpha)


def draw_argmax_circle(
    image_pil: Image.Image,
    attn_map: np.ndarray,
    color: Tuple[int, int, int] = (255, 230, 0),
    radius_frac: float = 0.07,
    width: int = 4,
    label: Optional[str] = None,
) -> Image.Image:
    """Draw a circle around the peak attention location.

    radius_frac: circle radius as fraction of min(H, W).
    """
    W, H = image_pil.size
    am = upsample_map(attn_map, (H, W))
    flat_idx = int(np.argmax(am))
    y, x = divmod(flat_idx, W)
    r = int(min(H, W) * radius_frac)

    out = image_pil.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    bbox = [x - r, y - r, x + r, y + r]
    draw.ellipse(bbox, outline=color, width=width)
    # also a small filled dot at the centroid
    dot = 4
    draw.ellipse([x - dot, y - dot, x + dot, y + dot], fill=color)

    if label:
        try:
            font = ImageFont.truetype("arial.ttf", size=max(14, int(min(H, W) * 0.04)))
        except Exception:
            font = ImageFont.load_default()
        draw.text((8, 8), label, fill=color, font=font)
    return out


# ---------------------------------------------------------------------------
# High-level convenience: one image -> paper-ready triptych
# ---------------------------------------------------------------------------

@dataclass
class VizResult:
    pred_idx: int
    pred_label: str
    confidence: float
    rollout_grid: np.ndarray          # [grid_h, grid_w], pre-upsample
    last_layer_attn: np.ndarray       # [grid_h, grid_w], CLS->patches in final layer (no rollout)
    per_head_last_layer: np.ndarray   # [num_heads, grid_h, grid_w] (final layer)
    triptych: Image.Image             # original | heatmap | annotated


def _patch_grid_dims(backbone: nn.Module, img_size: int) -> Tuple[int, int]:
    patch = backbone.patch_embed.patch_size
    if isinstance(patch, (tuple, list)):
        ph, pw = patch
    else:
        ph = pw = int(patch)
    return img_size // ph, img_size // pw


def _num_prefix_tokens(backbone: nn.Module) -> int:
    """CLS + register tokens (DINOv2 lvd142m has 0 register tokens)."""
    n = 1
    if getattr(backbone, "reg_token", None) is not None:
        n += backbone.reg_token.shape[1]
    if hasattr(backbone, "num_prefix_tokens"):
        n = int(backbone.num_prefix_tokens)
    return n


@torch.no_grad()
def make_figure(
    model: nn.Module,
    image_pil: Image.Image,
    classes: Sequence[str],
    img_size: int = 224,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
    device: Optional[torch.device] = None,
    discard_ratio: float = 0.0,
    head_fusion: str = "mean",
    heatmap_alpha: float = 0.38,
    cmap_name: str = "viridis",
    heatmap_norm: str = "mass",
    vmax_multiplier: float = 4.0,
    title: bool = True,
) -> VizResult:
    """Run model on a single PIL image and produce the visualization triptych."""
    device = device or torch.device("cpu")
    model = model.to(device).eval()

    # --- preprocess (identical to classmate's pipeline: resize-warp to 224x224, ImageNet norm) ---
    img224 = image_pil.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img224, dtype=np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    # --- forward ---
    logits = model(tensor)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(probs.argmax())
    pred_label = classes[pred_idx]
    confidence = float(probs[pred_idx])

    # --- attention extraction ---
    attentions = get_attention_maps(model)  # list of [1, heads, N, N]
    backbone = getattr(model, "backbone", model)
    grid_h, grid_w = _patch_grid_dims(backbone, img_size)
    num_prefix = _num_prefix_tokens(backbone)

    rollout = attention_rollout(attentions, head_fusion=head_fusion, discard_ratio=discard_ratio)
    rollout_grid = cls_to_patch_grid(rollout, grid_h, grid_w, num_prefix=num_prefix)[0].cpu().numpy()

    last = attentions[-1]  # [1, heads, N, N]
    per_head_last = last[0, :, 0, num_prefix : num_prefix + grid_h * grid_w].cpu().numpy().reshape(-1, grid_h, grid_w)
    last_layer_mean = per_head_last.mean(axis=0)

    # --- visuals at original image resolution (the user wanted overlays on the real image) ---
    orig = image_pil.convert("RGB")
    heat = overlay_heatmap(
        orig,
        rollout_grid,
        alpha=heatmap_alpha,
        cmap_name=cmap_name,
        norm_mode=heatmap_norm,
        vmax_multiplier=vmax_multiplier,
    )
    annotated = draw_argmax_circle(
        heat,
        rollout_grid,
        color=(255, 230, 0),
        radius_frac=0.07,
        width=4,
        label=(f"{pred_label}  {confidence*100:.1f}%" if title else None),
    )

    # side-by-side triptych
    W, H = orig.size
    canvas = Image.new("RGB", (W * 3 + 16, H), (24, 24, 24))
    canvas.paste(orig, (0, 0))
    canvas.paste(heat, (W + 8, 0))
    canvas.paste(annotated, (2 * W + 16, 0))

    return VizResult(
        pred_idx=pred_idx,
        pred_label=pred_label,
        confidence=confidence,
        rollout_grid=rollout_grid,
        last_layer_attn=last_layer_mean,
        per_head_last_layer=per_head_last,
        triptych=canvas,
    )


def per_head_grid_image(per_head: np.ndarray, base_image: Image.Image, alpha: float = 0.45,
                        cmap_name: str = "viridis", per_row: int = 3,
                        norm_mode: str = "mass", vmax_multiplier: float = 4.0) -> Image.Image:
    """Make a small grid visualizing each attention head (last layer, CLS row)."""
    n_heads = per_head.shape[0]
    rows = (n_heads + per_row - 1) // per_row
    base = base_image.convert("RGB")
    W, H = base.size
    pad = 4
    grid_w = per_row * W + (per_row - 1) * pad
    grid_h = rows * H + (rows - 1) * pad
    canvas = Image.new("RGB", (grid_w, grid_h), (24, 24, 24))
    for h in range(n_heads):
        r, c = divmod(h, per_row)
        overlay = overlay_heatmap(
            base,
            per_head[h],
            alpha=alpha,
            cmap_name=cmap_name,
            norm_mode=norm_mode,
            vmax_multiplier=vmax_multiplier,
        )
        draw = ImageDraw.Draw(overlay)
        try:
            font = ImageFont.truetype("arial.ttf", size=max(14, int(min(H, W) * 0.05)))
        except Exception:
            font = ImageFont.load_default()
        draw.text((6, 4), f"head {h}", fill=(255, 255, 255), font=font)
        canvas.paste(overlay, (c * (W + pad), r * (H + pad)))
    return canvas
