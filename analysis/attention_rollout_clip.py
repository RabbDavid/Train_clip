"""Attention rollout and quantitative focus metrics for DFN2B-CLIP/StreetCLIP.

This script is meant for the GPU machine after training has finished. It
creates the systematic evidence the report needs:

- prediction correctness and confidence,
- attention entropy / concentration metrics,
- qualitative original/heatmap/overlay panels,
- per-model summary charts.

Examples:
    python analysis/attention_rollout_clip.py ^
      --model streetclip ^
      --data-root TRAIN_DATASET/koglab_levi ^
      --model-dir MODEL/StreetCLIP ^
      --checkpoint runs_streetclip/20260519-221734/best.pt ^
      --out-dir analysis_outputs/streetclip_attention ^
      --max-samples-per-country 60 --batch-size 8

    python analysis/attention_rollout_clip.py ^
      --model dfn2b ^
      --data-root TRAIN_DATASET/koglab_levi ^
      --model-dir MODEL/DFN2B-CLIP-ViT-B-16 ^
      --checkpoint runs_clip/20260519-213048/best.pt ^
      --out-dir analysis_outputs/dfn2b_attention ^
      --max-samples-per-country 60 --batch-size 8
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.amp import autocast
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
COUNTRY_ALIASES = {"columbia": "colombia"}


def country_name(name: str) -> str:
    name = name.lower()
    if name.endswith("_test"):
        name = name[: -len("_test")]
    return COUNTRY_ALIASES.get(name, name)


def image_files(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def scan_test_samples(data_root: Path, classes: Optional[Sequence[str]]) -> Tuple[List[Tuple[Path, str]], List[str]]:
    requested = set(classes) if classes else None
    samples: List[Tuple[Path, str]] = []
    if (data_root / "test").exists():
        roots = [p for p in (data_root / "test").iterdir() if p.is_dir()]
    else:
        roots = [p for p in data_root.iterdir() if p.is_dir() and p.name.lower().endswith("_test")]

    for folder in sorted(roots):
        label = country_name(folder.name)
        if requested is not None and label not in requested:
            continue
        samples.extend((p, label) for p in image_files(folder))

    inferred_classes = sorted({label for _, label in samples})
    if not inferred_classes:
        raise SystemExit(f"No test images found under {data_root}")
    return samples, inferred_classes


def limit_per_class(samples: Sequence[Tuple[Path, str]], max_per_class: int, seed: int) -> List[Tuple[Path, str]]:
    if max_per_class <= 0:
        return list(samples)
    rng = random.Random(seed)
    grouped: Dict[str, List[Tuple[Path, str]]] = {}
    for sample in samples:
        grouped.setdefault(sample[1], []).append(sample)
    out: List[Tuple[Path, str]] = []
    for label in sorted(grouped):
        items = grouped[label]
        rng.shuffle(items)
        out.extend(items[:max_per_class])
    rng.shuffle(out)
    return out


def batches(items: Sequence[Tuple[Path, str]], batch_size: int) -> Iterable[List[Tuple[Path, str]]]:
    for idx in range(0, len(items), batch_size):
        yield list(items[idx : idx + batch_size])


def parse_amp_dtype(name: str):
    if name == "fp32":
        return None
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(name)


def safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in path.stem)[:160]


def load_images(paths: Sequence[Path]) -> List[Image.Image]:
    return [Image.open(p).convert("RGB") for p in paths]


def normalized_heatmap(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = values - values.min()
    denom = values.max()
    if denom > 0:
        values = values / denom
    return values


def heatmap_metrics(heatmap: np.ndarray) -> Dict[str, float]:
    flat = np.asarray(heatmap, dtype=np.float64).reshape(-1)
    flat = np.maximum(flat, 0)
    total = float(flat.sum())
    if total <= 0:
        return {
            "attention_entropy": 1.0,
            "attention_top1pct_mass": 0.0,
            "attention_top5pct_mass": 0.0,
            "attention_top10pct_mass": 0.0,
            "attention_peak_to_mean": 0.0,
        }

    prob = flat / total
    entropy = -float(np.sum(prob * np.log(prob + 1e-12))) / math.log(len(prob))
    sorted_prob = np.sort(prob)[::-1]

    def top_mass(frac: float) -> float:
        n = max(1, int(math.ceil(len(sorted_prob) * frac)))
        return float(sorted_prob[:n].sum())

    return {
        "attention_entropy": entropy,
        "attention_top1pct_mass": top_mass(0.01),
        "attention_top5pct_mass": top_mass(0.05),
        "attention_top10pct_mass": top_mass(0.10),
        "attention_peak_to_mean": float(prob.max() * len(prob)),
    }


def rollout_from_attentions(attentions: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return CLS-to-patch rollout for each batch item.

    attentions are expected as [B, heads, tokens, tokens].
    """
    if not attentions:
        raise ValueError("No attention tensors captured")
    first = attentions[0]
    batch, _, tokens, _ = first.shape
    device = first.device
    rollout = torch.eye(tokens, device=device).unsqueeze(0).repeat(batch, 1, 1)
    eye = torch.eye(tokens, device=device).unsqueeze(0)
    for attn in attentions:
        attn = attn.float().mean(dim=1)
        attn = attn + eye
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rollout = torch.bmm(attn, rollout)
    return rollout[:, 0, 1:]


def patch_rollout_to_image(mask: torch.Tensor, image_size: Tuple[int, int]) -> np.ndarray:
    n_patches = int(mask.numel())
    side = int(round(math.sqrt(n_patches)))
    if side * side != n_patches:
        raise ValueError(f"Cannot infer square patch grid from {n_patches} patches")
    grid = mask.reshape(1, 1, side, side)
    resized = F.interpolate(grid, size=(image_size[1], image_size[0]), mode="bilinear", align_corners=False)
    return normalized_heatmap(resized.squeeze().detach().cpu().numpy())


def colorize_heatmap(heatmap: np.ndarray) -> Image.Image:
    cmap = plt.get_cmap("magma")
    rgba = cmap(normalized_heatmap(heatmap))
    arr = (rgba[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(arr)


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float) -> Image.Image:
    heat = colorize_heatmap(heatmap).resize(image.size)
    return Image.blend(image.convert("RGB"), heat.convert("RGB"), alpha=alpha)


def font(size: int):
    for candidate in [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def save_panel(
    image: Image.Image,
    heatmap: np.ndarray,
    out_path: Path,
    title: str,
    alpha: float = 0.45,
) -> None:
    target_w = 420
    image_r = image.resize((target_w, int(image.height * target_w / image.width)))
    heat_r = colorize_heatmap(heatmap).resize(image_r.size)
    over_r = overlay_heatmap(image, heatmap, alpha=alpha).resize(image_r.size)

    title_h = 54
    gap = 12
    w = image_r.width * 3 + gap * 2
    h = image_r.height + title_h + 34
    panel = Image.new("RGB", (w, h), (250, 250, 250))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), title, fill=(20, 20, 20), font=font(20))
    for idx, (label, part) in enumerate([("image", image_r), ("rollout", heat_r), ("overlay", over_r)]):
        x = idx * (image_r.width + gap)
        panel.paste(part, (x, title_h))
        draw.text((x + 8, title_h + image_r.height + 5), label, fill=(40, 40, 40), font=font(17))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path, quality=92)


@contextmanager
def capture_openclip_attentions(model):
    """Patch OpenCLIP residual blocks to capture per-head attention weights."""
    visual = getattr(model, "visual", None)
    transformer = getattr(visual, "transformer", None)
    blocks = list(getattr(transformer, "resblocks", []))
    if not blocks:
        raise RuntimeError("Could not find OpenCLIP visual transformer residual blocks")

    captured: List[torch.Tensor] = []
    originals = []

    for block in blocks:
        original = block.attention
        originals.append((block, original))

        def patched_attention(*args, _block=block, **kwargs):
            q_x = kwargs.pop("q_x", None)
            if q_x is None:
                if not args:
                    raise TypeError("OpenCLIP attention patch did not receive q_x")
                q_x = args[0]
                args = args[1:]
            k_x = kwargs.pop("k_x", None)
            v_x = kwargs.pop("v_x", None)
            attn_mask = kwargs.pop("attn_mask", None)
            if args and attn_mask is None:
                attn_mask = args[0]
            k_x = q_x if k_x is None else k_x
            v_x = q_x if v_x is None else v_x
            out, weights = _block.attn(
                q_x,
                k_x,
                v_x,
                need_weights=True,
                average_attn_weights=False,
                attn_mask=attn_mask,
            )
            batch_dim = q_x.shape[0] if getattr(_block.attn, "batch_first", False) else q_x.shape[1]
            if weights.ndim == 4 and weights.shape[0] != batch_dim:
                # Some PyTorch versions may return [heads, batch, tokens, tokens].
                weights = weights.permute(1, 0, 2, 3)
            captured.append(weights.detach())
            return out

        block.attention = patched_attention

    try:
        yield captured
    finally:
        for block, original in originals:
            block.attention = original


class StreetCLIPEvaluator:
    def __init__(self, args, classes: Sequence[str], device: torch.device):
        sys.path.insert(0, str(ROOT / "Code"))
        from train_streetclip_country import StreetCLIPClassifier, load_clip_model
        from transformers import CLIPImageProcessor

        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        ckpt_classes = checkpoint.get("classes")
        if ckpt_classes:
            classes = ckpt_classes
        self.classes = list(classes)
        self.device = device
        self.amp_dtype = parse_amp_dtype(args.amp_dtype)
        clip = load_clip_model(args.model_dir, args.attn_implementation)
        self.processor = CLIPImageProcessor.from_pretrained(args.model_dir, local_files_only=args.local_files_only)
        self.model = StreetCLIPClassifier(clip, num_classes=len(self.classes), dropout=0.0)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(device).eval()

    def predict_with_rollout(self, images: Sequence[Image.Image]):
        inputs = self.processor(images=list(images), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, non_blocking=True)
        with torch.no_grad(), autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
            logits = self.model(pixel_values)
            vision_outputs = self.model.clip.vision_model(pixel_values=pixel_values, output_attentions=True, return_dict=True)
        probs = logits.softmax(dim=-1)
        rollout = rollout_from_attentions(list(vision_outputs.attentions))
        heatmaps = [patch_rollout_to_image(rollout[i], images[i].size) for i in range(len(images))]
        return probs.detach().cpu(), heatmaps


class DFN2BEvaluator:
    def __init__(self, args, classes: Sequence[str], device: torch.device):
        sys.path.insert(0, str(ROOT))
        import open_clip
        from train_clip_country import PROMPT_TEMPLATES, clip_logits, get_tokenizer, text_prototypes

        self.clip_logits = clip_logits
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        ckpt_classes = checkpoint.get("classes")
        if ckpt_classes:
            classes = ckpt_classes
        self.classes = list(classes)
        self.templates = checkpoint.get("templates", PROMPT_TEMPLATES)
        self.device = device
        self.amp_dtype = parse_amp_dtype(args.amp_dtype)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(f"local-dir:{args.model_dir}")
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(device).eval()
        tokenizer_args = argparse.Namespace(tokenizer=args.tokenizer, model_name=args.model_name)
        self.tokenizer = get_tokenizer(tokenizer_args)
        self.class_text = text_prototypes(self.model, self.tokenizer, self.classes, self.templates, device, self.amp_dtype)

    def predict_with_rollout(self, images: Sequence[Image.Image]):
        tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device, non_blocking=True)
        with torch.no_grad(), capture_openclip_attentions(self.model) as captured:
            captured.clear()
            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                logits = self.clip_logits(self.model, tensors, self.class_text)
            attentions = list(captured)
        probs = logits.softmax(dim=-1)
        rollout = rollout_from_attentions(attentions)
        heatmaps = [patch_rollout_to_image(rollout[i], images[i].size) for i in range(len(images))]
        return probs.detach().cpu(), heatmaps


def make_evaluator(args, classes: Sequence[str], device: torch.device):
    if args.model == "streetclip":
        return StreetCLIPEvaluator(args, classes, device)
    if args.model == "dfn2b":
        return DFN2BEvaluator(args, classes, device)
    raise ValueError(args.model)


def write_summary(metrics_path: Path, out_dir: Path, model_name: str) -> None:
    df = pd.read_csv(metrics_path)
    grouped = df.groupby("correct").agg(
        n=("correct", "size"),
        accuracy=("correct", "mean"),
        confidence=("confidence", "mean"),
        entropy=("attention_entropy", "mean"),
        top10_mass=("attention_top10pct_mass", "mean"),
        peak_to_mean=("attention_peak_to_mean", "mean"),
    )
    grouped.to_csv(out_dir / "attention_summary_by_correctness.csv")

    by_country = df.groupby("actual_country").agg(
        n=("correct", "size"),
        accuracy=("correct", "mean"),
        confidence=("confidence", "mean"),
        entropy=("attention_entropy", "mean"),
        top10_mass=("attention_top10pct_mass", "mean"),
    ).sort_values("accuracy", ascending=False)
    by_country.to_csv(out_dir / "attention_summary_by_country.csv")

    lines = [
        f"{model_name} attention rollout summary",
        "=" * 60,
        f"images: {len(df)}",
        f"accuracy on sampled images: {df['correct'].mean():.4f}",
        "",
        "By correctness:",
        grouped.to_string(),
        "",
        "Top countries by sampled accuracy:",
        by_country.head(8).to_string(),
        "",
        "Bottom countries by sampled accuracy:",
        by_country.tail(8).to_string(),
    ]
    (out_dir / "attention_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plt.figure(figsize=(5, 3.2))
    values = [
        df.loc[df["correct"] == False, "attention_entropy"],  # noqa: E712
        df.loc[df["correct"] == True, "attention_entropy"],  # noqa: E712
    ]
    plt.boxplot(values, labels=["wrong", "correct"], showfliers=False)
    plt.ylabel("normalized attention entropy")
    plt.title(f"{model_name}: attention focus by correctness")
    plt.tight_layout()
    plt.savefig(out_dir / "attention_entropy_correct_vs_wrong.png", dpi=180)
    plt.close()

    plt.figure(figsize=(5, 3.2))
    plt.scatter(df["confidence"], df["attention_entropy"], s=12, alpha=0.45)
    plt.xlabel("prediction confidence")
    plt.ylabel("normalized attention entropy")
    plt.title(f"{model_name}: confidence vs attention entropy")
    plt.tight_layout()
    plt.savefig(out_dir / "attention_entropy_vs_confidence.png", dpi=180)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["streetclip", "dfn2b"], required=True)
    ap.add_argument("--data-root", type=Path, default=Path("TRAIN_DATASET/koglab_levi"))
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--classes", default=None, help="optional comma-separated country list")
    ap.add_argument("--max-samples-per-country", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--save-panels", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--attn-implementation", choices=["auto", "eager", "sdpa", "flash_attention_2"], default="sdpa")
    ap.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--model-name", default="ViT-B-16")
    ap.add_argument("--tokenizer", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = args.out_dir / "heatmaps"
    panel_dir = args.out_dir / "panels"
    heatmap_dir.mkdir(exist_ok=True)
    panel_dir.mkdir(exist_ok=True)

    requested = [c.strip().lower() for c in args.classes.split(",")] if args.classes else None
    samples, inferred_classes = scan_test_samples(args.data_root, requested)
    samples = limit_per_class(samples, args.max_samples_per_country, args.seed)
    evaluator = make_evaluator(args, inferred_classes, device)
    class_to_idx = {c: i for i, c in enumerate(evaluator.classes)}

    rows: List[Dict[str, object]] = []
    panel_count = 0
    for batch in tqdm(list(batches(samples, args.batch_size)), desc=f"{args.model} attention rollout"):
        paths = [p for p, _ in batch]
        labels = [label for _, label in batch]
        images = load_images(paths)
        probs, heatmaps = evaluator.predict_with_rollout(images)
        top_values, top_indices = probs.topk(min(5, probs.shape[1]), dim=1)
        preds = probs.argmax(dim=1).tolist()

        for idx, (path, actual, image, heatmap) in enumerate(zip(paths, labels, images, heatmaps)):
            if actual not in class_to_idx:
                continue
            predicted = evaluator.classes[int(preds[idx])]
            confidence = float(probs[idx, preds[idx]])
            correct = predicted == actual
            heat_name = f"{args.model}_{country_name(actual)}_{safe_stem(path)}.npy"
            heat_path = heatmap_dir / heat_name
            np.save(heat_path, heatmap.astype(np.float32))

            metric_values = heatmap_metrics(heatmap)
            row = {
                "model": args.model,
                "file": str(path),
                "actual_country": actual,
                "predicted_country": predicted,
                "correct": bool(correct),
                "confidence": confidence,
                "top5_countries": "|".join(evaluator.classes[int(i)] for i in top_indices[idx].tolist()),
                "top5_confidences": "|".join(f"{float(v):.6f}" for v in top_values[idx].tolist()),
                "heatmap_file": str(heat_path),
                **metric_values,
            }
            rows.append(row)

            if panel_count < args.save_panels:
                title = f"{args.model} | true={actual} pred={predicted} conf={confidence:.2f} correct={correct}"
                panel_name = f"{panel_count:03d}_{args.model}_{actual}_pred-{predicted}_{safe_stem(path)}.jpg"
                save_panel(image, heatmap, panel_dir / panel_name, title)
                panel_count += 1

    metrics_path = args.out_dir / "attention_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=",")
        writer.writeheader()
        writer.writerows(rows)

    write_summary(metrics_path, args.out_dir, args.model)
    print(f"Wrote {metrics_path}")
    print(f"Wrote panels to {panel_dir}")


if __name__ == "__main__":
    main()
