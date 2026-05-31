"""Open-vocabulary object/concept attribution for attention heatmaps.

This is the optional "human ontology" layer for the report. It uses CLIPSeg to
ask broad visual questions such as road/sky/vegetation/building and measures how
much rollout attention lands on each concept mask.

Run this after `attention_rollout_clip.py`, because it consumes the saved
attention heatmaps.

Example:
    python analysis/object_concept_attention.py ^
      --attention-metrics analysis_outputs/streetclip_attention/attention_metrics.csv ^
      --out-dir analysis_outputs/streetclip_concepts ^
      --max-samples 500 ^
      --clipseg-model CIDAS/clipseg-rd64-refined ^
      --no-local-files-only
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.amp import autocast
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_CONCEPTS = [
    "road",
    "sky",
    "trees",
    "grass",
    "building",
    "vehicle",
    "traffic sign",
    "water",
    "mountain",
    "field",
]


def parse_concepts(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resize_mask(mask: torch.Tensor, size: Sequence[int]) -> np.ndarray:
    mask = mask[None, None].float()
    resized = F.interpolate(mask, size=tuple(size), mode="bilinear", align_corners=False)
    return resized.squeeze().detach().cpu().numpy().astype(np.float32)


def attention_mass(attention: np.ndarray, mask: np.ndarray) -> float:
    attention = np.maximum(attention.astype(np.float64), 0)
    mask = np.maximum(mask.astype(np.float64), 0)
    denom = float(attention.sum())
    if denom <= 0:
        return 0.0
    return float((attention * mask).sum() / denom)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attention-metrics", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--clipseg-model", default="CIDAS/clipseg-rd64-refined")
    ap.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--concepts", default=",".join(DEFAULT_CONCEPTS))
    ap.add_argument("--max-samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = ap.parse_args()

    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    concepts = parse_concepts(args.concepts)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.attention_metrics)
    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    df["correct"] = df["correct"].astype(str).str.lower().isin(["true", "1", "yes"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None if args.amp_dtype == "fp32" else (torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16)
    processor = CLIPSegProcessor.from_pretrained(args.clipseg_model, local_files_only=args.local_files_only)
    model = CLIPSegForImageSegmentation.from_pretrained(args.clipseg_model, local_files_only=args.local_files_only)
    model.to(device).eval()

    rows: List[Dict[str, object]] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="CLIPSeg concept attribution"):
        image_path = Path(row["file"])
        heatmap_path = Path(row["heatmap_file"])
        if not image_path.exists() or not heatmap_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        attention = np.load(heatmap_path).astype(np.float32)
        if attention.ndim != 2:
            continue

        images = [image] * len(concepts)
        inputs = processor(text=concepts, images=images, padding=True, return_tensors="pt")
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        with torch.no_grad(), autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            outputs = model(**inputs)
            masks = outputs.logits.sigmoid()

        base = {
            "model": row.get("model", ""),
            "file": row["file"],
            "actual_country": row["actual_country"],
            "predicted_country": row["predicted_country"],
            "correct": bool(row["correct"]),
            "confidence": float(row["confidence"]),
        }
        for concept, mask in zip(concepts, masks):
            mask_np = resize_mask(mask, attention.shape)
            rows.append({
                **base,
                "concept": concept,
                "attention_mass": attention_mass(attention, mask_np),
                "mask_mean": float(mask_np.mean()),
                "mask_coverage_0p5": float((mask_np >= 0.5).mean()),
            })

    out_csv = args.out_dir / "concept_attention.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model",
            "file",
            "actual_country",
            "predicted_country",
            "correct",
            "confidence",
            "concept",
            "attention_mass",
            "mask_mean",
            "mask_coverage_0p5",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    result = pd.DataFrame(rows)
    if result.empty:
        raise SystemExit("No concept rows were produced; check paths and CLIPSeg model availability.")

    summary = result.groupby(["concept", "correct"]).agg(
        n=("attention_mass", "size"),
        mean_attention_mass=("attention_mass", "mean"),
        mean_mask_coverage=("mask_coverage_0p5", "mean"),
    ).reset_index()
    summary.to_csv(args.out_dir / "concept_attention_summary_by_correctness.csv", index=False)

    pivot = summary.pivot(index="concept", columns="correct", values="mean_attention_mass").fillna(0)
    pivot = pivot.reindex(concepts)
    plt.figure(figsize=(8, 4.2))
    x = np.arange(len(pivot))
    wrong = pivot[False] if False in pivot.columns else np.zeros(len(pivot))
    correct = pivot[True] if True in pivot.columns else np.zeros(len(pivot))
    plt.bar(x - 0.18, wrong, width=0.36, label="wrong", color="#999999")
    plt.bar(x + 0.18, correct, width=0.36, label="correct", color="#D55E00")
    plt.xticks(x, pivot.index, rotation=35, ha="right")
    plt.ylabel("mean attention mass")
    plt.title("Attention mass by CLIPSeg concept")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "concept_attention_correct_vs_wrong.png", dpi=180)
    plt.close()

    lines = [
        "Concept attention summary",
        "=" * 60,
        f"images sampled: {result['file'].nunique()}",
        f"concepts: {', '.join(concepts)}",
        "",
        summary.to_string(index=False),
        "",
        "Interpretation note:",
        "Saved rollout heatmaps are normalized attention mass. These CLIPSeg masks are open-vocabulary predictions, not manually verified semantic labels.",
        "Use the numbers as approximate ontology-level evidence, not as proof that an object caused the prediction.",
    ]
    (args.out_dir / "concept_attention_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
