"""Quantitative attention-rollout analysis for the Street View classifier.

This complements the paper figures with dataset-level numbers:
  - does focused vs. diffuse rollout correlate with correct predictions?
  - do bottom/corner artifact regions get more attention in mistakes?

Example:
    python attention_quantitative_eval.py --per-country 80
    python attention_quantitative_eval.py --per-country 0 --batch-note full_test
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from attention_viz import attention_rollout, cls_to_patch_grid, enable_attention_capture, get_attention_maps
from load_classmate_h5 import CLASSMATE_CLASSES, GeoClassifier, load_from_h5


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
COUNTRY_ALIASES = {"columbia": "colombia"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def country_from_folder(folder: Path) -> str:
    country = folder.name.lower()
    if country.endswith("_test"):
        country = country[:-len("_test")]
    return COUNTRY_ALIASES.get(country, country)


def iter_images(data_root: Path, per_country: int, seed: int) -> Iterable[Tuple[Path, str]]:
    rng = random.Random(seed)
    for folder in sorted(data_root.iterdir()):
        if not folder.is_dir() or not folder.name.lower().endswith("_test"):
            continue
        country = country_from_folder(folder)
        if country not in CLASSMATE_CLASSES:
            continue
        images = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
        rng.shuffle(images)
        if per_country > 0:
            images = images[:per_country]
        for path in images:
            yield path, country


def preprocess(path: Path, img_size: int, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return ((arr - MEAN) / STD).to(device)


def region_sum(mass: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> float:
    return float(mass[y0:y1, x0:x1].sum() * 100.0)


def rollout_metrics(grid: np.ndarray) -> Dict[str, float]:
    mass = grid.astype(np.float64)
    mass = mass / max(float(mass.sum()), 1e-12)
    flat = mass.reshape(-1)
    h, w = mass.shape

    nonzero = flat[flat > 0]
    entropy = -float(np.sum(nonzero * np.log(nonzero))) / math.log(flat.size)
    top_sorted = np.sort(flat)[::-1]
    max_idx = int(flat.argmax())
    max_row, max_col = divmod(max_idx, w)

    corner = max(1, h // 8)
    center_y0, center_y1 = h // 4, (3 * h) // 4
    center_x0, center_x1 = w // 4, (3 * w) // 4
    bottom_y0 = (3 * h) // 4
    top_y1 = h // 4

    corner_mass = (
        mass[:corner, :corner].sum()
        + mass[:corner, -corner:].sum()
        + mass[-corner:, :corner].sum()
        + mass[-corner:, -corner:].sum()
    ) * 100.0

    return {
        "entropy_norm": entropy,
        "top1_mass_percent": float(top_sorted[:1].sum() * 100.0),
        "top5_mass_percent": float(top_sorted[:5].sum() * 100.0),
        "top10_mass_percent": float(top_sorted[:10].sum() * 100.0),
        "bottom_quarter_mass_percent": region_sum(mass, bottom_y0, h, 0, w),
        "top_quarter_mass_percent": region_sum(mass, 0, top_y1, 0, w),
        "center_mass_percent": region_sum(mass, center_y0, center_y1, center_x0, center_x1),
        "corner_mass_percent": float(corner_mass),
        "max_row_norm": float((max_row + 0.5) / h),
        "max_col_norm": float((max_col + 0.5) / w),
    }


@torch.no_grad()
def score_image(
    model: GeoClassifier,
    path: Path,
    actual: str,
    img_size: int,
    device: torch.device,
    head_fusion: str,
    discard_ratio: float,
) -> Dict[str, object]:
    x = preprocess(path, img_size, device)
    logits = model(x)
    probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred_idx = int(probs.argmax())
    pred = CLASSMATE_CLASSES[pred_idx]

    attentions = get_attention_maps(model)
    rollout = attention_rollout(attentions, head_fusion=head_fusion, discard_ratio=discard_ratio)
    grid = cls_to_patch_grid(rollout, img_size // 14, img_size // 14)[0].detach().cpu().numpy()

    row: Dict[str, object] = {
        "file": str(path),
        "actual_country": actual,
        "predicted_country": pred,
        "correct": int(pred == actual),
        "confidence": float(probs[pred_idx]),
    }
    row.update(rollout_metrics(grid))
    return row


def safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(score).rank(method="average").to_numpy()
    rank_sum_pos = float(ranks[y == 1].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def write_summary(df: pd.DataFrame, out_dir: Path, batch_note: str) -> None:
    metric_cols = [
        "entropy_norm",
        "top1_mass_percent",
        "top5_mass_percent",
        "top10_mass_percent",
        "bottom_quarter_mass_percent",
        "center_mass_percent",
        "corner_mass_percent",
    ]

    grouped = df.groupby("correct")[metric_cols].agg(["mean", "std", "count"])
    grouped.to_csv(out_dir / "attention_summary_by_correctness.csv", sep=";")

    y = df["correct"].to_numpy()
    auc_rows = []
    for col in ["confidence", *metric_cols]:
        auc = safe_auc(y, df[col].to_numpy())
        auc_rows.append({"metric": col, "auc_for_correct_prediction": "" if auc is None else f"{auc:.4f}"})
    pd.DataFrame(auc_rows).to_csv(out_dir / "attention_metric_auc.csv", index=False, sep=";")

    correct_df = df[df["correct"] == 1]
    wrong_df = df[df["correct"] == 0]
    lines = [
        f"Batch: {batch_note}",
        f"Images: {len(df)}",
        f"Accuracy: {df['correct'].mean() * 100.0:.2f}%",
        "",
        "Metric means: correct vs wrong",
    ]
    for col in metric_cols:
        c_mean = correct_df[col].mean()
        w_mean = wrong_df[col].mean()
        lines.append(f"  {col}: correct={c_mean:.3f}, wrong={w_mean:.3f}, diff={c_mean - w_mean:+.3f}")
    lines.append("")
    lines.append("AUC values are raw direction: 0.5 means no signal; below 0.5 means the inverse direction is more associated with correctness.")
    for row in auc_rows:
        lines.append(f"  {row['metric']}: {row['auc_for_correct_prediction']}")

    (out_dir / "attention_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    means = df.groupby("correct")[metric_cols].mean().rename(index={0: "wrong", 1: "correct"})
    ax = means.T.plot(kind="bar", figsize=(12, 6))
    ax.set_ylabel("metric value")
    ax.set_title("Attention rollout metrics: correct vs. wrong predictions")
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    plt.savefig(out_dir / "attention_correct_vs_wrong.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    colors = np.where(df["correct"].to_numpy() == 1, "#2b8a3e", "#c92a2a")
    plt.scatter(df["entropy_norm"], df["confidence"], c=colors, alpha=0.55, s=18)
    plt.xlabel("normalized rollout entropy (higher = more diffuse)")
    plt.ylabel("prediction confidence")
    plt.title("Attention diffusion vs. model confidence")
    plt.tight_layout()
    plt.savefig(out_dir / "attention_entropy_vs_confidence.png", dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path, default=Path("dino_geo_28_countries_full.weights.h5"))
    ap.add_argument("--data-root", type=Path, default=Path("TRAIN_DATASET/koglab_levi"))
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets/quant_attention"))
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--per-country", type=int, default=60, help="0 means all available test images")
    ap.add_argument("--head-fusion", choices=["mean", "max", "min"], default="mean")
    ap.add_argument("--discard-ratio", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-note", default="sampled_test")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GeoClassifier(num_classes=len(CLASSMATE_CLASSES))
    load_from_h5(model, args.h5)
    model.to(device).eval()
    enable_attention_capture(model)

    samples = list(iter_images(args.data_root, args.per_country, args.seed))
    rows: List[Dict[str, object]] = []
    for path, actual in tqdm(samples, desc=f"attention metrics on {device}"):
        try:
            rows.append(score_image(model, path, actual, args.img_size, device, args.head_fusion, args.discard_ratio))
        except Exception as exc:
            print(f"skip {path}: {exc}")

    if not rows:
        raise SystemExit("No images were processed.")

    csv_path = args.out_dir / "attention_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    df = pd.DataFrame(rows)
    write_summary(df, args.out_dir, args.batch_note)
    print(f"Saved metrics: {csv_path}")
    print(f"Saved summary: {args.out_dir / 'attention_summary.txt'}")


if __name__ == "__main__":
    main()
