"""Generate attention-visualization figures for the paper.

Loads the classmate's `.h5`, runs N images per country through the model, and
saves figures to `figures/`. By default samples a few correctly-classified and
a few misclassified examples per country — both make for good paper figures.

Usage examples
--------------
# Quick smoke test on a single image:
python run_viz.py --image TRAIN_DATASET/koglab_levi/japan_test/<some_file>.jpg

# Full per-country sweep (default: 3 correct + 2 wrong per country):
python run_viz.py --data-root TRAIN_DATASET/koglab_levi --per-country 3 --misclassified 2

# Higher-resolution attention (slower but crisper for paper figures):
python run_viz.py --img-size 518 --per-country 2
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image

from attention_viz import enable_attention_capture, make_figure, per_head_grid_image
from load_levi_dino_h5 import CLASSMATE_CLASSES, build_model_from_h5


COUNTRY_ALIASES = {"columbia": "colombia"}


def find_test_folders(data_root: Path) -> List[Tuple[str, Path]]:
    """Return list of (country, folder) for each `<country>_test` directory."""
    out = []
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name.lower()
        if name.endswith("_test"):
            country = name[:-len("_test")]
            country = COUNTRY_ALIASES.get(country, country)
            if country in CLASSMATE_CLASSES:
                out.append((country, child))
    return out


def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path, default=Path("dino_geo_28_countries_full.weights.h5"))
    ap.add_argument("--data-root", type=Path, default=Path("TRAIN_DATASET/koglab_levi"))
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    ap.add_argument("--image", type=Path, default=None, help="single image (overrides --data-root)")
    ap.add_argument("--per-country", type=int, default=3, help="N correctly-classified images per country")
    ap.add_argument("--misclassified", type=int, default=2, help="N misclassified images per country")
    ap.add_argument("--img-size", type=int, default=224, help="multiple of 14 (224, 336, 518)")
    ap.add_argument("--head-fusion", choices=["mean", "max", "min"], default="mean")
    ap.add_argument("--discard-ratio", type=float, default=0.0)
    ap.add_argument("--heatmap-norm", choices=["mass", "minmax", "percentile"], default="mass",
                    help="mass gives comparable colors; minmax is local contrast only")
    ap.add_argument("--cmap", default="viridis", help="matplotlib colormap for overlays")
    ap.add_argument("--heatmap-alpha", type=float, default=0.38)
    ap.add_argument("--vmax-multiplier", type=float, default=4.0,
                    help="for mass norm: color max is this times uniform patch mass")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-images-scanned-per-country", type=int, default=120,
                    help="cap how many images to score before picking samples (keeps runtime reasonable)")
    args = ap.parse_args()

    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.h5}")
    model = build_model_from_h5(args.h5, num_classes=len(CLASSMATE_CLASSES))
    model.eval()
    enable_attention_capture(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}.  img_size={args.img_size}.")

    # --- single-image mode ---
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        t0 = time.time()
        res = make_figure(model, img, CLASSMATE_CLASSES, img_size=args.img_size,
                           device=device, head_fusion=args.head_fusion, discard_ratio=args.discard_ratio,
                           heatmap_alpha=args.heatmap_alpha, cmap_name=args.cmap,
                           heatmap_norm=args.heatmap_norm, vmax_multiplier=args.vmax_multiplier)
        dt = time.time() - t0
        out_path = args.out_dir / f"{args.image.stem}__rollout.jpg"
        res.triptych.save(out_path, quality=90)
        head_path = args.out_dir / f"{args.image.stem}__per_head_last.jpg"
        per_head_grid_image(
            res.per_head_last_layer,
            img.resize((args.img_size, args.img_size)),
            alpha=args.heatmap_alpha,
            cmap_name=args.cmap,
            norm_mode=args.heatmap_norm,
            vmax_multiplier=args.vmax_multiplier,
        ).save(head_path, quality=90)
        print(f"  pred: {res.pred_label} ({res.confidence*100:.1f}%)  in {dt*1000:.0f}ms")
        print(f"  saved: {out_path}\n  saved: {head_path}")
        return

    # --- batch mode: per-country sampling ---
    folders = find_test_folders(args.data_root)
    if not folders:
        raise SystemExit(f"No <country>_test folders found under {args.data_root}")
    print(f"Found {len(folders)} country test folders.")

    summary_rows = []
    for country, folder in folders:
        imgs = list_images(folder)
        random.shuffle(imgs)
        imgs = imgs[: args.max_images_scanned_per_country]
        if not imgs:
            continue

        correct: List[Tuple[Path, "make_figure.return_type"]] = []
        wrong: List[Tuple[Path, "make_figure.return_type"]] = []

        for p in imgs:
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                print(f"  skip {p.name}: {e}")
                continue
            res = make_figure(model, img, CLASSMATE_CLASSES, img_size=args.img_size,
                               device=device, head_fusion=args.head_fusion, discard_ratio=args.discard_ratio,
                               heatmap_alpha=args.heatmap_alpha, cmap_name=args.cmap,
                               heatmap_norm=args.heatmap_norm, vmax_multiplier=args.vmax_multiplier)
            (correct if res.pred_label == country else wrong).append((p, res))
            if len(correct) >= args.per_country and len(wrong) >= args.misclassified:
                break

        country_dir = args.out_dir / country
        country_dir.mkdir(parents=True, exist_ok=True)

        for tag, picks in (("correct", correct[: args.per_country]), ("wrong", wrong[: args.misclassified])):
            for p, res in picks:
                base = f"{tag}__{p.stem}__pred-{res.pred_label}__{int(res.confidence*100)}"
                res.triptych.save(country_dir / f"{base}__rollout.jpg", quality=90)
                per_head = per_head_grid_image(
                    res.per_head_last_layer,
                    Image.open(p).convert("RGB").resize((args.img_size, args.img_size)),
                    alpha=args.heatmap_alpha,
                    cmap_name=args.cmap,
                    norm_mode=args.heatmap_norm,
                    vmax_multiplier=args.vmax_multiplier,
                )
                per_head.save(country_dir / f"{base}__per_head.jpg", quality=88)
                summary_rows.append({
                    "country": country,
                    "file": p.name,
                    "tag": tag,
                    "predicted": res.pred_label,
                    "confidence": f"{res.confidence:.4f}",
                })
        print(f"  {country:14s}  saved correct={len(correct[:args.per_country])}  wrong={len(wrong[:args.misclassified])}")

    with open(args.out_dir / "viz_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["country", "file", "tag", "predicted", "confidence"], delimiter=";")
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nWrote {len(summary_rows)} figures.\nIndex: {args.out_dir / 'viz_index.csv'}")


if __name__ == "__main__":
    main()
