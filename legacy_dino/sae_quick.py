"""Quick sparse autoencoder experiment on Levi's frozen DINOv2 model.

The script trains an SAE on final-layer patch-token embeddings and saves:
  - sae_outputs/sae_quick.pt
  - sae_outputs/loss.csv
  - sae_outputs/top_features.csv
  - sae_outputs/feature_XXX.jpg contact sheets with top activating patches
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_levi_dino_h5 import CLASSMATE_CLASSES, build_model_from_h5


MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


@dataclass
class TokenRecord:
    image: Path
    country: str
    patch_idx: int
    row: int
    col: int


class SAE(nn.Module):
    def __init__(self, d_in: int = 384, d_hidden: int = 1536):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_hidden)
        self.decoder = nn.Linear(d_hidden, d_in, bias=False)
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
        nn.init.constant_(self.encoder.bias, -1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = F.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z


def country_from_folder(folder: Path) -> str:
    name = folder.name.lower()
    if name == "columbia_test":
        return "colombia"
    return name[:-5] if name.endswith("_test") else name


def list_images_for_split(data_root: Path, split: str, max_per_country: int, seed: int) -> List[Tuple[Path, str]]:
    rng = random.Random(seed)
    out: List[Tuple[Path, str]] = []
    if split == "train" and (data_root / "train").exists():
        folders = [p for p in (data_root / "train").iterdir() if p.is_dir()]
    elif split == "test" and (data_root / "test").exists():
        folders = [p for p in (data_root / "test").iterdir() if p.is_dir()]
    elif split == "all":
        folders = [p for p in data_root.iterdir() if p.is_dir()]
    elif split == "train":
        folders = [p for p in data_root.iterdir() if p.is_dir() and not p.name.lower().endswith("_test")]
    else:
        folders = [p for p in data_root.iterdir() if p.is_dir() and p.name.lower().endswith("_test")]

    for folder in sorted(folders):
        country = country_from_folder(folder)
        if country not in CLASSMATE_CLASSES:
            continue
        images = [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        rng.shuffle(images)
        out.extend((p, country) for p in images[:max_per_country])
    rng.shuffle(out)
    return out


def preprocess(path: Path, img_size: int, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def final_patch_tokens(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    feats = model.backbone.forward_features(x)
    if isinstance(feats, dict):
        if "x_norm_patchtokens" in feats:
            return feats["x_norm_patchtokens"]
        if "x" in feats:
            return feats["x"][:, 1:]
    return feats[:, 1:] if feats.ndim == 3 else feats.unsqueeze(1)


def collect_tokens(
    model: nn.Module,
    images: Sequence[Tuple[Path, str]],
    img_size: int,
    patches_per_image: int,
    device: torch.device,
    seed: int,
) -> Tuple[torch.Tensor, List[TokenRecord]]:
    rng = random.Random(seed)
    tokens: List[torch.Tensor] = []
    records: List[TokenRecord] = []
    grid = img_size // 14
    for i, (path, country) in enumerate(images, 1):
        x = preprocess(path, img_size, device)
        patch_tokens = final_patch_tokens(model, x)[0].detach().cpu()
        picks = rng.sample(range(patch_tokens.shape[0]), k=min(patches_per_image, patch_tokens.shape[0]))
        tokens.append(patch_tokens[picks])
        for pidx in picks:
            records.append(TokenRecord(path, country, pidx, pidx // grid, pidx % grid))
        if i % 100 == 0:
            print(f"collected {i}/{len(images)} images -> {len(records)} tokens")
    return torch.cat(tokens, dim=0), records


def make_contact_sheet(rows: Sequence[Tuple[float, TokenRecord]], out_path: Path, img_size: int) -> None:
    thumb_w, thumb_h = 180, 120
    cols = 5
    cell_h = thumb_h + 26
    canvas = Image.new("RGB", (cols * thumb_w, ((len(rows) + cols - 1) // cols) * cell_h), (24, 24, 24))
    patch = img_size // 14
    sx, sy = 600 / img_size, 400 / img_size
    for i, (score, rec) in enumerate(rows):
        img = Image.open(rec.image).convert("RGB").resize((600, 400), Image.BILINEAR)
        draw = ImageDraw.Draw(img)
        x0 = int(rec.col * patch * sx)
        y0 = int(rec.row * patch * sy)
        x1 = int((rec.col + 1) * patch * sx)
        y1 = int((rec.row + 1) * patch * sy)
        draw.rectangle([x0, y0, x1, y1], outline=(255, 230, 0), width=5)
        img.thumbnail((thumb_w, thumb_h))
        x = (i % cols) * thumb_w
        y = (i // cols) * cell_h
        canvas.paste(img, (x, y))
        ImageDraw.Draw(canvas).text((x + 4, y + thumb_h + 4), f"{rec.country} {score:.2f}", fill=(255, 255, 255))
    canvas.save(out_path, quality=92)


def summarize_sae_features(z: torch.Tensor, records: Sequence[TokenRecord], top_k: int) -> List[Dict[str, object]]:
    """Summarize SAE features so contact sheets are backed by numbers."""
    out: List[Dict[str, object]] = []
    n_tokens = z.shape[0]
    k = min(top_k, n_tokens)
    for feat in range(z.shape[1]):
        values = z[:, feat]
        active = values > 0
        active_count = int(active.sum().item())
        top_scores, top_idxs = torch.topk(values, k=k)
        positive_pairs = [(float(score), int(idx)) for score, idx in zip(top_scores.tolist(), top_idxs.tolist()) if score > 0]
        countries = [records[idx].country for _, idx in positive_pairs]
        counts = Counter(countries)
        top_country, top_country_count = counts.most_common(1)[0] if counts else ("", 0)
        rows = [records[idx].row for _, idx in positive_pairs]
        cols = [records[idx].col for _, idx in positive_pairs]
        out.append({
            "feature": feat,
            "max_activation": f"{float(values.max().item()):.6f}",
            "mean_activation": f"{float(values.mean().item()):.6f}",
            "active_rate": f"{active_count / max(1, n_tokens):.6f}",
            "active_count": active_count,
            "top_k_positive": len(positive_pairs),
            "top_country": top_country,
            "top_country_fraction": f"{top_country_count / max(1, len(positive_pairs)):.6f}",
            "top_countries": "|".join(f"{country}:{count}" for country, count in counts.most_common(5)),
            "top_patch_row_mean": "" if not rows else f"{float(np.mean(rows)):.3f}",
            "top_patch_col_mean": "" if not cols else f"{float(np.mean(cols)):.3f}",
            "top_score_mean": "" if not positive_pairs else f"{float(np.mean([score for score, _ in positive_pairs])):.6f}",
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path, default=Path("dino_geo_28_countries_full.weights.h5"))
    ap.add_argument("--data-root", type=Path, default=Path("TRAIN_DATASET/koglab_levi"))
    ap.add_argument("--out-dir", type=Path, default=Path("sae_outputs"))
    ap.add_argument("--split", choices=["train", "test", "all"], default="train",
                    help="fit SAE on train activations by default; use test/all only for exploratory analysis")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--max-per-country", type=int, default=60)
    ap.add_argument("--patches-per-image", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=1536)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l1", type=float, default=5e-2)
    ap.add_argument("--top-k-features", type=int, default=12)
    ap.add_argument("--top-k-patches", type=int, default=10)
    ap.add_argument("--feature-summary-top-k", type=int, default=50,
                    help="top patches used for feature purity/position summaries")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model_from_h5(args.h5, num_classes=len(CLASSMATE_CLASSES))
    model.to(device).eval()

    split_used = args.split
    images = list_images_for_split(args.data_root, split_used, args.max_per_country, args.seed)
    if not images and args.split == "train":
        print("No train folders found; falling back to test folders for exploratory SAE fitting.")
        split_used = "test"
        images = list_images_for_split(args.data_root, split_used, args.max_per_country, args.seed)
    print(f"using {len(images)} {split_used} images on {device}")
    if not images:
        raise SystemExit(f"No images found for split={args.split} under {args.data_root}")
    tokens, records = collect_tokens(model, images, args.img_size, args.patches_per_image, device, args.seed)
    mean = tokens.mean(dim=0, keepdim=True)
    std = tokens.std(dim=0, keepdim=True).clamp_min(1e-5)
    tokens_n = (tokens - mean) / std

    sae = SAE(d_in=tokens_n.shape[1], d_hidden=args.hidden).to(device)
    loader = DataLoader(TensorDataset(tokens_n), batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(sae.parameters(), lr=args.lr)

    with open(args.out_dir / "loss.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "mse", "l1", "l0"], delimiter=";")
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            totals = {"loss": 0.0, "mse": 0.0, "l1": 0.0, "l0": 0.0, "n": 0}
            for (batch,) in loader:
                batch = batch.to(device)
                x_hat, z = sae(batch)
                mse = F.mse_loss(x_hat, batch)
                l1 = z.mean()
                loss = mse + args.l1 * l1
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    sae.decoder.weight.data = F.normalize(sae.decoder.weight.data, dim=0)
                totals["loss"] += loss.item() * batch.size(0)
                totals["mse"] += mse.item() * batch.size(0)
                totals["l1"] += l1.item() * batch.size(0)
                totals["l0"] += (z > 0).float().sum(dim=1).mean().item() * batch.size(0)
                totals["n"] += batch.size(0)
            row = {k: totals[k] / totals["n"] for k in ["loss", "mse", "l1", "l0"]}
            writer.writerow({"epoch": epoch, **{k: f"{v:.6f}" for k, v in row.items()}})
            print(f"epoch {epoch:02d}: loss={row['loss']:.4f} mse={row['mse']:.4f} l0={row['l0']:.1f}")

    with torch.no_grad():
        _, z = sae(tokens_n.to(device))
        z = z.cpu()

    feature_summary = summarize_sae_features(z, records, args.feature_summary_top_k)
    with open(args.out_dir / "feature_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(feature_summary[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(feature_summary)

    feature_strength = z.max(dim=0).values
    feature_ids = torch.topk(feature_strength, k=min(args.top_k_features, z.shape[1])).indices.tolist()

    with open(args.out_dir / "top_features.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "rank", "score", "image", "country", "patch_row", "patch_col"], delimiter=";")
        writer.writeheader()
        for feat in feature_ids:
            scores, idxs = torch.topk(z[:, feat], k=min(args.top_k_patches, z.shape[0]))
            rows = []
            for rank, (score, idx) in enumerate(zip(scores.tolist(), idxs.tolist()), 1):
                rec = records[idx]
                writer.writerow({
                    "feature": feat, "rank": rank, "score": f"{score:.6f}", "image": str(rec.image),
                    "country": rec.country, "patch_row": rec.row, "patch_col": rec.col,
                })
                rows.append((score, rec))
            make_contact_sheet(rows, args.out_dir / f"feature_{feat:04d}.jpg", args.img_size)

    torch.save({
        "model": sae.state_dict(),
        "mean": mean,
        "std": std,
        "args": vars(args),
        "split_used": split_used,
        "feature_ids": feature_ids,
    }, args.out_dir / "sae_quick.pt")
    notes = [
        "SAE quick experiment",
        "=" * 60,
        f"split used: {split_used}",
        f"images: {len(images)}",
        f"tokens: {len(records)}",
        f"input dimension: {tokens_n.shape[1]}",
        f"hidden SAE features: {args.hidden}",
        f"feature summary top-k patches: {args.feature_summary_top_k}",
        "",
        "Interpretation:",
        "The SAE is trained on DINO patch-token activations. The yellow boxes in",
        "feature contact sheets are the image patches whose internal activations",
        "most strongly activate a sparse feature. They are feature exemplars, not",
        "direct object labels and not causal explanations by themselves.",
        "",
        "Use feature_summary.csv to check whether a feature is sparse, whether its",
        "top patches concentrate in one country, and whether it has a positional",
        "bias. Contact sheets should be interpreted together with these numbers.",
    ]
    (args.out_dir / "SAE_INTERPRETATION_NOTE.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"saved SAE outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
