"""Train a supervised country classifier on top of a local StreetCLIP model.

StreetCLIP is a Hugging Face Transformers CLIP checkpoint, not an OpenCLIP
local-dir checkpoint. This script therefore uses transformers and expects the
model to be downloaded manually before training.

Expected local structure:
    MODEL/StreetCLIP/
      config.json
      preprocessor_config.json
      model.safetensors or pytorch_model.bin

Recommended run:
    python Code/train_streetclip_country.py ^
      --data-root TRAIN_DATASET/koglab_levi ^
      --model-dir MODEL/StreetCLIP ^
      --epochs 8 --batch-size 64 --grad-accum-steps 2 --unfreeze-vision-layers 4 ^
      --lr-head 1e-3 --lr-vision 1e-5 --amp-dtype bf16
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

try:
    from transformers import CLIPImageProcessor, CLIPModel
except ImportError as exc:  # pragma: no cover
    CLIPImageProcessor = None
    CLIPModel = None
    TRANSFORMERS_IMPORT_ERROR = exc
else:
    TRANSFORMERS_IMPORT_ERROR = None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
COUNTRY_ALIASES = {"columbia": "colombia"}


@dataclass
class RunConfig:
    data_root: str
    model_dir: str
    classes: List[str]
    img_count_train: int
    img_count_val: int
    img_count_test: int
    epochs: int
    batch_size: int
    grad_accum_steps: int
    num_workers: int
    prefetch_factor: int
    lr_head: float
    lr_vision: float
    weight_decay: float
    warmup_epochs: float
    unfreeze_vision_layers: int
    amp_dtype: str
    attn_implementation: str
    compile: bool
    seed: int


def country_name(name: str) -> str:
    name = name.lower()
    if name.endswith("_test"):
        name = name[:-len("_test")]
    return COUNTRY_ALIASES.get(name, name)


def image_files(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def scan_imagefolder(root: Path, classes: Optional[Sequence[str]] = None) -> List[Tuple[Path, str]]:
    wanted = set(classes) if classes else None
    samples: List[Tuple[Path, str]] = []
    if not root.exists():
        return samples
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        cname = country_name(child.name)
        if wanted is not None and cname not in wanted:
            continue
        samples.extend((p, cname) for p in image_files(child))
    return samples


def scan_classmate_layout(root: Path, classes: Optional[Sequence[str]] = None) -> Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]]]:
    wanted = set(classes) if classes else None
    train_samples: List[Tuple[Path, str]] = []
    test_samples: List[Tuple[Path, str]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.lower() == "test":
            continue
        cname = country_name(child.name)
        if wanted is not None and cname not in wanted:
            continue
        samples = [(p, cname) for p in image_files(child)]
        if child.name.lower().endswith("_test"):
            test_samples.extend(samples)
        else:
            train_samples.extend(samples)
    return train_samples, test_samples


def infer_samples(data_root: Path, classes_arg: Optional[str]) -> Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]], List[Tuple[Path, str]], List[str]]:
    requested = [c.strip().lower() for c in classes_arg.split(",")] if classes_arg else None
    if (data_root / "train").exists():
        train_samples = scan_imagefolder(data_root / "train", requested)
        val_samples = scan_imagefolder(data_root / "val", requested)
        test_samples = scan_imagefolder(data_root / "test", requested)
    else:
        train_samples, test_samples = scan_classmate_layout(data_root, requested)
        val_samples = []

    classes = requested or sorted({label for _, label in train_samples} or {label for _, label in test_samples})
    train_labels = {label for _, label in train_samples}
    test_labels = {label for _, label in test_samples}
    missing_train = sorted(test_labels - train_labels)
    if missing_train:
        print(f"WARNING: ignoring test classes with no train images: {missing_train}")
        classes = [c for c in classes if c in train_labels]
        test_samples = [(p, y) for p, y in test_samples if y in train_labels]

    if not classes:
        raise SystemExit(f"No classes found under {data_root}")
    return train_samples, val_samples, test_samples, classes


def limit_per_class(samples: List[Tuple[Path, str]], max_per_class: int, seed: int) -> List[Tuple[Path, str]]:
    if max_per_class <= 0:
        return samples
    rng = random.Random(seed)
    by_label: Dict[str, List[Tuple[Path, str]]] = defaultdict(list)
    for sample in samples:
        by_label[sample[1]].append(sample)
    out: List[Tuple[Path, str]] = []
    for items in by_label.values():
        rng.shuffle(items)
        out.extend(items[:max_per_class])
    rng.shuffle(out)
    return out


def stratified_split(samples: List[Tuple[Path, str]], classes: Sequence[str], val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    by_label: Dict[str, List[int]] = defaultdict(list)
    for idx, (_, label) in enumerate(samples):
        by_label[label].append(idx)
    train_idx, val_idx = [], []
    for label in classes:
        idxs = by_label.get(label, [])
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_fraction))) if len(idxs) > 1 else 0
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


class StreetViewDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[Path, str]], classes: Sequence[str], processor: CLIPImageProcessor):
        self.samples = list(samples)
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return pixel_values, self.class_to_idx[label], str(path)


class StreetCLIPClassifier(nn.Module):
    def __init__(self, clip: CLIPModel, num_classes: int, dropout: float):
        super().__init__()
        self.clip = clip
        dim = int(clip.config.projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.clip.get_image_features(pixel_values=pixel_values)
        feats = F.normalize(feats.float(), dim=-1)
        return self.head(self.dropout(feats))


def set_trainable(model: StreetCLIPClassifier, unfreeze_layers: int) -> None:
    for p in model.clip.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    vision = model.clip.vision_model
    if unfreeze_layers < 0:
        for p in vision.parameters():
            p.requires_grad = True
        for p in model.clip.visual_projection.parameters():
            p.requires_grad = True
        return

    if unfreeze_layers > 0:
        layers = list(vision.encoder.layers)
        for layer in layers[-unfreeze_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        for module in [vision.post_layernorm, model.clip.visual_projection]:
            for p in module.parameters():
                p.requires_grad = True


def make_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    no_decay = ("bias", "LayerNorm", "layer_norm", "position_embedding", "class_embedding")
    groups = [
        {"params": [], "lr": args.lr_head, "weight_decay": args.weight_decay},
        {"params": [], "lr": args.lr_head, "weight_decay": 0.0},
        {"params": [], "lr": args.lr_vision, "weight_decay": args.weight_decay},
        {"params": [], "lr": args.lr_vision, "weight_decay": 0.0},
    ]
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        clean_name = name.removeprefix("_orig_mod.")
        is_head = clean_name.startswith("head.")
        is_nodecay = p.ndim <= 1 or any(k in clean_name for k in no_decay)
        idx = (0 if is_head else 2) + (1 if is_nodecay else 0)
        groups[idx]["params"].append(p)
    groups = [g for g in groups if g["params"]]
    return torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-6, fused=args.fused_adamw and torch.cuda.is_available())


def dataloader_kwargs(num_workers: int, prefetch_factor: int) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def parse_amp_dtype(name: str):
    if name == "fp32":
        return None
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(name)


def cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def apply_lr(optimizer: torch.optim.Optimizer, step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> None:
    for group in optimizer.param_groups:
        group.setdefault("base_lr", group["lr"])
        group["lr"] = cosine_lr(step, total_steps, warmup_steps, group["base_lr"], min_lr_ratio)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp_dtype, classes: Sequence[str], desc: str):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    rows = []
    y_true, y_pred = [], []
    for pixel_values, labels, paths in tqdm(loader, desc=desc, leave=False):
        pixel_values = pixel_values.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(pixel_values)
            loss = F.cross_entropy(logits, labels)
        probs_tensor = logits.softmax(dim=-1).detach().cpu()
        topk_count = min(5, probs_tensor.shape[1])
        topk_confidences, topk_indices = probs_tensor.topk(topk_count, dim=1)
        probs = probs_tensor.numpy()
        preds = probs.argmax(axis=1)
        labs = labels.detach().cpu().numpy()
        loss_sum += float(loss.item()) * labels.numel()
        correct += int((preds == labs).sum())
        total += labels.numel()
        y_true.extend(labs.tolist())
        y_pred.extend(preds.tolist())
        for path, lab, pred, prob, top_idx, top_conf in zip(
            paths,
            labs.tolist(),
            preds.tolist(),
            probs,
            topk_indices.tolist(),
            topk_confidences.tolist(),
        ):
            rows.append({
                "file": path,
                "actual_country": classes[int(lab)],
                "predicted_country": classes[int(pred)],
                "confidence": float(prob[pred]),
                "top5_countries": "|".join(classes[int(i)] for i in top_idx),
                "top5_confidences": "|".join(f"{float(v):.6f}" for v in top_conf),
            })
    return correct / max(1, total), loss_sum / max(1, total), rows, np.array(y_true), np.array(y_pred)


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[str]) -> str:
    lines = [f"{'class':18s} precision recall f1 support"]
    correct_total = int((y_true == y_pred).sum())
    for idx, cls in enumerate(classes):
        tp = int(((y_true == idx) & (y_pred == idx)).sum())
        fp = int(((y_true != idx) & (y_pred == idx)).sum())
        fn = int(((y_true == idx) & (y_pred != idx)).sum())
        support = int((y_true == idx).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        lines.append(f"{cls:18s} {precision:8.3f} {recall:6.3f} {f1:6.3f} {support:7d}")
    lines.append("")
    lines.append(f"accuracy {correct_total / max(1, len(y_true)):.4f} ({correct_total}/{len(y_true)})")
    return "\n".join(lines) + "\n"


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[str], out_path: Path) -> None:
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for actual, predicted in zip(y_true.tolist(), y_pred.tolist()):
        matrix[int(actual), int(predicted)] += 1

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["actual_country", *classes])
        for idx, cls in enumerate(classes):
            writer.writerow([cls, *matrix[idx].tolist()])


def save_predictions(rows: List[Dict[str, object]], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "actual_country",
                "predicted_country",
                "confidence",
                "top5_countries",
                "top5_confidences",
            ],
            delimiter=";",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "file": row["file"],
                "actual_country": row["actual_country"],
                "predicted_country": row["predicted_country"],
                "confidence": f"{float(row['confidence']):.6f}",
                "top5_countries": row.get("top5_countries", ""),
                "top5_confidences": row.get("top5_confidences", ""),
            })


def unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def load_clip_model(model_dir: Path, attn_implementation: str):
    kwargs: Dict[str, object] = {"local_files_only": True}
    if attn_implementation != "auto":
        kwargs["attn_implementation"] = attn_implementation
    try:
        return CLIPModel.from_pretrained(model_dir, **kwargs)
    except (TypeError, ValueError) as exc:
        if attn_implementation == "sdpa":
            print(f"WARNING: SDPA attention request was not accepted by this Transformers build ({exc}). Falling back to model default.")
            return CLIPModel.from_pretrained(model_dir, local_files_only=True)
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("TRAIN_DATASET/koglab_levi"))
    ap.add_argument("--model-dir", type=Path, default=Path("MODEL/StreetCLIP"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs_streetclip"))
    ap.add_argument("--classes", default=None)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--max-train-per-class", type=int, default=0)
    ap.add_argument("--max-test-per-class", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--grad-accum-steps", type=int, default=2,
                    help="effective batch = batch-size * grad-accum-steps")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-vision", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--unfreeze-vision-layers", type=int, default=4, help="-1 = full vision tower; N = last N layers")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--attn-implementation", choices=["auto", "eager", "sdpa", "flash_attention_2"], default="sdpa",
                    help="Transformers attention backend. sdpa is the safe fast default; flash_attention_2 requires a working flash-attn install.")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--fused-adamw", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if CLIPModel is None or CLIPImageProcessor is None:
        raise SystemExit("Missing dependency: transformers. Run: pip install -r requirements.txt") from TRANSFORMERS_IMPORT_ERROR
    if not args.model_dir.is_dir():
        raise SystemExit(f"StreetCLIP model folder missing: {args.model_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = parse_amp_dtype(args.amp_dtype)
    run_dir = args.out_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_samples, val_samples, test_samples, classes = infer_samples(args.data_root, args.classes)
    train_samples = limit_per_class(train_samples, args.max_train_per_class, args.seed)
    val_samples = limit_per_class(val_samples, args.max_test_per_class, args.seed)
    test_samples = limit_per_class(test_samples, args.max_test_per_class, args.seed)

    print(f"Loading local StreetCLIP: {args.model_dir}")
    print(f"Attention backend request: {args.attn_implementation}")
    clip = load_clip_model(args.model_dir, args.attn_implementation)
    processor = CLIPImageProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = StreetCLIPClassifier(clip, num_classes=len(classes), dropout=args.dropout)
    set_trainable(model, args.unfreeze_vision_layers)
    model.to(device=device, memory_format=torch.channels_last)
    if args.compile:
        model = torch.compile(model)

    if not val_samples:
        train_idx, val_idx = stratified_split(train_samples, classes, args.val_fraction, args.seed)
        train_ds = Subset(StreetViewDataset(train_samples, classes, processor), train_idx)
        val_ds = Subset(StreetViewDataset(train_samples, classes, processor), val_idx)
    else:
        train_ds = StreetViewDataset(train_samples, classes, processor)
        val_ds = StreetViewDataset(val_samples, classes, processor)
    test_ds = StreetViewDataset(test_samples, classes, processor) if test_samples else None

    loader_common = dataloader_kwargs(args.num_workers, args.prefetch_factor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_common)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_common)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_common) if test_ds is not None else None

    cfg = RunConfig(
        data_root=str(args.data_root),
        model_dir=str(args.model_dir),
        classes=classes,
        img_count_train=len(train_ds),
        img_count_val=len(val_ds),
        img_count_test=len(test_ds) if test_ds else 0,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        lr_head=args.lr_head,
        lr_vision=args.lr_vision,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        unfreeze_vision_layers=args.unfreeze_vision_layers,
        amp_dtype=args.amp_dtype,
        attn_implementation=args.attn_implementation,
        compile=args.compile,
        seed=args.seed,
    )
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    print(f"Device: {device}")
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Images: train={len(train_ds)} val={len(val_ds)} test={len(test_ds) if test_ds else 0}")
    print(f"Raw train class counts: {dict(Counter(label for _, label in train_samples))}")

    optimizer = make_optimizer(model, args)
    scaler = GradScaler(device.type, enabled=(amp_dtype == torch.float16))
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = updates_per_epoch * args.epochs
    warmup_steps = int(updates_per_epoch * args.warmup_epochs)
    best_val_acc = -1.0
    best_path = run_dir / "best.pt"
    global_step = 0

    with open(run_dir / "epoch_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "lr_head", "lr_vision"], delimiter=";")
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss, total_n = 0.0, 0
            pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
            for step, (pixel_values, labels, _) in enumerate(pbar, 1):
                apply_lr(optimizer, global_step, total_steps, warmup_steps, args.min_lr_ratio)
                pixel_values = pixel_values.to(device, non_blocking=True, memory_format=torch.channels_last)
                labels = labels.to(device, non_blocking=True)
                with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    logits = model(pixel_values)
                    loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)
                    loss_for_backward = loss / args.grad_accum_steps
                if scaler.is_enabled():
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

                if step % args.grad_accum_steps == 0 or step == len(train_loader):
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), args.grad_clip)
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                total_loss += float(loss.item()) * labels.numel()
                total_n += labels.numel()
                pbar.set_postfix(loss=f"{total_loss / max(1, total_n):.4f}")

            val_acc, val_loss, _, _, _ = evaluate(model, val_loader, device, amp_dtype, classes, "val")
            train_loss = total_loss / max(1, total_n)
            lr_head = max(g["lr"] for g in optimizer.param_groups)
            lr_vision = min(g["lr"] for g in optimizer.param_groups)
            writer.writerow({
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_acc": f"{val_acc:.6f}",
                "lr_head": f"{lr_head:.8g}",
                "lr_vision": f"{lr_vision:.8g}",
            })
            f.flush()
            print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            state = {"model": unwrap_model(model).state_dict(), "classes": classes, "config": asdict(cfg), "epoch": epoch, "val_acc": val_acc}
            torch.save(state, run_dir / "last.pt")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(state, best_path)
                print(f"  saved new best: {best_path} ({best_val_acc:.4f})")

    ckpt = torch.load(best_path, map_location=device)
    unwrap_model(model).load_state_dict(ckpt["model"])
    if test_loader is not None:
        test_acc, test_loss, rows, y_true, y_pred = evaluate(model, test_loader, device, amp_dtype, classes, "test")
        print(f"Best checkpoint test: acc={test_acc:.4f} loss={test_loss:.4f}")
        save_predictions(rows, run_dir / "test_predictions_streetclip.csv")
        (run_dir / "test_report_streetclip.txt").write_text(per_class_report(y_true, y_pred, classes), encoding="utf-8")
        save_confusion_matrix(y_true, y_pred, classes, run_dir / "confusion_matrix_streetclip.csv")


if __name__ == "__main__":
    main()
