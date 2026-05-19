"""Fine-tune an OpenCLIP model for Street View country classification.

This is intentionally CLIP-native: classes are represented as text prompt
prototypes, images are classified by image-text similarity, and fine-tuning
updates the visual tower while keeping the language tower fixed by default.

Recommended first 5090 run:
    python train_clip_country.py ^
      --data-root TRAIN_DATASET/koglab_levi ^
      --model hf-hub:apple/DFN2B-CLIP-ViT-B-16 ^
      --epochs 12 --batch-size 192 --grad-accum-steps 1 ^
      --unfreeze-visual-layers -1 --lr-visual 1e-5 --lr-logit-scale 5e-5 ^
      --amp-dtype bf16

Use --zero-shot-only for a baseline without training.
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

try:
    import open_clip
except ImportError as exc:  # pragma: no cover - gives a useful runtime error
    open_clip = None
    OPENCLIP_IMPORT_ERROR = exc
else:
    OPENCLIP_IMPORT_ERROR = None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
COUNTRY_ALIASES = {"columbia": "colombia"}
DISPLAY_NAMES = {"usa": "the United States", "colombia": "Colombia"}

PROMPT_TEMPLATES = [
    "a Google Street View image from {country}",
    "a street view photo in {country}",
    "a road scene from {country}",
    "a geolocation game image from {country}",
    "an outdoor street scene in {country}",
    "a roadside landscape in {country}",
]


@dataclass
class RunConfig:
    data_root: str
    model: str
    pretrained: Optional[str]
    tokenizer: Optional[str]
    classes: List[str]
    prompt_templates: List[str]
    img_count_train: int
    img_count_val: int
    img_count_test: int
    epochs: int
    batch_size: int
    grad_accum_steps: int
    lr_visual: float
    lr_logit_scale: float
    weight_decay: float
    warmup_epochs: float
    layer_decay: float
    label_smoothing: float
    unfreeze_visual_layers: int
    amp_dtype: str
    compile: bool
    seed: int


def country_name(name: str) -> str:
    name = name.lower()
    if name.endswith("_test"):
        name = name[:-len("_test")]
    return COUNTRY_ALIASES.get(name, name)


def display_country(country: str) -> str:
    return DISPLAY_NAMES.get(country, country.replace("_", " ").title())


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
        if not child.is_dir():
            continue
        cname = country_name(child.name)
        if child.name.lower() == "test":
            continue
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

    if requested:
        classes = requested
    else:
        classes = sorted({label for _, label in train_samples} or {label for _, label in test_samples})

    train_labels = {label for _, label in train_samples}
    test_labels = {label for _, label in test_samples}
    missing_train = sorted(test_labels - train_labels)
    if missing_train:
        print(f"WARNING: {len(missing_train)} test classes have no train images and will be ignored: {missing_train}")
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
    for label, items in by_label.items():
        rng.shuffle(items)
        out.extend(items[:max_per_class])
    rng.shuffle(out)
    return out


def stratified_split(
    samples: List[Tuple[Path, str]],
    classes: Sequence[str],
    val_fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
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


class CountryDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[Path, str]], classes: Sequence[str], transform):
        self.samples = list(samples)
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.class_to_idx[label], str(path)


def parse_amp_dtype(name: str):
    if name == "fp32":
        return None
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unknown amp dtype: {name}")


def create_model_and_transforms(args):
    if args.model.startswith("hf-hub:") and args.pretrained in {None, "", "none"}:
        return open_clip.create_model_and_transforms(args.model)
    return open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)


def get_tokenizer(args):
    for name in [args.tokenizer, args.model, "ViT-B-16"]:
        if not name:
            continue
        try:
            return open_clip.get_tokenizer(name)
        except Exception:
            continue
    raise RuntimeError("Could not create an OpenCLIP tokenizer.")


def set_trainable(model: nn.Module, unfreeze_visual_layers: int, train_logit_scale: bool) -> None:
    for p in model.parameters():
        p.requires_grad = False

    visual = getattr(model, "visual", None)
    if visual is None:
        raise RuntimeError("OpenCLIP model has no visual tower.")

    if unfreeze_visual_layers < 0:
        for p in visual.parameters():
            p.requires_grad = True
    else:
        # ViT visual towers expose transformer.resblocks. Keep this explicit so
        # accidental architecture changes fail visibly rather than silently.
        resblocks = getattr(getattr(visual, "transformer", None), "resblocks", None)
        if resblocks is None:
            for p in visual.parameters():
                p.requires_grad = True
        else:
            if unfreeze_visual_layers > 0:
                for block in list(resblocks)[-unfreeze_visual_layers:]:
                    for p in block.parameters():
                        p.requires_grad = True
            for attr in ("ln_post", "proj"):
                obj = getattr(visual, attr, None)
                if isinstance(obj, nn.Module):
                    for p in obj.parameters():
                        p.requires_grad = True
                elif isinstance(obj, torch.Tensor) or isinstance(obj, nn.Parameter):
                    obj.requires_grad_(True)

    for name in ("logit_scale", "logit_bias"):
        p = getattr(model, name, None)
        if isinstance(p, nn.Parameter):
            p.requires_grad = train_logit_scale


def visual_depth(model: nn.Module) -> int:
    resblocks = getattr(getattr(getattr(model, "visual", None), "transformer", None), "resblocks", None)
    return len(resblocks) if resblocks is not None else 0


def layer_id_for_param(name: str, depth: int) -> int:
    if not name.startswith("visual."):
        return depth + 1
    marker = "visual.transformer.resblocks."
    if marker in name:
        rest = name.split(marker, 1)[1]
        try:
            return int(rest.split(".", 1)[0]) + 1
        except ValueError:
            return depth
    if any(k in name for k in ("ln_post", "proj")):
        return depth + 1
    return 0


def make_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    no_decay_keys = ("bias", "ln_", "norm", "bn", "positional_embedding", "class_embedding", "logit_scale", "logit_bias")
    depth = visual_depth(model)
    max_layer = depth + 1
    groups: Dict[Tuple[float, float], Dict[str, object]] = {}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_logit = name in {"logit_scale", "logit_bias"}
        base_lr = args.lr_logit_scale if is_logit else args.lr_visual
        lid = layer_id_for_param(name, depth)
        lr = base_lr * (args.layer_decay ** max(0, max_layer - lid)) if depth and not is_logit else base_lr
        wd = 0.0 if p.ndim <= 1 or any(k in name for k in no_decay_keys) else args.weight_decay
        key = (lr, wd)
        groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": wd, "names": []})
        groups[key]["params"].append(p)
        groups[key]["names"].append(name)

    if not groups:
        raise RuntimeError("No trainable parameters. Check --unfreeze-visual-layers.")

    use_fused = args.fused_adamw and torch.cuda.is_available()
    return torch.optim.AdamW(
        [{"params": g["params"], "lr": g["lr"], "weight_decay": g["weight_decay"]} for g in groups.values()],
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        fused=use_fused,
    )


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
def text_prototypes(
    model: nn.Module,
    tokenizer,
    classes: Sequence[str],
    templates: Sequence[str],
    device: torch.device,
    amp_dtype,
) -> torch.Tensor:
    model.eval()
    feats = []
    context_length = getattr(model, "context_length", 77)
    for cls in classes:
        prompts = [tpl.format(country=display_country(cls)) for tpl in templates]
        tokens = tokenizer(prompts, context_length=context_length).to(device)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            text = model.encode_text(tokens)
        text = F.normalize(text.float(), dim=-1)
        text = F.normalize(text.mean(dim=0), dim=0)
        feats.append(text)
    return torch.stack(feats, dim=0).to(device)


def clip_logits(model: nn.Module, images: torch.Tensor, class_text: torch.Tensor) -> torch.Tensor:
    image_features = model.encode_image(images)
    image_features = F.normalize(image_features.float(), dim=-1)
    logit_scale = model.logit_scale.exp() if hasattr(model, "logit_scale") else 100.0
    logits = logit_scale * image_features @ class_text.float().T
    logit_bias = getattr(model, "logit_bias", None)
    if logit_bias is not None:
        logits = logits + logit_bias
    return logits


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_text: torch.Tensor,
    device: torch.device,
    amp_dtype,
    desc: str,
) -> Tuple[float, float, List[Dict[str, object]], np.ndarray, np.ndarray]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    rows: List[Dict[str, object]] = []
    all_labels, all_preds = [], []
    for images, labels, paths in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = clip_logits(model, images, class_text)
            loss = F.cross_entropy(logits, labels)
        probs = logits.softmax(dim=-1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)
        labs = labels.detach().cpu().numpy()
        loss_sum += float(loss.item()) * labels.numel()
        correct += int((preds == labs).sum())
        total += labels.numel()
        all_labels.extend(labs.tolist())
        all_preds.extend(preds.tolist())
        for path, lab, pred, prob in zip(paths, labs.tolist(), preds.tolist(), probs):
            rows.append({
                "file": path,
                "actual_idx": int(lab),
                "predicted_idx": int(pred),
                "confidence": float(prob[pred]),
            })
    return correct / max(1, total), loss_sum / max(1, total), rows, np.array(all_labels), np.array(all_preds)


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


def save_predictions(rows: List[Dict[str, object]], probs_path: Path, classes: Sequence[str], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    # rows do not carry all probabilities to keep eval memory small; this CSV is
    # therefore a compact detailed prediction file.
    with open(probs_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "actual_country", "predicted_country", "confidence"],
            delimiter=";",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "file": row["file"],
                "actual_country": classes[int(row["actual_idx"])],
                "predicted_country": classes[int(row["predicted_idx"])],
                "confidence": f"{float(row['confidence']):.6f}",
            })


def load_state_dict_into(model: nn.Module, state: Dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({k: v.to(device) if torch.is_tensor(v) else v for k, v in state.items()})


def interpolate_state(
    initial: Dict[str, torch.Tensor],
    finetuned: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, ft_value in finetuned.items():
        init_value = initial.get(key)
        if (
            torch.is_tensor(ft_value)
            and torch.is_tensor(init_value)
            and ft_value.shape == init_value.shape
            and ft_value.is_floating_point()
        ):
            out[key] = (1.0 - alpha) * init_value + alpha * ft_value.cpu()
        else:
            out[key] = ft_value.cpu() if torch.is_tensor(ft_value) else ft_value
    return out


def parse_alpha_list(value: str) -> List[float]:
    if not value.strip():
        return []
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("TRAIN_DATASET/koglab_levi"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs_clip"))
    ap.add_argument("--model", default="hf-hub:apple/DFN2B-CLIP-ViT-B-16")
    ap.add_argument("--pretrained", default=None, help="OpenCLIP pretrained tag; leave empty for hf-hub models")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--classes", default=None, help="optional comma-separated class list")
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--max-train-per-class", type=int, default=0, help="debug cap; 0 keeps all")
    ap.add_argument("--max-test-per-class", type=int, default=0, help="debug cap; 0 keeps all")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=192)
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--lr-visual", type=float, default=1e-5)
    ap.add_argument("--lr-logit-scale", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--layer-decay", type=float, default=0.75)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--unfreeze-visual-layers", type=int, default=-1,
                    help="-1 = full visual tower; N = last N ViT blocks")
    ap.add_argument("--train-logit-scale", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--proj-l2", type=float, default=1e-4,
                    help="L2 regularization toward the original visual projection matrix")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--fused-adamw", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.98)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--wiseft-alphas", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--zero-shot-only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if open_clip is None:
        raise SystemExit(
            "Missing dependency: open_clip_torch. Install with:\n"
            "  python -m pip install open_clip_torch\n"
            "or run:\n"
            "  python -m pip install -r requirements.txt"
        ) from OPENCLIP_IMPORT_ERROR

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = parse_amp_dtype(args.amp_dtype)
    run_dir = args.out_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Loading OpenCLIP model: {args.model}")
    model, preprocess_train, preprocess_val = create_model_and_transforms(args)
    tokenizer = get_tokenizer(args)
    model.to(device=device, memory_format=torch.channels_last)

    train_samples, val_samples, test_samples, classes = infer_samples(args.data_root, args.classes)
    train_samples = limit_per_class(train_samples, args.max_train_per_class, args.seed)
    val_samples = limit_per_class(val_samples, args.max_test_per_class, args.seed)
    test_samples = limit_per_class(test_samples, args.max_test_per_class, args.seed)
    if not val_samples and train_samples:
        train_idx, val_idx = stratified_split(train_samples, classes, args.val_fraction, args.seed)
        train_ds = Subset(CountryDataset(train_samples, classes, preprocess_train), train_idx)
        val_ds = Subset(CountryDataset(train_samples, classes, preprocess_val), val_idx)
    else:
        train_ds = CountryDataset(train_samples, classes, preprocess_train)
        val_ds = CountryDataset(val_samples, classes, preprocess_val)
    test_ds = CountryDataset(test_samples, classes, preprocess_val) if test_samples else None

    loader_common = dict(num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_common) if len(train_ds) else None
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_common) if len(val_ds) else None
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_common) if test_ds is not None and len(test_ds) else None

    print(f"Classes ({len(classes)}): {classes}")
    print(f"Images: train={len(train_ds)} val={len(val_ds)} test={len(test_ds) if test_ds else 0}")
    print(f"Train class counts: {dict(Counter(label for _, label in train_samples))}")

    templates = PROMPT_TEMPLATES
    class_text = text_prototypes(model, tokenizer, classes, templates, device, amp_dtype)

    cfg = RunConfig(
        data_root=str(args.data_root),
        model=args.model,
        pretrained=args.pretrained,
        tokenizer=args.tokenizer,
        classes=classes,
        prompt_templates=templates,
        img_count_train=len(train_ds),
        img_count_val=len(val_ds),
        img_count_test=len(test_ds) if test_ds else 0,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr_visual=args.lr_visual,
        lr_logit_scale=args.lr_logit_scale,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        layer_decay=args.layer_decay,
        label_smoothing=args.label_smoothing,
        unfreeze_visual_layers=args.unfreeze_visual_layers,
        amp_dtype=args.amp_dtype,
        compile=args.compile,
        seed=args.seed,
    )
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if val_loader is not None:
        zs_acc, zs_loss, _, _, _ = evaluate(model, val_loader, class_text, device, amp_dtype, "zero-shot val")
        print(f"Zero-shot val: acc={zs_acc:.4f} loss={zs_loss:.4f}")
    if args.zero_shot_only:
        if test_loader is not None:
            acc, loss, rows, y_true, y_pred = evaluate(model, test_loader, class_text, device, amp_dtype, "zero-shot test")
            print(f"Zero-shot test: acc={acc:.4f} loss={loss:.4f}")
            save_predictions(rows, run_dir / "test_predictions_zero_shot.csv", classes, y_true, y_pred)
            (run_dir / "test_report_zero_shot.txt").write_text(per_class_report(y_true, y_pred, classes), encoding="utf-8")
        return

    if train_loader is None or val_loader is None:
        raise SystemExit("Training needs train and validation samples.")

    set_trainable(model, args.unfreeze_visual_layers, args.train_logit_scale)
    if args.compile:
        print("NOTE: --compile is currently not applied because OpenCLIP uses custom encode_image/encode_text methods here.")
    optimizer = make_optimizer(model, args)
    scaler = GradScaler(device.type, enabled=(amp_dtype == torch.float16))

    proj_initial = None
    visual = getattr(model, "visual", None)
    if visual is not None and hasattr(visual, "proj") and isinstance(visual.proj, torch.Tensor):
        proj_initial = visual.proj.detach().float().clone()

    total_steps = math.ceil(len(train_loader) / args.grad_accum_steps) * args.epochs
    warmup_steps = int(math.ceil(len(train_loader) / args.grad_accum_steps) * args.warmup_epochs)
    best_val = -1.0
    best_path = run_dir / "best.pt"
    global_step = 0

    with open(run_dir / "epoch_metrics.csv", "w", newline="", encoding="utf-8") as f_epoch:
        epoch_writer = csv.DictWriter(
            f_epoch,
            fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "lr"],
            delimiter=";",
        )
        epoch_writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss, total_n = 0.0, 0
            pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
            for step, (images, labels, _) in enumerate(pbar, 1):
                apply_lr(optimizer, global_step, total_steps, warmup_steps, args.min_lr_ratio)
                images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
                labels = labels.to(device, non_blocking=True)

                with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    logits = clip_logits(model, images, class_text)
                    loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)
                    if proj_initial is not None and args.proj_l2 > 0:
                        loss = loss + args.proj_l2 * F.mse_loss(visual.proj.float(), proj_initial)
                    loss_for_backward = loss / args.grad_accum_steps

                if scaler.is_enabled():
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

                total_loss += float(loss.item()) * labels.numel()
                total_n += labels.numel()

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
                    if hasattr(model, "logit_scale"):
                        model.logit_scale.data.clamp_(0, math.log(100.0))
                    global_step += 1

                lr_now = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(loss=f"{total_loss / max(1, total_n):.4f}", lr=f"{lr_now:.2e}")

            class_text = text_prototypes(model, tokenizer, classes, templates, device, amp_dtype)
            val_acc, val_loss, _, _, _ = evaluate(model, val_loader, class_text, device, amp_dtype, "val")
            train_loss = total_loss / max(1, total_n)
            lr_now = optimizer.param_groups[0]["lr"]
            epoch_writer.writerow({
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_acc": f"{val_acc:.6f}",
                "lr": f"{lr_now:.8g}",
            })
            f_epoch.flush()
            print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            state = {
                "model": model.state_dict(),
                "classes": classes,
                "templates": templates,
                "config": asdict(cfg),
                "epoch": epoch,
                "val_acc": val_acc,
            }
            torch.save(state, run_dir / "last.pt")
            if val_acc > best_val:
                best_val = val_acc
                torch.save(state, best_path)
                print(f"  saved new best: {best_path} ({best_val:.4f})")

    ckpt = torch.load(best_path, map_location="cpu")
    load_state_dict_into(model, ckpt["model"], device)
    class_text = text_prototypes(model, tokenizer, classes, templates, device, amp_dtype)
    if test_loader is not None:
        test_acc, test_loss, rows, y_true, y_pred = evaluate(model, test_loader, class_text, device, amp_dtype, "test")
        print(f"Best checkpoint test: acc={test_acc:.4f} loss={test_loss:.4f}")
        save_predictions(rows, run_dir / "test_predictions_detailed_clip.csv", classes, y_true, y_pred)
        (run_dir / "test_report_clip.txt").write_text(per_class_report(y_true, y_pred, classes), encoding="utf-8")

    alphas = parse_alpha_list(args.wiseft_alphas)
    if alphas and val_loader is not None:
        best_state = {k: v.detach().cpu().clone() for k, v in ckpt["model"].items()}
        wise_rows = []
        best_alpha, best_alpha_val = None, -1.0
        for alpha in alphas:
            mixed = interpolate_state(initial_state, best_state, alpha)
            load_state_dict_into(model, mixed, device)
            class_text = text_prototypes(model, tokenizer, classes, templates, device, amp_dtype)
            val_acc, val_loss, _, _, _ = evaluate(model, val_loader, class_text, device, amp_dtype, f"wise-ft val a={alpha:g}")
            row = {"alpha": alpha, "val_acc": val_acc, "val_loss": val_loss}
            if test_loader is not None:
                test_acc, test_loss, _, _, _ = evaluate(model, test_loader, class_text, device, amp_dtype, f"wise-ft test a={alpha:g}")
                row.update({"test_acc": test_acc, "test_loss": test_loss})
            wise_rows.append(row)
            if val_acc > best_alpha_val:
                best_alpha_val = val_acc
                best_alpha = alpha
                torch.save({"model": mixed, "classes": classes, "templates": templates, "alpha": alpha, "config": asdict(cfg)}, run_dir / "wiseft_best.pt")

        with open(run_dir / "wiseft_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for row in wise_rows for k in row}), delimiter=";")
            writer.writeheader()
            writer.writerows(wise_rows)
        print(f"Best WiSE-FT alpha by val: {best_alpha} val_acc={best_alpha_val:.4f}")


if __name__ == "__main__":
    main()
