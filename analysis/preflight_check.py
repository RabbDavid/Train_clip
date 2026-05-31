"""Preflight checks for the GPU analysis workflow.

This catches the boring failures before a multi-hour remote run: missing model
folders, missing checkpoints, missing dataset, missing Python packages.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import List, Optional


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def exists(path: Path, description: str, required: bool, lines: List[str]) -> bool:
    ok = path.exists()
    status = "OK" if ok else ("MISSING" if required else "optional missing")
    lines.append(f"[{status}] {description}: {path}")
    return ok or not required


def find_latest_run(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / "best.pt").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def count_test_images(data_root: Path) -> tuple[int, int]:
    if not data_root.exists():
        return 0, 0
    if (data_root / "test").exists():
        folders = [p for p in (data_root / "test").iterdir() if p.is_dir()]
    else:
        folders = [p for p in data_root.iterdir() if p.is_dir() and p.name.lower().endswith("_test")]
    count = 0
    for folder in folders:
        count += sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    return len(folders), count


def package_check(name: str, lines: List[str], required: bool = True) -> bool:
    ok = importlib.util.find_spec(name) is not None
    status = "OK" if ok else ("MISSING" if required else "optional missing")
    lines.append(f"[{status}] Python package: {name}")
    return ok or not required


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("TRAIN_DATASET/koglab_levi"))
    ap.add_argument("--dfn2b-model-dir", type=Path, default=Path("MODEL/DFN2B-CLIP-ViT-B-16"))
    ap.add_argument("--streetclip-model-dir", type=Path, default=Path("MODEL/StreetCLIP"))
    ap.add_argument("--dfn2b-run", type=Path, default=None)
    ap.add_argument("--streetclip-run", type=Path, default=None)
    ap.add_argument("--dino-h5", type=Path, default=Path("dino_geo_28_countries_full.weights.h5"))
    ap.add_argument("--out", type=Path, default=Path("preflight_report.txt"))
    args = ap.parse_args()

    lines: List[str] = ["GPU analysis preflight", "=" * 60]
    ok = True

    folders, images = count_test_images(args.data_root)
    ok &= exists(args.data_root, "dataset root", True, lines)
    lines.append(f"[INFO] test folders/images found: {folders} folders, {images} images")
    if images == 0:
        ok = False

    ok &= exists(args.dfn2b_model_dir, "DFN2B model folder", True, lines)
    ok &= exists(args.dfn2b_model_dir / "open_clip_config.json", "DFN2B OpenCLIP config", True, lines)
    ok &= exists(args.dfn2b_model_dir / "open_clip_pytorch_model.bin", "DFN2B OpenCLIP weights", True, lines)

    ok &= exists(args.streetclip_model_dir, "StreetCLIP model folder", True, lines)
    ok &= exists(args.streetclip_model_dir / "config.json", "StreetCLIP config", True, lines)
    ok &= exists(args.streetclip_model_dir / "preprocessor_config.json", "StreetCLIP processor config", True, lines)
    street_weight_ok = (args.streetclip_model_dir / "model.safetensors").exists() or (args.streetclip_model_dir / "pytorch_model.bin").exists()
    lines.append(f"[{'OK' if street_weight_ok else 'MISSING'}] StreetCLIP weights: model.safetensors or pytorch_model.bin")
    ok &= street_weight_ok

    dfn_run = args.dfn2b_run or find_latest_run(Path("runs_clip"))
    street_run = args.streetclip_run or find_latest_run(Path("runs_streetclip"))
    if dfn_run is None:
        lines.append("[MISSING] DFN2B run folder with best.pt under runs_clip/")
        ok = False
    else:
        lines.append(f"[OK] DFN2B run folder: {dfn_run}")
        ok &= exists(dfn_run / "best.pt", "DFN2B best checkpoint", True, lines)
        ok &= exists(dfn_run / "config.json", "DFN2B run config", True, lines)
    if street_run is None:
        lines.append("[MISSING] StreetCLIP run folder with best.pt under runs_streetclip/")
        ok = False
    else:
        lines.append(f"[OK] StreetCLIP run folder: {street_run}")
        ok &= exists(street_run / "best.pt", "StreetCLIP best checkpoint", True, lines)
        ok &= exists(street_run / "config.json", "StreetCLIP run config", True, lines)

    exists(args.dino_h5, "DINO legacy weights", False, lines)

    for pkg in ["torch", "torchvision", "numpy", "pandas", "PIL", "matplotlib", "transformers", "open_clip", "timm", "h5py"]:
        ok &= package_check(pkg, lines)

    package_check("seaborn", lines, required=False)

    if dfn_run and street_run:
        commands = {
            "streetclip_run": str(street_run),
            "dfn2b_run": str(dfn_run),
            "one_command_without_concepts": (
                "python analysis/run_gpu_analysis_workflow.py "
                f"--streetclip-run {street_run} --dfn2b-run {dfn_run} --skip-concepts"
            ),
            "one_command_with_concepts": (
                "python analysis/run_gpu_analysis_workflow.py "
                f"--streetclip-run {street_run} --dfn2b-run {dfn_run} --run-concepts"
            ),
        }
        lines.extend(["", "Suggested commands:", json.dumps(commands, indent=2)])

    lines.extend(["", f"RESULT: {'PASS' if ok else 'FAIL'}"])
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
