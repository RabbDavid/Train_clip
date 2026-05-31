"""Collect training/evaluation/analysis outputs into one final folder.

The script copies small report artifacts and optionally checkpoints. It never
copies MODEL/ or TRAIN_DATASET/.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Optional


def copy_if_exists(src: Path, dst: Path) -> Optional[Path]:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def copy_many(run_dir: Path, out_dir: Path, files: Iterable[str]) -> None:
    for name in files:
        copy_if_exists(run_dir / name, out_dir / name)


def run_text(command: list[str], cwd: Path) -> str:
    try:
        completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
        return completed.stdout.strip() + ("\n" + completed.stderr.strip() if completed.stderr.strip() else "")
    except Exception as exc:  # pragma: no cover - best effort environment capture
        return f"FAILED: {' '.join(command)}\n{exc}"


def zip_dir(folder: Path, out_zip: Path) -> None:
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in folder.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(folder.parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--streetclip-run", type=Path, required=True)
    ap.add_argument("--dfn2b-run", type=Path, required=True)
    ap.add_argument("--analysis-root", type=Path, default=Path("analysis_outputs"))
    ap.add_argument("--out-dir", type=Path, default=Path("FINAL_RESULTS"))
    ap.add_argument("--include-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--zip", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    street_out = args.out_dir / "streetclip"
    dfn_out = args.out_dir / "dfn2b_clip"

    street_files = [
        "config.json",
        "epoch_metrics.csv",
        "test_report_streetclip.txt",
        "test_predictions_streetclip.csv",
        "confusion_matrix_streetclip.csv",
    ]
    dfn_files = [
        "config.json",
        "epoch_metrics.csv",
        "test_report_clip.txt",
        "test_predictions_detailed_clip.csv",
        "confusion_matrix_clip.csv",
        "wiseft_metrics.csv",
    ]
    if args.include_checkpoints:
        street_files.append("best.pt")
        dfn_files.append("best.pt")

    copy_many(args.streetclip_run, street_out, street_files)
    copy_many(args.dfn2b_run, dfn_out, dfn_files)

    docs_out = args.out_dir / "documentation"
    for src in [
        Path("README.md"),
        Path("BENDE_GPU_RUN_REQUEST.md"),
        Path("FINAL_PDF_PLAN.md"),
        Path("FOLDER_STRUCTURE.txt"),
        Path("RUN_GPU_ANALYSIS.md"),
        Path("requirements.txt"),
        Path("analysis/README.md"),
        Path("analysis/INTERPRETABILITY_NOTES.md"),
    ]:
        copy_if_exists(src, docs_out / src.as_posix().replace("/", "__"))

    if args.analysis_root.exists():
        analysis_out = args.out_dir / "analysis_outputs"
        if analysis_out.exists():
            shutil.rmtree(analysis_out)
        ignore = shutil.ignore_patterns("*.npy")
        shutil.copytree(args.analysis_root, analysis_out, ignore=ignore)

    copy_if_exists(Path("preflight_report.txt"), args.out_dir / "preflight_report.txt")

    env_text = [
        "Python",
        sys.version,
        "",
        "pip freeze",
        run_text([sys.executable, "-m", "pip", "freeze"], Path.cwd()),
        "",
        "nvidia-smi",
        run_text(["nvidia-smi"], Path.cwd()),
    ]
    (args.out_dir / "environment.txt").write_text("\n".join(env_text) + "\n", encoding="utf-8")

    repo_text = [
        "git rev-parse HEAD",
        run_text(["git", "rev-parse", "HEAD"], Path.cwd()),
        "",
        "git status --short",
        run_text(["git", "status", "--short"], Path.cwd()),
        "",
        "git log --oneline -5",
        run_text(["git", "log", "--oneline", "-5"], Path.cwd()),
    ]
    (args.out_dir / "repo_info.txt").write_text("\n".join(repo_text) + "\n", encoding="utf-8")

    if args.zip:
        zip_path = args.out_dir.parent / "FINAL_RESULTS_CLIP_COUNTRY.zip"
        zip_dir(args.out_dir, zip_path)
        print(f"Wrote {zip_path}")
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
