"""One-command GPU workflow for post-training analysis.

This wrapper exists because the GPU machine may be operated remotely from a
phone. It runs the individual analysis scripts in the right order and stops on
the first failure.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def latest_run(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / "best.pt").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run(cmd: List[str]) -> None:
    print("\n" + "=" * 80)
    print("RUN:", " ".join(str(x) for x in cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)


def path_arg(path: Path) -> str:
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("TRAIN_DATASET/koglab_levi"))
    ap.add_argument("--dfn2b-model-dir", type=Path, default=Path("MODEL/DFN2B-CLIP-ViT-B-16"))
    ap.add_argument("--streetclip-model-dir", type=Path, default=Path("MODEL/StreetCLIP"))
    ap.add_argument("--dfn2b-run", type=Path, default=None)
    ap.add_argument("--streetclip-run", type=Path, default=None)
    ap.add_argument("--analysis-root", type=Path, default=Path("analysis_outputs"))
    ap.add_argument("--max-samples-per-country", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--save-panels", type=int, default=80)
    ap.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--heatmap-norm", choices=["mass", "percentile", "minmax"], default="mass",
                    help="panel color scaling; mass is comparable, minmax is local contrast only")
    ap.add_argument("--vmax-multiplier", type=float, default=4.0)
    ap.add_argument("--heatmap-alpha", type=float, default=0.38)
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--run-concepts", action="store_true", help="also run CLIPSeg concept attribution")
    ap.add_argument("--skip-concepts", action="store_true")
    ap.add_argument("--concept-max-samples", type=int, default=500)
    ap.add_argument("--run-dino", action="store_true", help="run legacy DINO analysis if dino h5 is present")
    ap.add_argument("--dino-h5", type=Path, default=Path("dino_geo_28_countries_full.weights.h5"))
    ap.add_argument("--skip-package", action="store_true")
    ap.add_argument("--include-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    dfn_run = args.dfn2b_run or latest_run(Path("runs_clip"))
    street_run = args.streetclip_run or latest_run(Path("runs_streetclip"))
    if dfn_run is None:
        raise SystemExit("No DFN2B run folder found. Pass --dfn2b-run.")
    if street_run is None:
        raise SystemExit("No StreetCLIP run folder found. Pass --streetclip-run.")

    py = sys.executable
    run([
        py,
        "analysis/preflight_check.py",
        "--data-root",
        path_arg(args.data_root),
        "--dfn2b-model-dir",
        path_arg(args.dfn2b_model_dir),
        "--streetclip-model-dir",
        path_arg(args.streetclip_model_dir),
        "--dfn2b-run",
        path_arg(dfn_run),
        "--streetclip-run",
        path_arg(street_run),
        "--dino-h5",
        path_arg(args.dino_h5),
    ])

    street_attention = args.analysis_root / "streetclip_attention"
    dfn_attention = args.analysis_root / "dfn2b_attention"
    comparison = args.analysis_root / "attention_comparison"

    run([
        py,
        "analysis/attention_rollout_clip.py",
        "--model",
        "streetclip",
        "--data-root",
        path_arg(args.data_root),
        "--model-dir",
        path_arg(args.streetclip_model_dir),
        "--checkpoint",
        path_arg(street_run / "best.pt"),
        "--out-dir",
        path_arg(street_attention),
        "--max-samples-per-country",
        str(args.max_samples_per_country),
        "--batch-size",
        str(args.batch_size),
        "--save-panels",
        str(args.save_panels),
        "--amp-dtype",
        args.amp_dtype,
        "--heatmap-norm",
        args.heatmap_norm,
        "--vmax-multiplier",
        str(args.vmax_multiplier),
        "--heatmap-alpha",
        str(args.heatmap_alpha),
        "--cmap",
        args.cmap,
    ])

    run([
        py,
        "analysis/attention_rollout_clip.py",
        "--model",
        "dfn2b",
        "--data-root",
        path_arg(args.data_root),
        "--model-dir",
        path_arg(args.dfn2b_model_dir),
        "--checkpoint",
        path_arg(dfn_run / "best.pt"),
        "--out-dir",
        path_arg(dfn_attention),
        "--max-samples-per-country",
        str(args.max_samples_per_country),
        "--batch-size",
        str(args.batch_size),
        "--save-panels",
        str(args.save_panels),
        "--amp-dtype",
        args.amp_dtype,
        "--heatmap-norm",
        args.heatmap_norm,
        "--vmax-multiplier",
        str(args.vmax_multiplier),
        "--heatmap-alpha",
        str(args.heatmap_alpha),
        "--cmap",
        args.cmap,
    ])

    run([
        py,
        "analysis/summarize_attention_runs.py",
        "--streetclip-metrics",
        path_arg(street_attention / "attention_metrics.csv"),
        "--dfn2b-metrics",
        path_arg(dfn_attention / "attention_metrics.csv"),
        "--out-dir",
        path_arg(comparison),
    ])

    if args.run_dino:
        if args.dino_h5.exists():
            run([
                py,
                "legacy_dino/attention_quantitative_eval.py",
                "--h5",
                path_arg(args.dino_h5),
                "--data-root",
                path_arg(args.data_root),
                "--out-dir",
                path_arg(args.analysis_root / "dino_attention"),
                "--per-country",
                str(args.max_samples_per_country),
                "--batch-note",
                f"sampled_{args.max_samples_per_country}_per_country",
            ])
            run([
                py,
                "legacy_dino/run_viz.py",
                "--h5",
                path_arg(args.dino_h5),
                "--data-root",
                path_arg(args.data_root),
                "--out-dir",
                path_arg(args.analysis_root / "dino_examples"),
                "--per-country",
                "2",
                "--misclassified",
                "1",
                "--heatmap-norm",
                "mass",
            ])
        else:
            print(f"Skipping DINO: {args.dino_h5} not found")

    if args.run_concepts and not args.skip_concepts:
        run([
            py,
            "analysis/object_concept_attention.py",
            "--attention-metrics",
            path_arg(street_attention / "attention_metrics.csv"),
            "--out-dir",
            path_arg(args.analysis_root / "streetclip_concepts"),
            "--no-local-files-only",
            "--max-samples",
            str(args.concept_max_samples),
        ])
        run([
            py,
            "analysis/object_concept_attention.py",
            "--attention-metrics",
            path_arg(dfn_attention / "attention_metrics.csv"),
            "--out-dir",
            path_arg(args.analysis_root / "dfn2b_concepts"),
            "--no-local-files-only",
            "--max-samples",
            str(args.concept_max_samples),
        ])

    if not args.skip_package:
        package_cmd = [
            py,
            "analysis/package_final_outputs.py",
            "--streetclip-run",
            path_arg(street_run),
            "--dfn2b-run",
            path_arg(dfn_run),
            "--analysis-root",
            path_arg(args.analysis_root),
            "--out-dir",
            "FINAL_RESULTS",
            "--zip",
        ]
        package_cmd.append("--include-checkpoints" if args.include_checkpoints else "--no-include-checkpoints")
        run(package_cmd)

    print("\nDONE. Send back FINAL_RESULTS_CLIP_COUNTRY.zip and the text summaries in analysis_outputs/.")


if __name__ == "__main__":
    main()
