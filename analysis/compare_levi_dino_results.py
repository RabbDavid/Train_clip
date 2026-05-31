"""Compare Levi's DINO result CSVs against DFN2B/StreetCLIP outputs.

This consumes already-generated prediction CSVs; it does not need to load model
weights or images.

Example:
    python analysis/compare_levi_dino_results.py ^
      --levi-results-dir "C:/tmp/levi_new_dino/Results" ^
      --streetclip-pred runs_streetclip/20260519-221734/test_predictions_streetclip.csv ^
      --dfn2b-pred runs_clip/20260519-213048/test_predictions_detailed_clip.csv ^
      --out-dir analysis_outputs/levi_dino_comparison
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_predictions(path: Path, model: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    rename = {
        "true_label": "actual_country",
        "pred_label": "predicted_country",
        "prediction": "predicted_country",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    required = {"actual_country", "predicted_country"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    out = df[["actual_country", "predicted_country"]].copy()
    if "file" in df.columns:
        out["file"] = df["file"]
    out["model"] = model
    out["correct"] = out["actual_country"] == out["predicted_country"]
    return out


def per_country_metrics(df: pd.DataFrame) -> pd.DataFrame:
    countries = sorted(set(df["actual_country"]) | set(df["predicted_country"]))
    rows: List[Dict[str, object]] = []
    for country in countries:
        actual = df["actual_country"] == country
        pred = df["predicted_country"] == country
        tp = int((actual & pred).sum())
        fp = int((~actual & pred).sum())
        fn = int((actual & ~pred).sum())
        support = int(actual.sum())
        if support == 0:
            continue
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        rows.append({
            "country": country,
            "correct": tp,
            "total": support,
            "accuracy": recall,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levi-results-dir", type=Path, required=True)
    ap.add_argument("--streetclip-pred", type=Path, default=None)
    ap.add_argument("--dfn2b-pred", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("analysis_outputs/levi_dino_comparison"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames: List[pd.DataFrame] = []
    for idx in range(6):
        path = args.levi_results_dir / f"{idx}_test_predictions_detailed.csv"
        if path.exists():
            frames.append(load_predictions(path, f"Levi-DINO-{idx}"))
    if args.dfn2b_pred and args.dfn2b_pred.exists():
        frames.append(load_predictions(args.dfn2b_pred, "DFN2B-CLIP"))
    if args.streetclip_pred and args.streetclip_pred.exists():
        frames.append(load_predictions(args.streetclip_pred, "StreetCLIP"))
    if not frames:
        raise SystemExit("No prediction CSVs found.")

    all_preds = pd.concat(frames, ignore_index=True)
    overall_rows = []
    country_rows = []
    for model, part in all_preds.groupby("model"):
        correct = int(part["correct"].sum())
        total = int(len(part))
        overall_rows.append({
            "model": model,
            "correct": correct,
            "total": total,
            "accuracy": correct / max(1, total),
        })
        per = per_country_metrics(part)
        per.insert(0, "model", model)
        country_rows.append(per)

    overall = pd.DataFrame(overall_rows).sort_values("accuracy", ascending=False)
    per_country = pd.concat(country_rows, ignore_index=True)
    overall.to_csv(args.out_dir / "overall_model_comparison.csv", index=False)
    per_country.to_csv(args.out_dir / "per_country_model_comparison.csv", index=False)

    plt.figure(figsize=(8, 4.2))
    colors = ["#D55E00" if m == "StreetCLIP" else "#0072B2" if m == "DFN2B-CLIP" else "#777777" for m in overall["model"]]
    plt.bar(overall["model"], overall["accuracy"], color=colors)
    plt.ylabel("test accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.title("Final held-out test accuracy comparison")
    for i, row in enumerate(overall.itertuples(index=False)):
        plt.text(i, row.accuracy + 0.015, f"{row.accuracy:.3f}\n({row.correct}/{row.total})", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out_dir / "overall_model_comparison.png", dpi=180)
    plt.close()

    selected = [m for m in ["StreetCLIP", "DFN2B-CLIP", "Levi-DINO-5"] if m in set(overall["model"])]
    if selected:
        pivot = per_country[per_country["model"].isin(selected)].pivot(index="country", columns="model", values="accuracy")
        if "StreetCLIP" in pivot.columns:
            pivot = pivot.sort_values("StreetCLIP", ascending=False)
        plt.figure(figsize=(12, 4.8))
        x = np.arange(len(pivot))
        width = 0.8 / len(selected)
        palette = {"StreetCLIP": "#D55E00", "DFN2B-CLIP": "#0072B2", "Levi-DINO-5": "#666666"}
        for j, model in enumerate(selected):
            plt.bar(x - 0.4 + width / 2 + j * width, pivot[model], width=width, label=model, color=palette.get(model))
        plt.ylabel("per-country test accuracy")
        plt.ylim(0, 1)
        plt.xticks(x, pivot.index, rotation=45, ha="right")
        plt.title("Per-country accuracy: best models")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_dir / "per_country_best_model_comparison.png", dpi=180)
        plt.close()

    lines = [
        "Levi DINO comparison summary",
        "=" * 60,
        "",
        overall.to_string(index=False, formatters={"accuracy": "{:.4f}".format}),
    ]
    if "StreetCLIP" in set(overall["model"]) and "Levi-DINO-5" in set(overall["model"]):
        acc = dict(zip(overall["model"], overall["accuracy"]))
        lines += [
            "",
            f"StreetCLIP - Levi-DINO-5 accuracy delta: {acc['StreetCLIP'] - acc['Levi-DINO-5']:+.4f}",
        ]
    (args.out_dir / "comparison_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote comparison outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
