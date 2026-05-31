"""Combine attention rollout runs into paper-ready comparison tables/charts."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_metrics(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["model_label"] = label
    df["correct"] = df["correct"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df


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
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def metric_direction(metric: str) -> str:
    if "entropy" in metric:
        return "lower means more concentrated"
    if "mass" in metric or "peak" in metric:
        return "higher means more concentrated"
    return "higher score direction"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--streetclip-metrics", type=Path, required=True)
    ap.add_argument("--dfn2b-metrics", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("analysis_outputs/attention_comparison"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.concat(
        [
            read_metrics(args.streetclip_metrics, "StreetCLIP"),
            read_metrics(args.dfn2b_metrics, "DFN2B-CLIP"),
        ],
        ignore_index=True,
    )

    by_model = df.groupby("model_label").agg(
        n=("correct", "size"),
        accuracy=("correct", "mean"),
        mean_confidence=("confidence", "mean"),
        mean_entropy=("attention_entropy", "mean"),
        mean_top10_mass=("attention_top10pct_mass", "mean"),
        mean_peak_to_mean=("attention_peak_to_mean", "mean"),
    )
    by_model.to_csv(args.out_dir / "attention_summary_by_model.csv")

    by_correct = df.groupby(["model_label", "correct"]).agg(
        n=("correct", "size"),
        mean_confidence=("confidence", "mean"),
        mean_entropy=("attention_entropy", "mean"),
        mean_top10_mass=("attention_top10pct_mass", "mean"),
        mean_peak_to_mean=("attention_peak_to_mean", "mean"),
    )
    by_correct.to_csv(args.out_dir / "attention_summary_by_model_and_correctness.csv")

    by_country = df.groupby(["model_label", "actual_country"]).agg(
        n=("correct", "size"),
        accuracy=("correct", "mean"),
        mean_entropy=("attention_entropy", "mean"),
        mean_top10_mass=("attention_top10pct_mass", "mean"),
    ).reset_index()
    by_country.to_csv(args.out_dir / "attention_summary_by_country.csv", index=False)

    metric_cols = [
        "confidence",
        "attention_entropy",
        "attention_top1pct_mass",
        "attention_top5pct_mass",
        "attention_top10pct_mass",
        "attention_peak_to_mean",
    ]
    auc_rows = []
    for model, part in df.groupby("model_label"):
        y = part["correct"].astype(int).to_numpy()
        for metric in metric_cols:
            if metric not in part.columns:
                continue
            auc = safe_auc(y, part[metric].to_numpy())
            auc_rows.append({
                "model_label": model,
                "metric": metric,
                "auc_for_correct_prediction": "" if auc is None else f"{auc:.4f}",
                "direction_note": metric_direction(metric),
            })
    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(args.out_dir / "attention_metric_auc_for_correctness.csv", index=False)

    corr_rows = []
    for model, part in by_country.groupby("model_label"):
        for metric in ["mean_entropy", "mean_top10_mass"]:
            if len(part) < 3:
                continue
            corr_rows.append({
                "model_label": model,
                "metric": metric,
                "pearson_corr_with_country_accuracy": f"{part['accuracy'].corr(part[metric], method='pearson'):.4f}",
                "spearman_corr_with_country_accuracy": f"{part['accuracy'].corr(part[metric], method='spearman'):.4f}",
                "n_countries": int(part["actual_country"].nunique()),
            })
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(args.out_dir / "country_accuracy_attention_correlation.csv", index=False)

    plt.figure(figsize=(6, 3.5))
    for model, color in [("DFN2B-CLIP", "#0072B2"), ("StreetCLIP", "#D55E00")]:
        part = df[df["model_label"] == model]
        plt.hist(part["attention_entropy"], bins=30, alpha=0.55, label=model, color=color)
    plt.xlabel("normalized attention entropy")
    plt.ylabel("image count")
    plt.title("Attention focus distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "attention_entropy_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 3.5))
    positions = []
    values = []
    labels = []
    colors = []
    for i, model in enumerate(["DFN2B-CLIP", "StreetCLIP"]):
        for j, correct in enumerate([False, True]):
            positions.append(i * 3 + j)
            values.append(df[(df["model_label"] == model) & (df["correct"] == correct)]["attention_entropy"])
            labels.append(f"{model}\n{'correct' if correct else 'wrong'}")
            colors.append("#D7E8F5" if model == "DFN2B-CLIP" else "#F9DCC7")
    boxes = plt.boxplot(values, positions=positions, patch_artist=True, showfliers=False)
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
    plt.xticks(positions, labels)
    plt.ylabel("normalized attention entropy")
    plt.title("Attention entropy: correct vs wrong predictions")
    plt.tight_layout()
    plt.savefig(args.out_dir / "attention_entropy_correct_vs_wrong_models.png", dpi=180)
    plt.close()

    pivot = by_country.pivot(index="actual_country", columns="model_label", values="accuracy").dropna()
    pivot = pivot.sort_values("StreetCLIP", ascending=False)
    plt.figure(figsize=(12, 4.5))
    x = range(len(pivot))
    plt.bar([i - 0.2 for i in x], pivot["DFN2B-CLIP"], width=0.4, label="DFN2B-CLIP", color="#0072B2")
    plt.bar([i + 0.2 for i in x], pivot["StreetCLIP"], width=0.4, label="StreetCLIP", color="#D55E00")
    plt.xticks(list(x), list(pivot.index), rotation=45, ha="right")
    plt.ylabel("sampled test accuracy")
    plt.title("Attention-run sample accuracy by country")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "attention_sample_accuracy_by_country.png", dpi=180)
    plt.close()

    for metric, filename, xlabel in [
        ("mean_entropy", "country_accuracy_vs_attention_entropy.png", "mean attention entropy"),
        ("mean_top10_mass", "country_accuracy_vs_top10_mass.png", "mean top-10% attention mass"),
    ]:
        plt.figure(figsize=(6.4, 4.2))
        for model, color in [("DFN2B-CLIP", "#0072B2"), ("StreetCLIP", "#D55E00")]:
            part = by_country[by_country["model_label"] == model]
            plt.scatter(part[metric], part["accuracy"], s=46, alpha=0.75, label=model, color=color)
        plt.xlabel(xlabel)
        plt.ylabel("sampled country accuracy")
        plt.title("Country accuracy vs attention focus")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_dir / filename, dpi=180)
        plt.close()

    lines = [
        "Attention comparison summary",
        "=" * 60,
        "",
        "By model:",
        by_model.to_string(),
        "",
        "By model and correctness:",
        by_correct.to_string(),
        "",
        "Metric AUC for correct prediction:",
        auc_df.to_string(index=False),
        "",
        "Country-level attention/performance correlation:",
        corr_df.to_string(index=False),
        "",
        "Interpretation note:",
        "Lower entropy means the rollout is more spatially concentrated. AUC values near 0.5 mean little association with correctness.",
        "These are focus/performance associations, not proof of causal importance.",
    ]
    (args.out_dir / "attention_comparison_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote comparison outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
