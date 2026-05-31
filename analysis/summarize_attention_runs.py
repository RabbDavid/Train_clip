"""Combine attention rollout runs into paper-ready comparison tables/charts."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_metrics(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["model_label"] = label
    df["correct"] = df["correct"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df


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
        "Interpretation note:",
        "Lower entropy means the rollout is more spatially concentrated. This is a focus metric, not a proof of causal importance.",
    ]
    (args.out_dir / "attention_comparison_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote comparison outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
