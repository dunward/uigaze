"""Result visualization for UIGaze experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def plot_model_comparison(
    results_csv: str | Path,
    output_path: str | Path = "results/model_comparison.png",
    metrics: list[str] | None = None,
):
    """Bar chart comparing models across metrics."""
    df = pd.read_csv(results_csv)

    if metrics is None:
        metrics = [c for c in ["CC", "SIM", "KL", "NSS", "AUC"] if c in df.columns]

    models = df["model"].unique()
    summary = df.groupby("model")[metrics].mean()

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        values = summary[metric].values
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        bars = ax.bar(range(len(models)), values, color=colors)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel("Score")

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle("VLM Saliency Prediction — Model Comparison", fontsize=14)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved model comparison to {output_path}")


def plot_category_heatmap(
    results_csv: str | Path,
    metric: str = "CC",
    output_path: str | Path = "results/category_heatmap.png",
):
    """Heatmap of metric scores by model and category."""
    df = pd.read_csv(results_csv)
    pivot = df.pivot_table(values=metric, index="model", columns="category", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9)

    plt.colorbar(im, ax=ax, label=metric)
    ax.set_title(f"{metric} by Model and UI Category", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved category heatmap to {output_path}")


def plot_saliency_comparison(
    image_path: str | Path,
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    output_path: str | Path,
    model_name: str = "VLM",
):
    """Side-by-side comparison: original image, predicted saliency, ground truth."""
    img = Image.open(image_path).convert("RGB")
    img_arr = np.array(img)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_arr)
    axes[0].set_title("Original UI", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(img_arr)
    axes[1].imshow(pred_map, cmap="jet", alpha=0.5)
    axes[1].set_title(f"Predicted ({model_name})", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(img_arr)
    axes[2].imshow(gt_map, cmap="jet", alpha=0.5)
    axes[2].set_title("Ground Truth (UEyes)", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
