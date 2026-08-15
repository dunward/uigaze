"""Generate camera-ready figures for the CHI paper (PDF, for Overleaf upload)."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.vlm.openrouter import GazePoint
from src.saliency.generator import generate_saliency_map
from src.analysis.visualizer import plot_saliency_comparison
from src.data_loader import load_dataset

ROOT = Path(__file__).parent.parent.parent
RESULTS = ROOT / "results" / "full"
OUT = ROOT / "results" / "paper_figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, GRAY_LINE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
# Leader, the model the coordinate-scale artifact had hidden, and the genuine laggard.
HIGHLIGHT = {"gpt-5.4": BLUE, "gemini-3.1-flash-lite": ORANGE, "ui-tars-1.5": AQUA}
LABELS = {"gpt-5.4": "GPT-5.4", "gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite",
          "ui-tars-1.5": "UI-TARS 1.5"}
DURS = ["1s", "3s", "7s"]

# Axis tick labels, kept identical across figures so the same model reads the
# same way everywhere.
SHORT = {
    "gpt-5.4": "GPT-5.4", "gpt-5.4-mini": "GPT-5.4-mini",
    "gemini-3.1-pro": "Gemini 3.1 Pro", "gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite",
    "claude-opus-4.6": "Claude Opus 4.6", "claude-sonnet-4.6": "Claude Sonnet 4.6",
    "qwen-3.5-plus": "Qwen 3.5 Plus", "qwen-3.5-flash": "Qwen 3.5 Flash",
    "ui-tars-1.5": "UI-TARS 1.5",
}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5,
    "axes.edgecolor": GRAY_LINE, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def style_ax(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def load_series(metric: str) -> pd.DataFrame:
    frames = []
    for d in DURS:
        df = pd.read_csv(RESULTS / d / "summary_by_model.csv")
        df["duration"] = d
        frames.append(df)
    return pd.concat(frames).pivot(index="model", columns="duration",
                                   values=f"{metric}_mean_mean")


def fig_duration():
    cc = load_series("CC")
    ccp = load_series("CC_partial")
    prior = [pd.read_csv(RESULTS / "baselines" / d / "summary.csv")
             .set_index("baseline").loc["mean_gt_cat", "CC_mean"] for d in DURS]
    ioc = pd.read_csv(RESULTS / "human_ceiling" / "ioc_summary.csv").set_index("duration")
    ceiling = [ioc.loc[d, "CC_mean"] for d in DURS]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    x = np.arange(3)

    def stagger(entries, min_gap):
        """entries: list of (y, label, color); returns label y-positions without overlap."""
        order = sorted(range(len(entries)), key=lambda i: entries[i][0])
        ys = [entries[i][0] for i in order]
        for j in range(1, len(ys)):
            if ys[j] - ys[j - 1] < min_gap:
                ys[j] = ys[j - 1] + min_gap
        out = [0.0] * len(entries)
        for pos, i in enumerate(order):
            out[i] = ys[pos]
        return out

    for ax, data, title in ((axes[0], cc, "Raw agreement (CC)"),
                            (axes[1], ccp, "Beyond-prior agreement (partial CC)")):
        style_ax(ax)
        ends = []
        for model in data.index:
            vals = data.loc[model, DURS].values
            if model in HIGHLIGHT:
                ax.plot(x, vals, color=HIGHLIGHT[model], linewidth=1.8, marker="o",
                        markersize=3.5, zorder=3)
                ends.append((vals[-1], LABELS[model], HIGHLIGHT[model]))
            else:
                ax.plot(x, vals, color=GRAY_LINE, linewidth=1.0, zorder=1)
        if ax is axes[0]:
            ends.append((prior[-1], "Dataset prior (LOO)", INK))
        gap = 0.045 if ax is axes[0] else 0.026
        for (y, label, color), ly in zip(ends, stagger(ends, gap)):
            ax.annotate(label, (x[-1] + 0.08, ly), color=color, fontsize=7.5, va="center")
        ax.set_xticks(x, DURS)
        ax.set_xlim(-0.15, 3.6)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Viewing duration", color=MUTED)

    axes[0].plot(x, ceiling, color=INK, linewidth=1.4, linestyle=":", zorder=2)
    axes[0].annotate("Human split-half IOC", (1.0, ceiling[1] + 0.03), color=INK,
                     fontsize=7.5, va="bottom", ha="center")
    axes[0].set_ylabel("CC vs. ground truth", color=INK)
    axes[0].set_ylim(0, 0.75)
    axes[1].axhline(0, color=GRAY_LINE, linewidth=0.8)
    axes[1].set_ylabel("Partial CC (prior controlled)", color=INK)
    axes[1].set_ylim(-0.02, 0.40)
    others = ccp.drop(index=list(HIGHLIGHT))
    axes[1].annotate("other models", (1.0, others.loc[:, "3s"].max() + 0.015),
                     color=MUTED, fontsize=7, ha="center", va="bottom")

    fig.savefig(OUT / "fig_duration_cc.pdf")
    plt.close(fig)


def fig_category():
    df = pd.read_csv(RESULTS / "7s" / "per_image_avg.csv")
    order = (df.groupby("model")["CC_mean"].mean().sort_values(ascending=True).index)
    cats = ["desktop", "mobile", "poster", "web"]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.6), sharey=True, sharex=True,
                             constrained_layout=True)
    for ax, cat in zip(axes, cats):
        style_ax(ax)
        sub = df[df["category"] == cat].groupby("model")["CC_mean"]
        means, ns, stds = sub.mean()[order], sub.count()[order], sub.std()[order]
        ci = 1.96 * stds / np.sqrt(ns)
        y = np.arange(len(order))
        ax.barh(y, means, xerr=ci, color=BLUE, height=0.62,
                error_kw={"ecolor": MUTED, "elinewidth": 0.8})
        ax.set_yticks(y, [SHORT[m] for m in order], fontsize=7)
        ax.set_title(cat, fontsize=9, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=GRAY_LINE, linewidth=0.8)
    axes[0].set_xlabel("CC at 7 s (mean, 95% CI)", color=MUTED)
    fig.savefig(OUT / "fig_category_cc.pdf")
    plt.close(fig)


def fig_scale_artifact():
    """Three-cell decomposition: parser effect and collection-time effect.

    Re-parsing the retained August responses with the original clamping parser
    supplies the middle bar, which holds collection time fixed and isolates the
    parser; comparing that bar to April holds the parser fixed and isolates time.
    """
    dec = pd.read_csv(RESULTS / "parser_vs_time" / "summary.csv").set_index("model")
    order = dec["aug_new"].sort_values().index
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(5.2, 2.4), constrained_layout=True)
    style_ax(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    bars = [("apr_old", GRAY_LINE, "April · clamping"),
            ("aug_old", ORANGE, "August · clamping"),
            ("aug_new", BLUE, "August · scale-aware")]
    for off, (col, color, label) in zip((-0.26, 0.0, 0.26), bars):
        ax.barh(y + off, dec.loc[order, col], height=0.24, color=color, label=label)

    for i, m in enumerate(order):
        p, t = dec.loc[m, "parser_effect"], dec.loc[m, "time_effect"]
        ax.annotate(f"parser {p:+.2f}   time {t:+.2f}",
                    (dec.loc[m, "aug_new"] + 0.01, i), fontsize=6.5,
                    color=INK, va="center")
    ax.set_yticks(y, [SHORT[m] for m in order], fontsize=7.5)
    ax.set_xlabel("CC at 7 s", color=INK)
    ax.set_xlim(0, 0.56)
    ax.legend(fontsize=6.5, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=3, borderaxespad=0,
              columnspacing=1.0, handlelength=1.2, handletextpad=0.4)
    fig.savefig(OUT / "fig_scale_artifact.pdf")
    plt.close(fig)


def qualitative_examples():
    """Representative (non-degenerate) best and worst GPT-5.4 cases at 7s."""
    dataset = load_dataset("data", duration="7s")
    by_id = {s.image_id: s for s in dataset}
    per = pd.read_csv(RESULTS / "7s" / "per_image_avg.csv")
    per = per[per["model"] == "gpt-5.4"]
    ent = pd.read_csv(RESULTS / "baselines" / "7s" / "gt_stats.csv")[
        ["image_id", "gt_entropy_norm"]]
    per = per.merge(ent, on="image_id")
    med = per["gt_entropy_norm"].median()

    rich = per[per["gt_entropy_norm"] >= med]
    best = rich[rich["category"] == "desktop"].nlargest(1, "CC_mean").iloc[0]
    worst = per[per["category"] == "mobile"].nsmallest(1, "CC_mean").iloc[0]

    for row, label in ((best, "best"), (worst, "worst")):
        s = by_id[row["image_id"]]
        pred_path = RESULTS / "predictions" / "gpt-5.4" / f"{s.image_id}_run01.json"
        points = [GazePoint(**p) for p in json.loads(pred_path.read_text())]
        gt = plt.imread(s.heatmap_path)
        gt = gt.mean(axis=2) if gt.ndim == 3 else gt
        h, w = gt.shape
        pred = generate_saliency_map(points, width=w, height=h)
        out_path = OUT / f"fig_qualitative_{label}_{s.image_id}_cc{row['CC_mean']:.3f}.png"
        plot_saliency_comparison(s.image_path, pred, gt / (gt.max() or 1), out_path,
                                 model_name="gpt-5.4")
        print(f"  {label}: {s.image_id} ({row['category']}) CC={row['CC_mean']:.3f} "
              f"entropy={row['gt_entropy_norm']:.3f} -> {out_path.name}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_duration()
    fig_category()
    fig_scale_artifact()
    qualitative_examples()
    print(f"Figures written to {OUT}/")
