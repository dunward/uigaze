"""Inferential statistics for model / duration / category comparisons.

All comparisons are paired at the image level (each image contributes one
score per condition, averaged across the 3 runs) so that image difficulty is
controlled for. Uses Friedman's test for the omnibus effect across >2
related conditions, followed by pairwise Wilcoxon signed-rank tests with
Holm correction. Effect sizes: rank-biserial correlation for Wilcoxon pairs,
Kendall's W for the Friedman omnibus. 95% CIs via BCa bootstrap on the mean
difference.
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

N_BOOT = 10000
RNG_SEED = 12345


def _wide(per_image_avg: pd.DataFrame, group_col: str, metric: str) -> pd.DataFrame:
    """Pivot to image_id x group_col, keeping only images present for all groups."""
    wide = per_image_avg.pivot_table(index="image_id", columns=group_col, values=f"{metric}_mean")
    return wide.dropna(axis=0, how="any")


def bootstrap_ci_diff(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT, seed: int = RNG_SEED):
    rng = np.random.default_rng(seed)
    diff = a - b
    n = len(diff)
    boot = rng.choice(diff, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation effect size for Wilcoxon."""
    diff = a - b
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = sstats.rankdata(np.abs(diff))
    r_plus = ranks[diff > 0].sum()
    r_minus = ranks[diff < 0].sum()
    return float((r_plus - r_minus) / ranks.sum())


def pairwise_wilcoxon(wide: pd.DataFrame) -> pd.DataFrame:
    groups = list(wide.columns)
    rows = []
    for g1, g2 in itertools.combinations(groups, 2):
        a, b = wide[g1].values, wide[g2].values
        try:
            stat, p = sstats.wilcoxon(a, b, method="approx", zero_method="wilcox")
        except ValueError:
            stat, p = np.nan, 1.0
        mean_diff, lo, hi = bootstrap_ci_diff(a, b)
        rows.append({
            "group_1": g1, "group_2": g2, "n": len(a),
            "mean_diff": mean_diff, "ci95_lo": lo, "ci95_hi": hi,
            "wilcoxon_stat": stat, "p_raw": p,
            "rank_biserial_r": rank_biserial(a, b),
        })
    df = pd.DataFrame(rows).sort_values("p_raw")
    reject, p_holm, _, _ = _holm(df["p_raw"].values)
    df["p_holm"] = p_holm
    df["sig_holm_05"] = reject
    return df.sort_values(["group_1", "group_2"]).reset_index(drop=True)


def _holm(pvals: np.ndarray):
    from statsmodels.stats.multitest import multipletests
    return multipletests(pvals, alpha=0.05, method="holm")


def friedman_omnibus(wide: pd.DataFrame) -> dict:
    arrays = [wide[c].values for c in wide.columns]
    stat, p = sstats.friedmanchisquare(*arrays)
    n, k = wide.shape
    kendall_w = stat / (n * (k - 1))
    return {"n_images": n, "n_groups": k, "chi2": float(stat), "p": float(p), "kendalls_w": float(kendall_w)}


def run_comparison(per_image_avg: pd.DataFrame, group_col: str, metric: str, label: str, out_dir: Path):
    wide = _wide(per_image_avg, group_col, metric)
    omnibus = friedman_omnibus(wide)
    pairwise = pairwise_wilcoxon(wide)

    out_dir.mkdir(parents=True, exist_ok=True)
    pairwise.to_csv(out_dir / f"{label}_pairwise.csv", index=False)
    pd.DataFrame([omnibus]).to_csv(out_dir / f"{label}_omnibus.csv", index=False)

    print(f"\n=== {label} ({metric}) ===")
    print(f"Friedman: chi2={omnibus['chi2']:.2f}, p={omnibus['p']:.2e}, "
          f"Kendall's W={omnibus['kendalls_w']:.3f}, n={omnibus['n_images']} images, "
          f"{omnibus['n_groups']} groups")
    sig = pairwise[pairwise["sig_holm_05"]]
    print(f"Pairwise (Holm-corrected): {sig.shape[0]}/{pairwise.shape[0]} pairs significant at alpha=.05")
    return omnibus, pairwise


def model_ranking_table(per_image_avg: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean +/- bootstrap 95% CI per model, for reporting alongside pairwise tests."""
    rows = []
    for model, g in per_image_avg.groupby("model"):
        vals = g[f"{metric}_mean"].dropna().values
        rng = np.random.default_rng(RNG_SEED)
        boot = rng.choice(vals, size=(N_BOOT, len(vals)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"model": model, "n": len(vals), "mean": vals.mean(),
                     "ci95_lo": lo, "ci95_hi": hi})
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


def run_all(results_dir: Path, out_root: Path, durations: list[str], metric: str = "CC"):
    for duration in durations:
        dur_dir = results_dir / duration
        per_image_avg = pd.read_csv(dur_dir / "per_image_avg.csv")
        out_dir = out_root / duration

        ranking = model_ranking_table(per_image_avg, metric)
        out_dir.mkdir(parents=True, exist_ok=True)
        ranking.to_csv(out_dir / "model_ranking.csv", index=False)
        print(f"\n### Duration {duration} - model ranking ({metric}, bootstrap 95% CI) ###")
        print(ranking.round(3).to_string(index=False))

        run_comparison(per_image_avg, "model", metric, "model", out_dir)
        run_comparison(per_image_avg, "category", metric, "category", out_dir)

    # Duration effect: needs per_image_avg merged across durations for same model
    frames = []
    for duration in durations:
        df = pd.read_csv(results_dir / duration / "per_image_avg.csv")
        df["duration"] = duration
        frames.append(df)
    all_dur = pd.concat(frames)

    for model in all_dur["model"].unique():
        sub = all_dur[all_dur["model"] == model]
        out_dir = out_root / "duration_effect"
        wide = _wide(sub, "duration", metric)
        if wide.shape[1] < 2:
            continue
        omnibus = friedman_omnibus(wide)
        pairwise = pairwise_wilcoxon(wide)
        out_dir.mkdir(parents=True, exist_ok=True)
        pairwise.to_csv(out_dir / f"{model}_pairwise.csv", index=False)
        pd.DataFrame([omnibus]).to_csv(out_dir / f"{model}_omnibus.csv", index=False)
        print(f"\n=== duration effect: {model} ({metric}) ===")
        print(f"Friedman: chi2={omnibus['chi2']:.2f}, p={omnibus['p']:.2e}, W={omnibus['kendalls_w']:.3f}")
        print(pairwise[["group_1", "group_2", "mean_diff", "ci95_lo", "ci95_hi", "p_holm", "sig_holm_05"]]
              .round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inferential statistics on saliency results")
    parser.add_argument("--target", default="full", choices=["pilot", "full"])
    parser.add_argument("--metric", default="CC", choices=["CC", "SIM", "KL", "NSS", "AUC", "CC_partial"])
    parser.add_argument("--durations", nargs="+", default=["1s", "3s", "7s"])
    args = parser.parse_args()

    base = Path(__file__).parent.parent.parent / "results" / args.target
    run_all(base, base / "stats" / args.metric, args.durations, metric=args.metric)
