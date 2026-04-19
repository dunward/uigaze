"""Regenerate summary CSVs and best/worst images from existing raw data."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vlm.openrouter import GazePoint
from src.saliency.generator import generate_saliency_map
from src.metrics.evaluator import evaluate_all
from src.analysis.visualizer import plot_saliency_comparison
from src.data_loader import load_dataset, sample_pilot

METRICS = ["CC", "SIM", "KL"]


def load_ground_truth(heatmap_path: Path) -> np.ndarray:
    img = Image.open(heatmap_path).convert("L")
    arr = np.array(img, dtype=np.float64)
    if arr.max() > 0:
        arr /= arr.max()
    return arr


def regenerate(target: str = "pilot", data_dir: str = "data"):
    results_dir = Path(__file__).parent.parent / "results" / target
    raw_dir = results_dir / "raw"

    if not raw_dir.exists():
        print(f"No raw data found at {raw_dir}")
        sys.exit(1)

    # Load all raw CSVs
    all_csvs = list(raw_dir.glob("*.csv"))
    if not all_csvs:
        print(f"No CSV files in {raw_dir}")
        sys.exit(1)

    df = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)
    print(f"Loaded {len(df)} results from {len(all_csvs)} model(s)")

    # Per-image stats
    image_avg = df.groupby(["model", "category", "image_id"])[METRICS].agg(["mean", "min", "max"])
    image_avg.columns = [f"{m}_{s}" for m, s in image_avg.columns]
    image_avg = image_avg.reset_index()
    image_avg.to_csv(results_dir / "per_image_avg.csv", index=False)

    # Summary by model
    mean_cols = [f"{m}_mean" for m in METRICS]
    model_summary = image_avg.groupby("model")[mean_cols].agg(["mean", "std"])
    model_summary.columns = [f"{m}_{s}" for m, s in model_summary.columns]
    model_summary = model_summary.reset_index()
    model_summary.to_csv(results_dir / "summary_by_model.csv", index=False)

    # Summary by model x category
    cat_summary = image_avg.groupby(["model", "category"])[mean_cols].agg(["mean", "std"])
    cat_summary.columns = [f"{m}_{s}" for m, s in cat_summary.columns]
    cat_summary = cat_summary.reset_index()
    cat_summary.to_csv(results_dir / "summary_by_category.csv", index=False)

    print("\nSummary CSVs regenerated.")

    # Best/worst images
    print("\nGenerating best/worst images...")
    dataset = load_dataset(data_dir, duration="3s")
    if target == "pilot":
        samples = sample_pilot(dataset, n_per_category=10)
    else:
        samples = dataset
    samples_by_id = {s.image_id: s for s in samples}

    img_dir = results_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for (model, category), group in image_avg.groupby(["model", "category"]):
        for label, idx in [("best", group["CC_mean"].idxmax()), ("worst", group["CC_mean"].idxmin())]:
            row = group.loc[idx]
            image_id = row["image_id"]
            sample = samples_by_id.get(image_id)
            if not sample:
                continue

            pred_path = results_dir / "predictions" / model / f"{image_id}_run01.json"
            if not pred_path.exists():
                print(f"  SKIP {model}/{image_id} (no prediction JSON)")
                continue

            points = [GazePoint(**p) for p in json.loads(pred_path.read_text())]
            gt = load_ground_truth(sample.heatmap_path)
            h, w = gt.shape
            pred = generate_saliency_map(points, width=w, height=h)

            cc = row["CC_mean"]
            out_path = img_dir / f"{model}_{category}_{label}_cc{cc:.3f}.png"
            plot_saliency_comparison(
                sample.image_path, pred, gt, out_path, model_name=model
            )
            print(f"  {label} ({category}): CC={cc:.3f} → {out_path.name}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY BY MODEL")
    print(f"{'='*60}")
    print(model_summary.to_string(index=False))

    print(f"\n{'='*60}")
    print("SUMMARY BY MODEL x CATEGORY")
    print(f"{'='*60}")
    print(cat_summary.to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate summaries and images from raw data")
    parser.add_argument("--target", default="pilot", choices=["pilot", "full"])
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    regenerate(target=args.target, data_dir=args.data_dir)
