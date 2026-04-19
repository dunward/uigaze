"""Pilot experiment: run VLM saliency prediction on a small subset of UEyes."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_dataset, sample_pilot
from src.vlm.openrouter import predict_batch, MODELS
from src.saliency.generator import generate_saliency_map
from src.metrics.evaluator import evaluate_all
from src.analysis.visualizer import plot_saliency_comparison

METRICS = ["CC", "SIM", "KL"]


def load_ground_truth(heatmap_path: Path) -> np.ndarray:
    """Load and normalize a ground truth heatmap image."""
    img = Image.open(heatmap_path).convert("L")
    arr = np.array(img, dtype=np.float64)
    max_val = arr.max()
    if max_val > 0:
        arr /= max_val
    return arr


def save_predictions(predictions, model_name, run, output_dir):
    """Save raw VLM gaze point predictions as JSON."""
    pred_dir = output_dir / "predictions" / model_name
    pred_dir.mkdir(parents=True, exist_ok=True)

    for image_id, points in predictions.items():
        data = [asdict(p) for p in points]
        path = pred_dir / f"{image_id}_run{run:02d}.json"
        path.write_text(json.dumps(data, indent=2))


def save_best_worst_images(df, pilot_samples, output_dir):
    """Save comparison images for best/worst CC per model x category."""
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    samples_by_id = {s.image_id: s for s in pilot_samples}

    # Use mean CC across runs per (model, category, image_id)
    grouped = df.groupby(["model", "category", "image_id"])["CC"].mean().reset_index()

    for (model, category), group in grouped.groupby(["model", "category"]):
        for label, idx in [("best", group["CC"].idxmax()), ("worst", group["CC"].idxmin())]:
            row = group.loc[idx]
            sample = samples_by_id.get(row["image_id"])
            if not sample:
                continue

            # Load the first run's prediction for this image
            pred_path = output_dir / "predictions" / model / f"{row['image_id']}_run01.json"
            if not pred_path.exists():
                continue

            points_data = json.loads(pred_path.read_text())
            from src.vlm.openrouter import GazePoint
            points = [GazePoint(**p) for p in points_data]

            gt = load_ground_truth(sample.heatmap_path)
            h, w = gt.shape
            pred = generate_saliency_map(points, width=w, height=h)

            out_path = img_dir / f"{model}_{category}_{label}.png"
            plot_saliency_comparison(
                sample.image_path, pred, gt, out_path, model_name=model
            )
            print(f"  Saved {label}: {out_path}")


def run_pilot(
    data_dir: str = "data",
    n_per_category: int = 10,
    n_runs: int = 10,
    concurrency: int = 10,
    models: list[str] | None = None,
    output_dir: str = "results",
):
    if models is None:
        models = list(MODELS.keys())

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset and sample pilot subset
    print("Loading UEyes dataset...")
    dataset = load_dataset(data_dir, duration="3s")
    pilot = sample_pilot(dataset, n_per_category=n_per_category)
    print(f"Pilot: {len(pilot)} images x {len(models)} models x {n_runs} runs")
    print(f"Total API calls: {len(pilot) * len(models) * n_runs}")
    print(f"Concurrency: {concurrency}")

    categories = sorted(set(s.category for s in pilot))
    for cat in categories:
        count = sum(1 for s in pilot if s.category == cat)
        print(f"  {cat}: {count} images")

    # --- Raw results ---
    all_results = []

    for model_name in models:
        for run in range(1, n_runs + 1):
            print(f"\n{'='*60}")
            print(f"Model: {model_name} | Run {run}/{n_runs}")
            print(f"{'='*60}")

            image_paths = [s.image_path for s in pilot]
            predictions = predict_batch(
                image_paths, model=model_name, concurrency=concurrency
            )

            # Save raw predictions
            save_predictions(predictions, model_name, run, output_path)

            for sample in pilot:
                points = predictions.get(sample.image_path.stem, [])
                if not points:
                    print(f"  SKIP {sample.image_id} (no predictions)")
                    continue

                gt = load_ground_truth(sample.heatmap_path)
                h, w = gt.shape
                pred = generate_saliency_map(points, width=w, height=h)
                metrics = evaluate_all(pred, gt)

                all_results.append({
                    "model": model_name,
                    "run": run,
                    "image_id": sample.image_id,
                    "category": sample.category,
                    "n_points": len(points),
                    **metrics,
                })

    df = pd.DataFrame(all_results)

    # --- Save raw metrics ---
    raw_path = output_path / "raw_results.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nRaw results saved to {raw_path}")

    # --- Per-image stats (across runs) ---
    image_avg = df.groupby(["model", "category", "image_id"])[METRICS].agg(["mean", "min", "max"])
    image_avg.columns = [f"{m}_{s}" for m, s in image_avg.columns]
    image_avg = image_avg.reset_index()

    image_avg_path = output_path / "per_image_avg.csv"
    image_avg.to_csv(image_avg_path, index=False)

    # --- Summary by model (based on per-image mean) ---
    mean_cols = [f"{m}_mean" for m in METRICS]
    model_summary = image_avg.groupby("model")[mean_cols].agg(["mean", "std"])
    model_summary.columns = [f"{m}_{s}" for m, s in model_summary.columns]
    model_summary = model_summary.reset_index()

    model_path = output_path / "summary_by_model.csv"
    model_summary.to_csv(model_path, index=False)

    # --- Summary by model x category ---
    cat_summary = image_avg.groupby(["model", "category"])[mean_cols].agg(["mean", "std"])
    cat_summary.columns = [f"{m}_{s}" for m, s in cat_summary.columns]
    cat_summary = cat_summary.reset_index()

    cat_path = output_path / "summary_by_category.csv"
    cat_summary.to_csv(cat_path, index=False)

    # --- Best/worst comparison images ---
    print("\nGenerating best/worst comparison images...")
    save_best_worst_images(df, pilot, output_path)

    # --- Print ---
    print(f"\n{'='*60}")
    print("SUMMARY BY MODEL")
    print(f"{'='*60}")
    print(model_summary.to_string(index=False))

    print(f"\n{'='*60}")
    print("SUMMARY BY MODEL x CATEGORY")
    print(f"{'='*60}")
    print(cat_summary.to_string(index=False))

    print(f"\nSaved:")
    print(f"  {raw_path}")
    print(f"  {image_avg_path}")
    print(f"  {model_path}")
    print(f"  {cat_path}")
    print(f"  {output_path / 'predictions/'} (raw JSON coordinates)")
    print(f"  {output_path / 'images/'} (best/worst comparisons)")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run UIGaze pilot experiment")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--n-per-category", type=int, default=10)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"Options: {list(MODELS.keys())}")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max parallel API requests")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    run_pilot(
        data_dir=args.data_dir,
        n_per_category=args.n_per_category,
        n_runs=args.n_runs,
        concurrency=args.concurrency,
        models=args.models,
        output_dir=args.output_dir,
    )
