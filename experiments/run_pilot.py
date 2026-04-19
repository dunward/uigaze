"""Pilot experiment: run VLM saliency prediction on a small subset of UEyes."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_dataset, sample_pilot
from src.vlm.openrouter import predict_saliency, MODELS
from src.saliency.generator import generate_saliency_map
from src.metrics.evaluator import evaluate_all

METRICS = ["CC", "SIM", "KL"]


def load_ground_truth(heatmap_path: Path) -> np.ndarray:
    """Load and normalize a ground truth heatmap image."""
    img = Image.open(heatmap_path).convert("L")
    arr = np.array(img, dtype=np.float64)
    max_val = arr.max()
    if max_val > 0:
        arr /= max_val
    return arr


def run_pilot(
    data_dir: str = "data",
    n_per_category: int = 10,
    n_runs: int = 10,
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

            for i, sample in enumerate(pilot):
                print(f"  [{i+1}/{len(pilot)}] {sample.image_id} ({sample.category})")

                try:
                    points = predict_saliency(sample.image_path, model=model_name)
                except Exception as e:
                    print(f"    ERROR: {e}")
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

    # --- Save raw ---
    raw_path = output_path / "raw_results.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nRaw results saved to {raw_path}")

    # --- Summary by model ---
    model_summary = df.groupby("model")[METRICS].agg(["mean", "std"])
    model_summary.columns = [f"{m}_{s}" for m, s in model_summary.columns]
    model_summary = model_summary.reset_index()

    model_path = output_path / "summary_by_model.csv"
    model_summary.to_csv(model_path, index=False)

    # --- Summary by model x category ---
    cat_summary = df.groupby(["model", "category"])[METRICS].agg(["mean", "std"])
    cat_summary.columns = [f"{m}_{s}" for m, s in cat_summary.columns]
    cat_summary = cat_summary.reset_index()

    cat_path = output_path / "summary_by_category.csv"
    cat_summary.to_csv(cat_path, index=False)

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
    print(f"  {model_path}")
    print(f"  {cat_path}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run UIGaze pilot experiment")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--n-per-category", type=int, default=10)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"Options: {list(MODELS.keys())}")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    run_pilot(
        data_dir=args.data_dir,
        n_per_category=args.n_per_category,
        n_runs=args.n_runs,
        models=args.models,
        output_dir=args.output_dir,
    )
