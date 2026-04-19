"""Pilot experiment: run VLM saliency prediction on a small subset of UEyes."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_dataset, sample_pilot
from src.vlm.openrouter import predict_batch, MODELS
from src.saliency.generator import generate_saliency_map
from src.metrics.evaluator import evaluate_all


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
    models: list[str] | None = None,
    output_dir: str = "results",
):
    """Run the pilot experiment.

    Args:
        data_dir: Path to UEyes data directory.
        n_per_category: Number of images per category for pilot.
        models: List of model short names to evaluate. Defaults to all.
        output_dir: Directory to save results.
    """
    if models is None:
        models = list(MODELS.keys())

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset and sample pilot subset
    print("Loading UEyes dataset...")
    dataset = load_dataset(data_dir, duration="3s")
    pilot = sample_pilot(dataset, n_per_category=n_per_category)
    print(f"Pilot subset: {len(pilot)} images")

    categories = sorted(set(s.category for s in pilot))
    for cat in categories:
        count = sum(1 for s in pilot if s.category == cat)
        print(f"  {cat}: {count} images")

    all_results = []

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        image_paths = [s.image_path for s in pilot]
        predictions = predict_batch(image_paths, model=model_name)

        for sample in pilot:
            gaze_points = predictions.get(sample.image_path.stem, [])

            if not gaze_points:
                print(f"  SKIP {sample.image_id} (no predictions)")
                continue

            # Load ground truth
            gt = load_ground_truth(sample.heatmap_path)
            height, width = gt.shape

            # Generate predicted saliency map at same resolution
            pred = generate_saliency_map(
                gaze_points, width=width, height=height
            )

            # Evaluate (without fixation map for now - CC, SIM, KL only)
            metrics = evaluate_all(pred, gt)

            result = {
                "model": model_name,
                "image_id": sample.image_id,
                "category": sample.category,
                "n_points": len(gaze_points),
                **metrics,
            }
            all_results.append(result)

    # Save results
    df = pd.DataFrame(all_results)
    csv_path = output_path / "pilot_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY (mean metrics)")
    print(f"{'='*60}")

    summary = df.groupby("model")[["CC", "SIM", "KL"]].mean()
    print(summary.to_string())

    print(f"\nBy category:")
    cat_summary = df.groupby(["model", "category"])[["CC", "SIM", "KL"]].mean()
    print(cat_summary.to_string())

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run UIGaze pilot experiment")
    parser.add_argument("--data-dir", default="data", help="UEyes data directory")
    parser.add_argument("--n-per-category", type=int, default=10)
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"Models to evaluate. Options: {list(MODELS.keys())}")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    run_pilot(
        data_dir=args.data_dir,
        n_per_category=args.n_per_category,
        models=args.models,
        output_dir=args.output_dir,
    )
