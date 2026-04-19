"""Single image test run for quick pipeline verification."""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_dataset, sample_pilot
from src.vlm.openrouter import predict_saliency, MODELS
from src.saliency.generator import generate_saliency_map
from src.metrics.evaluator import evaluate_all
from src.analysis.visualizer import plot_saliency_comparison


def run_single(
    data_dir: str = "data",
    model: str = "gpt-5.4-mini",
    image_id: str | None = None,
    output_dir: str = "results",
):
    dataset = load_dataset(data_dir, duration="3s")

    if image_id:
        matches = [s for s in dataset if s.image_id == image_id]
        if not matches:
            print(f"Image ID '{image_id}' not found in dataset.")
            sys.exit(1)
        sample = matches[0]
    else:
        sample = sample_pilot(dataset, n_per_category=1)[0]

    print(f"Image:    {sample.image_path}")
    print(f"Category: {sample.category}")
    print(f"Model:    {model}")
    print()

    # VLM prediction
    print("Calling VLM...")
    points = predict_saliency(sample.image_path, model=model)
    print(f"Got {len(points)} gaze points")
    print()

    # Load ground truth
    gt = np.array(Image.open(sample.heatmap_path).convert("L"), dtype=np.float64)
    if gt.max() > 0:
        gt /= gt.max()
    h, w = gt.shape

    # Generate predicted saliency map
    pred = generate_saliency_map(points, width=w, height=h)

    # Evaluate
    metrics = evaluate_all(pred, gt)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save comparison image
    output_path = Path(output_dir) / f"test_{sample.image_id}_{model}.png"
    plot_saliency_comparison(
        sample.image_path, pred, gt, output_path, model_name=model
    )
    print(f"\nSaved comparison to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single image test run")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model", default="gpt-5.4-mini", help=f"Options: {list(MODELS.keys())}")
    parser.add_argument("--image-id", default=None, help="Specific image ID to test")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    run_single(
        data_dir=args.data_dir,
        model=args.model,
        image_id=args.image_id,
        output_dir=args.output_dir,
    )
