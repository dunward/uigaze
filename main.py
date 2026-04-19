"""UIGaze: Benchmarking VLM Saliency Prediction on User Interfaces."""

import argparse
import sys
from pathlib import Path

from src.data_loader import load_dataset, sample_pilot
from src.vlm.openrouter import predict_saliency, MODELS
from src.saliency.generator import generate_saliency_map
from src.metrics.evaluator import evaluate_all
from src.analysis.visualizer import (
    plot_model_comparison,
    plot_category_heatmap,
)


def cmd_info(args):
    """Show dataset info."""
    dataset = load_dataset(args.data_dir, duration=args.duration)
    categories = {}
    for s in dataset:
        categories[s.category] = categories.get(s.category, 0) + 1

    print(f"UEyes dataset: {len(dataset)} samples")
    print(f"Duration: {args.duration}")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")


def cmd_visualize(args):
    """Generate visualization from results CSV."""
    csv_path = Path(args.results_csv)
    if not csv_path.exists():
        print(f"Results file not found: {csv_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    plot_model_comparison(csv_path, output_dir / "model_comparison.png")
    plot_category_heatmap(csv_path, output_path=output_dir / "category_heatmap.png")
    print("Visualization complete.")


def main():
    parser = argparse.ArgumentParser(
        description="UIGaze: Benchmarking VLM Saliency Prediction on User Interfaces"
    )
    subparsers = parser.add_subparsers(dest="command")

    # info command
    info_parser = subparsers.add_parser("info", help="Show dataset info")
    info_parser.add_argument("--data-dir", default="data")
    info_parser.add_argument("--duration", default="3s", choices=["1s", "3s", "7s"])

    # visualize command
    viz_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    viz_parser.add_argument("--results-csv", default="results/pilot_results.csv")
    viz_parser.add_argument("--output-dir", default="results")

    args = parser.parse_args()

    if args.command == "info":
        cmd_info(args)
    elif args.command == "visualize":
        cmd_visualize(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
