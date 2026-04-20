"""Pilot experiment: collect VLM saliency predictions on a small subset of UEyes."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_dataset, sample_pilot
from src.vlm.openrouter import predict_batch, MODELS

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "pilot"


def run_pilot(
    data_dir: str = "data",
    n_per_category: int = 10,
    n_runs: int = 10,
    concurrency: int = 10,
    models: list[str] | None = None,
):
    if models is None:
        models = list(MODELS.keys())

    pred_dir = OUTPUT_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

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

    for model_name in models:
        model_pred_dir = pred_dir / model_name
        model_pred_dir.mkdir(parents=True, exist_ok=True)

        # Resume: check which runs already have predictions saved
        completed_runs = set()
        for run in range(1, n_runs + 1):
            # A run is complete if all pilot images have a prediction JSON
            all_exist = all(
                (model_pred_dir / f"{s.image_id}_run{run:02d}.json").exists()
                for s in pilot
            )
            if all_exist:
                completed_runs.add(run)

        if completed_runs:
            print(f"\nResuming {model_name}: runs {sorted(completed_runs)} already done")

        for run in range(1, n_runs + 1):
            if run in completed_runs:
                print(f"\n  Skipping {model_name} run {run} (already done)")
                continue

            print(f"\n{'='*60}")
            print(f"Model: {model_name} | Run {run}/{n_runs}")
            print(f"{'='*60}")

            image_paths = [s.image_path for s in pilot]
            predictions = predict_batch(
                image_paths, model=model_name, concurrency=concurrency
            )

            # Save predictions as JSON
            saved = 0
            for sample in pilot:
                points = predictions.get(sample.image_path.stem, [])
                if not points:
                    print(f"  SKIP {sample.image_id} (no predictions)")
                    continue

                data = [asdict(p) for p in points]
                path = model_pred_dir / f"{sample.image_id}_run{run:02d}.json"
                path.write_text(json.dumps(data, indent=2))
                saved += 1

            print(f"  Run {run} saved ({saved} predictions)")

    print(f"\nDone. Predictions saved to {pred_dir}/")
    print(f"Run 'uv run python experiments/regenerate.py' to compute metrics and generate images.")


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
    args = parser.parse_args()

    run_pilot(
        data_dir=args.data_dir,
        n_per_category=args.n_per_category,
        n_runs=args.n_runs,
        concurrency=args.concurrency,
        models=args.models,
    )
