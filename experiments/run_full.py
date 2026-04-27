"""Full experiment: collect VLM saliency predictions on the entire UEyes dataset."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_dataset
from src.vlm.openrouter import predict_batch, MODELS

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "full"


def run_full(
    data_dir: str = "data",
    n_runs: int = 3,
    concurrency: int = 10,
    models: list[str] | None = None,
    categories: list[str] | None = None,
):
    if models is None:
        models = list(MODELS.keys())

    pred_dir = OUTPUT_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    print("Loading UEyes dataset...")
    dataset = load_dataset(data_dir, duration="3s", categories=categories)
    print(f"Full: {len(dataset)} images x {len(models)} models x {n_runs} runs")
    print(f"Total API calls: {len(dataset) * len(models) * n_runs}")
    print(f"Concurrency: {concurrency}")

    cat_counts: dict[str, int] = {}
    for s in dataset:
        cat_counts[s.category] = cat_counts.get(s.category, 0) + 1
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count} images")

    for model_name in models:
        model_pred_dir = pred_dir / model_name
        model_pred_dir.mkdir(parents=True, exist_ok=True)

        completed_runs = set()
        for run in range(1, n_runs + 1):
            all_exist = all(
                (model_pred_dir / f"{s.image_id}_run{run:02d}.json").exists()
                for s in dataset
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

            # Skip images already saved for this run (mid-run resume)
            pending = [
                s for s in dataset
                if not (model_pred_dir / f"{s.image_id}_run{run:02d}.json").exists()
            ]
            if not pending:
                print("  Already complete (all JSONs exist)")
                continue

            print(f"  Pending: {len(pending)}/{len(dataset)} images")

            image_paths = [s.image_path for s in pending]
            predictions = predict_batch(
                image_paths, model=model_name, concurrency=concurrency
            )

            saved = 0
            for sample in pending:
                points = predictions.get(sample.image_path.stem, [])
                if not points:
                    print(f"  SKIP {sample.image_id} (no predictions)")
                    continue

                data = [asdict(p) for p in points]
                path = model_pred_dir / f"{sample.image_id}_run{run:02d}.json"
                path.write_text(json.dumps(data, indent=2))
                saved += 1

            print(f"  Run {run} saved ({saved}/{len(pending)} predictions)")

    print(f"\nDone. Predictions saved to {pred_dir}/")
    print("Run 'uv run python experiments/fill_missing.py --target full --model <name>' to retry failed pairs.")
    print("Run 'uv run python experiments/regenerate.py --target full' to compute metrics.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run UIGaze full experiment")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Repetitions per image (default: 3, validated by pilot sensitivity check)")
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"Options: {list(MODELS.keys())}")
    parser.add_argument("--categories", nargs="+", default=None,
                        choices=["webpage", "desktop", "mobile", "poster"],
                        help="Filter by category. Default: all")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max parallel API requests")
    args = parser.parse_args()

    run_full(
        data_dir=args.data_dir,
        n_runs=args.n_runs,
        concurrency=args.concurrency,
        models=args.models,
        categories=args.categories,
    )
