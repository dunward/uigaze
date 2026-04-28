"""Regenerate metrics, summaries, and images from saved prediction JSONs."""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
DURATIONS = ["1s", "3s", "7s"]


def load_ground_truth(heatmap_path: Path) -> np.ndarray:
    img = Image.open(heatmap_path).convert("L")
    arr = np.array(img, dtype=np.float64)
    if arr.max() > 0:
        arr /= arr.max()
    return arr


def _compute_one(task):
    """Worker function: load prediction + GT, compute metrics. Runs in a subprocess."""
    pred_file, heatmap_path, model_name, image_id, run, category = task
    points_data = json.loads(Path(pred_file).read_text())
    points = [GazePoint(**p) for p in points_data]
    if not points:
        return None
    gt = load_ground_truth(Path(heatmap_path))
    h, w = gt.shape
    pred = generate_saliency_map(points, width=w, height=h)
    metrics = evaluate_all(pred, gt)
    return {
        "model": model_name,
        "run": run,
        "image_id": image_id,
        "category": category,
        "n_points": len(points),
        **metrics,
    }


def recompute_metrics(results_dir: Path, samples_by_id: dict, duration: str, data_dir: str, models: list[str] | None = None, workers: int | None = None):
    """Recompute metrics from prediction JSONs for a given duration (parallelized)."""
    pred_base = results_dir / "predictions"
    if not pred_base.exists():
        return pd.DataFrame()

    dur_dir = results_dir / duration / "raw"
    heatmap_dir = Path(data_dir) / "saliency_maps" / f"heatmaps_{duration}"

    all_results = []
    tasks = []

    for model_dir in sorted(pred_base.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        if models and model_name not in models:
            existing_csv = dur_dir / f"{model_name}.csv"
            if existing_csv.exists():
                existing = pd.read_csv(existing_csv)
                all_results.extend(existing.to_dict("records"))
            continue

        if models is None:
            existing_csv = dur_dir / f"{model_name}.csv"
            if existing_csv.exists():
                existing = pd.read_csv(existing_csv)
                all_results.extend(existing.to_dict("records"))
                print(f"  {model_name}: loaded from cache")
                continue

        for pred_file in sorted(model_dir.glob("*_run*.json")):
            stem = pred_file.stem
            parts = stem.rsplit("_run", 1)
            if len(parts) != 2:
                continue
            image_id = parts[0]
            run = int(parts[1])

            sample = samples_by_id.get(image_id)
            if not sample:
                continue

            heatmap_path = heatmap_dir / sample.image_path.name
            if not heatmap_path.exists():
                continue

            tasks.append((str(pred_file), str(heatmap_path), model_name, image_id, run, sample.category))

    if not tasks:
        return pd.DataFrame(all_results)

    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    print(f"  Computing {len(tasks)} predictions across {workers} workers...", flush=True)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_compute_one, t) for t in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                all_results.append(res)
            done += 1
            if done % 500 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                print(f"    {done}/{len(tasks)} ({rate:.0f}/s, eta {eta:.0f}s)", flush=True)

    return pd.DataFrame(all_results)


def regenerate(target: str = "pilot", data_dir: str = "data", durations: list[str] | None = None, models: list[str] | None = None, workers: int | None = None):
    results_dir = Path(__file__).parent.parent / "results" / target

    if durations is None:
        durations = DURATIONS

    # Load samples
    dataset = load_dataset(data_dir, duration="3s")
    if target == "pilot":
        samples = sample_pilot(dataset, n_per_category=10)
    else:
        samples = dataset
    samples_by_id = {s.image_id: s for s in samples}

    for duration in durations:
        print(f"\n{'='*60}")
        print(f"Duration: {duration}")
        print(f"{'='*60}")

        # Recompute metrics from JSONs
        df = recompute_metrics(results_dir, samples_by_id, duration, data_dir, models=models, workers=workers)
        if df.empty:
            print("  No results found.")
            continue

        print(f"  Computed {len(df)} results")

        # Output dir for this duration
        dur_dir = results_dir / duration
        dur_dir.mkdir(parents=True, exist_ok=True)

        # Raw results
        raw_dir = dur_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for model_name, model_df in df.groupby("model"):
            model_df.to_csv(raw_dir / f"{model_name}.csv", index=False)

        # Per-image stats
        image_avg = df.groupby(["model", "category", "image_id"])[METRICS].agg(["mean", "min", "max"])
        image_avg.columns = [f"{m}_{s}" for m, s in image_avg.columns]
        image_avg = image_avg.reset_index()
        image_avg.to_csv(dur_dir / "per_image_avg.csv", index=False)

        # Summary by model
        mean_cols = [f"{m}_mean" for m in METRICS]
        model_summary = image_avg.groupby("model")[mean_cols].agg(["mean", "std"])
        model_summary.columns = [f"{m}_{s}" for m, s in model_summary.columns]
        model_summary = model_summary.reset_index()
        model_summary.to_csv(dur_dir / "summary_by_model.csv", index=False)

        # Summary by model x category
        cat_summary = image_avg.groupby(["model", "category"])[mean_cols].agg(["mean", "std"])
        cat_summary.columns = [f"{m}_{s}" for m, s in cat_summary.columns]
        cat_summary = cat_summary.reset_index()
        cat_summary.to_csv(dur_dir / "summary_by_category.csv", index=False)

        # Best/worst images
        img_dir = dur_dir / "images"
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
                    continue

                points = [GazePoint(**p) for p in json.loads(pred_path.read_text())]

                heatmap_path = Path(data_dir) / "saliency_maps" / f"heatmaps_{duration}" / sample.image_path.name
                if not heatmap_path.exists():
                    continue

                gt = load_ground_truth(heatmap_path)
                h, w = gt.shape
                pred = generate_saliency_map(points, width=w, height=h)

                model_img_dir = img_dir / model
                model_img_dir.mkdir(parents=True, exist_ok=True)

                cc = row["CC_mean"]
                out_path = model_img_dir / f"{category}_{label}_cc{cc:.3f}.png"
                plot_saliency_comparison(
                    sample.image_path, pred, gt, out_path, model_name=model
                )

        # Print
        print(f"\n  SUMMARY BY MODEL ({duration})")
        print(model_summary.to_string(index=False))

    print(f"\nDone. Results saved to {results_dir}/{{1s,3s,7s}}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate metrics and images from prediction JSONs")
    parser.add_argument("--target", default="pilot", choices=["pilot", "full"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--durations", nargs="+", default=None,
                        choices=["1s", "3s", "7s"],
                        help="Durations to compute. Default: all")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to regenerate. Others loaded from cache. Default: all")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel worker processes. Default: cpu_count - 1")
    args = parser.parse_args()

    regenerate(target=args.target, data_dir=args.data_dir, durations=args.durations, models=args.models, workers=args.workers)
