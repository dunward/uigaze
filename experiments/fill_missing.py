"""Fill missing results for a model by re-running only failed pairs."""

import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_dataset, sample_pilot
from src.vlm.openrouter import predict_saliency_async, GazePoint
from src.saliency.generator import generate_saliency_map
from src.metrics.evaluator import evaluate_all

METRICS = ["CC", "SIM", "KL"]
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "pilot"


def load_ground_truth(heatmap_path: Path) -> np.ndarray:
    img = Image.open(heatmap_path).convert("L")
    arr = np.array(img, dtype=np.float64)
    if arr.max() > 0:
        arr /= arr.max()
    return arr


async def fill_missing(model: str, n_runs: int = 10, concurrency: int = 10, data_dir: str = "data"):
    raw_csv = OUTPUT_DIR / "raw" / f"{model}.csv"
    pred_dir = OUTPUT_DIR / "predictions" / model
    pred_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(data_dir, duration="3s")
    pilot = sample_pilot(dataset, n_per_category=10)
    samples_by_id = {s.image_id: s for s in pilot}
    all_image_ids = set(s.image_id for s in pilot)

    # Find missing pairs
    if raw_csv.exists():
        df = pd.read_csv(raw_csv)
        existing = set(zip(df["image_id"], df["run"]))
    else:
        df = pd.DataFrame()
        existing = set()

    missing = []
    for run in range(1, n_runs + 1):
        for img_id in all_image_ids:
            if (img_id, run) not in existing:
                missing.append((img_id, run))

    if not missing:
        print("Nothing to fill!")
        return

    print(f"Missing: {len(missing)} pairs")
    print(f"Current: {len(df)}/{len(all_image_ids) * n_runs}")
    print(f"Concurrency: {concurrency}")

    semaphore = asyncio.Semaphore(concurrency)
    results = []
    errors = 0

    async def process_one(img_id, run, idx):
        nonlocal errors
        async with semaphore:
            sample = samples_by_id[img_id]
            print(f"  [{idx+1}/{len(missing)}] {img_id} run {run}")
            try:
                points = await predict_saliency_async(sample.image_path, model=model)

                # Save prediction JSON
                pred_path = pred_dir / f"{img_id}_run{run:02d}.json"
                pred_path.write_text(json.dumps([asdict(p) for p in points], indent=2))

                gt = load_ground_truth(sample.heatmap_path)
                h, w = gt.shape
                pred = generate_saliency_map(points, width=w, height=h)
                metrics = evaluate_all(pred, gt)

                results.append({
                    "model": model,
                    "run": run,
                    "image_id": img_id,
                    "category": sample.category,
                    "n_points": len(points),
                    **metrics,
                })
            except Exception as e:
                print(f"    ERROR: {e}")
                errors += 1

    tasks = [process_one(img_id, run, i) for i, (img_id, run) in enumerate(missing)]
    await asyncio.gather(*tasks)

    # Append to CSV
    if results:
        fill_df = pd.DataFrame(results)
        if raw_csv.exists():
            fill_df.to_csv(raw_csv, mode="a", header=False, index=False)
        else:
            fill_df.to_csv(raw_csv, index=False)

    total = len(df) + len(results)
    print(f"\nDone: {len(results)} filled, {errors} errors")
    print(f"Total now: {total}/{len(all_image_ids) * n_runs}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fill missing pilot results")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    asyncio.run(fill_missing(
        model=args.model,
        n_runs=args.n_runs,
        concurrency=args.concurrency,
        data_dir=args.data_dir,
    ))
