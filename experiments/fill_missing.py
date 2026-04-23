"""Fill missing predictions for a model by re-running only failed pairs."""

import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_dataset, sample_pilot
from src.vlm.openrouter import predict_saliency_async

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "pilot"


async def fill_missing(model: str, n_runs: int = 10, concurrency: int = 10, data_dir: str = "data"):
    pred_dir = OUTPUT_DIR / "predictions" / model
    pred_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(data_dir, duration="3s")
    pilot = sample_pilot(dataset, n_per_category=10)
    samples_by_id = {s.image_id: s for s in pilot}
    all_image_ids = set(s.image_id for s in pilot)

    # Find missing pairs (check prediction JSON existence)
    existing = set()
    for run in range(1, n_runs + 1):
        for img_id in all_image_ids:
            if (pred_dir / f"{img_id}_run{run:02d}.json").exists():
                existing.add((img_id, run))

    missing = []
    for run in range(1, n_runs + 1):
        for img_id in all_image_ids:
            if (img_id, run) not in existing:
                missing.append((img_id, run))

    if not missing:
        print("Nothing to fill!")
        return

    print(f"Missing: {len(missing)} pairs")
    print(f"Current: {len(existing)}/{len(all_image_ids) * n_runs}")
    print(f"Concurrency: {concurrency}")

    semaphore = asyncio.Semaphore(concurrency)
    filled = 0
    errors = 0

    async def process_one(img_id, run, idx):
        nonlocal filled, errors
        async with semaphore:
            sample = samples_by_id[img_id]
            print(f"  [{idx+1}/{len(missing)}] {img_id} run {run}")
            try:
                points = await predict_saliency_async(sample.image_path, model=model)

                pred_path = pred_dir / f"{img_id}_run{run:02d}.json"
                pred_path.write_text(json.dumps([asdict(p) for p in points], indent=2))
                filled += 1
            except Exception as e:
                print(f"    ERROR [{type(e).__name__}]: {e}")
                errors += 1

    tasks = [process_one(img_id, run, i) for i, (img_id, run) in enumerate(missing)]
    await asyncio.gather(*tasks)

    print(f"\nDone: {filled} filled, {errors} errors")
    print(f"Total now: {len(existing) + filled}/{len(all_image_ids) * n_runs}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fill missing pilot predictions")
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
