"""Fill missing predictions for a model by re-running only failed pairs."""

import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_dataset, sample_pilot
from src.vlm.openrouter import predict_saliency_async


def _resolve_samples(target: str, data_dir: str):
    dataset = load_dataset(data_dir, duration="3s")
    if target == "pilot":
        return sample_pilot(dataset, n_per_category=10)
    return dataset


async def fill_missing(
    target: str,
    model: str,
    n_runs: int,
    concurrency: int = 10,
    data_dir: str = "data",
):
    output_dir = Path(__file__).parent.parent / "results" / target
    pred_dir = output_dir / "predictions" / model
    pred_dir.mkdir(parents=True, exist_ok=True)

    samples = _resolve_samples(target, data_dir)
    samples_by_id = {s.image_id: s for s in samples}
    all_image_ids = set(samples_by_id.keys())

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

    total = len(all_image_ids) * n_runs
    if not missing:
        print(f"Nothing to fill! ({len(existing)}/{total})")
        return

    print(f"Target: {target} | Model: {model}")
    print(f"Missing: {len(missing)} pairs")
    print(f"Current: {len(existing)}/{total}")
    print(f"Concurrency: {concurrency}")

    semaphore = asyncio.Semaphore(concurrency)
    filled = 0
    errors = 0

    async def process_one(img_id, run, idx):
        nonlocal filled, errors
        async with semaphore:
            sample = samples_by_id[img_id]
            print(f"  [{idx+1}/{len(missing)}] {img_id} run {run}", flush=True)
            try:
                points = await predict_saliency_async(sample.image_path, model=model)
                if not points:
                    print(f"    SKIP {img_id} run {run} (no predictions returned)", flush=True)
                    errors += 1
                    return

                pred_path = pred_dir / f"{img_id}_run{run:02d}.json"
                pred_path.write_text(json.dumps([asdict(p) for p in points], indent=2))
                filled += 1
            except Exception as e:
                print(f"    ERROR [{type(e).__name__}]: {e}", flush=True)
                errors += 1

    tasks = [process_one(img_id, run, i) for i, (img_id, run) in enumerate(missing)]
    await asyncio.gather(*tasks)

    print(f"\nDone: {filled} filled, {errors} errors")
    print(f"Total now: {len(existing) + filled}/{total}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fill missing predictions for pilot or full experiment")
    parser.add_argument("--target", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-runs", type=int, default=None,
                        help="Default: 10 for pilot, 3 for full")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    n_runs = args.n_runs
    if n_runs is None:
        n_runs = 10 if args.target == "pilot" else 3

    asyncio.run(fill_missing(
        target=args.target,
        model=args.model,
        n_runs=n_runs,
        concurrency=args.concurrency,
        data_dir=args.data_dir,
    ))
