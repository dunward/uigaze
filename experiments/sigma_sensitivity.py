"""Sigma sensitivity: recompute metrics with sigma in {20, 40, 60}px on a subset.

Appendix analysis showing how the Gaussian blur width interacts with each
model's output behavior (models with few unique coordinates are hit hardest
at small sigma, inflating KL for format rather than attention reasons).
"""

import argparse
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
from src.data_loader import load_dataset, sample_pilot

SIGMAS = [20.0, 40.0, 60.0]
DURATION = "7s"


def _one(task):
    pred_file, heatmap_path, model, image_id, category = task
    points = [GazePoint(**p) for p in json.loads(Path(pred_file).read_text())]
    if not points:
        return []
    arr = np.array(Image.open(heatmap_path).convert("L"), dtype=np.float64)
    gt = arr / arr.max() if arr.max() > 0 else arr
    h, w = gt.shape
    rows = []
    for sigma in SIGMAS:
        pred = generate_saliency_map(points, width=w, height=h, sigma=sigma)
        m = evaluate_all(pred, gt)
        rows.append({"model": model, "image_id": image_id, "category": category,
                     "sigma": sigma, **m})
    return rows


def main(n_per_category: int, workers: int | None):
    base = Path(__file__).parent.parent / "results" / "full"
    dataset = load_dataset("data", duration=DURATION)
    subset = sample_pilot(dataset, n_per_category=n_per_category, seed=42)
    by_id = {s.image_id: s for s in subset}

    tasks = []
    for model_dir in sorted((base / "predictions").iterdir()):
        if not model_dir.is_dir():
            continue
        for image_id, s in by_id.items():
            pred = model_dir / f"{image_id}_run01.json"
            if pred.exists():
                tasks.append((str(pred), str(s.heatmap_path), model_dir.name,
                              image_id, s.category))

    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    print(f"{len(tasks)} model-image pairs x {len(SIGMAS)} sigmas, {workers} workers")
    rows = []
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one, t) for t in tasks]
        for fut in as_completed(futures):
            rows.extend(fut.result())
            done += 1
            if done % 200 == 0 or done == len(tasks):
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(tasks)} ({rate:.1f}/s)", flush=True)

    df = pd.DataFrame(rows)
    out = base / "sigma_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "raw.csv", index=False)
    summary = df.groupby(["model", "sigma"])[["CC", "SIM", "KL"]].mean().round(3).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.pivot(index="model", columns="sigma", values="CC").round(3).to_string())
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-category", type=int, default=50)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    main(args.n_per_category, args.workers)
