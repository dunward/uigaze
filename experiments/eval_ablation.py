"""Evaluate ablation predictions against GT at all durations.

Key question: does the temporal-cue variant (first1s) shift predictions toward
early-viewing ground truth, and does relaxing the point-count constraint
(free_n) change alignment?
"""

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
from experiments.baselines import CANONICAL, resize_map
from src.data_loader import load_dataset

DURATIONS = ["1s", "3s", "7s"]
ROOT = Path(__file__).parent.parent


def _get_prior(npz_path: str):
    data = np.load(npz_path)
    return data["sum_global"], int(data["n_total"])


def _one(task):
    pred_file, model, variant, image_id, category, heatmaps, priors = task
    points = [GazePoint(**p) for p in json.loads(Path(pred_file).read_text())]
    if not points:
        return []
    rows = []
    for dur in DURATIONS:
        arr = np.array(Image.open(heatmaps[dur]).convert("L"), dtype=np.float64)
        gt = arr / arr.max() if arr.max() > 0 else arr
        h, w = gt.shape
        pred = generate_saliency_map(points, width=w, height=h)
        sum_global, n_total = _get_prior(priors[dur])
        loo = (sum_global - resize_map(gt, *CANONICAL)) / (n_total - 1)
        prior = resize_map(loo / loo.max(), h, w)
        m = evaluate_all(pred, gt, prior=prior)
        rows.append({"model": model, "variant": variant, "image_id": image_id,
                     "category": category, "duration": dur, "n_points": len(points), **m})
    return rows


def main():
    pred_base = ROOT / "results" / "ablation" / "predictions"
    out_dir = ROOT / "results" / "ablation"
    samples = {s.image_id: s for s in load_dataset("data", duration="7s")}
    heatdirs = {d: ROOT / "data" / "saliency_maps" / f"heatmaps_{d}" for d in DURATIONS}
    priors = {d: str(ROOT / "results" / "full" / "baselines" / d / "gt_priors.npz")
              for d in DURATIONS}

    tasks = []
    for model_dir in sorted(pred_base.iterdir()):
        if not model_dir.is_dir():
            continue
        for var_dir in sorted(model_dir.iterdir()):
            for f in sorted(var_dir.glob("*_run01.json")):
                image_id = f.stem.rsplit("_run", 1)[0]
                s = samples.get(image_id)
                if not s:
                    continue
                heatmaps = {d: heatdirs[d] / s.image_path.name for d in DURATIONS}
                if not all(p.exists() for p in heatmaps.values()):
                    continue
                tasks.append((str(f), model_dir.name, var_dir.name, image_id,
                              s.category, heatmaps, priors))

    workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"Evaluating {len(tasks)} predictions x {len(DURATIONS)} durations, {workers} workers")
    rows = []
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one, t) for t in tasks]
        for fut in as_completed(futures):
            rows.extend(fut.result())
            done += 1
            if done % 200 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} ({done / (time.time() - t0):.1f}/s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "raw_metrics.csv", index=False)
    summary = df.groupby(["model", "variant", "duration"])[
        ["CC", "CC_partial", "SIM", "KL", "n_points"]].mean().round(3).reset_index()
    summary.to_csv(out_dir / "summary.csv", index=False)

    print("\nCC by variant x duration (mean over models and images):")
    print(df.groupby(["variant", "duration"])["CC"].mean().round(3).unstack())
    print("\nCC_partial by variant x duration:")
    print(df.groupby(["variant", "duration"])["CC_partial"].mean().round(3).unstack())
    print("\nPer model at 1s (does first1s help early alignment?):")
    print(df[df.duration == "1s"].groupby(["model", "variant"])["CC"].mean().round(3).unstack())


if __name__ == "__main__":
    main()
