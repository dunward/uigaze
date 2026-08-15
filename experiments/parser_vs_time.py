"""Decompose the observed CC change into parser effect and collection-time effect.

We hold one factor fixed at a time:

    April responses x old parser  = published v1 numbers  (raw text not retained)
    August responses x old parser = computed here
    August responses x new parser = current numbers

    parser effect (time fixed)  = Aug/new  - Aug/old
    time effect   (parser fixed) = Aug/old - Apr/old

The old parser is reproduced verbatim from the pre-fix implementation: parse
JSON, then clamp every coordinate into [0, 1].
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

from src.vlm.openrouter import GazePoint, _parse_gaze_points
from src.saliency.generator import generate_saliency_map
from src.metrics.evaluator import compute_cc
from src.data_loader import load_dataset

MODELS = ["gemini-3.1-flash-lite", "gemini-3.1-pro", "ui-tars-1.5"]
ROOT = Path(__file__).parent.parent


def parse_old(content: str) -> list[GazePoint]:
    """The pre-fix parser: clamp out-of-range coordinates into [0, 1]."""
    parsed = json.loads(content.strip())
    if isinstance(parsed, dict):
        parsed = next((v for v in parsed.values() if isinstance(v, list)), [])
    points = []
    for p in parsed:
        if not (isinstance(p, dict) and all(k in p for k in ("x", "y", "intensity"))):
            continue
        points.append(GazePoint(
            x=max(0.0, min(1.0, float(p["x"]))),
            y=max(0.0, min(1.0, float(p["y"]))),
            intensity=max(0.0, min(1.0, float(p["intensity"]))),
        ))
    return points


def _one(task):
    raw_path, heatmap_path, image_path, model, image_id, run = task
    raw = Path(raw_path).read_text()
    arr = np.array(Image.open(heatmap_path).convert("L"), dtype=np.float64)
    gt = arr / arr.max() if arr.max() > 0 else arr
    h, w = gt.shape
    size = Image.open(image_path).size

    out = {"model": model, "image_id": image_id, "run": run}
    for tag, fn in (("old", lambda: parse_old(raw)),
                    ("new", lambda: _parse_gaze_points(raw, image_size=size, model=model))):
        try:
            pts = fn()
        except Exception:
            return None
        if not pts:
            return None
        out[f"CC_{tag}"] = compute_cc(generate_saliency_map(pts, width=w, height=h), gt)
    return out


def main(workers: int | None):
    samples = {s.image_id: s for s in load_dataset("data", duration="7s")}
    heat = ROOT / "data" / "saliency_maps" / "heatmaps_7s"

    tasks = []
    for model in MODELS:
        base = ROOT / "results" / "full" / "raw_responses" / model
        for run_dir in sorted(base.glob("run*")):
            run = int(run_dir.name.replace("run", ""))
            for f in sorted(run_dir.glob("*.txt")):
                s = samples.get(f.stem)
                if not s:
                    continue
                hp = heat / s.image_path.name
                if hp.exists():
                    tasks.append((str(f), str(hp), str(s.image_path), model, f.stem, run))

    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    print(f"{len(tasks)} raw responses x 2 parsers, {workers} workers")
    rows = []
    t0, done = [time.time()], 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_one, t) for t in tasks]):
            r = fut.result()
            if r:
                rows.append(r)
            done += 1
            if done % 2000 == 0 or done == len(tasks):
                rate = done / (time.time() - t0[0])
                print(f"  {done}/{len(tasks)} ({rate:.0f}/s, eta {(len(tasks)-done)/rate:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "full" / "parser_vs_time"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "raw.csv", index=False)

    per_image = df.groupby(["model", "image_id"])[["CC_old", "CC_new"]].mean().reset_index()
    apr = {m: pd.read_csv(ROOT / "results/full/clamped_era/summary_by_model_7s.csv")
           .set_index("model").loc[m, "CC_mean_mean"] for m in MODELS}

    print("\n=== CC at 7s: three-cell decomposition ===")
    print(f"{'model':24}{'Apr/old':>9}{'Aug/old':>9}{'Aug/new':>9}{'parser':>9}{'time':>9}")
    summary = []
    for m in MODELS:
        g = per_image[per_image.model == m]
        aug_old, aug_new = g.CC_old.mean(), g.CC_new.mean()
        parser_eff, time_eff = aug_new - aug_old, aug_old - apr[m]
        summary.append({"model": m, "apr_old": apr[m], "aug_old": aug_old, "aug_new": aug_new,
                        "parser_effect": parser_eff, "time_effect": time_eff, "n_images": len(g)})
        print(f"{m:24}{apr[m]:>9.3f}{aug_old:>9.3f}{aug_new:>9.3f}{parser_eff:>+9.3f}{time_eff:>+9.3f}")
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=None)
    main(p.parse_args().workers)
