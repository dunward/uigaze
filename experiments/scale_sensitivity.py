"""Sensitivity of the corrected results to the scale-inference decision.

Where more than one divisor is plausible for an axis, our rule resolves the tie
using each model's documented coordinate convention (Gemini: 0-1000 grid;
UI-TARS: absolute pixels). This script recomputes every corrected score under
the *opposite* resolution, to establish whether the reported gains depend on
that tie-break or survive it.
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

from src.vlm.openrouter import GazePoint, _axis_divisor, _rescale_value
from src.saliency.generator import generate_saliency_map
from src.metrics.evaluator import compute_cc
from src.data_loader import load_dataset

MODELS = ["gemini-3.1-flash-lite", "gemini-3.1-pro", "ui-tars-1.5"]
SPAN_MIN = 0.30
ROOT = Path(__file__).parent.parent


def alt_divisor(vals, dim, model):
    """The divisor our rule did NOT choose, among the plausible candidates."""
    chosen = _axis_divisor(vals, dim, model)
    m = max(abs(v) for v in vals)
    plausible = [d for d in (1.0, 10.0, 100.0, 1000.0, float(dim))
                 if SPAN_MIN <= m / d <= 1.0]
    others = [d for d in plausible if abs(d - chosen) > 1e-9]
    if not others:
        return chosen
    # the candidate whose result differs most from ours: the worst case
    return max(others, key=lambda d: abs(m / d - m / chosen))


def _one(task):
    raw_path, heatmap_path, image_path, model, image_id = task
    try:
        parsed = json.loads(Path(raw_path).read_text().strip())
    except Exception:
        return None
    if isinstance(parsed, dict):
        parsed = next((v for v in parsed.values() if isinstance(v, list)), [])
    pts = [p for p in parsed
           if isinstance(p, dict) and all(k in p for k in ("x", "y", "intensity"))]
    if not pts:
        return None

    w, h = Image.open(image_path).size
    xs = [float(p["x"]) for p in pts]
    ys = [float(p["y"]) for p in pts]
    inten = [float(p["intensity"]) for p in pts]
    di = _axis_divisor(inten, None, None)

    arr = np.array(Image.open(heatmap_path).convert("L"), dtype=np.float64)
    gt = arr / arr.max() if arr.max() > 0 else arr
    gh, gw = gt.shape

    out = {"model": model, "image_id": image_id}
    for tag, fx, fy in (("chosen", _axis_divisor, _axis_divisor),
                        ("alt", alt_divisor, alt_divisor)):
        dx, dy = fx(xs, w, model), fy(ys, h, model)
        gp = [GazePoint(x=max(0.0, min(1.0, _rescale_value(x, dx))),
                        y=max(0.0, min(1.0, _rescale_value(y, dy))),
                        intensity=max(0.0, min(1.0, _rescale_value(i, di))))
              for x, y, i in zip(xs, ys, inten)]
        out[f"CC_{tag}"] = compute_cc(generate_saliency_map(gp, width=gw, height=gh), gt)
    return out


def main():
    samples = {s.image_id: s for s in load_dataset("data", duration="7s")}
    heat = ROOT / "data" / "saliency_maps" / "heatmaps_7s"
    tasks = []
    for model in MODELS:
        base = ROOT / "results/full/raw_responses" / model
        for run_dir in sorted(base.glob("run*")):
            for f in sorted(run_dir.glob("*.txt")):
                s = samples.get(f.stem)
                if s and (heat / s.image_path.name).exists():
                    tasks.append((str(f), str(heat / s.image_path.name),
                                  str(s.image_path), model, f.stem))

    workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"{len(tasks)} responses (all runs), {workers} workers")
    rows, t0, done = [], time.time(), 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_one, t) for t in tasks]):
            r = fut.result()
            if r:
                rows.append(r)
            done += 1
            if done % 1000 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} ({done/(time.time()-t0):.0f}/s)", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "results/full/scale_audit"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "sensitivity.csv", index=False)

    df = df.groupby(["model", "image_id"])[["CC_chosen", "CC_alt"]].mean().reset_index()
    print("\n=== CC at 7s under our tie-break vs the opposite one (3-run mean) ===")
    print(f"{'model':24}{'chosen':>9}{'alternative':>13}{'delta':>9}{'n':>7}")
    for m, g in df.groupby("model"):
        print(f"{m:24}{g.CC_chosen.mean():>9.3f}{g.CC_alt.mean():>13.3f}"
              f"{g.CC_alt.mean()-g.CC_chosen.mean():>+9.3f}{len(g):>7}")
    print(f"\nSaved to {out}/sensitivity.csv")


if __name__ == "__main__":
    main()
