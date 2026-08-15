"""Audit the coordinate-scale inference heuristic on every stored raw response.

The corrected results rest on inferring, per response and per axis, which scale
a model answered on. This script quantifies how often that inference is
unambiguous, so the heuristic's reliability can be reported rather than assumed.

For each axis of each response we record:
  - the inferred divisor (1 = already normalized, else 10/100/1000/image dim)
  - whether the rescaled values land inside [0, 1] (a necessary condition for a
    correct inference: a wrong divisor generally leaves values out of range or
    collapses them into a narrow band near zero)
  - whether any *other* candidate divisor would also have been plausible, which
    is the operative notion of ambiguity: a decision is safe when exactly one
    candidate both fits the values into [0, 1] and leaves them spanning a
    reasonable fraction of the image rather than collapsed near zero
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

from src.vlm.openrouter import _axis_divisor, _rescale_value
from src.data_loader import load_dataset

MODELS = ["gemini-3.1-flash-lite", "gemini-3.1-pro", "ui-tars-1.5"]
ROOT = Path(__file__).parent.parent
# A candidate divisor is plausible when the rescaled values fit inside [0, 1]
# and still span at least SPAN_MIN of that range; anything smaller means the
# divisor is too large and has collapsed the points toward the origin.
SPAN_MIN = 0.30


def _one(task):
    raw_path, image_path, model, image_id, run = task
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
    out = {"model": model, "image_id": image_id, "run": run, "n_points": len(pts)}
    for axis, vals, dim in (("x", [float(p["x"]) for p in pts], w),
                            ("y", [float(p["y"]) for p in pts], h)):
        div = _axis_divisor(vals, dim, model)
        scaled = [_rescale_value(v, div) for v in vals]
        m = max(abs(v) for v in vals)

        # How many candidate divisors would also have been plausible?
        plausible = [d for d in (1.0, 10.0, 100.0, 1000.0, float(dim))
                     if SPAN_MIN <= m / d <= 1.0]
        # Worst disagreement between plausible candidates, in normalized units.
        disagree = 0.0
        if len(plausible) > 1:
            disagree = max(abs(m / a - m / b)
                           for i, a in enumerate(plausible) for b in plausible[i + 1:])

        out[f"{axis}_divisor"] = div
        out[f"{axis}_rescaled_ok"] = float(np.mean([0.0 <= v <= 1.0 for v in scaled]))
        out[f"{axis}_spread"] = float(np.std(scaled))
        out[f"{axis}_n_plausible"] = len(plausible)
        out[f"{axis}_disagreement"] = float(disagree)
    out["mixed_axes"] = out["x_divisor"] != out["y_divisor"]
    return out


def main():
    samples = {s.image_id: s for s in load_dataset("data", duration="7s")}
    tasks = []
    for model in MODELS:
        for run_dir in sorted((ROOT / "results/full/raw_responses" / model).glob("run*")):
            run = int(run_dir.name.replace("run", ""))
            for f in sorted(run_dir.glob("*.txt")):
                s = samples.get(f.stem)
                if s:
                    tasks.append((str(f), str(s.image_path), model, f.stem, run))

    workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"auditing {len(tasks)} responses, {workers} workers")
    rows, t0, done = [], time.time(), 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_one, t) for t in tasks]):
            r = fut.result()
            if r:
                rows.append(r)
            done += 1
            if done % 3000 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} ({done/(time.time()-t0):.0f}/s)", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "results/full/scale_audit"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "raw.csv", index=False)

    print("\n=== scale-inference audit ===")
    print(f"{'model':24}{'resp':>7}{'norm only%':>12}{'rescaled%':>11}"
          f"{'mixed axes%':>13}{'in-range%':>11}{'ambiguous%':>12}{'med disagree':>14}")
    summ = []
    for m, g in df.groupby("model"):
        norm_only = float(((g.x_divisor == 1) & (g.y_divisor == 1)).mean())
        rescaled = 1 - norm_only
        in_range = float(((g.x_rescaled_ok + g.y_rescaled_ok) / 2).mean())
        amb = (g.x_n_plausible > 1) | (g.y_n_plausible > 1)
        ambiguous = float(amb.mean())
        dis = pd.concat([g.x_disagreement, g.y_disagreement])
        med_dis = float(dis[dis > 0].median()) if (dis > 0).any() else 0.0
        summ.append({"model": m, "n_responses": len(g), "norm_only": norm_only,
                     "rescaled": rescaled, "mixed_axes": float(g.mixed_axes.mean()),
                     "in_range_after": in_range, "ambiguous": ambiguous,
                     "median_disagreement": med_dis})
        print(f"{m:24}{len(g):>7}{100*norm_only:>11.1f}%{100*rescaled:>10.1f}%"
              f"{100*g.mixed_axes.mean():>12.1f}%{100*in_range:>10.1f}%"
              f"{100*ambiguous:>11.1f}%{med_dis:>14.3f}")
    pd.DataFrame(summ).to_csv(out / "summary.csv", index=False)

    print("\ndivisors chosen (share of axes):")
    d = pd.concat([df.x_divisor, df.y_divisor]).value_counts(normalize=True)
    for k, v in d.items():
        label = {1.0: "1 (already normalized)", 10.0: "10", 100.0: "100 (percent)",
                 1000.0: "1000 (grid)"}.get(k, f"{k:.0f} (image dimension)")
        print(f"  {label:32}{100*v:>7.2f}%")
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
