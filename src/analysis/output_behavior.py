"""Analyze VLM output behavior from saved prediction JSONs.

Quantifies output-format pathologies (duplicate coordinates, grid snapping,
edge clamping, low dispersion) that can depress saliency metrics for reasons
unrelated to attention modeling.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

EXPECTED_RUNS = 3


def analyze_run(points: list[dict]) -> dict:
    n = len(points)
    xs = np.array([p["x"] for p in points])
    ys = np.array([p["y"] for p in points])
    intensities = np.array([p["intensity"] for p in points])

    coords = list(zip(xs.tolist(), ys.tolist()))
    coord_counts = Counter(coords)
    n_unique = len(coord_counts)
    max_dup = max(coord_counts.values())

    def frac_on_grid(v, step):
        return float(np.mean(np.abs(np.round(v / step) * step - v) < 1e-9))

    return {
        "n_points": n,
        "n_unique_coords": n_unique,
        "unique_ratio": n_unique / n,
        "max_duplicate_count": max_dup,
        "n_unique_x": len(set(xs.tolist())),
        "n_unique_y": len(set(ys.tolist())),
        "frac_grid_005": frac_on_grid(xs, 0.05) * frac_on_grid(ys, 0.05),
        "frac_edge": float(np.mean((xs <= 0.0) | (xs >= 1.0) | (ys <= 0.0) | (ys >= 1.0))),
        "std_x": float(xs.std()),
        "std_y": float(ys.std()),
        "n_unique_intensity": len(set(intensities.tolist())),
    }


def run(results_dir: Path, out_dir: Path, n_images_expected: int):
    pred_base = results_dir / "predictions"
    rows = []
    coverage = []

    for model_dir in sorted(pred_base.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        files = sorted(model_dir.glob("*_run*.json"))
        n_empty = 0
        for f in files:
            points = json.loads(f.read_text())
            image_id, run_no = f.stem.rsplit("_run", 1)
            if not points:
                n_empty += 1
                continue
            rows.append({"model": model, "image_id": image_id, "run": int(run_no),
                         **analyze_run(points)})
        expected = n_images_expected * EXPECTED_RUNS
        coverage.append({
            "model": model,
            "files_saved": len(files),
            "files_expected": expected,
            "empty_files": n_empty,
            "valid_response_rate": (len(files) - n_empty) / expected,
        })
        print(f"  {model}: {len(files)} files, {n_empty} empty", flush=True)

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "per_run.csv", index=False)

    agg_cols = [c for c in df.columns if c not in ("model", "image_id", "run")]
    summary = df.groupby("model")[agg_cols].mean().round(4)
    cov_df = pd.DataFrame(coverage).set_index("model")
    summary = summary.join(cov_df[["valid_response_rate", "empty_files"]])
    summary.reset_index().to_csv(out_dir / "summary_by_model.csv", index=False)

    print("\nOUTPUT BEHAVIOR SUMMARY BY MODEL")
    cols = ["n_points", "n_unique_coords", "unique_ratio", "max_duplicate_count",
            "n_unique_x", "n_unique_y", "frac_grid_005", "frac_edge",
            "std_x", "std_y", "valid_response_rate"]
    print(summary[cols].to_string())
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze VLM output behavior")
    parser.add_argument("--target", default="full", choices=["pilot", "full"])
    parser.add_argument("--n-images", type=int, default=1980)
    args = parser.parse_args()

    base = Path(__file__).parent.parent.parent / "results" / args.target
    run(base, base / "output_behavior", args.n_images)
