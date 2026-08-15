"""Inter-observer consistency (IOC) from raw UEyes eyetracker logs.

Provides the human upper bound for saliency prediction: how well does one
half of the participants who viewed an image predict the other half?

Steps:
  1. index    - parse all *_fixations.csv logs into one cached table.
  2. validate - reconstruct full-participant 7s maps for a sample and compare
                against the provided UEyes heatmaps under two coordinate
                mappings (direct screen-normalized vs letterbox-corrected).
  3. run      - split-half IOC per image x duration (CC/SIM/KL between the
                two half-maps, plus Spearman-Brown corrected CC).
"""

import argparse
import os
import pickle
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.evaluator import compute_cc, compute_sim, compute_kl, compute_cc_partial
from src.data_loader import load_dataset
from experiments.baselines import CANONICAL, resize_map

SCREEN_W, SCREEN_H = 1920, 1200
SIGMA = 40.0
N_SPLITS = 5
SEED = 42
DURATIONS = {"1s": 1.0, "3s": 3.0, "7s": 7.5}  # 7s window includes everything


def build_index(log_dir: Path, cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    frames = []
    files = sorted(log_dir.glob("*_fixations.csv"))
    for i, f in enumerate(files):
        participant = f.name.split("_")[1].lower()
        df = pd.read_csv(f, usecols=lambda c: c.split("(")[0] in
                         ("MEDIA_NAME", "FPOGX", "FPOGY", "FPOGS", "FPOGD", "FPOGV"))
        df.columns = [c.split("(")[0] for c in df.columns]
        df = df[df["FPOGV"] == 1].drop(columns=["FPOGV"])
        df["participant"] = participant
        frames.append(df)
        if (i + 1) % 100 == 0:
            print(f"  indexed {i + 1}/{len(files)} logs", flush=True)
    idx = pd.concat(frames, ignore_index=True)
    cache.write_bytes(pickle.dumps(idx))
    print(f"  index: {len(idx)} fixations, {idx['participant'].nunique()} participants, "
          f"{idx['MEDIA_NAME'].nunique()} media")
    return idx


def to_image_coords(fx: np.ndarray, fy: np.ndarray, w: int, h: int, letterbox: bool):
    """Map screen-normalized FPOG coords to image-normalized coords."""
    if not letterbox:
        return fx, fy
    scale = min(SCREEN_W / w, SCREEN_H / h)
    disp_w, disp_h = w * scale, h * scale
    off_x, off_y = (SCREEN_W - disp_w) / 2, (SCREEN_H - disp_h) / 2
    x = (fx * SCREEN_W - off_x) / disp_w
    y = (fy * SCREEN_H - off_y) / disp_h
    return x, y


def splat_map(fx, fy, weights, w: int, h: int) -> np.ndarray:
    canvas = np.zeros((h, w))
    px = np.clip(np.round(fx * (w - 1)).astype(int), 0, w - 1)
    py = np.clip(np.round(fy * (h - 1)).astype(int), 0, h - 1)
    np.add.at(canvas, (py, px), weights)
    sal = gaussian_filter(canvas, sigma=SIGMA)
    m = sal.max()
    return sal / m if m > 0 else sal


def load_gt(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"), dtype=np.float64)
    return arr / arr.max() if arr.max() > 0 else arr


def validate(index: pd.DataFrame, samples_by_name: dict, n: int = 30):
    """Compare full-participant reconstructions against provided heatmaps."""
    media_counts = index.groupby("MEDIA_NAME")["participant"].nunique()
    names = [m for m in media_counts.index if m in samples_by_name][:n]
    scores = {"direct": [], "letterbox": []}
    for name in names:
        s = samples_by_name[name]
        gt = load_gt(s.heatmap_path)
        h, w = gt.shape
        g = index[index["MEDIA_NAME"] == name]
        for mode, letterbox in (("direct", False), ("letterbox", True)):
            x, y = to_image_coords(g["FPOGX"].values, g["FPOGY"].values, w, h, letterbox)
            keep = (x >= -0.02) & (x <= 1.02) & (y >= -0.02) & (y <= 1.02)
            rec = splat_map(np.clip(x[keep], 0, 1), np.clip(y[keep], 0, 1),
                            g["FPOGD"].values[keep], w, h)
            scores[mode].append(compute_cc(rec, gt))
    for mode, vals in scores.items():
        print(f"  {mode}: mean CC vs provided 7s heatmap = {np.mean(vals):.3f} "
              f"(min {np.min(vals):.3f}, n={len(vals)})")
    best = max(scores, key=lambda m: np.mean(scores[m]))
    print(f"  -> chosen mapping: {best}")
    return best == "letterbox", {m: float(np.mean(v)) for m, v in scores.items()}


_PRIORS: dict[str, dict] = {}


def _prior_for(npz_path: str, gt: np.ndarray, h: int, w: int) -> np.ndarray:
    """Leave-one-out dataset prior resized to this image, as in baselines.py."""
    if npz_path not in _PRIORS:
        d = np.load(npz_path)
        _PRIORS[npz_path] = {"sum": d["sum_global"], "n": int(d["n_total"])}
    p = _PRIORS[npz_path]
    loo = (p["sum"] - resize_map(gt, *CANONICAL)) / (p["n"] - 1)
    return resize_map(loo / loo.max(), h, w)


def _ioc_one(task):
    name, heatmap_path, category, image_id, grp_pickle, letterbox, priors = task
    g = pickle.loads(grp_pickle)
    gt_arr = np.array(Image.open(heatmap_path).convert("L"), dtype=np.float64)
    gt_norm = gt_arr / gt_arr.max() if gt_arr.max() > 0 else gt_arr
    h, w = gt_arr.shape
    participants = sorted(g["participant"].unique())
    if len(participants) < 4:
        return []
    x, y = to_image_coords(g["FPOGX"].values, g["FPOGY"].values, w, h, letterbox)
    keep = (x >= -0.02) & (x <= 1.02) & (y >= -0.02) & (y <= 1.02)
    g = g.loc[keep].assign(ix=np.clip(x[keep], 0, 1), iy=np.clip(y[keep], 0, 1))

    # zlib.crc32 rather than hash(): Python randomizes str hashing per process,
    # which would make the split assignment irreproducible across runs.
    rng = np.random.default_rng(SEED + zlib.crc32(image_id.encode()) % 10_000)
    rows = []
    for dur_label, window in DURATIONS.items():
        gd = g[g["FPOGS"] < window]
        for split in range(N_SPLITS):
            perm = rng.permutation(participants)
            half_a = set(perm[: len(perm) // 2])
            ga, gb = gd[gd["participant"].isin(half_a)], gd[~gd["participant"].isin(half_a)]
            if ga.empty or gb.empty:
                continue
            ma = splat_map(ga["ix"].values, ga["iy"].values, ga["FPOGD"].values, w, h)
            mb = splat_map(gb["ix"].values, gb["iy"].values, gb["FPOGD"].values, w, h)
            cc = compute_cc(ma, mb)
            row = {
                "image_id": image_id, "category": category, "duration": dur_label,
                "split": split, "n_participants": len(participants),
                "CC": cc, "CC_sb": 2 * cc / (1 + cc) if cc > -1 else np.nan,
                "SIM": compute_sim(ma, mb), "KL": compute_kl(ma, mb),
            }
            # Ceiling for the primary metric: agreement between two halves of the
            # participants after removing what both share with the dataset prior.
            prior_path = priors.get(dur_label)
            if prior_path:
                prior = _prior_for(prior_path, gt_norm, h, w)
                row["CC_partial"] = compute_cc_partial(ma, mb, prior)
            rows.append(row)
    return rows


def run_ioc(index: pd.DataFrame, samples_by_name: dict, out_dir: Path,
            letterbox: bool, workers: int | None):
    # Per-duration leave-one-out priors, reused from the baseline computation.
    base = Path(__file__).parent.parent / "results" / "full" / "baselines"
    priors = {d: str(base / d / "gt_priors.npz") for d in DURATIONS
              if (base / d / "gt_priors.npz").exists()}
    if len(priors) < len(DURATIONS):
        print(f"  WARNING: priors found for {sorted(priors)} only; "
              "CC_partial skipped elsewhere. Run experiments/baselines.py first.")
    tasks = []
    for name, g in index.groupby("MEDIA_NAME"):
        s = samples_by_name.get(name)
        if s is None:
            continue
        tasks.append((name, str(s.heatmap_path), s.category, s.image_id,
                      pickle.dumps(g[["participant", "FPOGX", "FPOGY", "FPOGS", "FPOGD"]]),
                      letterbox, priors))
    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    print(f"  IOC on {len(tasks)} images x {len(DURATIONS)} durations x {N_SPLITS} splits, "
          f"{workers} workers")
    all_rows = []
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_ioc_one, t) for t in tasks]
        for fut in as_completed(futures):
            all_rows.extend(fut.result())
            done += 1
            if done % 200 == 0 or done == len(tasks):
                rate = done / (time.time() - t0)
                print(f"    {done}/{len(tasks)} ({rate:.1f} img/s, "
                      f"eta {(len(tasks) - done) / rate:.0f}s)", flush=True)

    df = pd.DataFrame(all_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "ioc_raw.csv", index=False)
    metrics = [c for c in ["CC", "CC_sb", "CC_partial", "SIM", "KL"] if c in df.columns]
    per_image = df.groupby(["duration", "category", "image_id"])[metrics].mean().reset_index()
    per_image.to_csv(out_dir / "ioc_per_image.csv", index=False)
    summary = per_image.groupby("duration")[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.reset_index().to_csv(out_dir / "ioc_summary.csv", index=False)
    cat = per_image.groupby(["duration", "category"])[metrics].mean().reset_index()
    cat.to_csv(out_dir / "ioc_by_category.csv", index=False)
    print("\n  IOC SUMMARY (split-half, mean over images)")
    print(summary.round(3).to_string())
    print(f"\n  mean participants/image: {df.groupby('image_id')['n_participants'].first().mean():.1f}")


def main():
    parser = argparse.ArgumentParser(description="Inter-observer consistency from UEyes logs")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(__file__).parent.parent / "results" / "full" / "human_ceiling"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building fixation index...")
    index = build_index(Path(args.data_dir) / "eyetracker_logs", out_dir / "fixation_index.pkl")

    dataset = load_dataset(args.data_dir, duration="7s")
    samples_by_name = {s.image_path.name: s for s in dataset}

    print("\nValidating coordinate mapping (n=30 images, all participants vs provided 7s map)...")
    letterbox, val_scores = validate(index, samples_by_name)
    pd.DataFrame([{"letterbox_chosen": letterbox, **val_scores}]).to_csv(
        out_dir / "validation.csv", index=False)

    if args.validate_only:
        return
    run_ioc(index, samples_by_name, out_dir, letterbox, args.workers)


if __name__ == "__main__":
    main()
