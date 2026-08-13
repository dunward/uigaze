"""Evaluate non-VLM baselines against UEyes ground truth.

Baselines:
    uniform      - constant map (chance reference for SIM/KL)
    noise        - per-pixel uniform random noise (chance reference for CC/AUC)
    center_gXX   - isotropic Gaussian centered on the image, sigma = XX% of min(h, w)
    mean_gt      - leave-one-out mean of all GT maps (dataset center prior)
    mean_gt_cat  - leave-one-out mean of same-category GT maps

Also computes per-image GT heatmap entropy per duration, to test whether the
duration effect (CC rising 1s -> 7s) is explained by GT maps spreading out.
"""

import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.evaluator import evaluate_all
from src.data_loader import load_dataset, sample_pilot

DURATIONS = ["1s", "3s", "7s"]
METRICS = ["CC", "SIM", "KL", "NSS", "AUC"]
CENTER_SIGMA_FRACS = [0.15, 0.25, 0.35]
CANONICAL = (512, 512)  # (h, w) normalized space for mean-GT priors


def load_map(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"), dtype=np.float64)
    if arr.max() > 0:
        arr /= arr.max()
    return arr


def load_fixmap(path: Path) -> np.ndarray:
    # Some fixmaps are JPEGs with compression noise; binarize at mid-gray.
    arr = np.array(Image.open(path).convert("L"))
    return (arr > 127).astype(np.float64)


def resize_map(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    img = Image.fromarray((arr * 255).astype(np.uint8))
    out = np.array(img.resize((w, h), Image.BILINEAR), dtype=np.float64)
    if out.max() > 0:
        out /= out.max()
    return out


def center_gaussian(h: int, w: int, sigma: float) -> np.ndarray:
    y = np.arange(h) - (h - 1) / 2
    x = np.arange(w) - (w - 1) / 2
    g = np.exp(-(y[:, None] ** 2 + x[None, :] ** 2) / (2 * sigma**2))
    return g / g.max()


def gt_entropy(gt: np.ndarray) -> tuple[float, float]:
    """Shannon entropy of the GT map as a probability distribution (bits)."""
    p = gt / (gt.sum() + 1e-10)
    p = p[p > 0]
    h = float(-(p * np.log2(p)).sum())
    return h, h / np.log2(gt.size)


_PRIORS: dict[str, dict] = {}


def _get_priors(npz_path: str) -> dict:
    if npz_path not in _PRIORS:
        data = np.load(npz_path, allow_pickle=True)
        _PRIORS[npz_path] = {k: data[k] for k in data.files}
    return _PRIORS[npz_path]


def _compute_one(task):
    heatmap_path, fixmap_path, image_id, category, priors_npz = task

    gt = load_map(Path(heatmap_path))
    h, w = gt.shape
    fixmap = None
    if fixmap_path and Path(fixmap_path).exists():
        fixmap = load_fixmap(Path(fixmap_path))
        if fixmap.shape != gt.shape or not fixmap.any():
            fixmap = None

    ent, ent_norm = gt_entropy(gt)

    baselines: dict[str, np.ndarray] = {}
    baselines["uniform"] = np.full((h, w), 0.5)
    seed = int(hashlib.md5(image_id.encode()).hexdigest()[:8], 16)
    baselines["noise"] = np.random.default_rng(seed).random((h, w))
    for frac in CENTER_SIGMA_FRACS:
        baselines[f"center_g{int(frac * 100)}"] = center_gaussian(h, w, frac * min(h, w))

    priors = _get_priors(priors_npz)
    gt_canon = resize_map(gt, *CANONICAL)
    n_total = int(priors["n_total"])
    loo_global = (priors["sum_global"] - gt_canon) / (n_total - 1)
    baselines["mean_gt"] = resize_map(loo_global / loo_global.max(), h, w)
    n_cat = int(priors[f"n_{category}"])
    if n_cat > 1:
        loo_cat = (priors[f"sum_{category}"] - gt_canon) / (n_cat - 1)
        baselines["mean_gt_cat"] = resize_map(loo_cat / loo_cat.max(), h, w)

    rows = []
    for name, pred in baselines.items():
        metrics = evaluate_all(pred, gt, fixation_map=fixmap)
        rows.append({"baseline": name, "image_id": image_id, "category": category, **metrics})

    stats = {"image_id": image_id, "category": category,
             "gt_entropy_bits": ent, "gt_entropy_norm": ent_norm,
             "height": h, "width": w}
    return rows, stats


def build_priors(samples, out_path: Path):
    """Accumulate resized GT sums (global and per-category) in canonical space."""
    sums = {"global": np.zeros(CANONICAL)}
    counts = {"global": 0}
    for i, s in enumerate(samples):
        gt = load_map(s.heatmap_path)
        gt_canon = resize_map(gt, *CANONICAL)
        sums["global"] += gt_canon
        counts["global"] += 1
        if s.category not in sums:
            sums[s.category] = np.zeros(CANONICAL)
            counts[s.category] = 0
        sums[s.category] += gt_canon
        counts[s.category] += 1
        if (i + 1) % 500 == 0:
            print(f"    prior accumulation {i + 1}/{len(samples)}", flush=True)

    payload = {"sum_global": sums["global"], "n_total": counts["global"]}
    for cat in counts:
        if cat != "global":
            payload[f"sum_{cat}"] = sums[cat]
            payload[f"n_{cat}"] = counts[cat]
    np.savez(out_path, **payload)
    return payload


def save_verify_images(samples, priors_npz: str, out_dir: Path):
    """Save baseline maps for one sample per category for visual inspection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    for s in samples:
        if s.category in seen:
            continue
        seen.add(s.category)
        rows, _ = _compute_one((str(s.heatmap_path), None, s.image_id, s.category, priors_npz))
        gt = load_map(s.heatmap_path)
        h, w = gt.shape
        priors = _get_priors(priors_npz)
        maps = {
            "gt": gt,
            "center_g25": center_gaussian(h, w, 0.25 * min(h, w)),
            "mean_gt": resize_map(priors["sum_global"] / int(priors["n_total"]), h, w),
            f"mean_gt_{s.category}": resize_map(
                priors[f"sum_{s.category}"] / int(priors[f"n_{s.category}"]), h, w),
        }
        for name, arr in maps.items():
            Image.fromarray((arr * 255).astype(np.uint8)).save(
                out_dir / f"{s.category}_{s.image_id}_{name}.png")


def run(target: str, data_dir: str, durations: list[str], workers: int | None):
    results_dir = Path(__file__).parent.parent / "results" / target / "baselines"
    results_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 4) - 1)

    for duration in durations:
        print(f"\n{'=' * 60}\nDuration: {duration}\n{'=' * 60}")
        dataset = load_dataset(data_dir, duration=duration)
        samples = sample_pilot(dataset, n_per_category=10) if target == "pilot" else dataset
        fixmap_dir = Path(data_dir) / "saliency_maps" / f"fixmaps_{duration}"

        dur_dir = results_dir / duration
        dur_dir.mkdir(parents=True, exist_ok=True)

        priors_npz = dur_dir / "gt_priors.npz"
        if not priors_npz.exists():
            print(f"  Building mean-GT priors from {len(samples)} maps...")
            build_priors(samples, priors_npz)

        tasks = []
        for s in samples:
            fixmap_path = fixmap_dir / s.image_path.name
            tasks.append((str(s.heatmap_path),
                          str(fixmap_path) if fixmap_path.exists() else None,
                          s.image_id, s.category, str(priors_npz)))

        print(f"  Evaluating {len(tasks)} images x {2 + len(CENTER_SIGMA_FRACS) + 2} baselines "
              f"across {workers} workers...")
        all_rows, all_stats = [], []
        t0 = time.time()
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_compute_one, t) for t in tasks]
            for fut in as_completed(futures):
                rows, stats = fut.result()
                all_rows.extend(rows)
                all_stats.append(stats)
                done += 1
                if done % 200 == 0 or done == len(tasks):
                    rate = done / (time.time() - t0)
                    eta = (len(tasks) - done) / rate if rate > 0 else 0
                    print(f"    {done}/{len(tasks)} ({rate:.1f} img/s, eta {eta:.0f}s)", flush=True)

        df = pd.DataFrame(all_rows)
        raw_dir = dur_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for name, bdf in df.groupby("baseline"):
            bdf.to_csv(raw_dir / f"{name}.csv", index=False)

        avail = [m for m in METRICS if m in df.columns]
        summary = df.groupby("baseline")[avail].agg(["mean", "std"])
        summary.columns = [f"{m}_{s}" for m, s in summary.columns]
        summary.reset_index().to_csv(dur_dir / "summary.csv", index=False)

        cat_summary = df.groupby(["baseline", "category"])[avail].agg(["mean", "std"])
        cat_summary.columns = [f"{m}_{s}" for m, s in cat_summary.columns]
        cat_summary.reset_index().to_csv(dur_dir / "summary_by_category.csv", index=False)

        pd.DataFrame(all_stats).to_csv(dur_dir / "gt_stats.csv", index=False)

        print(f"\n  BASELINE SUMMARY ({duration})")
        print(summary.round(3).to_string())
        ent = pd.DataFrame(all_stats)["gt_entropy_norm"]
        print(f"\n  GT entropy (normalized): mean={ent.mean():.4f} std={ent.std():.4f}")

    save_verify_images(
        sample_pilot(load_dataset(data_dir, duration=durations[-1]), n_per_category=1),
        str(results_dir / durations[-1] / "gt_priors.npz"),
        results_dir / "verify",
    )
    print(f"\nDone. Results in {results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate non-VLM baselines on UEyes GT")
    parser.add_argument("--target", default="full", choices=["pilot", "full"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--durations", nargs="+", default=DURATIONS, choices=DURATIONS)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    run(args.target, args.data_dir, args.durations, args.workers)
