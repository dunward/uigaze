"""Diagnose the coordinate-clamp suspicion by capturing raw VLM responses.

Re-queries a small image sample with the EXACT same prompt/params as the main
experiment, but stores the raw response text before any parsing/clamping, then
reports the distribution of raw coordinate values per model.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vlm.openrouter import (
    MODELS,
    RESPONSE_FORMAT,
    ResponseHealingPlugin,
    _build_messages,
    _encode_image,
    _get_client,
    _get_provider_prefs,
)
from src.data_loader import load_dataset, sample_pilot

SUSPECT_MODELS = ["gemini-3.1-pro", "gemini-3.1-flash-lite", "ui-tars-1.5"]


async def fetch_raw(image_path: Path, model: str) -> str:
    model_id = MODELS[model]
    image_uri = _encode_image(image_path)
    async with _get_client() as client:
        response = await client.chat.send_async(
            model=model_id,
            messages=_build_messages(image_uri),
            temperature=0.1,
            max_tokens=8192,
            provider=_get_provider_prefs(model_id),
            response_format=RESPONSE_FORMAT,
            plugins=[ResponseHealingPlugin(id="response-healing")],
        )
    return response.choices[0].message.content


def extract_xy(raw: str) -> tuple[list[float], list[float]]:
    parsed = json.loads(raw.strip())
    if isinstance(parsed, dict):
        parsed = next((v for v in parsed.values() if isinstance(v, list)), [])
    xs = [float(p["x"]) for p in parsed]
    ys = [float(p["y"]) for p in parsed]
    return xs, ys


async def main(n_per_category: int, models: list[str], concurrency: int):
    out_dir = Path(__file__).parent.parent / "results" / "full" / "diagnosis"
    dataset = load_dataset("data", duration="7s")
    samples = sample_pilot(dataset, n_per_category=n_per_category, seed=7)
    print(f"Diagnosing {len(samples)} images x {len(models)} models "
          f"({len(samples) * len(models)} calls)\n")

    sem = asyncio.Semaphore(concurrency)
    stats: dict[str, list] = {m: [] for m in models}

    async def one(sample, model):
        async with sem:
            dest = out_dir / model
            dest.mkdir(parents=True, exist_ok=True)
            f = dest / f"{sample.image_id}.txt"
            try:
                raw = await fetch_raw(sample.image_path, model)
                f.write_text(raw)
                xs, ys = extract_xy(raw)
                stats[model].append((sample.image_id, xs, ys))
            except Exception as e:
                print(f"  ERROR {model}/{sample.image_id}: {type(e).__name__}: {e}")

    await asyncio.gather(*(one(s, m) for s in samples for m in models))

    print(f"\n{'model':<24}{'imgs':>5}{'pts':>6}{'frac>1':>8}{'frac>=100':>10}"
          f"{'max_x':>9}{'max_y':>9}{'p95_x':>9}{'p95_y':>9}")
    report = {}
    for model in models:
        allx = np.array([v for _, xs, _ in stats[model] for v in xs])
        ally = np.array([v for _, _, ys in stats[model] for v in ys])
        if len(allx) == 0:
            print(f"{model:<24}  (no data)")
            continue
        both = np.concatenate([allx, ally])
        row = {
            "n_images": len(stats[model]),
            "n_points": len(allx),
            "frac_gt1": float((both > 1.0).mean()),
            "frac_ge100": float((both >= 100).mean()),
            "max_x": float(allx.max()), "max_y": float(ally.max()),
            "p95_x": float(np.percentile(allx, 95)), "p95_y": float(np.percentile(ally, 95)),
        }
        report[model] = row
        print(f"{model:<24}{row['n_images']:>5}{row['n_points']:>6}{row['frac_gt1']:>8.3f}"
              f"{row['frac_ge100']:>10.3f}{row['max_x']:>9.2f}{row['max_y']:>9.2f}"
              f"{row['p95_x']:>9.2f}{row['p95_y']:>9.2f}")

    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(f"\nRaw responses and summary saved to {out_dir}/")
    print("Verdict guide: frac_gt1 near 0 -> no clamp bug (true model behavior); "
          "frac_gt1 substantial -> coordinates were emitted on a larger scale and clamped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture raw VLM responses for clamp diagnosis")
    parser.add_argument("--n-per-category", type=int, default=4)
    parser.add_argument("--models", nargs="+", default=SUSPECT_MODELS)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(main(args.n_per_category, args.models, args.concurrency))
