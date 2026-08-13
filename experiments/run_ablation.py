"""Prompt ablation on a balanced 200-image subset.

Variants:
    base    - the original SALIENCY_PROMPT (30-50 points)
    free_n  - point-count constraint relaxed (5-80, "as many as needed")
    first1s - temporal cue: predict fixations within the FIRST 1 second

Models: top-3 by 7s CC (gpt-5.4, claude-opus-4.6, qwen-3.5-plus), 1 run each.
Predictions are saved under results/ablation/predictions/<model>/<variant>/ and
evaluated against all three GT durations with the standard pipeline.

NOTE: requires a working OPENROUTER_API_KEY. Run with --trial first (10 images)
to verify parsing and get a cost reading from the OpenRouter dashboard before
the full run.
"""

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vlm import openrouter as orx
from src.data_loader import load_dataset, sample_pilot

MODELS = ["gpt-5.4", "claude-opus-4.6", "qwen-3.5-plus"]
SEED = 202

FREE_N_PROMPT = orx.SALIENCY_PROMPT.replace(
    "- Output 30-50 points covering all areas of high visual saliency",
    "- Output as many points as needed to describe the attention distribution "
    "(anywhere from 5 to 80 points)",
)

FIRST1S_PROMPT = orx.SALIENCY_PROMPT.replace(
    "predict where users would look within the first few seconds.",
    "predict where users would look within the FIRST 1 SECOND of viewing - "
    "only the initial reflexive fixations, not later exploratory gaze.",
)

VARIANTS = {"base": orx.SALIENCY_PROMPT, "free_n": FREE_N_PROMPT, "first1s": FIRST1S_PROMPT}


async def run(n_per_category: int, concurrency: int, trial: bool):
    out_base = Path(__file__).parent.parent / "results" / "ablation" / "predictions"
    dataset = load_dataset("data", duration="7s")
    samples = sample_pilot(dataset, n_per_category=n_per_category, seed=SEED)
    if trial:
        samples = samples[:10]

    n_calls = sum(1 for _ in samples) * len(MODELS) * len(VARIANTS)
    existing = sum(1 for m in MODELS for v in VARIANTS
                   for s in samples
                   if (out_base / m / v / f"{s.image_id}_run01.json").exists())
    print(f"{len(samples)} images x {len(MODELS)} models x {len(VARIANTS)} variants "
          f"= {n_calls} calls ({existing} already done, {n_calls - existing} to go)")

    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def one(sample, model, variant, prompt):
        nonlocal done
        dest = out_base / model / variant
        dest.mkdir(parents=True, exist_ok=True)
        f = dest / f"{sample.image_id}_run01.json"
        if f.exists():
            return
        async with sem:
            try:
                points = await orx.predict_saliency_async(sample.image_path, model=model)
                f.write_text(json.dumps([asdict(p) for p in points]))
            except Exception as e:
                print(f"  ERROR {model}/{variant}/{sample.image_id}: "
                      f"{type(e).__name__}: {e}")
            done += 1
            if done % 50 == 0:
                print(f"  {done} calls completed", flush=True)

    # Variants run sequentially; the module prompt is swapped once per variant
    # (never per task) so all concurrent calls inside a variant see the same prompt.
    orig_prompt = orx.SALIENCY_PROMPT
    try:
        for variant, prompt in VARIANTS.items():
            print(f"\n--- variant: {variant} ---")
            orx.SALIENCY_PROMPT = prompt
            await asyncio.gather(*(one(s, m, variant, prompt)
                                   for s in samples for m in MODELS))
    finally:
        orx.SALIENCY_PROMPT = orig_prompt

    print(f"\nDone. Predictions in {out_base}/")
    print("Next: evaluate with experiments/eval_ablation.py (reuses regenerate logic).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt ablation subset run")
    parser.add_argument("--n-per-category", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--trial", action="store_true", help="10 images only")
    args = parser.parse_args()
    asyncio.run(run(args.n_per_category, args.concurrency, args.trial))
