# UIGaze

Benchmarking VLM saliency prediction on user interfaces against real human eye-tracking data (UEyes dataset).

## Setup

```bash
uv sync
cp .env.example .env  # add your OpenRouter API key
```

## Dataset

Download the [UEyes dataset](https://zenodo.org/record/8010312) (~12.9GB):

```bash
./scripts/download_data.sh
```

Or manually download and extract into `data/`:

```
data/
├── images/
├── saliency_maps/
│   ├── heatmaps_{1s,3s,7s}/
│   └── fixmaps_{1s,3s,7s}/
├── eyetracker_logs/
└── image_types.csv
```

## Run

### 1. Collect predictions

Pilot (40 images × 10 runs, sanity check):

```bash
uv run python experiments/run_pilot.py
uv run python experiments/run_pilot.py --models gpt-5.4-mini qwen-3.5-plus
```

Full (1,980 images × 3 runs):

```bash
uv run python experiments/run_full.py
uv run python experiments/run_full.py --models gpt-5.4 --concurrency 200
uv run python experiments/run_full.py --categories desktop poster
```

Available models: `gpt-5.4`, `gpt-5.4-mini`, `claude-opus-4.6`, `claude-sonnet-4.6`, `qwen-3.5-plus`, `qwen-3.5-flash`, `gemini-3.1-pro`, `gemini-3.1-flash-lite`, `ui-tars-1.5`

### 2. Generate metrics and images

```bash
# Pilot
uv run python experiments/regenerate.py --target pilot

# Full
uv run python experiments/regenerate.py --target full

# Specific duration
uv run python experiments/regenerate.py --target full --durations 3s
```

### 3. Fill missing predictions (if any failed)

```bash
uv run python experiments/fill_missing.py --target pilot --model gpt-5.4-mini
uv run python experiments/fill_missing.py --target full --model gpt-5.4 --concurrency 200
```

### Quick test (single image)

```bash
uv run python experiments/run_single.py --model gpt-5.4-mini
```

## Results

- [Full Results](FULL_RESULTS.md) — 1,980 images × 9 models × 3 runs × 3 durations
- [Pilot Results](PILOT_RESULTS.md) — 40 images × 9 models × 10 runs × 3 durations
