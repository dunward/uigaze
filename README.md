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

```bash
# All models (default 10 runs each)
uv run python experiments/run_pilot.py

# Specific models
uv run python experiments/run_pilot.py --models gpt-5.4-mini qwen-3.5-plus

# Custom runs / concurrency
uv run python experiments/run_pilot.py --models gpt-5.4-mini --n-runs 3 --concurrency 5
```

Available models: `gpt-5.4`, `gpt-5.4-mini`, `claude-opus-4.6`, `claude-sonnet-4.6`, `qwen-3.5-plus`, `gemini-3.1-pro`, `gemini-3.1-flash-lite`

### 2. Generate metrics and images

```bash
# All durations (1s, 3s, 7s)
uv run python experiments/regenerate.py

# Specific duration
uv run python experiments/regenerate.py --durations 3s
```

### 3. Fill missing predictions (if any failed)

```bash
uv run python experiments/fill_missing.py --model gpt-5.4-mini
```

### Quick test (single image)

```bash
uv run python experiments/run_single.py --model gpt-5.4-mini
```

## Results

- [Full Results](FULL_RESULTS.md) — 1,980 images × 9 models × 3 runs × 3 durations
- [Pilot Results](PILOT_RESULTS.md) — 40 images × 9 models × 10 runs × 3 durations
