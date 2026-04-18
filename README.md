# UIGaze

Benchmarking VLM saliency prediction on user interfaces against real human eye-tracking data (UEyes dataset).

## Setup

```bash
uv sync
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
│   ├── heatmaps/{1s,3s,7s}/
│   └── fixmaps/{1s,3s,7s}/
├── eyetracker_logs/
└── info.csv
```

## Run

```bash
# Set API key
export OPENROUTER_API_KEY=sk-or-...

# Pilot experiment
uv run python experiments/run_pilot.py
```
