# Pilot Results

Pilot experiment results for UIGaze — VLM saliency prediction benchmarked against UEyes eye-tracking ground truth.

## Config

- **Images**: 40 (10 per category: desktop, mobile, poster, web)
- **Runs**: 10 per model per image
- **Durations**: 1s, 3s, 7s (ground truth heatmap)
- **Sampling**: `seed=42`, consistent across all models
- **Models**: 7 (GPT-5.4, GPT-5.4-mini, Claude Opus 4.6, Claude Sonnet 4.6, Qwen 3.5 Plus, Gemini 3.1 Pro, Gemini 3.1 Flash Lite)

## Overall (by model)

### Duration: 1s

| Model | CC (↑) | SIM (↑) | KL (↓) |
|---|---|---|---|
| qwen-3.5-plus | **0.223** ± 0.119 | **0.293** ± 0.066 | **2.333** ± 0.987 |
| claude-opus-4.6 | 0.222 ± 0.154 | 0.292 ± 0.087 | 2.447 ± 1.115 |
| gpt-5.4 | 0.210 ± 0.139 | 0.287 ± 0.079 | 2.466 ± 1.132 |
| gpt-5.4-mini | 0.160 ± 0.111 | 0.262 ± 0.066 | 2.564 ± 1.031 |
| claude-sonnet-4.6 | 0.138 ± 0.147 | 0.244 ± 0.085 | 3.082 ± 1.590 |
| gemini-3.1-pro | 0.086 ± 0.144 | 0.160 ± 0.115 | 6.176 ± 3.171 |
| gemini-3.1-flash-lite | 0.053 ± 0.134 | 0.143 ± 0.131 | 6.674 ± 3.789 |

### Duration: 3s

| Model | CC (↑) | SIM (↑) | KL (↓) |
|---|---|---|---|
| **qwen-3.5-plus** | **0.354** ± 0.168 | **0.409** ± 0.084 | **1.548** ± 0.607 |
| gpt-5.4 | 0.329 ± 0.165 | 0.395 ± 0.089 | 1.725 ± 0.826 |
| claude-opus-4.6 | 0.327 ± 0.200 | 0.391 ± 0.113 | 1.839 ± 1.073 |
| gpt-5.4-mini | 0.248 ± 0.140 | 0.355 ± 0.081 | 1.925 ± 0.842 |
| claude-sonnet-4.6 | 0.219 ± 0.186 | 0.335 ± 0.113 | 2.455 ± 1.460 |
| gemini-3.1-pro | 0.142 ± 0.198 | 0.220 ± 0.161 | 5.585 ± 3.332 |
| gemini-3.1-flash-lite | 0.105 ± 0.222 | 0.198 ± 0.184 | 6.081 ± 3.992 |

### Duration: 7s

| Model | CC (↑) | SIM (↑) | KL (↓) |
|---|---|---|---|
| **qwen-3.5-plus** | **0.486** ± 0.167 | **0.536** ± 0.081 | **1.060** ± 0.550 |
| gpt-5.4 | 0.428 ± 0.169 | 0.508 ± 0.087 | 1.259 ± 0.710 |
| claude-opus-4.6 | 0.385 ± 0.227 | 0.483 ± 0.127 | 1.620 ± 1.097 |
| gpt-5.4-mini | 0.321 ± 0.162 | 0.457 ± 0.079 | 1.443 ± 0.746 |
| claude-sonnet-4.6 | 0.295 ± 0.227 | 0.434 ± 0.128 | 1.975 ± 1.420 |
| gemini-3.1-pro | 0.192 ± 0.238 | 0.283 ± 0.197 | 5.109 ± 3.258 |
| gemini-3.1-flash-lite | 0.156 ± 0.277 | 0.263 ± 0.228 | 5.439 ± 3.889 |

## By Category (3s)

### Desktop

| Model | CC (↑) | SIM (↑) | KL (↓) |
|---|---|---|---|
| gpt-5.4 | **0.425** ± 0.123 | **0.449** ± 0.066 | **1.133** ± 0.379 |
| claude-opus-4.6 | 0.398 ± 0.137 | 0.446 ± 0.074 | 1.226 ± 0.456 |
| qwen-3.5-plus | 0.352 ± 0.192 | 0.429 ± 0.074 | 1.249 ± 0.376 |
| claude-sonnet-4.6 | 0.297 ± 0.190 | 0.400 ± 0.088 | 1.417 ± 0.468 |
| gpt-5.4-mini | 0.293 ± 0.157 | 0.391 ± 0.059 | 1.380 ± 0.362 |
| gemini-3.1-pro | 0.202 ± 0.232 | 0.296 ± 0.178 | 4.403 ± 3.563 |
| gemini-3.1-flash-lite | 0.159 ± 0.237 | 0.275 ± 0.164 | 4.088 ± 3.688 |

### Mobile

| Model | CC (↑) | SIM (↑) | KL (↓) |
|---|---|---|---|
| **qwen-3.5-plus** | **0.307** ± 0.101 | **0.403** ± 0.047 | **1.977** ± 0.720 |
| gpt-5.4 | 0.266 ± 0.136 | 0.375 ± 0.068 | 2.349 ± 1.026 |
| claude-opus-4.6 | 0.236 ± 0.220 | 0.354 ± 0.128 | 2.542 ± 1.198 |
| gemini-3.1-pro | 0.233 ± 0.193 | 0.302 ± 0.138 | 3.778 ± 2.331 |
| gpt-5.4-mini | 0.182 ± 0.134 | 0.334 ± 0.079 | 2.567 ± 0.988 |
| claude-sonnet-4.6 | 0.139 ± 0.175 | 0.285 ± 0.110 | 3.612 ± 1.516 |
| gemini-3.1-flash-lite | 0.039 ± 0.204 | 0.110 ± 0.174 | 7.764 ± 3.368 |

### Poster

| Model | CC (↑) | SIM (↑) | KL (↓) |
|---|---|---|---|
| **qwen-3.5-plus** | **0.458** ± 0.141 | **0.451** ± 0.088 | **1.305** ± 0.541 |
| gpt-5.4 | 0.401 ± 0.131 | 0.437 ± 0.084 | 1.410 ± 0.508 |
| claude-opus-4.6 | 0.389 ± 0.165 | 0.423 ± 0.098 | 1.333 ± 0.489 |
| gpt-5.4-mini | 0.320 ± 0.110 | 0.391 ± 0.079 | 1.722 ± 0.729 |
| claude-sonnet-4.6 | 0.302 ± 0.167 | 0.380 ± 0.107 | 1.963 ± 1.237 |
| gemini-3.1-flash-lite | 0.180 ± 0.264 | 0.274 ± 0.205 | 5.030 ± 4.355 |
| gemini-3.1-pro | 0.148 ± 0.178 | 0.196 ± 0.147 | 6.179 ± 3.368 |

### Web

| Model | CC (↑) | SIM (↑) | KL (↓) |
|---|---|---|---|
| **qwen-3.5-plus** | **0.299** ± 0.193 | **0.355** ± 0.097 | **1.662** ± 0.510 |
| claude-opus-4.6 | 0.286 ± 0.241 | 0.340 ± 0.124 | 2.257 ± 1.289 |
| gpt-5.4 | 0.225 ± 0.185 | 0.320 ± 0.081 | 2.010 ± 0.695 |
| gpt-5.4-mini | 0.197 ± 0.121 | 0.305 ± 0.081 | 2.031 ± 0.765 |
| claude-sonnet-4.6 | 0.139 ± 0.162 | 0.275 ± 0.102 | 2.827 ± 1.444 |
| gemini-3.1-flash-lite | 0.041 ± 0.166 | 0.133 ± 0.146 | 7.442 ± 3.782 |
| gemini-3.1-pro | -0.015 ± 0.085 | 0.085 ± 0.068 | 7.981 ± 2.587 |

## Key Findings

1. **All models improve with longer duration** — 7s ground truth yields the highest CC across all models, suggesting VLM predictions align more closely with exploratory gaze patterns than initial fixations
2. **Qwen 3.5 Plus ranks first across all durations** — CC: 0.223 (1s) → 0.354 (3s) → 0.486 (7s)
3. **GPT-5.4 and Claude Opus 4.6 are close seconds** — strong performance especially on desktop UIs
4. **Poster UIs are easiest to predict** — simpler visual hierarchy leads to higher agreement
5. **Web UIs are hardest** — complex layouts with many competing elements reduce prediction accuracy
6. **Gemini models underperform significantly** — both Flash Lite and Pro show high KL divergence across all durations
7. **Model cost vs performance trade-off** — GPT-5.4-mini achieves reasonable results at much lower cost

## Sample Comparisons

Best/worst comparison images per model, category, and duration are available in:

```
results/pilot/{1s,3s,7s}/images/{model}/
```
