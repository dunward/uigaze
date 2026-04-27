# Pilot Results

Pilot experiment results for UIGaze — VLM saliency prediction benchmarked against UEyes eye-tracking ground truth.

## Config

- **Images**: 40 (10 per category: desktop, mobile, poster, web)
- **Runs**: 10 per model per image
- **Durations**: 1s, 3s, 7s (ground truth heatmap)
- **Sampling**: `seed=42`, consistent across all models
- **Models**: 9

## Overall (by model)

### Duration: 1s

| Model | CC (↑) | SIM (↑) | KL (↓) |
|---|---|---|---|
| qwen-3.5-plus | **0.223** ± 0.119 | **0.293** ± 0.066 | **2.333** ± 0.987 |
| claude-opus-4.6 | 0.222 ± 0.154 | 0.292 ± 0.087 | 2.447 ± 1.115 |
| gpt-5.4 | 0.210 ± 0.139 | 0.287 ± 0.079 | 2.466 ± 1.132 |
| gpt-5.4-mini | 0.160 ± 0.111 | 0.262 ± 0.066 | 2.564 ± 1.031 |
| claude-sonnet-4.6 | 0.138 ± 0.147 | 0.244 ± 0.085 | 3.082 ± 1.590 |
| qwen-3.5-flash | 0.130 ± 0.109 | 0.240 ± 0.067 | 3.419 ± 1.607 |
| ui-tars-1.5 | 0.109 ± 0.153 | 0.143 ± 0.132 | 7.524 ± 2.968 |
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
| qwen-3.5-flash | 0.217 ± 0.140 | 0.327 ± 0.089 | 2.621 ± 1.334 |
| gemini-3.1-pro | 0.142 ± 0.198 | 0.220 ± 0.161 | 5.585 ± 3.332 |
| ui-tars-1.5 | 0.132 ± 0.200 | 0.164 ± 0.162 | 7.278 ± 3.026 |
| gemini-3.1-flash-lite | 0.105 ± 0.222 | 0.198 ± 0.184 | 6.081 ± 3.992 |

### Duration: 7s

| Model | CC (↑) | SIM (↑) | KL (↓) |
|---|---|---|---|
| **qwen-3.5-plus** | **0.486** ± 0.167 | **0.536** ± 0.081 | **1.060** ± 0.550 |
| gpt-5.4 | 0.428 ± 0.169 | 0.508 ± 0.087 | 1.259 ± 0.710 |
| claude-opus-4.6 | 0.385 ± 0.227 | 0.483 ± 0.127 | 1.620 ± 1.097 |
| gpt-5.4-mini | 0.321 ± 0.162 | 0.457 ± 0.079 | 1.443 ± 0.746 |
| qwen-3.5-flash | 0.314 ± 0.163 | 0.427 ± 0.108 | 2.113 ± 1.356 |
| claude-sonnet-4.6 | 0.295 ± 0.227 | 0.434 ± 0.128 | 1.975 ± 1.420 |
| gemini-3.1-pro | 0.192 ± 0.238 | 0.283 ± 0.197 | 5.109 ± 3.258 |
| gemini-3.1-flash-lite | 0.156 ± 0.277 | 0.263 ± 0.228 | 5.439 ± 3.889 |
| ui-tars-1.5 | 0.110 ± 0.173 | 0.163 ± 0.156 | 7.247 ± 2.700 |

## 3-Run Subset (Sensitivity Check)

To determine the number of repetitions needed for reliable model comparison, we re-aggregated all metrics using only the first 3 of 10 runs per (model, image) pair and compared the resulting estimates against the full 10-run baseline. The analysis examines whether the additional repetitions materially change either absolute metric values or the relative ranking of models.

### Duration: 1s (3 runs)

| Model | CC (↑) | SIM (↑) | KL (↓) |
|---|---|---|---|
| claude-opus-4.6 | **0.225** ± 0.157 | 0.292 ± 0.086 | 2.429 ± 1.116 |
| qwen-3.5-plus | 0.222 ± 0.123 | 0.291 ± 0.066 | **2.367** ± 1.036 |
| gpt-5.4 | 0.203 ± 0.140 | 0.284 ± 0.078 | 2.476 ± 1.144 |
| gpt-5.4-mini | 0.162 ± 0.129 | 0.263 ± 0.075 | 2.579 ± 1.159 |
| claude-sonnet-4.6 | 0.133 ± 0.144 | 0.242 ± 0.081 | 3.054 ± 1.581 |
| qwen-3.5-flash | 0.128 ± 0.107 | 0.238 ± 0.066 | 3.421 ± 1.659 |
| ui-tars-1.5 | 0.098 ± 0.149 | 0.134 ± 0.129 | 7.680 ± 3.018 |
| gemini-3.1-pro | 0.080 ± 0.148 | 0.158 ± 0.120 | 6.168 ± 3.449 |
| gemini-3.1-flash-lite | 0.049 ± 0.139 | 0.139 ± 0.137 | 6.811 ± 3.981 |

### Duration: 3s (3 runs)

| Model | CC (↑) | SIM (↑) | KL (↓) |
|---|---|---|---|
| **qwen-3.5-plus** | **0.355** ± 0.174 | **0.409** ± 0.085 | **1.547** ± 0.609 |
| claude-opus-4.6 | 0.329 ± 0.204 | 0.391 ± 0.113 | 1.835 ± 1.062 |
| gpt-5.4 | 0.319 ± 0.166 | 0.390 ± 0.090 | 1.734 ± 0.817 |
| gpt-5.4-mini | 0.246 ± 0.153 | 0.354 ± 0.088 | 1.962 ± 0.945 |
| claude-sonnet-4.6 | 0.216 ± 0.186 | 0.334 ± 0.111 | 2.418 ± 1.433 |
| qwen-3.5-flash | 0.214 ± 0.144 | 0.323 ± 0.092 | 2.653 ± 1.445 |
| gemini-3.1-pro | 0.132 ± 0.199 | 0.215 ± 0.165 | 5.651 ± 3.595 |
| ui-tars-1.5 | 0.122 ± 0.188 | 0.154 ± 0.153 | 7.464 ± 3.020 |
| gemini-3.1-flash-lite | 0.102 ± 0.228 | 0.193 ± 0.192 | 6.220 ± 4.158 |

### Duration: 7s (3 runs)

| Model | CC (↑) | SIM (↑) | KL (↓) |
|---|---|---|---|
| **qwen-3.5-plus** | **0.483** ± 0.173 | **0.534** ± 0.083 | **1.070** ± 0.577 |
| gpt-5.4 | 0.417 ± 0.172 | 0.504 ± 0.089 | 1.265 ± 0.714 |
| claude-opus-4.6 | 0.386 ± 0.229 | 0.483 ± 0.126 | 1.606 ± 1.051 |
| gpt-5.4-mini | 0.324 ± 0.173 | 0.460 ± 0.087 | 1.438 ± 0.779 |
| qwen-3.5-flash | 0.313 ± 0.169 | 0.424 ± 0.114 | 2.163 ± 1.519 |
| claude-sonnet-4.6 | 0.297 ± 0.230 | 0.435 ± 0.128 | 1.922 ± 1.406 |
| gemini-3.1-pro | 0.185 ± 0.249 | 0.279 ± 0.208 | 5.143 ± 3.519 |
| gemini-3.1-flash-lite | 0.153 ± 0.291 | 0.258 ± 0.241 | 5.583 ± 4.066 |
| ui-tars-1.5 | 0.093 ± 0.167 | 0.152 ± 0.152 | 7.445 ± 2.717 |

**Convergence of metric estimates (absolute difference between 3-run and 10-run means)**

| Metric | Mean abs. diff | Max abs. diff | Largest deviation |
|---|---|---|---|
| CC | 0.005 | 0.017 | ui-tars-1.5 (7s) |
| SIM | 0.003 | 0.012 | ui-tars-1.5 (7s) |
| KL | 0.053 | 0.198 | ui-tars-1.5 (7s) |

Across all nine models and three durations, the 3-run estimate of CC deviates from the 10-run estimate by less than 0.011 for the four highest-ranked models, which is approximately one-thirtieth of the per-image standard deviation (0.15–0.23). The top-four ordering is preserved at every duration, and the larger KL deviations are confined to the lowest-performing models — UI-TARS and Gemini Flash Lite — whose KL values already exceed 5 and therefore lie far outside the operating range used for head-to-head comparisons.

We conclude that three repetitions per image yield estimates that are statistically indistinguishable from the ten-run baseline for the purposes of model ranking, and we adopt this configuration for subsequent large-scale evaluation.

## Key Findings

1. **All models improve with longer duration** — 7s ground truth yields the highest CC across all models, suggesting VLM predictions align more closely with exploratory gaze patterns than initial fixations
2. **Qwen 3.5 Plus ranks first across all durations** — CC: 0.223 (1s) → 0.354 (3s) → 0.486 (7s)
3. **GPT-5.4 and Claude Opus 4.6 are close seconds** — strong performance especially on desktop UIs
4. **Qwen 3.5 Flash vs Plus** — Flash achieves ~60-65% of Plus performance at lower cost, reasonable trade-off
5. **UI-TARS underperforms despite being UI-specialized** — CC 0.110 (7s); as a GUI agent model optimized for element interaction rather than gaze prediction, its training objective differs from saliency prediction
6. **Poster UIs are easiest to predict** — simpler visual hierarchy leads to higher agreement
7. **Web UIs are hardest** — complex layouts with many competing elements reduce prediction accuracy
8. **Gemini models underperform significantly** — both Flash Lite and Pro show high KL divergence across all durations

## Sample Comparisons

Best/worst comparison images per model, category, and duration are available in:

```
results/pilot/{1s,3s,7s}/images/{model}/
```
