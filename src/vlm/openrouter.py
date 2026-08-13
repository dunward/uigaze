import asyncio
import base64
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image
from openrouter import OpenRouter
from openrouter.components.chatformatjsonschemaconfig import (
    ChatFormatJSONSchemaConfig,
    ChatJSONSchemaConfig,
)
from openrouter.components.providerpreferences import ProviderPreferences
from openrouter.components.responsehealingplugin import ResponseHealingPlugin
from openrouter.utils.retries import BackoffStrategy, RetryConfig

load_dotenv()

SALIENCY_PROMPT = """\
You are an expert in visual attention and UI design.
Given this UI screenshot, predict where users would look within the first few seconds.
Output ONLY a JSON array of gaze points with NO additional text:
[{"x": <0.0-1.0>, "y": <0.0-1.0>, "intensity": <0.0-1.0>}, ...]

Rules:
- x: normalized horizontal position (0=left edge, 1=right edge)
- y: normalized vertical position (0=top edge, 1=bottom edge)
- intensity: predicted attention strength (1.0=highest attention)
- Output 30-50 points covering all areas of high visual saliency
- Focus on: text headings, buttons, images, navigation elements, and other salient UI components
- Distribute points to reflect the likely distribution of human visual attention
"""

MODELS = {
    # OpenAI
    "gpt-5.4": "openai/gpt-5.4-20260305",
    "gpt-5.4-mini": "openai/gpt-5.4-mini-20260317",
    # Google
    "gemini-3.1-pro": "google/gemini-3.1-pro-preview-20260219",
    "gemini-3.1-flash-lite": "google/gemini-3.1-flash-lite-preview-20260303",
    # Anthropic
    "claude-opus-4.6": "anthropic/claude-4.6-opus-20260205",
    "claude-sonnet-4.6": "anthropic/claude-4.6-sonnet-20260217",
    # Qwen
    "qwen-3.5-plus": "qwen/qwen3.5-plus-20260216",
    "qwen-3.5-flash": "qwen/qwen3.5-flash-20260224",
    # ByteDance
    "ui-tars-1.5": "bytedance/ui-tars-1.5-7b",
}

# Provider routing preferences per model prefix
PROVIDER_PREFS = {
    "google/": ProviderPreferences(
        order=["Google AI Studio"],
        allow_fallbacks=False,
    ),
}


RESPONSE_FORMAT = ChatFormatJSONSchemaConfig(
    type="json_schema",
    json_schema=ChatJSONSchemaConfig(
        name="gaze_points",
        strict=True,
        schema_={
            "type": "object",
            "properties": {
                "points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "intensity": {"type": "number"},
                        },
                        "required": ["x", "y", "intensity"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["points"],
            "additionalProperties": False,
        },
    ),
)


@dataclass
class GazePoint:
    x: float  # 0-1, left to right
    y: float  # 0-1, top to bottom
    intensity: float  # 0-1, attention strength


def _detect_media_type(data: bytes) -> str:
    """Detect image media type from file magic bytes."""
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if data[:2] == b'\xff\xd8':
        return "image/jpeg"
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return "image/webp"
    if data[:2] == b'BM':
        return "image/bmp"
    return "image/png"


# Anthropic limits images to 5MB after base64 encoding. Apply same cap globally.
_MAX_BASE64_BYTES = 5 * 1024 * 1024
_DOWNSCALE_LONG_SIDE = 2048


def _encode_image(image_path: Path) -> str:
    """Encode image to base64 data URI; auto-downscale if it exceeds provider size caps."""
    data = image_path.read_bytes()
    media_type = _detect_media_type(data)

    # Quick size check — base64 inflates ~33%, so raw bytes ~> 3.75MB will exceed the cap.
    if len(data) * 4 // 3 > _MAX_BASE64_BYTES:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        long = max(img.size)
        if long > _DOWNSCALE_LONG_SIDE:
            img.thumbnail((_DOWNSCALE_LONG_SIDE, _DOWNSCALE_LONG_SIDE))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90, optimize=True)
        data = buf.getvalue()
        media_type = "image/jpeg"

    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{media_type};base64,{b64}"


def _get_client(api_key: str | None = None) -> OpenRouter:
    """Create an OpenRouter client with retry config."""
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenRouter API key required. Set OPENROUTER_API_KEY env var or pass api_key."
        )

    return OpenRouter(
        api_key=api_key,
        http_referer="https://github.com/uigaze",
        x_open_router_title="UIGaze Research",
        retry_config=RetryConfig(
            "backoff",
            BackoffStrategy(
                initial_interval=2000,
                max_interval=30000,
                max_elapsed_time=120000,
                exponent=1.5,
            ),
            retry_connection_errors=True,
        ),
        timeout_ms=120000,
    )


def _build_messages(image_uri: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": SALIENCY_PROMPT},
                {"type": "image_url", "image_url": {"url": image_uri}},
            ],
        }
    ]


def _get_provider_prefs(model_id: str) -> ProviderPreferences | None:
    """Return provider routing preferences for a given model ID."""
    for prefix, prefs in PROVIDER_PREFS.items():
        if model_id.startswith(prefix):
            return prefs
    return None


def predict_saliency(
    image_path: str | Path,
    model: str = "gpt-5.4-mini",
    api_key: str | None = None,
) -> list[GazePoint]:
    """Call a VLM via OpenRouter to predict saliency points (sync)."""
    image_path = Path(image_path)
    model_id = MODELS.get(model, model)
    image_size = Image.open(image_path).size
    image_uri = _encode_image(image_path)

    provider = _get_provider_prefs(model_id)

    with _get_client(api_key) as client:
        response = client.chat.send(
            model=model_id,
            messages=_build_messages(image_uri),
            temperature=0.1,
            max_tokens=8192,
            provider=provider,
            response_format=RESPONSE_FORMAT,
            plugins=[ResponseHealingPlugin(id="response-healing")],
        )

    return _parse_gaze_points(response.choices[0].message.content,
                              image_size=image_size, model=model)


async def predict_saliency_async(
    image_path: str | Path,
    model: str = "gpt-5.4-mini",
    api_key: str | None = None,
    return_raw: bool = False,
) -> list[GazePoint] | tuple[list[GazePoint], str]:
    """Call a VLM via OpenRouter to predict saliency points (async)."""
    image_path = Path(image_path)
    model_id = MODELS.get(model, model)
    image_size = Image.open(image_path).size
    image_uri = _encode_image(image_path)

    provider = _get_provider_prefs(model_id)

    async with _get_client(api_key) as client:
        response = await client.chat.send_async(
            model=model_id,
            messages=_build_messages(image_uri),
            temperature=0.1,
            max_tokens=8192,
            provider=provider,
            response_format=RESPONSE_FORMAT,
            plugins=[ResponseHealingPlugin(id="response-healing")],
        )

    content = response.choices[0].message.content
    points = _parse_gaze_points(content, image_size=image_size, model=model)
    return (points, content) if return_raw else points


async def predict_batch_async(
    image_paths: list[Path],
    model: str = "gpt-5.4-mini",
    api_key: str | None = None,
    concurrency: int = 10,
    raw_dir: Path | None = None,
) -> dict[str, list[GazePoint]]:
    """Run saliency prediction on multiple images concurrently.

    Args:
        image_paths: List of image paths.
        model: Model short name or full OpenRouter model ID.
        api_key: OpenRouter API key.
        concurrency: Max parallel requests.

    Returns:
        Dict mapping image filename (stem) to gaze points.
    """
    semaphore = asyncio.Semaphore(concurrency)
    results = {}

    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)

    async def _predict_one(path: Path, idx: int):
        async with semaphore:
            print(f"  [{idx+1}/{len(image_paths)}] {path.name} ({model})")
            try:
                if raw_dir is not None:
                    points, raw = await predict_saliency_async(
                        path, model=model, api_key=api_key, return_raw=True)
                    (raw_dir / f"{path.stem}.txt").write_text(raw)
                else:
                    points = await predict_saliency_async(path, model=model, api_key=api_key)
                results[path.stem] = points
            except Exception as e:
                print(f"    ERROR [{type(e).__name__}]: {e}")
                results[path.stem] = []

    tasks = [_predict_one(p, i) for i, p in enumerate(image_paths)]
    await asyncio.gather(*tasks)
    return results


def predict_batch(
    image_paths: list[Path],
    model: str = "gpt-5.4-mini",
    api_key: str | None = None,
    concurrency: int = 10,
    raw_dir: Path | None = None,
) -> dict[str, list[GazePoint]]:
    """Run saliency prediction on multiple images concurrently (sync wrapper)."""
    return asyncio.run(
        predict_batch_async(image_paths, model=model, api_key=api_key,
                            concurrency=concurrency, raw_dir=raw_dir)
    )


# Models documented to emit absolute pixel coordinates (UI-TARS) vs the
# Gemini-style 0-1000 grid. Used only to disambiguate responses whose
# coordinates exceed 100 but stay within both candidate ranges.
_PIXEL_COORD_MODELS = ("ui-tars",)


def _axis_divisor(
    vals: list[float],
    dim: int | None,
    model: str | None,
) -> float:
    """Infer the large-scale divisor for one axis from its out-of-range values.

    Models sometimes ignore the normalized-coordinate instruction and answer on
    another scale — observed in raw responses: 0-10, percent, 0-1000 grid, and
    absolute pixels, with x and y frequently on DIFFERENT scales in the same
    response (e.g., normalized x with pixel y). Values <= 1.5 are treated as
    already normalized and never rescaled; the divisor below applies only to the
    axis's larger values.
    """
    big = [abs(v) for v in vals if abs(v) > 1.5]
    if not big:
        return 1.0
    m = max(big)
    if m <= 15:
        return 10.0
    if m <= 150:
        return 100.0
    if dim:
        fits_pixels = m <= dim * 1.05
        exceeds_grid = m > 1000
        prefers_pixels = model and any(k in model for k in _PIXEL_COORD_MODELS)
        if fits_pixels and (exceeds_grid or prefers_pixels):
            return float(dim)
    if m <= 1050:
        return 1000.0
    return float(dim) if dim else m  # last resort: squash into range


def _rescale_value(v: float, divisor: float) -> float:
    """Rescale one coordinate: values already in [0, 1.5] pass through."""
    return v if abs(v) <= 1.5 else v / divisor


def _parse_gaze_points(
    content: str,
    image_size: tuple[int, int] | None = None,
    model: str | None = None,
) -> list[GazePoint]:
    """Parse VLM response text into GazePoint list (scale-aware)."""
    text = content.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"    RAW RESPONSE: {text}")
        raise

    # Handle both raw array and wrapped object (e.g. {"gaze_points": [...]})
    if isinstance(parsed, list):
        points_data = parsed
    elif isinstance(parsed, dict):
        # Find the first list value in the dict
        points_data = next(
            (v for v in parsed.values() if isinstance(v, list)), []
        )
    else:
        raise ValueError(f"Unexpected JSON type: {type(parsed)}")

    valid = [p for p in points_data
             if isinstance(p, dict) and all(k in p for k in ("x", "y", "intensity"))]
    if len(valid) < len(points_data):
        print(f"    WARN: skipped {len(points_data) - len(valid)} malformed points")
    if not valid:
        return []

    xs = [float(p["x"]) for p in valid]
    ys = [float(p["y"]) for p in valid]
    w, h = image_size if image_size else (None, None)
    div_x = _axis_divisor(xs, w, model)
    div_y = _axis_divisor(ys, h, model)

    intensities = [float(p["intensity"]) for p in valid]
    div_i = _axis_divisor(intensities, None, None)

    gaze_points = []
    for x, y, intensity in zip(xs, ys, intensities):
        gaze_points.append(GazePoint(
            x=max(0.0, min(1.0, _rescale_value(x, div_x))),
            y=max(0.0, min(1.0, _rescale_value(y, div_y))),
            intensity=max(0.0, min(1.0, _rescale_value(intensity, div_i))),
        ))

    return gaze_points
