import asyncio
import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
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


def _encode_image(image_path: Path) -> str:
    """Encode image to base64 data URI with actual format detection."""
    data = image_path.read_bytes()
    media_type = _detect_media_type(data)
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

    return _parse_gaze_points(response.choices[0].message.content)


async def predict_saliency_async(
    image_path: str | Path,
    model: str = "gpt-5.4-mini",
    api_key: str | None = None,
) -> list[GazePoint]:
    """Call a VLM via OpenRouter to predict saliency points (async)."""
    image_path = Path(image_path)
    model_id = MODELS.get(model, model)
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

    return _parse_gaze_points(response.choices[0].message.content)


async def predict_batch_async(
    image_paths: list[Path],
    model: str = "gpt-5.4-mini",
    api_key: str | None = None,
    concurrency: int = 10,
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

    async def _predict_one(path: Path, idx: int):
        async with semaphore:
            print(f"  [{idx+1}/{len(image_paths)}] {path.name} ({model})")
            try:
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
) -> dict[str, list[GazePoint]]:
    """Run saliency prediction on multiple images concurrently (sync wrapper)."""
    return asyncio.run(
        predict_batch_async(image_paths, model=model, api_key=api_key, concurrency=concurrency)
    )


def _parse_gaze_points(content: str) -> list[GazePoint]:
    """Parse VLM response text into GazePoint list."""
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

    gaze_points = []
    for p in points_data:
        x = max(0.0, min(1.0, float(p["x"])))
        y = max(0.0, min(1.0, float(p["y"])))
        intensity = max(0.0, min(1.0, float(p["intensity"])))
        gaze_points.append(GazePoint(x=x, y=y, intensity=intensity))

    return gaze_points
