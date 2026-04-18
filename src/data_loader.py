from dataclasses import dataclass
from pathlib import Path
import random

import pandas as pd


@dataclass
class Sample:
    image_path: Path
    heatmap_path: Path
    category: str
    image_id: str


def load_dataset(
    data_dir: str | Path,
    duration: str = "3s",
    categories: list[str] | None = None,
) -> list[Sample]:
    """Load UEyes dataset samples.

    Expected data_dir structure:
        data/
        ├── images/           # UI screenshots (1920x1200)
        ├── saliency_maps/
        │   ├── 1s/
        │   ├── 3s/
        │   └── 7s/
        └── info.csv          # image_id, category, ...

    Args:
        data_dir: Path to the UEyes data directory.
        duration: Heatmap duration - "1s", "3s", or "7s".
        categories: Filter by category. None = all categories.

    Returns:
        List of Sample objects with matched image/heatmap pairs.
    """
    data_dir = Path(data_dir)
    info_path = data_dir / "info.csv"

    if not info_path.exists():
        raise FileNotFoundError(
            f"info.csv not found at {info_path}. "
            "Please download the UEyes dataset and place it in the data/ directory."
        )

    info = pd.read_csv(info_path)

    if categories:
        info = info[info["category"].isin(categories)]

    images_dir = data_dir / "images"
    heatmaps_dir = data_dir / "saliency_maps" / duration

    samples = []
    for _, row in info.iterrows():
        image_id = str(row["image_id"])
        category = row["category"]

        # Try common image extensions
        image_path = _find_image(images_dir, image_id)
        heatmap_path = _find_image(heatmaps_dir, image_id)

        if image_path and heatmap_path:
            samples.append(
                Sample(
                    image_path=image_path,
                    heatmap_path=heatmap_path,
                    category=category,
                    image_id=image_id,
                )
            )

    if not samples:
        raise ValueError(
            f"No valid samples found in {data_dir}. "
            "Check that images/ and saliency_maps/ directories contain matching files."
        )

    return samples


def sample_pilot(
    dataset: list[Sample],
    n_per_category: int = 10,
    seed: int = 42,
) -> list[Sample]:
    """Sample a pilot subset with equal representation per category."""
    rng = random.Random(seed)
    by_category: dict[str, list[Sample]] = {}
    for s in dataset:
        by_category.setdefault(s.category, []).append(s)

    pilot = []
    for cat, items in sorted(by_category.items()):
        n = min(n_per_category, len(items))
        pilot.extend(rng.sample(items, n))

    return pilot


def _find_image(directory: Path, image_id: str) -> Path | None:
    """Find an image file by ID, trying common extensions."""
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
        path = directory / f"{image_id}{ext}"
        if path.exists():
            return path
    # Also try without extension matching (glob)
    matches = list(directory.glob(f"{image_id}.*"))
    return matches[0] if matches else None
