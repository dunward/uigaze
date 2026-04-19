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
        ├── images/                        # UI screenshots (1920x1200)
        ├── saliency_maps/
        │   ├── heatmaps_{1s,3s,7s}/       # continuous heatmaps
        │   ├── fixmaps_{1s,3s,7s}/        # binary fixation maps
        │   └── overlay_heatmaps_{1s,3s,7s}/
        ├── scanpaths/
        ├── eyetracker_logs/
        └── image_types.csv                # semicolon-separated metadata

    Args:
        data_dir: Path to the UEyes data directory.
        duration: Heatmap duration - "1s", "3s", or "7s".
        categories: Filter by category. None = all categories.

    Returns:
        List of Sample objects with matched image/heatmap pairs.
    """
    data_dir = Path(data_dir)
    info_path = data_dir / "image_types.csv"

    if not info_path.exists():
        raise FileNotFoundError(
            f"image_types.csv not found at {info_path}. "
            "Please download the UEyes dataset and place it in the data/ directory."
        )

    info = pd.read_csv(info_path, sep=";")

    if categories:
        info = info[info["Category"].isin(categories)]

    images_dir = data_dir / "images"
    heatmaps_dir = data_dir / "saliency_maps" / f"heatmaps_{duration}"

    samples = []
    for _, row in info.iterrows():
        image_name = str(row["Image Name"])
        image_id = Path(image_name).stem
        category = row["Category"]

        image_path = images_dir / image_name
        heatmap_path = heatmaps_dir / image_name

        if image_path.exists() and heatmap_path.exists():
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
