import numpy as np
from scipy.ndimage import gaussian_filter


def generate_saliency_map(
    gaze_points: list,
    width: int = 1920,
    height: int = 1200,
    sigma: float = 40.0,
) -> np.ndarray:
    """Generate a saliency heatmap from VLM-predicted gaze points.

    Each gaze point is placed on a canvas and weighted by its intensity,
    then the entire map is smoothed with a Gaussian filter.

    Args:
        gaze_points: List of GazePoint (x, y, intensity) with normalized coords.
        width: Output map width in pixels.
        height: Output map height in pixels.
        sigma: Gaussian kernel standard deviation (pixels).

    Returns:
        Normalized saliency map as float64 array of shape (height, width), range [0, 1].
    """
    saliency = np.zeros((height, width), dtype=np.float64)

    for point in gaze_points:
        px = int(round(point.x * (width - 1)))
        py = int(round(point.y * (height - 1)))
        px = max(0, min(width - 1, px))
        py = max(0, min(height - 1, py))
        saliency[py, px] += point.intensity

    saliency = gaussian_filter(saliency, sigma=sigma)

    max_val = saliency.max()
    if max_val > 0:
        saliency /= max_val

    return saliency
