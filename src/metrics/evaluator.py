import numpy as np
from scipy.stats import pearsonr


def compute_cc(pred: np.ndarray, gt: np.ndarray) -> float:
    """Correlation Coefficient between predicted and ground truth saliency maps."""
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()

    if pred_flat.std() == 0 or gt_flat.std() == 0:
        return 0.0

    cc, _ = pearsonr(pred_flat, gt_flat)
    return float(cc)


def compute_sim(pred: np.ndarray, gt: np.ndarray) -> float:
    """Similarity metric (histogram intersection of normalized distributions)."""
    # Normalize to probability distributions
    pred_norm = pred / (pred.sum() + 1e-10)
    gt_norm = gt / (gt.sum() + 1e-10)

    return float(np.minimum(pred_norm, gt_norm).sum())


def compute_kl(pred: np.ndarray, gt: np.ndarray) -> float:
    """KL Divergence from ground truth to predicted saliency map.

    KL(gt || pred) - measures how much information is lost when pred
    is used to approximate gt.
    """
    eps = 1e-10
    pred_norm = pred / (pred.sum() + eps) + eps
    gt_norm = gt / (gt.sum() + eps) + eps

    return float(np.sum(gt_norm * np.log(gt_norm / pred_norm)))


def compute_nss(pred: np.ndarray, fixation_map: np.ndarray) -> float:
    """Normalized Scanpath Saliency.

    Args:
        pred: Predicted saliency map.
        fixation_map: Binary fixation map (1 at fixation locations, 0 elsewhere).
    """
    if pred.std() == 0:
        return 0.0

    pred_normalized = (pred - pred.mean()) / pred.std()
    fixation_locations = fixation_map > 0

    if not fixation_locations.any():
        return 0.0

    return float(pred_normalized[fixation_locations].mean())


def compute_auc(pred: np.ndarray, fixation_map: np.ndarray) -> float:
    """Area Under the ROC Curve (Judd variant).

    Args:
        pred: Predicted saliency map.
        fixation_map: Binary fixation map.
    """
    fixation_locations = fixation_map.flatten() > 0
    pred_flat = pred.flatten()

    n_fixations = fixation_locations.sum()
    if n_fixations == 0 or n_fixations == len(pred_flat):
        return 0.5

    pos_values = pred_flat[fixation_locations]
    neg_values = pred_flat[~fixation_locations]

    # Sort thresholds from predicted values at fixation points
    thresholds = np.sort(pos_values)[::-1]

    tp_rate = np.zeros(len(thresholds) + 2)
    fp_rate = np.zeros(len(thresholds) + 2)
    tp_rate[0], fp_rate[0] = 0.0, 0.0
    tp_rate[-1], fp_rate[-1] = 1.0, 1.0

    n_neg = len(neg_values)
    for i, thresh in enumerate(thresholds):
        tp_rate[i + 1] = (i + 1) / n_fixations
        fp_rate[i + 1] = (neg_values >= thresh).sum() / n_neg

    return float(np.trapz(tp_rate, fp_rate))


def evaluate_all(
    pred: np.ndarray,
    gt: np.ndarray,
    fixation_map: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all saliency metrics.

    Args:
        pred: Predicted saliency map.
        gt: Ground truth saliency map (continuous).
        fixation_map: Binary fixation map. If None, NSS and AUC are skipped.

    Returns:
        Dict with metric names as keys and scores as values.
    """
    results = {
        "CC": compute_cc(pred, gt),
        "SIM": compute_sim(pred, gt),
        "KL": compute_kl(pred, gt),
    }

    if fixation_map is not None:
        results["NSS"] = compute_nss(pred, fixation_map)
        results["AUC"] = compute_auc(pred, fixation_map)

    return results
