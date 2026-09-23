"""Calibration and comparisons with thresholds fixed outside the test set."""
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_curve

from .experiment_v3 import calculate_metrics, fit_calibration


def fit_operating_points(labels, logits, config):
    result = fit_calibration(labels, logits, config["target_recall"])
    negative = np.sort(expit(np.asarray(logits)[np.asarray(labels) == 0] / result["temperature"]))
    allowed = int(np.floor(config["target_fpr"] * len(negative)))
    cutoff = float(np.nextafter(negative[len(negative) - allowed - 1], np.inf))
    result["fpr_threshold"] = cutoff
    result["target_fpr"] = config["target_fpr"]
    result["fpr_threshold_rule"] = "conservative_negative_quantile_fitted_on_calibration"
    return result


def measured(labels, probabilities, threshold):
    result = calculate_metrics(labels, probabilities, threshold)
    tn, fp, fn, tp = np.asarray(result["confusion_matrix"]).ravel()
    result["false_positive_rate"] = float(fp / (tn + fp)) if tn + fp else None
    result["precision_vulnerable"] = float(tp / (tp + fp)) if tp + fp else 0.0
    return result


def evaluate_logits(labels, logits, calibration):
    probabilities = expit(np.asarray(logits, dtype=np.float64) / calibration["temperature"])
    point = None
    if len(np.unique(labels)) == 2:
        fpr, recall, _ = roc_curve(labels, probabilities)
        point = float(recall[fpr <= 0.1].max())
    return {
        "raw": measured(labels, expit(logits), 0.5),
        "calibrated": measured(labels, probabilities, calibration["threshold"]),
        "fpr_operating_point": measured(labels, probabilities, calibration["fpr_threshold"]),
        "roc_recall_at_fpr_0_10": point,
        "roc_point_note": "descriptive_test_curve_not_a_deployable_test_fitted_threshold",
    }


def bootstrap_delta(labels, groups, probabilities_a, thresholds_a, probabilities_b, thresholds_b, repetitions=1000, seed=42):
    """Paired cluster bootstrap of mean per-seed F1 and recall differences."""
    labels = np.asarray(labels)
    groups, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    outcomes = []

    def scores(probs, cutoffs, weights):
        values = []
        for probability, cutoff in zip(probs, cutoffs):
            pred = probability >= cutoff
            tp = weights[(labels == 1) & pred].sum()
            tn = weights[(labels == 0) & ~pred].sum()
            fp = weights[(labels == 0) & pred].sum()
            fn = weights[(labels == 1) & ~pred].sum()
            f1 = 0.5 * (2 * tp / max(2 * tp + fp + fn, 1) + 2 * tn / max(2 * tn + fp + fn, 1))
            values.append((f1, tp / max(tp + fn, 1)))
        return np.mean(values, axis=0)

    for _ in range(repetitions):
        weights = np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))[inverse]
        outcomes.append(scores(probabilities_a, thresholds_a, weights) - scores(probabilities_b, thresholds_b, weights))
    intervals = np.percentile(outcomes, [2.5, 97.5], axis=0)
    point = scores(probabilities_a, thresholds_a, np.ones(len(labels))) - scores(probabilities_b, thresholds_b, np.ones(len(labels)))
    return {name: {"difference": float(point[i]), "ci95": intervals[:, i].tolist()} for i, name in enumerate(("f1_macro", "recall_vulnerable"))}
