"""Calibration and binary classification metrics independent of model frameworks."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)


def fit_calibration(labels, logits, target_recall=0.9):
    labels, logits = np.asarray(labels), np.asarray(logits, dtype=np.float64)
    if set(labels.tolist()) != {0, 1} or not np.isfinite(logits).all():
        raise ValueError("Calibration needs finite logits and both classes.")

    def objective(log_temperature):
        scaled = logits / np.exp(log_temperature)
        return float(np.mean(np.logaddexp(0, scaled) - labels * scaled))

    fit = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    temperature = float(np.exp(fit.x)) if fit.success and objective(fit.x) <= objective(0) else 1.0
    probabilities = expit(logits / temperature)
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    eligible = np.flatnonzero(recall[:-1] >= target_recall)
    best = max(eligible, key=lambda i: (precision[i], thresholds[i]))
    threshold = float(thresholds[best])
    return {
        "method": "temperature_scaling",
        "temperature": temperature,
        "threshold": threshold,
        "threshold_objective": "maximum_precision_subject_to_target_recall_on_calibration",
        "target_recall": target_recall,
        "fit_split": "calibration",
        "samples": len(labels),
        "raw": calculate_metrics(labels, expit(logits), 0.5),
        "calibrated": calculate_metrics(labels, probabilities, threshold),
    }


def calculate_metrics(labels, probabilities, threshold):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(np.int32)
    reliability, ece = [], 0.0
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (probabilities >= lower) & (
            (probabilities < upper) if upper < 1 else (probabilities <= upper)
        )
        if mask.any():
            mean = float(probabilities[mask].mean())
            rate = float(labels[mask].mean())
            count = int(mask.sum())
            ece += count / len(labels) * abs(mean - rate)
            reliability.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": count,
                    "predicted_probability": mean,
                    "observed_positive_rate": rate,
                }
            )
    both = len(np.unique(labels)) == 2
    return {
        "samples": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(
            f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)
        ),
        "recall_vulnerable": float(recall_score(labels, predictions, zero_division=0))
        if np.any(labels == 1)
        else None,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=["no_annotated_vulnerability", "vulnerability_annotated"],
            zero_division=0,
            output_dict=True,
        ),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if both else None,
        "average_precision": float(average_precision_score(labels, probabilities))
        if np.any(labels == 1)
        else None,
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "ece_10_bins": float(ece),
        "reliability_bins": reliability,
        "threshold": float(threshold),
        "single_class_subset": not both,
    }


def grouped_metrics(records, probabilities, threshold):
    result = {}
    for field in ("source", "granularity", "label_confidence"):
        result[field] = {}
        for value in sorted({r[field] for r in records}):
            indices = [i for i, r in enumerate(records) if r[field] == value]
            result[field][value] = calculate_metrics(
                [records[i]["label"] for i in indices], probabilities[indices], threshold
            )
    return result
