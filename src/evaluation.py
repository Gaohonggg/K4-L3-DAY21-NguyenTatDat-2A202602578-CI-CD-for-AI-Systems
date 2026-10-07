"""Holdout evaluation and threshold/reporting/distribution bonuses."""

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

if __package__:
    from .constants import F1_THRESHOLD
else:
    from constants import F1_THRESHOLD
REFERENCE_POSITIVE_RATE = 0.248
POSITIVE_RATE_TOLERANCE = 0.05


def distribution_report(targets) -> dict:
    """Detect an absolute shift of more than five percentage points."""
    rate = float(np.mean(targets))
    deviation = abs(rate - REFERENCE_POSITIVE_RATE)
    warning = deviation > POSITIVE_RATE_TOLERANCE and not math.isclose(
        deviation, POSITIVE_RATE_TOLERANCE, rel_tol=0.0, abs_tol=1e-12,
    )
    return {
        "positive_rate": rate,
        "reference_positive_rate": REFERENCE_POSITIVE_RATE,
        "positive_rate_deviation": deviation,
        "distribution_warning": bool(warning),
    }


def evaluate(targets, predictions, positive_probabilities) -> tuple[dict, str]:
    """Keep default-rule metrics separate from exploratory holdout tuning."""
    probabilities = np.asarray(positive_probabilities)
    curve = []
    for index in range(17):
        threshold = round(0.1 + index * 0.05, 2)
        threshold_predictions = (probabilities > threshold).astype(int)
        curve.append({
            "threshold": threshold,
            "f1_score": float(f1_score(targets, threshold_predictions, zero_division=0)),
        })
    # Ties prefer the default threshold's neighborhood, then the lower value.
    best = max(curve, key=lambda row: (
        row["f1_score"], -abs(row["threshold"] - 0.5), -row["threshold"],
    ))
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    report = {
        "f1_score": float(f1_score(targets, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "decision_threshold": 0.5,
        "best_threshold": best["threshold"],
        "best_threshold_f1_score": best["f1_score"],
        "f1_score_at_0_5": next(row["f1_score"] for row in curve if row["threshold"] == 0.5),
        "threshold_search_split": "holdout (exploratory; not an independent estimate)",
        "threshold_curve": curve,
        "confusion_matrix": matrix.tolist(),
    }
    detail = (
        "Evaluation split: holdout\nDecision rule: model.predict (default threshold)\n"
        "Confusion matrix: rows=true, columns=predicted; class order=[0, 1]\n"
        f"{np.array2string(matrix)}\n\n"
        + classification_report(
            targets, predictions, labels=[0, 1],
            target_names=["thu_nhap_thap", "thu_nhap_cao"],
            digits=4, zero_division=0,
        )
        + "\nThreshold sweep uses observed holdout labels. Its best F1 is\n"
        "exploratory; deployed model and quality gate keep the default rule.\n"
    )
    return report, detail
