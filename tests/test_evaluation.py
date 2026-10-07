import numpy as np
import pytest

from src.evaluation import distribution_report, evaluate


def test_threshold_search_does_not_replace_default_f1():
    targets = np.array([1, 1, 0, 0])
    probabilities = np.array([0.3, 0.4, 0.1, 0.2])
    default_predictions = (probabilities > 0.5).astype(int)
    report, detail = evaluate(targets, default_predictions, probabilities)
    assert report["f1_score"] == 0.0
    assert report["accuracy"] == 0.5
    assert report["f1_score_at_0_5"] == 0.0
    assert report["best_threshold_f1_score"] == 1.0
    assert report["best_threshold"] == 0.25
    assert len(report["threshold_curve"]) == 17
    assert report["threshold_curve"][0]["threshold"] == 0.1
    assert report["threshold_curve"][-1]["threshold"] == 0.9
    assert report["confusion_matrix"] == [[2, 0], [2, 0]]
    assert "thu_nhap_thap" in detail and "thu_nhap_cao" in detail
    assert "precision" in detail and "recall" in detail


@pytest.mark.parametrize("positive_count, warning", [
    (248, False), (298, False), (198, False), (299, True), (197, True),
])
def test_distribution_warning_is_strictly_more_than_five_percentage_points(positive_count, warning):
    targets = np.zeros(1000, dtype=int)
    targets[:positive_count] = 1
    report = distribution_report(targets)
    assert report["positive_rate"] == pytest.approx(positive_count / 1000)
    assert report["distribution_warning"] is warning
