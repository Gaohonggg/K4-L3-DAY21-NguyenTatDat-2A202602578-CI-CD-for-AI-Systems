"""Training integration tests with isolated data, outputs and MLflow storage."""

import json

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from src.schema import FEATURE_NAMES
from src.train import train


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("training")
    rng = np.random.default_rng(0)
    features = rng.random((200, len(FEATURE_NAMES)))
    frame = pd.DataFrame(features, columns=FEATURE_NAMES)
    # A learnable binary target tests actual training, without a quality-gate
    # assertion on randomly generated data.
    frame["target"] = (features[:, 0] + features[:, 1] > 1.0).astype(int)
    train_path, eval_path = root / "train.csv", root / "holdout.csv"
    frame.iloc[:160].to_csv(train_path, index=False)
    frame.iloc[160:].to_csv(eval_path, index=False)
    original_uri = mlflow.get_tracking_uri()
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("MLFLOW_TRACKING_URI", (root / "tracking").as_uri())
            patch.setenv("MLFLOW_ARTIFACT_ROOT", str(root / "artifacts"))
            patch.setenv("MLFLOW_EXPERIMENT_NAME", "test-adult-income")
            score = train(
                {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
                data_path=str(train_path), eval_path=str(eval_path),
                output_dir=root / "outputs", model_dir=root / "models", run_name="test-run",
            )
            client = mlflow.tracking.MlflowClient()
            report = json.loads((root / "outputs" / "report.json").read_text())
            run = client.get_run(report["mlflow_run_id"])
    finally:
        mlflow.set_tracking_uri(original_uri)
    return root, score, frame.iloc[160:], run


def test_train_returns_float(trained_run):
    _, score, _, _ = trained_run
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_report_file_created(trained_run):
    root, score, _, run = trained_run
    report = json.loads((root / "outputs" / "report.json").read_text())
    assert report["f1_score"] == score
    assert 0.0 <= report["accuracy"] <= 1.0
    assert report["train_samples"] == 160 and report["eval_samples"] == 40
    assert report["feature_names"] == FEATURE_NAMES
    assert report["decision_threshold"] == 0.5
    assert run.info.status == "FINISHED"
    assert run.data.metrics["f1_score"] == score
    assert run.data.metrics["accuracy"] == report["accuracy"]
    detail = (root / "outputs" / "detail.txt").read_text()
    assert "Confusion matrix" in detail
    assert "thu_nhap_cao" in detail


def test_model_file_created(trained_run):
    root, score, evaluation, _ = trained_run
    model = joblib.load(root / "models" / "model.joblib")
    predictions = model.predict(evaluation[FEATURE_NAMES])
    report = json.loads((root / "outputs" / "report.json").read_text())
    assert f1_score(evaluation["target"], predictions, zero_division=0) == score
    assert accuracy_score(evaluation["target"], predictions) == report["accuracy"]
