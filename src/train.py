"""Train Adult income models and record reproducible MLflow experiments."""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier
import yaml

# Keep both the lab's script entry point and `python -m src.train` working.
if __package__:
    from .evaluation import F1_THRESHOLD, distribution_report, evaluate
    from .schema import FEATURE_NAMES, load_dataset
else:
    from evaluation import F1_THRESHOLD, distribution_report, evaluate
    from schema import FEATURE_NAMES, load_dataset

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_tracking() -> None:
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_tracking_uri(uri)
    name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "adult-income")
    # Remote servers manage artifact locations. Local tracking uses an absolute
    # artifact path so the UI can retrieve artifacts regardless of working dir.
    if uri.startswith(("sqlite:", "file:")):
        if mlflow.get_experiment_by_name(name) is None:
            root = Path(os.environ.get("MLFLOW_ARTIFACT_ROOT", str(PROJECT_ROOT / "mlartifacts"))).resolve()
            root.mkdir(parents=True, exist_ok=True)
            mlflow.create_experiment(name, artifact_location=root.as_uri())
    mlflow.set_experiment(name)


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
    *,
    output_dir: str | Path = "outputs",
    model_dir: str | Path = "models",
    run_name: str | None = None,
) -> float:
    """Return default-rule holdout F1 and log exploratory bonus metrics separately."""
    allowed = {"n_estimators", "learning_rate", "max_depth"}
    if not isinstance(params, dict) or set(params) != allowed:
        raise ValueError(f"Model parameters must be exactly {sorted(allowed)}")
    features, targets = load_dataset(data_path)
    eval_features, eval_targets = load_dataset(eval_path)
    if targets.nunique() != 2:
        raise ValueError("Training data must contain both target classes")

    # Bonus 5 runs before fitting. A distribution shift warns but does not block.
    distribution = distribution_report(targets)
    LOGGER.info("Training samples: %d | Positive rate: %.2f%%", len(targets), 100 * distribution["positive_rate"])
    if distribution["distribution_warning"]:
        LOGGER.warning(
            "DATA DISTRIBUTION WARNING: positive rate %.2f%% differs from 24.8%% "
            "by %.2f percentage points (> 5).",
            100 * distribution["positive_rate"], 100 * distribution["positive_rate_deviation"],
        )

    _configure_tracking()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({**params, "random_state": RANDOM_STATE, "decision_threshold": 0.5})
        model = GradientBoostingClassifier(**params, random_state=RANDOM_STATE)
        model.fit(features, targets)
        predictions = model.predict(eval_features)
        probabilities = model.predict_proba(eval_features)[:, 1]
        report, detail = evaluate(eval_targets, predictions, probabilities)
        report.update(distribution)
        report.update({
            "params": dict(params), "random_state": RANDOM_STATE,
            "feature_names": FEATURE_NAMES,
            "train_samples": len(targets), "eval_samples": len(eval_targets),
            "data_path": str(data_path), "eval_path": str(eval_path),
            "train_sha256": _sha256(data_path), "holdout_sha256": _sha256(eval_path),
            "mlflow_run_id": run.info.run_id, "git_commit": os.environ.get("GITHUB_SHA"),
            "quality_gate_threshold": F1_THRESHOLD,
            "quality_gate_passed": report["f1_score"] >= F1_THRESHOLD,
        })
        output_path, model_path = Path(output_dir), Path(model_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path / "model.joblib")
        with (output_path / "report.json").open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        (output_path / "detail.txt").write_text(detail, encoding="utf-8")

        mlflow.log_metrics({key: report[key] for key in (
            "f1_score", "accuracy", "precision", "recall", "best_threshold",
            "best_threshold_f1_score", "f1_score_at_0_5", "positive_rate",
        )})
        mlflow.log_metrics({"train_samples": len(targets), "eval_samples": len(eval_targets)})
        mlflow.set_tags({
            "train_sha256": report["train_sha256"], "holdout_sha256": report["holdout_sha256"],
            "threshold_search_split": "holdout-exploratory",
        })
        mlflow.sklearn.log_model(
            model, "model", signature=infer_signature(eval_features, predictions),
            input_example=eval_features.head(3),
        )
        mlflow.log_artifact(str(output_path / "report.json"), "evaluation")
        mlflow.log_artifact(str(output_path / "detail.txt"), "evaluation")
        LOGGER.info("MLflow run: %s", run.info.run_id)
        LOGGER.info("F1: %.4f | Accuracy: %.4f", report["f1_score"], report["accuracy"])
        LOGGER.info(
            "Exploratory best threshold: %.2f | F1: %.4f | Default 0.5 F1: %.4f",
            report["best_threshold"], report["best_threshold_f1_score"], report["f1_score_at_0_5"],
        )
        LOGGER.info("Quality gate >= %.2f: %s", F1_THRESHOLD, "PASS" if report["quality_gate_passed"] else "FAIL")
        LOGGER.info("Report: %s | Model: %s", output_path / "report.json", model_path / "model.joblib")
        print(detail, flush=True)
        return float(report["f1_score"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--data-path", default="data/train_batch1.csv")
    parser.add_argument("--eval-path", default="data/holdout.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--run-name")
    parser.add_argument("--n-estimators", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-depth", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    with open(args.params, encoding="utf-8") as stream:
        params = yaml.safe_load(stream)
    if not isinstance(params, dict):
        parser.error("The params YAML must contain a mapping")
    for key in ("n_estimators", "learning_rate", "max_depth"):
        value = getattr(args, key)
        if value is not None:
            params[key] = value
    train(
        params, args.data_path, args.eval_path,
        output_dir=args.output_dir, model_dir=args.model_dir, run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
