"""Quality checks, GitHub credential export and versioned S3 publication."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re

from .constants import F1_THRESHOLD as QUALITY_THRESHOLD


def check_quality(report: dict) -> float:
    score = report.get("f1_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("f1_score must be a JSON number")
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("f1_score must be finite and within [0, 1]")
    if score < QUALITY_THRESHOLD:
        raise ValueError(f"FAILED: f1_score {score:.4f} < {QUALITY_THRESHOLD}. Release blocked.")
    return float(score)


def publish_model(bucket: str, version_id: str, model_path: Path, report_path: Path, client=None) -> dict:
    """Validate before upload; promote one report pointer after versioned objects."""
    if not bucket or not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", version_id):
        raise ValueError("A bucket and a safe, unique version ID are required")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    score = check_quality(report)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    prefix = f"artifacts/releases/{version_id}"
    deployment = {**report, "deployment_id": version_id, "model_key": f"{prefix}/model.joblib", "model_sha256": digest}
    payload = json.dumps(deployment, indent=2, allow_nan=False).encode("utf-8")
    if client is None:
        import boto3
        client = boto3.client("s3")
    client.upload_file(str(model_path), bucket, deployment["model_key"])
    client.put_object(Bucket=bucket, Key=f"{prefix}/report.json", Body=payload, ContentType="application/json")
    detail_path = report_path.with_name("detail.txt")
    if detail_path.exists():
        client.upload_file(str(detail_path), bucket, f"{prefix}/detail.txt")
    # Required canonical model path remains available for the submission.
    client.upload_file(str(model_path), bucket, "artifacts/current/model.joblib")
    # Serving reads this single pointer, then downloads its immutable model key.
    client.put_object(Bucket=bucket, Key="artifacts/current/report.json", Body=payload, ContentType="application/json")
    report_path.with_name("deployment.json").write_bytes(payload + b"\n")
    print(f"Published {version_id}: F1={score:.4f}, SHA256={digest}", flush=True)
    return deployment


def export_github_credentials() -> None:
    """Decode the lab's JSON Secret without exposing credentials in logs."""
    credentials = json.loads(os.environ["STORAGE_CREDENTIALS"])
    names = {
        "aws_access_key_id": "AWS_ACCESS_KEY_ID",
        "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
    }
    if "aws_session_token" in credentials:
        names["aws_session_token"] = "AWS_SESSION_TOKEN"
    lines = []
    for source, destination in names.items():
        value = credentials.get(source)
        if not isinstance(value, str) or not value or any(char in value for char in "\r\n"):
            raise ValueError(f"Invalid or missing credential field: {source}")
        # GitHub masks these values before subsequent steps print any output.
        escaped = value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::add-mask::{escaped}", flush=True)
        lines.append(f"{destination}={value}\n")
    with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as stream:
        stream.writelines(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gate = commands.add_parser("check-quality")
    gate.add_argument("--report", default="outputs/report.json", type=Path)
    commands.add_parser("export-credentials")
    publish = commands.add_parser("publish")
    publish.add_argument("--bucket", default=os.environ.get("ARTIFACT_BUCKET"))
    publish.add_argument("--version-id", required=True)
    publish.add_argument("--model", default="models/model.joblib", type=Path)
    publish.add_argument("--report", default="outputs/report.json", type=Path)
    args = parser.parse_args()
    if args.command == "export-credentials":
        export_github_credentials()
    elif args.command == "check-quality":
        score = check_quality(json.loads(args.report.read_text(encoding="utf-8")))
        print(f"PASSED: f1_score {score:.4f} >= {QUALITY_THRESHOLD}")
    else:
        publish_model(args.bucket, args.version_id, args.model, args.report)


if __name__ == "__main__":
    main()
