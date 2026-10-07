import hashlib
import json

import pytest

from src.artifacts import check_quality, export_github_credentials, publish_model


@pytest.mark.parametrize("score", [0.65, 0.7149, 1.0])
def test_quality_gate_accepts_passing_f1(score):
    assert check_quality({"f1_score": score}) == score


@pytest.mark.parametrize("score", [0.6499, -1, 1.1, float("nan"), float("inf"), "0.9", None, True])
def test_quality_gate_rejects_low_or_invalid_f1(score):
    with pytest.raises(ValueError):
        check_quality({"f1_score": score})


class RecordingStorage:
    def __init__(self):
        self.events = []

    def upload_file(self, filename, bucket, key):
        self.events.append(("upload", bucket, key))

    def put_object(self, **kwargs):
        self.events.append(("put", kwargs["Bucket"], kwargs["Key"]))
        if kwargs["Key"] == "artifacts/current/report.json":
            self.current_report = json.loads(kwargs["Body"])


def test_rejected_model_never_changes_cloud_storage(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"f1_score": 0.60}))
    client = RecordingStorage()
    with pytest.raises(ValueError, match="Release blocked"):
        publish_model("test-bucket", "run-1", tmp_path / "missing-model.joblib", report, client)
    assert client.events == []


def test_publication_promotes_manifest_after_versioned_model(tmp_path):
    model, report = tmp_path / "model.joblib", tmp_path / "report.json"
    model.write_bytes(b"test-artifact")
    report.write_text(json.dumps({"f1_score": 0.71, "accuracy": 0.87}))
    client = RecordingStorage()
    deployment = publish_model("test-bucket", "run-1", model, report, client)
    assert client.events[0] == ("upload", "test-bucket", "artifacts/releases/run-1/model.joblib")
    assert client.events[-1] == ("put", "test-bucket", "artifacts/current/report.json")
    assert client.current_report["model_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert deployment["deployment_id"] == "run-1"
    assert json.loads((tmp_path / "deployment.json").read_text()) == deployment


def test_credentials_export_supports_session_tokens(tmp_path, monkeypatch, capsys):
    environment = tmp_path / "github-env"
    monkeypatch.setenv("GITHUB_ENV", str(environment))
    monkeypatch.setenv("STORAGE_CREDENTIALS", json.dumps({
        "aws_access_key_id": "test-id",
        "aws_secret_access_key": "test-secret",
        "aws_session_token": "test-token",
    }))
    export_github_credentials()
    assert environment.read_text().splitlines() == [
        "AWS_ACCESS_KEY_ID=test-id", "AWS_SECRET_ACCESS_KEY=test-secret", "AWS_SESSION_TOKEN=test-token",
    ]
    assert "::add-mask::test-secret" in capsys.readouterr().out


def test_credentials_with_newlines_are_rejected(tmp_path, monkeypatch, capsys):
    environment = tmp_path / "github-env"
    monkeypatch.setenv("GITHUB_ENV", str(environment))
    monkeypatch.setenv("STORAGE_CREDENTIALS", json.dumps({
        "aws_access_key_id": "test-id", "aws_secret_access_key": "value\nINJECTED=1",
    }))
    with pytest.raises(ValueError):
        export_github_credentials()
    assert not environment.exists()
