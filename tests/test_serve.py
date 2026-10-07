import hashlib
from io import BytesIO
import json

from fastapi.testclient import TestClient
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src import serve
from src.schema import FEATURE_NAMES


class Predictor:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, features):
        assert list(features.columns) == FEATURE_NAMES
        assert features.shape == (1, 10)
        return np.array([self.prediction])


def _application(prediction=1):
    return serve.create_app(lambda: serve.ModelRuntime(Predictor(prediction), "test-version", "0" * 64))


@pytest.mark.parametrize("prediction,label", [(0, "thu_nhap_thap"), (1, "thu_nhap_cao")])
def test_score_preserves_contract_and_returns_model_version(prediction, label):
    with TestClient(_application(prediction)) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/version").json()["deployment_id"] == "test-version"
        response = client.post("/score", json={"features": [60, 2, 5, 2, 4, 0, 1, 0, 0, 45]})
        assert response.status_code == 200
        assert response.json() == {"prediction": prediction, "label": label}
        assert response.headers["X-Model-Version"] == "test-version"


@pytest.mark.parametrize("length", [0, 9, 11])
def test_wrong_feature_count_returns_400(length):
    with TestClient(_application()) as client:
        assert client.post("/score", json={"features": [1] * length}).status_code == 400


@pytest.mark.parametrize("value", [True, "NaN", "1", None])
def test_non_numeric_features_return_422(value):
    with TestClient(_application()) as client:
        assert client.post("/score", json={"features": [value] + [1] * 9}).status_code == 422


def test_non_finite_json_input_returns_422():
    with TestClient(_application()) as client:
        response = client.post(
            "/score", content='{"features":[NaN,1,1,1,1,1,1,1,1,1]}',
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


def test_health_is_unavailable_without_loaded_model():
    # A client outside its context does not run lifespan startup.
    client = TestClient(_application())
    assert client.get("/healthz").status_code == 503


class DownloadStorage:
    def __init__(self, model, manifest):
        self.model, self.manifest = model, manifest

    def get_object(self, **kwargs):
        assert kwargs["Key"] == "artifacts/current/report.json"
        return {"Body": BytesIO(json.dumps(self.manifest).encode())}

    def download_file(self, bucket, key, filename):
        assert key == "artifacts/releases/test-release/model.joblib"
        with open(filename, "wb") as stream:
            stream.write(self.model)


@pytest.fixture
def storage_setup(tmp_path, monkeypatch):
    frame = pd.DataFrame(np.ones((4, 10)), columns=FEATURE_NAMES)
    frame["age"] = [20, 30, 60, 70]
    model = DecisionTreeClassifier(random_state=42).fit(frame, [0, 0, 1, 1])
    source = tmp_path / "source.joblib"
    joblib.dump(model, source)
    payload = source.read_bytes()
    manifest = {
        "f1_score": 0.71, "deployment_id": "test-release",
        "model_key": "artifacts/releases/test-release/model.joblib",
        "model_sha256": hashlib.sha256(payload).hexdigest(),
    }
    storage = DownloadStorage(payload, manifest)
    destination = tmp_path / "models" / "model.joblib"
    monkeypatch.setenv("ARTIFACT_BUCKET", "test-bucket")
    monkeypatch.setenv("MODEL_PATH", str(destination))
    monkeypatch.setattr(serve.boto3, "client", lambda service: storage)
    return storage, destination


def test_startup_downloads_and_verifies_versioned_model(storage_setup):
    _, destination = storage_setup
    runtime = serve.load_runtime()
    assert destination.exists()
    assert runtime.deployment_id == "test-release"
    assert list(runtime.model.feature_names_in_) == FEATURE_NAMES


def test_checksum_failure_preserves_existing_model(storage_setup):
    storage, destination = storage_setup
    destination.parent.mkdir()
    destination.write_bytes(b"existing-model")
    storage.manifest["model_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        serve.load_runtime()
    assert destination.read_bytes() == b"existing-model"
    assert list(destination.parent.iterdir()) == [destination]
