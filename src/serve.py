"""Serve a verified Adult income model from a versioned S3 release."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Callable

import boto3
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, FiniteFloat, field_validator

from .artifacts import check_quality
from .schema import FEATURE_NAMES


@dataclass(frozen=True)
class ModelRuntime:
    model: object
    deployment_id: str
    model_sha256: str


def load_runtime() -> ModelRuntime:
    """Read one deployment manifest and atomically cache its verified model."""
    bucket = os.environ.get("ARTIFACT_BUCKET")
    if not bucket:
        raise ValueError("ARTIFACT_BUCKET is required at startup")
    client = boto3.client("s3")
    response = client.get_object(Bucket=bucket, Key="artifacts/current/report.json")
    body = response["Body"]
    try:
        report = json.loads(body.read())
    finally:
        body.close()
    check_quality(report)
    key, digest = report.get("model_key", ""), report.get("model_sha256", "")
    deployment_id = report.get("deployment_id", "")
    if not isinstance(deployment_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", deployment_id):
        raise ValueError("Invalid deployment ID")
    if key != f"artifacts/releases/{deployment_id}/model.joblib":
        raise ValueError("Model key does not match the versioned release")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid model SHA256")
    path = Path(os.environ.get("MODEL_PATH", "~/models/model.joblib")).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".joblib", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        client.download_file(bucket, key, str(temporary))
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
            raise ValueError("Downloaded model checksum does not match the manifest")
        model = joblib.load(temporary)
        if getattr(model, "n_features_in_", None) != len(FEATURE_NAMES):
            raise ValueError("Model must accept exactly ten features")
        if list(getattr(model, "feature_names_in_", [])) != FEATURE_NAMES:
            raise ValueError("Model feature order does not match the API contract")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return ModelRuntime(model, deployment_id, digest)


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: list[FiniteFloat]

    @field_validator("features", mode="before")
    @classmethod
    def numeric_features(cls, values):
        if isinstance(values, list) and any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in values
        ):
            raise ValueError("Features must contain numbers, not strings or booleans")
        return values


def create_app(loader: Callable[[], ModelRuntime] = load_runtime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.runtime = loader()
        yield
        application.state.runtime = None

    application = FastAPI(title="Adult Income API", lifespan=lifespan)
    application.state.runtime = None

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        # Omit raw input/context, which may contain non-JSON finite values.
        return JSONResponse(status_code=422, content={"detail": [
            {"loc": item["loc"], "msg": item["msg"], "type": item["type"]}
            for item in error.errors()
        ]})

    def runtime() -> ModelRuntime:
        loaded = application.state.runtime
        if loaded is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        return loaded

    @application.get("/healthz")
    def healthz():
        runtime()
        return {"status": "ok"}

    @application.get("/version")
    def version():
        loaded = runtime()
        return {"deployment_id": loaded.deployment_id, "model_sha256": loaded.model_sha256}

    @application.post("/score")
    def score(request: ScoreRequest, response: Response):
        if len(request.features) != len(FEATURE_NAMES):
            raise HTTPException(status_code=400, detail="Expected 10 features (adult income)")
        loaded = runtime()
        features = pd.DataFrame([request.features], columns=FEATURE_NAMES)
        prediction = int(loaded.model.predict(features)[0])
        if prediction not in (0, 1):
            raise HTTPException(status_code=500, detail="Model returned an invalid class")
        response.headers["X-Model-Version"] = loaded.deployment_id
        return {"prediction": prediction, "label": "thu_nhap_cao" if prediction else "thu_nhap_thap"}

    return application


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
