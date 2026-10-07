"""Shared schema for prepared Adult data and serving inputs."""

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "age", "workclass", "education_num", "marital_status", "occupation",
    "relationship", "sex", "capital_gain", "capital_loss", "hours_per_week",
]


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Reject malformed data and return features in the API contract's order."""
    frame = pd.read_csv(path)
    expected = set(FEATURE_NAMES + ["target"])
    if set(frame.columns) != expected or len(frame.columns) != len(expected):
        raise ValueError(f"{path}: expected columns {FEATURE_NAMES + ['target']}")
    if frame.empty:
        raise ValueError(f"{path}: dataset is empty")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        raise ValueError(f"{path}: all columns must be numeric")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: missing or non-finite values are not allowed")
    if not frame["target"].isin([0, 1]).all():
        raise ValueError(f"{path}: target must contain only 0 and 1")
    return frame[FEATURE_NAMES], frame["target"].astype(int)
