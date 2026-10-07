import numpy as np
import pandas as pd
import pytest

from src.schema import FEATURE_NAMES, load_dataset


def _frame():
    frame = pd.DataFrame(np.ones((2, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    frame["target"] = [0, 1]
    return frame


def test_feature_order_is_restored(tmp_path):
    path = tmp_path / "data.csv"
    _frame().iloc[:, ::-1].to_csv(path, index=False)
    features, targets = load_dataset(path)
    assert list(features.columns) == FEATURE_NAMES
    assert targets.tolist() == [0, 1]


@pytest.mark.parametrize("problem", ["missing_column", "extra_column", "non_numeric", "infinite", "bad_target", "empty"])
def test_invalid_datasets_are_rejected(tmp_path, problem):
    frame = _frame()
    if problem == "missing_column":
        frame = frame.drop(columns=["age"])
    elif problem == "extra_column":
        frame["unexpected"] = 1
    elif problem == "non_numeric":
        frame["age"] = ["invalid", "value"]
    elif problem == "infinite":
        frame.loc[0, "age"] = np.inf
    elif problem == "bad_target":
        frame.loc[0, "target"] = 2
    elif problem == "empty":
        frame = frame.iloc[:0]
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_dataset(path)
