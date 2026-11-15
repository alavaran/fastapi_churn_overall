
import pandas as pd
import pytest

from dataset import (
    FEATURES,
    TARGET,
    prepare_dataset,
)
from errors import ServiceError


def test_prepare_dataset_splits_data(
    sample_dataset,
    monkeypatch,
):
    monkeypatch.setattr(
        "dataset.load_dataset",
        lambda: sample_dataset,
    )

    X_train, X_test, y_train, y_test = prepare_dataset()

    assert len(X_train) + len(X_test) == len(sample_dataset)

    assert X_train.columns.tolist() == FEATURES
    assert X_test.columns.tolist() == FEATURES

    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_prepare_dataset_rejects_empty_dataset(monkeypatch):
    monkeypatch.setattr(
        "dataset.load_dataset",
        lambda: pd.DataFrame(),
    )

    with pytest.raises(ServiceError) as exc_info:
        prepare_dataset()

    assert exc_info.value.code == "DATASET_EMPTY"
    assert exc_info.value.status_code == 422