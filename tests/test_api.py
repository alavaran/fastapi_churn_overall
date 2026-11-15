
from fastapi.testclient import TestClient

import pandas as pd
import main

from main import app


def test_predict_without_trained_model():

    with TestClient(app) as client:
        app.state.model_artifact = None
        response = client.post(
            "/predict",
            json={
                "monthly_fee": 49.99,
                "usage_hours": 12.5,
                "support_requests": 4,
                "account_age_months": 3,
                "failed_payments": 2,
                "region": "europe",
                "device_type": "mobile",
                "payment_method": "card",
                "autopay_enabled": 0,
            },
        )

    assert response.status_code == 503
    assert response.json()["code"] == "MODEL_NOT_TRAINED"


def test_predict_invalid_data():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "monthly_fee": "invalid",
                "usage_hours": 12.5,
                "support_requests": 4,
                "account_age_months": 3,
                "failed_payments": 2,
                "region": "europe",
                "device_type": "mobile",
                "payment_method": "card",
                "autopay_enabled": 0,
            },
        )

    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


def test_full_training_and_prediction_flow(
    sample_dataset,
    monkeypatch,
    tmp_path,
):
    # Подменяем загрузку датасета тестовыми данными
    monkeypatch.setattr(
        main,
        "prepare_dataset",
        lambda: (
            sample_dataset.drop(columns=["churn"]),
            sample_dataset.drop(columns=["churn"]),
            sample_dataset["churn"],
            sample_dataset["churn"],
        ),
    )

    # Используем отдельный файл истории обучений
    monkeypatch.setattr(
        "training_history.HISTORY_PATH",
        tmp_path / "training_history.json",
    )

    # Используем отдельный файл модели
    monkeypatch.setattr(
        "model_storage.MODEL_PATH",
        tmp_path / "churn_model.joblib",
    )

    with TestClient(app) as client:
        train_response = client.post(
            "/model/train",
            json={
                "model_type": "logreg",
                "hyperparameters": {},
            },
        )

        assert train_response.status_code == 200

        status_response = client.get("/model/status")
        assert status_response.status_code == 200
        assert status_response.json()["trained"] is True

        prediction_response = client.post(
            "/predict",
            json={
                "monthly_fee": 49.99,
                "usage_hours": 12.5,
                "support_requests": 4,
                "account_age_months": 3,
                "failed_payments": 2,
                "region": "europe",
                "device_type": "mobile",
                "payment_method": "card",
                "autopay_enabled": 0,
            },
        )

        assert prediction_response.status_code == 200

        result = prediction_response.json()

        assert result["churn"] in [0, 1]
        assert 0 <= result["probability_churn"] <= 1
        assert 0 <= result["probability_no_churn"] <= 1
