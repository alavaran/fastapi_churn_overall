
import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Body
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dataset import (
    prepare_dataset,
    get_class_distribution,
    load_dataset,
)
from model import train_churn_model
from model_storage import save_churn_model, load_churn_model
from schemas import (
    FeatureVectorChurn,
    DatasetRowChurn,
    PredictionResponseChurn,
)


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# Application lifespan
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model_artifact = load_churn_model()

        if app.state.model_artifact is not None:
            logger.info("Churn model loaded successfully")
        else:
            logger.info("No trained churn model found")

    except Exception:
        logger.exception("Failed to load churn model")
        app.state.model_artifact = None

    yield


app = FastAPI(
    title="Churn Prediction API",
    description="ML service for customer churn prediction",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# Root
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "ml churn service is running"
    }


# --------------------------------------------------
# Dataset endpoints
# --------------------------------------------------

@app.get("/dataset/preview")
def dataset_preview(
    limit: int = Query(default=5, ge=1, le=100)
):
    try:
        df = load_dataset()

        return {
            "rows": df.head(limit).to_dict(orient="records")
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Dataset file not found",
        )


@app.get("/dataset/info")
def dataset_info():
    try:
        df = load_dataset()

        if "churn" not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="Target column 'churn' is missing",
            )

        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": df.columns.tolist(),
            "churn_distribution": (
                df["churn"].value_counts().sort_index().to_dict()
            ),
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Dataset file not found",
        )


@app.post(
    "/dataset/validate",
    response_model=DatasetRowChurn,
)
def validate_dataset_row(row: DatasetRowChurn):
    return row


@app.get("/dataset/split-info")
def dataset_split_info():
    try:
        X_train, X_test, y_train, y_test = prepare_dataset()

        return {
            "train": {
                "rows": len(X_train),
                "columns": len(X_train.columns),
                "churn_distribution": get_class_distribution(y_train),
            },
            "test": {
                "rows": len(X_test),
                "columns": len(X_test.columns),
                "churn_distribution": get_class_distribution(y_test),
            },
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Dataset file not found",
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


# --------------------------------------------------
# Model training
# --------------------------------------------------

@app.post("/model/train")
def train_model():
    try:
        X_train, X_test, y_train, y_test = prepare_dataset()

        model = train_churn_model(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(
                float(accuracy_score(y_test, predictions)), 4
            ),
            "precision": round(
                float(precision_score(
                    y_test, predictions, zero_division=0
                )), 4
            ),
            "recall": round(
                float(recall_score(
                    y_test, predictions, zero_division=0
                )), 4
            ),
            "f1": round(
                float(f1_score(
                    y_test, predictions, zero_division=0
                )), 4
            ),
        }

        # ROC-AUC невозможно вычислить, если в тесте один класс
        if y_test.nunique() == 2:
            metrics["roc_auc"] = round(
                float(roc_auc_score(y_test, probabilities)), 4
            )
        else:
            metrics["roc_auc"] = None

        artifact = save_churn_model(model, metrics)

        # Обновляем состояние приложения после сохранения
        app.state.model_artifact = artifact

        return {
            "status": "success",
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "metrics": metrics,
            "trained_at": artifact["trained_at"],
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Dataset file not found",
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except OSError:
        logger.exception("Failed to save churn model")
        raise HTTPException(
            status_code=500,
            detail="Failed to save churn model",
        )


# --------------------------------------------------
# Model status
# --------------------------------------------------

@app.get("/model/status")
def model_status():
    artifact = app.state.model_artifact

    if artifact is None:
        return {
            "trained": False,
            "trained_at": None,
            "metrics": None,
        }

    return {
        "trained": True,
        "trained_at": artifact["trained_at"],
        "metrics": artifact["metrics"],
    }


# --------------------------------------------------
# Churn prediction
# --------------------------------------------------

@app.post(
    "/predict",
    response_model=(
        PredictionResponseChurn
        | list[PredictionResponseChurn]
    ),
)
def predict_churn(
    payload: FeatureVectorChurn | list[FeatureVectorChurn] = Body(
        ...,
        openapi_examples={
            "single_client": {
                "summary": "Один клиент",
                "value": {
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
            },
            "multiple_clients": {
                "summary": "Несколько клиентов",
                "value": [
                    {
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
                    {
                        "monthly_fee": 19.99,
                        "usage_hours": 80.0,
                        "support_requests": 0,
                        "account_age_months": 24,
                        "failed_payments": 0,
                        "region": "asia",
                        "device_type": "desktop",
                        "payment_method": "paypal",
                        "autopay_enabled": 1,
                    },
                ],
            },
        },
    ),
):
    artifact = app.state.model_artifact

    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail="Churn model is not trained yet",
        )

    # Проверяем, один клиент пришёл или список
    is_batch = isinstance(payload, list)

    clients = payload if is_batch else [payload]

    if not clients:
        raise HTTPException(
            status_code=400,
            detail="Client list cannot be empty",
        )

    # Pydantic objects -> dictionaries -> DataFrame
    df = pd.DataFrame([
        client.model_dump()
        for client in clients
    ])

    # Получаем обученный Pipeline
    model = artifact["model"]

    # Предсказываем классы и вероятности
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)

    # Определяем позиции классов в predict_proba
    class_indices = {
        class_label: index
        for index, class_label in enumerate(model.classes_)
    }

    if 0 not in class_indices or 1 not in class_indices:
        raise HTTPException(
            status_code=500,
            detail="Model does not support both churn classes",
        )

    results = []

    for prediction, probability in zip(
        predictions,
        probabilities,
    ):
        results.append(
            PredictionResponseChurn(
                churn=int(prediction),
                probability_no_churn=float(
                    probability[class_indices[0]]
                ),
                probability_churn=float(
                    probability[class_indices[1]]
                ),
            )
        )

    return results if is_batch else results[0]
