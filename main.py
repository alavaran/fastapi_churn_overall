import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Body, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from logging_config import configure_logging
from errors import ServiceError

from dataset import (
    prepare_dataset,
    get_class_distribution,
    load_dataset,
    FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

from model import train_churn_model
from model_storage import save_churn_model, load_churn_model
from training_history import (
    save_training_record,
    get_training_history,
)

from schemas import (
    TrainingConfigChurn,
    FeatureVectorChurn,
    DatasetRowChurn,
    PredictionResponseChurn,
    ErrorResponse,
    TrainingHistoryResponse,
)

# --------------------------------------------------
# Logging
# --------------------------------------------------
configure_logging()
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


@app.exception_handler(ServiceError)
async def service_error_handler(
    request: Request,
    exc: ServiceError,
):
    logger.warning(
        "Service error: %s - %s",
        exc.code,
        exc.message,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.code,
            "message": exc.message,
            "details": exc.details,
        },
    )

# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health_check():
    model_available = app.state.model_artifact is not None

    try:
        dataset = load_dataset()
        dataset_loaded = not dataset.empty

    except Exception:
        logger.exception("Health check: dataset loading failed")
        dataset_loaded = False

    service_healthy = model_available and dataset_loaded

    return {
        "status": "healthy" if service_healthy else "degraded",
        "model_available": model_available,
        "dataset_loaded": dataset_loaded,
    }

@app.exception_handler(HTTPException)
async def http_error_handler(
    request: Request,
    exc: HTTPException,
):
    
    logger.warning(
        "HTTP error: %s - %s",
        exc.status_code,
        exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        headers=exc.headers,
        content={
            "code": "HTTP_ERROR",
            "message": str(exc.detail),
            "details": None,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
):
    logger.warning(
        "Request validation failed: %s",
        exc.errors(),
    )

    details = [
        {
            "field": ".".join(str(part) for part in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors()
    ]

    return JSONResponse(
        status_code=422,
        content={
            "code": "VALIDATION_ERROR",
            "message": "Переданы некорректные данные",
            "details": details,
        },
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(
    request: Request,
    exc: Exception,
):
    logger.exception("Unexpected application error")

    return JSONResponse(
        status_code=500,
        content={
            "code": "INTERNAL_SERVER_ERROR",
            "message": "Внутренняя ошибка сервера",
            "details": None,
        },
    )


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


@app.post(
    "/model/train",
    responses={
        422: {
            "model": ErrorResponse,
            "description": "Ошибка подготовки датасета",
            "content": {
                "application/json": {
                    "example": {
                        "code": "DATASET_EMPTY",
                        "message": "Датасет пуст",
                        "details": None,
                    }
                }
            },
        },
        500: {
            "model": ErrorResponse,
            "description": "Ошибка обучения модели",
            "content": {
                "application/json": {
                    "example": {
                        "code": "TRAINING_FAILED",
                        "message": "Не удалось обучить модель",
                        "details": None,
                    }
                }
            },
        },
    }
)
def train_model(config: TrainingConfigChurn):
    try:
        logger.info(
            "Starting model training: model_type=%s",
            config.model_type,
        )
        # Prepare data, train the model, and evaluate it.
        X_train, X_test, y_train, y_test = prepare_dataset()

        model = train_churn_model(
            X_train,
            y_train,
            config
        )

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1": float(
                f1_score(y_test, predictions, zero_division=0)
            ),
            "roc_auc": float(
                roc_auc_score(y_test, probabilities[:, 1])
            ),
        }

        actual_hyperparameters = (
            model.named_steps["classifier"].get_params(deep=False)
        )

        artifact = save_churn_model(
            model=model,
            metrics=metrics,
            model_type=config.model_type,
            hyperparameters=actual_hyperparameters,
        )

        save_training_record(
            model_type=config.model_type,
            hyperparameters=actual_hyperparameters,
            metrics=metrics,
        )

        app.state.model_artifact = artifact

        logger.info(
            "Model training completed: model_type=%s, metrics=%s",
            config.model_type,
            metrics,
        )

        return {
            "message": "Model trained successfully",
            "model_type": artifact["model_type"],
            "hyperparameters": artifact["hyperparameters"],
            "metrics": artifact["metrics"],
        }

    except ServiceError:
        raise

    except Exception:
        logger.exception("Model training failed")

        raise ServiceError(
            code="TRAINING_FAILED",
            message="Не удалось обучить модель",
            status_code=500,
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
            "model_type": None,
            "hyperparameters": None,
            "trained_at": None,
            "metrics": None,
        }

    return {
        "trained": True,
        "model_type": artifact.get("model_type"),
        "hyperparameters": artifact.get("hyperparameters"),
        "trained_at": artifact.get("trained_at"),
        "metrics": artifact.get("metrics"),
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
                        "monthly_fee": 29.99,
                        "usage_hours": 35.0,
                        "support_requests": 1,
                        "account_age_months": 18,
                        "failed_payments": 0,
                        "region": "america",
                        "device_type": "desktop",
                        "payment_method": "card",
                        "autopay_enabled": 1,
                    },
                ],
            },
        },
    ),
):
    logger.info("Received prediction request")
    artifact = app.state.model_artifact

    if artifact is None:
        raise ServiceError(
            code="MODEL_NOT_TRAINED",
            message="Модель ещё не обучена",
            status_code=503,
        )

    is_batch = isinstance(payload, list)
    clients = payload if is_batch else [payload]

    if not clients:
        raise ServiceError(
            code="EMPTY_CLIENT_LIST",
            message="Список клиентов не должен быть пустым",
            status_code=400,
        )

    df = pd.DataFrame(
        [client.model_dump() for client in clients],
        columns=FEATURES,
    )

    model = artifact["model"]

    try:
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)

        class_indices = {
            class_label: index
            for index, class_label in enumerate(model.classes_)
        }

        if 0 not in class_indices or 1 not in class_indices:
            raise ServiceError(
                code="INVALID_MODEL_CLASSES",
                message="Модель должна поддерживать оба класса churn: 0 и 1",
                status_code=500,
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

    except ServiceError as exc:
        logger.warning(
            "Prediction service error: %s - %s",
            exc.code,
            exc.message,
        )
        raise

    except Exception:
        logger.exception("Prediction failed")

        raise ServiceError(
            code="PREDICTION_FAILED",
            message="Не удалось выполнить предсказание",
            status_code=500,
        )
    
    logger.info(
        "Prediction completed successfully for %s client(s)",
        len(clients),
    )
    return results if is_batch else results[0]



@app.get("/model/schema")
def get_model_schema():
    fields = FeatureVectorChurn.model_fields

    features = []

    for feature_name in FEATURES:
        field = fields[feature_name]

        features.append({
            "name": feature_name,
            "type": field.annotation.__name__,
            "category": (
                "numeric"
                if feature_name in NUMERIC_FEATURES
                else "categorical"
            ),
            "required": field.is_required(),
        })

    return {
        "features": features,
        "target": {
            "name": "churn",
            "type": "int",
            "values": [0, 1],
        },
    }


@app.get(
    "/model/metrics",
    response_model=TrainingHistoryResponse,
)
def get_model_metrics(
    model_type: str | None = Query(
        default=None,
        description="Filter by model type: logreg or random_forest",
    ),
    limit: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Number of latest training records",
    ),
):
    if model_type is not None and model_type not in {
        "logreg",
        "random_forest",
    }:
        raise ServiceError(
            code="INVALID_MODEL_TYPE",
            message="Unsupported model type",
            status_code=422,
            details={
                "allowed_values": ["logreg", "random_forest"]
            },
        )

    records = get_training_history(
        model_type=model_type,
        limit=limit,
    )

    return {
        "total": len(records),
        "records": records,
    }
