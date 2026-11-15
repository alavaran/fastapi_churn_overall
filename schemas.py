from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Literal

class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Any | None = None


class FeatureVectorChurn(BaseModel):
    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

class DatasetRowChurn(FeatureVectorChurn):
    churn: int

class PredictionResponseChurn(BaseModel):
    churn: int
    probability_no_churn: float
    probability_churn: float

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "churn": 1,
                    "probability_no_churn": 0.18,
                    "probability_churn": 0.82,
                }
            ]
        }
    )

class TrainingConfigChurn(BaseModel):
    model_type: Literal["logreg", "random_forest"]
    hyperparameters: dict[str, Any] = Field(default_factory=dict)

class TrainingHistoryRecord(BaseModel):
    timestamp: str
    model_type: str
    hyperparameters: dict[str, Any]
    metrics: dict[str, float]


class TrainingHistoryResponse(BaseModel):
    total: int
    records: list[TrainingHistoryRecord]