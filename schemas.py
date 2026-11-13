from pydantic import BaseModel, ConfigDict


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