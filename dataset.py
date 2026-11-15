import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from errors import ServiceError

logger = logging.getLogger(__name__)

TARGET = "churn"

NUMERIC_FEATURES = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]

CATEGORICAL_FEATURES = [
    "region",
    "device_type",
    "payment_method",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DATASET_PATH = Path(__file__).parent / "data" / "churn_dataset.csv"


def load_dataset() -> pd.DataFrame:
    logger.info("Loading churn dataset")

    try:
        df = pd.read_csv(DATASET_PATH)
        logger.info(
            "Churn dataset loaded successfully: %s rows",
            len(df),
        )
        return df

    except pd.errors.EmptyDataError:
        logger.error("Churn dataset is empty")
        return pd.DataFrame()

    except FileNotFoundError:
        logger.exception("Churn dataset file not found")
        raise
    

def prepare_dataset():
    df = load_dataset()

    if df.empty:
        raise ServiceError(
            code="DATASET_EMPTY",
            message="Датасет пуст",
            status_code=422,
        )

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {missing_columns}"
        )

    df = df.dropna(subset=[TARGET])

    if df.empty:
        raise ServiceError(
            code="DATASET_EMPTY",
            message="В датасете отсутствуют строки с корректным target",
            status_code=422,
        )

    if df[TARGET].nunique() < 2:
        raise ValueError("Dataset must contain both churn classes")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def get_class_distribution(y: pd.Series) -> dict:
    return y.value_counts().sort_index().to_dict()
