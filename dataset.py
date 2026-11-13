import pandas as pd

from pathlib import Path
from schemas import DatasetRowChurn

from sklearn.model_selection import train_test_split

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


DATASET_PATH = Path(__file__).parent / "data" / "churn_dataset.csv"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    return df


def load_dataset_rows() -> list[DatasetRowChurn]:
    df = load_dataset()

    rows = df.to_dict(orient="records")

    return [
        DatasetRowChurn.model_validate(row)
        for row in rows
    ]

def prepare_dataset():
    df = load_dataset()

    # Разделяем признаки и целевую переменную
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # Удаляем строки, где неизвестна целевая переменная
    valid_target = y.notna()
    X = X.loc[valid_target].copy()
    y = y.loc[valid_target].copy()

    # Делим данные на train и test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Заполняем пропуски числовых признаков медианами train
    for column in NUMERIC_FEATURES:
        median = X_train[column].median()
        X_train[column] = X_train[column].fillna(median)
        X_test[column] = X_test[column].fillna(median)

    # Заполняем пропуски категориальных признаков модой train
    for column in CATEGORICAL_FEATURES:
        mode = X_train[column].mode()

        if not mode.empty:
            most_frequent = mode.iloc[0]
            X_train[column] = X_train[column].fillna(most_frequent)
            X_test[column] = X_test[column].fillna(most_frequent)

    return X_train, X_test, y_train, y_test

def get_class_distribution(y: pd.Series) -> dict:
    return y.value_counts().sort_index().to_dict()

def prepare_dataset():
    df = load_dataset()

    if df.empty:
        raise ValueError("Dataset is empty")

    if TARGET not in df.columns:
        raise ValueError("Target column 'churn' is missing")

    df = df.dropna(subset=[TARGET])

    if df.empty:
        raise ValueError("Dataset has no valid target values")

    if df[TARGET].nunique() < 2:
        raise ValueError("Dataset must contain both churn classes")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test