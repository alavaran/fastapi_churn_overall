import logging

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dataset import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from schemas import TrainingConfigChurn

logger = logging.getLogger(__name__)


def train_churn_model(X_train, y_train, config: TrainingConfigChurn):
    logger.info("Starting model training")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    if config.model_type == "logreg":
        default_params = {
            "max_iter": 1000,
            "random_state": 42,
        }
        classifier_class = LogisticRegression

    elif config.model_type == "random_forest":
        default_params = {
            "n_estimators": 100,
            "max_depth": None,
            "random_state": 42,
            "n_jobs": -1,
        }
        classifier_class = RandomForestClassifier

    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")

    model_params = {
        **default_params,
        **config.hyperparameters,
    }

    classifier = classifier_class(**model_params)

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    model_pipeline.fit(X_train, y_train)

    logger.info("Model training completed successfully")

    return model_pipeline
