
from pathlib import Path
from datetime import datetime, timezone

import joblib


MODEL_PATH = Path(__file__).parent / "models" / "churn_model.joblib"


def save_churn_model(model, metrics, model_type, hyperparameters):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "model_type": model_type,
        "hyperparameters": hyperparameters,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }

    joblib.dump(artifact, MODEL_PATH)

    return artifact


def load_churn_model():
    if not MODEL_PATH.exists():
        return None

    artifact = joblib.load(MODEL_PATH)

    return artifact
