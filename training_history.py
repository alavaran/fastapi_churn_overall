import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HISTORY_PATH = Path(__file__).parent / "models" / "training_history.json"


def load_training_history() -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []

    with open(HISTORY_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_training_record(
    model_type: str,
    hyperparameters: dict,
    metrics: dict,
) -> dict:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    history = load_training_history()

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_type": model_type,
        "hyperparameters": hyperparameters,
        "metrics": metrics,
    }

    history.append(record)

    with open(HISTORY_PATH, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4, ensure_ascii=False)

    return record


def get_training_history(
    model_type: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    history = load_training_history()

    if model_type is not None:
        history = [
            record
            for record in history
            if record["model_type"] == model_type
        ]

    return history[-limit:][::-1]
