
from sklearn.pipeline import Pipeline

from model import train_churn_model
from schemas import TrainingConfigChurn


def test_train_logistic_regression(sample_dataset):
    X = sample_dataset.drop(columns=["churn"])
    y = sample_dataset["churn"]

    config = TrainingConfigChurn(
        model_type="logreg",
        hyperparameters={},
    )

    model = train_churn_model(X, y, config)

    assert isinstance(model, Pipeline)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    assert len(predictions) == len(y)
    assert probabilities.shape == (len(y), 2)


def test_train_random_forest(sample_dataset):
    X = sample_dataset.drop(columns=["churn"])
    y = sample_dataset["churn"]

    config = TrainingConfigChurn(
        model_type="random_forest",
        hyperparameters={
            "n_estimators": 10,
            "max_depth": 3,
        },
    )

    model = train_churn_model(X, y, config)

    assert isinstance(model, Pipeline)
    assert len(model.predict(X)) == len(y)
