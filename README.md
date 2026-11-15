# Customer Churn Prediction API

A machine learning REST API built with **Python, FastAPI, Pandas and Scikit-learn** to predict customer churn.

The service supports model training, single and batch predictions, model metrics, training history, structured error handling and Docker deployment.

## Features

- Train machine learning models through REST API.
- Supported classifiers: Logistic Regression and Random Forest.
- Scikit-learn preprocessing Pipeline with numerical and categorical feature handling.
- Single-customer and batch churn predictions.
- Churn probability for each prediction.
- Model status and feature schema endpoints.
- Training metrics and persistent training history.
- Structured API error responses.
- Application logging and health checks.
- Docker support.
- Unit and integration tests with Pytest.

## Tech Stack

- Python
- FastAPI
- Pydantic
- Pandas
- Scikit-learn
- Joblib
- Pytest
- Docker

## Project Structure

```text
fastapi_churn_overall/
│
├── data/
│   └── churn_dataset.csv
│
├── models/
│   ├── churn_model.joblib
│   └── training_history.json
│
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_dataset.py
│   └── test_model.py
│
├── main.py
├── dataset.py
├── model.py
├── model_storage.py
├── schemas.py
├── errors.py
├── training_history.py
├── logging_config.py
│
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## Dataset

The service uses `data/churn_dataset.csv`.

Each row represents one customer. The target variable is `churn`:

- `0` — customer retained.
- `1` — customer churned.

### Features

| Feature | Description | Type |
|---|---|---|
| `monthly_fee` | Customer's monthly subscription fee | Numeric |
| `usage_hours` | Customer's usage hours | Numeric |
| `support_requests` | Number of support requests | Numeric |
| `account_age_months` | Account age in months | Numeric |
| `failed_payments` | Number of failed payments | Numeric |
| `autopay_enabled` | Whether automatic payments are enabled (0/1) | Numeric |
| `region` | Customer's region | Categorical |
| `device_type` | Device type: mobile, desktop or tablet | Categorical |
| `payment_method` | Payment method: card, PayPal or crypto | Categorical |
| `churn` | Target variable (0/1) | Target |

The preprocessing pipeline handles numerical and categorical features separately.

## Installation and Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/alavaran/fastapi_churn_overall
cd fastapi_churn_overall
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the API

```bash
python -m uvicorn main:app --reload
```

The API will be available at:

- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health check: http://localhost:8000/health

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Root endpoint |
| GET | `/health` | Service health check |
| POST | `/model/train` | Train a churn prediction model |
| GET | `/model/status` | Get model status |
| GET | `/model/schema` | Get expected feature schema |
| GET | `/model/metrics` | Get training metrics and history |
| POST | `/predict` | Predict customer churn |

## Model Training

Train a Logistic Regression model:

**POST** `/model/train`

Request:

```json
{
  "model_type": "logreg",
  "hyperparameters": {}
}
```

The service also supports:

```json
{
  "model_type": "random_forest",
  "hyperparameters": {}
}
```

Hyperparameters are passed to the corresponding Scikit-learn classifier.

After successful training, the service saves the model and records the training timestamp, model type, hyperparameters and evaluation metrics.

## Churn Prediction

**POST** `/predict`

The endpoint accepts either a single customer object or a list of customer objects.

### Single customer

Request:

```json
{
  "monthly_fee": 49.99,
  "usage_hours": 12.5,
  "support_requests": 4,
  "account_age_months": 3,
  "failed_payments": 2,
  "autopay_enabled": 0,
  "region": "europe",
  "device_type": "mobile",
  "payment_method": "card"
}
```

The response includes:

- `churn` — predicted class (0 or 1).
- `probability_no_churn` — probability of customer retention.
- `probability_churn` — probability of customer churn.

### Batch prediction

The same endpoint accepts an array of customer objects:

```json
[
  {
    "monthly_fee": 49.99,
    "usage_hours": 12.5,
    "support_requests": 4,
    "account_age_months": 3,
    "failed_payments": 2,
    "autopay_enabled": 0,
    "region": "europe",
    "device_type": "mobile",
    "payment_method": "card"
  },
  {
    "monthly_fee": 79.99,
    "usage_hours": 35.0,
    "support_requests": 1,
    "account_age_months": 18,
    "failed_payments": 0,
    "autopay_enabled": 1,
    "region": "asia",
    "device_type": "desktop",
    "payment_method": "paypal"
  }
]
```

An empty list is rejected with a structured validation error.

A model must be trained or loaded before predictions can be made.

## Model Metrics and Training History

**GET** `/model/metrics`

Returns training records containing:

- Training timestamp (UTC).
- Model type.
- Hyperparameters.
- Evaluation metrics.

Optional query parameters:

- `model_type` — filter by `logreg` or `random_forest`.
- `limit` — number of latest records (1–100).

Example:

```text
GET /model/metrics?model_type=logreg&limit=5
```

Training history is stored in `models/training_history.json`.

## Error Handling

The API uses a consistent error response structure:

```json
{
  "code": "MODEL_NOT_TRAINED",
  "message": "Model has not been trained",
  "details": null
}
```

HTTP validation errors and application errors are handled centrally. Internal exception details are not exposed to API clients.

## Docker

### 1. Build the image

```bash
docker build -t churn-api .
```

### 2. Run the container

```bash
docker run -d --name churn-api -p 8000:8000 churn-api
```

### 3. Check the service

Open:

- http://localhost:8000/health
- http://localhost:8000/docs

### 4. View container logs

```bash
docker logs churn-api
```

### 5. Stop the container

```bash
docker stop churn-api
```

To remove it:

```bash
docker rm churn-api
```

## Testing

The project uses Pytest for unit and integration testing.

Run all tests:

```bash
python -m pytest -v
```

The test suite covers:

- Dataset preparation and validation.
- Model training with supported classifiers.
- API prediction without a trained model.
- Invalid prediction input.
- End-to-end model training and prediction flow.

## Model Persistence

The trained Scikit-learn Pipeline is serialized using Joblib.

The service can load a previously saved model during application startup, allowing predictions without retraining when a compatible model artifact is available.

Training history is persisted separately as JSON.

## Future Improvements

- CI/CD integration with GitHub Actions.
- Model performance monitoring.
- Data drift detection.
- Model versioning.
- Automated retraining workflows.

---

**Project status:** Completed educational project.