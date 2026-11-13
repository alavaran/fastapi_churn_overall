# Customer Churn Prediction API

A machine learning service for predicting customer churn probability using **Python, FastAPI, and scikit-learn**.

The project demonstrates an end-to-end ML workflow: dataset loading, preprocessing, model training, evaluation, persistence, and real-time predictions through a REST API.

## Features

- **REST API** built with FastAPI.
- **Data validation** using Pydantic.
- **Dataset inspection** and validation endpoints.
- **Automated preprocessing** with scikit-learn Pipeline and ColumnTransformer.
- **Machine learning model** based on Logistic Regression.
- **Model evaluation** using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- **Model persistence** with Joblib.
- **Single and batch predictions**.
- **Churn probability estimation** for each customer.
- **Interactive API documentation** with Swagger UI.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| FastAPI | REST API framework |
| Pydantic | Request and response validation |
| Pandas | Data processing |
| scikit-learn | Machine learning and preprocessing |
| Joblib | Model serialization |
| Uvicorn | ASGI server |
| Pytest | Automated testing |
| Docker | Containerization |

## Project Structure

```text
fastapi_churn_overall/
│
├── data/
│   └── churn_dataset.csv
│
├── models/
│   └── churn_model.joblib
│
├── tests/
│
├── main.py
├── schemas.py
├── dataset.py
├── model.py
├── model_storage.py
│
├── requirements.txt
├── Dockerfile
└── README.md
```

*The structure may evolve as the project develops.*

## Dataset

The service uses a customer churn dataset containing numerical and categorical features.

| Feature | Description | Type |
|---|---|---|
| monthly_fee | Monthly subscription fee | Numerical |
| usage_hours | Customer usage hours | Numerical |
| support_requests | Number of support requests | Numerical |
| account_age_months | Account age in months | Numerical |
| failed_payments | Number of failed payments | Numerical |
| region | Customer region | Categorical |
| device_type | Device used by customer | Categorical |
| payment_method | Payment method | Categorical |
| autopay_enabled | Automatic payment enabled | Binary |
| churn | Customer churn indicator | Target |

Target variable:
- `0` — customer stays.
- `1` — customer churns.

## Machine Learning Pipeline

The model uses a scikit-learn Pipeline with separate preprocessing for numerical and categorical features.

**Numerical features:**
- Missing value imputation using median.
- Feature scaling using StandardScaler.

**Categorical features:**
- Missing value imputation using most frequent value.
- One-hot encoding with unknown category handling.

**Classifier:**
- Logistic Regression.

The preprocessing steps are fitted on the training data to avoid data leakage.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Service health check |
| GET | `/dataset/preview` | Preview dataset records |
| GET | `/dataset/info` | Dataset information |
| GET | `/dataset/validate` | Dataset validation |
| GET | `/dataset/split-info` | Training/test split information |
| POST | `/model/train` | Train the churn prediction model |
| GET | `/model/status` | Get model status and metrics |
| POST | `/predict` | Predict churn for one or multiple customers |

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/fastapi_churn_overall.git
cd fastapi_churn_overall
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## Example: Train the Model

```bash
curl -X POST "http://127.0.0.1:8000/model/train"
```

## Example: Single Customer Prediction

**Request:**

```json
{
  "monthly_fee": 49.99,
  "usage_hours": 12.5,
  "support_requests": 4,
  "account_age_months": 3,
  "failed_payments": 2,
  "region": "europe",
  "device_type": "mobile",
  "payment_method": "card",
  "autopay_enabled": 0
}
```

**Response:**

```json
{
  "churn": 1,
  "probability_no_churn": 0.2932,
  "probability_churn": 0.7068
}
```

The probabilities represent the model's estimated likelihood of each outcome.

## Example: Batch Prediction

The `/predict` endpoint also supports a list of customers.

```json
[
  {
    "monthly_fee": 49.99,
    "usage_hours": 12.5,
    "support_requests": 4,
    "account_age_months": 3,
    "failed_payments": 2,
    "region": "europe",
    "device_type": "mobile",
    "payment_method": "card",
    "autopay_enabled": 0
  },
  {
    "monthly_fee": 29.99,
    "usage_hours": 85.0,
    "support_requests": 0,
    "account_age_months": 24,
    "failed_payments": 0,
    "region": "asia",
    "device_type": "desktop",
    "payment_method": "paypal",
    "autopay_enabled": 1
  }
]
```

The response contains an individual prediction for each customer, preserving the input order.

## Model Evaluation

The model is evaluated using:

- **Accuracy** — overall prediction accuracy.
- **Precision** — proportion of predicted churn cases that are actual churn cases.
- **Recall** — proportion of actual churn cases identified by the model.
- **F1-score** — harmonic mean of Precision and Recall.
- **ROC-AUC** — ability to distinguish between churn and non-churn customers.

For churn prediction, Recall and F1-score are particularly important because missing customers who are likely to churn can have a significant business impact.

## Error Handling

The API validates incoming data and returns appropriate HTTP errors.

For example:
- `422 Unprocessable Entity` — invalid request data.
- `503 Service Unavailable` — model has not been trained or loaded yet.

## Development Roadmap

- [x] FastAPI application setup.
- [x] Dataset loading and inspection.
- [x] Data validation and preprocessing.
- [x] Train/test split.
- [x] Scikit-learn Pipeline.
- [x] Model training and evaluation.
- [x] Model persistence.
- [x] Single customer prediction.
- [x] Batch prediction.
- [x] API documentation and examples.
- [ ] Configurable model selection and hyperparameters.
- [ ] Extended automated API tests.
- [ ] Docker deployment.
- [ ] Final integration and deployment improvements.

## Project Goals

This project is built as a practical exercise in developing and deploying machine learning services.

It focuses on:
- Designing REST APIs for ML applications.
- Building reusable preprocessing and training pipelines.
- Preventing data leakage.
- Persisting and serving trained models.
- Validating API inputs and outputs.
- Evaluating classification models from a business perspective.

---

**Status:** In development.
