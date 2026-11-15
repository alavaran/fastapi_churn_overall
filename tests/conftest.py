
import pandas as pd
import pytest


@pytest.fixture
def sample_dataset():
    return pd.DataFrame({
        "monthly_fee": [20.0, 30.0, 40.0, 50.0, 25.0, 60.0],
        "usage_hours": [10.0, 20.0, 30.0, 40.0, 15.0, 50.0],
        "support_requests": [0, 1, 2, 3, 1, 4],
        "account_age_months": [24, 18, 12, 6, 20, 3],
        "failed_payments": [0, 0, 1, 2, 0, 3],
        "region": [
            "europe", "asia", "america",
            "europe", "asia", "america",
        ],
        "device_type": [
            "mobile", "desktop", "tablet",
            "mobile", "desktop", "tablet",
        ],
        "payment_method": [
            "card", "paypal", "crypto",
            "card", "paypal", "crypto",
        ],
        "autopay_enabled": [1, 1, 0, 0, 1, 0],
        "churn": [0, 0, 0, 1, 0, 1],
    })
