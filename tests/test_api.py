"""Tests for the customer churn prediction API."""

import joblib
from fastapi.testclient import TestClient

from src.app.main import app, MODEL_PATH

client = TestClient(app)

VALID_CUSTOMER = {
    "tenure_months": 24,
    "monthly_charges": 75.50,
    "support_calls": 2,
    "contract": "month-to-month",
    "internet_service": "fiber",
}


def test_health_endpoint_contract():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
    }


def test_model_loads_successfully():
    model = joblib.load(MODEL_PATH)

    assert model is not None
    assert hasattr(model, "predict_proba")
    assert model.n_features_in_ == 5


def test_valid_prediction_contract():
    response = client.post("/predict", json=VALID_CUSTOMER)

    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"churn_probability", "prediction"}
    assert 0 <= result["churn_probability"] <= 1
    assert result["prediction"] in (0, 1)


def test_missing_required_feature_is_rejected():
    customer = VALID_CUSTOMER.copy()
    del customer["tenure_months"]

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_negative_tenure_is_rejected():
    customer = VALID_CUSTOMER.copy()
    customer["tenure_months"] = -1

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_prediction_is_deterministic():
    first = client.post("/predict", json=VALID_CUSTOMER)
    second = client.post("/predict", json=VALID_CUSTOMER)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
