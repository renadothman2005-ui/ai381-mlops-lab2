"""FastAPI service for customer churn prediction."""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "model.joblib"

FEATURES = [
    "tenure_months",
    "monthly_charges",
    "support_calls",
    "contract",
    "internet_service",
]

app = FastAPI(
    title="AI381 Customer Churn API",
    version="1.0.0",
)

# Loading the model at startup/import makes missing or invalid models visible.
model = joblib.load(MODEL_PATH)


class CustomerInput(BaseModel):
    """Validated features required for one churn prediction."""

    model_config = ConfigDict(extra="forbid")

    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0, le=10000)
    support_calls: int = Field(ge=0, le=1000)
    contract: str = Field(min_length=1, max_length=50)
    internet_service: str = Field(min_length=1, max_length=50)


class PredictionOutput(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    prediction: int = Field(ge=0, le=1)


@app.get("/health")
def health():
    """Report that the service is running and its model is loaded."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    """Return the predicted churn probability and binary prediction."""
    data = customer.model_dump()
    X = pd.DataFrame([data], columns=FEATURES)

    probability = float(model.predict_proba(X)[0, 1])

    return {
        "churn_probability": round(probability, 6),
        "prediction": int(probability >= 0.5),
    }
