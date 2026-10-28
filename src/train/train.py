"""Train and save the customer churn prediction pipeline."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "churn_v2.csv"
MODEL_PATH = ROOT / "model.joblib"

NUMERIC = ["tenure_months", "monthly_charges", "support_calls"]
CATEGORICAL = ["contract", "internet_service"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "churn"


def train_model():
    """Train the model, evaluate it, and save the fitted pipeline."""
    df = pd.read_csv(DATA_PATH)

    # churn_v2 uses monthly_fee; the model expects monthly_charges.
    df = df.rename(columns={"monthly_fee": "monthly_charges"})

    required = FEATURES + [TARGET]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if df[required].isnull().any().any():
        raise ValueError("Training data contains missing required values.")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessing),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = probabilities >= 0.5

    print(f"F1 : {f1_score(y_test, predictions):.3f}")
    print(f"AUC: {roc_auc_score(y_test, probabilities):.3f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    return model


if __name__ == "__main__":
    train_model()
