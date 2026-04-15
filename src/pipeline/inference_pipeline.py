from pathlib import Path

import joblib
import pandas as pd

from src.pipeline.training_pipeline import (
    CUSTOMERS_PATH,
    MODEL_PATH,
    TICKETS_PATH,
    FEATURE_COLUMNS,
    build_training_dataset,
    ensure_model_artifact,
    load_source_data,
    validate_inference_schema,
)


class CustomerNotFoundError(ValueError):
    pass


def load_model_artifact(model_path: Path = MODEL_PATH) -> dict:
    ensure_model_artifact()
    return joblib.load(model_path)


def build_customer_feature_frame(
    customer_id: str,
    customers_path: Path = CUSTOMERS_PATH,
    tickets_path: Path = TICKETS_PATH,
) -> pd.DataFrame:
    customers, tickets = load_source_data(customers_path, tickets_path)
    dataset = build_training_dataset(customers, tickets)
    customer_frame = dataset.loc[dataset["customer_id"] == customer_id, FEATURE_COLUMNS].copy()
    if customer_frame.empty:
        raise CustomerNotFoundError(f"Customer {customer_id} not found")
    validate_inference_schema(customer_frame)
    return customer_frame


def predict_customer_churn(customer_id: str) -> dict:
    artifact = load_model_artifact()
    feature_frame = build_customer_feature_frame(customer_id)
    probability = float(artifact["model"].predict_proba(feature_frame)[0, 1])
    prediction = int(probability >= 0.5)
    risk_category = "HIGH" if probability >= 0.75 else "MEDIUM" if probability >= 0.4 else "LOW"
    return {
        "customer_id": customer_id,
        "prediction": prediction,
        "churn_probability": round(probability, 4),
        "risk_category": risk_category,
        "model_version": "churn-model-v1",
    }
