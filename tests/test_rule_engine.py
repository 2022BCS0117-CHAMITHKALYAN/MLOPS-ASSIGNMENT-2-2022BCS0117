import pandas as pd
import pytest

from src.pipeline.inference_pipeline import build_customer_feature_frame, predict_customer_churn
from src.pipeline.training_pipeline import FEATURE_COLUMNS, SchemaValidationError, train_model, validate_inference_schema


def test_training_pipeline_creates_metrics(tmp_path):
    output = train_model(
        model_path=tmp_path / "model.joblib",
        metrics_path=tmp_path / "metrics.json",
    )
    assert output.model_path.exists()
    assert output.metrics["f1"] >= 0
    assert output.metrics["roc_auc"] >= 0
    assert output.metrics["precision_recall_auc"] >= 0


def test_inference_frame_matches_training_schema():
    frame = build_customer_feature_frame("6467-CHFZW")
    assert list(frame.columns) == FEATURE_COLUMNS


def test_predict_customer_churn_response_shape():
    result = predict_customer_churn("6467-CHFZW")
    assert result["customer_id"] == "6467-CHFZW"
    assert 0 <= result["churn_probability"] <= 1
    assert result["risk_category"] in {"LOW", "MEDIUM", "HIGH"}


def test_schema_validation_rejects_missing_features():
    with pytest.raises(SchemaValidationError):
        validate_inference_schema(pd.DataFrame([{"tenure": 12}]))
