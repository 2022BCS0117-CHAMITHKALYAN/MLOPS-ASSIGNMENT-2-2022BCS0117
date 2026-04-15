from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_api_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "service running"
    assert response.json()["model_artifact"] == "churn_model.joblib"


def test_valid_prediction():
    response = client.post(
        "/predict-risk",
        json={"customer_id": "6467-CHFZW"}
    )
    assert response.status_code == 200
    assert "risk_category" in response.json()
    assert "churn_probability" in response.json()
    assert response.json()["model_version"] == "churn-model-v1"


def test_invalid_customer():
    response = client.post(
        "/predict-risk",
        json={"customer_id": "INVALID"}
    )
    assert response.status_code == 404


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert "metrics" in body
    assert "f1" in body["metrics"]


def test_monitoring_summary():
    response = client.get("/monitoring-summary")
    assert response.status_code == 200
    body = response.json()
    assert "top_feature_drift" in body
    assert "retraining_recommended" in body
