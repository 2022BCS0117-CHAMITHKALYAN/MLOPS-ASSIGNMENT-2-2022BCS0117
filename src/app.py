from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.pipeline.inference_pipeline import CustomerNotFoundError, predict_customer_churn
from src.pipeline.monitoring import generate_monitoring_snapshot
from src.pipeline.training_pipeline import MODEL_PATH, METRICS_PATH, ensure_model_artifact
import json

app = FastAPI(
    title="Churn Risk Prediction Service",
    description="ML-based churn risk prediction API with reproducible feature pipeline",
    version="2.0"
)
ensure_model_artifact()

class CustomerRequest(BaseModel):
    customer_id: str

@app.get("/")
def health_check():
    return {"status": "service running", "model_artifact": str(MODEL_PATH.name)}

@app.get("/model-info")
def model_info():
    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {
        "model_path": str(MODEL_PATH),
        "metrics": metrics,
    }

@app.get("/monitoring-summary")
def monitoring_summary():
    return generate_monitoring_snapshot()

@app.post("/predict-risk")
def predict_risk(request: CustomerRequest):
    try:
        return predict_customer_churn(request.customer_id)
    except CustomerNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
