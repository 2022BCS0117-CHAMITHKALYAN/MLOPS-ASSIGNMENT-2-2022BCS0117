import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "refined"
MODEL_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "artifacts" / "metrics"
EXPERIMENTS_DIR = BASE_DIR / "artifacts" / "experiments"
REGISTRY_DIR = BASE_DIR / "artifacts" / "model_registry"

CUSTOMERS_PATH = DATA_DIR / "customers.csv"
TICKETS_PATH = DATA_DIR / "support_tickets.csv"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
METRICS_PATH = METRICS_DIR / "training_metrics.json"
EXPERIMENT_LOG_PATH = EXPERIMENTS_DIR / "experiments.jsonl"
REGISTRY_PATH = REGISTRY_DIR / "registry.json"

SENTIMENT_MAP = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "senior_citizen",
    "tickets_last_7_days",
    "tickets_last_30_days",
    "tickets_last_90_days",
    "ticket_sentiment_score",
    "avg_days_between_tickets",
    "monthly_charge_delta",
    "billing_ticket_count",
    "complaint_ticket_count",
    "technical_ticket_count",
]
CATEGORICAL_FEATURES = [
    "contract_type",
    "payment_method",
    "paperless_billing",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
REQUIRED_CUSTOMER_COLUMNS = {
    "customer_id",
    "contract_type",
    "tenure",
    "monthly_charges",
    "total_charges",
    "PaymentMethod",
    "PaperlessBilling",
    "SeniorCitizen",
    "Churn",
}
REQUIRED_TICKET_COLUMNS = {
    "ticket_id",
    "customer_id",
    "ticket_type",
    "sentiment",
    "created_at",
}


class SchemaValidationError(ValueError):
    pass


@dataclass
class TrainingOutput:
    model_path: Path
    metrics_path: Path
    registry_path: Path
    feature_columns: list[str]
    metrics: dict[str, Any]


def validate_customer_schema(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_CUSTOMER_COLUMNS.difference(df.columns))
    if missing:
        raise SchemaValidationError(f"Missing customer columns: {missing}")


def validate_ticket_schema(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_TICKET_COLUMNS.difference(df.columns))
    if missing:
        raise SchemaValidationError(f"Missing ticket columns: {missing}")


def validate_inference_schema(df: pd.DataFrame) -> None:
    missing = sorted(set(FEATURE_COLUMNS).difference(df.columns))
    if missing:
        raise SchemaValidationError(f"Missing inference feature columns: {missing}")


def _ensure_directories() -> None:
    for path in [MODEL_DIR, METRICS_DIR, EXPERIMENTS_DIR, REGISTRY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_source_data(
    customers_path: Path = CUSTOMERS_PATH,
    tickets_path: Path = TICKETS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    customers = pd.read_csv(customers_path)
    tickets = pd.read_csv(tickets_path)
    validate_customer_schema(customers)
    validate_ticket_schema(tickets)
    tickets["created_at"] = pd.to_datetime(tickets["created_at"], errors="coerce")
    if tickets["created_at"].isna().any():
        raise SchemaValidationError("Ticket timestamps contain invalid values")
    return customers, tickets


def _calculate_ticket_aggregates(tickets: pd.DataFrame, reference_time: datetime) -> pd.DataFrame:
    tickets = tickets.copy()
    tickets["sentiment_value"] = tickets["sentiment"].map(SENTIMENT_MAP).fillna(0.0)
    tickets = tickets.sort_values(["customer_id", "created_at"])

    window_7 = reference_time - timedelta(days=7)
    window_30 = reference_time - timedelta(days=30)
    window_90 = reference_time - timedelta(days=90)

    def _per_customer(group: pd.DataFrame) -> pd.Series:
        created = group["created_at"]
        counts = {
            "tickets_last_7_days": int((created >= window_7).sum()),
            "tickets_last_30_days": int((created >= window_30).sum()),
            "tickets_last_90_days": int((created >= window_90).sum()),
            "ticket_sentiment_score": float(group["sentiment_value"].mean()),
        }

        category_counts = group["ticket_type"].value_counts()
        counts["billing_ticket_count"] = int(category_counts.get("billing", 0))
        counts["complaint_ticket_count"] = int(category_counts.get("complaint", 0))
        counts["technical_ticket_count"] = int(category_counts.get("technical", 0))

        gaps = created.sort_values().diff().dropna().dt.total_seconds() / 86400.0
        counts["avg_days_between_tickets"] = float(gaps.mean()) if not gaps.empty else 90.0
        return pd.Series(counts)

    aggregates = (
        tickets.groupby("customer_id", group_keys=False)[["created_at", "sentiment_value", "ticket_type"]]
        .apply(_per_customer)
    )
    return aggregates.reset_index()


def build_training_dataset(
    customers: pd.DataFrame,
    tickets: pd.DataFrame,
    reference_time: datetime | None = None,
) -> pd.DataFrame:
    reference_time = reference_time or tickets["created_at"].max().to_pydatetime()
    ticket_features = _calculate_ticket_aggregates(tickets, reference_time)

    dataset = customers.merge(ticket_features, on="customer_id", how="left")
    dataset = dataset.rename(
        columns={
            "PaymentMethod": "payment_method",
            "PaperlessBilling": "paperless_billing",
            "SeniorCitizen": "senior_citizen",
        }
    )
    dataset["monthly_charge_delta"] = dataset["monthly_charges"] - (
        dataset["total_charges"] / dataset["tenure"].clip(lower=1)
    )

    fill_values = {
        "tickets_last_7_days": 0,
        "tickets_last_30_days": 0,
        "tickets_last_90_days": 0,
        "ticket_sentiment_score": 0.0,
        "billing_ticket_count": 0,
        "complaint_ticket_count": 0,
        "technical_ticket_count": 0,
        "avg_days_between_tickets": 90.0,
    }
    dataset = dataset.fillna(fill_values)
    return dataset


def create_training_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=10,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def _record_experiment(run_record: dict[str, Any]) -> None:
    with EXPERIMENT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(run_record) + "\n")


def _update_model_registry(run_record: dict[str, Any]) -> None:
    registry = {
        "latest_model": run_record["model_version"],
        "stages": {
            "Staging": run_record["model_version"],
            "Production": run_record["model_version"],
            "Archived": [],
        },
        "models": [run_record],
    }
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def train_model(
    customers_path: Path = CUSTOMERS_PATH,
    tickets_path: Path = TICKETS_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> TrainingOutput:
    _ensure_directories()
    customers, tickets = load_source_data(customers_path, tickets_path)
    dataset = build_training_dataset(customers, tickets)

    X = dataset[FEATURE_COLUMNS].copy()
    y = dataset["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    validate_inference_schema(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = create_training_pipeline()
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "f1": round(float(f1_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "precision_recall_auc": round(float(average_precision_score(y_test, probabilities)), 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    artifact = {
        "model": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "dataset_version": "refined-customers-support_tickets-v1",
        "reference_time": tickets["created_at"].max().isoformat(),
        "metrics": metrics,
        "baseline_numeric_means": {
            column: round(float(X[column].mean()), 4) for column in NUMERIC_FEATURES
        },
    }
    joblib.dump(artifact, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    run_record = {
        "run_id": datetime.utcnow().strftime("%Y%m%d%H%M%S"),
        "model_version": "churn-model-v1",
        "model_path": str(model_path.relative_to(BASE_DIR)),
        "dataset_version": artifact["dataset_version"],
        "feature_list": FEATURE_COLUMNS,
        "hyperparameters": {
            "n_estimators": 250,
            "max_depth": 10,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
        },
        "metrics": metrics,
        "registered_at": artifact["trained_at"],
    }
    _record_experiment(run_record)
    _update_model_registry(run_record)

    return TrainingOutput(
        model_path=model_path,
        metrics_path=metrics_path,
        registry_path=REGISTRY_PATH,
        feature_columns=FEATURE_COLUMNS,
        metrics=metrics,
    )


def ensure_model_artifact() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    train_model()
    return MODEL_PATH
