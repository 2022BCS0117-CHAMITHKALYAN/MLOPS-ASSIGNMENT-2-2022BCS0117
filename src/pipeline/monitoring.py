from src.pipeline.inference_pipeline import load_model_artifact
from src.pipeline.training_pipeline import NUMERIC_FEATURES, build_training_dataset, load_source_data


def generate_monitoring_snapshot() -> dict:
    artifact = load_model_artifact()
    customers, tickets = load_source_data()
    dataset = build_training_dataset(customers, tickets)

    current_means = {column: round(float(dataset[column].mean()), 4) for column in NUMERIC_FEATURES}
    baseline_means = artifact.get("baseline_numeric_means", {})
    feature_drift = {}

    for column in NUMERIC_FEATURES:
        baseline = float(baseline_means.get(column, 0.0))
        current = current_means[column]
        feature_drift[column] = round(current - baseline, 4)

    top_drift = sorted(
        feature_drift.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:5]

    return {
        "dataset_version": artifact.get("dataset_version"),
        "model_metrics": artifact.get("metrics", {}),
        "top_feature_drift": [{"feature": name, "mean_delta": delta} for name, delta in top_drift],
        "latency_ms_target": 200,
        "memory_mb_target": 512,
        "retraining_recommended": any(abs(delta) > 5 for _, delta in top_drift),
    }
