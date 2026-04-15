import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.training_pipeline import train_model


if __name__ == "__main__":
    output = train_model()
    print(f"Model saved to: {output.model_path}")
    print(f"Metrics saved to: {output.metrics_path}")
    print(output.metrics)
