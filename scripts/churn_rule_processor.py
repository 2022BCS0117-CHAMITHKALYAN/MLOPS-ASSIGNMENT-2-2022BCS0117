import pandas as pd
from pathlib import Path

# Base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent


# -----------------------------
# Rule Function
# -----------------------------
def evaluate_risk(customer):

    tickets = customer["tickets_last_30_days"]
    contract = customer["contract_type"]
    complaint = customer["complaint_ticket"]   # ✅ FIXED

    # Rule 1: High risk (many tickets)
    if tickets > 5:
        return "HIGH"

    # Rule 2: High risk (month-to-month + complaint)
    if contract == "Month-to-month" and complaint == 1:
        return "HIGH"

    # Rule 3: Medium risk
    if tickets >= 3:
        return "MEDIUM"

    # Default
    return "LOW"
# -----------------------------
# Main Processing Function
# -----------------------------
def generate_risk_labels(input_path, output_path):

    # Load dataset
    df = pd.read_csv(input_path)

    print("Dataset Loaded Successfully ✅")
    print("Columns:", df.columns.tolist())

    # -----------------------------
    # Validate Required Columns
    # -----------------------------
    required_columns = [
        "tickets_last_30_days",
        "contract_type",
        "complaint_ticket"
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"❌ Missing required column: {col}")

    # -----------------------------
    # Apply Rule Engine
    # -----------------------------
    df["risk_label"] = df.apply(evaluate_risk, axis=1)

    # -----------------------------
    # Save Output
    # -----------------------------
    df.to_csv(output_path, index=False)

    print("\n✅ Risk classification completed")
    print("Output saved to:", output_path)

    print("\n📊 Risk Distribution:")
    print(df["risk_label"].value_counts())


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":

    input_file = BASE_DIR / "data" / "refined" / "customer_behavior_data.csv"
    output_file = BASE_DIR / "data" / "refined" / "customer_risk_output.csv"

    generate_risk_labels(input_file, output_file)