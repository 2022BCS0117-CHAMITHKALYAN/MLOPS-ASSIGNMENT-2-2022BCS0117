import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Number of services
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]

    df["num_services"] = df[service_cols].apply(
        lambda row: sum([1 for val in row if val in ["Yes", "DSL", "Fiber optic"]]),
        axis=1
    )

    # 2. Contract type
    df["is_monthly_contract"] = df["Contract"].apply(
        lambda x: 1 if x == "Month-to-month" else 0
    )

    # 3. Avg charge per tenure
    df["avg_charge_per_tenure"] = df["TotalCharges"] / (df["tenure"] + 1)

    # 4. Tenure groups
    def tenure_group(x):
        if x < 12:
            return "0-1yr"
        elif x < 24:
            return "1-2yr"
        elif x < 48:
            return "2-4yr"
        else:
            return "4+yr"

    df["tenure_group"] = df["tenure"].apply(tenure_group)

    # 5. Support features
    df["has_support"] = df[["TechSupport", "OnlineSecurity"]].apply(
        lambda row: 1 if "Yes" in row.values else 0,
        axis=1
    )

    logging.info("Feature engineering completed")

    return df
