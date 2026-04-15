# As ..src/feature_pipeline.py
import pandas as pd
from datetime import datetime, timedelta


def build_features(customers_path, tickets_path, output_path):

    # Load datasets
    customers = pd.read_csv(customers_path)
    tickets = pd.read_csv(tickets_path)

    tickets["created_at"] = pd.to_datetime(tickets["created_at"])

    now = datetime.now()
    window_7 = now - timedelta(days=7)
    window_30 = now - timedelta(days=30)
    window_90 = now - timedelta(days=90)

    # -----------------------------
    # Ticket frequency features
    #2022BCS0117 CHAMIH KALYAN
    # -----------------------------

    tickets_7 = tickets[tickets["created_at"] > window_7]
    tickets_30 = tickets[tickets["created_at"] > window_30]
    tickets_90 = tickets[tickets["created_at"] > window_90]

    f7 = tickets_7.groupby("customer_id").size().reset_index(name="tickets_last_7_days")
    f30 = tickets_30.groupby("customer_id").size().reset_index(name="tickets_last_30_days")
    f90 = tickets_90.groupby("customer_id").size().reset_index(name="tickets_last_90_days")

    # -----------------------------
    # Complaint Indiacator
    #2022BCS0117 CHAMITH KALYAN
    # -----------------------------

    complaints = tickets[tickets["ticket_type"] == "complaint"]

    complaint_feature = complaints.groupby("customer_id").size().reset_index(name="complaint_count")
    complaint_feature["complaint_ticket"] = 1
    complaint_feature = complaint_feature[["customer_id", "complaint_ticket"]]

    # -----------------------------
    # Sentiment feature
    #2022BCS0117 CHAMITH KALYAN
    # -----------------------------

    negative = tickets[tickets["sentiment"] == "negative"]

    neg_counts = negative.groupby("customer_id").size().reset_index(name="negative_tickets")
    total_counts = tickets.groupby("customer_id").size().reset_index(name="total_tickets")

    sentiment_feature = total_counts.merge(neg_counts, on="customer_id", how="left")
    sentiment_feature["negative_tickets"] = sentiment_feature["negative_tickets"].fillna(0)

    sentiment_feature["negative_ratio"] = (
        sentiment_feature["negative_tickets"] / sentiment_feature["total_tickets"]
    )

    sentiment_feature = sentiment_feature[["customer_id", "negative_ratio"]]

    # -----------------------------
    # Merge all features
    # -----------------------------

    features = customers[
        ["customer_id", "contract_type", "monthly_charges", "tenure", "Churn"]
    ]

    features = features.merge(f7, on="customer_id", how="left")
    features = features.merge(f30, on="customer_id", how="left")
    features = features.merge(f90, on="customer_id", how="left")
    features = features.merge(complaint_feature, on="customer_id", how="left")
    features = features.merge(sentiment_feature, on="customer_id", how="left")

    # Replace missing values
    features = features.fillna(0)

    # Save feature dataset
    features.to_csv(output_path, index=False)

    print("Feature dataset saved to:", output_path)
    print("Shape:", features.shape)


if __name__ == "__main__":

    build_features(
        customers_path="../data/refined/customers.csv",
        tickets_path="../data/refined/support_tickets.csv",
        output_path="../data/refined/customer_behavior_data.csv",
    )
    print('Feature Dataset created.')