def evaluate_risk(row):
    tickets = row["tickets_last_30_days"]
    contract = row["contract_type"]
    complaint = row["complaint_ticket"]

    if tickets > 5:
        return "HIGH"

    if contract == "Month-to-month" and complaint == 1:
        return "HIGH"

    if tickets >= 3:
        return "MEDIUM"

    return "LOW"