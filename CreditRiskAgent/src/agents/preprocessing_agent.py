def preprocessing_agent(state):
    data = state["customer_data"]

    features = {
        "age": float(data["age"]),
        "income": float(data["income"]),
        "loan_amount": float(data["loan_amount"]),
        "debt_to_income": float(data["existing_debt"]) / (float(data["income"]) + 1),
        "credit_utilization": float(data["existing_debt"]) / (float(data["loan_amount"]) + 1),
        "repayment_history": 0.9 if data["credit_score"] > 700 else 0.6,
        "employment_encoded": 1 if data["employment_status"] == "employed" else 0,
        "credit_history_encoded": 1 if data["credit_score"] > 650 else 0
    }

    return {
        "engineered_features": features
    }