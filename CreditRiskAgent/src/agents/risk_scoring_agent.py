# src/agents/risk_scoring_agent.py

import os
import joblib
import pandas as pd

MODEL_PATH = "data/artifacts/xgboost.pkl"

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded")


def risk_scoring_agent(state):

    features = state.get("engineered_features", {})

    df = pd.DataFrame([{
        "age": features.get("age"),
        "income": features.get("income"),
        "existing_debt": features.get("income") * features.get("debt_to_income"),
        "credit_score": 700 if features.get("credit_history_encoded") else 600,
        "loan_amount": features.get("loan_amount"),
        "employment_encoded": features.get("employment_encoded"),
        "debt_to_income": features.get("debt_to_income")
    }])

    if model:
        pd_score = model.predict_proba(df)[0][1]
    else:
        pd_score = 0.5

    if pd_score < 0.3:
        category = "low"
    elif pd_score < 0.6:
        category = "medium"
    else:
        category = "high"

    return {
        "risk_score": {
            "pd": round(float(pd_score), 4),
            "risk_category": category,
            "prediction": int(pd_score > 0.5)
        }
    }