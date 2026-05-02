# src/ml/train.py

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

os.makedirs("data/artifacts", exist_ok=True)


def generate_data(n=10000):
    np.random.seed(42)

    data = pd.DataFrame({
        "age": np.random.randint(21, 65, n),
        "income": np.random.randint(20000, 200000, n),
        "existing_debt": np.random.randint(0, 100000, n),
        "credit_score": np.random.randint(300, 900, n),
        "loan_amount": np.random.randint(1000, 100000, n),
        "employment_encoded": np.random.choice([0, 1], n)
    })

    # Feature engineering
    data["debt_to_income"] = data["existing_debt"] / (data["income"] + 1)

    # Target logic (REALISTIC)
    risk = (
        (data["credit_score"] < 600).astype(int) * 0.4 +
        (data["debt_to_income"] > 0.5).astype(int) * 0.3 +
        (data["loan_amount"] > data["income"]).astype(int) * 0.3 +
        (data["employment_encoded"] == 0).astype(int) * 0.2
    )

    data["default"] = (risk > 0.5).astype(int)

    return data


def train():
    print("\n=== TRAINING MODEL (ALIGNED FEATURES) ===\n")

    df = generate_data()

    features = [
        "age",
        "income",
        "existing_debt",
        "credit_score",
        "loan_amount",
        "employment_encoded",
        "debt_to_income"
    ]

    X = df[features]
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    print(f"ROC-AUC: {round(auc, 4)}")

    joblib.dump(model, "data/artifacts/xgboost.pkl")

    print("✅ Model saved to data/artifacts/xgboost.pkl")


if __name__ == "__main__":
    train()