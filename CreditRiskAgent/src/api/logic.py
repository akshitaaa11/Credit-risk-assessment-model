def analyze_credit_risk(request: dict) -> dict:
    score = 100
    reasons = []

    if request["credit_score"] < 600:
        score -= 30
        reasons.append("Low credit score")
    elif request["credit_score"] < 700:
        score -= 15
        reasons.append("Fair credit score")

    debt_to_income = (request["existing_debt"] + request["loan_amount"]) / request["income"]

    if debt_to_income > 0.5:
        score -= 25
        reasons.append("High debt-to-income ratio")
    elif debt_to_income > 0.3:
        score -= 10
        reasons.append("Moderate debt-to-income ratio")

    if request["employment_status"] == "unemployed":
        score -= 20
        reasons.append("Unemployed status")
    elif request["employment_status"] == "student":
        score -= 10
        reasons.append("Student status")

    if request["age"] < 25:
        score -= 10
        reasons.append("Limited credit history")

    score = max(0, min(100, score))

    if score >= 80:
        risk_level = "Low"
        approval_status = "Approved"
    elif score >= 60:
        risk_level = "Medium"
        approval_status = "Review Needed"
    else:
        risk_level = "High"
        approval_status = "Rejected"

    return {
        "risk_level": risk_level,
        "approval_status": approval_status,
        "risk_score": score,
        "debt_to_income_ratio": round(debt_to_income, 2),
        "reasons": reasons
    }