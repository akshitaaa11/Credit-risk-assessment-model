# src/graph/workflow.py

from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END

# ✅ function-based agent
from src.agents.risk_scoring_agent import risk_scoring_agent


# -------------------------------
# STATE
# -------------------------------
class CreditRiskState(TypedDict, total=False):
    customer_data: Dict[str, Any]
    engineered_features: Dict[str, float]
    risk_score: Dict[str, Any]
    explanation: str
    stress_results: Dict[str, float]
    monitoring_status: str
    final_decision: Dict[str, Any]


# -------------------------------
# 1. PREPROCESSING
# -------------------------------
def preprocessing_agent(state: CreditRiskState) -> CreditRiskState:
    data = state.get("customer_data", {})

    features = {
        "age": float(data.get("age", 0)),
        "income": float(data.get("income", 0)),
        "loan_amount": float(data.get("loan_amount", 0)),

        "debt_to_income": float(data.get("existing_debt", 0)) / (float(data.get("income", 1)) + 1),
        "credit_utilization": float(data.get("existing_debt", 0)) / (float(data.get("loan_amount", 1)) + 1),

        "repayment_history": 0.9 if data.get("credit_score", 0) > 700 else 0.6,
        "employment_encoded": 1 if data.get("employment_status") == "employed" else 0,
        "credit_history_encoded": 1 if data.get("credit_score", 0) > 650 else 0
    }

    return {"engineered_features": features}


# -------------------------------
# 2. RISK SCORING
# -------------------------------
def risk_scoring_node(state: CreditRiskState) -> CreditRiskState:
    return risk_scoring_agent(state)


# -------------------------------
# 3. EXPLAINABILITY
# -------------------------------
def explainability_agent(state: CreditRiskState) -> CreditRiskState:
    features = state.get("engineered_features", {})

    reasons = []

    if features.get("debt_to_income", 0) > 0.5:
        reasons.append("High debt-to-income ratio increased risk")

    if features.get("repayment_history", 1) < 0.7:
        reasons.append("Poor repayment history increased risk")

    if features.get("credit_history_encoded", 1) == 0:
        reasons.append("Weak credit history increased risk")

    if not reasons:
        reasons.append("Stable financial profile")

    return {"explanation": " | ".join(reasons[:3])}


# -------------------------------
# 4. STRESS TEST
# -------------------------------
def stress_test_agent(state: CreditRiskState) -> CreditRiskState:
    base_pd = state["risk_score"]["pd"]

    shock_pd = min(base_pd * 1.2, 1.0)

    return {
        "stress_results": {
            "base": base_pd,
            "shock_20pct": shock_pd
        }
    }


# -------------------------------
# 5. MONITORING
# -------------------------------
def monitoring_agent(state: CreditRiskState) -> CreditRiskState:
    return {"monitoring_status": "healthy"}


# -------------------------------
# 6. DECISION ROUTER
# -------------------------------
def router_node(state: CreditRiskState) -> CreditRiskState:
    pd = state["risk_score"]["pd"]

    if pd < 0.3:
        action = "APPROVE"
        confidence = 0.9
    elif pd < 0.6:
        action = "HOLD"
        confidence = 0.7
    else:
        action = "REJECT"
        confidence = 0.92

    return {
        "final_decision": {
            "action": action,
            "confidence": confidence
        }
    }


# -------------------------------
# WORKFLOW
# -------------------------------
def build_workflow():
    graph = StateGraph(CreditRiskState)

    graph.add_node("preprocessing", preprocessing_agent)
    graph.add_node("risk_scoring", risk_scoring_node)
    graph.add_node("explainability", explainability_agent)
    graph.add_node("stress_test", stress_test_agent)
    graph.add_node("monitoring", monitoring_agent)
    graph.add_node("router", router_node)

    graph.set_entry_point("preprocessing")

    graph.add_edge("preprocessing", "risk_scoring")
    graph.add_edge("risk_scoring", "explainability")
    graph.add_edge("explainability", "stress_test")
    graph.add_edge("stress_test", "monitoring")
    graph.add_edge("monitoring", "router")
    graph.add_edge("router", END)

    return graph.compile()


workflow = build_workflow()


# -------------------------------
# TEST
# -------------------------------
if __name__ == "__main__":
    test_data = {
        "age": 25,
        "income": 50000,
        "existing_debt": 20000,
        "credit_score": 650,
        "loan_amount": 30000,
        "employment_status": "employed"
    }

    result = workflow.invoke({"customer_data": test_data})

    print("\n===== OUTPUT =====")
    print(result)