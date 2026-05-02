from src.graph.workflow import workflow
from src.graph.state import create_initial_state


def main():
    borrower_data = {
        "age": 28,
        "income": 60000,
        "loan_amount": 25000,
        "employment_status": "employed",
        "credit_history": "good",
        "credit_utilization": 0.4,
        "repayment_history": 0.9
    }

    # Create proper state (VERY IMPORTANT)
    state = create_initial_state(
        borrower_id="B001",
        borrower_data=borrower_data
    )

    result = workflow.invoke(state)

    print("\n===== FINAL OUTPUT =====\n")
    print("Engineered Features:")
    print(result.get("engineered_features"))

    print("\nPD Score:")
    print(result.get("pd_score"))

    print("\nRisk Class:")
    print(result.get("risk_class"))

    print("\n========================\n")


if __name__ == "__main__":
    main()