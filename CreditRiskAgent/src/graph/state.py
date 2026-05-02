"""
State definition for the Agentic Credit Risk Assessment System.

This module defines the state structure used by LangGraph to orchestrate
the multi-agent credit risk workflow. Each agent updates specific fields
as the borrower progresses through the pipeline.
"""

from typing import TypedDict, Optional, Dict, List, Any
from typing_extensions import Annotated
from langgraph.graph.message import add_messages


class CreditRiskState(TypedDict):
    """
    State schema for the credit risk assessment workflow.
    
    This state is passed between agents and updated incrementally as
    each stage completes. LangGraph manages state transitions automatically.
    
    Fields:
    -------
    borrower_id : str
        Unique identifier for the borrower being assessed
    
    borrower_data : Optional[Dict[str, Any]]
        Raw borrower input data (demographics, financials, credit history)
    
    preprocessed_data : Optional[Dict[str, Any]]
        Cleaned and encoded data after preprocessing agent
        (missing values handled, outliers removed, categorical encoding)
    
    engineered_features : Optional[Dict[str, float]]
        Derived financial features (debt-to-income ratio, utilization, etc.)
        Created by feature engineering agent
    
    pd_score : Optional[float]
        Probability of Default score (0.0 to 1.0) from ML models
        Primary output from risk scoring agent
    
    risk_class : Optional[str]
        Risk classification: "Low", "Medium", "High", "Critical"
        Derived from pd_score using thresholds
    
    stress_test_results : Optional[Dict[str, Dict[str, float]]]
        Risk scores under economic scenarios:
        {
            "baseline": {"pd_score": 0.25, "risk_class": "Medium"},
            "mild_recession": {"pd_score": 0.32, ...},
            "severe_recession": {"pd_score": 0.48, ...},
            "high_interest": {"pd_score": 0.35, ...}
        }
    
    monitoring_metrics : Optional[Dict[str, Any]]
        Real-time monitoring data:
        {
            "psi_score": 0.08,  # Population Stability Index
            "feature_drift": {"income": 0.12, "debt": 0.05},
            "last_checked": "2024-01-15T10:30:00Z"
        }
    
    drift_detected : bool
        Flag indicating if significant drift was detected (default: False)
        Triggers re-scoring when True
    
    alerts : List[Dict[str, str]]
        System alerts and warnings:
        [
            {"level": "warning", "message": "Income decreased by 25%"},
            {"level": "critical", "message": "PSI threshold exceeded"}
        ]
    
    decision : Optional[str]
        Final decision: "Approved", "Rejected", "Manual Review"
    
    explanations : Optional[Dict[str, Any]]
        Explainability outputs:
        {
            "shap_values": {...},
            "top_features": [("debt_ratio", 0.35), ("income", -0.22)],
            "natural_language": "Risk is elevated due to high debt ratio..."
        }
    
    requires_rescoring : bool
        Flag to trigger workflow re-entry (default: False)
        Set by monitoring agent when drift exceeds threshold
    
    messages : Annotated[List, add_messages]
        Conversation history between agents and system
        LangGraph automatically manages message accumulation
    """
    
    # Identifiers
    borrower_id: str
    
    # Data pipeline outputs
    borrower_data: Optional[Dict[str, Any]]
    preprocessed_data: Optional[Dict[str, Any]]
    engineered_features: Optional[Dict[str, float]]
    
    # Risk assessment outputs
    pd_score: Optional[float]
    risk_class: Optional[str]
    
    # Stress testing outputs
    stress_test_results: Optional[Dict[str, Dict[str, float]]]
    
    # Monitoring outputs
    monitoring_metrics: Optional[Dict[str, Any]]
    drift_detected: bool
    alerts: List[Dict[str, str]]
    
    # Final decision
    decision: Optional[str]
    
    # Explainability
    explanations: Optional[Dict[str, Any]]
    
    # Workflow control
    requires_rescoring: bool
    
    # Message history (LangGraph managed)
    messages: Annotated[List, add_messages]


# Type aliases for common state components
RawBorrowerData = Dict[str, Any]
ProcessedData = Dict[str, Any]
FeatureDict = Dict[str, float]
StressTestResults = Dict[str, Dict[str, float]]
MonitoringMetrics = Dict[str, Any]
Explanations = Dict[str, Any]
AlertList = List[Dict[str, str]]


def create_initial_state(borrower_id: str, borrower_data: RawBorrowerData) -> CreditRiskState:
    """
    Create initial state for a new borrower assessment.
    
    Parameters:
    -----------
    borrower_id : str
        Unique borrower identifier
    borrower_data : RawBorrowerData
        Raw borrower data dictionary
    
    Returns:
    --------
    CreditRiskState
        Initialized state ready for agent processing
    
    Example:
    --------
    >>> data = {
    ...     "age": 35,
    ...     "income": 50000,
    ...     "credit_limit": 20000,
    ...     "balance": 12000
    ... }
    >>> state = create_initial_state("B00123", data)
    """
    return CreditRiskState(
        borrower_id=borrower_id,
        borrower_data=borrower_data,
        preprocessed_data=None,
        engineered_features=None,
        pd_score=None,
        risk_class=None,
        stress_test_results=None,
        monitoring_metrics=None,
        drift_detected=False,
        alerts=[],
        decision=None,
        explanations=None,
        requires_rescoring=False,
        messages=[]
    )
