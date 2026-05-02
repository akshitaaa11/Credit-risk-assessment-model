# src/api/app.py

import sys
import os

# Fix import path (IMPORTANT for Windows projects)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any

# Import your LangGraph workflow
from src.graph.workflow import workflow


# ---------------------------------------
# FASTAPI APP
# ---------------------------------------
app = FastAPI(
    title="Credit Risk API",
    description="LangGraph-powered Credit Risk Assessment",
    version="1.0"
)

# Enable CORS (for Streamlit frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------
# REQUEST SCHEMA
# ---------------------------------------
class CreditRequest(BaseModel):
    applicant_name: str = Field(..., example="Utkarsh")
    age: int = Field(..., example=25)
    income: float = Field(..., example=50000)
    existing_debt: float = Field(..., example=20000)
    credit_score: int = Field(..., example=650)
    loan_amount: float = Field(..., example=30000)
    employment_status: str = Field(..., example="employed")


# ---------------------------------------
# ROOT ENDPOINT
# ---------------------------------------
@app.get("/")
def root():
    return {"message": "Credit Risk API is running"}


# ---------------------------------------
# HEALTH CHECK
# ---------------------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}


# ---------------------------------------
# MAIN ANALYSIS ENDPOINT
# ---------------------------------------
@app.post("/analyze")
def analyze(data: CreditRequest):

    try:
        # Convert request to dictionary
        input_data = data.dict()

        # Call LangGraph workflow
        result: Dict[str, Any] = workflow.invoke({
            "customer_data": input_data
        })

        # Extract outputs safely
        decision = result.get("final_decision", {})
        risk = result.get("risk_score", {})
        explanation = result.get("explanation", "")
        stress = result.get("stress_results", {})

        return {
            "applicant_name": input_data.get("applicant_name"),
            "decision": decision.get("action"),
            "confidence": decision.get("confidence"),
            "pd": risk.get("pd"),
            "risk_category": risk.get("risk_category"),
            "explanation": explanation,
            "stress_results": stress
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )