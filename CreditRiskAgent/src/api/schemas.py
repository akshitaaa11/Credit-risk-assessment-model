from pydantic import BaseModel, Field
from typing import List

class CreditRiskRequest(BaseModel):
    applicant_name: str = Field(..., min_length=1)
    age: int = Field(..., ge=18, le=100)
    income: float = Field(..., gt=0)
    existing_debt: float = Field(..., ge=0)
    credit_score: int = Field(..., ge=300, le=900)
    loan_amount: float = Field(..., gt=0)
    employment_status: str = Field(..., pattern="^(employed|unemployed|self-employed|student)$")

class CreditRiskResponse(BaseModel):
    applicant_name: str
    risk_level: str
    approval_status: str
    risk_score: int
    debt_to_income_ratio: float
    reasons: List[str]