from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ProfitInputs(BaseModel):
    funded_amnt: Optional[float] = Field(default=None, gt=0)
    term_months: Optional[int] = Field(default=None, gt=0)
    installment: Optional[float] = Field(default=None, gt=0)


class RejectedRiskRequest(ProfitInputs):
    model_config = ConfigDict(extra="forbid")

    amount_requested: float = Field(gt=0)
    risk_score: float = Field(ge=300, le=900)
    dti: float = Field(ge=0)
    zip_code: str = Field(min_length=3)
    state: str = Field(min_length=2, max_length=2)
    employment_length: str


class AcceptedScoreRequest(ProfitInputs):
    model_config = ConfigDict(extra="forbid")

    loan_amnt: float = Field(gt=0)
    int_rate: float = Field(ge=0)
    annual_inc: float = Field(ge=0)
    dti: float = Field(ge=0)
    fico_range_low: float = Field(ge=300, le=900)
    fico_range_high: float = Field(ge=300, le=900)
    delinq_2yrs: float = 0
    inq_last_6mths: float = 0
    open_acc: float = 0
    pub_rec: float = 0
    revol_bal: float = 0
    revol_util: float = 0
    total_acc: float = 0
    mort_acc: float = 0
    acc_open_past_24mths: float = 0
    pub_rec_bankruptcies: float = 0
    grade: str
    sub_grade: str
    emp_length: str
    home_ownership: str
    verification_status: str
    purpose: str
    addr_state: str
    application_type: str = "Individual"
    initial_list_status: str = "w"


class ScoreResponse(BaseModel):
    p_default: float
    decision: str
    reason: str
    expected_profit: Optional[float] = None
    expected_return: Optional[float] = None
    lgd: Optional[float] = None
    required_return: Optional[float] = None
    approval_rule: Optional[str] = None
    model_note: Optional[str] = None
