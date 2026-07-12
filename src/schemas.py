from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

"""
- defines which schema classes are optional and what makes valid input for proper API response.
- also defines what fields the API returns from scorer.py
"""
class AcceptedScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loan_amnt: float = Field(gt=0)
    int_rate: float = Field(ge=0)
    annual_inc: float = Field(ge=0)
    dti: float = Field(ge=0)
    fico_range_low: float = Field(ge=300, le=900)
    fico_range_high: float = Field(ge=300, le=900)
    delinq_2yrs: Optional[float] = Field(default=None, ge=0)
    inq_last_6mths: Optional[float] = Field(default=None, ge=0)
    open_acc: Optional[float] = Field(default=None, ge=0)
    pub_rec: Optional[float] = Field(default=None, ge=0)
    revol_bal: Optional[float] = Field(default=None, ge=0)
    revol_util: Optional[float] = Field(default=None, ge=0)
    total_acc: Optional[float] = Field(default=None, ge=0)
    mort_acc: Optional[float] = Field(default=None, ge=0)
    acc_open_past_24mths: Optional[float] = Field(default=None, ge=0)
    pub_rec_bankruptcies: Optional[float] = Field(default=None, ge=0)
    grade: str
    sub_grade: str
    emp_length: str
    home_ownership: str
    verification_status: str
    purpose: str
    addr_state: str
    application_type: Optional[str] = None
    initial_list_status: Optional[str] = None


class ScoreResponse(BaseModel):
    p_default: float
    p_non_default: float
    decision_margin: float = Field(
        description="2 * abs(p_default - 0.5), a scaled distance from 50/50 default probability; not statistical confidence."
    )
    risk_band: str
    model_version: Optional[str] = None
    model_type: Optional[str] = None
    calibration_method: Optional[str] = None
    scoring_note: str
