"""main.py - FastAPI app.   Local:  python -m uvicorn main:app --reload"""
from typing import Literal, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
import engine

app = FastAPI(title="Student Early-Warning API",
              description="Predicts Dropout / Enrolled / Graduate from admission, semester-1 and semester-2 data "
                          "(Logistic Regression, UCI 697, nested-CV validated).", version="2.0")


class Semester(BaseModel):
    registered: int = Field(..., ge=0, le=40, description="Subjects registered")
    passed: int = Field(..., ge=0, le=40, description="Subjects passed")
    sgpa: float = Field(..., ge=0, le=10, description="SGPA on a 0-10 scale (0 if nothing was evaluated)")

    @model_validator(mode="after")
    def passed_not_more_than_registered(self):
        if self.passed > self.registered:
            raise ValueError("passed cannot be greater than registered")
        return self


class Student(BaseModel):
    previous_qualification_percent: float = Field(..., ge=0, le=100, description="12th / previous qualification %")
    admission_percent: Optional[float] = Field(None, ge=0, le=100, description="Admission score %, leave out if unknown")
    hostel: int = Field(..., ge=0, le=1, description="1 = studying away from home / hostel")
    first_generation: int = Field(..., ge=0, le=1, description="1 = neither parent attended higher education")
    scholarship: int = Field(..., ge=0, le=1)
    fee_dues: int = Field(0, ge=0, le=1, description="1 = fees pending (used from semester 1 onwards)")
    semester1: Optional[Semester] = Field(None, description="Leave out at admission time (checkpoint A)")
    semester2: Optional[Semester] = Field(None, description="Add after semester 2 (checkpoint C)")
    # ---- new in v2 (all optional; leave out if unknown) ----
    age_at_enrollment: Optional[int] = Field(None, ge=14, le=80, description="Age in years at admission")
    gender: Optional[Literal["male", "female"]] = Field(None, description="Leave out if not recorded")
    special_needs: Optional[int] = Field(None, ge=0, le=1, description="1 = has a recorded special educational need")
    international: Optional[int] = Field(None, ge=0, le=1, description="1 = not a national of the country (usually 0 in India)")
    mother_occupation_group: Optional[int] = Field(None, ge=0, le=9, description="Broad occupation group 1-9 (NCO-2015 division / ISCO major group). 0 or omitted = unknown")
    father_occupation_group: Optional[int] = Field(None, ge=0, le=9, description="Same scale as mother_occupation_group")

    model_config = {"json_schema_extra": {"example": {
        "previous_qualification_percent": 72, "admission_percent": 68, "hostel": 1, "first_generation": 1,
        "scholarship": 0, "fee_dues": 1,
        "age_at_enrollment": 19, "gender": "female", "special_needs": 0, "international": 0,
        "mother_occupation_group": 9, "father_occupation_group": 7,
        "semester1": {"registered": 6, "passed": 3, "sgpa": 5.5},
        "semester2": {"registered": 6, "passed": 2, "sgpa": 4.8}}}}


@app.get("/")
def root():
    return {"message": "Student Early-Warning API is running. Open /docs to test.", "endpoints": ["/health", "/predict", "/docs"]}


@app.get("/health")
def health():
    return {"status": "ok", "checkpoints_loaded": list(engine.BUNDLE["models"])}


@app.post("/predict")
def predict(student: Student):
    try:
        return engine.predict(student.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
