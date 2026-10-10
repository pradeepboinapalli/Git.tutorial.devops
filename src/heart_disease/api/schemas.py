from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    age: float = Field(..., ge=0, le=120)
    sex: float = Field(..., ge=0, le=1)
    cp: float = Field(..., ge=0, le=3)
    trestbps: float = Field(..., ge=80, le=250)
    chol: float = Field(..., ge=100, le=600)
    fbs: float = Field(..., ge=0, le=1)
    restecg: float = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=60, le=220)
    exang: float = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: float = Field(..., ge=0, le=2)
    ca: float = Field(..., ge=0, le=4)
    thal: float = Field(..., ge=0, le=3)


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    probabilities: dict[str, float]
