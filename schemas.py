from pydantic import BaseModel
from typing import Optional


class EnrollResponse(BaseModel):
    patient_id: str
    patient_name: str
    embeddings_stored: int
    message: str


class TrainResponse(BaseModel):
    patient_id: str
    model_accuracy: float
    embeddings_used: int
    message: str


class VerifyResponse(BaseModel):
    patient_id: str
    is_match: bool
    confidence: float
    message: str
    # Debug fields — visible in API response, useful during testing
    svm_confidence: Optional[float] = None
    cosine_confidence: Optional[float] = None


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
