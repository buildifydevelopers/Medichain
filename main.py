"""
MediChain — Patient Face Verification Backend
FaceNet (InceptionResnetV1) + Per-Patient SVM Classifier
FastAPI | Railway-Ready | Firebase Storage
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import logging

from face_service import FaceService
from models.schemas import EnrollResponse, TrainResponse, VerifyResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MediChain Face Verification API",
    description="FaceNet + SVM patient identity verification for MediChain",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

face_service = FaceService()


@app.get("/")
async def root():
    return {"status": "MediChain Face API running", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": face_service.model_loaded}


@app.post("/enroll", response_model=EnrollResponse)
async def enroll_patient(
    patient_id: str = Form(..., description="Unique patient ID (ABHA ID or UUID)"),
    patient_name: str = Form(..., description="Patient full name"),
    photos: list[UploadFile] = File(..., description="3-10 face photos for enrollment")
):
    """
    Enroll a new patient by uploading 3-10 face photos.
    Extracts FaceNet embeddings and stores them in Firebase.
    Minimum 3 photos required for reliable SVM training.
    """
    if len(photos) < 3:
        raise HTTPException(
            status_code=400,
            detail="Minimum 3 photos required for reliable enrollment. Please provide 3-10 photos."
        )
    if len(photos) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 photos allowed per enrollment session."
        )

    photo_bytes_list = []
    for photo in photos:
        if photo.content_type not in ["image/jpeg", "image/png", "image/webp"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {photo.content_type}. Use JPEG, PNG, or WebP."
            )
        photo_bytes_list.append(await photo.read())

    result = await face_service.enroll_patient(patient_id, patient_name, photo_bytes_list)

    if not result["success"]:
        raise HTTPException(status_code=422, detail=result["error"])

    return EnrollResponse(
        patient_id=patient_id,
        patient_name=patient_name,
        embeddings_stored=result["embeddings_count"],
        message=f"Successfully enrolled {result['embeddings_count']} face embeddings. Call /train to activate."
    )


@app.post("/train", response_model=TrainResponse)
async def train_patient_svm(
    patient_id: str = Form(..., description="Patient ID to train SVM for")
):
    """
    Train a per-patient SVM classifier using stored embeddings.
    Must be called after /enroll. Creates a binary classifier (this patient vs. not).
    """
    result = await face_service.train_svm(patient_id)

    if not result["success"]:
        raise HTTPException(status_code=422, detail=result["error"])

    return TrainResponse(
        patient_id=patient_id,
        model_accuracy=result["cv_accuracy"],
        embeddings_used=result["embeddings_used"],
        message="SVM trained successfully. Patient verification is now active."
    )


@app.post("/verify", response_model=VerifyResponse)
async def verify_patient(
    patient_id: str = Form(..., description="Patient ID to verify against"),
    photo: UploadFile = File(..., description="Live face photo from camera")
):
    """
    Verify if the person in the photo matches the enrolled patient.
    Returns binary yes/no with confidence score.
    Target accuracy: 98-99%.
    """
    if photo.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use JPEG, PNG, or WebP."
        )

    photo_bytes = await photo.read()
    result = await face_service.verify_patient(patient_id, photo_bytes)

    if not result["success"]:
        raise HTTPException(status_code=422, detail=result["error"])

    return VerifyResponse(
        patient_id=patient_id,
        is_match=result["is_match"],
        confidence=result["confidence"],
        message="Identity verified" if result["is_match"] else "Identity NOT verified — face does not match records"
    )


@app.delete("/patient/{patient_id}")
async def delete_patient(patient_id: str):
    """Remove all face data for a patient (GDPR / DPDP Act compliance)."""
    result = await face_service.delete_patient(patient_id)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["error"])
    return {"message": f"All face data deleted for patient {patient_id}"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
