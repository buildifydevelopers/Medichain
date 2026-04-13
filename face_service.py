"""
FaceService — Core ML Logic
FaceNet (InceptionResnetV1, pretrained on VGGFace2) → 512-dim embeddings
Per-patient SVM (RBF kernel) → binary yes/no classifier
Firebase → embedding + model storage
"""

import io
import logging
import pickle
import numpy as np
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

import cv2
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from services.firebase_service import FirebaseService

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound ML operations
_executor = ThreadPoolExecutor(max_workers=2)


class FaceService:
    """
    Senior-level design decisions:
    - MTCNN for face detection (handles rotation, partial occlusion)
    - InceptionResnetV1 pretrained on VGGFace2 (512-dim, ~99.6% LFW accuracy)
    - Per-patient SVM with RBF kernel (learns face variance per individual)
    - Negative samples = embeddings from other enrolled patients (smart negatives)
    - All ML runs in thread pool to keep FastAPI async non-blocking
    """

    EMBEDDING_DIM = 512
    SVM_CONFIDENCE_THRESHOLD = 0.70   # Tune: higher = stricter
    MIN_FACE_SIZE = 80                 # pixels — reject tiny/blurry faces
    IMAGE_SIZE = 160                   # FaceNet input size

    def __init__(self):
        self.firebase = FirebaseService()
        self._mtcnn: Optional[MTCNN] = None
        self._facenet: Optional[InceptionResnetV1] = None
        self.model_loaded = False
        self._load_models()

    def _load_models(self):
        """Load MTCNN + FaceNet. Called once at startup."""
        try:
            logger.info("Loading MTCNN face detector...")
            self._mtcnn = MTCNN(
                image_size=self.IMAGE_SIZE,
                margin=20,            # Extra context around face
                min_face_size=self.MIN_FACE_SIZE,
                thresholds=[0.6, 0.7, 0.7],  # P-Net, R-Net, O-Net thresholds
                factor=0.709,
                post_process=True,    # Normalize to [-1, 1]
                keep_all=False,       # Only largest face
                device=self._get_device()
            )

            logger.info("Loading FaceNet (InceptionResnetV1, VGGFace2)...")
            self._facenet = InceptionResnetV1(
                pretrained='vggface2',
                classify=False        # Embedding mode, not classification
            ).eval().to(self._get_device())

            self.model_loaded = True
            logger.info(f"Models loaded successfully on {self._get_device()}")

        except Exception as e:
            logger.error(f"CRITICAL: Failed to load models: {e}")
            self.model_loaded = False

    @staticmethod
    def _get_device() -> str:
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    # ─────────────────────────────────────────────
    # PRIVATE: Core ML operations (run in executor)
    # ─────────────────────────────────────────────

    def _bytes_to_pil(self, image_bytes: bytes) -> Image.Image:
        """Convert raw bytes to PIL RGB image."""
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return img

    def _extract_embedding(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Full pipeline: bytes → detect face → FaceNet embedding.
        Returns 512-dim numpy array or None if no face detected.
        """
        try:
            img = self._bytes_to_pil(image_bytes)

            # MTCNN: detect + align + crop face
            face_tensor = self._mtcnn(img)  # Returns [1, 3, 160, 160] or None

            if face_tensor is None:
                logger.warning("No face detected in image")
                return None

            # FaceNet: extract 512-dim embedding
            with torch.no_grad():
                face_tensor = face_tensor.unsqueeze(0).to(self._get_device())
                embedding = self._facenet(face_tensor)
                embedding = embedding.cpu().numpy().flatten()  # (512,)

            # L2 normalize (cosine similarity becomes dot product)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return None

    def _train_svm_sync(self, patient_id: str, patient_embeddings: list,
                        negative_embeddings: list) -> dict:
        """
        Train RBF-SVM binary classifier.
        Positive: patient's own embeddings
        Negative: embeddings from ALL other enrolled patients
        Uses StandardScaler + SVC in a Pipeline for robustness.
        """
        n_positive = len(patient_embeddings)
        n_negative = len(negative_embeddings)

        if n_positive < 3:
            return {"success": False, "error": "Need at least 3 enrolled photos to train"}

        # If no negatives exist yet (first patient), create synthetic negatives
        # by adding Gaussian noise to existing embeddings
        if n_negative < 3:
            logger.info("Generating synthetic negatives (first patient)")
            synthetic_negatives = []
            for emb in patient_embeddings:
                for _ in range(5):
                    noise = np.random.normal(0, 0.3, emb.shape)
                    synthetic = emb + noise
                    synthetic = synthetic / np.linalg.norm(synthetic)
                    synthetic_negatives.append(synthetic)
            negative_embeddings = synthetic_negatives

        # Balance dataset
        n_neg_needed = min(n_positive * 3, len(negative_embeddings))
        neg_indices = np.random.choice(len(negative_embeddings), n_neg_needed, replace=False)
        selected_negatives = [negative_embeddings[i] for i in neg_indices]

        X = np.array(patient_embeddings + selected_negatives)
        y = np.array([1] * n_positive + [0] * len(selected_negatives))

        # Pipeline: normalize → SVM
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                kernel='rbf',
                C=10.0,           # Regularization (higher = tighter boundary)
                gamma='scale',    # Auto-scale with feature count
                probability=True, # Enable probability estimates for confidence
                class_weight='balanced'
            ))
        ])

        # Cross-validation accuracy
        cv_scores = cross_val_score(pipeline, X, y, cv=min(5, n_positive), scoring='accuracy')
        cv_accuracy = float(np.mean(cv_scores))

        # Train on full data
        pipeline.fit(X, y)

        return {
            "success": True,
            "pipeline": pipeline,
            "cv_accuracy": cv_accuracy,
            "embeddings_used": n_positive
        }

    # ─────────────────────────────────────────────
    # PUBLIC: Async API methods
    # ─────────────────────────────────────────────

    async def enroll_patient(self, patient_id: str, patient_name: str,
                              photos: list[bytes]) -> dict:
        """Extract embeddings from all photos and store in Firebase."""
        loop = asyncio.get_event_loop()

        embeddings = []
        failed_photos = 0

        for i, photo_bytes in enumerate(photos):
            embedding = await loop.run_in_executor(
                _executor, self._extract_embedding, photo_bytes
            )
            if embedding is not None:
                embeddings.append(embedding.tolist())
            else:
                failed_photos += 1
                logger.warning(f"Photo {i+1}: no face detected, skipping")

        if len(embeddings) < 3:
            return {
                "success": False,
                "error": f"Only {len(embeddings)} valid faces detected from {len(photos)} photos. "
                         f"Need at least 3. Ensure good lighting, face clearly visible."
            }

        # Store embeddings in Firebase
        store_result = await self.firebase.store_embeddings(
            patient_id=patient_id,
            patient_name=patient_name,
            embeddings=embeddings
        )

        if not store_result["success"]:
            return {"success": False, "error": store_result["error"]}

        logger.info(f"Enrolled {len(embeddings)} embeddings for patient {patient_id}")
        return {"success": True, "embeddings_count": len(embeddings)}

    async def train_svm(self, patient_id: str) -> dict:
        """Load embeddings from Firebase, train SVM, store model back."""
        loop = asyncio.get_event_loop()

        # Fetch this patient's embeddings
        patient_data = await self.firebase.get_patient_embeddings(patient_id)
        if not patient_data["success"]:
            return {"success": False, "error": f"Patient {patient_id} not found. Enroll first."}

        patient_embeddings = [np.array(e) for e in patient_data["embeddings"]]

        # Fetch all OTHER patients' embeddings as negatives
        all_others = await self.firebase.get_all_other_embeddings(patient_id)
        negative_embeddings = [np.array(e) for e in all_others.get("embeddings", [])]

        # Train SVM (CPU-bound, run in executor)
        train_result = await loop.run_in_executor(
            _executor,
            self._train_svm_sync,
            patient_id,
            patient_embeddings,
            negative_embeddings
        )

        if not train_result["success"]:
            return train_result

        # Serialize and store model in Firebase
        model_bytes = pickle.dumps(train_result["pipeline"])
        store_result = await self.firebase.store_svm_model(patient_id, model_bytes)

        if not store_result["success"]:
            return {"success": False, "error": "Model trained but failed to save: " + store_result["error"]}

        logger.info(f"SVM trained for {patient_id} — CV accuracy: {train_result['cv_accuracy']:.3f}")
        return {
            "success": True,
            "cv_accuracy": train_result["cv_accuracy"],
            "embeddings_used": train_result["embeddings_used"]
        }

    async def verify_patient(self, patient_id: str, photo_bytes: bytes) -> dict:
        """
        Full verification pipeline:
        photo → embedding → load patient SVM → predict → return match + confidence
        """
        loop = asyncio.get_event_loop()

        # Step 1: Extract embedding from live photo
        embedding = await loop.run_in_executor(
            _executor, self._extract_embedding, photo_bytes
        )

        if embedding is None:
            return {
                "success": False,
                "error": "No face detected in photo. Ensure face is clearly visible with good lighting."
            }

        # Step 2: Load patient's SVM model from Firebase
        model_data = await self.firebase.get_svm_model(patient_id)
        if not model_data["success"]:
            return {
                "success": False,
                "error": f"No trained model found for patient {patient_id}. "
                         f"Please enroll and train first."
            }

        pipeline: Pipeline = pickle.loads(model_data["model_bytes"])

        # Step 3: Predict (run in executor — synchronous sklearn)
        def predict():
            X = embedding.reshape(1, -1)
            proba = pipeline.predict_proba(X)[0]  # [prob_negative, prob_positive]
            confidence = float(proba[1])           # Probability of being THIS patient
            is_match = confidence >= self.SVM_CONFIDENCE_THRESHOLD
            return is_match, confidence

        is_match, confidence = await loop.run_in_executor(_executor, predict)

        logger.info(f"Verify {patient_id}: match={is_match}, confidence={confidence:.3f}")
        return {
            "success": True,
            "is_match": is_match,
            "confidence": round(confidence, 4)
        }

    async def delete_patient(self, patient_id: str) -> dict:
        """Delete all face data (DPDP Act compliance)."""
        return await self.firebase.delete_patient(patient_id)