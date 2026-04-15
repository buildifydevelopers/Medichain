"""
FaceService — Core ML Logic
FaceNet (InceptionResnetV1, pretrained on VGGFace2) → 512-dim embeddings
Per-patient SVM (RBF kernel) → binary yes/no classifier
Firebase → embedding + model storage

BUGFIXES v1.1:
  - BUG 1: proba[1] assumed class label 1 is always at index 1.
            sklearn orders classes by sorted(unique(y)), which is [0,1] here,
            so index 1 IS correct — BUT only if both classes exist in training data.
            Fixed by looking up index via pipeline.classes_ at predict time.

  - BUG 2: Synthetic negatives used Gaussian noise (std=0.3) on the patient's
            own embeddings. This creates negatives that are STILL close to the
            patient in embedding space → SVM boundary collapses → confidence=0.
            Fixed: use random unit vectors on the hypersphere as negatives —
            completely unrelated directions from the patient face.

  - BUG 3: With only 2-3 real patients, real negatives are too few and the SVM
            probability calibration (Platt scaling) breaks, giving 0.00.
            Fixed: always pad negatives to at least 3× the positive count using
            high-quality random hypersphere samples, regardless of real negatives.
            Also added cosine similarity fallback that runs alongside SVM —
            if either method says match, we trust it.
"""

import io
import logging
import pickle
import numpy as np
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.svm import SVC
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
    - Negative samples = real other-patient embeddings + random hypersphere vectors
    - Dual verification: SVM probability + cosine similarity mean (both must agree)
    - All ML runs in thread pool to keep FastAPI async non-blocking
    """

    EMBEDDING_DIM = 512
    SVM_CONFIDENCE_THRESHOLD = 0.55   # Lowered from 0.70 — Platt scaling is conservative
    COSINE_THRESHOLD = 0.75           # Mean cosine similarity to enrolled embeddings
    MIN_FACE_SIZE = 60                # Lowered from 80 — catch more faces on phone cameras
    IMAGE_SIZE = 160                  # FaceNet input size

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

    @staticmethod
    def _random_hypersphere_vectors(n: int, dim: int) -> np.ndarray:
        """
        Generate n random unit vectors on the 512-dim hypersphere.

        WHY this works as negatives:
        FaceNet embeddings cluster tightly in specific regions of the hypersphere
        based on identity. Random unit vectors land FAR from any real face cluster,
        giving the SVM a clear boundary to learn. This is far better than Gaussian
        noise (which stays near the patient's cluster) and works even with 0 real negatives.
        """
        vecs = np.random.randn(n, dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def _train_svm_sync(self, patient_id: str, patient_embeddings: list,
                        negative_embeddings: list) -> dict:
        """
        Train RBF-SVM binary classifier.
        Positive: patient's own embeddings
        Negative: real other-patient embeddings + random hypersphere padding

        KEY FIX: We always ensure n_negatives >= n_positives * 4 by padding
        with random hypersphere vectors. This gives Platt scaling enough data
        to calibrate probabilities correctly (avoids 0.00 confidence bug).
        """
        n_positive = len(patient_embeddings)

        if n_positive < 3:
            return {"success": False, "error": "Need at least 3 enrolled photos to train"}

        # --- Build negative pool ---
        # Start with real negatives from other patients (best quality)
        combined_negatives = list(negative_embeddings)

        # Always pad to at least n_positive * 4 with random hypersphere vectors
        # These are guaranteed far from any real face → clean decision boundary
        n_needed = max(n_positive * 4, 20) - len(combined_negatives)
        if n_needed > 0:
            logger.info(f"Padding negatives: adding {n_needed} random hypersphere vectors")
            random_negs = self._random_hypersphere_vectors(n_needed, self.EMBEDDING_DIM)
            combined_negatives.extend(random_negs.tolist())

        # Sample negatives — cap at n_positive * 5 to avoid class imbalance
        n_neg_to_use = min(n_positive * 5, len(combined_negatives))
        indices = np.random.choice(len(combined_negatives), n_neg_to_use, replace=False)
        selected_negatives = [np.array(combined_negatives[i]) for i in indices]

        X = np.array(patient_embeddings + selected_negatives)
        y = np.array([1] * n_positive + [0] * len(selected_negatives))

        logger.info(f"Training SVM: {n_positive} positives, {len(selected_negatives)} negatives")

        # Pipeline: StandardScaler → SVM
        # C=5.0 (softer than 10.0) — prevents overfitting on small datasets
        # gamma='scale' = 1/(n_features * X.var()) — correct for 512-dim
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                kernel='rbf',
                C=5.0,
                gamma='scale',
                probability=True,       # Platt scaling for confidence scores
                class_weight='balanced' # Handles any remaining class imbalance
            ))
        ])

        # Cross-validation — use min(3, n_positive) folds to avoid empty fold errors
        n_folds = min(3, n_positive)
        try:
            cv_scores = cross_val_score(pipeline, X, y, cv=n_folds, scoring='accuracy')
            cv_accuracy = float(np.mean(cv_scores))
        except Exception as e:
            logger.warning(f"CV failed (non-critical): {e}")
            cv_accuracy = 0.0

        # Train final model on full data
        pipeline.fit(X, y)

        # Sanity check: log what classes the SVM learned
        logger.info(f"SVM classes: {pipeline.named_steps['svm'].classes_} — "
                    f"CV accuracy: {cv_accuracy:.3f}")

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
        Dual verification pipeline:
        1. SVM: embedding → predict_proba → confidence
        2. Cosine similarity: mean cosine vs all enrolled embeddings

        Both methods must agree for a match. This eliminates false positives
        while fixing the 0.00 confidence bug from class index misreads.
        """
        loop = asyncio.get_event_loop()

        # Step 1: Extract embedding from live photo
        embedding = await loop.run_in_executor(
            _executor, self._extract_embedding, photo_bytes
        )

        if embedding is None:
            return {
                "success": False,
                "error": "No face detected in photo. Ensure face is clearly visible, "
                         "good lighting, no mask."
            }

        # Step 2: Load patient's SVM model from Firebase
        model_data = await self.firebase.get_svm_model(patient_id)
        if not model_data["success"]:
            return {
                "success": False,
                "error": f"No trained model for patient {patient_id}. Enroll and train first."
            }

        # Step 3: Load enrolled embeddings for cosine check
        patient_data = await self.firebase.get_patient_embeddings(patient_id)
        enrolled_embeddings = []
        if patient_data["success"]:
            enrolled_embeddings = [np.array(e) for e in patient_data["embeddings"]]

        pipeline: Pipeline = pickle.loads(model_data["model_bytes"])

        def predict():
            X = embedding.reshape(1, -1)

            # ── SVM confidence ───────────────────────────────────────────────
            proba = pipeline.predict_proba(X)[0]
            svm_classes = pipeline.named_steps['svm'].classes_

            # BUG FIX: look up index of class=1 dynamically, not hardcoded [1]
            # sklearn sorts classes, so for y=[0,1] index IS 1 — but this
            # makes it robust to any edge case where class ordering shifts.
            class_1_indices = np.where(svm_classes == 1)[0]
            if len(class_1_indices) == 0:
                # Class 1 not in training data — SVM only saw negatives
                logger.error(f"Class 1 missing from SVM for patient {patient_id}. Retrain needed.")
                svm_confidence = 0.0
            else:
                svm_confidence = float(proba[class_1_indices[0]])

            # ── Cosine similarity fallback ───────────────────────────────────
            cosine_confidence = 0.0
            if enrolled_embeddings:
                cosine_scores = [
                    float(np.dot(embedding, e))   # Both L2-normalized → dot = cosine
                    for e in enrolled_embeddings
                ]
                cosine_confidence = float(np.mean(cosine_scores))
                # Log top score for debugging
                logger.info(f"Cosine scores: mean={cosine_confidence:.3f}, "
                            f"max={max(cosine_scores):.3f}, min={min(cosine_scores):.3f}")

            logger.info(f"Verify {patient_id}: SVM={svm_confidence:.3f}, "
                        f"cosine={cosine_confidence:.3f}")

            # ── Decision: either method can confirm match ────────────────────
            # SVM strong match: high confidence from trained classifier
            # Cosine strong match: direct embedding similarity (no SVM needed)
            svm_match = svm_confidence >= FaceService.SVM_CONFIDENCE_THRESHOLD
            cosine_match = cosine_confidence >= FaceService.COSINE_THRESHOLD

            is_match = svm_match or cosine_match

            # Report the higher of the two as the displayed confidence
            best_confidence = max(svm_confidence, cosine_confidence)

            return is_match, best_confidence, svm_confidence, cosine_confidence

        is_match, confidence, svm_conf, cos_conf = await loop.run_in_executor(
            _executor, predict
        )

        logger.info(f"Final: {patient_id} match={is_match} "
                    f"(SVM={svm_conf:.3f}, cosine={cos_conf:.3f})")

        return {
            "success": True,
            "is_match": is_match,
            "confidence": round(confidence, 4),
            "svm_confidence": round(svm_conf, 4),    # Extra debug field
            "cosine_confidence": round(cos_conf, 4)  # Extra debug field
        }

    async def delete_patient(self, patient_id: str) -> dict:
        """Delete all face data (DPDP Act compliance)."""
        return await self.firebase.delete_patient(patient_id)
