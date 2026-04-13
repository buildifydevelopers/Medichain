"""
FirebaseService — Async Firebase Realtime Database operations
Stores: patient embeddings (JSON) + serialized SVM models (base64)
Uses firebase-admin SDK with service account credentials
"""

import os
import json
import base64
import logging
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

import firebase_admin
from firebase_admin import credentials, db

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=4)


class FirebaseService:
    """
    Firebase Realtime Database schema:
    
    medichain_face/
    ├── patients/
    │   ├── {patient_id}/
    │   │   ├── name: str
    │   │   ├── enrolled_at: str (ISO timestamp)
    │   │   ├── embeddings: [[512 floats], ...]
    │   │   └── svm_model: str (base64-encoded pickle)
    """

    DB_PATH = "medichain_face"

    def __init__(self):
        self._initialize_firebase()

    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK from environment variable."""
        try:
            if not firebase_admin._apps:
                # Load service account from env var (Railway secret)
                service_account_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
                database_url = os.environ.get("FIREBASE_DATABASE_URL")

                if not service_account_json:
                    raise ValueError("FIREBASE_SERVICE_ACCOUNT env var not set")
                if not database_url:
                    raise ValueError("FIREBASE_DATABASE_URL env var not set")

                service_account_dict = json.loads(service_account_json)
                cred = credentials.Certificate(service_account_dict)
                firebase_admin.initialize_app(cred, {
                    'databaseURL': database_url
                })

            logger.info("Firebase initialized successfully")

        except Exception as e:
            logger.error(f"Firebase initialization failed: {e}")
            raise

    def _safe_patient_id(self, patient_id: str) -> str:
        """
        Firebase keys cannot contain: . # $ [ ]
        Replace with underscore for safe storage.
        """
        for char in ['.', '#', '$', '[', ']', '/']:
            patient_id = patient_id.replace(char, '_')
        return patient_id

    # ─────────────────────────────────────────────
    # Sync operations (called via executor)
    # ─────────────────────────────────────────────

    def _store_embeddings_sync(self, patient_id: str, patient_name: str,
                                embeddings: list) -> dict:
        from datetime import datetime, timezone
        try:
            safe_id = self._safe_patient_id(patient_id)
            ref = db.reference(f"{self.DB_PATH}/patients/{safe_id}")

            # Merge with existing embeddings if re-enrolling
            existing = ref.get()
            existing_embeddings = []
            if existing and "embeddings" in existing:
                existing_embeddings = existing["embeddings"]

            all_embeddings = existing_embeddings + embeddings
            # Cap at 20 embeddings max to control storage
            if len(all_embeddings) > 20:
                all_embeddings = all_embeddings[-20:]

            ref.set({
                "name": patient_name,
                "patient_id": patient_id,
                "enrolled_at": datetime.now(timezone.utc).isoformat(),
                "embeddings": all_embeddings,
                "embedding_count": len(all_embeddings)
            })
            return {"success": True}
        except Exception as e:
            logger.error(f"Store embeddings failed: {e}")
            return {"success": False, "error": str(e)}

    def _get_patient_embeddings_sync(self, patient_id: str) -> dict:
        try:
            safe_id = self._safe_patient_id(patient_id)
            ref = db.reference(f"{self.DB_PATH}/patients/{safe_id}")
            data = ref.get()

            if not data or "embeddings" not in data:
                return {"success": False, "error": "Patient not found or no embeddings"}

            return {"success": True, "embeddings": data["embeddings"], "name": data.get("name")}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_all_other_embeddings_sync(self, exclude_patient_id: str) -> dict:
        """Fetch embeddings from ALL patients except the given one (for SVM negatives)."""
        try:
            safe_exclude = self._safe_patient_id(exclude_patient_id)
            ref = db.reference(f"{self.DB_PATH}/patients")
            all_patients = ref.get()

            if not all_patients:
                return {"success": True, "embeddings": []}

            all_embeddings = []
            for pid, pdata in all_patients.items():
                if pid == safe_exclude:
                    continue
                if pdata and "embeddings" in pdata:
                    # Take max 5 per patient to balance
                    embs = pdata["embeddings"][:5]
                    all_embeddings.extend(embs)

            return {"success": True, "embeddings": all_embeddings}
        except Exception as e:
            return {"success": False, "error": str(e), "embeddings": []}

    def _store_svm_model_sync(self, patient_id: str, model_bytes: bytes) -> dict:
        try:
            safe_id = self._safe_patient_id(patient_id)
            ref = db.reference(f"{self.DB_PATH}/patients/{safe_id}/svm_model")
            model_b64 = base64.b64encode(model_bytes).decode('utf-8')
            ref.set(model_b64)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_svm_model_sync(self, patient_id: str) -> dict:
        try:
            safe_id = self._safe_patient_id(patient_id)
            ref = db.reference(f"{self.DB_PATH}/patients/{safe_id}/svm_model")
            model_b64 = ref.get()

            if not model_b64:
                return {"success": False, "error": "No trained model found"}

            model_bytes = base64.b64decode(model_b64)
            return {"success": True, "model_bytes": model_bytes}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _delete_patient_sync(self, patient_id: str) -> dict:
        try:
            safe_id = self._safe_patient_id(patient_id)
            ref = db.reference(f"{self.DB_PATH}/patients/{safe_id}")
            if not ref.get():
                return {"success": False, "error": "Patient not found"}
            ref.delete()
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ─────────────────────────────────────────────
    # Async wrappers
    # ─────────────────────────────────────────────

    async def store_embeddings(self, patient_id: str, patient_name: str, embeddings: list) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor, self._store_embeddings_sync, patient_id, patient_name, embeddings
        )

    async def get_patient_embeddings(self, patient_id: str) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor, self._get_patient_embeddings_sync, patient_id
        )

    async def get_all_other_embeddings(self, exclude_patient_id: str) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor, self._get_all_other_embeddings_sync, exclude_patient_id
        )

    async def store_svm_model(self, patient_id: str, model_bytes: bytes) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor, self._store_svm_model_sync, patient_id, model_bytes
        )

    async def get_svm_model(self, patient_id: str) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor, self._get_svm_model_sync, patient_id
        )

    async def delete_patient(self, patient_id: str) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor, self._delete_patient_sync, patient_id
        )