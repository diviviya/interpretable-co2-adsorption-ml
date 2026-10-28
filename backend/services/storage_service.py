"""
JSON-based persistent storage service.
Handles thread-safe reading, writing, and initialization of data files.
Zero SQL required.
"""
from pathlib import Path
import json
import threading
from datetime import datetime, timezone
import hashlib
import secrets

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = PROJECT_ROOT / "backend" / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()

def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return key.hex(), salt

def _verify_password(password: str, hashed_hex: str, salt: str) -> bool:
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return secrets.compare_digest(key.hex(), hashed_hex)

def _read_json(file_path: Path, default=None):
    if not file_path.exists():
        return default if default is not None else []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else []

def _write_json(file_path: Path, data):
    tmp_path = file_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    tmp_path.replace(file_path)

class JSONStorage:
    USERS_FILE = STORAGE_DIR / "users.json"
    PREDICTIONS_FILE = STORAGE_DIR / "predictions.json"
    MODELS_FILE = STORAGE_DIR / "models.json"
    JOBS_FILE = STORAGE_DIR / "training_jobs.json"
    AUDIT_FILE = STORAGE_DIR / "audit_logs.json"
    DATASETS_FILE = STORAGE_DIR / "datasets.json"

    @classmethod
    def init_storage(cls):
        with _lock:
            # 1. Initialize Users (default admin user: admin / admin123)
            users = _read_json(cls.USERS_FILE, [])
            if not any(u.get("username") == "admin" for u in users):
                pwd_hash, salt = _hash_password("admin123")
                users.append({
                    "id": 1,
                    "username": "admin",
                    "role": "admin",
                    "password_hash": pwd_hash,
                    "salt": salt,
                    "created_at": datetime.now(timezone.utc).isoformat()
                })
                _write_json(cls.USERS_FILE, users)

            # 2. Initialize Model Versions
            models = _read_json(cls.MODELS_FILE, [])
            if not models:
                models = [
                    {
                        "version": "v1.0-ensemble",
                        "name": "Validation-Optimized Weighted Ensemble",
                        "status": "active",
                        "metrics": {"R2": 0.8173, "RMSE": 0.7064, "MAE": 0.5220},
                        "weights": {
                            "CatBoost": 0.5464,
                            "LightGBM": 0.2766,
                            "XGBoost": 0.1770
                        },
                        "created_at": "2026-10-21T20:35:48Z",
                        "activated_at": datetime.now(timezone.utc).isoformat(),
                        "activated_by": "system"
                    },
                    {
                        "version": "v0.9-catboost-solo",
                        "name": "Tuned CatBoost Baseline",
                        "status": "candidate",
                        "metrics": {"R2": 0.8019, "RMSE": 0.7355, "MAE": 0.5445},
                        "weights": {"CatBoost": 1.0},
                        "created_at": "2026-10-20T18:00:00Z",
                        "activated_at": None,
                        "activated_by": None
                    }
                ]
                _write_json(cls.MODELS_FILE, models)

            if not cls.PREDICTIONS_FILE.exists():
                _write_json(cls.PREDICTIONS_FILE, [])
            if not cls.JOBS_FILE.exists():
                _write_json(cls.JOBS_FILE, [])
            if not cls.AUDIT_FILE.exists():
                _write_json(cls.AUDIT_FILE, [])
            if not cls.DATASETS_FILE.exists():
                _write_json(cls.DATASETS_FILE, [])

    # USERS
    @classmethod
    def get_user_by_username(cls, username: str):
        with _lock:
            users = _read_json(cls.USERS_FILE, [])
            for u in users:
                if u["username"].lower() == username.lower():
                    return u
            return None

    @classmethod
    def create_user(cls, username: str, password: str, role: str = "user"):
        with _lock:
            users = _read_json(cls.USERS_FILE, [])
            if any(u["username"].lower() == username.lower() for u in users):
                raise ValueError("Username already taken")
            pwd_hash, salt = _hash_password(password)
            new_user = {
                "id": len(users) + 1,
                "username": username,
                "role": role,
                "password_hash": pwd_hash,
                "salt": salt,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            users.append(new_user)
            _write_json(cls.USERS_FILE, users)
            return {"id": new_user["id"], "username": new_user["username"], "role": new_user["role"]}

    # PREDICTION HISTORY
    @classmethod
    def record_prediction(cls, username: str | None, inputs: dict, prediction_result: float, model_version: str):
        with _lock:
            history = _read_json(cls.PREDICTIONS_FILE, [])
            entry = {
                "id": len(history) + 1,
                "username": username or "anonymous",
                "inputs": inputs,
                "predicted_uptake": round(prediction_result, 4),
                "model_version": model_version,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            history.append(entry)
            _write_json(cls.PREDICTIONS_FILE, history)
            return entry

    @classmethod
    def get_predictions(cls, username: str | None = None, limit: int = 50):
        with _lock:
            history = _read_json(cls.PREDICTIONS_FILE, [])
            if username and username != "admin":
                history = [h for h in history if h.get("username") == username]
            return list(reversed(history[-limit:]))

    # MODEL VERSIONS
    @classmethod
    def get_model_versions(cls):
        with _lock:
            return _read_json(cls.MODELS_FILE, [])

    @classmethod
    def get_active_model_version(cls):
        with _lock:
            models = _read_json(cls.MODELS_FILE, [])
            for m in models:
                if m.get("status") == "active":
                    return m
            return models[0] if models else None

    @classmethod
    def activate_model_version(cls, version: str, username: str):
        with _lock:
            models = _read_json(cls.MODELS_FILE, [])
            target = next((m for m in models if m["version"] == version), None)
            if not target:
                raise ValueError(f"Model version '{version}' not found")

            # Set previous active to archived/inactive
            prev_active = None
            for m in models:
                if m.get("status") == "active":
                    prev_active = m["version"]
                    m["status"] = "archived"

            target["status"] = "active"
            target["activated_at"] = datetime.now(timezone.utc).isoformat()
            target["activated_by"] = username
            _write_json(cls.MODELS_FILE, models)

            # Record audit log
            cls.record_audit(username, "ACTIVATE_MODEL", f"Activated model {version}. Previous active: {prev_active}")
            return target

    @classmethod
    def rollback_model_version(cls, target_version: str, username: str):
        with _lock:
            models = _read_json(cls.MODELS_FILE, [])
            target = next((m for m in models if m["version"] == target_version), None)
            if not target:
                raise ValueError(f"Target rollback version '{target_version}' not found")

            current_active = None
            for m in models:
                if m.get("status") == "active":
                    current_active = m["version"]
                    m["status"] = "archived"

            target["status"] = "active"
            target["activated_at"] = datetime.now(timezone.utc).isoformat()
            target["activated_by"] = username
            _write_json(cls.MODELS_FILE, models)

            cls.record_audit(username, "ROLLBACK_MODEL", f"Rolled back from {current_active} to {target_version}")
            return target

    # TRAINING JOBS
    @classmethod
    def create_training_job(cls, dataset_name: str, username: str):
        with _lock:
            jobs = _read_json(cls.JOBS_FILE, [])
            job_id = f"job-{secrets.token_hex(4)}"
            new_job = {
                "job_id": job_id,
                "dataset_name": dataset_name,
                "status": "pending",
                "progress_pct": 0,
                "created_by": username,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": None,
                "metrics": None,
                "error": None
            }
            jobs.append(new_job)
            _write_json(cls.JOBS_FILE, jobs)
            return new_job

    @classmethod
    def update_training_job(cls, job_id: str, updates: dict):
        with _lock:
            jobs = _read_json(cls.JOBS_FILE, [])
            job = next((j for j in jobs if j["job_id"] == job_id), None)
            if job:
                job.update(updates)
                _write_json(cls.JOBS_FILE, jobs)
                return job
            return None

    @classmethod
    def get_training_job(cls, job_id: str):
        with _lock:
            jobs = _read_json(cls.JOBS_FILE, [])
            return next((j for j in jobs if j["job_id"] == job_id), None)

    # AUDIT LOGS
    @classmethod
    def record_audit(cls, username: str, action: str, details: str):
        logs = _read_json(cls.AUDIT_FILE, [])
        logs.append({
            "id": len(logs) + 1,
            "username": username,
            "action": action,
            "details": details,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        _write_json(cls.AUDIT_FILE, logs)

    @classmethod
    def get_audit_logs(cls, limit: int = 50):
        with _lock:
            logs = _read_json(cls.AUDIT_FILE, [])
            return list(reversed(logs[-limit:]))

# Initialize storage on module import
JSONStorage.init_storage()
