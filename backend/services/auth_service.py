"""JWT authentication and role-aware request dependencies."""
from datetime import datetime, timedelta, timezone
import os
import secrets
from pathlib import Path

import jwt
from dotenv import load_dotenv
from fastapi import Header, HTTPException, status

from backend.services.storage_service import JSONStorage, _verify_password

# Load the same backend-local environment file used by the storage bootstrap.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256").strip()
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
JWT_SECRET = os.getenv("JWT_SECRET", "").strip()

if JWT_ALGORITHM != "HS256":
    raise RuntimeError("Only HS256 is supported for JWT signing.")
if JWT_EXPIRATION_HOURS < 1:
    raise RuntimeError("JWT_EXPIRATION_HOURS must be at least 1.")
if ENVIRONMENT == "production":
    if len(JWT_SECRET) < 32:
        raise RuntimeError(
            "Production requires JWT_SECRET to contain at least 32 characters."
        )
else:
    # Local development can run without a .env file, but tokens will not
    # survive a process restart unless a stable JWT_SECRET is configured.
    JWT_SECRET = JWT_SECRET or secrets.token_urlsafe(48)


def authenticate_user(username: str, password: str):
    user = JSONStorage.get_user_by_username(username)
    if not user:
        return None
    if not _verify_password(password, user["password_hash"], user["salt"]):
        return None
    return user


def create_access_token(user: dict) -> str:
    payload = {
        "sub": user["username"],
        "id": user["id"],
        "role": user["role"],
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired. Please log in again.",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
        )


def get_current_user_optional(authorization: str | None = Header(None)) -> dict | None:
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        try:
            payload = decode_access_token(parts[1])
            return JSONStorage.get_user_by_username(payload["sub"])
        except HTTPException:
            return None
    return None


def get_current_user(authorization: str | None = Header(None)) -> dict:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header (Bearer token required).",
        )
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected: 'Bearer <token>'.",
        )
    payload = decode_access_token(parts[1])
    user = JSONStorage.get_user_by_username(payload["sub"])
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User associated with this token no longer exists.",
        )
    return user


def get_current_admin(authorization: str | None = Header(None)) -> dict:
    user = get_current_user(authorization)
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrative privileges required to access this endpoint.",
        )
    return user
