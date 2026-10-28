
from datetime import datetime, timedelta, timezone
import jwt
from fastapi import Header, HTTPException, status
from backend.services.storage_service import JSONStorage, _verify_password

JWT_SECRET = "glassformai-secret-key-co2-adsorption-2026"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

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
        "iat": datetime.now(timezone.utc)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired. Please log in again."
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token."
        )

def get_current_user_optional(authorization: str | None = Header(None)) -> dict | None:
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        token = parts[1]
        try:
            payload = decode_access_token(token)
            user = JSONStorage.get_user_by_username(payload["sub"])
            return user
        except HTTPException:
            return None
    return None

def get_current_user(authorization: str | None = Header(None)) -> dict:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header (Bearer token required)."
        )
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected: 'Bearer <token>'."
        )
    token = parts[1]
    payload = decode_access_token(token)
    user = JSONStorage.get_user_by_username(payload["sub"])
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User associated with this token no longer exists."
        )
    return user

def get_current_admin(authorization: str | None = Header(None)) -> dict:
    user = get_current_user(authorization)
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrative privileges required to access this endpoint."
        )
    return user
