"""
security.py: the admin key check.

Every /api/admin/* route lists `require_admin` as a dependency, so FastAPI
runs this check BEFORE the route's own code. If the key is wrong, the route
never runs.
"""
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import settings

# auto_error=False: if the header is missing, give us None instead of
# FastAPI's default 403, so we can always answer 401 ourselves.
bearer = HTTPBearer(auto_error=False)


def require_admin(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
    # 1. The server itself has no key set -> admin is switched off entirely.
    if not settings.admin_key:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "Admin is disabled: ADMIN_KEY is not set on the server.")

    # 2. No "Authorization: Bearer ..." header, or the key doesn't match.
    if creds is None or not secrets.compare_digest(creds.credentials, settings.admin_key):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid admin key.",
                            headers={"WWW-Authenticate": "Bearer"})