"""
security.py: who is making this request?

Routes ask for a user by listing a dependency:
  current_user  - any logged-in account (predict, batch, history)
  require_admin - an account with role "admin" (everything under /api/admin)

FastAPI runs these BEFORE the route's own code. If the check fails, the
route never runs. The browser sends its session token in the header
"Authorization: Bearer <token>".
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .services import auth_service

# auto_error=False: if the header is missing, give us None instead of
# FastAPI's default 403, so we can answer 401 with a clear message.
bearer = HTTPBearer(auto_error=False)


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Log in to use this.",
                            headers={"WWW-Authenticate": "Bearer"})
    user = auth_service.user_for_token(creds.credentials)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Your session has ended. Log in again.",
                            headers={"WWW-Authenticate": "Bearer"})
    return user


def require_admin(user: dict = Depends(current_user)) -> dict:
    # 403 (not 401): we know who you are, you're just not allowed here.
    if user["role"] != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admins only.")
    return user
