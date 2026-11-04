"""
auth.py: sign up, log in, log out, and "who am I".
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel

from ..security import bearer, current_user
from ..services import auth_service

router = APIRouter(prefix="/api/auth")


class Credentials(BaseModel):
    username: str
    password: str


class LoginRequest(Credentials):
    # The admin login page sends admin=true, so a normal account gets a
    # clear "not an admin" answer there instead of a confusing half-login.
    admin: bool = False


def _session_reply(user: dict) -> dict:
    return {"token": auth_service.start_session(user["id"]), "user": user}


@router.post("/signup")
def signup(req: Credentials):
    username = req.username.strip()
    problem = auth_service.check_new_account(username, req.password)
    if problem:
        raise HTTPException(400, problem)
    try:
        user = auth_service.create_user(username, req.password)   # always role "user"
    except ValueError as e:
        raise HTTPException(409, str(e))
    return _session_reply(user)   # signing up also logs you in


@router.post("/login")
def login(req: LoginRequest):
    user = auth_service.authenticate(req.username.strip(), req.password)
    if user is None:
        # Same message for unknown user and wrong password: don't reveal
        # which usernames exist.
        raise HTTPException(401, "Wrong username or password.")
    if req.admin and user["role"] != "admin":
        raise HTTPException(403, "This account is not an admin. Use the normal login page.")
    return _session_reply(user)


@router.get("/me")
def me(user: dict = Depends(current_user)):
    return {"user": user}


@router.post("/logout")
def logout(creds: HTTPAuthorizationCredentials | None = Depends(bearer)):
    if creds is not None:
        auth_service.end_session(creds.credentials)
    return {"ok": True}
