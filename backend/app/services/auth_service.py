"""
auth_service.py: user accounts and login sessions, stored in SQLite
(the same backend/data/app.db file as the prediction history).

Two tables:
  users    - one row per account: username, password hash, role ("user" or "admin")
  sessions - one row per logged-in browser: a hash of its token + expiry time

Passwords are never stored. We store a PBKDF2-SHA256 hash: the password is
mixed with a random "salt" and hashed 260,000 times, which makes guessing
passwords from a stolen database very slow. Python's hashlib has this built
in, so no extra package is needed.

Session tokens work like a temporary ticket: after logging in, the browser
gets a long random token and sends it with every request. We store only a
hash of the token, so a stolen database can't be used to log in either.
"""
import hashlib
import hmac
import re
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone

from ..config import settings
from .history_service import DB_PATH

PBKDF2_ROUNDS = 260_000
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")
MIN_PASSWORD, MAX_PASSWORD = 8, 128


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> datetime:
    return datetime.now(timezone.utc)


def init_db():
    conn = _connect()
    try:
        # COLLATE NOCASE: "Adhithya" and "adhithya" count as the same username.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                role          TEXT NOT NULL DEFAULT 'user',
                created_at    TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id    INTEGER NOT NULL REFERENCES users(id),
                expires_at TEXT NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()


# ----------------------------------------------------------------------
# Passwords
# ----------------------------------------------------------------------
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ROUNDS)
    # Everything needed to check it later is kept in one string.
    return f"pbkdf2_sha256${PBKDF2_ROUNDS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, rounds, salt_hex, digest_hex = stored.split("$")
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(rounds))
    except ValueError:
        return False
    # compare_digest takes the same time whether the first or last byte
    # differs, so timing can't leak how close a guess was.
    return hmac.compare_digest(digest.hex(), digest_hex)


# ----------------------------------------------------------------------
# Accounts
# ----------------------------------------------------------------------
def _public(row) -> dict:
    return {"id": row["id"], "username": row["username"], "role": row["role"]}


def check_new_account(username: str, password: str) -> str | None:
    """Returns an error message, or None if the details are acceptable."""
    if not USERNAME_RE.match(username):
        return "Username must be 3–30 characters: letters, numbers, _ . or -"
    if not (MIN_PASSWORD <= len(password) <= MAX_PASSWORD):
        return f"Password must be at least {MIN_PASSWORD} characters."
    return None


def create_user(username: str, password: str, role: str = "user") -> dict:
    """Raises ValueError if the username is taken."""
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            (username, hash_password(password), role, _now().isoformat(timespec="seconds")),
        )
        conn.commit()
        return {"id": cur.lastrowid, "username": username, "role": role}
    except sqlite3.IntegrityError:
        raise ValueError("That username is already taken.")
    finally:
        conn.close()


def authenticate(username: str, password: str) -> dict | None:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    finally:
        conn.close()
    if row is None:
        # Still do the slow hash, so "no such user" and "wrong password"
        # take the same time and can't be told apart.
        verify_password(password, hash_password("dummy-password"))
        return None
    return _public(row) if verify_password(password, row["password_hash"]) else None


def ensure_admin() -> str:
    """Create or update the admin account from ADMIN_USERNAME / ADMIN_PASSWORD
    in backend/.env. Runs at every startup, so changing the password in .env
    and restarting is how you change the admin password."""
    name, pw = settings.admin_username.strip(), settings.admin_password
    if not name or not pw:
        return "No ADMIN_USERNAME / ADMIN_PASSWORD in backend/.env: admin login is disabled."
    problem = check_new_account(name, pw)
    if problem:
        return f"Admin account not created: {problem}"
    conn = _connect()
    try:
        row = conn.execute("SELECT id FROM users WHERE username = ?", (name,)).fetchone()
        if row:
            conn.execute("UPDATE users SET password_hash = ?, role = 'admin' WHERE id = ?",
                         (hash_password(pw), row["id"]))
        else:
            conn.execute(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, 'admin', ?)",
                (name, hash_password(pw), _now().isoformat(timespec="seconds")),
            )
        conn.commit()
    finally:
        conn.close()
    return f"Admin account '{name}' is ready."


# ----------------------------------------------------------------------
# Sessions
# ----------------------------------------------------------------------
def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def start_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    expires = _now() + timedelta(days=settings.session_days)
    conn = _connect()
    try:
        # Tidy up: remove this server's expired sessions while we're here.
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (_now().isoformat(),))
        conn.execute("INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
                     (_token_hash(token), user_id, expires.isoformat()))
        conn.commit()
    finally:
        conn.close()
    return token


def user_for_token(token: str) -> dict | None:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = ? AND s.expires_at > ?",
            (_token_hash(token), _now().isoformat()),
        ).fetchone()
    finally:
        conn.close()
    return _public(row) if row else None


def end_session(token: str) -> None:
    conn = _connect()
    try:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))
        conn.commit()
    finally:
        conn.close()
