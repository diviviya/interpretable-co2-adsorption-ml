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

from ..config import REPO_ROOT, settings
from .history_service import DB_PATH

# Your private list of team accounts (not in GitHub). Format: one account per
# line, "username password", optionally followed by "admin".
USERS_FILE = REPO_ROOT / "backend" / "users.txt"

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


def ensure_file_users() -> list[str]:
    """Create or update every account listed in backend/users.txt.

    Runs at every startup, so the same usernames and passwords work again on
    any computer or server you copy the project (and this file) to.
    - New name in the file   -> account created (role "user", or "admin" if marked)
    - Existing account       -> password set to the one in the file
    - Marked "admin"         -> made admin. Not marked -> role left as it is,
                                so promotions from the Admin page are kept.
    - Name removed from file -> account is kept (delete nothing by accident).
    Returns one line per problem/summary for the startup log."""
    if not USERS_FILE.exists():
        return []
    messages, created, updated = [], 0, 0
    conn = _connect()
    try:
        for line_no, raw in enumerate(USERS_FILE.read_text(encoding="utf-8-sig").splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue                      # blank lines and comments
            parts = line.split()
            if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2].lower() != "admin"):
                messages.append(f"users.txt line {line_no}: expected 'username password' or 'username password admin', skipped.")
                continue
            username, password = parts[0], parts[1]
            make_admin = len(parts) == 3
            problem = check_new_account(username, password)
            if problem:
                messages.append(f"users.txt line {line_no} ({username}): {problem} Skipped.")
                continue
            if _is_main_admin(username):
                messages.append(f"users.txt line {line_no}: {username} is the main admin from .env, skipped.")
                continue
            row = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if row:
                conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(password), row["id"]))
                if make_admin:
                    conn.execute("UPDATE users SET role = 'admin' WHERE id = ?", (row["id"],))
                updated += 1
            else:
                conn.execute(
                    "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                    (username, hash_password(password), "admin" if make_admin else "user",
                     _now().isoformat(timespec="seconds")),
                )
                created += 1
        conn.commit()
    finally:
        conn.close()
    messages.append(f"users.txt: {created} account(s) created, {updated} updated.")
    return messages


# ----------------------------------------------------------------------
# Managing users (admin page)
# ----------------------------------------------------------------------
def _is_main_admin(username: str) -> bool:
    # The account from backend/.env. It always stays admin, so the site
    # can never end up with nobody able to manage it.
    return bool(settings.admin_username) and username.lower() == settings.admin_username.strip().lower()


def list_users() -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute("""
            SELECT u.id, u.username, u.role, u.created_at, COUNT(p.id) AS predictions
            FROM users u LEFT JOIN predictions p ON p.user_id = u.id
            GROUP BY u.id ORDER BY u.role = 'admin' DESC, u.username COLLATE NOCASE
        """).fetchall()
    finally:
        conn.close()
    return [{"id": r["id"], "username": r["username"], "role": r["role"],
             "created_at": r["created_at"], "predictions": r["predictions"],
             "main_admin": _is_main_admin(r["username"])} for r in rows]


def set_role(user_id: int, role: str, acting_user_id: int) -> None:
    """Make someone an admin or a normal user. Raises ValueError with a
    message the page can show if the change isn't allowed."""
    if role not in ("admin", "user"):
        raise ValueError("Role must be 'admin' or 'user'.")
    if user_id == acting_user_id:
        raise ValueError("You can't change your own role. Ask another admin.")
    conn = _connect()
    try:
        row = conn.execute("SELECT username, role FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            raise LookupError("No such user.")
        if role == "user" and _is_main_admin(row["username"]):
            raise ValueError("The main admin (set in backend/.env) always stays an admin.")
        if role == "user" and row["role"] == "admin":
            admins = conn.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'").fetchone()[0]
            if admins <= 1:
                raise ValueError("There must always be at least one admin.")
        conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
        conn.commit()
    finally:
        conn.close()
    # No need to log them out: every request reads the role fresh from the
    # database (user_for_token), so the change applies on their next click.


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
