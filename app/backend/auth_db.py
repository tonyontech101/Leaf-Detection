"""
Local account store for the Leaf Detection app.

Everything here is offline and dependency-free: a single SQLite file
(``config.AUTH_DB_FILE``) holds users, login sessions, and enrolled face
embeddings. Passwords are stored as salted PBKDF2-HMAC-SHA256 hashes using
only the Python standard library.

Tables
  users            one row per account (name, email, password hash+salt)
  sessions         opaque bearer tokens -> user, with an expiry
  face_embeddings  zero or more face vectors per user (float32 blobs)

The module is safe to import before the database exists; ``init_db`` is called
lazily on first use and creates the schema.
"""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from . import config

_lock = threading.Lock()
_initialised = False


# --------------------------------------------------------------------------
# Connection + schema
# --------------------------------------------------------------------------
def _connect() -> sqlite3.Connection:
    """Open a connection with sane defaults and row access by name."""
    config.ensure_dirs()
    conn = sqlite3.connect(str(config.AUTH_DB_FILE))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    """Create the schema once (idempotent, thread-safe)."""
    global _initialised
    if _initialised:
        return
    with _lock:
        if _initialised:
            return
        with _connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    name          TEXT    NOT NULL,
                    email         TEXT    NOT NULL UNIQUE,
                    password_hash TEXT    NOT NULL,
                    password_salt TEXT    NOT NULL,
                    created_at    REAL    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    token      TEXT    PRIMARY KEY,
                    user_id    INTEGER NOT NULL,
                    created_at REAL    NOT NULL,
                    expires_at REAL    NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id    INTEGER NOT NULL,
                    vector     BLOB    NOT NULL,
                    dim        INTEGER NOT NULL,
                    created_at REAL    NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_user
                    ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_face_user
                    ON face_embeddings(user_id);
                """
            )
        _initialised = True


# --------------------------------------------------------------------------
# Password hashing (PBKDF2-HMAC-SHA256, stdlib only)
# --------------------------------------------------------------------------
def _hash_password(password: str, salt: bytes) -> str:
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, config.PBKDF2_ITERATIONS
    )
    return dk.hex()


def _verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    candidate = _hash_password(password, salt)
    # constant-time comparison to avoid timing leaks
    return secrets.compare_digest(candidate, hash_hex)


# --------------------------------------------------------------------------
# Users
# --------------------------------------------------------------------------
def normalize_email(email: str) -> str:
    return email.strip().lower()


def email_exists(email: str) -> bool:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM users WHERE email = ?", (normalize_email(email),)
        ).fetchone()
    return row is not None


def create_user(name: str, email: str, password: str) -> int:
    """Insert a new account and return its id. Raises ValueError if taken."""
    init_db()
    salt = secrets.token_bytes(16)
    pw_hash = _hash_password(password, salt)
    try:
        with _connect() as conn:
            cur = conn.execute(
                "INSERT INTO users (name, email, password_hash, password_salt, "
                "created_at) VALUES (?, ?, ?, ?, ?)",
                (name.strip(), normalize_email(email), pw_hash, salt.hex(),
                 time.time()),
            )
            return int(cur.lastrowid)
    except sqlite3.IntegrityError:
        raise ValueError("An account with this email already exists.")


def verify_credentials(email: str, password: str) -> Optional[dict]:
    """Return the user dict on a correct password, else None."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (normalize_email(email),)
        ).fetchone()
    if row is None:
        return None
    if not _verify_password(password, row["password_salt"], row["password_hash"]):
        return None
    return _public_user(row)


def get_user(user_id: int) -> Optional[dict]:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return _public_user(row) if row else None


def _public_user(row: sqlite3.Row) -> dict:
    """Project a user row to a safe, serialisable dict (no secrets)."""
    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "created_at": row["created_at"],
        "has_face": face_count(row["id"]) > 0,
    }


# --------------------------------------------------------------------------
# Sessions (opaque bearer tokens)
# --------------------------------------------------------------------------
def create_session(user_id: int) -> Tuple[str, float]:
    """Create a session token for a user; returns (token, expires_at)."""
    init_db()
    token = secrets.token_urlsafe(32)
    now = time.time()
    expires = now + config.SESSION_TTL_SECONDS
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            (token, user_id, now, expires),
        )
    return token, expires


def user_for_token(token: str) -> Optional[dict]:
    """Resolve a bearer token to a user, honouring expiry. None if invalid."""
    if not token:
        return None
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT user_id, expires_at FROM sessions WHERE token = ?", (token,)
        ).fetchone()
        if row is None:
            return None
        if row["expires_at"] < time.time():
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            return None
    return get_user(row["user_id"])


def delete_session(token: str) -> None:
    if not token:
        return
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def purge_expired_sessions() -> int:
    """Remove expired sessions; returns the number deleted."""
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM sessions WHERE expires_at < ?", (time.time(),)
        )
        return cur.rowcount


# --------------------------------------------------------------------------
# Face embeddings
# --------------------------------------------------------------------------
def add_face_embedding(user_id: int, vector: np.ndarray) -> None:
    """Store one L2-normalised face embedding (float32) for a user."""
    init_db()
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO face_embeddings (user_id, vector, dim, created_at) "
            "VALUES (?, ?, ?, ?)",
            (user_id, vec.tobytes(), int(vec.shape[0]), time.time()),
        )


def face_count(user_id: int) -> int:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM face_embeddings WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return int(row["n"])


def all_face_embeddings() -> List[Tuple[int, np.ndarray]]:
    """Return [(user_id, vector), ...] for every enrolled face."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT user_id, vector, dim FROM face_embeddings"
        ).fetchall()
    out: List[Tuple[int, np.ndarray]] = []
    for r in rows:
        vec = np.frombuffer(r["vector"], dtype=np.float32)
        if vec.shape[0] == r["dim"]:
            out.append((int(r["user_id"]), vec))
    return out
