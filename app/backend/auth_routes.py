"""
Authentication API for the Leaf Detection app.

Routes (all under /api/auth)
  POST /signup       create an account (email + password), optionally enrol a face
  POST /login        email + password  -> bearer token
  POST /login-face   face image        -> bearer token (matches an enrolled user)
  POST /enroll-face  add a face to the signed-in account (auth required)
  GET  /me           current account (auth required)
  POST /logout       invalidate the current token (auth required)

Tokens are returned in the JSON body and expected back as
``Authorization: Bearer <token>``. The frontend stores the token in
localStorage. Face images are sent as base64 data URLs captured from the
browser webcam.
"""
from __future__ import annotations

import base64
import binascii
import re
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from . import auth_db, face_auth, leaf_utils

router = APIRouter(prefix="/api/auth", tags=["auth"])

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_MIN_PASSWORD = 8
_MAX_IMAGE_BYTES = 8 * 1024 * 1024   # 8 MB guard for face captures


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------
class SignupBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=_MIN_PASSWORD, max_length=200)
    face_image: Optional[str] = None      # optional base64 data URL


class LoginBody(BaseModel):
    email: str
    password: str


class FaceBody(BaseModel):
    face_image: str                        # base64 data URL (required)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _decode_data_url(data_url: str) -> bytes:
    """Decode a base64 data URL (or raw base64) to raw image bytes."""
    if not data_url:
        raise HTTPException(status_code=400, detail="No image provided.")
    payload = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Invalid image encoding.")
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image.")
    if len(raw) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Face image too large.")
    return raw


def _face_vector_from_data_url(data_url: str):
    """Decode -> RGB -> detect+embed face. Raises HTTPException on failure."""
    raw = _decode_data_url(data_url)
    try:
        rgb = leaf_utils.load_rgb(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="Unsupported or corrupt image.")
    try:
        return face_auth.embed_face(rgb)
    except face_auth.FaceError as e:
        raise HTTPException(status_code=422, detail=str(e))


def _bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


def require_user(authorization: Optional[str] = Header(default=None)) -> dict:
    """FastAPI dependency: resolve the bearer token to a user or 401."""
    token = _bearer_token(authorization)
    user = auth_db.user_for_token(token) if token else None
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return user


def optional_user(authorization: Optional[str] = Header(default=None)) -> Optional[dict]:
    """Like require_user but returns None instead of raising."""
    token = _bearer_token(authorization)
    return auth_db.user_for_token(token) if token else None


def _session_response(user: dict) -> dict:
    token, expires = auth_db.create_session(user["id"])
    return {"token": token, "expires_at": expires, "user": user}


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@router.post("/signup")
def signup(body: SignupBody) -> dict:
    email = auth_db.normalize_email(body.email)
    if not _EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    if auth_db.email_exists(email):
        raise HTTPException(status_code=409,
                            detail="An account with this email already exists.")

    # If a face was supplied, validate it BEFORE creating the account so we
    # don't leave a half-enrolled user when the capture has no detectable face.
    face_vector = None
    if body.face_image:
        face_vector = _face_vector_from_data_url(body.face_image)

    try:
        user_id = auth_db.create_user(body.name, email, body.password)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    if face_vector is not None:
        auth_db.add_face_embedding(user_id, face_vector)

    user = auth_db.get_user(user_id)
    return _session_response(user)


@router.post("/login")
def login(body: LoginBody) -> dict:
    user = auth_db.verify_credentials(body.email, body.password)
    if user is None:
        # Same message for unknown email and wrong password (no user enumeration).
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return _session_response(user)


@router.post("/login-face")
def login_face(body: FaceBody) -> dict:
    vector = _face_vector_from_data_url(body.face_image)
    user_id, similarity = face_auth.match_face(vector)
    if user_id is None:
        raise HTTPException(
            status_code=401,
            detail="Face not recognized. Try again or sign in with your password.",
        )
    user = auth_db.get_user(user_id)
    if user is None:                      # enrolled user was deleted
        raise HTTPException(status_code=401, detail="Face not recognized.")
    resp = _session_response(user)
    resp["similarity"] = round(similarity, 4)
    return resp


@router.post("/enroll-face")
def enroll_face(body: FaceBody, user: dict = Depends(require_user)) -> dict:
    vector = _face_vector_from_data_url(body.face_image)
    auth_db.add_face_embedding(user["id"], vector)
    return {"ok": True, "has_face": True, "faces": auth_db.face_count(user["id"])}


@router.get("/me")
def me(user: dict = Depends(require_user)) -> dict:
    return {"user": user}


@router.post("/logout")
def logout(authorization: Optional[str] = Header(default=None)) -> dict:
    token = _bearer_token(authorization)
    if token:
        auth_db.delete_session(token)
    return {"ok": True}
