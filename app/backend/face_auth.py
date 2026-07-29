"""
Face embedding + matching for the optional face-login mode.

Pipeline (all offline, no new dependencies):
  1. detect the largest face in an RGB image with OpenCV's bundled Haar
     cascade (ships inside opencv-python-headless -> cv2.data.haarcascades)
  2. crop it with a small margin
  3. embed the crop with the SAME ConvNeXt backbone used for leaves
     (embedding.embed -> L2-normalised vector)
  4. compare against enrolled users by cosine similarity (a dot product,
     since every vector is L2-normalised)

Because the backbone is a general-purpose ImageNet network rather than a
dedicated face-recognition model, and there is no liveness check, this is a
convenience feature only. See the SECURITY NOTE in config.py.
"""
from __future__ import annotations

import threading
from typing import Optional, Tuple

import cv2
import numpy as np

from . import config, embedding
from . import auth_db

_cascade = None
_cascade_lock = threading.Lock()


class FaceError(Exception):
    """Raised when no usable face can be extracted from an image."""


def _get_cascade() -> "cv2.CascadeClassifier":
    """Load the frontal-face Haar cascade once (thread-safe)."""
    global _cascade
    if _cascade is not None:
        return _cascade
    with _cascade_lock:
        if _cascade is None:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            clf = cv2.CascadeClassifier(path)
            if clf.empty():
                raise FaceError("Face detector could not be loaded.")
            _cascade = clf
    return _cascade


def _largest_face(rgb: np.ndarray) -> Tuple[int, int, int, int]:
    """Return the (x, y, w, h) of the largest detected face, or raise."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _get_cascade().detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(config.FACE_MIN_SIZE, config.FACE_MIN_SIZE),
    )
    if len(faces) == 0:
        raise FaceError(
            "No face detected. Center your face, ensure good lighting, and "
            "look straight at the camera."
        )
    # pick the biggest box (closest / most prominent face)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def _crop_with_margin(rgb: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    m = config.FACE_CROP_MARGIN
    H, W = rgb.shape[:2]
    x0 = max(0, int(x - m * w))
    y0 = max(0, int(y - m * h))
    x1 = min(W, int(x + w + m * w))
    y1 = min(H, int(y + h + m * h))
    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0:
        raise FaceError("Face crop was empty.")
    return crop


def embed_face(rgb: np.ndarray) -> np.ndarray:
    """
    Detect the largest face in an RGB image and return its L2-normalised
    embedding. Raises FaceError if no usable face is found.
    """
    box = _largest_face(rgb)
    crop = _crop_with_margin(rgb, box)
    return embedding.embed(crop)


def match_face(vector: np.ndarray) -> Tuple[Optional[int], float]:
    """
    Compare a face embedding against every enrolled user.

    Returns (user_id, similarity) for the best match whose similarity meets
    ``config.FACE_MATCH_THRESHOLD``. If nobody clears the threshold, returns
    (None, best_similarity) so callers can log/inspect the near-miss score.
    """
    enrolled = auth_db.all_face_embeddings()
    if not enrolled:
        return None, 0.0

    query = np.asarray(vector, dtype=np.float32).reshape(-1)

    # Aggregate the best similarity per user (a user may enrol several shots).
    best_by_user: dict[int, float] = {}
    for user_id, vec in enrolled:
        if vec.shape[0] != query.shape[0]:
            continue
        sim = float(np.dot(query, vec))
        if sim > best_by_user.get(user_id, -1.0):
            best_by_user[user_id] = sim

    if not best_by_user:
        return None, 0.0

    best_user = max(best_by_user, key=best_by_user.get)
    best_sim = best_by_user[best_user]
    if best_sim >= config.FACE_MATCH_THRESHOLD:
        return best_user, best_sim
    return None, best_sim
