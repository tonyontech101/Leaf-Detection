"""
Image utilities: EXIF-safe loading, leaf segmentation, and colour-based
health heuristics. Pure OpenCV / Pillow / NumPy so it runs fully offline.
"""
from __future__ import annotations

import io
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

# Enable HEIC/HEIF decoding (iPhone photos) if pillow-heif is installed.
# The dataset ships some images in .HEIC format; without this they cannot be
# read. Falls back silently if the optional dependency is unavailable.
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:  # pragma: no cover - optional dependency
    pass


def load_rgb(data: bytes) -> np.ndarray:
    """Decode image bytes to an EXIF-corrected RGB numpy array (H, W, 3)."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)          # respect phone orientation
    img = img.convert("RGB")
    return np.asarray(img)


def segment_leaf(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate the leaf from the background using HSV colour thresholding.

    Returns (masked_rgb, mask) where mask is uint8 {0,255}. If segmentation
    finds too little foreground, we fall back to the full image so we never
    discard a valid leaf.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    # Vegetation tends to be green; diseased leaves add yellow/brown.
    # Capture a broad plant-colour band, then clean up with morphology.
    lower = np.array([15, 25, 25], dtype=np.uint8)     # yellow-green start
    upper = np.array([95, 255, 255], dtype=np.uint8)   # through green
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep only the largest connected component (the leaf).
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        clean = np.zeros_like(mask)
        cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
        mask = clean

    coverage = float(mask.mean()) / 255.0
    if coverage < 0.05:                 # segmentation failed - use whole image
        mask = np.full(rgb.shape[:2], 255, dtype=np.uint8)

    masked = cv2.bitwise_and(rgb, rgb, mask=mask)
    return masked, mask


def health_heuristic(rgb: np.ndarray, mask: np.ndarray) -> dict:
    """
    Rule-based colour analysis of the segmented leaf. This is NOT a medical
    diagnosis - it is a transparent estimate of visible discoloration.

    Returns fractions of healthy-green, yellow, and brown/necrotic pixels
    within the leaf mask, plus a coarse status label.
    """
    leaf = mask > 0
    total = int(leaf.sum())
    if total == 0:
        return {"status": "UNKNOWN", "green": 0.0, "yellow": 0.0, "brown": 0.0,
                "discoloration": 0.0}

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)

    green = leaf & (h >= 35) & (h <= 85) & (s >= 40)
    yellow = leaf & (h >= 20) & (h < 35) & (s >= 40)
    brown = leaf & (((h < 20) | (h > 170)) & (s >= 30) & (v < 200))

    g = float(green.sum()) / total
    y = float(yellow.sum()) / total
    b = float(brown.sum()) / total
    discoloration = round(y + b, 4)

    if discoloration < 0.10:
        status = "HEALTHY"
    elif discoloration < 0.30:
        status = "MINOR_ISSUES"
    else:
        status = "UNHEALTHY"

    return {
        "status": status,
        "green": round(g, 4),
        "yellow": round(y, 4),
        "brown": round(b, 4),
        "discoloration": discoloration,
    }
