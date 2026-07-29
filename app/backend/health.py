"""
Leaf-status analysis.

Combines two fully-local signals:
  1. A transparent OpenCV colour heuristic (always available).
  2. A local vision-language model served by Ollama on localhost (optional).

Both run offline. If the VLM is unavailable we still return the heuristic
result so the feature degrades gracefully instead of failing.
"""
from __future__ import annotations

import base64
import re
from typing import Optional

import numpy as np
import requests

from . import config
from . import leaf_utils

_DISCLAIMER = (
    "This is an automated visual estimate, not a professional diagnosis. "
    "Confirm with an expert before acting."
)


def _encode_jpeg(rgb: np.ndarray) -> str:
    """Encode an RGB array to base64 JPEG for the Ollama API."""
    import cv2
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Failed to encode image for VLM")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def vlm_available() -> bool:
    """Check whether the local Ollama VLM is reachable and pulled."""
    try:
        r = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3)
        if r.status_code != 200:
            return False
        names = [m.get("name", "") for m in r.json().get("models", [])]
        return any(n.split(":")[0] == config.VLM_MODEL for n in names)
    except requests.RequestException:
        return False


def _query_vlm(rgb: np.ndarray) -> Optional[dict]:
    """Ask the local VLM to describe the leaf. Returns None if unavailable."""
    try:
        payload = {
            "model": config.VLM_MODEL,
            "prompt": config.VLM_PROMPT,
            "images": [_encode_jpeg(rgb)],
            "stream": False,
            # Deterministic decoding: small vision models otherwise
            # occasionally emit an immediate stop token (empty response).
            "options": {"temperature": 0, "num_predict": 150},
        }
        r = requests.post(f"{config.OLLAMA_HOST}/api/generate",
                          json=payload, timeout=config.VLM_TIMEOUT_SECONDS)
        r.raise_for_status()
        text = r.json().get("response", "").strip()
    except (requests.RequestException, ValueError):
        return None

    # A blank generation (small models sometimes emit nothing) is treated as
    # "unavailable" so the UI degrades gracefully instead of showing an empty
    # description.
    if not text:
        return None

    status_match = re.search(r"STATUS:\s*([A-Z_]+)", text, re.IGNORECASE)
    obs_match = re.search(r"OBSERVATIONS:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    status = status_match.group(1).upper() if status_match else "UNKNOWN"
    if status not in {"HEALTHY", "MINOR_ISSUES", "UNHEALTHY"}:
        status = "UNKNOWN"
    observations = obs_match.group(1).strip() if obs_match else text
    return {"status": status, "observations": observations}


def _combine(heuristic_status: str, vlm_status: Optional[str]) -> str:
    """Merge the two signals, taking the more severe when they disagree."""
    order = {"HEALTHY": 0, "MINOR_ISSUES": 1, "UNHEALTHY": 2}
    candidates = [s for s in (heuristic_status, vlm_status)
                  if s in order]
    if not candidates:
        return "UNKNOWN"
    return max(candidates, key=lambda s: order[s])


def analyze(rgb: np.ndarray, mask: np.ndarray) -> dict:
    """
    Produce the leaf-status report. `mask` comes from leaf_utils.segment_leaf
    so the heuristic and the VLM look at the same leaf.
    """
    heuristic = leaf_utils.health_heuristic(rgb, mask)
    vlm = _query_vlm(rgb)

    final_status = _combine(heuristic["status"],
                            vlm["status"] if vlm else None)

    return {
        "status": final_status,
        "heuristic": heuristic,
        "vlm": vlm,                       # None when the local model is off
        "vlm_used": vlm is not None,
        "disclaimer": _DISCLAIMER,
    }
