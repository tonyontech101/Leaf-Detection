"""
Central configuration for the Leaf Detection app.

Importing this module first sets environment variables that force every ML
library into OFFLINE mode, so that once models are downloaded during setup,
the app never attempts a network call at runtime.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# 1. Force offline mode for all ML libraries BEFORE they get imported.
#    These env vars are read by torch / huggingface at import time.
# --------------------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Keep torch hub / torchvision from reaching out once weights are cached.
os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[2] / "app" / "artifacts" / "torch_cache"))

# --------------------------------------------------------------------------
# 2. Project paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = PROJECT_ROOT / "app"
FRONTEND_DIR = APP_DIR / "frontend"
ARTIFACTS_DIR = APP_DIR / "artifacts"
THUMBNAILS_DIR = ARTIFACTS_DIR / "thumbnails"
TORCH_CACHE_DIR = ARTIFACTS_DIR / "torch_cache"

# The dataset folder shipped with the project. Each subfolder is one class
# (species + condition, e.g. "Piper Betle Healthy") and holds its images.
DATASET_DIR = PROJECT_ROOT / "Original Dataset"

# Generated artifacts (created by scripts/preprocess_build_index.py)
INDEX_FILE = ARTIFACTS_DIR / "index.npz"          # embeddings + labels + thumb names
LABELS_FILE = ARTIFACTS_DIR / "labels.json"       # class list + counts
MANIFEST_FILE = ARTIFACTS_DIR / "manifest.json"   # offline-readiness manifest

# --------------------------------------------------------------------------
# 3. Model configuration
# --------------------------------------------------------------------------
# Feature extractor used for species kNN + similarity search.
#
# Configurable via the LEAF_EMBEDDING_MODEL env var. Supported torchvision
# backbones (downloaded once during setup, then cached offline):
#
#   mobilenet_v3_large  dim  960  fastest, lowest accuracy (legacy default)
#   convnext_tiny       dim  768  strong accuracy, good CPU speed
#   convnext_small      dim  768  stronger accuracy (default), a bit slower
#   efficientnet_v2_s   dim 1280  strong accuracy, medium speed
#
# NOTE: changing this REQUIRES re-running setup (to cache the new weights)
# and rebuilding the index, since embeddings from different backbones are
# not comparable. See app/scripts/preprocess_build_index.py.
_EMBEDDING_DIMS = {
    "mobilenet_v3_large": 960,
    "convnext_tiny": 768,
    "convnext_small": 768,
    "efficientnet_v2_s": 1280,
}
EMBEDDING_MODEL = os.environ.get("LEAF_EMBEDDING_MODEL", "convnext_small")
if EMBEDDING_MODEL not in _EMBEDDING_DIMS:
    raise ValueError(
        f"Unsupported LEAF_EMBEDDING_MODEL '{EMBEDDING_MODEL}'. "
        f"Choose one of: {', '.join(_EMBEDDING_DIMS)}"
    )
EMBEDDING_DIM = _EMBEDDING_DIMS[EMBEDDING_MODEL]
IMAGE_SIZE = 224

# Nearest-neighbour settings
KNN_NEIGHBORS = 9         # votes used for species prediction
SIMILAR_RESULTS = 6       # similar images returned to the UI

# --------------------------------------------------------------------------
# 4. Local VLM (leaf-status analysis) via Ollama - runs on localhost only.
# --------------------------------------------------------------------------
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
VLM_MODEL = os.environ.get("LEAF_VLM_MODEL", "moondream")  # small, offline-friendly
VLM_TIMEOUT_SECONDS = 60
# Small local vision models (e.g. moondream) reliably answer a single, direct
# question but tend to return an EMPTY response when asked to follow a rigid
# multi-part output format. We therefore ask one plain-language question for
# the free-text observation and let the OpenCV colour heuristic decide the
# overall status (see health._combine). If a model does volunteer a
# "STATUS: <word>" line, health._query_vlm still parses and uses it.
VLM_PROMPT = (
    "Describe the condition of this plant leaf, including any yellowing, "
    "browning, spots, or wilting."
)

# --------------------------------------------------------------------------
# 5. Server
# --------------------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 8000


def ensure_dirs() -> None:
    """Create artifact directories if they do not exist."""
    for d in (ARTIFACTS_DIR, THUMBNAILS_DIR, TORCH_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
