"""
FastAPI application for the Leaf Detection app.

Endpoints
  GET  /api/health-check   offline-readiness + component status
  POST /api/analyze        upload/capture a leaf image -> species + status + similar
  GET  /thumbs/<file>      dataset thumbnails for the "similar images" gallery
  GET  /                   the PWA frontend

The app binds to 127.0.0.1 only. It performs NO outbound network calls; the
only HTTP it makes is to a local Ollama instance on localhost for the VLM.
"""
from __future__ import annotations

# Importing config first forces offline env vars before torch is imported.
from . import config

import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import health, inference, leaf_utils, plant_info

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("leaf")

app = FastAPI(title="Leaf Detection (offline)", version="1.0.0")

MAX_UPLOAD_BYTES = 15 * 1024 * 1024   # 15 MB guard against oversized uploads


@app.get("/api/health-check")
def health_check() -> dict:
    """Report whether the app is offline-ready and which parts are live."""
    return {
        "status": "ok",
        "index_ready": inference._index.ready,
        "vlm_available": health.vlm_available(),
        "vlm_model": config.VLM_MODEL,
    }


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...)) -> JSONResponse:
    """Analyse a single leaf image: species, health status, similar leaves."""
    if not inference._index.ready:
        raise HTTPException(
            status_code=503,
            detail="Embedding index missing. Run setup before using the app.",
        )

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 15 MB).")

    try:
        rgb = leaf_utils.load_rgb(data)
    except Exception:
        raise HTTPException(status_code=400, detail="Unsupported or corrupt image.")

    # Segment once, reuse for both species ID and health analysis.
    masked, mask = leaf_utils.segment_leaf(rgb)

    try:
        species = inference.identify(rgb, masked, mask)
    except (FileNotFoundError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=str(e))

    status = health.analyze(rgb, mask)

    # --- AI-generated leaf description + care guidance (offline) ---
    # The species card's "description" now shows the local vision model's
    # free-text observation of the captured leaf, replacing the static
    # reference-dataset blurb and the scientific-name line.
    observations = (status.get("vlm") or {}).get("observations", "").strip()
    species["description"] = observations
    status["care"] = plant_info.care(status.get("status", "UNKNOWN"), observations)

    return JSONResponse({"species": species, "health": status})


# --- static frontend + thumbnails (mounted last so /api/* wins) ---
if config.THUMBNAILS_DIR.exists():
    app.mount("/thumbs", StaticFiles(directory=str(config.THUMBNAILS_DIR)),
              name="thumbs")

if config.FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True),
              name="frontend")


@app.on_event("startup")
def _startup_readiness() -> None:
    """Log an explicit offline-readiness summary on boot."""
    idx = inference._index.ready
    vlm = health.vlm_available()
    log.info("=" * 56)
    log.info(" Leaf Detection - startup readiness")
    log.info("   embedding index : %s", "READY" if idx else "MISSING (run setup)")
    log.info("   local VLM       : %s", "READY" if vlm else "not running (heuristic only)")
    log.info("   species ID + similarity work fully offline.")
    if not idx:
        log.warning("   -> Run: python -m app.scripts.preprocess_build_index")
    log.info("=" * 56)
