"""
Verify that the app is fully OFFLINE-READY.

Checks that every piece needed at runtime already exists on disk / locally:
  1. torchvision MobileNetV3 weights cached in TORCH_HOME
  2. embedding index + labels + thumbnails
  3. local Ollama VLM pulled (optional but recommended)

Writes app/artifacts/manifest.json and prints a clear verdict. Exit code 0
means the app will run with no internet; non-zero means something still needs
a one-time download.

  python -m app.scripts.verify_offline
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from app.backend import config, health


def _torch_weights_cached() -> bool:
    """True if the configured backbone's weights are in the local torch cache."""
    hub_dir = config.TORCH_CACHE_DIR / "hub" / "checkpoints"
    if not hub_dir.exists():
        return False
    # torchvision checkpoints are named like "convnext_small-0c510722.pth",
    # so the model key is a prefix of the cached filename.
    return any(config.EMBEDDING_MODEL in p.name for p in hub_dir.iterdir())


def _index_ready() -> bool:
    return (config.INDEX_FILE.exists()
            and config.LABELS_FILE.exists()
            and config.THUMBNAILS_DIR.exists()
            and any(config.THUMBNAILS_DIR.iterdir()))


def main() -> int:
    config.ensure_dirs()

    checks = {
        "embedding_weights_cached": _torch_weights_cached(),
        "embedding_index_built": _index_ready(),
        "vlm_pulled": health.vlm_available(),
    }

    # Species ID + similarity are the core; VLM is an enhancement.
    core_ready = checks["embedding_weights_cached"] and checks["embedding_index_built"]
    fully_ready = core_ready and checks["vlm_pulled"]

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "core_offline_ready": core_ready,
        "fully_offline_ready": fully_ready,
        "vlm_model": config.VLM_MODEL,
    }
    with open(config.MANIFEST_FILE, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print("\n" + "=" * 60)
    print(" OFFLINE READINESS REPORT")
    print("=" * 60)
    for name, ok in checks.items():
        print(f"  [{'OK ' if ok else 'XX '}] {name}")
    print("-" * 60)
    if fully_ready:
        print("  RESULT: FULLY OFFLINE READY — no internet needed to run.")
    elif core_ready:
        print("  RESULT: CORE OFFLINE READY — species ID + similarity work offline.")
        print("          VLM leaf-status not pulled; run setup to enable it,")
        print("          or the app falls back to the colour heuristic.")
    else:
        print("  RESULT: NOT READY — run: python -m app.scripts.setup_offline")
    print("=" * 60 + "\n")

    return 0 if core_ready else 1


if __name__ == "__main__":
    sys.exit(main())
