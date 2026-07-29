"""
One-time ONLINE setup. Run this once WITH internet; afterwards the app runs
fully offline forever.

Steps:
  1. Download the MobileNetV3 feature-extractor weights into the local cache.
  2. Pull the local VLM into Ollama (for leaf-status analysis).
  3. Build the embedding index from the dataset.
  4. Verify offline readiness and write the manifest.

  python -m app.scripts.setup_offline

Prerequisites (installed once with internet):
  * pip install -r requirements.txt
  * Ollama installed and running  (https://ollama.com)  - optional but
    required for the VLM leaf-status feature.
"""
from __future__ import annotations

import sys
import time

import requests

from app.backend import config


def step_download_embedding_weights() -> bool:
    print(f"\n[1/4] Downloading feature-extractor weights ({config.EMBEDDING_MODEL})...")
    try:
        # Instantiating the model with pretrained weights triggers a one-time
        # download into TORCH_HOME; subsequent loads are offline.
        from app.backend import embedding
        import numpy as np
        embedding._build()
        # sanity forward pass on a dummy image
        _ = embedding.embed(np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
                                     dtype=np.uint8))
        print("      done.")
        return True
    except Exception as e:
        print(f"      FAILED: {e}")
        return False


def step_pull_vlm() -> bool:
    print(f"\n[2/4] Pulling local VLM '{config.VLM_MODEL}' via Ollama...")
    # Verify Ollama is reachable first.
    try:
        requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=5).raise_for_status()
    except requests.RequestException:
        print(f"      Ollama not reachable at {config.OLLAMA_HOST}.")
        print("      Install & start Ollama, then re-run setup to enable the VLM.")
        print("      (The app still works offline using the colour heuristic.)")
        return False

    try:
        # Stream the pull so we can show progress and know when it finishes.
        with requests.post(f"{config.OLLAMA_HOST}/api/pull",
                           json={"name": config.VLM_MODEL},
                           stream=True, timeout=None) as r:
            r.raise_for_status()
            last = ""
            for line in r.iter_lines():
                if not line:
                    continue
                import json
                msg = json.loads(line).get("status", "")
                if msg and msg != last:
                    print(f"      {msg}")
                    last = msg
        print("      done.")
        return True
    except requests.RequestException as e:
        print(f"      FAILED: {e}")
        return False


def step_build_index() -> bool:
    print("\n[3/4] Building embedding index from the dataset...")
    try:
        from app.scripts import preprocess_build_index
        preprocess_build_index.main()
        return True
    except SystemExit as e:
        print(f"      FAILED: {e}")
        return False
    except Exception as e:
        print(f"      FAILED: {e}")
        return False


def step_verify() -> int:
    print("\n[4/4] Verifying offline readiness...")
    from app.scripts import verify_offline
    return verify_offline.main()


def main() -> int:
    print("=" * 60)
    print(" LEAF DETECTION - ONE-TIME OFFLINE SETUP")
    print(" Run this once with internet; then it works offline forever.")
    print("=" * 60)

    start = time.time()
    step_download_embedding_weights()
    step_pull_vlm()
    step_build_index()
    code = step_verify()

    print(f"\nSetup finished in {time.time() - start:.0f}s.")
    if code == 0:
        print("You can now disconnect from the internet and run:  python run.py")
    else:
        print("Some core components are missing - see the report above.")
    return code


if __name__ == "__main__":
    sys.exit(main())
