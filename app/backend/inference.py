"""
Inference: species identification via k-nearest-neighbour voting over the
precomputed dataset embeddings, plus similar-image retrieval. All data is
loaded from local artifacts - no network access.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from . import config
from . import embedding
from . import leaf_utils


class _Index:
    """In-memory embedding index loaded from artifacts/index.npz."""

    def __init__(self) -> None:
        self.embeddings: Optional[np.ndarray] = None   # (N, D) float32, L2-normalised
        self.label_ids: Optional[np.ndarray] = None    # (N,) int
        self.class_names: Optional[np.ndarray] = None   # (C,) str
        self.thumbs: Optional[np.ndarray] = None        # (N,) str filenames
        self._loaded = False
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if not config.INDEX_FILE.exists():
                raise FileNotFoundError(
                    f"Embedding index not found at {config.INDEX_FILE}. "
                    "Run: python -m app.scripts.preprocess_build_index"
                )
            data = np.load(config.INDEX_FILE, allow_pickle=True)
            self.embeddings = data["embeddings"].astype(np.float32)
            self.label_ids = data["label_ids"].astype(np.int64)
            self.class_names = data["class_names"]
            self.thumbs = data["thumbs"]

            # Guard against an index built with a different backbone: the
            # stored vectors must match the current model's feature dimension,
            # otherwise the similarity dot product fails with a cryptic error.
            index_dim = self.embeddings.shape[1]
            if index_dim != config.EMBEDDING_DIM:
                self.embeddings = None
                raise RuntimeError(
                    f"Index dimension ({index_dim}) does not match the current "
                    f"backbone '{config.EMBEDDING_MODEL}' ({config.EMBEDDING_DIM}). "
                    "Rebuild it: python -m app.scripts.preprocess_build_index"
                )

            self._loaded = True

    @property
    def ready(self) -> bool:
        return config.INDEX_FILE.exists()


_index = _Index()


def identify(rgb: np.ndarray,
             masked: Optional[np.ndarray] = None,
             mask: Optional[np.ndarray] = None) -> dict:
    """
    Run the full visual pipeline on an RGB image:
      1. segment the leaf (skipped if `masked`/`mask` are supplied)
      2. embed it
      3. kNN vote for species (+ confidence)
      4. gather the most similar dataset leaves

    Returns a dict ready to serialise to JSON.
    """
    _index.load()

    if masked is None or mask is None:
        masked, mask = leaf_utils.segment_leaf(rgb)
    query = embedding.embed(masked)

    # Cosine similarity == dot product (all vectors are L2-normalised).
    sims = _index.embeddings @ query                      # (N,)
    order = np.argsort(-sims)                              # descending

    # --- species vote among the K nearest neighbours ---
    k = min(config.KNN_NEIGHBORS, order.shape[0])
    knn = order[:k]
    votes: dict[int, float] = {}
    for idx in knn:
        lbl = int(_index.label_ids[idx])
        # weight by similarity so closer matches count more
        votes[lbl] = votes.get(lbl, 0.0) + float(sims[idx])

    best_label = max(votes, key=votes.get)
    total_weight = sum(votes.values()) or 1.0
    confidence = round(votes[best_label] / total_weight, 4)

    # top-3 alternatives for transparency
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])[:3]
    top_species = [
        {"species": str(_index.class_names[lbl]),
         "score": round(w / total_weight, 4)}
        for lbl, w in ranked
    ]

    # --- similar images (skip duplicates of the same file if any) ---
    similar = []
    for idx in order[: config.SIMILAR_RESULTS]:
        similar.append({
            "species": str(_index.class_names[int(_index.label_ids[idx])]),
            "thumb": str(_index.thumbs[idx]),
            "similarity": round(float(sims[idx]), 4),
        })

    return {
        "species": str(_index.class_names[best_label]),
        "confidence": confidence,
        "top_species": top_species,
        "similar": similar,
        "leaf_coverage": round(float(mask.mean()) / 255.0, 4),
    }
