"""
Measure species-identification accuracy of the current embedding index.

This runs a *leave-one-out* evaluation directly on the vectors in
``artifacts/index.npz``: every dataset image is used as a query against all
the others (itself excluded), and we check whether the kNN vote recovers its
true species. It reproduces the exact voting logic used at runtime
(``inference.identify``), so the numbers reflect real app behaviour.

Because it operates on the already-built index, it is fast (a few matrix
multiplications) and needs no images or model — just the index. Run it before
and after changing ``config.EMBEDDING_MODEL`` to quantify the improvement.

  python -m app.scripts.evaluate_accuracy

Output: top-1 accuracy, kNN-vote accuracy, top-3 accuracy, and the weakest
classes so you can see where confusion remains.
"""
from __future__ import annotations

import sys

import numpy as np

from app.backend import config


def _load_index():
    if not config.INDEX_FILE.exists():
        sys.exit(
            f"Index not found at {config.INDEX_FILE}. "
            "Build it first: python -m app.scripts.preprocess_build_index"
        )
    data = np.load(config.INDEX_FILE, allow_pickle=True)
    return (
        data["embeddings"].astype(np.float32),
        data["label_ids"].astype(np.int64),
        data["class_names"],
    )


def main() -> int:
    embeddings, labels, class_names = _load_index()
    n, d = embeddings.shape
    num_classes = len(class_names)
    print(f"Index: {n} images, {num_classes} classes, {d}-D "
          f"({config.EMBEDDING_MODEL}).")
    if n < 2:
        sys.exit("Need at least two indexed images to evaluate.")

    k = min(config.KNN_NEIGHBORS, n - 1)

    # Full cosine-similarity matrix (vectors are already L2-normalised).
    # Mask the diagonal so an image never retrieves itself.
    sims = embeddings @ embeddings.T                 # (N, N)
    np.fill_diagonal(sims, -np.inf)

    # Sort neighbours once; take the top-k for voting and the top-3 for
    # the "correct species somewhere in the shortlist" metric.
    order = np.argsort(-sims, axis=1)                # (N, N) descending

    top1_hits = 0
    vote_hits = 0
    top3_hits = 0
    per_class_total = np.zeros(num_classes, dtype=np.int64)
    per_class_vote_hit = np.zeros(num_classes, dtype=np.int64)

    for i in range(n):
        true = labels[i]
        per_class_total[true] += 1
        neigh = order[i, :k]

        # 1-NN
        if labels[neigh[0]] == true:
            top1_hits += 1

        # similarity-weighted kNN vote (matches inference.identify)
        votes: dict[int, float] = {}
        for j in neigh:
            lbl = int(labels[j])
            votes[lbl] = votes.get(lbl, 0.0) + float(sims[i, j])
        predicted = max(votes, key=votes.get)
        if predicted == true:
            vote_hits += 1
            per_class_vote_hit[true] += 1

        # top-3 vote classes
        ranked = sorted(votes, key=votes.get, reverse=True)[:3]
        if true in ranked:
            top3_hits += 1

    print("\n" + "=" * 56)
    print(" SPECIES ACCURACY (leave-one-out kNN)")
    print("=" * 56)
    print(f"  1-NN accuracy        : {top1_hits / n:6.2%}")
    print(f"  kNN-vote accuracy    : {vote_hits / n:6.2%}   (k={k}, runtime metric)")
    print(f"  top-3 vote accuracy  : {top3_hits / n:6.2%}")
    print("-" * 56)

    # Show the 8 weakest classes by per-class vote accuracy.
    with np.errstate(invalid="ignore", divide="ignore"):
        per_class_acc = np.where(per_class_total > 0,
                                 per_class_vote_hit / per_class_total, 1.0)
    weakest = np.argsort(per_class_acc)[:8]
    print("  Weakest classes (vote accuracy):")
    for c in weakest:
        print(f"    {str(class_names[c]):28s} "
              f"{per_class_acc[c]:6.2%}  ({per_class_vote_hit[c]}/{per_class_total[c]})")
    print("=" * 56 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
