"""
Build the embedding index from the "Original Dataset".

For every dataset image this script:
  * loads and EXIF-corrects it,
  * segments the leaf,
  * computes an L2-normalised MobileNetV3 embedding,
  * writes a small thumbnail for the "similar images" gallery.

Outputs (into app/artifacts/):
  index.npz     embeddings, label ids, class names, thumbnail filenames
  labels.json   class list + per-class image counts
  thumbnails/   128px JPEG thumbnails referenced by the index

Run once (after setup downloads the model weights):
  python -m app.scripts.preprocess_build_index
"""
from __future__ import annotations

import json
import sys

import numpy as np
from PIL import Image

from app.backend import config, embedding, leaf_utils

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}
THUMB_SIZE = 128


def _iter_dataset():
    """Yield (species_name, image_path) for every image in the dataset."""
    if not config.DATASET_DIR.exists():
        sys.exit(f"Dataset folder not found: {config.DATASET_DIR}")
    for species_dir in sorted(p for p in config.DATASET_DIR.iterdir() if p.is_dir()):
        for img_path in sorted(species_dir.iterdir()):
            if img_path.suffix.lower() in VALID_EXT:
                yield species_dir.name, img_path


def _save_thumbnail(rgb: np.ndarray, name: str) -> None:
    thumb = Image.fromarray(rgb)
    thumb.thumbnail((THUMB_SIZE, THUMB_SIZE))
    thumb.save(config.THUMBNAILS_DIR / name, "JPEG", quality=80)


def main() -> None:
    config.ensure_dirs()

    items = list(_iter_dataset())
    if not items:
        sys.exit("No images found in the dataset.")

    class_names = sorted({name for name, _ in items})
    class_to_id = {name: i for i, name in enumerate(class_names)}

    print(f"Found {len(items)} images across {len(class_names)} species.")
    print("Building embeddings (first run downloads model weights if needed)...")

    embeddings: list[np.ndarray] = []
    label_ids: list[int] = []
    thumbs: list[str] = []
    counts = {name: 0 for name in class_names}
    failed = 0

    for i, (species, path) in enumerate(items):
        try:
            with open(path, "rb") as fh:
                rgb = leaf_utils.load_rgb(fh.read())
            masked, _ = leaf_utils.segment_leaf(rgb)
            vec = embedding.embed(masked)
        except Exception as e:                      # skip unreadable files
            failed += 1
            print(f"  ! skipped {path.name}: {e}")
            continue

        thumb_name = f"{i:06d}.jpg"
        _save_thumbnail(rgb, thumb_name)

        embeddings.append(vec)
        label_ids.append(class_to_id[species])
        thumbs.append(thumb_name)
        counts[species] += 1

        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(items)}")

    if not embeddings:
        sys.exit("No embeddings were produced - aborting.")

    np.savez_compressed(
        config.INDEX_FILE,
        embeddings=np.stack(embeddings).astype(np.float32),
        label_ids=np.asarray(label_ids, dtype=np.int64),
        class_names=np.asarray(class_names, dtype=object),
        thumbs=np.asarray(thumbs, dtype=object),
    )

    with open(config.LABELS_FILE, "w", encoding="utf-8") as fh:
        json.dump({"classes": class_names, "counts": counts,
                   "total": len(embeddings), "skipped": failed},
                  fh, indent=2, ensure_ascii=False)

    print(f"\nDone. Indexed {len(embeddings)} images "
          f"({failed} skipped) into {config.INDEX_FILE}")


if __name__ == "__main__":
    main()
