"""
Feature extractor for species identification and similarity search.

The backbone is selected by ``config.EMBEDDING_MODEL`` and downloaded once
during setup, then loaded from the local TORCH_HOME cache. We take the pooled
features *before* the classifier head and L2-normalise them so cosine
similarity reduces to a dot product.

Supported backbones (see config.py):
    mobilenet_v3_large, convnext_tiny, convnext_small, efficientnet_v2_s

Adding another torchvision backbone only requires a new entry in
``_build_backbone`` returning a module that maps (N, 3, H, W) -> (N, D).
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from . import config

_model = None
_preprocess = None
_lock = threading.Lock()


def _build_backbone(name: str):
    """
    Construct a torchvision backbone whose forward pass returns a pooled
    feature vector (N, D) with the classifier head removed.

    Returns (module, feature_dim).
    """
    import torch
    from torchvision import models

    if name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        net = models.mobilenet_v3_large(weights=weights)
        # forward: features -> avgpool -> flatten -> classifier
        net.classifier = torch.nn.Identity()      # -> (N, 960)
        return net, 960

    if name in ("convnext_tiny", "convnext_small"):
        if name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            net = models.convnext_tiny(weights=weights)
        else:
            weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1
            net = models.convnext_small(weights=weights)
        # classifier = Sequential(LayerNorm2d, Flatten, Linear). Drop only the
        # Linear so we keep the normalisation + flatten -> (N, 768).
        net.classifier[2] = torch.nn.Identity()
        return net, 768

    if name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        net = models.efficientnet_v2_s(weights=weights)
        # forward: features -> avgpool -> flatten -> classifier
        net.classifier = torch.nn.Identity()       # -> (N, 1280)
        return net, 1280

    raise ValueError(f"Unsupported embedding backbone: {name}")


def _build():
    """Lazily construct the model + preprocessing transform (thread-safe)."""
    global _model, _preprocess
    if _model is not None:
        return

    import torch
    from torchvision import transforms

    with _lock:
        if _model is not None:      # re-check inside the lock
            return

        net, dim = _build_backbone(config.EMBEDDING_MODEL)
        net.eval()

        if dim != config.EMBEDDING_DIM:      # guard against config drift
            raise RuntimeError(
                f"Backbone '{config.EMBEDDING_MODEL}' produces {dim}-D features "
                f"but config.EMBEDDING_DIM is {config.EMBEDDING_DIM}."
            )

        _preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
        _model = net


def embed(rgb: np.ndarray) -> np.ndarray:
    """
    Compute an L2-normalised embedding for an RGB numpy image (H, W, 3).
    Returns a 1-D float32 vector of length config.EMBEDDING_DIM.
    """
    _build()
    import torch

    with torch.no_grad():
        tensor = _preprocess(rgb).unsqueeze(0)      # (1, 3, H, W)
        feats = _model(tensor).reshape(-1).numpy().astype(np.float32)

    norm = np.linalg.norm(feats)
    if norm > 0:
        feats = feats / norm
    return feats
