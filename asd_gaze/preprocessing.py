"""Model inputs for one (stimulus, fixations) record.

The Android app ports these functions to Java (``Preprocessing.java`` and ``PilResize.java``).
``scripts/make_parity_golden.py`` writes reference values from this module, and the app's unit
tests check the Java port against them, so change both sides together.

Fixations are ``(x, y, duration_ms)`` rows in stimulus pixel coordinates, as stored in the
Saliency4ASD scanpath files. The heatmap divides those coordinates by the 1280x1024 screen of
the eye tracker. A 1024x768 picture therefore covers only part of the 224x224 grid. The released
models were trained with this mapping, so inference must keep it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .utils import (
    duration_log_normalize,
    imagenet_normalize,
    jet_colormap,
    local_minmax,
    pad_or_truncate,
    weighted_gaussian_heatmap,
)

SCREEN_WIDTH = 1280.0
SCREEN_HEIGHT = 1024.0
IMAGE_SIZE = 224
HEATMAP_SIGMA = 10.0
MAX_SEQ_LEN = 25
BLEND_ALPHA = 0.55


def build_sequence(fixations: np.ndarray, max_seq_len: int = MAX_SEQ_LEN) -> np.ndarray:
    """Gaze sequence of shape (max_seq_len, 3).

    x and y are min-max scaled over all fixations of the record, the duration becomes
    log1p(ms) / 10, and the rows are truncated or zero-padded to ``max_seq_len``.
    """
    fixations = np.asarray(fixations, dtype=np.float32).reshape(-1, 3)
    if len(fixations) == 0:
        return np.zeros((max_seq_len, 3), dtype=np.float32)

    seq = np.zeros((len(fixations), 3), dtype=np.float32)
    seq[:, 0] = local_minmax(fixations[:, 0])
    seq[:, 1] = local_minmax(fixations[:, 1])
    seq[:, 2] = duration_log_normalize(fixations[:, 2])
    return pad_or_truncate(seq, max_seq_len)


def load_stimulus(source: str | Path | Image.Image) -> np.ndarray:
    """Stimulus as a (3, 224, 224) float32 array in [0, 1], resized with Pillow's bilinear filter."""
    image = source if isinstance(source, Image.Image) else Image.open(source)
    image = image.convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0


def build_heatmap(fixations: np.ndarray) -> np.ndarray:
    """Duration-weighted Gaussian heatmap (224, 224), peak-normalised to 1."""
    return weighted_gaussian_heatmap(
        fixations=np.asarray(fixations, dtype=np.float32).reshape(-1, 3),
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        sigma=HEATMAP_SIGMA,
        screen_width=SCREEN_WIDTH,
        screen_height=SCREEN_HEIGHT,
    )


def build_visual_tensor(image_chw: np.ndarray, fixations: np.ndarray, blend_alpha: float = BLEND_ALPHA) -> np.ndarray:
    """Stimulus blended with the jet-coloured heatmap, then ImageNet-normalised: (3, 224, 224)."""
    heatmap_rgb = jet_colormap(build_heatmap(fixations))
    blended = blend_alpha * image_chw + (1.0 - blend_alpha) * heatmap_rgb
    return imagenet_normalize(blended)


__all__ = [
    'BLEND_ALPHA',
    'HEATMAP_SIGMA',
    'IMAGE_SIZE',
    'MAX_SEQ_LEN',
    'SCREEN_HEIGHT',
    'SCREEN_WIDTH',
    'build_heatmap',
    'build_sequence',
    'build_visual_tensor',
    'load_stimulus',
]
