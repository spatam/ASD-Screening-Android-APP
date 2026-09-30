from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.ndimage
import torch


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def detect_device(preferred: str = 'auto') -> torch.device:
    preferred = preferred.lower()
    if preferred == 'mps':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        raise RuntimeError('The requested MPS device is not available.')
    if preferred == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        raise RuntimeError('The requested CUDA device is not available.')
    if preferred == 'cpu':
        return torch.device('cpu')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


select_device = detect_device


def natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', value)]


def local_minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    lo = float(values.min())
    hi = float(values.max())
    if abs(hi - lo) < 1e-9:
        return np.full_like(values, 0.5, dtype=np.float32)
    out = (values - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def duration_log_normalize(values: np.ndarray) -> np.ndarray:
    return (np.log1p(np.clip(values, 0.0, None)) / 10.0).astype(np.float32)


def pad_or_truncate(array: np.ndarray, target_len: int) -> np.ndarray:
    if len(array) >= target_len:
        return array[:target_len].astype(np.float32)
    pad = np.zeros((target_len - len(array), array.shape[1]), dtype=np.float32)
    return np.concatenate([array.astype(np.float32), pad], axis=0)


def weighted_gaussian_heatmap(
    fixations: np.ndarray,
    height: int,
    width: int,
    sigma: float,
    screen_width: float,
    screen_height: float,
) -> np.ndarray:
    if fixations.size == 0:
        return np.zeros((height, width), dtype=np.float32)

    hmap = np.zeros((height, width), dtype=np.float32)
    xs = np.clip((fixations[:, 0] / screen_width) * (width - 1), 0, width - 1).astype(int)
    ys = np.clip((fixations[:, 1] / screen_height) * (height - 1), 0, height - 1).astype(int)
    weights = np.log1p(np.clip(fixations[:, 2], 0.0, None)).astype(np.float32)
    np.add.at(hmap, (ys, xs), weights)

    if float(hmap.max()) <= 0.0:
        return hmap

    radius = max(1, int(math.ceil(3.0 * sigma)))
    kernel = gaussian_kernel1d(radius, sigma)
    hmap = separable_convolution(hmap, kernel)
    peak = float(hmap.max())
    if peak > 0.0:
        hmap /= peak
    return hmap.astype(np.float32)


def gaussian_kernel1d(radius: int, sigma: float) -> np.ndarray:
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def separable_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # mode='nearest' matches the edge clamping used in the original separable loop.
    tmp = scipy.ndimage.convolve1d(image, kernel, axis=1, mode='nearest')
    out = scipy.ndimage.convolve1d(tmp, kernel, axis=0, mode='nearest')
    return out.astype(np.float32)


def jet_colormap(hmap: np.ndarray) -> np.ndarray:
    x = np.clip(hmap.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=0).astype(np.float32)


def imagenet_normalize(chw: np.ndarray) -> np.ndarray:
    mean = IMAGENET_MEAN[:, None, None]
    std = IMAGENET_STD[:, None, None]
    return ((chw - mean) / std).astype(np.float32)


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        params = [parameter for parameter in params if parameter.requires_grad]
    return int(sum(parameter.numel() for parameter in params))


def find_workspace_root(start: Path) -> Path:
    current = start.resolve()
    for parent in [current] + list(current.parents):
        if (parent / 'paper').exists() and (parent / 'code').exists():
            return parent
    raise FileNotFoundError('No workspace root found starting from the current path.')


def youden_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = labels.astype(np.int32)
    probs = probs.astype(np.float64)
    thresholds = np.unique(probs)
    best_thr = 0.5
    best_score = -np.inf
    for thr in thresholds:
        pred = (probs >= thr).astype(np.int32)
        tp = np.logical_and(pred == 1, labels == 1).sum()
        tn = np.logical_and(pred == 0, labels == 0).sum()
        fp = np.logical_and(pred == 1, labels == 0).sum()
        fn = np.logical_and(pred == 0, labels == 1).sum()
        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        score = tpr + tnr - 1.0
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr


def binary_classification_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (probs >= threshold).astype(np.int32)
    labels = labels.astype(np.int32)
    tp = np.logical_and(pred == 1, labels == 1).sum()
    tn = np.logical_and(pred == 0, labels == 0).sum()
    fp = np.logical_and(pred == 1, labels == 0).sum()
    fn = np.logical_and(pred == 0, labels == 1).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    acc = (tp + tn) / max(len(labels), 1)
    return {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
    }


def aggregate_subject_probs(
    groups: Iterable[str],
    labels: Iterable[int],
    probs: Iterable[float],
    space: str = 'logit',
) -> tuple[np.ndarray, np.ndarray]:
    """Pool per-image probabilities into one score per group (subject).

    ``space='logit'`` returns the sigmoid of the mean logit. The paper reports per-subject AUC
    with this convention (0.9592 under the deployment protocol) and the Android app scores a
    session the same way. ``space='prob'`` averages probabilities (0.9643 on the same data).
    """
    if space not in {'logit', 'prob'}:
        raise ValueError(f"space must be 'logit' or 'prob', got {space!r}")
    buckets: dict[str, list[float]] = {}
    subject_labels: dict[str, int] = {}
    for group, label, prob in zip(groups, labels, probs):
        buckets.setdefault(group, []).append(float(prob))
        subject_labels[group] = int(label)

    ordered = sorted(buckets)
    labels_out = np.array([subject_labels[group] for group in ordered], dtype=np.int32)
    if space == 'prob':
        probs_out = np.array([np.mean(buckets[group]) for group in ordered], dtype=np.float32)
    else:
        eps = 1e-6
        scores = []
        for group in ordered:
            p = np.clip(np.asarray(buckets[group], dtype=np.float64), eps, 1.0 - eps)
            scores.append(1.0 / (1.0 + np.exp(-np.mean(np.log(p / (1.0 - p))))))
        probs_out = np.array(scores, dtype=np.float32)
    return labels_out, probs_out
