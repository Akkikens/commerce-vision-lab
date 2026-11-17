"""Foreground segmentation (background removal)."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image


class Segmenter(Protocol):
    name: str

    def mask(self, image: Image.Image) -> np.ndarray:
        """Return a boolean HxW foreground mask."""
        ...


class RembgSegmenter:
    """U^2-Net via rembg (``gen`` extra). ``u2netp`` is the 4.7 MB variant."""

    def __init__(self, model: str = "u2net", threshold: int = 128):
        from rembg import new_session

        self.name = f"rembg-{model}"
        self.session = new_session(model)
        self.threshold = threshold

    def mask(self, image: Image.Image) -> np.ndarray:
        from rembg import remove

        m = remove(image.convert("RGB"), session=self.session, only_mask=True)
        return np.asarray(m.convert("L")) >= self.threshold


class WhiteBackgroundSegmenter:
    """Model-free baseline: anything not near-white is foreground.

    Works on studio packshots on white and is used in tests; it fails on white
    garments, which is exactly why the real pipeline uses rembg.
    """

    name = "white-threshold"

    def __init__(self, tol: int = 18):
        self.tol = tol

    def mask(self, image: Image.Image) -> np.ndarray:
        a = np.asarray(image.convert("RGB"), dtype=np.int16)
        return (255 - a).max(axis=2) > self.tol


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0
