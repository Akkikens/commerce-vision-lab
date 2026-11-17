"""Automated quality gate for generated product images.

Three checks, each computed on the *final* output image:

1. ``prompt_alignment``: CLIP cosine(output, "<product>, <scene>") must beat the
   output's similarity to a neutral control caption by a margin. A raw CLIP cosine is
   poorly calibrated across prompts; the margin is not.
2. ``mask_iou``: re-segment the output and compare with where we placed the product.
   Catches a product that was repainted, merged into the scene, or lost.
3. ``product_similarity``: CLIP cosine between the product region of the output and
   the original cut-out (both on a flat grey ground). A coarse semantic check.
4. ``product_ssim`` and ``colour_delta_e``: pixel-level structure and colour fidelity
   inside the placed product mask. CLIP embeddings barely move when img2img redraws
   a garment's details (ties, seams, prints) or a model's face; these do.

Anything that fails a check goes to the human review queue with the reasons.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image

from ..embeddings import Embedder
from .segment import Segmenter, mask_iou

NEUTRAL_CAPTION = "a photo"


@dataclass
class Thresholds:
    """Defaults are the values calibrated by scripts/eval_generation.py on a
    calibration split (results/generation.json), not on the reported test split.
    ``prompt_margin`` is set a priori."""

    prompt_margin: float = 0.03
    mask_iou: float = 0.9572
    product_similarity: float = 0.9485
    product_ssim: float = 0.8413
    colour_delta_e: float = 3.9113  # maximum mean CIE76 delta E


@dataclass
class GateResult:
    prompt_alignment: float
    prompt_margin: float
    mask_iou: float
    product_similarity: float
    product_ssim: float
    colour_delta_e: float
    passed: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _on_grey(rgb: Image.Image, mask: np.ndarray) -> Image.Image:
    a = np.asarray(rgb.convert("RGB")).copy()
    a[~mask] = 128
    ys, xs = np.nonzero(mask)
    if len(ys):
        a = a[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    return Image.fromarray(a)


def _box(x: np.ndarray, r: int) -> np.ndarray:
    """Mean filter with a (2r+1)^2 window via integral images (edge-padded)."""
    p = np.pad(x, r + 1, mode="edge").cumsum(0).cumsum(1)
    k = 2 * r + 1
    out = p[k:, k:] - p[:-k, k:] - p[k:, :-k] + p[:-k, :-k]
    return out[: x.shape[0], : x.shape[1]] / (k * k)


def masked_ssim(a: Image.Image, b: Image.Image, mask: np.ndarray, r: int = 3) -> float:
    """Mean SSIM (grayscale, box window) over pixels whose whole window is inside the mask.

    Eroding the mask by the window radius keeps background pixels (which legitimately
    differ between the scene and the reference) out of the product score.
    """
    x = np.asarray(a.convert("L"), dtype=np.float64)
    y = np.asarray(b.convert("L"), dtype=np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = _box(x, r), _box(y, r)
    vx = _box(x * x, r) - mx * mx
    vy = _box(y * y, r) - my * my
    cxy = _box(x * y, r) - mx * my
    ssim = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))
    inner = _box(mask.astype(np.float64), r) > 1 - 1e-9
    if not inner.any():
        inner = mask
    return float(ssim[inner].mean()) if inner.any() else 0.0


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    c = rgb / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def mean_delta_e(a: Image.Image, b: Image.Image, mask: np.ndarray) -> float:
    la = _srgb_to_lab(np.asarray(a.convert("RGB"), dtype=np.float64)[mask])
    lb = _srgb_to_lab(np.asarray(b.convert("RGB"), dtype=np.float64)[mask])
    return float(np.linalg.norm(la - lb, axis=-1).mean()) if len(la) else float("inf")


class QualityGate:
    def __init__(self, embedder: Embedder, segmenter: Segmenter,
                 thresholds: Thresholds | None = None):
        self.embedder = embedder
        self.segmenter = segmenter
        self.t = thresholds or Thresholds()

    def evaluate(self, output: Image.Image, placed_mask: np.ndarray, layer: Image.Image,
                 product_desc: str, scene_prompt: str) -> GateResult:
        img_e = self.embedder.embed_images([output])[0]
        txt = self.embedder.embed_texts([f"{product_desc}, {scene_prompt}", NEUTRAL_CAPTION])
        align = float(img_e @ txt[0])
        margin = float(align - img_e @ txt[1])

        out_mask = self.segmenter.mask(output)
        iou = mask_iou(out_mask, placed_mask)

        a, b = self.embedder.embed_images([_on_grey(output, placed_mask),
                                           _on_grey(layer.convert("RGB"), placed_mask)])
        prod_sim = float(a @ b)
        ref = layer.convert("RGB")
        ssim = masked_ssim(output, ref, placed_mask)
        de = mean_delta_e(output, ref, placed_mask)

        reasons = []
        if margin < self.t.prompt_margin:
            reasons.append(f"prompt margin {margin:.3f} < {self.t.prompt_margin}")
        if iou < self.t.mask_iou:
            reasons.append(f"mask IoU {iou:.3f} < {self.t.mask_iou}")
        if prod_sim < self.t.product_similarity:
            reasons.append(f"product similarity {prod_sim:.3f} < {self.t.product_similarity}")
        if ssim < self.t.product_ssim:
            reasons.append(f"product SSIM {ssim:.3f} < {self.t.product_ssim}")
        if de > self.t.colour_delta_e:
            reasons.append(f"colour delta E {de:.1f} > {self.t.colour_delta_e}")
        return GateResult(align, margin, iou, prod_sim, ssim, de, not reasons, reasons)
