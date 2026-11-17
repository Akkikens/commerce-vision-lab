"""Compose a product cut-out onto a generated scene, optionally harmonise, then gate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .backgrounds import SceneGenerator
from .quality import GateResult, QualityGate
from .segment import Segmenter

SCENES = {
    "studio": "a minimalist yoga studio with light wooden floor and white walls, soft "
              "natural window light, professional product photography",
    "outdoor": "a sunny beach boardwalk at golden hour, blurred background, professional "
               "fashion photography",
    "urban": "a clean city street with concrete wall, overcast soft light, editorial "
             "fashion photography",
}


@dataclass
class GenerationResult:
    image: Image.Image
    placed_mask: np.ndarray
    cutout: Image.Image
    prompt: str
    variant: str
    gate: GateResult | None = None


def place(product: Image.Image, mask: np.ndarray, canvas: tuple[int, int],
          fill: float = 0.85) -> tuple[Image.Image, np.ndarray]:
    """Crop to the mask's bounding box, scale to ``fill`` of canvas height, bottom-centre.

    Returns an RGBA layer the size of the canvas and the placed boolean mask.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        raise ValueError("segmenter found no foreground")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgba = product.convert("RGBA")
    alpha = Image.fromarray((mask * 255).astype(np.uint8))
    rgba.putalpha(alpha)
    crop = rgba.crop((x0, y0, x1, y1))
    cw, ch = canvas
    scale = min(fill * ch / crop.height, 0.9 * cw / crop.width)
    crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))),
                       Image.LANCZOS)
    layer = Image.new("RGBA", canvas, (0, 0, 0, 0))
    ox = (cw - crop.width) // 2
    oy = ch - crop.height - int(0.04 * ch)
    layer.paste(crop, (ox, oy), crop)
    placed = np.asarray(layer.getchannel("A")) >= 128
    return layer, placed


class ProductScenePipeline:
    """cut-out -> scene -> composite [-> img2img harmonise] -> quality gate."""

    def __init__(self, segmenter: Segmenter, generator: SceneGenerator, gate: QualityGate,
                 canvas: tuple[int, int] = (512, 512)):
        self.segmenter = segmenter
        self.generator = generator
        self.gate = gate
        self.canvas = canvas

    def run(self, product: Image.Image, product_desc: str, scene: str = "studio",
            seed: int = 0, harmonize_strength: float = 0.0) -> GenerationResult:
        scene_prompt = SCENES.get(scene, scene)
        mask = self.segmenter.mask(product)
        layer, placed = place(product, mask, self.canvas)
        background = self.generator.generate(scene_prompt, self.canvas, seed)
        composite = background.convert("RGBA")
        composite.alpha_composite(layer)
        out = composite.convert("RGB")
        full_prompt = f"{product_desc}, {scene_prompt}"
        variant = "composite"
        if harmonize_strength > 0:
            # No re-paste afterwards: whatever img2img does to the product is what the
            # gate sees, so the preservation checks can actually fail.
            out = self.generator.refine(out, full_prompt, harmonize_strength, seed)
            variant = f"harmonized@{harmonize_strength:g}"
        cutout = layer.convert("RGB")
        res = GenerationResult(out, placed, cutout, full_prompt, variant)
        res.gate = self.gate.evaluate(out, placed, layer, product_desc, scene_prompt)
        return res
