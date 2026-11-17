"""Scene generators behind a provider interface.

``SDTurboGenerator`` runs stabilityai/sd-turbo locally (diffusers, MPS/CUDA/CPU).
``StubGenerator`` returns a deterministic gradient so tests need no model.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np
from PIL import Image


class SceneGenerator(Protocol):
    name: str

    def generate(self, prompt: str, size: tuple[int, int], seed: int) -> Image.Image: ...

    def refine(self, image: Image.Image, prompt: str, strength: float, seed: int) -> Image.Image:
        """Image-to-image pass over a whole composite (lighting/shadow harmonisation)."""
        ...


class StubGenerator:
    name = "stub-gradient"

    def generate(self, prompt: str, size: tuple[int, int], seed: int) -> Image.Image:
        h = hashlib.sha1(f"{prompt}|{seed}".encode()).digest()
        top, bottom = np.array(list(h[:3]), np.float32), np.array(list(h[3:6]), np.float32)
        w, hgt = size
        t = np.linspace(0, 1, hgt, dtype=np.float32)[:, None, None]
        grad = (1 - t) * top + t * bottom
        return Image.fromarray(np.broadcast_to(grad, (hgt, w, 3)).astype(np.uint8))

    def refine(self, image: Image.Image, prompt: str, strength: float, seed: int) -> Image.Image:
        # Blend toward a generated scene in proportion to strength: a crude stand-in
        # for img2img that, like the real thing, damages the product at high strength.
        scene = self.generate(prompt, image.size, seed)
        return Image.blend(image.convert("RGB"), scene, float(strength))


class SDTurboGenerator:
    """SD-Turbo (1-4 step distilled SD 2.1, 512px). Tiny settings to stay laptop-friendly."""

    def __init__(self, model_id: str = "stabilityai/sd-turbo", device: str | None = None,
                 steps: int = 1):
        import torch
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

        from ..embeddings import pick_device

        self.torch = torch
        self.device = device or pick_device()
        # fp16 on MPS can produce NaN/black images with some VAEs; fp32 is safe and
        # still fast at 1-4 steps.
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.t2i = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=dtype, variant="fp16"
        ).to(self.device)
        self.t2i.set_progress_bar_config(disable=True)
        self.i2i = AutoPipelineForImage2Image.from_pipe(self.t2i)
        self.i2i.set_progress_bar_config(disable=True)
        self.steps = steps
        self.name = f"sd-turbo-{steps}step"

    def _gen(self, seed: int):
        return self.torch.Generator("cpu").manual_seed(seed)

    def generate(self, prompt: str, size: tuple[int, int], seed: int) -> Image.Image:
        w, h = size
        return self.t2i(prompt=prompt, width=w, height=h, num_inference_steps=self.steps,
                        guidance_scale=0.0, generator=self._gen(seed)).images[0]

    def refine(self, image: Image.Image, prompt: str, strength: float, seed: int) -> Image.Image:
        # SD-Turbo img2img runs int(steps * strength) denoising steps; keep that >= 1.
        steps = max(2, int(np.ceil(1.0 / strength)))
        return self.i2i(prompt=prompt, image=image.convert("RGB"), strength=strength,
                        num_inference_steps=steps, guidance_scale=0.0,
                        generator=self._gen(seed)).images[0]
