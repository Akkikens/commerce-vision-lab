"""Image/text embedders behind one small protocol.

``OpenClipEmbedder`` runs real CLIP/SigLIP models (needs the ``ml`` extra).
``HashEmbedder`` is a deterministic stand-in so the API and tests run without torch.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from PIL import Image


def l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


class Embedder(Protocol):
    name: str
    dim: int

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray: ...


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class OpenClipEmbedder:
    """Wraps an open_clip model. Embeddings are L2-normalised float32."""

    def __init__(self, arch: str, pretrained: str | None, name: str, device: str | None = None,
                 batch_size: int = 64):
        import open_clip
        import torch

        self.torch = torch
        self.name = name
        self.device = device or pick_device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.batch_size = batch_size
        with torch.no_grad():
            self.dim = int(self.embed_texts(["probe"]).shape[1])

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = self.torch.stack(
                    [self.preprocess(im.convert("RGB")) for im in images[i : i + self.batch_size]]
                ).to(self.device)
                out.append(self.model.encode_image(batch).float().cpu().numpy())
        return l2norm(np.concatenate(out))

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                toks = self.tokenizer(list(texts[i : i + self.batch_size])).to(self.device)
                out.append(self.model.encode_text(toks).float().cpu().numpy())
        return l2norm(np.concatenate(out))


def load_embedder(key: str) -> OpenClipEmbedder:
    from .config import MODELS

    arch, pretrained = MODELS[key]
    return OpenClipEmbedder(arch, pretrained, name=key)


class HashEmbedder:
    """Deterministic, model-free embedder for tests and the stub API.

    Images are embedded from a coarse colour histogram, so visually similar stub
    images land near each other; texts are embedded from hashed tokens. It has no
    semantic understanding and exists only to exercise the plumbing.
    """

    name = "hash-stub"

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        rows = []
        for im in images:
            small = np.asarray(im.convert("RGB").resize((8, 8)), dtype=np.float32) / 255.0
            v = small.reshape(-1)[: self.dim]
            v = np.pad(v, (0, max(0, self.dim - v.size)))
            rows.append(v + 1e-3)
        return l2norm(np.stack(rows))

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for tok in t.lower().split():
                h = int(hashlib.sha1(tok.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0
            rows.append(v + 1e-3)
        return l2norm(np.stack(rows))
