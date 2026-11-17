"""Wire the components together for the API, with a stub backend for tests/CI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from .classify import LinearProbe, ZeroShotClassifier
from .config import ARTIFACT_DIR, CACHE_DIR, DEFAULT_MODEL
from .copywriting.providers import LLMProvider, StubProvider
from .copywriting.writer import CopyWriter
from .embeddings import Embedder, HashEmbedder
from .generation.backgrounds import SceneGenerator, StubGenerator
from .generation.pipeline import ProductScenePipeline
from .generation.quality import QualityGate
from .generation.review_queue import ReviewQueue
from .generation.segment import Segmenter, WhiteBackgroundSegmenter
from .search import VectorIndex


@dataclass
class CatalogItem:
    id: int
    name: str
    attributes: dict[str, str]


@dataclass
class Services:
    backend: str
    embedder: Embedder
    index: VectorIndex
    catalog: dict[int, CatalogItem]
    zero_shot: dict[str, ZeroShotClassifier]
    probes: dict[str, LinearProbe]
    pipeline: ProductScenePipeline
    review_queue: ReviewQueue
    copywriter: CopyWriter
    extras: dict = field(default_factory=dict)


STUB_COLOURS = {"Black": (20, 20, 20), "Red": (200, 30, 30), "Blue": (30, 60, 200),
                "Green": (30, 150, 60), "Yellow": (230, 210, 40)}
STUB_TYPES = ["Tshirts", "Shorts", "Dresses"]


def stub_product_image(colour: str, article_type: str, size=(96, 128)) -> Image.Image:
    """Flat-colour garment silhouette on white: enough for plumbing tests."""
    im = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(im)
    w, h = size
    rgb = STUB_COLOURS[colour]
    if article_type == "Tshirts":
        d.rectangle([w * 0.25, h * 0.2, w * 0.75, h * 0.8], fill=rgb)
        d.rectangle([w * 0.1, h * 0.2, w * 0.9, h * 0.35], fill=rgb)
    elif article_type == "Shorts":
        d.rectangle([w * 0.25, h * 0.3, w * 0.75, h * 0.5], fill=rgb)
        d.rectangle([w * 0.25, h * 0.5, w * 0.45, h * 0.7], fill=rgb)
        d.rectangle([w * 0.55, h * 0.5, w * 0.75, h * 0.7], fill=rgb)
    else:
        d.polygon([(w * 0.4, h * 0.15), (w * 0.6, h * 0.15), (w * 0.85, h * 0.9),
                   (w * 0.15, h * 0.9)], fill=rgb)
    return im


def build_stub_services(review_path=None) -> Services:
    emb = HashEmbedder()
    catalog: dict[int, CatalogItem] = {}
    images = []
    pid = 1
    for t in STUB_TYPES:
        for c in STUB_COLOURS:
            catalog[pid] = CatalogItem(pid, f"{c} {t}", {"category": t, "colour": c})
            images.append(stub_product_image(c, t))
            pid += 1
    index = VectorIndex(emb.dim)
    index.add(list(catalog), emb.embed_images(images))
    seg = WhiteBackgroundSegmenter()
    zs = {"category": ZeroShotClassifier.build(emb, "category", STUB_TYPES),
          "colour": ZeroShotClassifier.build(emb, "colour", list(STUB_COLOURS))}
    # A probe trained on the stub images makes /classify meaningful in tests.
    probes = {"colour": _numpy_probe(emb.embed_images(images),
                                     [catalog[i].attributes["colour"] for i in catalog])}
    queue = ReviewQueue(review_path or ARTIFACT_DIR / "review_queue.jsonl")
    return Services("stub", emb, index, catalog, zs, probes,
                    ProductScenePipeline(seg, StubGenerator(), QualityGate(emb, seg),
                                         canvas=(128, 128)),
                    queue, CopyWriter(StubProvider()))


def _numpy_probe(X: np.ndarray, y: list[str]) -> LinearProbe:
    """Nearest-centroid probe expressed as a linear layer (no sklearn needed)."""
    labels = sorted(set(y))
    cents = np.stack([X[[i for i, v in enumerate(y) if v == lab]].mean(0) for lab in labels])
    coef = cents * 20.0
    intercept = -0.5 * (coef * cents).sum(1)
    return LinearProbe(labels, coef.astype(np.float32), intercept.astype(np.float32))


def build_real_services(model_key: str = DEFAULT_MODEL, llm: LLMProvider | None = None,
                        generate: bool = True) -> Services:
    """Real models. Needs the ``ml`` extra (and ``gen`` for /generate)."""
    from .data import load_catalog
    from .embeddings import load_embedder
    from .evalutil import cached_image_embeddings

    emb = load_embedder(model_key)
    products = load_catalog()
    vecs, _ = cached_image_embeddings(model_key, products, emb)
    index = VectorIndex(emb.dim)
    index.add([p.id for p in products], vecs)
    catalog = {p.id: CatalogItem(p.id, p.name, p.attributes()) for p in products}
    zs = {"category": ZeroShotClassifier.build(emb, "category",
                                               sorted({p.article_type for p in products})),
          "colour": ZeroShotClassifier.build(emb, "colour", sorted({p.colour for p in products}))}
    probes = {}
    for task in ("category", "colour", "pattern"):
        path = CACHE_DIR / f"probe_{model_key}_{task}.json"
        if path.exists():
            probes[task] = LinearProbe.load(path)
    seg: Segmenter
    gen: SceneGenerator
    if generate:
        from .generation.backgrounds import SDTurboGenerator
        from .generation.segment import RembgSegmenter

        seg, gen = RembgSegmenter("u2net"), SDTurboGenerator()
    else:
        seg, gen = WhiteBackgroundSegmenter(), StubGenerator()
    return Services("real", emb, index, catalog, zs, probes,
                    ProductScenePipeline(seg, gen, QualityGate(emb, seg)),
                    ReviewQueue(ARTIFACT_DIR / "review_queue.jsonl"),
                    CopyWriter(llm or StubProvider()))


def build_services_from_env() -> Services:
    backend = os.environ.get("CVLAB_BACKEND", "stub")
    if backend == "stub":
        return build_stub_services()
    from .copywriting.providers import get_provider

    return build_real_services(
        model_key=os.environ.get("CVLAB_MODEL", DEFAULT_MODEL),
        llm=get_provider(os.environ.get("CVLAB_LLM", "stub")),
        generate=os.environ.get("CVLAB_GENERATE", "1") == "1",
    )
