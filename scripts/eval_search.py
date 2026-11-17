"""Text->image and image->image retrieval metrics over a FAISS index.

Writes results/search.json. Run: python scripts/eval_search.py
"""

from __future__ import annotations

import os
import sys

# faiss-cpu and torch each bundle libomp on macOS; this documented workaround lets
# both load. The script cross-checks faiss rankings against numpy below.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import random
import time
from collections import defaultdict

import numpy as np
from PIL import ImageOps

from cvlab.classify import label_text
from cvlab.config import MODELS, SEED
from cvlab.data import load_catalog
from cvlab.embeddings import load_embedder
from cvlab.evalutil import cached_image_embeddings, write_result
from cvlab.search import VectorIndex, precision_at_k, recall_at_k

KS = (1, 5, 10)


def augment(img, rng: random.Random):
    """Crop 80-90%, maybe mirror, slight brightness change: a re-shot/re-cropped listing."""
    w, h = img.size
    s = rng.uniform(0.8, 0.9)
    cw, ch = int(w * s), int(h * s)
    x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
    out = img.crop((x, y, x + cw, y + ch)).resize((w, h))
    if rng.random() < 0.5:
        out = ImageOps.mirror(out)
    from PIL import ImageEnhance

    return ImageEnhance.Brightness(out).enhance(rng.uniform(0.85, 1.15))


def ranked_ids(index: VectorIndex, q: np.ndarray, k: int, exclude: list[int] | None = None):
    hits = index.search(q, k + (1 if exclude else 0))
    out = []
    for i, row in enumerate(hits):
        ids = [h.product_id for h in row if not exclude or h.product_id != exclude[i]]
        out.append(ids[:k])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--n-image-queries", type=int, default=500)
    args = ap.parse_args()

    products = load_catalog(n_max=args.n)
    ids = [p.id for p in products]
    by_name: dict[str, set[int]] = defaultdict(set)
    by_type_colour: dict[tuple[str, str], set[int]] = defaultdict(set)
    for p in products:
        by_name[p.name.strip().lower()].add(p.id)
        by_type_colour[(p.article_type, p.colour)].add(p.id)

    # Text queries 1: the listing title (deduped). A hit is any item with that exact title.
    names = sorted(by_name)
    # Text queries 2: shopper-style "<colour> <type>" for every combo with >= 5 items.
    combos = sorted(k for k, v in by_type_colour.items() if len(v) >= 5)
    combo_queries = [f"{c.lower()} {label_text(t)}" for t, c in combos]

    rng = random.Random(SEED)
    q_idx = rng.sample(range(len(products)), args.n_image_queries)

    out = {"catalog_size": len(products), "index": None, "models": {},
           "queries": {"title": len(names), "attribute": len(combo_queries),
                       "image": args.n_image_queries}}
    for model_key in args.models:
        embedder = load_embedder(model_key)
        emb, _ = cached_image_embeddings(model_key, products, embedder)
        index = VectorIndex(emb.shape[1])
        t0 = time.perf_counter()
        index.add(ids, emb)
        build_ms = 1000 * (time.perf_counter() - t0)
        out["index"] = index.backend
        # Sanity check: faiss top-10 must equal exact numpy top-10.
        probe = emb[:50]
        np_top = np.argsort(-(probe @ emb.T), axis=1)[:, :10]
        fa_top = [[ids.index(h.product_id) for h in row] for row in index.search(probe, 10)]
        agree = float(np.mean([set(a) == set(b) for a, b in zip(np_top.tolist(), fa_top,
                                                                 strict=True)]))
        assert agree >= 0.98, f"faiss/numpy disagreement: {agree}"
        res: dict = {}

        qe = embedder.embed_texts(names)
        t1 = time.perf_counter()
        ranked = ranked_ids(index, qe, max(KS))
        search_ms = 1000 * (time.perf_counter() - t1) / len(names)
        rel = [by_name[n] for n in names]
        res["text_title"] = {f"recall@{k}": round(recall_at_k(ranked, rel, k), 4) for k in KS}

        qe = embedder.embed_texts(combo_queries)
        ranked = ranked_ids(index, qe, 10)
        rel = [by_type_colour[c] for c in combos]
        res["text_attribute"] = {"precision@10": round(precision_at_k(ranked, rel, 10), 4),
                                 "recall@1": round(recall_at_k(ranked, rel, 1), 4)}

        aug = embedder.embed_images([augment(products[i].image(), rng) for i in q_idx])
        ranked = ranked_ids(index, aug, max(KS))
        rel = [{products[i].id} for i in q_idx]
        res["image_near_duplicate"] = {f"recall@{k}": round(recall_at_k(ranked, rel, k), 4)
                                       for k in KS}

        ranked = ranked_ids(index, emb[q_idx], 10, exclude=[products[i].id for i in q_idx])
        rel = [by_type_colour[(products[i].article_type, products[i].colour)] - {products[i].id}
               for i in q_idx]
        rel_type = [{p.id for p in products if p.article_type == products[i].article_type}
                    - {products[i].id} for i in q_idx]
        res["image_similar"] = {"precision@10_same_type_and_colour":
                                round(precision_at_k(ranked, rel, 10), 4),
                                "precision@10_same_type":
                                round(precision_at_k(ranked, rel_type, 10), 4)}
        res["faiss_numpy_top10_agreement"] = agree
        res["latency_ms_per_query_search_only"] = round(search_ms, 3)
        res["index_build_ms"] = round(build_ms, 2)
        out["models"][model_key] = res
        print(model_key, res)
        del embedder
    print("wrote", write_result("search", out))


if __name__ == "__main__":
    main()
