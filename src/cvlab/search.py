"""Vector index for product search (FAISS inner product, numpy fallback)."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .embeddings import l2norm


@dataclass
class Hit:
    product_id: int
    score: float


class VectorIndex:
    """Exact cosine-similarity index over L2-normalised vectors.

    Uses ``faiss.IndexFlatIP`` when faiss is installed, else a numpy matmul. At
    catalog sizes in the low hundreds of thousands exact search is fine; an IVF or
    HNSW index (or pgvector) would replace this beyond that.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.ids: list[int] = []
        self._vecs = np.zeros((0, dim), dtype=np.float32)
        try:
            import faiss

            if sys.platform == "darwin":
                # faiss and torch both bundle libomp on macOS; multi-threaded faiss
                # next to torch segfaults. Exact search at this scale is fast single-threaded.
                faiss.omp_set_num_threads(1)
            self._faiss = faiss.IndexFlatIP(dim)
        except ImportError:  # pragma: no cover - exercised when faiss is absent
            self._faiss = None

    @property
    def backend(self) -> str:
        return "faiss" if self._faiss is not None else "numpy"

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, ids: Sequence[int], vecs: np.ndarray) -> None:
        vecs = l2norm(vecs)
        if vecs.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vecs.shape[1]}")
        self.ids.extend(int(i) for i in ids)
        if self._faiss is not None:
            self._faiss.add(vecs)
        else:
            self._vecs = np.vstack([self._vecs, vecs])

    def search(self, queries: np.ndarray, k: int = 10) -> list[list[Hit]]:
        queries = l2norm(np.atleast_2d(queries))
        k = min(k, len(self.ids))
        if k == 0:
            return [[] for _ in range(len(queries))]
        if self._faiss is not None:
            scores, idx = self._faiss.search(queries, k)
        else:
            sims = queries @ self._vecs.T
            idx = np.argsort(-sims, axis=1)[:, :k]
            scores = np.take_along_axis(sims, idx, axis=1)
        return [[Hit(self.ids[j], float(s)) for j, s in zip(row_i, row_s, strict=True) if j >= 0]
                for row_i, row_s in zip(idx, scores, strict=True)]


def recall_at_k(ranked: Sequence[Sequence[int]], relevant: Sequence[set[int]], k: int) -> float:
    """Fraction of queries with at least one relevant id in the top k."""
    hits = [bool(set(r[:k]) & rel) for r, rel in zip(ranked, relevant, strict=True)]
    return float(np.mean(hits))


def precision_at_k(ranked: Sequence[Sequence[int]], relevant: Sequence[set[int]], k: int) -> float:
    vals = [len(set(r[:k]) & rel) / k for r, rel in zip(ranked, relevant, strict=True)]
    return float(np.mean(vals))
