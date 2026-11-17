import numpy as np

from cvlab.search import VectorIndex, precision_at_k, recall_at_k


def test_index_returns_exact_neighbours():
    rng = np.random.default_rng(1)
    vecs = rng.normal(size=(50, 16)).astype(np.float32)
    idx = VectorIndex(16)
    idx.add(list(range(100, 150)), vecs)
    hits = idx.search(vecs[7], k=3)[0]
    assert hits[0].product_id == 107
    assert hits[0].score > hits[1].score >= hits[2].score
    sims = (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)) @ (vecs[7] / np.linalg.norm(vecs[7]))
    assert [h.product_id for h in hits] == [100 + i for i in np.argsort(-sims)[:3]]


def test_index_k_larger_than_index_and_empty():
    idx = VectorIndex(4)
    assert idx.search(np.ones(4), k=5) == [[]]
    idx.add([1, 2], np.eye(4)[:2])
    assert len(idx.search(np.ones(4), k=10)[0]) == 2


def test_dim_mismatch_raises():
    idx = VectorIndex(4)
    try:
        idx.add([1], np.ones((1, 3)))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_retrieval_metrics():
    ranked = [[1, 2, 3], [4, 5, 6]]
    relevant = [{3}, {9}]
    assert recall_at_k(ranked, relevant, 1) == 0.0
    assert recall_at_k(ranked, relevant, 3) == 0.5
    assert precision_at_k(ranked, [{1, 2}, {4}], 2) == (1.0 + 0.5) / 2
