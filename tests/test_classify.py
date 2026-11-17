import numpy as np
import pytest

from cvlab.classify import LinearProbe, ZeroShotClassifier, accuracy, label_text, macro_f1
from cvlab.embeddings import HashEmbedder


def test_accuracy_and_macro_f1():
    y = ["a", "a", "b", "b"]
    p = ["a", "b", "b", "b"]
    assert accuracy(y, p) == 0.75
    # a: P=1, R=.5, F1=.667 ; b: P=.667, R=1, F1=.8
    assert macro_f1(y, p) == pytest.approx((2 / 3 + 0.8) / 2)


def test_macro_f1_penalises_majority_guessing():
    y = ["a"] * 9 + ["b"]
    p = ["a"] * 10
    assert accuracy(y, p) == 0.9
    assert macro_f1(y, p) < 0.5


def test_label_text_maps_awkward_labels():
    assert label_text("Tshirts") == "t-shirts"
    assert label_text("Jeans") == "jeans"


def test_zero_shot_picks_most_similar_prompt():
    emb = HashEmbedder()
    zs = ZeroShotClassifier.build(emb, "colour", ["Red", "Blue"])
    assert zs.text_emb.shape == (2, emb.dim)
    np.testing.assert_allclose(np.linalg.norm(zs.text_emb, axis=1), 1.0, rtol=1e-5)
    assert zs.predict(zs.text_emb) == ["Red", "Blue"]


def test_linear_probe_numpy_inference_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (30, 8)) + 3, rng.normal(0, 1, (30, 8)) - 3])
    probe = LinearProbe(["neg", "pos"], np.stack([-np.ones(8), np.ones(8)]).astype(np.float32),
                        np.zeros(2, np.float32))
    preds = probe.predict(X)
    assert preds[:30] == ["pos"] * 30 and preds[30:] == ["neg"] * 30
    np.testing.assert_allclose(probe.proba(X).sum(1), 1.0, rtol=1e-5)
    path = tmp_path / "p.json"
    probe.save(path)
    assert LinearProbe.load(path).predict(X) == preds
