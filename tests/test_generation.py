import numpy as np
import pytest
from PIL import Image

from cvlab.embeddings import HashEmbedder
from cvlab.generation.backgrounds import StubGenerator
from cvlab.generation.pipeline import ProductScenePipeline, place
from cvlab.generation.quality import QualityGate, Thresholds, masked_ssim, mean_delta_e
from cvlab.generation.review_queue import ReviewQueue
from cvlab.generation.segment import WhiteBackgroundSegmenter, mask_iou
from cvlab.services import stub_product_image


def test_mask_iou():
    a = np.zeros((4, 4), bool)
    a[:2] = True
    b = np.zeros((4, 4), bool)
    b[1:3] = True
    assert mask_iou(a, b) == pytest.approx(4 / 12)
    assert mask_iou(a, a) == 1.0


def test_white_segmenter_finds_garment():
    img = stub_product_image("Red", "Tshirts")
    m = WhiteBackgroundSegmenter().mask(img)
    assert 0.1 < m.mean() < 0.6
    assert not m[0, 0]


def test_place_bottom_centres_and_scales():
    img = stub_product_image("Blue", "Dresses")
    m = WhiteBackgroundSegmenter().mask(img)
    layer, placed = place(img, m, (200, 200))
    ys, xs = np.nonzero(placed)
    assert layer.size == (200, 200)
    assert ys.max() > 180 and ys.max() - ys.min() > 150
    assert abs((xs.min() + xs.max()) / 2 - 100) < 5


def test_place_rejects_empty_mask():
    with pytest.raises(ValueError):
        place(Image.new("RGB", (10, 10), "white"), np.zeros((10, 10), bool), (32, 32))


def test_ssim_and_delta_e_detect_changes():
    rng = np.random.default_rng(0)
    a = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    mask = np.ones((64, 64), bool)
    assert masked_ssim(a, a, mask) == pytest.approx(1.0)
    assert mean_delta_e(a, a, mask) == pytest.approx(0.0, abs=1e-9)
    b = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    assert masked_ssim(a, b, mask) < 0.2
    assert mean_delta_e(a, b, mask) > 20


def _pipeline():
    emb, seg = HashEmbedder(), WhiteBackgroundSegmenter()
    # Prompt alignment is meaningless with the hash embedder; disable that check here.
    gate = QualityGate(emb, seg, Thresholds(prompt_margin=-1.0))
    return ProductScenePipeline(seg, StubGenerator(), gate, canvas=(128, 128))


def test_composite_preserves_product_and_passes_preservation_checks():
    res = _pipeline().run(stub_product_image("Black", "Tshirts"), "a black t-shirt", seed=3)
    g = res.gate
    assert res.variant == "composite"
    assert g.product_ssim > 0.95 and g.colour_delta_e < 1.0 and g.product_similarity > 0.99


def test_heavy_refine_is_caught_by_gate():
    res = _pipeline().run(stub_product_image("Black", "Tshirts"), "a black t-shirt", seed=3,
                          harmonize_strength=0.8)
    assert res.variant == "harmonized@0.8"
    assert not res.gate.passed
    assert any("delta E" in r or "SSIM" in r for r in res.gate.reasons)


def test_review_queue_lifecycle(tmp_path):
    q = ReviewQueue(tmp_path / "q.jsonl")
    a = q.submit({"x": 1})
    b = q.submit({"x": 2})
    assert {r["review_id"] for r in q.pending()} == {a, b}
    q.decide(a, approved=False, reviewer="me")
    assert [r["review_id"] for r in q.pending()] == [b]
    with pytest.raises(KeyError):
        q.decide("nope", True, "me")


def test_default_thresholds_match_committed_calibration():
    import json
    from pathlib import Path

    path = Path(__file__).parent.parent / "results" / "generation.json"
    calibrated = json.loads(path.read_text())["thresholds"]
    defaults = Thresholds()
    for k, v in calibrated.items():
        assert getattr(defaults, k) == pytest.approx(v, abs=1e-3), k
