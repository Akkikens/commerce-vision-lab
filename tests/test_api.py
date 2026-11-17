import base64
import io

from PIL import Image

from cvlab.services import stub_product_image


def png(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "stub" and body["index_size"] == 15


def test_classify_returns_zero_shot_and_probe(client):
    r = client.post("/classify", files={"image": ("x.png", png(stub_product_image("Green", "Shorts")))})
    assert r.status_code == 200
    body = r.json()
    assert set(body["zero_shot"]) == {"category", "colour"}
    assert body["linear_probe"]["colour"]["label"] == "Green"


def test_classify_rejects_non_image(client):
    r = client.post("/classify", files={"image": ("x.txt", b"hello")})
    assert r.status_code == 400


def test_image_search_finds_identical_product_first(client, services):
    r = client.post("/search/image", files={"image": ("x.png", png(stub_product_image("Red", "Dresses")))},
                    data={"k": "3"})
    assert r.status_code == 200
    top = r.json()["hits"][0]
    assert top["attributes"] == {"category": "Dresses", "colour": "Red"}


def test_text_search_validates_and_returns_k(client):
    assert client.post("/search", json={"text": ""}).status_code == 422
    r = client.post("/search", json={"text": "red dress", "k": 4})
    assert r.status_code == 200 and len(r.json()["hits"]) == 4


def test_generate_heavy_refine_goes_to_review_queue(client):
    img = png(stub_product_image("Blue", "Tshirts"))
    r = client.post("/generate", files={"image": ("x.png", img)},
                    data={"product_desc": "a blue t-shirt", "harmonize_strength": "0.8"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "pending_review" and body["review_id"]
    Image.open(io.BytesIO(base64.b64decode(body["image_png_base64"]))).verify()
    pending = client.get("/review/pending").json()
    assert [p["review_id"] for p in pending] == [body["review_id"]]
    d = client.post(f"/review/{body['review_id']}", json={"approved": True, "reviewer": "qa"})
    assert d.json()["status"] == "approved"
    assert client.get("/review/pending").json() == []
    assert client.post("/review/unknown", json={"approved": True, "reviewer": "qa"}).status_code == 404


def test_generate_rejects_blank_image(client):
    r = client.post("/generate", files={"image": ("x.png", png(Image.new("RGB", (32, 32), "white")))},
                    data={"product_desc": "nothing"})
    assert r.status_code == 422


def test_copy_endpoint_uses_guarded_writer(client):
    r = client.post("/copy", json={"category": "Tops", "colour": "Black", "gender": "Women"})
    assert r.status_code == 200
    body = r.json()
    assert "black" in body["alt_text"].lower() and body["final_violations"] == []
