"""FastAPI service: /classify, /search, /generate, /copy and the review queue."""

from __future__ import annotations

import base64
import io
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from .services import Services, build_services_from_env

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class SearchRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    k: int = Field(10, ge=1, le=100)


class CopyRequest(BaseModel):
    category: str
    colour: str
    gender: str | None = None
    usage: str | None = None
    pattern: str | None = None
    materials: str | None = None


class ReviewDecision(BaseModel):
    approved: bool
    reviewer: str = Field(min_length=1, max_length=100)


async def _read_image(upload: UploadFile) -> Image.Image:
    data = await upload.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image too large")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError as e:
        raise HTTPException(400, "not a valid image") from e


def create_app(services: Services | None = None) -> FastAPI:
    app = FastAPI(title="commerce-vision-lab", version="0.1.0",
                  description="Learning project. Not used by a real brand.")
    app.state.services = services

    def svc(request: Request) -> Services:
        if request.app.state.services is None:
            request.app.state.services = build_services_from_env()
        return request.app.state.services

    @app.get("/health")
    def health(request: Request) -> dict:
        s = svc(request)
        return {"status": "ok", "backend": s.backend, "embedder": s.embedder.name,
                "index_size": len(s.index), "index_backend": s.index.backend}

    @app.post("/classify")
    async def classify(request: Request, image: Annotated[UploadFile, File()]) -> dict:
        s = svc(request)
        vec = s.embedder.embed_images([await _read_image(image)])
        out: dict = {"zero_shot": {}, "linear_probe": {}}
        for task, zs in s.zero_shot.items():
            scores = zs.scores(vec)[0]
            top = scores.argsort()[::-1][:3]
            out["zero_shot"][task] = [{"label": zs.labels[i], "score": round(float(scores[i]), 4)}
                                      for i in top]
        for task, probe in s.probes.items():
            p = probe.proba(vec)[0]
            i = int(p.argmax())
            out["linear_probe"][task] = {"label": probe.labels[i],
                                         "confidence": round(float(p[i]), 4)}
        return out

    def _hits(s: Services, vec, k: int) -> list[dict]:
        return [{"product_id": h.product_id, "score": round(h.score, 4),
                 "name": s.catalog[h.product_id].name,
                 "attributes": s.catalog[h.product_id].attributes}
                for h in s.index.search(vec, k)[0]]

    @app.post("/search")
    def search_text(request: Request, body: SearchRequest) -> dict:
        s = svc(request)
        return {"query": body.text, "hits": _hits(s, s.embedder.embed_texts([body.text]), body.k)}

    @app.post("/search/image")
    async def search_image(request: Request, image: Annotated[UploadFile, File()],
                           k: Annotated[int, Form(ge=1, le=100)] = 10) -> dict:
        s = svc(request)
        return {"hits": _hits(s, s.embedder.embed_images([await _read_image(image)]), k)}

    @app.post("/generate")
    async def generate(request: Request, image: Annotated[UploadFile, File()],
                       product_desc: Annotated[str, Form(min_length=1, max_length=200)],
                       scene: Annotated[str, Form(max_length=300)] = "studio",
                       seed: Annotated[int, Form()] = 0,
                       harmonize_strength: Annotated[float, Form(ge=0.0, le=0.9)] = 0.0) -> dict:
        s = svc(request)
        try:
            res = s.pipeline.run(await _read_image(image), product_desc, scene, seed,
                                 harmonize_strength)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        buf = io.BytesIO()
        res.image.save(buf, format="PNG")
        review_id = None
        if not res.gate.passed:
            review_id = s.review_queue.submit({"product_desc": product_desc, "scene": scene,
                                               "variant": res.variant,
                                               "reasons": res.gate.reasons})
        return {"variant": res.variant, "gate": res.gate.to_dict(), "review_id": review_id,
                "status": "approved_auto" if res.gate.passed else "pending_review",
                "image_png_base64": base64.b64encode(buf.getvalue()).decode()}

    @app.post("/copy")
    def copy(request: Request, body: CopyRequest) -> dict:
        s = svc(request)
        attrs = {k: v for k, v in body.model_dump().items() if v}
        return s.copywriter.write(attrs).__dict__

    @app.get("/review/pending")
    def review_pending(request: Request) -> list[dict]:
        return svc(request).review_queue.pending()

    @app.post("/review/{review_id}")
    def review_decide(request: Request, review_id: str, body: ReviewDecision) -> dict:
        try:
            svc(request).review_queue.decide(review_id, body.approved, body.reviewer)
        except KeyError as e:
            raise HTTPException(404, "unknown review id") from e
        return {"review_id": review_id, "status": "approved" if body.approved else "rejected"}

    return app


app = create_app()
