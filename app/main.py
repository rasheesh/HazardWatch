"""HazardWatch API — FastAPI wrapper around the BertBiGRU hazard classifier.

Run:  uvicorn app.main:app --host 127.0.0.1 --port 8000
Docs: http://127.0.0.1:8000/docs   Prototype UI: http://127.0.0.1:8000/

Mutating endpoints (/ingest, /ingest-batch, /save) require the X-API-Key header.
The key comes from the HAZARDWATCH_API_KEY env var (dev default below).
"""

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.ml import THRESHOLD, HazardEngine

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
API_KEY = os.environ.get("HAZARDWATCH_API_KEY", "dev-key-hazardwatch")

engine: HazardEngine | None = None
load_error: str | None = None


def _load_engine():
    global engine, load_error
    try:
        engine = HazardEngine()
    except Exception as e:  # surface the failure via /health instead of a dead process
        load_error = f"{type(e).__name__}: {e}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Bind the port immediately; load models in the background (~1-2 min).
    # Endpoints return 503 until the engine is ready — poll /health.
    threading.Thread(target=_load_engine, daemon=True).start()
    yield
    if engine is not None and engine.dirty:  # don't silently lose ingested reviews
        engine.save()


def get_engine() -> HazardEngine:
    if engine is None:
        if load_error:
            raise HTTPException(status_code=500, detail=f"model failed to load — {load_error}")
        raise HTTPException(status_code=503, detail="model is still loading, retry shortly")
    return engine


app = FastAPI(
    title="HazardWatch API",
    description="Product-review hazard detection (BERT+BiGRU) with sentence-transformer "
    "similarity search over previously flagged hazards, plus product-level aggregation.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: str | None = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


class ReviewIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000, description="Review text to analyse")


class SimilarIn(ReviewIn):
    k: int = Field(5, ge=1, le=50, description="Number of similar hazards to return")


class IngestIn(ReviewIn):
    parent_asin: str | None = Field(None, description="Product id the review belongs to")


class IngestBatchIn(BaseModel):
    items: list[IngestIn] = Field(..., min_length=1, max_length=500)


@app.get("/health")
def health():
    if engine is None:
        status = "error" if load_error else "loading"
        return {"status": status, "model_loaded": False, "detail": load_error}
    return {"status": "ok", "model_loaded": True, **engine.stats()}


@app.post("/predict")
def predict(body: ReviewIn):
    """Hazard probability for a single review."""
    prob = get_engine().predict_prob(body.text)
    return {
        "hazard_prob": round(prob, 4),
        "is_hazard": prob >= THRESHOLD,
        "threshold": THRESHOLD,
    }


@app.post("/similar-reviews")
def similar_reviews(body: SimilarIn):
    """k most similar previously-flagged hazards (cosine similarity)."""
    return {"query": body.text, "results": get_engine().search_similar(body.text, body.k)}


@app.post("/analyze")
def analyze(body: SimilarIn):
    """Predict + similar hazards in one call (used by the prototype UI)."""
    eng = get_engine()
    prob = eng.predict_prob(body.text)
    return {
        "hazard_prob": round(prob, 4),
        "is_hazard": prob >= THRESHOLD,
        "threshold": THRESHOLD,
        "similar": eng.search_similar(body.text, body.k),
    }


@app.get("/products")
def products(limit: int = 20):
    """Products ranked by hazard risk (report count x mean hazard probability)."""
    return {"products": get_engine().product_summary(limit)}


@app.get("/products/{asin}")
def product_detail(asin: str):
    """All flagged reviews + per-year report timeline for one product."""
    detail = get_engine().product_detail(asin)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"No flagged reviews for product {asin}")
    return detail


@app.post("/ingest", dependencies=[Depends(require_api_key)])
def ingest(body: IngestIn):
    """Classify a review; if it's a hazard (and not a near-duplicate), add it to the index."""
    return get_engine().add_to_index(body.text, body.parent_asin)


@app.post("/ingest-batch", dependencies=[Depends(require_api_key)])
def ingest_batch(body: IngestBatchIn):
    eng = get_engine()
    results = [eng.add_to_index(item.text, item.parent_asin) for item in body.items]
    return {
        "processed": len(results),
        "flagged": sum(1 for r in results if r["flagged"]),
        "duplicates": sum(1 for r in results if r.get("duplicate")),
        "index_size": eng.index.ntotal,
        "results": results,
    }


@app.post("/save", dependencies=[Depends(require_api_key)])
def save():
    """Persist the stores to models/ (previous versions are backed up automatically)."""
    try:
        return get_engine().save()
    except AssertionError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


# assets referenced by the prototype UI (e.g. /static/hero.png)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
