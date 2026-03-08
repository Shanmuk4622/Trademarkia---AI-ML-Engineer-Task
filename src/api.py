"""
api.py — Part 4: FastAPI Service

Endpoints:
  POST   /query          — semantic search with cache lookup
  GET    /cache/stats    — cache statistics
  DELETE /cache          — flush cache + reset stats

State management:
  All heavy objects (model, ChromaDB collection, cluster centres, cache) are
  initialised ONCE in the lifespan context manager and stored on app.state.
  Endpoints access them via request.app.state.* — no module-level globals,
  no repeated model loading, no race conditions on initialisation.

The SemanticCache is a singleton (one instance per process) attached to app.state.cache.
Python's GIL + the cache's internal threading.Lock protect concurrent writes.
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List

import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.data_pipeline import get_collection, get_embedding_model, CHROMA_DIR
from src.clustering import load_cluster_data
from src.semantic_cache import SemanticCache, DEFAULT_THRESHOLD

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CLUSTERS_DIR = PROJECT_ROOT / "clusters"
STATIC_DIR = PROJECT_ROOT / "src" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Lifespan: initialise all heavy resources once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Everything inside the 'try' runs at startup; 'finally' runs at shutdown.

    We use app.state instead of module globals to:
    1. Avoid import-time side effects (model downloads, disk I/O)
    2. Make dependencies explicit and testable (TestClient can override app.state)
    3. Ensure there is exactly one instance of each resource per process
    """
    log.info("=== Startup: loading models and data ===")

    # 1. Embedding model
    log.info("Loading sentence-transformers model...")
    app.state.model = get_embedding_model()

    # 2. ChromaDB collection
    log.info("Connecting to ChromaDB...")
    try:
        app.state.collection = get_collection()
        log.info(f"ChromaDB ready — {app.state.collection.count()} documents indexed")
    except Exception as e:
        log.error(f"ChromaDB not ready: {e}. Run scripts/run_pipeline.py first.")
        raise RuntimeError("ChromaDB collection not found. Run the pipeline first.") from e

    # 3. Fuzzy cluster data (centres needed for cache lookup)
    log.info("Loading cluster centres...")
    try:
        U, centres, doc_ids, categories = load_cluster_data()
        app.state.cluster_U = U              # (k, n_docs) — full membership matrix
        app.state.cluster_centres = centres  # (k, emb_dim) — for new query assignment
        app.state.cluster_doc_ids = doc_ids
        app.state.cluster_categories = categories
        log.info(f"Loaded {U.shape[0]} clusters, {U.shape[1]} documents")
    except Exception as e:
        log.error(f"Cluster data not found: {e}. Run scripts/run_pipeline.py first.")
        raise RuntimeError("Cluster assignments not found. Run the pipeline first.") from e

    # 4. Semantic cache (restored from disk if available)
    log.info("Initialising semantic cache...")
    app.state.cache = SemanticCache.load_from_disk(
        centres=app.state.cluster_centres,
        threshold=DEFAULT_THRESHOLD,
    )
    log.info(f"Cache ready — {len(app.state.cache._entries)} existing entries")

    log.info("=== Startup complete. Ready to serve requests. ===")
    log.info(">>> Open your browser at: http://localhost:8000 <<<")
    yield

    # Shutdown
    log.info("=== Shutdown: persisting cache ===")
    app.state.cache._save()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="20 Newsgroups Semantic Search",
    description=(
        "Fuzzy-clustered semantic search with a cluster-aware semantic cache. "
        "Built for the Trademarkia AI/ML Engineer task."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language query")
    threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Similarity threshold override (0.0–1.0). "
            "Defaults to the cache's configured threshold (0.80). "
            "Lower = more cache hits but potentially less precise matches."
        ),
    )
    n_results: int = Field(5, ge=1, le=20, description="Number of search results to return on a miss")


class SearchResult(BaseModel):
    id: int
    category: str
    similarity: float
    text: str


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str]
    similarity_score: Optional[float]
    result: List[SearchResult]
    dominant_cluster: int


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float
    threshold: float


# ---------------------------------------------------------------------------
# Helper: embed + assign cluster
# ---------------------------------------------------------------------------

def embed_query(query: str, model) -> np.ndarray:
    """Embed a query string and return an L2-normalised float32 vector."""
    vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    return vec[0].astype(np.float32)


def search_corpus(
    query_vec: np.ndarray,
    collection,
    dominant_cluster: int,
    cluster_doc_ids,
    cluster_U: np.ndarray,
    categories,
    n_results: int = 5,
) -> List[dict]:
    """
    Query ChromaDB for the most semantically similar documents.

    We do a simple ANN search across all documents (ChromaDB handles the HNSW
    index internally). The cluster information is returned in the response for
    transparency, but the raw vector search already gives the best results.

    Returns a formatted list of dictionaries containing the best matching documents.
    """
    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]  # cosine distance = 1 - cosine_similarity

    results_list = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        similarity = round(1.0 - dist, 4)
        results_list.append({
            "id": i + 1,
            "category": meta["category"],
            "similarity": similarity,
            "text": doc.strip()
        })

    return results_list


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    """Serves the interactive frontend UI."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return "<h1>UI not found</h1><p>Please ensure src/static/index.html exists.</p>"

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(body: QueryRequest, request: Request):
    """
    Semantic search with cluster-aware cache.

    Flow:
      1. Embed the query
      2. Look up the cache (cluster-bucketed cosine similarity)
      3a. Cache HIT  → return cached result immediately
      3b. Cache MISS → search ChromaDB, store result, return
    """
    cache: SemanticCache = request.app.state.cache
    model = request.app.state.model
    collection = request.app.state.collection
    cluster_U = request.app.state.cluster_U
    cluster_doc_ids = request.app.state.cluster_doc_ids
    cluster_categories = request.app.state.cluster_categories

    # Apply per-request threshold override if provided
    original_threshold = cache.threshold
    if body.threshold is not None:
        cache.threshold = body.threshold

    try:
        # 1. Embed
        query_vec = embed_query(body.query, model)

        # 2. Assign dominant cluster (for response + cache routing)
        dominant, membership = cache._assign_cluster(query_vec)

        # 3. Cache lookup
        hit = cache.lookup(query_vec)

        if hit is not None:
            cached_entry, score = hit
            cache.record_hit()
            return QueryResponse(
                query=body.query,
                cache_hit=True,
                matched_query=cached_entry.query,
                similarity_score=round(score, 4),
                result=cached_entry.result,
                dominant_cluster=dominant,
            )

        # 4. Cache miss — search corpus
        cache.record_miss()
        result = search_corpus(
            query_vec,
            collection,
            dominant,
            cluster_doc_ids,
            cluster_U,
            cluster_categories,
            n_results=body.n_results,
        )

        # 5. Store in cache (lock-protected)
        cache.insert(query=body.query, vector=query_vec, result=result)

        return QueryResponse(
            query=body.query,
            cache_hit=False,
            matched_query=None,
            similarity_score=None,
            result=result,
            dominant_cluster=dominant,
        )

    finally:
        # Restore threshold if it was overridden per-request
        if body.threshold is not None:
            cache.threshold = original_threshold


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats(request: Request):
    """
    Return current cache statistics.

    This endpoint is read-only and never modifies cache state.
    """
    cache: SemanticCache = request.app.state.cache
    stats = cache.stats
    return CacheStatsResponse(**stats)


@app.delete("/cache")
async def flush_cache(request: Request):
    """
    Flush the entire cache and reset all counters.

    This is a destructive operation. All cached queries and their results
    will be lost. The on-disk JSON file is also cleared.
    """
    cache: SemanticCache = request.app.state.cache
    cache.flush()
    return {"message": "Cache flushed successfully.", "status": "ok"}


@app.get("/health")
async def health_check(request: Request):
    """Quick health check — returns model and collection status."""
    try:
        n_docs = request.app.state.collection.count()
        n_clusters = request.app.state.cluster_U.shape[0]
        cache_entries = len(request.app.state.cache._entries)
        return {
            "status": "healthy",
            "indexed_documents": n_docs,
            "n_clusters": n_clusters,
            "cached_queries": cache_entries,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
