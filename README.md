# 20 Newsgroups — Semantic Search System

**Trademarkia AI/ML Engineer Task** | Built by Shanmuk

A lightweight semantic search system with fuzzy clustering, a semantic cache, and a FastAPI service — all built on the 20 Newsgroups dataset (~20k news articles across 20 categories).

---

## Quick Start

### Prerequisites
- Conda environment `cv_conda` (or a Python 3.10+ venv)
- Dataset at `twenty+newsgroups/20_newsgroups.tar.gz`

### 1. Install Dependencies

**Option A — conda (recommended):**
```bash
conda activate cv_conda
pip install -r requirements.txt
```

**Option B — fresh venv (Windows):**
```bash
scripts\setup_venv.bat
venv\Scripts\activate.bat
```

### 2. Run the Pipeline (Parts 1 & 2)

This processes the corpus, generates embeddings, indexes them in ChromaDB, and runs fuzzy clustering. **Run once before starting the API.**

```bash
conda activate cv_conda
python scripts/run_pipeline.py
```

Quick smoke test (500 docs, ~2 min):
```bash
python scripts/run_pipeline.py --smoke-test
```

### 3. Start the API (Parts 3 & 4)

```bash
conda activate cv_conda
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

The API will be live at `http://localhost:8000`. Interactive docs: `http://localhost:8000/docs`

---

## API Endpoints

### `POST /query`
Semantic search with cluster-aware cache.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "NASA space shuttle launch"}'
```

**Response:**
```json
{
  "query": "NASA space shuttle launch",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": "[1] Category: sci.space | Similarity: 0.92\n    ...",
  "dominant_cluster": 3
}
```

On a cache hit (same or similar query asked again):
```json
{
  "query": "space shuttle NASA mission",
  "cache_hit": true,
  "matched_query": "NASA space shuttle launch",
  "similarity_score": 0.91,
  "result": "...",
  "dominant_cluster": 3
}
```

Optional: override the similarity threshold per-request:
```bash
curl -X POST http://localhost:8000/query \
  -d '{"query": "...", "threshold": 0.90}'
```

### `GET /cache/stats`
```bash
curl http://localhost:8000/cache/stats
```
```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405,
  "threshold": 0.8
}
```

### `DELETE /cache`
```bash
curl -X DELETE http://localhost:8000/cache
```

---

## Architecture

```
Raw corpus (20_newsgroups.tar.gz)
    │
    ▼ src/data_pipeline.py (Part 1)
Clean text → all-MiniLM-L6-v2 embeddings → ChromaDB
    │
    ▼ src/clustering.py (Part 2)
5k-sample FCM → cluster centres → full-corpus cmeans_predict
→ (k, n_docs) soft membership matrix
    │
    ▼ src/api.py + src/semantic_cache.py (Parts 3 & 4)
FastAPI lifespan loads model + ChromaDB + clusters once
Every /query → embed → cluster bucket lookup → cosine similarity
    ├── HIT  → return cached result
    └── MISS → ChromaDB ANN search → store → return
```

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | 384-dim, purpose-built for retrieval, ~3× faster than large models |
| Vector store | ChromaDB | File-backed, zero infra, cosine HNSW index |
| Clustering approach | Two-phase FCM | Phase 1: fit on 5k sample (fast); Phase 2: `cmeans_predict` on full corpus |
| Cluster count | Empirical (silhouette on k∈[5,25]) | Avoids assuming the 20 ground-truth labels reflect true semantic structure |
| Cache lookup | Cluster-bucketed cosine | O(N/k) not O(N) — with k=15 gives ~15× speedup |
| Cache threshold | 0.80 (configurable) | Sweet spot: semantically equivalent rephrasings match, dissimilar queries miss |
| Cache persistence | Pure `json` | No Redis, no SQLite — strictly from scratch |
| State management | `app.state` singleton | Single model/cache instance, no global variables, no race conditions |

---

## Running Tests

```bash
conda activate cv_conda
python -m pytest tests/ -v
```

---

## Docker (Bonus)

**Prerequisite:** Run the pipeline first so `embeddings/`, `clusters/`, and `cache/` are populated.

```bash
# Build & run
docker build -t trademarkia-search .
docker run -p 8000:8000 \
  -v $(pwd)/embeddings:/app/embeddings \
  -v $(pwd)/clusters:/app/clusters \
  -v $(pwd)/cache:/app/cache \
  trademarkia-search

# Or with docker-compose:
docker-compose up --build
```

---

## Project Structure

```
├── src/
│   ├── data_pipeline.py    # Part 1: load, clean, embed, index
│   ├── clustering.py       # Part 2: two-phase fuzzy clustering
│   ├── semantic_cache.py   # Part 3: cluster-aware semantic cache
│   └── api.py              # Part 4: FastAPI service
├── scripts/
│   ├── run_pipeline.py     # Master pipeline runner (Parts 1+2)
│   └── setup_venv.bat      # Windows venv setup
├── tests/
│   ├── test_cache.py       # SemanticCache unit tests
│   └── test_api.py         # API integration tests
├── notebooks/              # cluster_analysis.ipynb (analysis)
├── embeddings/             # ChromaDB persistence (auto-created)
├── clusters/               # FCM assignments + metadata (auto-created)
├── cache/                  # Cache JSON state (auto-created)
├── twenty+newsgroups/      # Raw dataset
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
