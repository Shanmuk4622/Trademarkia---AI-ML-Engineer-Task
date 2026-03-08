# 20 Newsgroups — Semantic Search Engine

**Trademarkia AI/ML Engineer Task** | Built by Shanmuk

A production-ready semantic search engine on the 20 Newsgroups dataset (~20k articles). Includes two-phase fuzzy clustering, a custom cluster-aware semantic cache, and a rich interactive web UI.

---

## ⚡ Quick Start (Docker — Recommended)

You only need **Docker Desktop** installed. One command does everything:

```bash
git clone https://github.com/Shanmuk4622/Trademarkia---AI-ML-Engineer-Task.git
cd Trademarkia---AI-ML-Engineer-Task
docker-compose up --build
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser.

> **First-run:** The container builds the vector DB and clusters automatically (~15 min CPU). All later `docker-compose up` calls are **instant** — data is cached in volume mounts.

---

## ⚡ GPU Fast-Start (2 min instead of 15)

Docker on Windows cannot access your NVIDIA GPU. Run step 1 natively to use it:

```bash
# Step 1 — clone and run pipeline with GPU (~2 min)
git clone https://github.com/Shanmuk4622/Trademarkia---AI-ML-Engineer-Task.git
cd Trademarkia---AI-ML-Engineer-Task
conda activate cv_conda          # environment with CUDA PyTorch
pip install -r requirements.txt  # only on first run in a new clone
python scripts/run_pipeline.py

# Step 2 — Docker finds pre-built data, skips pipeline, ready in ~5 sec
docker-compose up --build
```

> **Troubleshooting:** If you see `module 'torch.library' has no attribute 'register_fake'`, run:
> `pip install torchvision==0.17.2 --force-reinstall`
> This pins torchvision to the exact version compatible with `torch==2.2.2`.

**Once running, explore:**
1. **[http://localhost:8000](http://localhost:8000)** — Interactive Search UI
2. **[http://localhost:8000/docs](http://localhost:8000/docs)** — Swagger API Explorer
3. **[http://localhost:8000/cache/stats](http://localhost:8000/cache/stats)** — Live cache telemetry

---

## 🧪 Sample Queries to Test the Semantic Cache

Try these pairs in the UI to see the cache in action:

**Pair 1 — Apple / Mac Hardware**
1. `what is the word apple means` → hits the corpus (~300ms)
2. `Macintosh display problems` → **⚡ CACHE HIT** (same semantic cluster, <1ms)

**Pair 2 — Space / NASA**
1. `NASA space shuttle launch Mars mission` → new cache entry
2. `space shuttle rocket NASA orbit` → **⚡ CACHE HIT**

**Pair 3 — Threshold boundary (should MISS)**
1. `commercial satellite launch SpaceX` → **CACHE MISS** (correctly identified as different intent from the NASA result above — demonstrates the 0.80 threshold heuristic)

---

## 🛠 Manual Run (without Docker)

```bash
conda activate cv_conda
pip install -r requirements.txt
python scripts/run_pipeline.py          # Part 1 + 2: embed & cluster (run once)
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

---

## 🧠 Architecture & Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | 384-dim, purpose-built for retrieval, ~3× faster than large models |
| Vector store | ChromaDB | File-backed, zero infra, cosine HNSW index |
| Clustering | Two-phase FCM | Sample 5k → fit centers → predict on all 14k |
| Cluster count | k=13 (silhouette) | Avoids assuming 20 labels = true semantic structure |
| Cache lookup | Cluster-bucketed cosine | O(N/k) vs O(N) — ~13× speedup as cache grows |
| Cache threshold | 0.80 | Sweet spot: rephrasings match, dissimilar queries separate cleanly |
| Cache persistence | Pure JSON | No Redis, no SQLite — built from scratch as required |
| State management | `app.state` singleton | Single model+cache instance, no race conditions |

See `OverviewReadme.md` for a deep-dive per-file architecture explanation.  
See `notebooks/heuristic_analysis.ipynb` for the threshold tuning analysis.

---

## Project Structure

```
src/
  data_pipeline.py   Part 1: load → clean → embed → ChromaDB
  clustering.py      Part 2: two-phase fuzzy C-Means
  semantic_cache.py  Part 3: cluster-aware semantic cache (from scratch)
  api.py             Part 4: FastAPI service + interactive UI
  static/index.html  Web UI
scripts/
  run_pipeline.py    Master pipeline runner (Parts 1 + 2)
  analyze_thresholds.py  Cache threshold sweep study
notebooks/
  heuristic_analysis.ipynb  Tuning decisions proof
tests/
  test_cache.py   SemanticCache unit tests
  test_api.py     API integration tests
Dockerfile         Standard image (data via volume mounts)
Dockerfile.bundled Fat image with data baked in
docker-compose.yml
```
