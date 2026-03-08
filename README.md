# 20 Newsgroups — Semantic Search Engine & Cluster-Aware Cache

**Trademarkia AI/ML Engineer Task** | Built by Shanmuk

A production-ready semantic search engine built from scratch on the 20 Newsgroups dataset (~20k articles). Includes a two-phase fuzzy clustering system, a custom cluster-aware semantic cache, and a rich interactive web UI.

---

## ⚡ Quick Start: One-Command Docker Setup

Everything runs in a single command. You only need **Docker Desktop** installed.

```bash
git clone https://github.com/Shanmuk4622/Trademarkia---AI-ML-Engineer-Task.git
cd Trademarkia---AI-ML-Engineer-Task
docker-compose up --build
```

Then open **[http://localhost:8000](http://localhost:8000)** in your browser.

> **First-run note:** On first run, the container automatically builds the vector database and fuzzy clusters from the raw corpus (~15 min on CPU, ~2-3 min with GPU). All subsequent `docker-compose up` calls are **instant** because the data is persisted via volume mounts.

**How to explore once it's running:**
1. **[http://localhost:8000](http://localhost:8000)** — Interactive Semantic Search UI
2. **[http://localhost:8000/docs](http://localhost:8000/docs)** — Swagger API Explorer
3. **[http://localhost:8000/cache/stats](http://localhost:8000/cache/stats)** — Live cache telemetry

---

## 🧪 Sample Queries to Test the Semantic Cache

To see the power of the custom Semantic Cache (Part 3), try entering these pairs of sentences into the Web UI. You will see how the system recognizes identical semantic intent despite wildly different phrasing.

**Test Case 1: The "Apple / Mac" Relationship**
1. Search: `what is the word apple means` *(Notice it hits the hardware clusters, takes ~300ms)*
2. Search: `Macintosh display problems` *(⚡ **CACHE HIT** - Same semantic neighborhood, takes 0-1ms)*

**Test Case 2: The Space / Orbit Relationship**
1. Search: `NASA space shuttle launch Mars mission` *(Creates a new cache entry)*
2. Search: `space shuttle rocket NASA orbit` *(⚡ **CACHE HIT** - Recognizes identical intent and instantly returns the previous result)*

**Test Case 3: Fuzzy Boundary Discrimination (The "Threshold" Test)**
1. Search: `commercial satellite launch SpaceX` 
*(🚨 **CACHE MISS** - Even though it contains "satellite" and "launch", the custom threshold heuristic of `0.80` correctly identifies that SpaceX commercial telemetry is a different intent than a NASA science mission, bypassing the cache to fetch accurate ChromaDB results).*

---

## 🛠 Manual Installation (Without Docker)

If you prefer to run the code natively:

**1. Install Dependencies**
```bash
conda create -n cv_conda python=3.10 -y
conda activate cv_conda
pip install -r requirements.txt
```

**2. Run the Data Pipeline (Only required once)**
*This cleans the corpus, builds the embeddings, populates ChromaDB, and calculates the Fuzzy C-Means clusters.*
```bash
python scripts/run_pipeline.py
```

**3. Start the Server**
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
Then visit `http://localhost:8000` in your browser.

---

## 🧠 Architecture & Design Decisions

### Part 1: Embeddings & Vector Store
* **Model:** `sentence-transformers/all-MiniLM-L6-v2`. Chosen because it is explicitly fine-tuned for semantic retrieval geometry, outputting rapid 384-dimensional vectors rather than sluggish 1536-dim LLM vectors.
* **Cleaning:** Raw NNTP email headers (`From:`, `Subject:`) and nested quoted replies are explicitly stripped via regex. This guarantees the model clusters on the actual linguistic *content* of the message, rather than cheating by clustering author email domains.
* **Store:** ChromaDB indexing on local disk.

### Part 2: Fuzzy Clustering
* **Approach:** Two-phase Fuzzy C-Means. (1) Fit the centers on a 5k representative sub-sample to save hours of compute, (2) use `cmeans_predict` to broadcast the soft boundaries to the remaining 15k documents.
* **Results:** The notebook `notebooks/heuristic_analysis.ipynb` proves that boundary documents (e.g., Gun Control texts) possess simultaneous fuzzy membership in both the *Politics* and *Religion* clusters, saving the nuance that hard K-Means obliterates.

### Part 3: The Semantic Cache (From Scratch)
* **Data Structure:** Two dicts `_entries` and `_cluster_index`. No Redis used.
* **Cluster-Aware Lookup:** Queries are assigned a dominant cluster upon arrival. The cache performs Cosine Similarity lookups *only* against prior queries in the same semantic bucket `O(N/k)`.
* **The Threshold Heuristic (0.80):** We explicitly swept this parameter in `scripts/analyze_thresholds.py`. At `0.70`, the system suffers *False Sharing* (confusing unrelated topics). At `0.90`, it suffers *Missed Opportunity* (requiring exact wording). The `0.80` sweet spot perfectly hugs the intent boundary.

### Part 4: FastAPI Service
* **State Management:** The model, Vector DB, and Semantic Cache are instantiated exactly once inside a FastAPI `@asynccontextmanager lifespan` and attached to `app.state`. This prevents race conditions and eliminates repeated multi-gigabyte loading times.
