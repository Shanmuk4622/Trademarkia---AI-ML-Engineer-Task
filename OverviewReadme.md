# Codebase Architecture Overview

This document provides a deep-dive explanation of how the 20 Newsgroups Semantic Search Engine actually works under the hood. It serves as a technical companion to the main `README.md`.

## Directory Structure

```text
Trademarkia---AI-ML-Engineer-Task/
├── src/
│   ├── data_pipeline.py    (Part 1)
│   ├── clustering.py       (Part 2)
│   ├── semantic_cache.py   (Part 3)
│   ├── api.py              (Part 4)
│   └── static/index.html   (Web UI)
├── scripts/
│   ├── run_pipeline.py
│   ├── analyze_thresholds.py
│   └── setup_venv.bat
├── notebooks/
│   └── heuristic_analysis.ipynb
├── Dockerfile & docker-compose.yml
└── persistent_data/ (auto-generated)
    ├── embeddings/  (ChromaDB)
    ├── clusters/    (FCM Matrices)
    └── cache/       (JSON State)
```

---

## Part 1: Data Pipeline (`src/data_pipeline.py`)

**Goal:** Clean the noisy 20 Newsgroups corpus, embed it, and store it for vector search.

**How it works:**
1. **Loading:** `load_corpus_from_tarball` extracts `20_newsgroups.tar.gz` directly into memory.
2. **Cleaning Heuristics:** The `clean_article` function aggressively drops email headers (`From:`, `Subject:`) and nested quoted replies. *Why?* Because raw headers leak the ground-truth category or group authors together trivially. We only want to search over true semantic textual content. Very short documents (<50 chars) are also dropped to prevent noise.
3. **Embedding:** `SentenceTransformer('all-MiniLM-L6-v2')` converts the cleaned strings into 384-dimensional dense vectors. *Why this model?* It is explicitly fine-tuned on 1B+ pairs for Semantic Textual Similarity (STS), unlike raw generative LLMs, and its small vector size maximizes ChromaDB ANN lookup speed.
4. **Storage:** The vectors and textual metadata are inserted into `ChromaDB` backed by the local `embeddings/` directory.

---

## Part 2: Fuzzy Clustering (`src/clustering.py`)

**Goal:** Discover the true underlying semantic structure of the dataset without assuming every document strictly belongs to only 1 of the 20 ground-truth labels. 

**How it works:**
1. **K-Selection:** `find_optimal_k_and_centres` runs Silhouette analysis. We found that `k=13` semantic clusters better represent the corpus than the 20 human labels.
2. **Two-Phase Architecture:** Running Fuzzy C-Means (FCM) on 15,000+ texts is incredibly slow. We train the fuzzy cluster centers on a random `5,000` document sample, and then use `skfuzzy.cmeans_predict` to incredibly rapidly broadcast the fuzzy boundaries to the remaining 10,000+ documents.
3. **The Membership Matrix:** The output is a matrix `U` saved to `clusters/assignments.npz`. `U[c, i]` holds the probability that document `i` belongs to cluster `c`.
4. **Boundary Analysis:** The `compute_cluster_entropy` function looks for vectors with maximum uncertainty. As shown in `metadata.json`, articles discussing Gun Control sit safely on the boundary, reporting near-equal 7.6% memberships in both the politics and religion clusters.

---

## Part 3: The Semantic Cache (`src/semantic_cache.py`)

**Goal:** Build a caching layer from scratch that recognizes semantic synonyms, preventing expensive vector-database lookups when a user rephrases a previous query.

**How it works:**
1. **The Architecture:** The cache explicitly rejects Redis. It uses a pure Python singleton holding two dicts: `_entries` (for O(1) string lookups) and `_cluster_index`.
2. **The Cluster-Bucket Lookup:** When `lookup(vector)` runs, it *first* asks the FCM model from Part 2 which cluster this vector belongs to. It then *only* does Cosine Similarity math against previous queries that inhabit that exact same cluster. This drops the lookup time from `O(N)` queries to `O(N/k)`.
3. **The Tuneable Threshold (`0.80`):** The `_cosine` function returns `1.0` for exact matches. We swept the threshold via `scripts/analyze_thresholds.py`. We found that setting it to `0.80` creates a Goldilocks zone: it successfully matches `"Macintosh display problems"` with `"Apple monitor issues"`, but correctly separates `"SpaceX satellite"` from `"NASA Mars launch"`.
4. **Persistence:** `_save()` dumps the dicts atomically to `cache/cache_state.json`.

---

## Part 4: FastAPI Service (`src/api.py`)

**Goal:** Expose the system securely to the web.

**How it works:**
1. **The Lifespan Manager:** The `@asynccontextmanager lifespan` runs before the server accepts traffic. It loads the `all-MiniLM` model, the ChromaDB index, and the Cache JSON exactly once, pinning them to `app.state`. This prevents race conditions and makes the API requests `O(1)` in initialization.
2. **The Endpoints:**
    - `GET /`: Serves the Interactive UI from `src/static/index.html`.
    - `POST /query`: Handles the workflow: `Embed Query -> Cache Lookup? -> If Miss, ChromaDB ANN Search -> Persist to Cache -> Return JSON Array`.
    - `GET /cache/stats`: Returns telemetry (Hits, Misses, Threshold limits).
3. **The User Interface:** `src/static/index.html` leverages TailWindCSS to create a beautiful Glassmorphism frontend that parses the JSON output into visually distinct cards and exposes the cache hit/miss telemetry live.
