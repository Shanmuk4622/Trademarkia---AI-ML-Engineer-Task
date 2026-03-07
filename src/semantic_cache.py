"""
semantic_cache.py — Part 3: Semantic Cache (Built From Scratch)

Design decisions:

1. DATA STRUCTURE:
   Two co-ordinated dictionaries:
   - _entries: maps query string → CacheEntry (with vector, result, metadata)
   - _cluster_index: maps dominant_cluster_id → list of query strings in that bucket

   This gives O(bucket_size) lookup instead of O(N) over the full cache, where
   bucket_size ≈ N/k. With k=15 clusters and N=1000 entries, that's a ~15× speedup.

2. CLUSTER-AWARE LOOKUP:
   Every query is assigned to a dominant cluster (argmax of its fuzzy membership vector).
   On lookup, we only compare the incoming query against entries that share the same
   dominant cluster. This is correct because:
   - Documents/queries in the same semantic region will have the same dominant cluster
   - A query about "space shuttles" won't waste comparisons against cached "hockey" queries
   The cluster membership is computed using the pre-fitted FCM centres from Part 2.

3. SIMILARITY METRIC: Cosine similarity via dot product.
   Because our embeddings are L2-normalised (unit vectors), cosine(a,b) = dot(a,b).
   This avoids an expensive norm computation on every cache lookup.

4. THRESHOLD (the tunable parameter at the heart of the system):
   - threshold=0.70: many false positives (semantically different queries map to same cached result)
   - threshold=0.80: good balance — semantically equivalent rephrasings match, dissimilar queries miss
   - threshold=0.90: conservative — only near-verbatim matches hit; good for high-precision use cases
   - threshold=0.95: essentially exact-match only; eliminates the semantic benefit of the cache
   Default is 0.80. The API exposes it so callers can tune per-request (see api.py).

5. PERSISTENCE:
   Pure Python json only. No SQLite, no Redis, no external libraries.
   Layout: cache/cache_state.json
   {
     "threshold": 0.80,
     "hit_count": 17,
     "miss_count": 25,
     "entries": {
       "<query_string>": {
         "result": "...",
         "dominant_cluster": 3,
         "vector": [0.1, 0.2, ...],
         "matched_query": null,
         "similarity_score": null
       },
       ...
     }
   }

6. THREAD SAFETY:
   Python's GIL protects individual dict reads/writes. We additionally use a threading.Lock
   around the (check → miss path → insert) sequence to prevent two concurrent misses
   computing and inserting the same query. This is the correct minimal locking —
   read-only cache hits do NOT acquire the lock.
"""

import json
import logging
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict

import skfuzzy as fuzz

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_FILE = CACHE_DIR / "cache_state.json"

DEFAULT_THRESHOLD = 0.80  # see decision note #4 above


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    query: str
    vector: List[float]           # normalised embedding
    result: str                   # the computed answer/result string
    dominant_cluster: int         # argmax of fuzzy membership
    membership_vector: List[float] # full fuzzy membership distribution


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------

class SemanticCache:
    """
    Cluster-aware semantic cache with cosine-similarity lookup.

    All state is stored in two plain Python dicts; persistence is via JSON.
    No external libraries beyond numpy are used.
    """

    def __init__(
        self,
        centres: np.ndarray,
        threshold: float = DEFAULT_THRESHOLD,
        fuzziness: float = 2.0,
    ):
        """
        Args:
            centres:   FCM cluster centres, shape (k, emb_dim). Required for
                       assigning incoming queries to clusters.
            threshold: Cosine similarity threshold for a cache hit.
            fuzziness: FCM fuzzifier parameter m (must match the one used in training).
        """
        self._entries: Dict[str, CacheEntry] = {}
        # cluster_id → list of query strings whose dominant cluster is cluster_id
        self._cluster_index: Dict[int, List[str]] = {}
        self._centres = centres          # (k, emb_dim)
        self.threshold = threshold
        self._fuzziness = fuzziness
        self.hit_count: int = 0
        self.miss_count: int = 0
        self._lock = threading.Lock()    # guards miss-path (check → insert)

    # ------------------------------------------------------------------
    # Cluster assignment
    # ------------------------------------------------------------------

    def _assign_cluster(self, vector: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Use cmeans_predict to get the fuzzy membership of a single query vector
        against the stored cluster centres, then return (dominant_cluster, membership).

        We pass the single vector as shape (emb_dim, 1) — skfuzzy expects (dim, n_samples).
        """
        vec_col = vector.reshape(-1, 1)  # (emb_dim, 1)
        U, _, _, _, _, _ = fuzz.cmeans_predict(
            vec_col,
            self._centres,
            m=self._fuzziness,
            error=1e-5,
            maxiter=100,
        )
        membership = U[:, 0]           # shape (k,)
        dominant = int(np.argmax(membership))
        return dominant, membership

    # ------------------------------------------------------------------
    # Cosine similarity (dot product on unit vectors)
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two L2-normalised vectors.
        Since both vectors are unit vectors, this is just a dot product.
        Vectors are stored normalised, so no re-normalisation needed here.
        """
        return float(np.dot(a, b))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(
        self, vector: np.ndarray
    ) -> Optional[Tuple[CacheEntry, float]]:
        """
        Search the cache for a semantically similar cached query.

        Algorithm:
          1. Assign the incoming vector to a dominant cluster
          2. Retrieve only the entries in that cluster's bucket
          3. Compute cosine similarity against each entry in the bucket
          4. Return the best match if it exceeds self.threshold

        Returns:
            (entry, similarity_score) if cache hit, else None.
        """
        dominant, _ = self._assign_cluster(vector)
        bucket = self._cluster_index.get(dominant, [])

        best_score = -1.0
        best_entry: Optional[CacheEntry] = None

        for query_key in bucket:
            entry = self._entries.get(query_key)
            if entry is None:
                continue
            cached_vec = np.array(entry.vector, dtype=np.float32)
            score = self._cosine(vector, cached_vec)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self.threshold and best_entry is not None:
            return best_entry, best_score
        return None

    def insert(self, query: str, vector: np.ndarray, result: str) -> CacheEntry:
        """
        Store a new query+result in the cache.

        Protected by a lock: two concurrent misses for the same query won't
        race to insert duplicate entries.
        """
        with self._lock:
            # Double-check: another thread may have inserted while we waited
            hit = self.lookup(vector)
            if hit is not None:
                return hit[0]

            dominant, membership = self._assign_cluster(vector)
            entry = CacheEntry(
                query=query,
                vector=vector.tolist(),
                result=result,
                dominant_cluster=dominant,
                membership_vector=membership.tolist(),
            )
            self._entries[query] = entry
            if dominant not in self._cluster_index:
                self._cluster_index[dominant] = []
            self._cluster_index[dominant].append(query)

            self._save()
            return entry

    def record_hit(self) -> None:
        """Increment hit counter (called by API after a successful lookup)."""
        with self._lock:
            self.hit_count += 1
            self._save_stats_only()

    def record_miss(self) -> None:
        """Increment miss counter (called by API on a cache miss)."""
        with self._lock:
            self.miss_count += 1
            self._save_stats_only()

    def flush(self) -> None:
        """Clear all cache entries and reset counters."""
        with self._lock:
            self._entries.clear()
            self._cluster_index.clear()
            self.hit_count = 0
            self.miss_count = 0
            self._save()
        log.info("Cache flushed.")

    @property
    def stats(self) -> Dict[str, Any]:
        total = self.hit_count + self.miss_count
        return {
            "total_entries": len(self._entries),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(self.hit_count / total, 4) if total > 0 else 0.0,
            "threshold": self.threshold,
        }

    # ------------------------------------------------------------------
    # Persistence (pure JSON)
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """
        Persist full cache state to JSON.
        Only called inside the write lock.
        """
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        state = {
            "threshold": self.threshold,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "entries": {
                k: asdict(v) for k, v in self._entries.items()
            },
        }
        tmp = CACHE_FILE.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(state, f)
        # Atomic rename — avoids corrupt state if the process is killed mid-write
        tmp.replace(CACHE_FILE)

    def _save_stats_only(self) -> None:
        """
        Lightweight save: only update counters in the JSON file.
        Reads existing file and updates hit/miss counts to avoid rewriting
        the full entry list on every hit.
        """
        if not CACHE_FILE.exists():
            self._save()
            return
        try:
            with open(CACHE_FILE, "r") as f:
                state = json.load(f)
            state["hit_count"] = self.hit_count
            state["miss_count"] = self.miss_count
            tmp = CACHE_FILE.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(state, f)
            tmp.replace(CACHE_FILE)
        except Exception:
            self._save()  # fallback to full save

    @classmethod
    def load_from_disk(cls, centres: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> "SemanticCache":
        """
        Load cache state from cache/cache_state.json.
        If the file doesn't exist, returns a fresh empty cache.
        """
        instance = cls(centres=centres, threshold=threshold)
        if not CACHE_FILE.exists():
            log.info("No existing cache file found. Starting fresh.")
            return instance

        try:
            with open(CACHE_FILE, "r") as f:
                state = json.load(f)

            instance.hit_count = state.get("hit_count", 0)
            instance.miss_count = state.get("miss_count", 0)
            # Restore threshold from saved state (user may have changed it)
            instance.threshold = state.get("threshold", threshold)

            for query, entry_dict in state.get("entries", {}).items():
                entry = CacheEntry(**entry_dict)
                instance._entries[query] = entry
                dominant = entry.dominant_cluster
                if dominant not in instance._cluster_index:
                    instance._cluster_index[dominant] = []
                instance._cluster_index[dominant].append(query)

            log.info(
                f"Restored cache: {len(instance._entries)} entries, "
                f"{instance.hit_count} hits, {instance.miss_count} misses"
            )
        except Exception as e:
            log.warning(f"Failed to load cache from disk ({e}). Starting fresh.")

        return instance
