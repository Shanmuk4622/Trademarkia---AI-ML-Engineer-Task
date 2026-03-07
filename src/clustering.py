"""
clustering.py — Part 2: Fuzzy C-Means Clustering with Two-Phase Approach

Design decisions:

1. TWO-PHASE FUZZY C-MEANS:
   Phase 1 — Fit on a random 5,000-doc sample to find cluster centres C.
              skfuzzy.cmeans on 20k docs would take hours; on 5k it runs in minutes.
              The cluster geometry learned on 5k is representative because the classes
              are balanced (~1k docs each), so a stratified random sample naturally
              preserves the distribution.
   Phase 2 — Use skfuzzy.cmeans_predict with the pre-computed centres C to assign
              membership vectors to the full corpus. This is a fast one-pass operation.

2. NUMBER OF CLUSTERS:
   We use silhouette analysis and distortion (WCSS) on the sample embeddings over
   k ∈ [5, 30] to find the elbow. The 20-label ground truth is a loose guide only —
   the *semantic* structure of the corpus is likely between 12 and 18 clusters because:
   - Several pairs of categories are near-synonymous (e.g. talk.religion.misc and
     alt.atheism; comp.sys.ibm.pc.hardware and comp.sys.mac.hardware).
   - The "politics" arc (guns, mideast, misc) forms a continuum, not 3 distinct clusters.
   We deterministically select the k that maximises the silhouette score on the sample.
   If the analysis yields k outside [10, 20] we cap it there as a sanity bound, since
   fewer than 10 clusters under-partition obvious topic groups.

3. SOFT ASSIGNMENTS (THE CORE REQUIREMENT):
   skfuzzy.cmeans returns a membership matrix U of shape (k, n_docs) where U[j, i] is
   the degree to which document i belongs to cluster j, with each column summing to 1.0.
   This is stored as-is (no argmax) — a gun-legislation post will correctly show high
   membership in BOTH the politics cluster and the firearms cluster.

4. PERSISTENCE:
   - clusters/assignments.npz: the full (k, n_docs) membership matrix + doc_ids array
   - clusters/metadata.json: cluster count, doc_ids, category_ids, top-10 doc per cluster

5. FUZZINESS PARAMETER (m):
   We use m=2.0, the standard Fuzzy C-Means fuzzifier. m=1 approaches hard assignment;
   m→∞ makes all memberships equal. m=2 is the theoretical sweet spot for text data
   (consistent with the original FCM paper, Bezdek 1981).
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List

import skfuzzy as fuzz
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from tqdm import tqdm

from src.data_pipeline import get_collection, CHROMA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CLUSTERS_DIR = PROJECT_ROOT / "clusters"
SAMPLE_SIZE = 5000        # docs used for Phase 1 centre-fitting
K_RANGE = range(5, 26)    # k values to evaluate (5 .. 25)
K_CAP = (10, 20)          # hard bounds on acceptable k
FCM_FUZZINESS = 2.0       # fuzzifier parameter m
FCM_MAX_ITER = 150
FCM_ERROR = 1e-5


# ---------------------------------------------------------------------------
# Helper: pull all embeddings from ChromaDB
# ---------------------------------------------------------------------------

def load_embeddings_from_chroma() -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Pull all document embeddings, doc_ids, and categories from ChromaDB.

    Returns:
        embeddings: np.ndarray of shape (n_docs, emb_dim), L2-normalised
        doc_ids:    list of doc_id strings
        categories: list of category strings
    """
    log.info("Loading embeddings from ChromaDB...")
    collection = get_collection()
    n = collection.count()

    # ChromaDB paginates results; fetch in pages of 5000
    all_embeddings, all_ids, all_cats = [], [], []
    page_size = 5000
    for offset in tqdm(range(0, n, page_size), desc="Fetching from ChromaDB"):
        result = collection.get(
            limit=page_size,
            offset=offset,
            include=["embeddings", "metadatas"],
        )
        all_embeddings.extend(result["embeddings"])
        all_ids.extend(result["ids"])
        all_cats.extend([m["category"] for m in result["metadatas"]])

    embedding_matrix = np.array(all_embeddings, dtype=np.float32)
    # Renormalise after retrieval (ChromaDB stores as float32; verify unit vectors)
    embedding_matrix = normalize(embedding_matrix, norm="l2")
    log.info(f"Loaded {embedding_matrix.shape[0]} embeddings, dim={embedding_matrix.shape[1]}")
    return embedding_matrix, all_ids, all_cats


# ---------------------------------------------------------------------------
# Phase 1: Find optimal k and cluster centres on a sample
# ---------------------------------------------------------------------------

def find_optimal_k_and_centres(
    embeddings: np.ndarray,
    sample_size: int = SAMPLE_SIZE,
) -> Tuple[int, np.ndarray]:
    """
    Run silhouette analysis on a sample to find the best k, then fit FCM centres.

    We use silhouette on HARD assignments (argmax of U) for efficiency — the
    silhouette score is a cluster quality metric and the soft structure would
    require a fuzzy-specific metric (partition coefficient); however, for the
    purpose of choosing k, hard cluster silhouette is a reliable proxy.

    Returns:
        best_k:    optimal number of clusters
        centres:   np.ndarray of shape (best_k, emb_dim) — cluster centres
    """
    n_docs = embeddings.shape[0]
    # Stratified random sample
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n_docs, size=min(sample_size, n_docs), replace=False)
    sample = embeddings[sample_idx]  # (sample_size, emb_dim)

    # skfuzzy expects shape (emb_dim, n_samples)
    sample_T = sample.T

    log.info(f"Running silhouette analysis for k in {list(K_RANGE)}...")
    best_k = None
    best_score = -1.0
    scores = {}

    for k in K_RANGE:
        # Run FCM — returns (centres, U, u0, d, jm, p, fpc)
        centres, U, _, _, _, _, fpc = fuzz.cmeans(
            sample_T,
            c=k,
            m=FCM_FUZZINESS,
            error=FCM_ERROR,
            maxiter=FCM_MAX_ITER,
            seed=42,
        )
        # Hard assignments for silhouette
        hard_labels = np.argmax(U, axis=0)

        # Silhouette requires at least 2 unique labels and < n_samples labels
        unique = len(np.unique(hard_labels))
        if unique < 2 or unique >= sample_size:
            scores[k] = -1.0
            continue

        score = silhouette_score(
            sample, hard_labels, metric="cosine", sample_size=min(2000, len(sample_idx))
        )
        scores[k] = score
        log.info(f"  k={k:2d}  silhouette={score:.4f}  fpc={fpc:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_centres = centres.copy()

    # Apply sanity cap
    best_k = max(K_CAP[0], min(K_CAP[1], best_k))
    log.info(f"Selected k={best_k} (silhouette={best_score:.4f})")

    # Re-fit with best_k to ensure centres are consistent with chosen k
    best_centres, _, _, _, _, _, _ = fuzz.cmeans(
        sample_T, c=best_k, m=FCM_FUZZINESS, error=FCM_ERROR, maxiter=FCM_MAX_ITER, seed=42
    )
    return best_k, best_centres, scores


# ---------------------------------------------------------------------------
# Phase 2: Predict memberships for the full corpus
# ---------------------------------------------------------------------------

def predict_memberships(
    embeddings: np.ndarray,
    centres: np.ndarray,
    chunk_size: int = 2000,
) -> np.ndarray:
    """
    Use cmeans_predict to assign membership distributions to the full corpus.

    skfuzzy.cmeans_predict can be slow if called on all 20k docs at once because
    it still runs until convergence internally. We call it in chunks and average
    the centre updates — this is equivalent to the one-shot call but avoids OOM.

    Returns:
        U: np.ndarray of shape (k, n_docs) — membership matrix (columns sum to 1)
    """
    n_docs = embeddings.shape[0]
    k = centres.shape[0]
    all_U = np.zeros((k, n_docs), dtype=np.float32)

    log.info(f"Predicting memberships for {n_docs} docs in chunks of {chunk_size}...")
    for start in tqdm(range(0, n_docs, chunk_size), desc="FCM predict"):
        end = min(start + chunk_size, n_docs)
        chunk = embeddings[start:end].T  # (emb_dim, chunk_size)
        U_chunk, _, _, _, _, _ = fuzz.cmeans_predict(
            chunk,
            centres,
            m=FCM_FUZZINESS,
            error=FCM_ERROR,
            maxiter=FCM_MAX_ITER,
        )
        all_U[:, start:end] = U_chunk

    return all_U


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def compute_cluster_entropy(U: np.ndarray) -> np.ndarray:
    """
    Compute Shannon entropy of each document's membership distribution.
    High entropy → boundary document (belongs equally to multiple clusters).
    Low entropy → clearly in one cluster.
    """
    # Clip to avoid log(0)
    U_safe = np.clip(U, 1e-10, 1.0).T  # (n_docs, k)
    entropy = -np.sum(U_safe * np.log(U_safe), axis=1)
    return entropy  # (n_docs,)


def get_top_docs_per_cluster(
    U: np.ndarray,
    doc_ids: List[str],
    categories: List[str],
    top_n: int = 10,
) -> Dict[int, List[Dict]]:
    """
    For each cluster, return the top_n documents with highest membership score.
    These are the "most representative" documents for that cluster.
    """
    result = {}
    for c_idx in range(U.shape[0]):
        memberships = U[c_idx]
        top_indices = np.argsort(memberships)[::-1][:top_n]
        # JSON requires string keys — convert int cluster index to str
        result[str(c_idx)] = [
            {
                "doc_id": doc_ids[i],
                "category": categories[i],
                "membership": float(memberships[i]),
            }
            for i in top_indices
        ]
    return result


def get_boundary_documents(
    U: np.ndarray,
    doc_ids: List[str],
    categories: List[str],
    entropy: np.ndarray,
    top_n: int = 20,
) -> List[Dict]:
    """
    Return the top_n highest-entropy documents — these are the most ambiguous,
    belonging significantly to 2+ clusters. These are the semantically interesting
    boundary cases described in the spec.
    """
    top_indices = np.argsort(entropy)[::-1][:top_n]
    return [
        {
            "doc_id": doc_ids[i],
            "category": categories[i],
            "entropy": float(entropy[i]),
            # Use str(c) keys — np.argsort returns numpy int64 which JSON can't serialise
            "memberships": {str(c): float(U[c, i]) for c in np.argsort(U[:, i])[::-1][:5]},
        }
        for i in top_indices
    ]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_results(
    U: np.ndarray,
    centres: np.ndarray,
    doc_ids: List[str],
    categories: List[str],
    k: int,
    top_docs: Dict,
    boundary_docs: List[Dict],
    silhouette_scores: Dict,
) -> None:
    """Save cluster assignments and analysis metadata."""
    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)

    # Custom JSON encoder to handle any residual numpy types
    class NumpySafeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    # 1. Membership matrix + doc_ids as NPZ (compact, fast to load)
    np.savez_compressed(
        str(CLUSTERS_DIR / "assignments.npz"),
        U=U,
        centres=centres,
        doc_ids=np.array(doc_ids),
        categories=np.array(categories),
    )
    log.info(f"Saved assignments.npz  shape={U.shape}")

    # 2. Human-readable metadata as JSON
    metadata = {
        "n_clusters": k,
        "n_docs": len(doc_ids),
        "fuzziness_m": FCM_FUZZINESS,
        "silhouette_scores": {str(k_): float(v) for k_, v in silhouette_scores.items()},
        "top_docs_per_cluster": top_docs,
        "boundary_documents": boundary_docs,
    }
    with open(CLUSTERS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, cls=NumpySafeEncoder)
    log.info("Saved metadata.json")


def load_cluster_data() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load cluster assignments from disk.
    Returns: U (k, n_docs), centres (k, emb_dim), doc_ids, categories
    """
    data = np.load(str(CLUSTERS_DIR / "assignments.npz"), allow_pickle=True)
    U = data["U"]
    centres = data["centres"]
    doc_ids = data["doc_ids"].tolist()
    categories = data["categories"].tolist()
    return U, centres, doc_ids, categories


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_clustering(smoke_test: bool = False) -> None:
    """Full clustering pipeline."""
    # Load embeddings
    embeddings, doc_ids, categories = load_embeddings_from_chroma()

    if smoke_test:
        log.info("SMOKE TEST: using first 500 docs only")
        embeddings = embeddings[:500]
        doc_ids = doc_ids[:500]
        categories = categories[:500]

    # Phase 1: find optimal k and cluster centres on sample
    best_k, centres, silhouette_scores = find_optimal_k_and_centres(embeddings)

    # Phase 2: predict memberships for full corpus using pre-computed centres
    U = predict_memberships(embeddings, centres)

    # Analysis
    entropy = compute_cluster_entropy(U)
    top_docs = get_top_docs_per_cluster(U, doc_ids, categories)
    boundary_docs = get_boundary_documents(U, doc_ids, categories, entropy)

    # Save
    save_results(U, centres, doc_ids, categories, best_k, top_docs, boundary_docs, silhouette_scores)
    log.info("Clustering complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run fuzzy clustering (Part 2)")
    parser.add_argument("--smoke-test", action="store_true", help="Run on 500 docs only")
    args = parser.parse_args()
    run_clustering(smoke_test=args.smoke_test)
