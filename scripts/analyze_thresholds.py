"""
analyze_thresholds.py

Demonstrates the effect of the primary tunable heuristic in Part 3:
the Semantic Cache Similarity Threshold.

We will simulate a realistic query workload with rephrasings and orthogonal topics:
1. Seed the cache with a base query.
2. Sweep the threshold from 0.70 to 0.95.
3. Show which rephrased queries result in a HIT (and thus inherit the first query's result)
   and which result in a MISS (and trigger a fresh ChromaDB search).
"""
import sys
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_pipeline import get_embedding_model
from src.clustering import load_cluster_data
from src.semantic_cache import SemanticCache

def cosine_sim(a, b):
    return float(np.dot(a, b))

def main():
    print("Loading embedding model and cluster centres...")
    model = get_embedding_model()
    U, centres, doc_ids, categories = load_cluster_data()
    
    # 1. Base query we will cache
    base_query = "NASA space shuttle launch Mars mission"
    base_vec = model.encode([base_query], normalize_embeddings=True)[0]
    
    # 2. Test queries (rephrased variants and unrelated)
    test_queries = [
        # Near exact
        "space shuttle rocket NASA orbit",
        "NASA mars mission launch details",
        # Semantically related but different intent
        "commercial satellite launch SpaceX",
        "budget cuts to planetary science missions",
        # Completely unrelated
        "Ford Mustang engine repair",
    ]
    
    # Pre-compute vectors
    test_vecs = [model.encode([q], normalize_embeddings=True)[0] for q in test_queries]
    
    # Show baseline similarities
    print(f"\nBase Query: '{base_query}'")
    print("-" * 50)
    print("Baseline Cosine Similarities against Base Query:")
    for q, vec in zip(test_queries, test_vecs):
        sim = cosine_sim(base_vec, vec)
        print(f"  {sim:.3f} | {q}")
    print("-" * 50)
    
    # 3. Sweep the threshold
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    print("\nSimulating Cache Lookup across Thresholds:")
    print("  Threshold dictates the boundary of 'semantic equivalence'.")
    print("  Too low = False Sharing (unrelated query gets wrong cached result)")
    print("  Too high = Missed Opportunity (rephrased query computes from scratch)")
    print("=" * 80)
    
    for t in thresholds:
        cache = SemanticCache(centres=centres, threshold=t)
        # Manually assign cluster instead of search_corpus so we don't need ChromaDB for this test
        dummy_result = [{"id": 0, "category": "test", "similarity": 1.0, "text": "dummy"}]
        cache.insert(base_query, base_vec, dummy_result)
        
        print(f"\nTHRESHOLD: {t:.2f}")
        for q, vec in zip(test_queries, test_vecs):
            hit = cache.lookup(vec)
            if hit:
                print(f"  [HIT]  {q}")
            else:
                print(f"  [MISS] {q}")

if __name__ == "__main__":
    main()
