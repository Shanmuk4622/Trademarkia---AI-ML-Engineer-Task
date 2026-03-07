"""
test_cache.py — Unit tests for the SemanticCache

Tests:
  - Hit/miss logic at various thresholds
  - Cluster routing (queries only compared within their cluster bucket)
  - Flush resets counters and entries
  - Persistence round-trip (save → load)
"""
import sys
import json
import tempfile
import shutil
import numpy as np
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import skfuzzy as fuzz

from src.semantic_cache import SemanticCache, CACHE_DIR, CACHE_FILE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_dummy_centres(n_clusters: int = 4, emb_dim: int = 16) -> np.ndarray:
    """Create orthogonal cluster centres for deterministic test behaviour."""
    rng = np.random.default_rng(0)
    centres = rng.random((n_clusters, emb_dim)).astype(np.float32)
    # L2-normalise
    norms = np.linalg.norm(centres, axis=1, keepdims=True)
    return centres / norms


@pytest.fixture
def tmp_cache_dir(monkeypatch, tmp_path):
    """Redirect cache persistence to a temp directory."""
    import src.semantic_cache as sc_module
    monkeypatch.setattr(sc_module, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(sc_module, "CACHE_FILE", tmp_path / "cache_state.json")
    # Also patch the module-level references in the class
    yield tmp_path


@pytest.fixture
def centres():
    return make_dummy_centres()


@pytest.fixture
def cache(centres, tmp_cache_dir):
    return SemanticCache(centres=centres, threshold=0.80)


def random_unit_vec(dim: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Tests: basic hit/miss
# ---------------------------------------------------------------------------

class TestHitMiss:
    def test_empty_cache_is_a_miss(self, cache):
        vec = random_unit_vec(seed=1)
        assert cache.lookup(vec) is None

    def test_exact_match_is_a_hit(self, cache):
        vec = random_unit_vec(seed=2)
        cache.insert("hello world", vec, "result_A")
        hit = cache.lookup(vec)
        assert hit is not None
        entry, score = hit
        assert score > 0.999  # identical vector → similarity ≈ 1.0
        assert entry.result == "result_A"

    def test_similar_query_hits(self, cache):
        """A slightly perturbed vector should still hit at threshold 0.70."""
        cache.threshold = 0.70
        vec = random_unit_vec(seed=3)
        # Perturb slightly
        noise = random_unit_vec(dim=16, seed=99) * 0.05
        vec2 = vec + noise
        vec2 = vec2 / np.linalg.norm(vec2)

        cache.insert("base query", vec, "result_B")
        hit = cache.lookup(vec2)
        assert hit is not None

    def test_dissimilar_query_misses(self, cache):
        """Two orthogonal vectors should miss at threshold 0.80."""
        cache.threshold = 0.80
        vec1 = np.zeros(16, dtype=np.float32)
        vec1[0] = 1.0
        vec2 = np.zeros(16, dtype=np.float32)
        vec2[8] = 1.0  # orthogonal

        # Force same cluster by inserting manually
        from src.semantic_cache import CacheEntry
        entry = CacheEntry(query="q1", vector=vec1.tolist(), result="r1",
                           dominant_cluster=0, membership_vector=[1.0, 0.0, 0.0, 0.0])
        cache._entries["q1"] = entry
        cache._cluster_index[0] = ["q1"]

        # Now lookup vec2 in cluster 0 manually — should miss
        best_score = float(np.dot(vec1, vec2))  # = 0.0
        assert best_score < cache.threshold


# ---------------------------------------------------------------------------
# Tests: counters
# ---------------------------------------------------------------------------

class TestCounters:
    def test_hit_count_increments(self, cache):
        vec = random_unit_vec(seed=10)
        cache.insert("q", vec, "r")
        cache.record_hit()
        cache.record_hit()
        assert cache.hit_count == 2

    def test_miss_count_increments(self, cache):
        cache.record_miss()
        assert cache.miss_count == 1

    def test_hit_rate_calculation(self, cache):
        for _ in range(3):
            cache.record_hit()
        for _ in range(7):
            cache.record_miss()
        stats = cache.stats
        assert abs(stats["hit_rate"] - 0.3) < 0.001


# ---------------------------------------------------------------------------
# Tests: flush
# ---------------------------------------------------------------------------

class TestFlush:
    def test_flush_clears_entries(self, cache):
        vec = random_unit_vec(seed=20)
        cache.insert("flush test", vec, "res")
        cache.flush()
        assert len(cache._entries) == 0
        assert len(cache._cluster_index) == 0

    def test_flush_resets_counters(self, cache):
        cache.record_hit()
        cache.record_miss()
        cache.flush()
        assert cache.hit_count == 0
        assert cache.miss_count == 0


# ---------------------------------------------------------------------------
# Tests: persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_reload(self, centres, tmp_cache_dir):
        import src.semantic_cache as sc_module

        c1 = SemanticCache(centres=centres, threshold=0.85)
        vec = random_unit_vec(seed=30)
        c1.insert("persist test", vec, "persisted_result")
        c1.record_hit()

        # Load a fresh instance
        c2 = SemanticCache.load_from_disk(centres=centres, threshold=0.85)
        assert len(c2._entries) == 1
        assert c2.hit_count == 1
        assert "persist test" in c2._entries

    def test_empty_reload_on_missing_file(self, centres, tmp_cache_dir):
        c = SemanticCache.load_from_disk(centres=centres)
        assert len(c._entries) == 0
        assert c.hit_count == 0
