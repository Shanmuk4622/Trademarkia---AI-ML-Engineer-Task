"""
test_api.py — Integration tests for the FastAPI endpoints

Uses FastAPI's TestClient to hit the real endpoints with mock state objects.
The heavy models (SentenceTransformer, ChromaDB) are replaced with lightweight
stubs so tests run in milliseconds without GPU or disk I/O.
"""
import sys
import json
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.semantic_cache import SemanticCache


EMB_DIM = 16
N_CLUSTERS = 4


def make_centres() -> np.ndarray:
    rng = np.random.default_rng(42)
    c = rng.random((N_CLUSTERS, EMB_DIM)).astype(np.float32)
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def make_mock_state(tmp_path):
    """Build a minimal app.state-like namespace with all required attributes."""
    centres = make_centres()
    cache = SemanticCache(centres=centres, threshold=0.80)
    # Monkeypatch cache dir to tmp
    import src.semantic_cache as sc
    sc.CACHE_DIR = tmp_path
    sc.CACHE_FILE = tmp_path / "cache_state.json"

    # Mock embedding model: always returns the same deterministic vector
    mock_model = MagicMock()
    fixed_vec = np.ones(EMB_DIM, dtype=np.float32) / np.sqrt(EMB_DIM)
    mock_model.encode.return_value = np.array([fixed_vec])

    # Mock ChromaDB collection
    mock_collection = MagicMock()
    mock_collection.count.return_value = 100
    mock_collection.query.return_value = {
        "documents": [["Sample document about space exploration."]],
        "metadatas": [[{"category": "sci.space"}]],
        "distances": [[0.05]],
    }

    # Mock cluster data — random U matrix
    rng = np.random.default_rng(1)
    U = rng.dirichlet(np.ones(N_CLUSTERS), size=100).T.astype(np.float32)

    state = MagicMock()
    state.model = mock_model
    state.collection = mock_collection
    state.cache = cache
    state.cluster_centres = centres
    state.cluster_U = U
    state.cluster_doc_ids = [f"cat/doc{i}" for i in range(100)]
    state.cluster_categories = ["sci.space"] * 100
    return state


@pytest.fixture
def client(tmp_path):
    """TestClient with mocked app.state (no real model loading)."""
    state = make_mock_state(tmp_path)
    app.state.model = state.model
    app.state.collection = state.collection
    app.state.cache = state.cache
    app.state.cluster_centres = state.cluster_centres
    app.state.cluster_U = state.cluster_U
    app.state.cluster_doc_ids = state.cluster_doc_ids
    app.state.cluster_categories = state.cluster_categories

    # Skip lifespan (we manually set state above)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests: POST /query
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    def test_first_query_is_cache_miss(self, client):
        resp = client.post("/query", json={"query": "nasa space shuttle"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["cache_hit"] is False
        assert data["matched_query"] is None
        assert data["similarity_score"] is None
        assert "result" in data
        assert isinstance(data["dominant_cluster"], int)

    def test_second_identical_query_is_cache_hit(self, client):
        payload = {"query": "nasa space shuttle"}
        client.post("/query", json=payload)
        resp2 = client.post("/query", json=payload)
        assert resp2.status_code == 200
        data = resp2.json()
        assert data["cache_hit"] is True
        assert data["matched_query"] == payload["query"]
        assert data["similarity_score"] >= 0.99

    def test_query_response_fields_present(self, client):
        resp = client.post("/query", json={"query": "hockey playoffs"})
        assert resp.status_code == 200
        data = resp.json()
        required_keys = {"query", "cache_hit", "matched_query", "similarity_score", "result", "dominant_cluster"}
        assert required_keys.issubset(data.keys())


# ---------------------------------------------------------------------------
# Tests: GET /cache/stats
# ---------------------------------------------------------------------------

class TestCacheStats:
    def test_initial_stats(self, client):
        resp = client.get("/cache/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_entries"] == 0
        assert data["hit_count"] == 0
        assert data["miss_count"] == 0
        assert data["hit_rate"] == 0.0

    def test_stats_after_miss_and_hit(self, client):
        client.post("/query", json={"query": "test query stats"})  # miss
        client.post("/query", json={"query": "test query stats"})  # hit
        stats = client.get("/cache/stats").json()
        assert stats["miss_count"] == 1
        assert stats["hit_count"] == 1
        assert abs(stats["hit_rate"] - 0.5) < 0.01


# ---------------------------------------------------------------------------
# Tests: DELETE /cache
# ---------------------------------------------------------------------------

class TestFlushCache:
    def test_flush_clears_cache(self, client):
        client.post("/query", json={"query": "to be deleted"})
        del_resp = client.delete("/cache")
        assert del_resp.status_code == 200
        stats = client.get("/cache/stats").json()
        assert stats["total_entries"] == 0
        assert stats["hit_count"] == 0
        assert stats["miss_count"] == 0
