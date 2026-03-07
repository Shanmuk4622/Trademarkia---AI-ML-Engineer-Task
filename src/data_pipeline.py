"""
data_pipeline.py — Part 1: Corpus Loading, Cleaning, Embedding & ChromaDB Indexing

Design decisions (justified here as per the spec):

1. DATA SOURCE: We load from the local tarball (twenty+newsgroups/20_newsgroups.tar.gz)
   instead of sklearn's fetch_20newsgroups. This gives us full control over the raw bytes
   and avoids any sklearn preprocessing that would hide our cleaning logic.

2. CLEANING STRATEGY:
   - Strip email headers (From, Subject, Organization, Lines, NNTP-Posting-Host etc.)
     because they leak category identity — a model that fits on headers isn't learning
     semantics, it's memorising metadata.
   - Strip quoted reply blocks (lines starting with ">") — these are noise copied from
     other articles and would cause embedding similarity to reflect quoting chains rather
     than original thought.
   - Discard documents with fewer than 50 whitespace-separated tokens after cleaning.
     Very short posts carry too little signal for a 384-dim embedding to represent
     meaningfully; they also tend to be "me too" or spam posts.
   - Near-duplicate removal: hash the first 200 cleaned characters. Crossposted articles
     (~4% per dataset description) would act as anchor points that artificially bridge
     unrelated clusters.
   - Decode as latin-1 with replacement — the dataset predates Unicode and some bytes
     are raw 8-bit chars; latin-1 is the standard for this corpus.

3. EMBEDDING MODEL: all-MiniLM-L6-v2
   - 384 dimensions vs 768 for large models — half the storage/compute, minimal quality
     loss for retrieval tasks (MTEB rank ~40 vs ~35 for large at 3× the cost).
   - Designed specifically for semantic similarity, not generation.
   - Batch size 64 balances GPU VRAM and CPU throughput; progress tracked with tqdm.

4. VECTOR STORE: ChromaDB (local persistent mode)
   - No external service needed — survives process restarts via on-disk SQLite+parquet.
   - Supports metadata filtering (by category, doc_id) for downstream cluster-aware
     retrieval without a separate index.
   - Collection name: "newsgroups" — one collection, all docs.

5. IDs: we use "{category}/{filename}" as the ChromaDB ID. This is human-readable,
   collision-free (filenames are unique per group), and encodes provenance for analysis.
"""

import os
import re
import tarfile
import hashlib
import logging
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TARBALL = PROJECT_ROOT / "twenty+newsgroups" / "20_newsgroups.tar.gz"
CHROMA_DIR = PROJECT_ROOT / "embeddings"
COLLECTION_NAME = "newsgroups"
EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
MIN_TOKENS = 50  # discard docs shorter than this after cleaning

# Email header fields to strip (case-insensitive)
HEADER_FIELDS = {
    "from", "subject", "organization", "lines", "nntp-posting-host",
    "message-id", "references", "date", "distribution", "reply-to",
    "newsgroups", "x-newsreader", "x-mailer", "path", "summary",
    "keywords", "mime-version", "content-type", "article-i.d.",
    "sender", "approved", "followup-to", "expires", "supersedes",
    "archive-name", "version", "x-posted-to", "return-path",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Extract & clean corpus
# ---------------------------------------------------------------------------

def _strip_email_headers(text: str) -> str:
    """
    Remove the RFC-822 style header block at the top of each article.
    The header block ends at the first blank line.
    """
    lines = text.splitlines()
    in_header = True
    body_lines = []
    for line in lines:
        if in_header:
            # Blank line marks end of header block
            if line.strip() == "":
                in_header = False
            else:
                # Keep if it doesn't look like a header field
                # (header fields start with "Word:" pattern)
                field = line.split(":")[0].strip().lower()
                if field not in HEADER_FIELDS and not re.match(r'^[A-Za-z0-9\-]+\s*:', line):
                    in_header = False  # malformed header — treat rest as body
                    body_lines.append(line)
        else:
            body_lines.append(line)
    return "\n".join(body_lines)


def _strip_quotes_and_sigs(text: str) -> str:
    """
    Remove quoted reply blocks (lines starting with >) and signature blocks
    (everything after a line that is exactly '-- ' or '-----------').
    These are noise: quotes copy other articles verbatim (embedding pollution),
    sigs are boilerplate unrelated to the post's actual topic.
    """
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Stop at signature delimiters
        if stripped in ("--", "----------", "--------------------------------------------------------------------------"):
            break
        # Skip quoted lines
        if stripped.startswith(">"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple blank lines and trim."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def clean_article(raw: str) -> str:
    """Full cleaning pipeline for one article."""
    text = raw
    text = _strip_email_headers(text)
    text = _strip_quotes_and_sigs(text)
    text = _normalize_whitespace(text)
    return text


def load_corpus_from_tarball(tarball_path: Path) -> List[Tuple[str, str, str]]:
    """
    Load and clean all articles from the tarball.

    Returns:
        List of (doc_id, category, cleaned_text) tuples.
        doc_id format: "<category>/<filename>"
    """
    log.info(f"Opening tarball: {tarball_path}")
    docs = []
    seen_hashes: set = set()
    n_too_short = 0
    n_duplicate = 0

    with tarfile.open(tarball_path, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        log.info(f"Total article files in archive: {len(members)}")

        for member in tqdm(members, desc="Loading & cleaning"):
            # Extract category and doc filename from path
            # Path format: "20_newsgroups/<category>/<filename>"
            parts = member.name.split("/")
            if len(parts) < 3:
                continue
            category = parts[1]
            filename = parts[2]
            doc_id = f"{category}/{filename}"

            try:
                raw_bytes = tf.extractfile(member).read()
                raw_text = raw_bytes.decode("latin-1", errors="replace")
            except Exception:
                continue

            cleaned = clean_article(raw_text)
            tokens = cleaned.split()

            # Filter: too short — not enough semantic signal
            if len(tokens) < MIN_TOKENS:
                n_too_short += 1
                continue

            # Near-duplicate filter: hash first 200 chars of cleaned text
            # Crossposted articles appear in multiple groups with identical bodies
            fingerprint = hashlib.md5(cleaned[:200].encode()).hexdigest()
            if fingerprint in seen_hashes:
                n_duplicate += 1
                continue
            seen_hashes.add(fingerprint)

            docs.append((doc_id, category, cleaned))

    log.info(
        f"Loaded {len(docs)} usable articles "
        f"(dropped {n_too_short} too-short, {n_duplicate} duplicates)"
    )
    return docs


# ---------------------------------------------------------------------------
# Step 2: Embed & index in ChromaDB
# ---------------------------------------------------------------------------

def embed_and_index(
    docs: List[Tuple[str, str, str]],
    chroma_dir: Path,
    model_name: str = EMBED_MODEL,
    batch_size: int = BATCH_SIZE,
) -> chromadb.Collection:
    """
    Embed all documents and store them in a ChromaDB collection.

    Returns the ChromaDB collection object.
    """
    log.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    log.info(f"Initialising ChromaDB at: {chroma_dir}")
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    # Delete existing collection if it exists (idempotent re-runs)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        log.info("Dropping existing collection for clean re-index...")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        # Cosine distance is the right metric for normalised sentence embeddings.
        # L2 distance is sensitive to magnitude, which is not meaningful after
        # mean-pooling + L2-normalisation that sentence-transformers applies.
        metadata={"hnsw:space": "cosine"},
    )

    doc_ids = [d[0] for d in docs]
    categories = [d[1] for d in docs]
    texts = [d[2] for d in docs]

    log.info(f"Embedding {len(texts)} documents in batches of {batch_size}...")
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        # normalize_embeddings=True ensures unit vectors → cosine similarity
        # becomes a simple dot product (faster at query time)
        embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.extend(embs.tolist())

    log.info("Inserting into ChromaDB...")
    # Insert in batches to avoid memory spikes
    insert_batch = 500
    for i in tqdm(range(0, len(docs), insert_batch), desc="Inserting"):
        collection.add(
            ids=doc_ids[i : i + insert_batch],
            embeddings=all_embeddings[i : i + insert_batch],
            documents=texts[i : i + insert_batch],
            metadatas=[
                {"category": cat, "doc_index": i + j}
                for j, cat in enumerate(categories[i : i + insert_batch])
            ],
        )

    log.info(f"ChromaDB collection '{COLLECTION_NAME}' now has {collection.count()} documents.")
    return collection


# ---------------------------------------------------------------------------
# Public API used by other modules
# ---------------------------------------------------------------------------

def get_chroma_client() -> chromadb.PersistentClient:
    """Return a ChromaDB client pointing at the project's embeddings directory."""
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection() -> chromadb.Collection:
    """Return the newsgroups ChromaDB collection (must have been indexed first)."""
    client = get_chroma_client()
    return client.get_collection(COLLECTION_NAME)


def get_embedding_model() -> SentenceTransformer:
    """Return the shared embedding model instance."""
    return SentenceTransformer(EMBED_MODEL)


def run_pipeline(smoke_test: bool = False) -> None:
    """
    Full data pipeline entry point.

    Args:
        smoke_test: If True, only process the first 500 docs (for CI/testing).
    """
    docs = load_corpus_from_tarball(TARBALL)
    if smoke_test:
        log.info("SMOKE TEST MODE: limiting to 500 documents")
        docs = docs[:500]
    embed_and_index(docs, CHROMA_DIR)
    log.info("Data pipeline complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the data pipeline (Part 1)")
    parser.add_argument("--smoke-test", action="store_true", help="Run on 500 docs only")
    args = parser.parse_args()
    run_pipeline(smoke_test=args.smoke_test)
