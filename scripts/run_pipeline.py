"""
run_pipeline.py — Master pipeline runner

Runs Part 1 (data pipeline) and Part 2 (clustering) in sequence.
Usage:
    conda activate cv_conda
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --smoke-test   # 500 docs, for testing
"""
import sys
import argparse
import logging
from pathlib import Path

# Make sure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline import run_pipeline
from src.clustering import run_clustering

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full data + clustering pipeline for the 20 Newsgroups system"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on 500 docs only (fast, for CI/testing)",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip Part 1 (assume ChromaDB is already populated)",
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip Part 2 (assume cluster assignments already exist)",
    )
    args = parser.parse_args()

    if not args.skip_embedding:
        log.info("=" * 60)
        log.info("PART 1: Data Pipeline (load → clean → embed → index)")
        log.info("=" * 60)
        run_pipeline(smoke_test=args.smoke_test)

    if not args.skip_clustering:
        log.info("=" * 60)
        log.info("PART 2: Fuzzy Clustering (sample → fit → predict → analyse)")
        log.info("=" * 60)
        run_clustering(smoke_test=args.smoke_test)

    log.info("=" * 60)
    log.info("Pipeline complete. To start the API server:")
    log.info("  conda activate cv_conda")
    log.info("  uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
