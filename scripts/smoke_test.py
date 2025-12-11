#!/usr/bin/env python3
"""
Lightweight smoke test to verify critical imports and database connectivity.
Run in CI and before scaling paid services.
"""

import os
import sys
from pathlib import Path

from loguru import logger
from sqlalchemy import text

# Ensure src package on path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.database import init_db, get_session  # noqa: E402
from src.predictor import TrumpPostPredictor  # noqa: E402


def main():
    logger.info("Starting smoke test...")

    # Initialize database (creates tables if needed)
    logger.info("Initializing database...")
    init_db()

    # Simple query to ensure DB is reachable
    session = get_session()
    try:
        session.execute(text("SELECT 1"))
    finally:
        session.close()
    logger.info("Database connectivity OK")

    # Ensure predictor pipeline can be constructed (without running external APIs)
    os.environ.setdefault('SMOKE_TEST', 'true')
    predictor = TrumpPostPredictor()
    logger.info("Predictor initialized with timing model: {}", type(predictor.timing_model).__name__)

    logger.success("Smoke test passed!")


if __name__ == "__main__":
    main()

