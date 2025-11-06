#!/usr/bin/env python3
"""
Cron job script for automated prediction validation.
Runs once, validates predictions, and exits.

Designed for Render.com Cron Jobs (free tier).
Schedule: Every hour via render.yaml

This script:
1. Finds predictions whose predicted time has passed
2. Matches them to actual posts
3. Calculates accuracy metrics
4. Updates prediction records
5. Logs results
6. Exits cleanly
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Configure logger for cron (writes to stdout for Render logs)
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)

from scripts.validate_predictions import main as validate_main


def main_cron():
    """Main cron job entry point"""
    start_time = datetime.now()

    logger.info("="*60)
    logger.info("Prediction Validation Cron Job Starting")
    logger.info(f"Execution time: {start_time}")
    logger.info("="*60)

    try:
        # Run validation
        logger.info("Starting prediction validation...")
        logger.info("="*60)

        # Call main validation function
        validate_main()

        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()

        # Log results
        logger.info("="*60)
        logger.success("Validation cron job completed successfully!")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info("="*60)

        # Exit with success code
        sys.exit(0)

    except Exception as e:
        logger.error(f"Validation cron job failed: {e}")
        logger.exception("Full traceback:")

        # Exit with error code
        sys.exit(1)


if __name__ == "__main__":
    main_cron()
