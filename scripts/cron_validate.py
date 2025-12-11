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

from scripts.validate_predictions import run_validation
from src.utils.run_log import CronRunLogger
from src.utils.alerts import get_alert_manager


def main_cron():
    """Main cron job entry point"""
    start_time = datetime.now()

    logger.info("="*60)
    logger.info("Prediction Validation Cron Job Starting")
    logger.info(f"Execution time: {start_time}")
    logger.info("="*60)

    exit_code = 0
    alert_manager = get_alert_manager()

    try:
        with CronRunLogger("validate") as run_log:
            logger.info("Starting prediction validation...")
            logger.info("="*60)

            summary, duration = run_validation()

            logger.info("="*60)
            logger.success("Validation cron job completed successfully!")
            logger.info(f"Duration: {duration:.2f} seconds")
            logger.info("="*60)

            run_log.update(
                records_processed=summary.get('validated', 0),
                extra_metadata={
                    'duration_seconds': duration,
                    'matched': summary.get('matched', 0),
                    'correct': summary.get('correct', 0),
                    'accuracy': summary.get('accuracy')
                }
            )
            if summary.get('matched'):
                alert_manager.check_accuracy_threshold(
                    summary.get('accuracy'),
                    {'matched': summary.get('matched'), 'correct': summary.get('correct')}
                )
    except Exception as e:
        exit_code = 1
        logger.error(f"Validation cron job failed: {e}")
        logger.exception("Full traceback:")
        alert_manager.check_cron_failures("validate")

    sys.exit(exit_code)


if __name__ == "__main__":
    main_cron()
