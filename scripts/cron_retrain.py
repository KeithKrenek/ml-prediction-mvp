#!/usr/bin/env python3
"""
Cron job script for automated model retraining.
Runs once, retrains models, and exits.

Designed for Render.com Cron Jobs (free tier).
Schedule: Weekly (default: every Sunday at 3am) via render.yaml

This script:
1. Checks if retraining is due (based on last training date)
2. Loads latest data from database
3. Trains new model versions
4. Evaluates and compares with production
5. Auto-promotes if better
6. Logs all results
7. Exits cleanly
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

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

from scripts.retrain_models import main as retrain_main
from src.data.database import get_session, ModelVersion
from src.models.model_registry import ModelRegistry


def should_retrain(model_type: str, min_days_between: int = 7) -> bool:
    """
    Check if enough time has passed since last training.

    Args:
        model_type: 'timing' or 'content'
        min_days_between: Minimum days between retraining (default: 7)

    Returns:
        True if retraining is due
    """
    session = get_session()
    try:
        # Get latest model for this type
        latest_model = (
            session.query(ModelVersion)
            .filter_by(model_type=model_type)
            .order_by(ModelVersion.trained_at.desc())
            .first()
        )

        if not latest_model:
            logger.info(f"No existing {model_type} model found - retraining recommended")
            return True

        # Check time since last training
        days_since_training = (datetime.now(timezone.utc) - latest_model.trained_at).days

        logger.info(f"{model_type.title()} model last trained {days_since_training} days ago")

        if days_since_training >= min_days_between:
            logger.info(f"Retraining is due (>= {min_days_between} days)")
            return True
        else:
            logger.info(f"Retraining not due yet ({min_days_between - days_since_training} days remaining)")
            return False

    finally:
        session.close()


def main_cron():
    """Main cron job entry point"""
    start_time = datetime.now()

    logger.info("="*60)
    logger.info("Model Retraining Cron Job Starting")
    logger.info(f"Execution time: {start_time}")
    logger.info("="*60)

    try:
        # Check if API key is configured
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if not anthropic_key or anthropic_key == 'your_anthropic_api_key_here':
            logger.warning("ANTHROPIC_API_KEY not configured - content model will not be updated")

        # Check if retraining is due
        timing_due = should_retrain('timing', min_days_between=7)
        content_due = should_retrain('content', min_days_between=7)

        if not timing_due and not content_due:
            logger.info("No retraining needed at this time")
            logger.info("="*60)
            sys.exit(0)

        # Run retraining
        logger.info("Starting retraining process...")
        logger.info("="*60)

        # Call main retraining function
        retrain_main()

        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()

        # Log results
        logger.info("="*60)
        logger.success("Retraining cron job completed successfully!")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info("="*60)

        # Exit with success code
        sys.exit(0)

    except Exception as e:
        logger.error(f"Retraining cron job failed: {e}")
        logger.exception("Full traceback:")

        # Exit with error code
        sys.exit(1)


if __name__ == "__main__":
    main_cron()
