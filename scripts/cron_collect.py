#!/usr/bin/env python3
"""
Cron job script for data collection.
Runs once, collects data, and exits.

Designed for Render.com Cron Jobs (free tier).
Schedule: Every 30 minutes via render.yaml

This script:
1. Initializes the collector
2. Runs one collection cycle
3. Logs results
4. Exits cleanly
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

from src.data.collector import DataCollector
from src.utils.run_log import CronRunLogger
from src.utils.alerts import get_alert_manager


def main():
    """Main cron job entry point"""
    start_time = datetime.now()

    logger.info("="*60)
    logger.info("Data Collection Cron Job Starting")
    logger.info(f"Execution time: {start_time}")
    logger.info("="*60)

    exit_code = 0
    try:
        with CronRunLogger("collector") as run_log:
            collector = DataCollector()

            has_keys = (
                (collector.apify_token and collector.apify_token != 'your_apify_token_here') or
                (collector.scrapecreators_key and collector.scrapecreators_key != 'your_scrapecreators_key_here')
            )

            if not has_keys:
                logger.warning("No API keys configured - will use GitHub archive only")

            new_posts = collector.collect_once()
            duration = (datetime.now() - start_time).total_seconds()

            logger.info("="*60)
            logger.success("Cron job completed successfully!")
            logger.info(f"New posts collected: {new_posts}")
            logger.info(f"Duration: {duration:.2f} seconds")
            logger.info("="*60)

            run_log.update(
                records_processed=new_posts,
                extra_metadata={
                    'duration_seconds': duration,
                    'had_api_keys': has_keys
                }
            )
    except Exception as e:
        exit_code = 1
        logger.error(f"Cron job failed: {e}")
        logger.exception("Full traceback:")
        get_alert_manager().check_cron_failures("collector")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
