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


def main():
    """Main cron job entry point"""
    start_time = datetime.now()
    
    logger.info("="*60)
    logger.info("Data Collection Cron Job Starting")
    logger.info(f"Execution time: {start_time}")
    logger.info("="*60)
    
    try:
        # Initialize collector
        collector = DataCollector()
        
        # Check if any API keys are configured
        has_keys = (
            (collector.apify_token and collector.apify_token != 'your_apify_token_here') or
            (collector.scrapecreators_key and collector.scrapecreators_key != 'your_scrapecreators_key_here')
        )
        
        if not has_keys:
            logger.warning("No API keys configured - will use GitHub archive only")
        
        # Run one collection cycle
        new_posts = collector.collect_once()
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        
        # Log results
        logger.info("="*60)
        logger.success(f"Cron job completed successfully!")
        logger.info(f"New posts collected: {new_posts}")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info("="*60)
        
        # Exit with success code
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Cron job failed: {e}")
        logger.exception("Full traceback:")
        
        # Exit with error code
        sys.exit(1)


if __name__ == "__main__":
    main()
