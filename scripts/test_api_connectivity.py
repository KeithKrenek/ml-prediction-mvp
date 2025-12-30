#!/usr/bin/env python3
"""
Test script to verify API connectivity and response format.
Run this to diagnose data collection issues.
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime
import requests

sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from dotenv import load_dotenv

load_dotenv()


def test_scrapecreators():
    """Test ScrapeCreators API"""
    logger.info("Testing ScrapeCreators API...")

    api_key = os.getenv('SCRAPECREATORS_API_KEY')
    if not api_key or api_key == 'your_scrapecreators_key_here':
        logger.error("SCRAPECREATORS_API_KEY not configured")
        return False

    try:
        url = "https://api.scrapecreators.com/v1/truthsocial/user/posts"
        headers = {'x-api-key': api_key}
        params = {'user_id': '107780257626128497'}

        logger.info(f"Requesting: {url}")
        logger.info(f"Params: {params}")

        response = requests.get(url, headers=headers, params=params, timeout=30)

        logger.info(f"Status Code: {response.status_code}")
        logger.info(f"Response Headers: {dict(response.headers)}")

        if response.status_code != 200:
            logger.error(f"API returned error: {response.text}")
            return False

        data = response.json()
        logger.info(f"Response Keys: {list(data.keys())}")

        posts = data.get('posts', [])
        logger.success(f"Got {len(posts)} posts")

        if posts:
            sample = posts[0]
            logger.info("\nSample Post Structure:")
            logger.info(json.dumps(sample, indent=2, default=str))

            # Validate required fields
            required = ['id', 'created_at', 'content']
            missing = [f for f in required if f not in sample]
            if missing:
                logger.error(f"Missing required fields: {missing}")
                return False

            # Test date parsing
            try:
                datetime.fromisoformat(sample['created_at'].replace('Z', '+00:00'))
                logger.success("Date format is valid")
            except Exception as e:
                logger.error(f"Invalid date format: {e}")
                return False

        return True

    except Exception as e:
        logger.error(f"Error testing ScrapeCreators: {e}")
        logger.exception("Full traceback:")
        return False


def test_database():
    """Test database connectivity"""
    logger.info("\nTesting database...")

    try:
        from src.data.database import get_session, Post

        session = get_session()
        count = session.query(Post).count()
        logger.success(f"Database connected. Total posts: {count}")

        if count > 0:
            latest = session.query(Post).order_by(Post.created_at.desc()).first()
            logger.info(f"Latest post: {latest.created_at}")
            logger.info(f"Post ID: {latest.post_id}")

        session.close()
        return True

    except Exception as e:
        logger.error(f"Database error: {e}")
        logger.exception("Full traceback:")
        return False


def main():
    logger.info("="*60)
    logger.info("API Connectivity Test")
    logger.info("="*60)

    scrapecreators_ok = test_scrapecreators()
    db_ok = test_database()

    logger.info("\n" + "="*60)
    logger.info("Test Results:")
    logger.info(f"  ScrapeCreators API: {'✓' if scrapecreators_ok else '✗'}")
    logger.info(f"  Database: {'✓' if db_ok else '✗'}")
    logger.info("="*60)

    if scrapecreators_ok and db_ok:
        logger.success("\n✓ All tests passed!")
        sys.exit(0)
    else:
        logger.error("\n✗ Some tests failed - check logs above")
        sys.exit(1)


if __name__ == "__main__":
    main()
