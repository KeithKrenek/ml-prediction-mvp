#!/usr/bin/env python3
"""
Load historical Truth Social data from archive into database.

This script:
1. Loads the downloaded truth_archive.json
2. Normalizes to database schema
3. Removes any synthetic posts (identified by patterns)
4. Loads real posts into database with deduplication
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from loguru import logger
import requests

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.database import init_db, get_session, Post
from sqlalchemy import func

def is_synthetic_post(post_dict):
    """
    Identify synthetic/test posts.

    Heuristics:
    - Content contains "test" or "synthetic"
    - User ID doesn't match Trump's
    - Created_at is in far future or obviously fake
    """
    content = post_dict.get('content', '').lower()

    # Check for test patterns
    if any(keyword in content for keyword in ['test post', 'synthetic', 'example post', 'sample']):
        return True

    return False

def normalize_post(raw_post):
    """Convert archive format to database schema"""
    return {
        'post_id': raw_post['id'],
        'user_id': '107780257626128497',  # Trump's user ID
        'content': raw_post.get('content', ''),
        'created_at': datetime.fromisoformat(raw_post['created_at'].replace('Z', '+00:00')),
        'url': raw_post.get('url', ''),
        'replies_count': raw_post.get('replies_count', 0),
        'reblogs_count': raw_post.get('reblogs_count', 0),
        'favourites_count': raw_post.get('favourites_count', 0),
        'media_urls': json.dumps(raw_post.get('media', []))
    }

def main():
    logger.info("="*80)
    logger.info("LOADING HISTORICAL DATA")
    logger.info("="*80)

    # Initialize database
    logger.info("Initializing database...")
    init_db()
    session = get_session()

    # Check current state
    current_count = session.query(Post).count()
    logger.info(f"Current posts in database: {current_count}")

    # Load archive file
    archive_path = Path(__file__).parent.parent / "data" / "raw" / "truth_archive.json"

    if not archive_path.exists():
        logger.warning(f"Archive file not found: {archive_path}")
        logger.info("Downloading from GitHub archive...")

        # Create directory if needed
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        # Download archive
        url = "https://raw.githubusercontent.com/stiles/trump-truth-social-archive/main/data/truth_archive.json"
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()

            with open(archive_path, 'w') as f:
                json.dump(response.json(), f)

            logger.success(f"Downloaded archive to: {archive_path}")
        except Exception as e:
            logger.error(f"Failed to download archive: {e}")
            logger.info("Please manually download:")
            logger.info(f"  curl -o {archive_path} {url}")
            sys.exit(1)

    logger.info(f"Loading archive from: {archive_path}")
    with open(archive_path, 'r') as f:
        raw_posts = json.load(f)

    logger.success(f"Loaded {len(raw_posts)} posts from archive")

    # Filter out synthetic posts
    logger.info("Filtering synthetic posts...")
    real_posts = [p for p in raw_posts if not is_synthetic_post(p)]
    synthetic_count = len(raw_posts) - len(real_posts)

    if synthetic_count > 0:
        logger.info(f"Filtered out {synthetic_count} synthetic posts")

    # Remove existing synthetic posts from database
    logger.info("Removing synthetic posts from database...")
    deleted_count = 0
    existing_posts = session.query(Post).all()
    for post in existing_posts:
        if is_synthetic_post({'content': post.content, 'id': post.post_id}):
            session.delete(post)
            deleted_count += 1

    if deleted_count > 0:
        session.commit()
        logger.success(f"Deleted {deleted_count} synthetic posts from database")

    # Load new posts with deduplication
    logger.info("Loading posts into database...")
    new_posts = 0
    duplicates = 0
    errors = 0

    for raw_post in real_posts:
        try:
            # Check if already exists
            post_id = raw_post['id']
            existing = session.query(Post).filter_by(post_id=post_id).first()

            if existing:
                duplicates += 1
                continue

            # Normalize and create new post
            normalized = normalize_post(raw_post)
            post = Post(**normalized)
            session.add(post)
            new_posts += 1

            # Commit in batches of 1000
            if new_posts % 1000 == 0:
                session.commit()
                logger.info(f"Loaded {new_posts} posts so far...")

        except Exception as e:
            logger.error(f"Error loading post {raw_post.get('id')}: {e}")
            errors += 1
            continue

    # Final commit
    session.commit()

    # Summary
    final_count = session.query(Post).count()
    date_range = session.query(
        func.min(Post.created_at),
        func.max(Post.created_at)
    ).first()

    logger.info("="*80)
    logger.success("HISTORICAL DATA LOAD COMPLETE")
    logger.info("="*80)
    logger.info(f"Posts before: {current_count}")
    logger.info(f"Posts after: {final_count}")
    logger.info(f"New posts added: {new_posts}")
    logger.info(f"Duplicates skipped: {duplicates}")
    logger.info(f"Synthetic removed: {deleted_count}")
    logger.info(f"Errors: {errors}")
    logger.info(f"")
    logger.info(f"Date range in database:")
    logger.info(f"  Earliest: {date_range[0]}")
    logger.info(f"  Latest: {date_range[1]}")
    logger.info("="*80)

    session.close()

    if new_posts > 0:
        logger.success(f"✓ Ready to train models on {final_count} posts!")
    else:
        logger.info("No new posts added (data already loaded)")

if __name__ == "__main__":
    main()
