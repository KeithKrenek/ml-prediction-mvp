"""
Real-time data collection service for Trump Truth Social posts.

Implements:
- Polling strategy (check every 5 minutes)
- Multiple data sources (Apify, ScrapeCreators, GitHub archive)
- Automatic fallback between sources
- Rate limiting and error handling
- Background scheduling
"""

import os
import sys
from pathlib import Path
import time
import json
import schedule
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict
import requests

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from loguru import logger
from dotenv import load_dotenv

from src.data.database import get_session, Post

# Load environment
load_dotenv()


class DataCollector:
    """
    Collects Trump's Truth Social posts from multiple sources.
    """
    
    def __init__(self):
        self.apify_token = os.getenv('APIFY_API_TOKEN')
        self.scrapecreators_key = os.getenv('SCRAPECREATORS_API_KEY')
        self.trump_user_id = '107780257626128497'
        
        # Track last successful fetch time
        self.last_fetch_time = None
        
        logger.info("Data Collector initialized")
    
    def fetch_from_github_archive(self) -> List[Dict]:
        """
        Fetch from Trump Truth Social Archive (GitHub).
        This archive was maintained until October 2024.
        """
        logger.info("Fetching from GitHub archive...")
        
        try:
            # Fetch latest archive file
            url = "https://raw.githubusercontent.com/stiles/trump-truth-social-archive/main/data/posts.json"
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            posts = response.json()
            logger.success(f"Fetched {len(posts)} posts from GitHub archive")
            
            return posts
        
        except Exception as e:
            logger.error(f"Failed to fetch from GitHub archive: {e}")
            return []
    
    def fetch_from_apify(self) -> List[Dict]:
        """
        Fetch using Apify Truth Social scraper.
        Requires APIFY_API_TOKEN in environment.
        """
        if not self.apify_token or self.apify_token == 'your_apify_token_here':
            logger.warning("Apify token not configured, skipping...")
            return []
        
        logger.info("Fetching from Apify...")
        
        try:
            # Start Apify actor run
            start_url = f"https://api.apify.com/v2/acts/louisdeconinck~truth-social-scraper/runs"
            
            payload = {
                "startUrls": [
                    {"url": f"https://truthsocial.com/@realDonaldTrump"}
                ],
                "maxItems": 50
            }
            
            headers = {
                "Authorization": f"Bearer {self.apify_token}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(start_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            run_data = response.json()
            run_id = run_data['data']['id']
            
            logger.info(f"Apify run started: {run_id}")
            
            # Poll for results (wait up to 2 minutes)
            for _ in range(24):  # 24 * 5 seconds = 2 minutes
                time.sleep(5)
                
                status_url = f"https://api.apify.com/v2/acts/louisdeconinck~truth-social-scraper/runs/{run_id}"
                status_response = requests.get(status_url, headers=headers, timeout=30)
                status_data = status_response.json()
                
                status = status_data['data']['status']
                
                if status == 'SUCCEEDED':
                    # Get results
                    dataset_id = status_data['data']['defaultDatasetId']
                    results_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
                    
                    results_response = requests.get(results_url, headers=headers, timeout=30)
                    posts = results_response.json()
                    
                    logger.success(f"Fetched {len(posts)} posts from Apify")
                    return posts
                
                elif status in ['FAILED', 'ABORTED', 'TIMED-OUT']:
                    logger.error(f"Apify run {status}")
                    return []
            
            logger.warning("Apify run timed out")
            return []
        
        except Exception as e:
            logger.error(f"Failed to fetch from Apify: {e}")
            return []
    
    def fetch_from_scrapecreators(self) -> List[Dict]:
        """
        Fetch using ScrapeCreators API.
        Requires SCRAPECREATORS_API_KEY in environment.
        """
        if not self.scrapecreators_key or self.scrapecreators_key == 'your_scrapecreators_key_here':
            logger.warning("ScrapeCreators key not configured, skipping...")
            return []
        
        logger.info("Fetching from ScrapeCreators...")
        
        try:
            url = f"https://api.scrapecreators.com/v1/truthsocial/user/posts"
            
            headers = {
                'x-api-key': self.scrapecreators_key
            }
            
            params = {
                'user_id': self.trump_user_id
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            posts = data.get('posts', [])
            
            logger.success(f"Fetched {len(posts)} posts from ScrapeCreators")
            return posts
        
        except Exception as e:
            logger.error(f"Failed to fetch from ScrapeCreators: {e}")
            return []
    
    def normalize_post_data(self, post: Dict, source: str) -> Dict:
        """
        Normalize post data from different sources into consistent format.
        """
        # Different sources have different field names
        if source == 'github':
            return {
                'post_id': post.get('id', post.get('post_id')),
                'user_id': self.trump_user_id,
                'content': post.get('content', post.get('text', '')),
                'created_at': datetime.fromisoformat(post['created_at'].replace('Z', '+00:00')),
                'url': post.get('url', ''),
                'replies_count': post.get('replies_count', 0),
                'reblogs_count': post.get('reblogs_count', 0),
                'favourites_count': post.get('favourites_count', 0),
                'media_urls': json.dumps(post.get('media_attachments', [])) if post.get('media_attachments') else None
            }
        
        elif source == 'apify':
            return {
                'post_id': post.get('id', post.get('postId')),
                'user_id': self.trump_user_id,
                'content': post.get('text', post.get('content', '')),
                'created_at': datetime.fromisoformat(post['createdAt'].replace('Z', '+00:00')),
                'url': post.get('url', ''),
                'replies_count': post.get('repliesCount', 0),
                'reblogs_count': post.get('reblogsCount', 0),
                'favourites_count': post.get('favoritesCount', 0),
                'media_urls': json.dumps(post.get('media', [])) if post.get('media') else None
            }
        
        elif source == 'scrapecreators':
            return {
                'post_id': post.get('id'),
                'user_id': self.trump_user_id,
                'content': post.get('content', ''),
                'created_at': datetime.fromisoformat(post['created_at'].replace('Z', '+00:00')),
                'url': post.get('url', ''),
                'replies_count': post.get('replies_count', 0),
                'reblogs_count': post.get('reblogs_count', 0),
                'favourites_count': post.get('favourites_count', 0),
                'media_urls': json.dumps(post.get('media_attachments', [])) if post.get('media_attachments') else None
            }
        
        return post
    
    def save_posts(self, posts: List[Dict], source: str) -> int:
        """
        Save posts to database, avoiding duplicates.
        Returns number of new posts saved.
        """
        if not posts:
            return 0
        
        session = get_session()
        new_posts = 0
        
        try:
            for post_data in posts:
                try:
                    # Normalize data
                    normalized = self.normalize_post_data(post_data, source)
                    
                    # Check if post already exists
                    existing = session.query(Post).filter_by(post_id=normalized['post_id']).first()
                    
                    if existing:
                        continue
                    
                    # Create new post
                    post = Post(**normalized)
                    session.add(post)
                    new_posts += 1
                
                except Exception as e:
                    logger.warning(f"Failed to save post: {e}")
                    continue
            
            session.commit()
            logger.success(f"Saved {new_posts} new posts from {source}")
        
        except Exception as e:
            logger.error(f"Database error: {e}")
            session.rollback()
        
        finally:
            session.close()
        
        return new_posts
    
    def collect_once(self) -> int:
        """
        Run one collection cycle, trying all sources.
        Returns total number of new posts collected.
        """
        logger.info("="*60)
        logger.info(f"Starting collection cycle at {datetime.now()}")
        logger.info("="*60)
        
        total_new = 0
        
        # Try each source in order of preference
        sources = [
            ('scrapecreators', self.fetch_from_scrapecreators),
            ('apify', self.fetch_from_apify),
            ('github', self.fetch_from_github_archive),
        ]
        
        for source_name, fetch_func in sources:
            try:
                logger.info(f"\nTrying source: {source_name}")
                posts = fetch_func()
                
                if posts:
                    new_posts = self.save_posts(posts, source_name)
                    total_new += new_posts
                    
                    # If we got new posts, no need to try other sources
                    if new_posts > 0:
                        logger.success(f"✅ Got {new_posts} new posts from {source_name}")
                        break
                else:
                    logger.info(f"No posts from {source_name}, trying next source...")
            
            except Exception as e:
                logger.error(f"Error with {source_name}: {e}")
                continue
        
        self.last_fetch_time = datetime.now()
        
        logger.info(f"\nCollection cycle complete. Total new posts: {total_new}")
        logger.info("="*60 + "\n")
        
        return total_new
    
    def start_continuous_collection(self, interval_minutes: int = 5):
        """
        Start continuous collection on a schedule.
        
        Args:
            interval_minutes: How often to check for new posts (default: 5)
        """
        logger.info(f"Starting continuous collection (every {interval_minutes} minutes)...")
        
        # Schedule collection
        schedule.every(interval_minutes).minutes.do(self.collect_once)
        
        # Run once immediately
        self.collect_once()
        
        # Keep running
        while True:
            try:
                schedule.run_pending()
                time.sleep(30)  # Check schedule every 30 seconds
            except KeyboardInterrupt:
                logger.info("Collection stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                time.sleep(60)  # Wait a minute before retrying


def main():
    """Main entry point for data collection service"""
    logger.info("Trump Truth Social Data Collector")
    logger.info("="*60)
    
    collector = DataCollector()
    
    # Check if any API keys are configured
    has_apify = collector.apify_token and collector.apify_token != 'your_apify_token_here'
    has_scrapecreators = collector.scrapecreators_key and collector.scrapecreators_key != 'your_scrapecreators_key_here'
    
    if not (has_apify or has_scrapecreators):
        logger.warning("\n⚠️  No API keys configured!")
        logger.info("\nWill use GitHub archive (historical data only)")
        logger.info("\nTo enable real-time collection, add to .env:")
        logger.info("  - APIFY_API_TOKEN=your_token")
        logger.info("  - SCRAPECREATORS_API_KEY=your_key")
        logger.info("\n")
    
    # Collect once for testing
    logger.info("Running initial collection...")
    collector.collect_once()
    
    # Ask if user wants continuous collection
    logger.info("\n" + "="*60)
    logger.info("Initial collection complete!")
    logger.info("\nStart continuous collection? (checks every 5 minutes)")
    logger.info("Press Ctrl+C to stop at any time")
    logger.info("="*60 + "\n")
    
    # For automated deployment, check environment variable
    if os.getenv('RUN_CONTINUOUS', 'false').lower() == 'true':
        logger.info("RUN_CONTINUOUS=true, starting automatic collection...")
        collector.start_continuous_collection(interval_minutes=5)
    else:
        logger.info("Run with RUN_CONTINUOUS=true to enable automatic collection")
        logger.info("Or use: python -c 'from collector import DataCollector; DataCollector().start_continuous_collection()'")


if __name__ == "__main__":
    main()
