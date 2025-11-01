"""
Download and process historical Truth Social data.
Primary source: TruthSocial 2024 Election Initiative dataset
"""

import os
import sys
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import requests
from loguru import logger

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from data.database import init_db, get_session, Post

# Configure logger
logger.add("logs/data_download.log", rotation="10 MB")


class HistoricalDataDownloader:
    """Download and process historical Truth Social data"""
    
    def __init__(self, data_dir="data/raw/historical"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset URLs
        self.datasets = {
            'trump_archive': {
                'url': 'https://raw.githubusercontent.com/stiles/trump-truth-social-archive/main/data/posts.json',
                'description': 'Trump Truth Social Archive (discontinued Oct 2024)',
                'format': 'json'
            },
            # We'll use direct GitHub API to check for CSV files
            'election_initiative_github': {
                'owner': 'kashish-s',
                'repo': 'TruthSocial_2024ElectionInitiative',
                'description': 'TruthSocial 2024 Election Initiative'
            }
        }
    
    def download_trump_archive(self):
        """Download Trump's archived posts"""
        logger.info("Downloading Trump Truth Social Archive...")
        
        try:
            url = self.datasets['trump_archive']['url']
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Downloaded {len(data)} posts from Trump Archive")
            
            # Save raw data
            output_file = self.data_dir / 'trump_posts_archive.json'
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.success(f"Saved to {output_file}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to download Trump Archive: {e}")
            return None
    
    def download_election_initiative(self):
        """
        Download TruthSocial 2024 Election Initiative data.
        Note: This is a large dataset, we'll provide instructions for manual download.
        """
        logger.info("TruthSocial 2024 Election Initiative dataset instructions:")
        logger.info("This dataset is large (1.5M posts). Please download manually:")
        logger.info("1. Visit: https://github.com/kashish-s/TruthSocial_2024ElectionInitiative")
        logger.info("2. Clone the repository or download specific files")
        logger.info("3. Alternative: Download from Kaggle mirror")
        logger.info("4. Place CSV files in: data/raw/historical/")
        
        # Check if files already exist
        csv_files = list(self.data_dir.glob('*.csv'))
        if csv_files:
            logger.info(f"Found {len(csv_files)} CSV files already downloaded")
            return True
        else:
            logger.warning("No CSV files found. Please download manually.")
            return False
    
    def process_trump_posts(self, data):
        """Process downloaded posts and filter for Trump"""
        if not data:
            logger.warning("No data to process")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Parse timestamps
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
        elif 'date' in df.columns:
            df['created_at'] = pd.to_datetime(df['date'])
        
        # Sort by timestamp
        df = df.sort_values('created_at')
        
        logger.info(f"Processed {len(df)} Trump posts")
        logger.info(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}")
        
        return df
    
    def load_to_database(self, df):
        """Load processed posts into database"""
        logger.info("Loading posts to database...")
        
        engine = init_db()
        session = get_session(engine)
        
        posts_added = 0
        posts_skipped = 0
        
        for _, row in df.iterrows():
            # Create post ID if not present
            post_id = row.get('id', row.get('post_id', f"post_{row['created_at'].timestamp()}"))
            
            # Check if post already exists
            existing = session.query(Post).filter_by(post_id=str(post_id)).first()
            if existing:
                posts_skipped += 1
                continue
            
            # Create new post
            post = Post(
                post_id=str(post_id),
                user_id=row.get('user_id', '107780257626128497'),  # Trump's user ID
                content=row.get('content', row.get('text', '')),
                created_at=row['created_at'],
                url=row.get('url', ''),
                replies_count=row.get('replies_count', 0),
                reblogs_count=row.get('reblogs_count', row.get('retweets', 0)),
                favourites_count=row.get('favourites_count', row.get('favorites', 0)),
                media_urls=json.dumps(row.get('media', [])) if 'media' in row else '',
                fetched_at=datetime.utcnow()
            )
            
            session.add(post)
            posts_added += 1
            
            if posts_added % 100 == 0:
                session.commit()
                logger.info(f"Added {posts_added} posts...")
        
        session.commit()
        session.close()
        
        logger.success(f"Database load complete: {posts_added} added, {posts_skipped} skipped")
        return posts_added
    
    def run_full_pipeline(self):
        """Run complete download and processing pipeline"""
        logger.info("="*50)
        logger.info("Starting Historical Data Download Pipeline")
        logger.info("="*50)
        
        # Initialize database
        init_db()
        
        # Download Trump Archive
        trump_data = self.download_trump_archive()
        
        if trump_data:
            # Process and load to database
            df = self.process_trump_posts(trump_data)
            self.load_to_database(df)
            
            # Save processed CSV
            output_csv = self.data_dir / 'trump_posts_processed.csv'
            df.to_csv(output_csv, index=False)
            logger.success(f"Saved processed data to {output_csv}")
        
        # Check for Election Initiative data
        self.download_election_initiative()
        
        logger.info("="*50)
        logger.info("Pipeline Complete!")
        logger.info("="*50)


def main():
    """Main entry point"""
    downloader = HistoricalDataDownloader()
    downloader.run_full_pipeline()


if __name__ == "__main__":
    main()
