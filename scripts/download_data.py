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
    """Download and process historical Truth Social and Twitter data"""

    def __init__(self, data_dir="data/raw/historical"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Dataset URLs
        self.datasets = {
            'trump_archive': {
                'url': 'https://raw.githubusercontent.com/stiles/trump-truth-social-archive/main/data/posts.json',
                'description': 'Trump Truth Social Archive (Feb 2022 - Oct 2024)',
                'format': 'json'
            },
            # Trump 2024 Campaign Dataset (most recent!)
            'trump_2024_campaign': {
                'url': 'https://huggingface.co/datasets/muhammetakkurt/trump-2024-campaign-truthsocial-truths/resolve/main/trump_truthsocial_dataset.csv',
                'description': 'Trump 2024 Campaign Truth Social Posts (Recent)',
                'format': 'csv'
            },
            # Trump Twitter Archive - Complete with deleted tweets
            'twitter_before_office': {
                'url': 'https://raw.githubusercontent.com/MarkHershey/CompleteTrumpTweetsArchive/master/data/realDonaldTrump_bf_office.csv',
                'description': 'Trump Twitter Archive - Before Office (2009-2017)',
                'format': 'csv'
            },
            'twitter_in_office': {
                'url': 'https://raw.githubusercontent.com/MarkHershey/CompleteTrumpTweetsArchive/master/data/realDonaldTrump_in_office.csv',
                'description': 'Trump Twitter Archive - In Office (2017-2021)',
                'format': 'csv'
            },
            # TruthSocial 2024 Election Initiative (1.5M posts, Feb-Oct 2024)
            'election_initiative_github': {
                'owner': 'kashish-s',
                'repo': 'TruthSocial_2024ElectionInitiative',
                'description': 'TruthSocial 2024 Election Initiative (1.5M posts)',
                'note': 'Large dataset - includes many accounts, filter for Trump'
            }
        }
    
    def download_trump_archive(self):
        """Download Trump's archived Truth Social posts"""
        logger.info("Downloading Trump Truth Social Archive...")

        try:
            url = self.datasets['trump_archive']['url']
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()
            logger.info(f"Downloaded {len(data)} posts from Trump Truth Social Archive")

            # Save raw data
            output_file = self.data_dir / 'trump_posts_archive.json'
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.success(f"Saved to {output_file}")
            return data

        except Exception as e:
            logger.error(f"Failed to download Trump Archive: {e}")
            return None

    def download_twitter_archive(self):
        """Download Trump's Twitter archive (includes deleted tweets)"""
        logger.info("Downloading Trump Twitter Archive...")

        all_tweets = []

        # Download before office tweets (2009-2017)
        try:
            logger.info("Downloading tweets from before office (2009-2017)...")
            url = self.datasets['twitter_before_office']['url']
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            # Save raw CSV
            output_file = self.data_dir / 'trump_twitter_before_office.csv'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response.text)

            # Parse CSV
            df_before = pd.read_csv(output_file)
            logger.info(f"Downloaded {len(df_before)} tweets from before office")
            all_tweets.append(df_before)

        except Exception as e:
            logger.error(f"Failed to download before-office tweets: {e}")

        # Download in office tweets (2017-2021)
        try:
            logger.info("Downloading tweets from in office (2017-2021)...")
            url = self.datasets['twitter_in_office']['url']
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            # Save raw CSV
            output_file = self.data_dir / 'trump_twitter_in_office.csv'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response.text)

            # Parse CSV
            df_office = pd.read_csv(output_file)
            logger.info(f"Downloaded {len(df_office)} tweets from in office")
            all_tweets.append(df_office)

        except Exception as e:
            logger.error(f"Failed to download in-office tweets: {e}")

        # Combine all tweets
        if all_tweets:
            combined = pd.concat(all_tweets, ignore_index=True)
            logger.success(f"Total Twitter archive: {len(combined)} tweets")

            # Save combined CSV
            output_file = self.data_dir / 'trump_twitter_combined.csv'
            combined.to_csv(output_file, index=False)
            logger.success(f"Saved combined archive to {output_file}")

            return combined
        else:
            logger.warning("No Twitter data downloaded")
            return None

    def download_trump_2024_campaign(self):
        """Download Trump's 2024 campaign Truth Social posts (RECENT DATA!)"""
        logger.info("Downloading Trump 2024 Campaign Truth Social data...")

        try:
            url = self.datasets['trump_2024_campaign']['url']
            logger.info(f"Downloading from Hugging Face: {url}")
            response = requests.get(url, timeout=120)
            response.raise_for_status()

            # Save raw CSV
            output_file = self.data_dir / 'trump_2024_campaign.csv'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response.text)

            # Parse CSV
            df = pd.read_csv(output_file)
            logger.success(f"Downloaded {len(df)} posts from 2024 campaign")

            return df

        except Exception as e:
            logger.error(f"Failed to download 2024 campaign data: {e}")
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

    def process_twitter_data(self, df):
        """Process Twitter archive data to match our database schema"""
        if df is None or df.empty:
            logger.warning("No Twitter data to process")
            return pd.DataFrame()

        logger.info("Processing Twitter archive data...")

        # Standardize column names to match our schema
        # Twitter CSV columns may vary, so we'll handle common variations
        column_mapping = {
            'text': 'content',
            'created_at': 'created_at',
            'id': 'post_id',
            'id_str': 'post_id',
            'retweet_count': 'reblogs_count',
            'favorite_count': 'favourites_count',
            'reply_count': 'replies_count',
            'isRetweet': 'is_retweet',
            'isDeleted': 'is_deleted'
        }

        # Rename columns
        df_processed = df.copy()
        for old_col, new_col in column_mapping.items():
            if old_col in df_processed.columns:
                df_processed.rename(columns={old_col: new_col}, inplace=True)

        # Parse timestamps
        if 'created_at' in df_processed.columns:
            df_processed['created_at'] = pd.to_datetime(df_processed['created_at'], errors='coerce')

        # Ensure we have required columns
        if 'post_id' not in df_processed.columns:
            logger.error("Missing 'post_id' or 'id' column in Twitter data")
            return pd.DataFrame()

        if 'content' not in df_processed.columns:
            logger.error("Missing 'text' or 'content' column in Twitter data")
            return pd.DataFrame()

        # Add platform indicator
        df_processed['platform'] = 'twitter'

        # Sort by timestamp
        df_processed = df_processed.sort_values('created_at')

        logger.info(f"Processed {len(df_processed)} Twitter posts")
        if not df_processed.empty and 'created_at' in df_processed.columns:
            logger.info(f"Date range: {df_processed['created_at'].min()} to {df_processed['created_at'].max()}")

        return df_processed

    def process_truthsocial_data(self, df):
        """Process Truth Social data to match our database schema"""
        if df is None or df.empty:
            logger.warning("No Truth Social data to process")
            return pd.DataFrame()

        logger.info("Processing Truth Social data...")

        # Standardize column names for Truth Social
        column_mapping = {
            'content': 'content',
            'text': 'content',
            'body': 'content',
            'created_at': 'created_at',
            'date': 'created_at',
            'published_at': 'created_at',
            'id': 'post_id',
            'truth_id': 'post_id',
            'post_id': 'post_id',
            'repost_count': 'reblogs_count',
            'reposts_count': 'reblogs_count',
            'reblogs_count': 'reblogs_count',
            'like_count': 'favourites_count',
            'likes_count': 'favourites_count',
            'favourites_count': 'favourites_count',
            'reply_count': 'replies_count',
            'replies_count': 'replies_count',
            'media_url': 'url',
            'post_url': 'url'
        }

        # Rename columns
        df_processed = df.copy()
        for old_col, new_col in column_mapping.items():
            if old_col in df_processed.columns and new_col not in df_processed.columns:
                df_processed.rename(columns={old_col: new_col}, inplace=True)

        # Parse timestamps
        if 'created_at' in df_processed.columns:
            df_processed['created_at'] = pd.to_datetime(df_processed['created_at'], errors='coerce')

        # Ensure we have required columns
        if 'post_id' not in df_processed.columns:
            # Generate post IDs if missing
            df_processed['post_id'] = [f"ts_{i}_{int(pd.Timestamp.now().timestamp())}" for i in range(len(df_processed))]
            logger.warning("Generated post_ids as they were missing")

        if 'content' not in df_processed.columns:
            logger.error("Missing content column in Truth Social data")
            return pd.DataFrame()

        # Add platform indicator
        df_processed['platform'] = 'truthsocial'

        # Sort by timestamp
        if 'created_at' in df_processed.columns:
            df_processed = df_processed.sort_values('created_at')

        logger.info(f"Processed {len(df_processed)} Truth Social posts")
        if not df_processed.empty and 'created_at' in df_processed.columns:
            valid_dates = df_processed['created_at'].dropna()
            if not valid_dates.empty:
                logger.info(f"Date range: {valid_dates.min()} to {valid_dates.max()}")

        return df_processed
    
    def load_to_database(self, df, source_name="unknown"):
        """Load processed posts into database"""
        logger.info(f"Loading posts from {source_name} to database...")

        engine = init_db()
        session = get_session(engine)

        posts_added = 0
        posts_skipped = 0
        posts_errors = 0

        for _, row in df.iterrows():
            try:
                # Create post ID if not present
                post_id = row.get('post_id', row.get('id', f"post_{row.get('created_at', datetime.utcnow()).timestamp()}"))

                # Skip if no valid created_at
                if pd.isna(row.get('created_at')):
                    posts_errors += 1
                    continue

                # Check if post already exists
                existing = session.query(Post).filter_by(post_id=str(post_id)).first()
                if existing:
                    posts_skipped += 1
                    continue

                # Create new post
                post = Post(
                    post_id=str(post_id),
                    user_id=row.get('user_id', '107780257626128497'),  # Trump's user ID
                    content=row.get('content', row.get('text', ''))[:5000],  # Limit content length
                    created_at=row['created_at'],
                    url=row.get('url', ''),
                    replies_count=int(row.get('replies_count', 0)) if pd.notna(row.get('replies_count')) else 0,
                    reblogs_count=int(row.get('reblogs_count', row.get('retweet_count', 0))) if pd.notna(row.get('reblogs_count', row.get('retweet_count', 0))) else 0,
                    favourites_count=int(row.get('favourites_count', row.get('favorite_count', 0))) if pd.notna(row.get('favourites_count', row.get('favorite_count', 0))) else 0,
                    media_urls=json.dumps(row.get('media', [])) if 'media' in row else '',
                    fetched_at=datetime.utcnow()
                )

                session.add(post)
                posts_added += 1

                if posts_added % 500 == 0:
                    session.commit()
                    logger.info(f"Added {posts_added} posts from {source_name}...")

            except Exception as e:
                posts_errors += 1
                if posts_errors < 10:  # Only log first 10 errors
                    logger.warning(f"Error processing row: {e}")
                continue

        session.commit()
        session.close()

        logger.success(f"Database load complete for {source_name}: {posts_added} added, {posts_skipped} skipped, {posts_errors} errors")
        return posts_added
    
    def run_full_pipeline(self):
        """Run complete download and processing pipeline"""
        logger.info("="*50)
        logger.info("Starting Historical Data Download Pipeline")
        logger.info("="*50)

        # Initialize database
        init_db()

        total_posts_added = 0

        # 1. Download 2024 Campaign Data (MOST RECENT - PRIORITY!)
        logger.info("\n### Downloading Trump 2024 Campaign Data (RECENT!) ###")
        campaign_2024_data = self.download_trump_2024_campaign()

        if campaign_2024_data is not None and not campaign_2024_data.empty:
            # Process and load to database
            df_2024 = self.process_truthsocial_data(campaign_2024_data)
            if not df_2024.empty:
                added = self.load_to_database(df_2024, source_name="2024 Campaign")
                total_posts_added += added

                # Save processed CSV
                output_csv = self.data_dir / '2024_campaign_processed.csv'
                df_2024.to_csv(output_csv, index=False)
                logger.success(f"Saved processed 2024 campaign data to {output_csv}")

        # 2. Download Truth Social Archive (Feb 2022 - Oct 2024)
        logger.info("\n### Downloading Truth Social Archive (2022-2024) ###")
        trump_data = self.download_trump_archive()

        if trump_data:
            # Process and load to database
            df_truth = self.process_trump_posts(trump_data)
            if not df_truth.empty:
                added = self.load_to_database(df_truth, source_name="Truth Social Archive")
                total_posts_added += added

                # Save processed CSV
                output_csv = self.data_dir / 'trump_posts_processed.csv'
                df_truth.to_csv(output_csv, index=False)
                logger.success(f"Saved processed Truth Social data to {output_csv}")

        # 3. Download Twitter Archive (historical context - 2009-2021)
        logger.info("\n### Downloading Twitter Archive (2009-2021 - Historical) ###")
        twitter_data = self.download_twitter_archive()

        if twitter_data is not None and not twitter_data.empty:
            # Process and load to database
            df_twitter = self.process_twitter_data(twitter_data)
            if not df_twitter.empty:
                added = self.load_to_database(df_twitter, source_name="Twitter Archive")
                total_posts_added += added

                # Save processed CSV
                output_csv = self.data_dir / 'twitter_posts_processed.csv'
                df_twitter.to_csv(output_csv, index=False)
                logger.success(f"Saved processed Twitter data to {output_csv}")

        # 4. Check for Election Initiative data (optional - very large)
        logger.info("\n### Checking for TruthSocial 2024 Election Initiative ###")
        self.download_election_initiative()

        logger.info("="*50)
        logger.info(f"Pipeline Complete! Total posts added: {total_posts_added}")
        logger.info("="*50)
        logger.info("\nData Timeline Coverage:")
        logger.info("  - 2024 Campaign: Most recent Truth Social posts")
        logger.info("  - 2022-2024: Truth Social archive (Feb 2022 - Oct 2024)")
        logger.info("  - 2009-2021: Twitter archive (historical context)")
        logger.info("\nRecommendation: Focus model training on 2022-2024 data for best accuracy!")

        return total_posts_added


def main():
    """Main entry point"""
    downloader = HistoricalDataDownloader()
    downloader.run_full_pipeline()


if __name__ == "__main__":
    main()
