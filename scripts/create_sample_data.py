"""
Create sample dataset for testing when historical data isn't available.
"""

import sys
import os
import json
from datetime import datetime, timedelta
import random

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from data.database import init_db, get_session, Post

# Sample Trump-style posts
SAMPLE_POSTS = [
    "The Fake News Media is at it again! They will do anything to hide the TRUTH from the American people. We are Making America Great Again!",
    "Just had a tremendous meeting with great American patriots. The best is yet to come!",
    "Witch Hunt continues. Total disgrace. But we're WINNING anyway!",
    "Our economy is BOOMING. Jobs are back. America First!",
    "The Radical Left doesn't want you to know how well we're doing. Record numbers!",
    "MAKE AMERICA GREAT AGAIN! The momentum is incredible.",
    "Thank you to all the amazing supporters. You are the best!",
    "Mainstream Media refuses to cover our SUCCESS. Sad!",
    "America is back and better than ever. We will never stop fighting for YOU!",
    "Incredible rally today. Thousands of patriots. Movement like never before!",
    "The Swamp is desperate. We're draining it anyway. MAGA!",
    "Big announcement coming soon. You're going to love it!",
    "Crooked politicians don't stand a chance against the TRUTH!",
    "Record-breaking numbers across the board. They said it couldn't be done!",
    "The Silent Majority is no longer silent. America is AWAKE!",
    "Our movement is unstoppable. The best is yet to come, believe me!",
    "Fake polls, fake news. We know the REAL numbers!",
    "Standing up for hardworking Americans. Always have, always will!",
    "The establishment is panicking. Good! We're winning BIGLY!",
    "God Bless America! Together we are unstoppable.",
    "Great conversation with patriots today. The energy is incredible!",
    "They're trying to silence us. It won't work. America First!",
    "Record donations from REAL Americans. The movement grows stronger!",
    "We're fighting for YOU. Never forget that. MAGA!",
    "The Radical Left's worst nightmare: TRUTH!",
    "America deserves better and we're delivering. Promises made, promises kept!",
    "Tremendous support from all across this great nation!",
    "The corrupt system thought they could stop us. They were WRONG!",
    "Stay tuned. Big things happening. America is back!",
    "Your favorite President is hard at work for YOU!",
]


def create_sample_data(num_days=180):
    """
    Create sample data spanning the specified number of days.
    Posts are distributed with realistic patterns (more during day, clusters).
    """
    print("Creating sample dataset...")
    
    # Initialize database
    init_db()
    session = get_session()
    
    # Check if data already exists
    existing = session.query(Post).count()
    if existing > 0:
        print(f"Database already contains {existing} posts. Skipping creation.")
        session.close()
        return
    
    # Generate posts
    start_date = datetime.now() - timedelta(days=num_days)
    posts_created = 0
    
    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        
        # Random number of posts per day (0-5, with bias toward 1-3)
        num_posts_today = random.choices([0, 1, 2, 3, 4, 5], weights=[10, 25, 30, 20, 10, 5])[0]
        
        for _ in range(num_posts_today):
            # Posts more likely during daytime hours (6 AM - 11 PM)
            hour = random.choices(
                range(24),
                weights=[1,1,1,1,1,2,5,8,10,10,8,10,8,6,5,4,6,8,10,8,6,4,2,1]
            )[0]
            minute = random.randint(0, 59)
            
            post_time = current_date.replace(hour=hour, minute=minute, second=0)
            
            # Select random post content
            content = random.choice(SAMPLE_POSTS)
            
            # Generate realistic engagement metrics
            favourites = random.randint(1000, 50000)
            reblogs = random.randint(100, 10000)
            replies = random.randint(50, 5000)
            
            # Create post with unique ID
            post = Post(
                post_id=f"sample_{posts_created}_{int(post_time.timestamp())}",  # Unique ID
                user_id="107780257626128497",  # Trump's user ID
                content=content,
                created_at=post_time,
                url=f"https://truthsocial.com/@realDonaldTrump/posts/{int(post_time.timestamp())}",
                replies_count=replies,
                reblogs_count=reblogs,
                favourites_count=favourites,
                media_urls="",
                fetched_at=datetime.now()
            )
            
            session.add(post)
            posts_created += 1
    
    session.commit()
    session.close()
    
    print(f"✓ Created {posts_created} sample posts spanning {num_days} days")
    print(f"✓ Date range: {start_date.date()} to {datetime.now().date()}")
    print(f"✓ Database: data/trump_predictions.db")
    
    return posts_created


if __name__ == "__main__":
    create_sample_data(num_days=180)
