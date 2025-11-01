#!/usr/bin/env python3
"""
Complete setup and test script for MVP.
Downloads sample data, trains models, makes first prediction.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from src.data.database import get_session, Post, init_db

# Sample Trump-style posts for testing (real examples from his Twitter/Truth Social style)
SAMPLE_POSTS = [
    "The Fake News Media is at it again! They don't want you to know the TRUTH about what's happening in our Country. America is doing better than ever before!",
    "Just had a great meeting with fantastic business leaders. They all agree - our Economy is the strongest it's ever been. Jobs are coming back. MAGA!",
    "The Democrats are trying to STEAL the election again! We cannot let them get away with it. We need TRANSPARENCY and FAIRNESS!",
    "Our Military is now the most powerful in the World! We are respected again. Other countries are paying their fair share. America First!",
    "Crooked Hillary and the Radical Left are destroying our Country. We must FIGHT BACK and save America from Socialism!",
    "GREAT news for American Workers! More jobs, higher wages, lower taxes. This is what happens when you put AMERICA FIRST!",
    "The Border is a disaster under Biden. We had it SECURED. Now it's wide open. This is a national security crisis!",
    "Witch Hunt! The Fake News and Radical Democrats are coming after me because they can't beat me fair and square. Total Hoax!",
    "Stock Market at ALL TIME HIGH! 401Ks looking great. This is what winning looks like. America is BACK!",
    "Big Tech is censoring Conservative voices! This is a THREAT to our Free Speech and Democracy. We must take action NOW!",
    "Just spoke to many world leaders. They all say the same thing - they miss the Trump Administration. America was RESPECTED!",
    "The Mainstream Media won't report this, but the American People know the TRUTH. We are Making America Great Again!",
    "Energy Independence! Lower gas prices! Better for American families. This is what happens with smart leadership!",
    "China is ripping us off! We need FAIR TRADE DEALS that put American Workers first. No more being taken advantage of!",
    "Law and Order! We must support our brave Police Officers who keep us safe. Defund the Police is the dumbest idea ever!",
    "Our Veterans deserve the BEST care. We are fighting for them every single day. America takes care of its heroes!",
    "Fake Polls! The Media is lying about my popularity. The SILENT MAJORITY is with me and we are going to WIN!",
    "Infrastructure Week! We are rebuilding America with beautiful new roads, bridges, and airports. Unlike others, I GET THINGS DONE!",
    "The Swamp is trying to stop me, but they can't! The American People are too smart. We see through their lies!",
    "God Bless America! God Bless our great Military! Together, we will Keep America Great! #MAGA #USA",
]


def create_synthetic_data(num_posts=100):
    """
    Create synthetic posting data with realistic temporal patterns.
    
    Trump's actual posting patterns:
    - More active in mornings (6-10 AM EST)
    - Very active evenings (7-11 PM EST)
    - Weekend activity
    - Bursts of activity (multiple posts in short time)
    """
    logger.info(f"Creating {num_posts} synthetic posts...")
    
    posts = []
    current_time = datetime.now() - timedelta(days=90)  # Start 90 days ago
    
    # Posting likelihood by hour (EST)
    hour_weights = {
        0: 0.3, 1: 0.2, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.2,
        6: 0.6, 7: 0.8, 8: 1.0, 9: 0.9, 10: 0.7, 11: 0.5,
        12: 0.4, 13: 0.3, 14: 0.3, 15: 0.4, 16: 0.5, 17: 0.6,
        18: 0.7, 19: 0.9, 20: 1.0, 21: 0.9, 22: 0.7, 23: 0.5,
    }
    
    post_id = 1
    
    while len(posts) < num_posts:
        # Random time advance (6-48 hours typical)
        hours_advance = random.choice([
            random.uniform(2, 8),    # 30% - short gap
            random.uniform(8, 24),   # 50% - medium gap  
            random.uniform(24, 48),  # 20% - long gap
        ])
        
        current_time += timedelta(hours=hours_advance)
        
        # Check if we should post based on hour of day
        hour = current_time.hour
        if random.random() < hour_weights.get(hour, 0.3):
            post = {
                'post_id': f'synthetic_{post_id:04d}',
                'user_id': '107780257626128497',  # Trump's real ID
                'content': random.choice(SAMPLE_POSTS),
                'created_at': current_time,
                'url': f'https://truthsocial.com/@realDonaldTrump/posts/{post_id}',
                'replies_count': random.randint(100, 5000),
                'reblogs_count': random.randint(500, 10000),
                'favourites_count': random.randint(1000, 50000),
                'media_urls': None,
            }
            
            posts.append(post)
            post_id += 1
            
            # 20% chance of burst posting (multiple posts quickly)
            if random.random() < 0.2:
                burst_size = random.randint(2, 4)
                for _ in range(burst_size):
                    if len(posts) >= num_posts:
                        break
                    
                    current_time += timedelta(minutes=random.randint(5, 30))
                    post = {
                        'post_id': f'synthetic_{post_id:04d}',
                        'user_id': '107780257626128497',
                        'content': random.choice(SAMPLE_POSTS),
                        'created_at': current_time,
                        'url': f'https://truthsocial.com/@realDonaldTrump/posts/{post_id}',
                        'replies_count': random.randint(100, 5000),
                        'reblogs_count': random.randint(500, 10000),
                        'favourites_count': random.randint(1000, 50000),
                        'media_urls': None,
                    }
                    posts.append(post)
                    post_id += 1
    
    logger.success(f"Created {len(posts)} synthetic posts")
    return posts[:num_posts]


def save_posts_to_db(posts):
    """Save posts to database"""
    logger.info("Saving posts to database...")
    
    session = get_session()
    
    for post_data in posts:
        # Check if post already exists
        existing = session.query(Post).filter_by(post_id=post_data['post_id']).first()
        if existing:
            continue
        
        post = Post(**post_data)
        session.add(post)
    
    session.commit()
    session.close()
    
    logger.success(f"Saved {len(posts)} posts to database")


def check_environment():
    """Check if environment is properly set up"""
    logger.info("Checking environment...")
    
    # Check for .env file
    env_path = Path(__file__).parent.parent / '.env'
    if not env_path.exists():
        logger.warning(".env file not found!")
        return False
    
    # Check for API key
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key or api_key == 'your_anthropic_api_key_here':
        logger.warning("ANTHROPIC_API_KEY not set in .env file!")
        return False
    
    logger.success("Environment OK!")
    return True


def main():
    """Main setup and test flow"""
    logger.info("="*60)
    logger.info(" "*20 + "MVP SETUP & TEST")
    logger.info("="*60)
    
    # Step 1: Check environment
    logger.info("\n[1/5] Checking environment...")
    env_ok = check_environment()
    
    if not env_ok:
        logger.error("\n❌ Environment setup incomplete!")
        logger.info("\nPlease:")
        logger.info("  1. Copy .env.example to .env")
        logger.info("  2. Add your ANTHROPIC_API_KEY to .env")
        logger.info("\nThen run this script again.")
        return
    
    # Step 2: Initialize database
    logger.info("\n[2/5] Initializing database...")
    init_db()
    logger.success("Database initialized!")
    
    # Step 3: Create sample data
    logger.info("\n[3/5] Creating sample data...")
    posts = create_synthetic_data(num_posts=150)
    save_posts_to_db(posts)
    
    # Step 4: Train models and make prediction
    logger.info("\n[4/5] Training models and making first prediction...")
    
    # Import here to ensure DB is ready
    from src.predictor import TrumpPostPredictor
    
    predictor = TrumpPostPredictor()
    predictor.train_models()
    predictor.save_models()
    
    prediction = predictor.predict(save_to_db=True)
    
    # Step 5: Display results
    logger.info("\n[5/5] Results:")
    if prediction:
        predictor.display_prediction(prediction)
        logger.success("\n✅ Setup complete! System is working!")
        logger.info("\nNext steps:")
        logger.info("  - Run 'python src/predictor.py' to make more predictions")
        logger.info("  - Run 'python scripts/add_fastapi.py' to add REST API")
        logger.info("  - Run 'streamlit run dashboard/app.py' to launch dashboard")
    else:
        logger.error("\n❌ Prediction failed! Check logs above.")
    
    logger.info("\n" + "="*60)


if __name__ == "__main__":
    main()
