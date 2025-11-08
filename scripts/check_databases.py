#!/usr/bin/env python3
"""
Check which database files exist and their post counts.
This helps identify database path issues.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.database import init_db, get_session, Post, get_engine

def check_database(db_path):
    """Check a specific database file"""
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        engine = create_engine(f'sqlite:///{db_path}')
        Session = sessionmaker(bind=engine)
        session = Session()

        count = session.query(Post).count()
        session.close()

        return count
    except Exception as e:
        return f"Error: {e}"

# Check using the module's default
print("="*80)
print("DATABASE PATH ANALYSIS")
print("="*80)

# Get the engine to see what URL it's using
engine = get_engine()
print(f"\nDefault database URL from get_engine():")
print(f"  {engine.url}")

# Initialize and check count
init_db()
session = get_session()
count = session.query(Post).count()
session.close()

print(f"\nPosts in default database: {count}")

# Check common locations
print(f"\nChecking for database files in common locations:")

repo_root = Path(__file__).parent.parent
possible_paths = [
    repo_root / 'data' / 'trump_predictions.db',
    repo_root / 'trump_predictions.db',
    Path('./data/trump_predictions.db'),
    Path('./trump_predictions.db'),
    Path('data/trump_predictions.db'),
]

for path in possible_paths:
    if path.exists():
        abs_path = path.absolute()
        file_size = os.path.getsize(abs_path) / (1024*1024)  # MB
        post_count = check_database(abs_path)

        print(f"\n✓ Found: {abs_path}")
        print(f"  Size: {file_size:.2f} MB")
        print(f"  Posts: {post_count}")

print("\n" + "="*80)
print(f"Repository root: {repo_root.absolute()}")
print("="*80)
