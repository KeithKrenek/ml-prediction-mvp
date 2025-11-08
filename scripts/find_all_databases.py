#!/usr/bin/env python3
"""
Find ALL SQLite database files in the project directory tree.
This helps identify hidden or misplaced databases.
"""

import os
from pathlib import Path

def find_all_databases(root_path):
    """Recursively find all .db files"""
    db_files = []

    for dirpath, dirnames, filenames in os.walk(root_path):
        # Skip venv and node_modules
        dirnames[:] = [d for d in dirnames if d not in ['venv', 'node_modules', '.git', '__pycache__']]

        for filename in filenames:
            if filename.endswith('.db'):
                full_path = Path(dirpath) / filename
                size_mb = os.path.getsize(full_path) / (1024*1024)
                db_files.append((full_path, size_mb))

    return db_files

print("="*80)
print("SEARCHING FOR ALL DATABASE FILES")
print("="*80)

repo_root = Path(__file__).parent.parent
print(f"\nSearching in: {repo_root.absolute()}")
print(f"This may take a few seconds...\n")

db_files = find_all_databases(repo_root)

if not db_files:
    print("No .db files found!")
else:
    print(f"Found {len(db_files)} database file(s):\n")

    for db_path, size_mb in sorted(db_files, key=lambda x: x[1], reverse=True):
        print(f"  {db_path}")
        print(f"    Size: {size_mb:.2f} MB")

        # Try to count posts
        try:
            import sys
            sys.path.append(str(Path(__file__).parent.parent))

            from sqlalchemy import create_engine, text
            from sqlalchemy.orm import sessionmaker

            engine = create_engine(f'sqlite:///{db_path}')
            Session = sessionmaker(bind=engine)
            session = Session()

            result = session.execute(text("SELECT COUNT(*) FROM posts"))
            count = result.scalar()
            session.close()

            print(f"    Posts: {count}")
        except Exception as e:
            print(f"    Posts: Unable to count ({type(e).__name__})")

        print()

print("="*80)
print("RECOMMENDED ACTION:")
print("="*80)
print("Keep ONLY the largest file (should be ~16-17 MB with 29,400 posts)")
print("Delete all other .db files to avoid confusion")
print("="*80)
