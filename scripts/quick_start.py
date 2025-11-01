#!/usr/bin/env python3
"""
Quick start script to set up and test the Trump prediction system.
"""

import os
import sys
from pathlib import Path

def main():
    print("\n" + "="*60)
    print(" "*15 + "🚀 TRUMP POST PREDICTOR SETUP")
    print("="*60 + "\n")
    
    # Check Python version
    import sys
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher required")
        print(f"   Current version: {sys.version}")
        return
    print("✓ Python version OK")
    
    # Check for .env file
    if not Path('.env').exists():
        print("\n📝 Creating .env file...")
        with open('.env', 'w') as f:
            f.write("# Anthropic API Key (required)\n")
            f.write("ANTHROPIC_API_KEY=your_api_key_here\n")
            f.write("\n# Database (default: SQLite)\n")
            f.write("DATABASE_URL=sqlite:///./data/trump_predictions.db\n")
        print("✓ Created .env file")
        print("\n⚠️  IMPORTANT: Edit .env and add your Anthropic API key!")
        print("   Get your key at: https://console.anthropic.com/\n")
    else:
        print("✓ .env file exists")
    
    # Check for API key
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('ANTHROPIC_API_KEY', '')
    if api_key == 'your_api_key_here' or not api_key:
        print("\n❌ Anthropic API key not set!")
        print("   Please edit .env and add your API key")
        print("   Then run this script again\n")
        return
    print("✓ API key configured")
    
    # Check for data
    import sqlite3
    db_path = 'data/trump_predictions.db'
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM posts")
        count = cursor.fetchone()[0]
        conn.close()
        
        if count == 0:
            print("\n📥 No data found. Creating sample dataset...")
            os.system(f"{sys.executable} scripts/create_sample_data.py")
        else:
            print(f"✓ Database contains {count} posts")
    except:
        print("\n📥 Creating sample dataset...")
        os.system(f"{sys.executable} scripts/create_sample_data.py")
    
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE!")
    print("="*60)
    print("\nReady to make predictions! Run:")
    print(f"\n   {sys.executable} src/predictor.py\n")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
