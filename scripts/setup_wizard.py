"""
Interactive setup wizard for Windows users.
Guides through environment setup step by step.
"""

import os
import sys
from pathlib import Path


def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")


def print_step(number, total, text):
    """Print a step indicator"""
    print(f"\n[Step {number}/{total}] {text}")
    print("-" * 60)


def check_python_version():
    """Check if Python version is adequate"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10 or higher is required!")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        print("\n   Download Python from: https://www.python.org/downloads/")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def create_env_file():
    """Interactively create .env file"""
    env_path = Path(__file__).parent.parent / '.env'
    
    if env_path.exists():
        overwrite = input("\n.env file already exists. Overwrite? (y/N): ")
        if overwrite.lower() != 'y':
            print("Keeping existing .env file.")
            return True
    
    print("\nLet's set up your environment variables.")
    print("You can leave optional fields empty by pressing Enter.\n")
    
    # Required: Anthropic API key
    while True:
        anthropic_key = input("Enter your Anthropic API key (required): ").strip()
        if anthropic_key:
            break
        print("❌ API key is required! Get one at: https://console.anthropic.com/")
    
    # Optional: Data collection API keys
    print("\n--- Optional: For real-time data collection ---")
    apify_token = input("Apify API token (press Enter to skip): ").strip()
    scrapecreators_key = input("ScrapeCreators API key (press Enter to skip): ").strip()
    
    # Write .env file
    with open(env_path, 'w') as f:
        f.write("# API Keys\n")
        f.write(f"ANTHROPIC_API_KEY={anthropic_key}\n\n")
        
        f.write("# Data Collection (Optional - uses historical data if not set)\n")
        f.write(f"APIFY_API_TOKEN={apify_token if apify_token else 'your_apify_token_here'}\n")
        f.write(f"SCRAPECREATORS_API_KEY={scrapecreators_key if scrapecreators_key else 'your_scrapecreators_key_here'}\n\n")
        
        f.write("# Database (defaults to SQLite for MVP)\n")
        f.write("DATABASE_URL=sqlite:///./data/trump_predictions.db\n\n")
        
        f.write("# Model Configuration\n")
        f.write("TIMING_MODEL_TYPE=prophet\n")
        f.write("CONTENT_MODEL_TYPE=claude_api\n\n")
        
        f.write("# Deployment\n")
        f.write("ENVIRONMENT=development\n")
        f.write("API_HOST=0.0.0.0\n")
        f.write("API_PORT=8000\n")
    
    print("\n✅ .env file created successfully!")
    return True


def check_directory_structure():
    """Ensure all necessary directories exist"""
    dirs = [
        'data/raw',
        'data/processed',
        'models',
        'logs',
    ]
    
    for dir_path in dirs:
        path = Path(__file__).parent.parent / dir_path
        path.mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure verified")
    return True


def main():
    """Main setup wizard"""
    print_header("Trump Post Predictor - Setup Wizard")
    print("This wizard will guide you through the initial setup.\n")
    
    # Step 1: Check Python
    print_step(1, 4, "Checking Python version")
    if not check_python_version():
        input("\nPress Enter to exit...")
        return
    
    # Step 2: Directory structure
    print_step(2, 4, "Setting up directories")
    check_directory_structure()
    
    # Step 3: Create .env file
    print_step(3, 4, "Configuring environment")
    if not create_env_file():
        print("\n❌ Setup incomplete. Please try again.")
        input("\nPress Enter to exit...")
        return
    
    # Step 4: Next steps
    print_step(4, 4, "Setup complete!")
    
    print("\n✅ All done! Your environment is configured.")
    print("\nNext steps:")
    print("\n1. Activate virtual environment:")
    print("   venv\\Scripts\\activate")
    print("\n2. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n3. Run setup and test:")
    print("   python scripts\\setup_and_test.py")
    print("\n4. Start making predictions!")
    print("   python src\\predictor.py")
    
    print("\n" + "="*60)
    print("  For detailed instructions, see: WINDOWS_QUICKSTART.md")
    print("="*60 + "\n")
    
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
