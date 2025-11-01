# Quick Command Reference

## Initial Setup (One Time)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup environment file
cp .env.example .env
# Then edit .env and add your ANTHROPIC_API_KEY

# Create sample data
python scripts/create_sample_data.py
```

## Daily Usage

```bash
# Activate environment (always do this first!)
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Make a prediction
python src/predictor.py

# Test timing model only
python src/models/timing_model.py

# Test content generation only  
python src/models/content_model.py

# Check database
python -c "from src.data.database import *; s=get_session(); print(f'Posts: {s.query(Post).count()}')"
```

## Development

```bash
# Recreate database
rm data/trump_predictions.db  # Windows: del data\trump_predictions.db
python scripts/create_sample_data.py

# Run with custom config
python src/predictor.py

# View logs
cat logs/data_download.log  # Windows: type logs\data_download.log
```

## Verification

```bash
# Check Python version (need 3.10+)
python --version

# Check installed packages
pip list | grep -E "prophet|anthropic|sqlalchemy"  # Windows: pip list | findstr "prophet anthropic"

# Test database connection
python -c "from src.data.database import init_db; init_db(); print('✓ Database OK')"

# Verify API key (without revealing it)
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('✓ API key loaded' if os.getenv('ANTHROPIC_API_KEY') else '✗ API key missing')"
```

## Common Issues

```bash
# If venv activation fails (Windows)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# If Prophet installation fails
pip install pystan==2.19.1.1
pip install prophet

# If database is locked
# Close all Python processes, then:
rm data/trump_predictions.db
python scripts/create_sample_data.py

# Reset everything
rm -rf venv/ data/ models/ logs/  # Windows: rmdir /s venv data models logs
# Then start over with setup commands
```

## Project Commands

| Command | Purpose |
|---------|---------|
| `python src/predictor.py` | Make timing + content prediction |
| `python scripts/create_sample_data.py` | Generate 373 sample posts |
| `python scripts/quick_start.py` | Check setup status |
| `python src/models/timing_model.py` | Train/test timing model |
| `python src/models/content_model.py` | Test content generation |

## File Locations

| Path | Contents |
|------|----------|
| `.env` | **YOUR API KEY** (edit this!) |
| `config/config.yaml` | Model configuration |
| `data/trump_predictions.db` | SQLite database |
| `models/timing_model.pkl` | Trained Prophet model |
| `logs/` | Application logs |
| `src/predictor.py` | **MAIN ENTRY POINT** |

## Configuration Quick Edits

```yaml
# config/config.yaml

# Use more examples for content generation
content_model:
  claude_api:
    num_examples: 15  # Default: 10

# Adjust model temperature (creativity)
content_model:
  claude_api:
    temperature: 0.9  # Default: 0.8, Higher = more creative

# Change timing model sensitivity
timing_model:
  prophet:
    changepoint_prior_scale: 0.1  # Default: 0.05, Higher = more flexible
```

## Next Steps Roadmap

1. ✅ MVP working locally
2. ⬜ Add FastAPI REST API
3. ⬜ Create Streamlit dashboard
4. ⬜ Deploy to Render.com
5. ⬜ Add real-time data collection
6. ⬜ Upgrade to Neural TPP
7. ⬜ Fine-tune content model
8. ⬜ Add evaluation metrics
9. ⬜ Implement CI/CD
10. ⬜ Write blog post

---

**Remember**: Always activate your virtual environment before running any commands!

Windows: `venv\Scripts\activate`
Mac/Linux: `source venv/bin/activate`
