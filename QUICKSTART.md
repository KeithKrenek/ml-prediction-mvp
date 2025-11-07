# Quick Start Guide - Training Models

## Prerequisites Setup

### Step 1: Update Dependencies

```bash
# Upgrade to latest pip
python -m pip install --upgrade pip

# Install/upgrade key dependencies with fixes
pip install --upgrade cmdstanpy>=1.2.0
pip install --upgrade anthropic>=0.40.0

# Install all requirements
pip install -r requirements.txt
```

### Step 2: Load Historical Data (CRITICAL!)

**You currently have only 294 posts - you need 29,400!**

```bash
# This will auto-download and load the archive
python scripts/load_historical_data.py

# Expected output:
# Downloading from GitHub archive...
# ✓ Downloaded archive
# ✓ Loaded 29,400 posts
# ✓ Ready to train models!
```

## Training Models

### Step 3: Train Initial Models

```bash
python scripts/retrain_models.py
```

**Expected output:**
- Training on ~23,520 posts (80% of 29,400)
- Test set: ~5,880 posts
- NTPP model training: 2-3 minutes
- MAE: 1.5-2.5 hours
- Within-6h accuracy: 60-70%

## Troubleshooting

### Prophet Error: `'Prophet' object has no attribute 'stan_backend'`

**Fix:**
```bash
# Install cmdstanpy backend
pip install --upgrade cmdstanpy

# Install cmdstan compiler (required once)
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

**Alternative - Use NTPP instead (recommended):**
Edit `config/config.yaml`:
```yaml
timing_model:
  type: "ntpp"  # Use NTPP instead of Prophet
```

### Anthropic Error: `got an unexpected keyword argument 'proxies'`

**Fix:**
```bash
# Upgrade Anthropic SDK
pip install --upgrade "anthropic>=0.40.0"
```

### Only 294 Posts in Database

**You need to load historical data first!**
```bash
python scripts/load_historical_data.py
```

This downloads 29,400+ posts from the GitHub archive.

## Verification

After training, verify models exist:

```bash
# Check timing models
ls -lh models/timing/

# Check content models
ls -lh models/content/

# Test prediction
python -c "
from src.predictor import TrumpPredictor
p = TrumpPredictor()
result = p.predict()
print(f'Next post in: {result[\"timing_hours_from_now\"]:.1f} hours')
print(f'Content: {result[\"predicted_content\"][:100]}...')
"
```

## Next Steps

1. ✅ Models trained locally
2. Deploy to Render (push to git)
3. Enable cron jobs for automated predictions
4. Monitor validation metrics
5. Iterate based on real accuracy data
