# 🚀 Quick Start Guide - Windows Edition

Get your MVP running in **under 30 minutes**!

## Prerequisites

- ✅ Python 3.10 or higher ([Download](https://www.python.org/downloads/))
- ✅ Git ([Download](https://git-scm.com/download/win))
- ✅ Anthropic API key ([Get one](https://console.anthropic.com/))

## Step 1: Clone and Setup (5 minutes)

Open **PowerShell** or **Command Prompt**:

```powershell
# Navigate to where you want the project
cd C:\Users\YourName\Projects

# Clone the repository (or navigate to existing folder)
cd trump-prediction-mvp

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies (this may take 2-3 minutes)
pip install -r requirements.txt
```

## Step 2: Configure Environment (2 minutes)

Create a `.env` file in the project root:

```powershell
# Copy the example file
copy .env.example .env

# Edit .env with Notepad
notepad .env
```

Add your Anthropic API key:
```
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here
```

Save and close Notepad.

## Step 3: Initialize and Test (5 minutes)

```powershell
# Run the setup script - this will:
# - Initialize the database
# - Create sample data
# - Train models
# - Make your first prediction!
python scripts\setup_and_test.py
```

You should see output like:
```
============================================================
                    PREDICTION RESULTS
============================================================

🕐 PREDICTED TIME:
   Monday, November 04, 2025 at 08:30 AM
   Confidence: 75.3%

📝 PREDICTED CONTENT:
   The Fake News Media is at it again! They don't want you to 
   know the TRUTH. America is doing better than ever. MAGA!
   Confidence: 70.0%

📊 OVERALL CONFIDENCE: 72.6%
============================================================
```

🎉 **Success!** Your basic prediction system is working!

## Step 4: Start the API (2 minutes)

In a **new terminal** (keep the first one open):

```powershell
# Activate virtual environment
cd C:\Users\YourName\Projects\trump-prediction-mvp
venv\Scripts\activate

# Start the API server
uvicorn src.api.main:app --reload
```

Open your browser to: http://localhost:8000/docs

You'll see the interactive API documentation!

### Test the API:

1. Click on **"POST /predict"**
2. Click **"Try it out"**
3. Click **"Execute"**
4. See your prediction in the response!

## Step 5: Launch the Dashboard (2 minutes)

In **another new terminal**:

```powershell
# Activate virtual environment
cd C:\Users\YourName\Projects\trump-prediction-mvp
venv\Scripts\activate

# Start Streamlit dashboard
streamlit run dashboard\app.py
```

Your browser should automatically open to the dashboard!

**What you can do:**
- 🎯 Make predictions
- 📊 View analytics
- 📝 See recent posts
- ⚙️ Learn about the system

## Step 6: Start Data Collection (Optional - 2 minutes)

For **real-time** post collection:

```powershell
# In a new terminal
venv\Scripts\activate

# Run the collector
python src\data\collector.py
```

**Note:** Without API keys, this will use historical data from GitHub archive.

To enable real-time collection, add to `.env`:
```
APIFY_API_TOKEN=your_apify_token
# or
SCRAPECREATORS_API_KEY=your_scrapecreators_key
```

---

## 🎯 Quick Reference

### Making Predictions

**Command Line:**
```powershell
python src\predictor.py
```

**API:**
```powershell
curl -X POST http://localhost:8000/predict
```

**Dashboard:**
- Go to http://localhost:8501
- Click "Make Prediction" button

### Running Components Simultaneously

You'll need **4 terminal windows** for the complete system:

**Terminal 1 - API Server:**
```powershell
venv\Scripts\activate
uvicorn src.api.main:app --reload
```

**Terminal 2 - Dashboard:**
```powershell
venv\Scripts\activate
streamlit run dashboard\app.py
```

**Terminal 3 - Data Collector (optional):**
```powershell
venv\Scripts\activate
python src\data\collector.py
```

**Terminal 4 - Your work:**
```powershell
venv\Scripts\activate
# Run tests, make predictions, etc.
```

---

## 📦 What You Have Now

✅ **Working Timing Model** - Predicts when next post will occur
✅ **Working Content Model** - Generates post content
✅ **REST API** - FastAPI with auto-generated docs
✅ **Web Dashboard** - Beautiful Streamlit interface
✅ **Data Collection** - Automated post fetching
✅ **Database** - SQLite with post history

---

## 🚀 Next Steps

### 1. Deploy to Production (Day 7)

**Create Render.com account:**
1. Go to https://render.com
2. Sign up with GitHub
3. Click "New +" → "Blueprint"
4. Connect your repo
5. Render will auto-deploy from `render.yaml`!

**Set environment variables in Render:**
- `ANTHROPIC_API_KEY`
- `APIFY_API_TOKEN` (optional)
- `SCRAPECREATORS_API_KEY` (optional)

### 2. Deploy Dashboard to Streamlit Cloud (Day 7)

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select your repo
5. Set main file path: `dashboard/app.py`
6. Deploy!

### 3. Improve Models (Week 2+)

**Timing Model Upgrade:**
```powershell
# Install EasyTPP
pip install easytpp --break-system-packages

# Train Neural TPP model
python scripts\train_neural_tpp.py
```

**Content Model Upgrade:**
```powershell
# Fine-tune GPT-2 locally
python scripts\finetune_gpt2.py

# Or use Colab for Phi-3
# (script will open Colab notebook)
python scripts\open_colab_training.py
```

---

## ⚠️ Troubleshooting

### "Python not found"
- Install Python from python.org
- Check "Add Python to PATH" during installation

### "pip not found"  
```powershell
python -m pip install --upgrade pip
```

### "Module not found"
```powershell
# Make sure virtual environment is activated
venv\Scripts\activate

# Reinstall requirements
pip install -r requirements.txt
```

### "Port already in use"
```powershell
# Kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Database locked
```powershell
# Close all running instances and delete the database
del data\trump_predictions.db

# Reinitialize
python scripts\setup_and_test.py
```

---

## 💡 Tips

**Development Workflow:**
1. Keep API running in one terminal
2. Keep dashboard running in another
3. Code changes auto-reload
4. Test in browser immediately

**Windows-Specific:**
- Use PowerShell (not Command Prompt) for better experience
- Consider Windows Terminal for tabbed terminals
- Use `Ctrl+C` to stop running services

**Performance:**
- First prediction takes ~30 seconds (model training)
- Subsequent predictions take ~5 seconds
- Dashboard loads instantly after first use

---

## 🎓 Learning Resources

**Understanding the Code:**
- `src/predictor.py` - Main prediction pipeline
- `src/models/timing_model.py` - Prophet forecasting
- `src/models/content_model.py` - Claude API integration
- `src/api/main.py` - FastAPI endpoints
- `dashboard/app.py` - Streamlit UI

**Next Skills to Learn:**
1. Neural Temporal Point Processes (EasyTPP)
2. LLM Fine-tuning (QLoRA, LoRA)
3. RAG Systems (ChromaDB, LangChain)
4. MLOps (Prefect, Evidently)

---

## 📊 Expected Performance

**Timing Prediction:**
- MAE: 8-12 hours (baseline Prophet)
- Can be improved to 4-6 hours with Neural TPP

**Content Generation:**
- BERTScore: 0.60-0.70 (few-shot Claude)
- Can be improved to 0.70-0.80 with fine-tuning

**Cost:**
- **Development:** ~$10-15/month (just Claude API)
- **Production:** ~$20-25/month (add data collection)

---

## ✅ Success Checklist

After completing this guide, you should have:

- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] `.env` file configured with API key
- [ ] Database initialized with sample data
- [ ] Models trained and making predictions
- [ ] API server running on localhost:8000
- [ ] Dashboard running on localhost:8501
- [ ] Made at least one successful prediction
- [ ] Viewed prediction in dashboard
- [ ] Tested API endpoints in browser

---

## 🎉 You're Ready!

You now have a **working ML prediction system** with:
- Real ML models
- Production-ready API
- Beautiful dashboard
- Automated data collection
- Ready to deploy

**Time to completion:** ~20-30 minutes
**Next milestone:** Deploy to production (another ~30 minutes)

Questions? Check the main README or create an issue on GitHub!
