# Trump Truth Social Post Prediction MVP

A real-time ML pipeline that predicts **when** Trump will post on Truth Social and **what** he'll say.

## 🎯 Quick Start (5 Minutes to First Prediction!)

### Prerequisites
- Python 3.10 or higher
- Anthropic API key ([Get one here](https://console.anthropic.com/))

### Installation

1. **Clone or navigate to the project**
```bash
cd trump-prediction-mvp
```

2. **Install Python dependencies**

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**On Mac/Linux:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Anthropic API key
# ANTHROPIC_API_KEY=your_key_here
```

**Windows users:** You can edit `.env` with Notepad or any text editor.

4. **Download historical data and run first prediction**
```bash
# Download Trump's historical posts
python scripts/download_data.py

# Train models and make first prediction
python src/predictor.py
```

🎉 **That's it!** You should see your first prediction in seconds.

---

## 📁 Project Structure

```
trump-prediction-mvp/
├── config/
│   └── config.yaml              # Model configuration
├── data/
│   ├── raw/                     # Raw data downloads
│   ├── processed/               # Processed datasets
│   └── trump_predictions.db     # SQLite database
├── models/                      # Saved model files
├── src/
│   ├── data/
│   │   └── database.py          # Database models (SQLAlchemy)
│   ├── models/
│   │   ├── timing_model.py      # Prophet-based timing prediction
│   │   └── content_model.py     # Claude-based content generation
│   └── predictor.py             # Main prediction pipeline
├── scripts/
│   └── download_data.py         # Data download script
├── logs/                        # Application logs
├── requirements.txt
└── README.md
```

---

## 🔧 How It Works

### Architecture Overview

```
┌─────────────────┐
│  Historical     │
│  Data (GitHub)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   SQLite DB     │  ← Stores posts and predictions
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Prediction Pipeline             │
│                                         │
│  ┌──────────────┐   ┌──────────────┐  │
│  │   Prophet    │   │  Claude API  │  │
│  │ (Timing)     │   │ (Content)    │  │
│  └──────┬───────┘   └──────┬───────┘  │
│         │                   │          │
│         └─────────┬─────────┘          │
│                   ▼                    │
│         ┌─────────────────┐            │
│         │   Prediction    │            │
│         │  (Time + Text)  │            │
│         └─────────────────┘            │
└─────────────────────────────────────────┘
```

### Models

**Timing Model (Prophet)**
- Predicts when next post will occur
- Uses Facebook Prophet for time series forecasting
- Captures daily/weekly patterns automatically
- ~12-16 hour accuracy (baseline)

**Content Model (Claude API)**
- Generates post content in Trump's style
- Uses few-shot prompting with real examples
- 10 example posts for style learning
- Can be upgraded to fine-tuned model later

---

## 🚀 Usage

### Make a Single Prediction

```bash
python src/predictor.py
```

Output example:
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

### Automated Predictions (NEW! ⏰)

### Fast Smoke Test

Run the lightweight smoke test to ensure imports and DB connectivity work (used by CI):

```bash
python scripts/smoke_test.py
```


The system now supports **automatic prediction scheduling**:

**Option 1: In-Process Scheduler (Development)**
```bash
# Enable scheduling in config/config.yaml:
# scheduling:
#   enabled: true
#   prediction_interval_hours: 6

# Start the API (scheduler starts automatically)
uvicorn src.api.main:app --reload
```

**Option 2: Cron Job (Production - Recommended)**
- Predictions run automatically every 6 hours on Render.com
- Configured in `render.yaml` (already set up)
- Completely free on Render's free tier
- See [SCHEDULING.md](SCHEDULING.md) for full documentation

**Control via API:**
```bash
# Check scheduler status
curl http://localhost:8000/scheduler/status

# Cron health summary (new)
curl http://localhost:8000/health/cron

# Enable/disable scheduler
curl -X POST http://localhost:8000/scheduler/control \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'

# Manually trigger prediction
curl -X POST http://localhost:8000/scheduler/trigger
```

📖 **See [SCHEDULING.md](SCHEDULING.md) for complete scheduling documentation**

### Configuration

Edit `config/config.yaml` to customize model behavior and scheduling intervals.

---

## 💰 Cost Breakdown (MVP)

| Service | Cost | Notes |
|---------|------|-------|
| Claude API | ~$10-20/month | Few predictions daily |
| Database | $0 | SQLite (local) |
| Compute | $0 | Run locally |
| Data Collection | $0 | Historical data (free) |
| **Total** | **$10-20/month** | ✅ Well under budget |

---

## 🎯 Next Steps

1. **Test locally** - Verify predictions work
2. **Add FastAPI** - Create REST API
3. **Deploy** - Push to Render.com
4. **Add real-time collection** - Implement polling
5. **Create dashboard** - Build Streamlit UI

---

## 📝 License

This is an educational/portfolio project. Use responsibly and respect platform terms of service.
