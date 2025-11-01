# 🚀 Deployment Guide

Complete guide for deploying your MVP to production.

## Overview

We'll deploy three components:
1. **API** → Render.com (free tier)
2. **Dashboard** → Streamlit Cloud (free)
3. **Database** → Render PostgreSQL or SQLite

**Total Cost:** $0-7/month (free for 90 days, then optional $7/month for persistent database)

---

## Part 1: Deploy API to Render.com

### Step 1: Prepare Repository

```bash
# Make sure all changes are committed
git add .
git commit -m "Add production deployment config"
git push origin main
```

### Step 2: Create Render Account

1. Go to https://render.com
2. Click "Get Started"
3. Sign up with your GitHub account
4. Authorize Render to access your repositories

### Step 3: Deploy from Blueprint

1. Click "**New +**" → "**Blueprint**"
2. Connect your repository
3. Render will detect `render.yaml` and show you:
   - `trump-prediction-api` (Web Service)
   - `trump-prediction-collector` (Background Worker)
4. Click "**Apply**"

### Step 4: Configure Environment Variables

For **trump-prediction-api**:
1. Go to service → "Environment"
2. Add these variables:

```
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
DATABASE_URL=sqlite:///./data/trump_predictions.db
ENVIRONMENT=production
```

For **trump-prediction-collector**:
1. Go to service → "Environment"
2. Add same variables plus:

```
RUN_CONTINUOUS=true
APIFY_API_TOKEN=your-token-here (optional)
SCRAPECREATORS_API_KEY=your-key-here (optional)
```

### Step 5: Deploy!

1. Click "**Manual Deploy**" → "**Deploy latest commit**"
2. Wait 3-5 minutes for build to complete
3. Your API will be live at: `https://trump-prediction-api.onrender.com`

### Test Your API

```bash
# Health check
curl https://trump-prediction-api.onrender.com/health

# Make prediction
curl -X POST https://trump-prediction-api.onrender.com/predict

# View docs
# Open: https://trump-prediction-api.onrender.com/docs
```

---

## Part 2: Deploy Dashboard to Streamlit Cloud

### Step 1: Prepare Streamlit Config

Create `dashboard/.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[server]
headless = true
port = 8501
enableCORS = false
```

Commit and push:
```bash
git add dashboard/.streamlit/config.toml
git commit -m "Add Streamlit config"
git push
```

### Step 2: Create Streamlit Account

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "**New app**"

### Step 3: Configure App

1. **Repository:** Select your repo
2. **Branch:** main
3. **Main file path:** `dashboard/app.py`
4. Click "**Advanced settings**"

Add to `secrets.toml`:
```toml
# Leave empty for now - dashboard reads from database
```

5. Click "**Deploy!**"

### Step 4: Access Dashboard

Your dashboard will be live at:
`https://share.streamlit.io/yourusername/trump-prediction-mvp`

---

## Part 3: Optional - PostgreSQL Database

### For Persistent Storage (Recommended for Production)

#### Option A: Render PostgreSQL (Free for 90 days)

1. In Render Dashboard, click "**New +**" → "**PostgreSQL**"
2. Name: `trump-prediction-db`
3. Plan: Free
4. Click "**Create Database**"

5. Copy the "**External Database URL**"
6. Update both services' `DATABASE_URL` to this URL

#### Option B: Neon PostgreSQL (Always Free)

1. Go to https://neon.tech
2. Sign up (free tier: 0.5GB storage, 1 compute)
3. Create new project: "Trump Predictions"
4. Copy connection string
5. Update `DATABASE_URL` in Render

---

## Part 4: Set Up Custom Domain (Optional)

### For API (using Render)

1. Buy domain from Namecheap, Google Domains, etc.
2. In Render, go to your service → "Settings"
3. Add custom domain: `api.yourdomain.com`
4. Add DNS records as shown by Render

### For Dashboard (using Streamlit)

Streamlit Cloud doesn't support custom domains on free tier.
Upgrade to Teams ($250/month) or self-host.

---

## Part 5: Monitoring & Maintenance

### Set Up Uptime Monitoring

**Option 1: UptimeRobot (Free)**
1. Go to https://uptimerobot.com
2. Add monitors for:
   - API: `https://your-api.onrender.com/health`
   - Dashboard: `https://your-dashboard.streamlit.app`
3. Get alerts via email

**Option 2: Better Uptime (Free)**
1. Go to https://betteruptime.com
2. More features, prettier dashboard
3. Integrates with Slack

### View Logs

**Render:**
- Go to service → "Logs" tab
- Real-time streaming logs
- Can search and filter

**Streamlit:**
- Go to app → "Manage app" → "Logs"
- Shows errors and prints

### Update Models

```bash
# SSH into Render (not available on free tier)
# Or use GitHub Actions to retrain automatically

# Create .github/workflows/retrain.yml
```

---

## Part 6: Continuous Deployment

### Automatic Deployments

Already configured! Every push to `main` triggers:
1. Render rebuilds and redeploys API
2. Render restarts data collector
3. Streamlit Cloud updates dashboard

### Manual Control

To **disable** auto-deploy:
- Render: Service → Settings → Build & Deploy → Disable "Auto-Deploy"
- Streamlit: App → Settings → Disable "Always rerun"

---

## Part 7: Cost Optimization

### Free Tier Limits

**Render Free Tier:**
- 750 hours/month (enough for 1 service 24/7)
- Spins down after 15 min inactivity
- Restarts in ~30 seconds on request

**Streamlit Cloud Free:**
- Unlimited public apps
- 1GB RAM per app
- Community support

### Staying Under Budget

**To stay at $0/month:**
- Use SQLite (data stored on disk)
- Use Render free tier (spins down when idle)
- Only run collector when needed

**To spend $7/month:**
- Add Render PostgreSQL for persistence
- Better for production use

**To spend $15-20/month:**
- Add Claude API costs (~$10-15 for testing)
- All other services still free

---

## Part 8: Troubleshooting

### API Won't Start

**Check logs:**
```bash
# In Render, go to Logs tab
# Common issues:
```

1. **Missing environment variables**
   - Solution: Add in Render Dashboard → Environment

2. **Database connection failed**
   - Solution: Check DATABASE_URL format

3. **Port binding error**
   - Solution: Use `$PORT` (Render sets this automatically)

### Dashboard Shows Errors

1. **Can't connect to API**
   - Update API URL in `dashboard/app.py`
   - Make sure API is deployed and running

2. **Database empty**
   - Run data collector first
   - Or run `setup_and_test.py` locally then deploy

### Collector Not Running

1. **Check Render Worker logs**
2. **Verify RUN_CONTINUOUS=true**
3. **Ensure API keys are set**

---

## Part 9: Performance Tuning

### Reduce API Response Time

```python
# Add caching in src/api/main.py
from functools import lru_cache

@lru_cache(maxsize=128)
def get_cached_prediction():
    # Cache predictions for 5 minutes
    pass
```

### Optimize Database Queries

```python
# Add indexes in src/data/database.py
Index('idx_created_at', Post.created_at)
Index('idx_predicted_at', Prediction.predicted_at)
```

### Reduce Memory Usage

```python
# Quantize models
import pickle
import gzip

# Save compressed models
with gzip.open('model.pkl.gz', 'wb') as f:
    pickle.dump(model, f)
```

---

## Part 10: Scaling Up

### When You're Ready to Scale

**Upgrade to Paid Tier ($7-25/month):**
- Persistent PostgreSQL database
- No spin-down delays
- More CPU/RAM
- Background workers

**Add Features:**
- Email notifications when predictions ready
- Webhook support for integrations
- Historical accuracy tracking
- A/B testing different models

**Advanced:**
- Deploy to AWS/GCP with Terraform
- Add Redis for caching
- Implement model versioning
- Set up Prometheus monitoring

---

## 📊 Deployment Checklist

Before going live:

- [ ] All code committed and pushed
- [ ] Environment variables set in Render
- [ ] API deployed and responding to /health
- [ ] Database initialized with data
- [ ] Data collector running
- [ ] Dashboard deployed to Streamlit
- [ ] Uptime monitoring configured
- [ ] Tested end-to-end: make prediction via API
- [ ] Tested dashboard: view predictions
- [ ] Documentation updated with live URLs

After going live:

- [ ] Monitor logs for first 24 hours
- [ ] Test from different locations/devices
- [ ] Share with friends for feedback
- [ ] Add to your portfolio
- [ ] Write blog post about the experience

---

## 🎉 Success!

You now have a **production ML system** running in the cloud!

**Your URLs:**
- API: `https://trump-prediction-api.onrender.com`
- Docs: `https://trump-prediction-api.onrender.com/docs`
- Dashboard: `https://[your-app].streamlit.app`

**Share your project:**
- Add to LinkedIn
- Post on Twitter/X
- Submit to Show HN (Hacker News)
- Add to your resume

**Total deployment time:** ~30-45 minutes
**Total cost:** $0-7/month

Questions? Check the main README or create an issue!
