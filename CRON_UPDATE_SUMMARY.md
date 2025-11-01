# 🔄 Cron Job Update Summary

## Changes Made for Render.com Free Tier Compatibility

You correctly identified that Render.com requires a **paid Starter plan** ($7/month) for background workers. I've updated the project to use **Cron Jobs** instead, which are **completely FREE** on Render! ✅

---

## Files Changed/Created

### 1. **render.yaml** (MODIFIED)
**Changed:**
- Service type: `worker` → `cron`
- Added schedule: `*/30 * * * *` (every 30 minutes)
- Removed `RUN_CONTINUOUS` environment variable
- Updated startCommand to use new cron script

**Result:** Both API and collector now deploy on FREE tier!

### 2. **scripts/cron_collect.py** (NEW ✨)
**Purpose:** Dedicated entry point for Cron Jobs
- Runs one collection cycle
- Logs to stdout (visible in Render logs)
- Proper error handling and exit codes
- Duration tracking

**Usage:**
```bash
# Locally
python scripts/cron_collect.py

# On Render
# Runs automatically every 30 minutes
```

### 3. **DEPLOYMENT.md** (UPDATED)
**Updated sections:**
- Deployment instructions mention Cron Job (not background worker)
- Environment variables section (removed RUN_CONTINUOUS)
- Troubleshooting section for Cron Jobs
- Cost comparisons updated

### 4. **CRON_SETUP.md** (NEW ✨)
**Comprehensive guide covering:**
- How Cron Jobs work
- Schedule configuration
- Monitoring and logs
- Cost comparison
- Optimization tips
- Troubleshooting
- Testing locally

### 5. **src/data/collector.py** (NO CHANGES NEEDED ✅)
The `collect_once()` method was already perfect for cron usage!

---

## How to Deploy

### Step 1: Update Your Repository

```bash
# Pull latest changes
git pull

# Or if you're updating manually, replace these files:
# - render.yaml
# - scripts/cron_collect.py (new)
# - DEPLOYMENT.md
# - CRON_SETUP.md (new)
```

### Step 2: Push to GitHub

```bash
git add .
git commit -m "Switch to Cron Job for data collection (free tier)"
git push origin main
```

### Step 3: Deploy to Render

**Option A: Fresh Deploy (Recommended)**
1. Delete existing "trump-prediction-collector" worker service
2. Go to Render → New + → Blueprint
3. Select your repo
4. Render will detect the updated `render.yaml`
5. You'll see:
   - `trump-prediction-api` (Web Service) - FREE ✅
   - `trump-prediction-collector` (Cron Job) - FREE ✅

**Option B: Update Existing**
1. Render auto-deploys on push (if configured)
2. Or manually: Service → Manual Deploy → Deploy latest commit

### Step 4: Set Environment Variables

**For both services:**
```
ANTHROPIC_API_KEY=your-key-here
DATABASE_URL=sqlite:///./data/trump_predictions.db
ENVIRONMENT=production
```

**For collector (optional, for real-time data):**
```
APIFY_API_TOKEN=your-token-here
SCRAPECREATORS_API_KEY=your-key-here
```

---

## Cron Job Configuration

### Default Schedule
**Every 30 minutes:** `*/30 * * * *`

This is perfect because:
- ✅ Trump posts a few times per day (not every minute)
- ✅ Catches new posts within 30 minutes
- ✅ Minimal API usage
- ✅ Completely free

### Alternative Schedules

**Every hour (more conservative):**
```yaml
schedule: "0 * * * *"
```

**Every 15 minutes (more frequent):**
```yaml
schedule: "*/15 * * * *"
```

**Business hours only (9 AM - 5 PM, Mon-Fri):**
```yaml
schedule: "*/30 9-17 * * 1-5"
```

To change: Edit `render.yaml` → commit → push

---

## Cost Comparison

### Before (Background Worker)
```
Render Starter Plan: $7/month
Total: $17-22/month
```

### After (Cron Job) ✅
```
Render Free Tier: $0/month
Total: $10-15/month (Claude API only!)
```

**You save $84/year!** 💰

---

## How It Works

```
┌─────────────────────────────────────────────────────┐
│                  Render.com Cron                     │
│                                                       │
│  Schedule: */30 * * * * (every 30 minutes)          │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│            scripts/cron_collect.py                   │
│                                                       │
│  1. Initialize DataCollector                        │
│  2. Run collect_once()                              │
│  3. Log results                                     │
│  4. Exit with status code                           │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│          src/data/collector.py                       │
│                                                       │
│  Try sources in order:                              │
│  1. ScrapeCreators API                              │
│  2. Apify                                           │
│  3. GitHub Archive (fallback)                       │
│                                                       │
│  Save new posts to database                         │
└─────────────────────────────────────────────────────┘
```

---

## Monitoring

### View Logs

**In Render Dashboard:**
1. Click "trump-prediction-collector" (Cron Job)
2. Go to "Logs" tab
3. See execution history and output

**Example log output:**
```
2025-11-01 10:00:00 | INFO | Data Collection Cron Job Starting
2025-11-01 10:00:05 | SUCCESS | Fetched 10 posts from scrapecreators
2025-11-01 10:00:06 | SUCCESS | Saved 3 new posts
2025-11-01 10:00:06 | INFO | Duration: 6.24 seconds
```

### Execution History

- Green checkmark ✅ = Success (exit code 0)
- Red X ❌ = Failed (exit code 1)
- Click on any run to see full logs

### Manual Trigger (for testing)

**In Render Dashboard:**
1. Go to Cron Job service
2. Click "Manual Deploy"
3. Click "Deploy latest commit"
4. This runs immediately (doesn't wait for schedule)

---

## Testing Locally

### Test the cron script:
```bash
# Activate your environment
venv\Scripts\activate

# Run once
python scripts\cron_collect.py
```

**Expected output:**
```
============================================================
Data Collection Cron Job Starting
Execution time: 2025-11-01 10:00:00
============================================================
Trying source: scrapecreators
Saved 3 new posts from scrapecreators
============================================================
Cron job completed successfully!
New posts collected: 3
Duration: 5.43 seconds
============================================================
```

---

## Troubleshooting

### Cron Job Not Running

**Problem:** Service shows as created but no logs
- **Solution:** Check schedule syntax in render.yaml
- **Check:** Must be valid cron expression: `*/30 * * * *`

**Problem:** Logs show "No API keys configured"
- **Solution:** This is OK! It will use GitHub archive (historical data)
- **To fix:** Add APIFY_API_TOKEN or SCRAPECREATORS_API_KEY

**Problem:** Exit code 1 (failed)
- **Solution:** Click on failed run → View logs → Check error message
- **Common:** Database connection issues, missing dependencies

### No New Posts Collected

**Most likely:** Trump hasn't posted in the last 30 minutes!
- Check database for last post time
- This is expected behavior

**Alternative:** Check API keys are valid
- Test locally: `python scripts\cron_collect.py`

---

## Next Steps

### 1. Deploy with Cron Job
```bash
git add .
git commit -m "Use Cron Job for free tier deployment"
git push origin main
```

### 2. Verify Deployment
- Check both services are "Live" in Render
- Web Service: `https://your-api.onrender.com/health`
- Cron Job: Check logs after first run (within 30 min)

### 3. Monitor First Few Runs
- Watch logs in Render dashboard
- Verify posts are being collected
- Check API `/posts/recent` endpoint

### 4. Celebrate! 🎉
- **FREE** deployment achieved!
- **$7/month saved**
- **Production-ready** system

---

## Summary

### What Changed
- ✅ Background Worker → Cron Job
- ✅ Continuous running → Periodic execution
- ✅ $7/month → $0/month

### What Stayed the Same
- ✅ Same collection logic
- ✅ Same data sources
- ✅ Same API functionality
- ✅ Same quality of data

### Total Cost Now
- **Compute:** $0 (Render free tier)
- **Database:** $0 (SQLite)
- **APIs:** $10-15/month (Claude + optional data collection)
- **Total:** $10-15/month ✅

---

## Documentation

**Read these for more details:**
- `CRON_SETUP.md` - Complete Cron Job guide
- `DEPLOYMENT.md` - Updated deployment instructions
- `render.yaml` - Configuration file

**Test locally:**
```bash
python scripts\cron_collect.py
```

**Questions?** Everything is documented in the files above!

---

Great catch on the free tier limitation! This Cron Job approach is actually **better** than a background worker for this use case:

1. ✅ **Free** - Saves $7/month
2. ✅ **Efficient** - Only runs when needed
3. ✅ **Sufficient** - Trump doesn't post every minute
4. ✅ **Scalable** - Easy to adjust frequency

Happy deploying! 🚀
