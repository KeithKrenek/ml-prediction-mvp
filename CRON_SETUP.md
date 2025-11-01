# 🔄 Cron Job Configuration Guide

## Overview

The data collector runs as a **Render.com Cron Job** (free tier) instead of a background worker. This saves $7/month while providing sufficient data collection frequency.

---

## How It Works

### Execution Flow

```
Every 30 minutes:
├── Render triggers cron_collect.py
├── Script initializes DataCollector
├── Tries data sources in order:
│   ├── 1. ScrapeCreators API
│   ├── 2. Apify Scraper
│   └── 3. GitHub Archive (fallback)
├── Saves new posts to database
└── Exits with status code
```

### Schedule

**Default:** `*/30 * * * *` (every 30 minutes)

**Cron syntax:**
```
┌───────────── minute (0 - 59)
│ ┌───────────── hour (0 - 23)
│ │ ┌───────────── day of month (1 - 31)
│ │ │ ┌───────────── month (1 - 12)
│ │ │ │ ┌───────────── day of week (0 - 6) (Sunday to Saturday)
│ │ │ │ │
* * * * *
```

**Common alternatives:**
- Every 15 minutes: `*/15 * * * *`
- Every hour: `0 * * * *`
- Every 2 hours: `0 */2 * * *`
- Business hours only: `0 9-17 * * 1-5` (9 AM - 5 PM, Mon-Fri)

---

## Configuration Files

### 1. render.yaml

```yaml
services:
  - type: cron
    name: trump-prediction-collector
    schedule: "*/30 * * * *"  # ← Change here for different frequency
    startCommand: python scripts/cron_collect.py
```

### 2. scripts/cron_collect.py

**Purpose:** Entry point for Cron Job
- Runs one collection cycle
- Logs to stdout (visible in Render logs)
- Exits cleanly with status code

**Key features:**
- ✅ Proper error handling
- ✅ Duration tracking
- ✅ Success/failure reporting
- ✅ Graceful exit

### 3. src/data/collector.py

**Purpose:** Core collection logic
- `collect_once()` method runs one cycle
- Tries multiple data sources with fallback
- Handles rate limiting and errors
- Saves new posts to database

---

## Environment Variables

### Required for Production

```bash
# In Render Dashboard → Cron Job → Environment
ANTHROPIC_API_KEY=sk-ant-...        # For content model (API access)
DATABASE_URL=sqlite:///./data/...   # Database connection
```

### Optional (for real-time collection)

```bash
APIFY_API_TOKEN=apify_api_...       # $5/month free tier
SCRAPECREATORS_API_KEY=sc_...       # Pay as you go
```

**Without API keys:** Uses GitHub archive only (historical data up to Oct 2024)

---

## Monitoring & Logs

### View Logs in Render

1. Go to Render Dashboard
2. Click on "trump-prediction-collector" (Cron Job)
3. Click "Logs" tab

**What you'll see:**
```
2025-11-01 10:00:00 | INFO     | Data Collection Cron Job Starting
2025-11-01 10:00:00 | INFO     | Trying source: scrapecreators
2025-11-01 10:00:05 | SUCCESS  | Fetched 10 posts from scrapecreators
2025-11-01 10:00:06 | SUCCESS  | Saved 3 new posts from scrapecreators
2025-11-01 10:00:06 | SUCCESS  | Cron job completed successfully!
2025-11-01 10:00:06 | INFO     | New posts collected: 3
2025-11-01 10:00:06 | INFO     | Duration: 6.24 seconds
```

### Check Execution History

**In Render Dashboard:**
- "Cron Jobs" section shows last runs
- Green checkmark = Success (exit code 0)
- Red X = Failed (exit code 1)
- Click on run to see logs

---

## Cost Comparison

### Background Worker (OLD)
```
Render Starter Plan: $7/month
Runs continuously: 24/7
Uses: ~730 hours/month
```

### Cron Job (NEW ✅)
```
Render Free Tier: $0/month
Runs periodically: 30-min intervals
Uses: ~0.5 hours/month (48 runs × ~40 seconds each)
```

**Savings:** $7/month = $84/year 💰

---

## Testing Locally

### Run Once
```bash
python scripts/cron_collect.py
```

### Simulate Cron Schedule (for testing)
```bash
# Install schedule package (already in requirements.txt)
pip install schedule

# Run collector every 30 minutes locally
python src/data/collector.py
# (This uses the continuous mode with schedule library)
```

---

## Deployment Checklist

Before deploying:
- [ ] Update `render.yaml` with desired schedule
- [ ] Commit and push changes
- [ ] Set environment variables in Render
- [ ] Deploy via Render Blueprint

After deployment:
- [ ] Check Cron Job logs for first run
- [ ] Verify posts are being saved to database
- [ ] Confirm no errors in logs
- [ ] Test API to see new posts: `GET /posts/recent`

---

## Troubleshooting

### Cron Job Not Running

**Check 1: Verify service is created**
- Go to Render Dashboard
- Should see "trump-prediction-collector" with type "Cron"

**Check 2: Check schedule syntax**
```yaml
schedule: "*/30 * * * *"  # Correct
schedule: "every 30 minutes"  # Wrong - must use cron syntax
```

**Check 3: View logs for errors**
- Cron service → Logs
- Look for Python errors or API failures

**Check 4: Manually trigger**
- Cron service → "Manual Deploy" → "Deploy latest commit"
- This runs immediately for testing

### No New Posts Collected

**Scenario 1: No API keys configured**
```
Solution: Add APIFY_API_TOKEN or SCRAPECREATORS_API_KEY
OR accept historical data only from GitHub archive
```

**Scenario 2: Trump hasn't posted recently**
```
Solution: This is expected! He doesn't post continuously.
Check database to see last post time.
```

**Scenario 3: API rate limits**
```
Check logs for "429 Too Many Requests"
Solution: Reduce cron frequency or upgrade API plan
```

### Database Connection Errors

**Error: "database is locked"**
```
Cause: SQLite doesn't handle concurrent writes well
Solution: 
1. Use PostgreSQL for production (see DEPLOYMENT.md)
2. Or keep SQLite but ensure API doesn't write during cron run
```

**Fix: Add PostgreSQL**
```yaml
# In render.yaml, uncomment:
databases:
  - name: trump-prediction-db
    plan: free  # Free for 90 days
```

Then update DATABASE_URL in both services:
```
DATABASE_URL=postgresql://user:pass@host/db
```

---

## Optimization Tips

### Reduce Frequency (Lower Costs)

Trump typically posts 2-10 times per day. Collection frequency options:

**Conservative (recommended):**
```yaml
schedule: "0 * * * *"  # Every hour (24 runs/day)
```

**Balanced (default):**
```yaml
schedule: "*/30 * * * *"  # Every 30 min (48 runs/day)
```

**Aggressive:**
```yaml
schedule: "*/15 * * * *"  # Every 15 min (96 runs/day)
```

### Smart Scheduling

**Peak hours only:**
```yaml
# Every 30 min during 6 AM - 11 PM ET
schedule: "*/30 6-23 * * *"
```

**Weekdays only:**
```yaml
# Every hour, Mon-Fri
schedule: "0 * * * 1-5"
```

### Conditional Collection

Modify `scripts/cron_collect.py` to check last post time:

```python
from datetime import datetime, timedelta

# Skip if last post was < 1 hour ago
session = get_session()
last_post = session.query(Post).order_by(Post.created_at.desc()).first()

if last_post and (datetime.now() - last_post.created_at) < timedelta(hours=1):
    logger.info("Recent post exists, skipping collection")
    sys.exit(0)
```

---

## Monitoring & Alerts

### Set Up Uptime Monitoring

**UptimeRobot (Free):**
1. Create HTTP(s) monitor
2. URL: Your API health endpoint
3. Alert if down > 5 minutes

**Better Uptime:**
- More features, prettier UI
- Slack integration

### Email Alerts

Add to `scripts/cron_collect.py`:

```python
if new_posts > 0:
    send_email(
        subject=f"New Trump post detected! ({new_posts} posts)",
        body=f"Collected at {datetime.now()}"
    )
```

---

## Advanced: Webhook Trigger

Instead of time-based cron, trigger on actual events:

1. Use Render Web Service instead of Cron
2. Create webhook endpoint:

```python
@app.post("/webhook/collect")
async def trigger_collection():
    collector = DataCollector()
    new_posts = collector.collect_once()
    return {"new_posts": new_posts}
```

3. Call from external monitoring service when post detected

---

## Summary

### Current Setup
- ✅ Cron Job runs every 30 minutes
- ✅ Completely free (no paid tier needed)
- ✅ Sufficient for Trump's posting frequency
- ✅ Easy to adjust schedule in render.yaml

### Key Files
- `render.yaml` - Cron schedule configuration
- `scripts/cron_collect.py` - Execution entry point
- `src/data/collector.py` - Collection logic

### Cost
- **$0/month** (vs $7/month for background worker)
- Only Claude API costs ($10-15/month)

---

## Quick Reference

**View logs:**
```
Render Dashboard → Cron Job → Logs
```

**Change frequency:**
```
Edit render.yaml → Update schedule → git push
```

**Manual run:**
```
Render Dashboard → Cron Job → Manual Deploy
```

**Test locally:**
```bash
python scripts/cron_collect.py
```

---

For more details, see:
- `DEPLOYMENT.md` - Full deployment guide
- `render.yaml` - Configuration file
- `scripts/cron_collect.py` - Cron script
- `src/data/collector.py` - Collection logic
