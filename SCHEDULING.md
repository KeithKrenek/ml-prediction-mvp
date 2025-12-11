# Automated Prediction Scheduling

This document describes the automated prediction scheduling system that enables the Trump Post Predictor to make predictions automatically at regular intervals.

## Overview

The system provides **two scheduling mechanisms**:

1. **In-Process Scheduler** - APScheduler integrated into the FastAPI application (for development/single-server deployments)
2. **Cron Job Scheduler** - Render.com cron jobs (for production/cloud deployments)

Both approaches ensure predictions are made automatically without manual intervention.

---

## 1. In-Process Scheduler (APScheduler)

The FastAPI application includes an integrated APScheduler that runs predictions on a configurable schedule.

### How It Works

- **Scheduler**: BackgroundScheduler runs in a separate thread within the API process
- **Trigger**: Configurable interval (default: every 6 hours) or cron expression
- **Queue Management**: Thread lock prevents overlapping prediction jobs
- **Automatic Start**: Scheduler starts when the API starts (if enabled in config)

### Configuration

Edit `config/config.yaml`:

```yaml
scheduling:
  enabled: true  # Enable/disable automatic predictions
  prediction_interval_hours: 6  # Make predictions every 6 hours
  cron_expression: null  # Optional: Use cron instead of interval
  max_concurrent_predictions: 1  # Prevent overlapping jobs
  retry_on_failure: true
  retry_delay_minutes: 30
```

### Control via API

#### Check Scheduler Status

```bash
GET /scheduler/status
```

**Response:**
```json
{
  "enabled": true,
  "running": true,
  "next_run_time": "2025-11-06T18:00:00",
  "prediction_interval_hours": 6.0,
  "cron_expression": null,
  "total_scheduled_predictions": 42,
  "last_scheduled_prediction_time": "2025-11-06T12:00:00"
}
```

#### Enable/Disable Scheduler

```bash
POST /scheduler/control
Content-Type: application/json

{
  "enabled": true  # or false to disable
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Scheduler started",
  "enabled": true,
  "next_run_time": "2025-11-06T18:00:00"
}
```

#### Manually Trigger Prediction

```bash
POST /scheduler/trigger
```

This immediately runs a scheduled prediction without waiting for the next scheduled time.

### Local Development

1. Start the API:
   ```bash
   python -m uvicorn src.api.main:app --reload
   ```

2. The scheduler will start automatically if `scheduling.enabled: true` in config

3. Check logs for:
   ```
   Setting up prediction scheduler...
   Using interval trigger: every 6 hours
   Prediction scheduler started successfully!
   Next scheduled prediction: 2025-11-06T18:00:00
   ```

4. Monitor scheduled predictions in logs:
   ```
   ============================================================
   Starting scheduled prediction job
   ============================================================
   Making scheduled prediction...
   Scheduled prediction completed successfully!
   Prediction ID: 123e4567-e89b-12d3-a456-426614174000
   Predicted time: 2025-11-08T14:30:00
   Overall confidence: 72.6%
   ============================================================
   ```

### Pros and Cons

**Pros:**
- ✅ Runs within the API process (no separate infrastructure)
- ✅ Easy to control via API endpoints
- ✅ Immediate status updates
- ✅ Good for development and testing

**Cons:**
- ❌ Stops when API restarts (Render.com free tier spins down after 15 min)
- ❌ Not suitable for Render.com free tier in production
- ❌ Single point of failure

---

## 2. Cron Job Scheduler (Render.com)

For production deployments on Render.com, use the dedicated cron job service.

### How It Works

- **Service**: Separate Render.com Cron Job service
- **Schedule**: Every 6 hours (00:00, 06:00, 12:00, 18:00 UTC)
- **Script**: `scripts/cron_predict.py` runs independently
- **Cost**: **FREE** on Render.com (cron jobs are free)
- **Reliability**: Runs even when API is spun down

### Configuration

The cron job is defined in `render.yaml`:

```yaml
services:
  - type: cron
    name: trump-prediction-scheduler
    runtime: python
    plan: starter
    schedule: "0 */6 * * *"  # Every 6 hours
    buildCommand: pip install -r requirements.txt
    startCommand: python scripts/cron_predict.py
    envVars:
      - key: ANTHROPIC_API_KEY
        sync: false  # Set in Render dashboard
      - key: DATABASE_URL
        sync: false
    autoDeploy: true
```

### Customizing the Schedule

Edit the `schedule` field in `render.yaml` using standard cron syntax:

| Schedule | Cron Expression | Description |
|----------|----------------|-------------|
| Every 6 hours | `0 */6 * * *` | At 00:00, 06:00, 12:00, 18:00 |
| Every 12 hours | `0 */12 * * *` | At 00:00 and 12:00 |
| Every 3 hours | `0 */3 * * *` | Every 3 hours |
| Twice daily | `0 8,20 * * *` | At 8:00 AM and 8:00 PM |
| Once daily | `0 9 * * *` | At 9:00 AM daily |
| Every hour | `0 * * * *` | At the start of every hour |

After editing, commit and push to GitHub. Render will automatically redeploy the cron job with the new schedule.

### Monitoring Cron Jobs

1. **Render Dashboard**:
   - Go to your Render dashboard
   - Select the `trump-prediction-scheduler` service
   - View logs to see execution history

2. **Logs show**:
   - When the job started
   - Prediction ID generated
   - Predicted time and content
   - Confidence scores
   - Execution duration

Example log output:
```
============================================================
Prediction Cron Job Starting
Execution time: 2025-11-06T12:00:05
============================================================
Initializing predictor...
Using existing trained models
Making prediction...
============================================================
Prediction completed successfully!
Prediction ID: 123e4567-e89b-12d3-a456-426614174000
Predicted time: 2025-11-08T14:30:00
Predicted content: The Fake News Media is at it again! They don't want you to know...
Timing confidence: 75.3%
Content confidence: 70.0%
Overall confidence: 72.6%
Duration: 4.32 seconds
============================================================
```

### Cron Health Endpoint

- The FastAPI service now exposes `GET /health/cron`, summarizing:
  - Last run timestamp & status for each cron job
  - Consecutive failures (useful for alerts)
  - Number of runs in the past 24 hours
- Use it with Render health checks or UptimeRobot to ensure cron jobs are firing before/after upgrading paid plans.

```bash
curl https://trump-prediction-api.onrender.com/health/cron | jq
```

### Testing Cron Script Locally

Test the cron script before deploying:

```bash
# Make sure environment variables are set
export ANTHROPIC_API_KEY=your_key_here
export DATABASE_URL=your_database_url

# Run the cron script
python scripts/cron_predict.py
```

The script should:
1. Initialize the predictor
2. Check for/train models
3. Make a prediction
4. Save to database
5. Log results
6. Exit with code 0 (success)

### Pros and Cons

**Pros:**
- ✅ Runs independently of API (doesn't require API to be up)
- ✅ **FREE** on Render.com
- ✅ Reliable execution even with free tier spin-down
- ✅ Survives API restarts
- ✅ Simple to monitor via Render logs

**Cons:**
- ❌ Requires separate Render service
- ❌ Less flexible control (requires git push to change schedule)
- ❌ Slight delay in status updates (not real-time)

---

## 3. Hybrid Approach (Recommended for Production)

**Best practice**: Use **both** mechanisms:

1. **Cron Job** (primary): Handles scheduled predictions reliably every 6 hours
2. **In-Process Scheduler** (secondary): Allows on-demand control and testing

### Setup

1. **Enable cron job** in Render.com:
   - Deploy with `render.yaml` (already configured)
   - Set environment variables in Render dashboard

2. **Disable in-process scheduler** in production:
   - Edit `config/config.yaml`:
     ```yaml
     scheduling:
       enabled: false  # Disable to avoid duplicate predictions
     ```
   - This prevents duplicate predictions from both systems

3. **Use API endpoints** for manual predictions:
   - `POST /predict` - Make immediate prediction
   - `POST /scheduler/trigger` - Test the scheduled prediction function

### Benefits

- ✅ Reliable automatic predictions (cron job)
- ✅ Manual control when needed (API endpoints)
- ✅ Cost-effective (all free tier)
- ✅ No duplicate predictions

---

## 4. Environment Variables

Both scheduling systems require the same environment variables:

### Required

- `ANTHROPIC_API_KEY` - For Claude API content generation
- `DATABASE_URL` - Database connection string (SQLite or PostgreSQL)

### Optional

- `ENVIRONMENT` - Set to `production` for production mode
- `PYTHON_VERSION` - Python version (default: 3.11.0)

Set these in:
- **Local**: `.env` file
- **Render**: Dashboard → Service → Environment → Environment Variables

---

## 5. Database Persistence

All predictions (manual and scheduled) are saved to the database with:

- `prediction_id` - Unique identifier
- `predicted_at` - When prediction was made
- `predicted_time` - Forecasted post time
- `predicted_content` - Generated content
- `timing_confidence` - Timing model confidence
- `content_confidence` - Content model confidence
- `timing_model_version` - Model used
- `content_model_version` - Model used

This allows:
- Historical tracking of predictions
- Accuracy evaluation (when actual posts arrive)
- Performance monitoring over time

---

## 6. Troubleshooting

### Scheduler Not Starting

**Check logs for:**
```
Prediction scheduling is disabled in configuration
```

**Solution**: Set `scheduling.enabled: true` in `config/config.yaml`

---

### Cron Job Failing

**Check Render logs for:**
```
ANTHROPIC_API_KEY not configured
```

**Solution**: Set `ANTHROPIC_API_KEY` in Render dashboard environment variables

---

### Duplicate Predictions

**Symptom**: Predictions made twice at the same time

**Cause**: Both in-process scheduler and cron job are running

**Solution**: Disable one of them:
- Set `scheduling.enabled: false` in config (disables in-process)
- OR remove cron job from `render.yaml` (disables Render cron)

---

### Predictions Skipped

**Check logs for:**
```
Scheduled prediction skipped - previous job still running
```

**Cause**: Previous prediction hasn't finished yet

**Solution**: This is normal behavior (queue management working correctly). If persistent:
- Increase `prediction_interval_hours` to allow more time
- Check if model training is taking too long
- Verify database connection is fast

---

## 7. Next Steps

With automated scheduling implemented, consider:

1. **Automated Model Retraining** - Retrain models weekly with new data
2. **Prediction Validation** - Automatically match predictions to actual posts
3. **Accuracy Tracking** - Monitor prediction accuracy over time
4. **Alerts/Notifications** - Send notifications when predictions are made
5. **Context Integration** - Integrate real news sources for better predictions

See the main README and implementation roadmap for details.

---

## 8. Summary

| Feature | In-Process Scheduler | Cron Job Scheduler |
|---------|---------------------|-------------------|
| **Runs when API is down** | ❌ No | ✅ Yes |
| **Free on Render.com** | ✅ Yes | ✅ Yes |
| **Control via API** | ✅ Yes | ❌ No (requires git push) |
| **Setup complexity** | ✅ Easy | ⚠️ Moderate |
| **Production ready** | ⚠️ Limited (free tier) | ✅ Yes |
| **Recommended for** | Development, Testing | Production |

**Recommendation**: Use **Cron Job Scheduler** for production deployments on Render.com.
