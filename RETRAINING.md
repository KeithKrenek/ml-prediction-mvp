# Automated Model Retraining

This document describes the comprehensive automated model retraining system that continuously improves prediction accuracy by training on new data, evaluating performance, and automatically promoting better models.

## Overview

The retraining system provides:

1. **Automated Weekly Retraining** - Models retrained on schedule with latest data
2. **Model Versioning** - Complete version history and metadata tracking
3. **Performance Evaluation** - Rigorous test set evaluation with multiple metrics
4. **Automatic Promotion** - Better models automatically deployed to production
5. **Rollback Capability** - Revert to previous versions if needed
6. **API Management** - Full control via REST API endpoints

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   RETRAINING PIPELINE                        │
│                                                              │
│  1. Load Latest Data      → Database (new posts)            │
│  2. Train/Test Split      → 80/20 split by time             │
│  3. Train New Model       → Prophet/Claude API              │
│  4. Evaluate Performance  → Test set metrics                │
│  5. Compare with Prod     → Score comparison                │
│  6. Auto-Promote          → If >2% better                   │
│  7. Archive Old Versions  → Keep last 10 versions           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Database Schema

### ModelVersion Table

Tracks all trained model versions:

| Field | Type | Description |
|-------|------|-------------|
| `version_id` | String | Unique ID (e.g., prophet_timing_20251106_120000_a3f2) |
| `model_type` | String | 'timing' or 'content' |
| `algorithm` | String | 'prophet', 'claude_api', etc. |
| `trained_at` | DateTime | Training timestamp |
| `status` | String | 'trained', 'active', 'archived', 'failed' |
| `is_production` | Boolean | Currently in production |
| `num_training_samples` | Integer | Training data size |
| `training_duration_seconds` | Float | Training time |
| `hyperparameters` | JSON | Model configuration |
| `file_path` | String | Path to saved model file |

### TrainingRun Table

Tracks individual training runs with metrics:

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | String | Unique run ID |
| `model_version_id` | String | Links to ModelVersion |
| `status` | String | 'running', 'completed', 'failed' |
| `test_mae_hours` | Float | Mean absolute error (timing) |
| `test_within_6h_accuracy` | Float | % predictions within 6 hours |
| `promoted_to_production` | Boolean | Was this model promoted? |
| `improvement_percentage` | Float | vs previous model |
| `promotion_reason` | Text | Why promoted/rejected |

### ModelEvaluation Table

Stores detailed evaluation results:

| Field | Type | Description |
|-------|------|-------------|
| `model_version_id` | String | Model being evaluated |
| `mae_hours` | Float | Mean absolute error |
| `within_6h_accuracy` | Float | % within 6 hours |
| `within_12h_accuracy` | Float | % within 12 hours |
| `within_24h_accuracy` | Float | % within 24 hours |
| `overall_score` | Float | Composite score (0-1) |
| `predictions_json` | JSON | Sample predictions for analysis |

---

## Configuration

Edit `config/config.yaml`:

```yaml
retraining:
  enabled: true  # Enable/disable retraining
  schedule_days: 7  # Retrain every N days

  # Data configuration
  test_split: 0.2  # 20% test set
  min_training_samples: 50  # Minimum data required

  # Evaluation
  evaluation_max_predictions: 20  # Predictions to make during eval

  # Auto-promotion
  auto_promote: true  # Automatically promote better models
  min_improvement_threshold: 2.0  # Minimum 2% improvement required

  # Version management
  keep_versions: 10  # Keep 10 most recent versions
  archive_old_versions: true  # Archive old versions
```

---

## Deployment

### Render.com Cron Job (Recommended)

The retraining cron job is configured in `render.yaml`:

```yaml
- type: cron
  name: trump-model-retrainer
  runtime: python
  plan: starter
  schedule: "0 3 * * 0"  # Every Sunday at 3:00 AM UTC
  buildCommand: pip install -r requirements.txt
  startCommand: python scripts/cron_retrain.py
```

**Schedule Options:**

| Schedule | Cron Expression | Description |
|----------|----------------|-------------|
| Weekly (default) | `0 3 * * 0` | Every Sunday at 3am |
| Bi-weekly | `0 3 */14 * *` | Every 14 days at 3am |
| Monthly | `0 3 1 * *` | 1st of month at 3am |
| Daily (testing) | `0 3 * * *` | Every day at 3am |

### Local Testing

Run retraining manually:

```bash
# Make sure you have enough data
python -c "from src.data.database import get_session, Post; print(f'{get_session().query(Post).count()} posts')"

# Run retraining
python scripts/retrain_models.py
```

Expected output:
```
============================================================
TIMING MODEL RETRAINING
============================================================
Loading and splitting data...
Loaded 373 posts from 2022-02-21 to 2024-10-31
Split data: 298 train, 75 test
Training new timing model on 298 samples...
Model training complete!
Trained model saved: models/timing/prophet_timing_20251106_120000_a3f2.pkl
Model version registered: prophet_timing_20251106_120000_a3f2
Evaluating model on test set...
MAE: 8.45h | Within 6h: 62.5% | Overall score: 0.7123
Checking if new model should be promoted...
✓ Model promoted to production! Reason: New model is 5.2% better
============================================================
TIMING MODEL RETRAINING COMPLETE
============================================================
```

---

## API Endpoints

### GET /models/production

Get current production models:

```bash
curl http://localhost:8000/models/production
```

Response:
```json
{
  "timing_model": {
    "version_id": "prophet_timing_20251106_120000_a3f2",
    "algorithm": "prophet",
    "trained_at": "2025-11-06T12:00:00Z",
    "promoted_at": "2025-11-06T12:05:00Z",
    "num_training_samples": 298
  },
  "content_model": {
    "version_id": "claude_api_content_20251106_120530_b4e9",
    "algorithm": "claude_api",
    "trained_at": "2025-11-06T12:05:30Z",
    "promoted_at": "2025-11-06T12:07:00Z",
    "num_training_samples": 298
  }
}
```

### GET /models/versions

List all model versions:

```bash
curl "http://localhost:8000/models/versions?model_type=timing&limit=5"
```

Response:
```json
{
  "versions": [
    {
      "version_id": "prophet_timing_20251106_120000_a3f2",
      "model_type": "timing",
      "algorithm": "prophet",
      "trained_at": "2025-11-06T12:00:00Z",
      "status": "active",
      "is_production": true,
      "num_training_samples": 298,
      "training_duration_seconds": 12.5
    },
    ...
  ],
  "count": 5
}
```

### GET /models/training-history

View recent training runs:

```bash
curl "http://localhost:8000/models/training-history?limit=10"
```

Response:
```json
{
  "training_runs": [
    {
      "run_id": "run_20251106_120000_a3f2",
      "model_version_id": "prophet_timing_20251106_120000_a3f2",
      "started_at": "2025-11-06T12:00:00Z",
      "completed_at": "2025-11-06T12:03:45Z",
      "duration_seconds": 225.3,
      "status": "completed",
      "test_mae_hours": 8.45,
      "test_within_6h_accuracy": 0.625,
      "promoted_to_production": true,
      "promotion_reason": "New model is 5.2% better"
    }
  ],
  "count": 10
}
```

### POST /models/retrain

Manually trigger retraining:

```bash
curl -X POST http://localhost:8000/models/retrain
```

Response:
```json
{
  "status": "started",
  "message": "Model retraining started in background",
  "note": "Check /models/training-history for progress"
}
```

### POST /models/promote/{version_id}

Manually promote a model:

```bash
curl -X POST "http://localhost:8000/models/promote/prophet_timing_20251106_120000_a3f2"
```

Response:
```json
{
  "status": "success",
  "message": "Model prophet_timing_20251106_120000_a3f2 promoted to production",
  "version_id": "prophet_timing_20251106_120000_a3f2"
}
```

### GET /models/compare/{version_a}/{version_b}

Compare two models:

```bash
curl "http://localhost:8000/models/compare/prophet_timing_OLD/prophet_timing_NEW"
```

Response:
```json
{
  "version_a": {
    "version_id": "prophet_timing_OLD",
    "score": 0.6753
  },
  "version_b": {
    "version_id": "prophet_timing_NEW",
    "score": 0.7123
  },
  "winner": "prophet_timing_NEW",
  "improvement_percentage": 5.2,
  "recommendation": "promote",
  "reason": "New model is 5.2% better"
}
```

---

## Evaluation Metrics

### Timing Model Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **MAE (hours)** | Mean absolute error | < 12 hours |
| **RMSE (hours)** | Root mean squared error | < 16 hours |
| **Within 6h** | % predictions within 6 hours | > 50% |
| **Within 12h** | % predictions within 12 hours | > 70% |
| **Within 24h** | % predictions within 24 hours | > 85% |
| **Overall Score** | Composite score (0-1) | > 0.65 |

**Overall Score Formula:**
```
overall_score = 0.7 * within_6h_accuracy + 0.3 * (1 - normalized_mae)
```

### Content Model Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Length Similarity** | Length match with actual | > 0.7 |
| **Word Overlap** | Shared words (Jaccard) | > 0.3 |
| **Overall Score** | Average of metrics | > 0.5 |

*Note: BERTScore and BLEU can be added for more sophisticated evaluation*

---

## Auto-Promotion Logic

A new model is automatically promoted if:

1. ✅ Evaluation completes successfully
2. ✅ Overall score is at least 2% better than production
3. ✅ `auto_promote: true` in config

Example decision logic:

```
Current Production Score: 0.6753
New Model Score: 0.7123

Improvement = ((0.7123 - 0.6753) / 0.6753) * 100 = 5.2%

Decision: PROMOTE (5.2% > 2.0% threshold)
```

If improvement < 2%:
- Model is registered but NOT promoted
- Manual review recommended
- Can be promoted via API if desired

---

## Model Files and Storage

### Directory Structure

```
models/
├── timing/
│   ├── prophet_timing_20251106_120000_a3f2.pkl  # Versioned
│   ├── prophet_timing_20251103_030000_b7d4.pkl
│   └── ...
├── content/
│   ├── claude_api_content_20251106_120530_b4e9.pkl
│   └── ...
├── timing_model.pkl  # Symlink to production timing model
└── content_model.pkl  # Symlink to production content model
```

### Model File Contents

**Timing Model (Prophet):**
```python
{
    'model': <Prophet model object>,
    'config': {...},  # Hyperparameters
    'last_trained': datetime,
    'features_df': <training data>
}
```

**Content Model (Claude API):**
```python
{
    'model_type': 'claude_api',
    'example_posts': [...],  # Sample posts
    'config': {...},
    'version_id': '...'
}
```

---

## Monitoring and Alerts

### Render.com Dashboard

1. Go to https://dashboard.render.com
2. Select `trump-model-retrainer` service
3. View logs for:
   - Training start/completion
   - Evaluation metrics
   - Promotion decisions
   - Any errors

### Key Log Messages

**Success:**
```
✓ Model promoted to production! Reason: New model is 5.2% better
TIMING MODEL RETRAINING COMPLETE
Duration: 225.3 seconds
```

**No Promotion:**
```
✗ Model not promoted. Reason: New model is 0.8% better (below 2.0% threshold)
Manual review recommended
```

**Failure:**
```
ERROR: Model training failed: Insufficient data
Collect more data before retraining
```

---

## Troubleshooting

### Insufficient Data

**Error:** "Insufficient data for training: 35 posts"

**Solution:** Collect more data. Minimum 50 posts required.

```bash
# Check data count
python -c "from src.data.database import get_session, Post; print(get_session().query(Post).count())"

# If low, run data collection
python scripts/cron_collect.py
```

---

### Retraining Takes Too Long

**Symptom:** Retraining exceeds 10-minute timeout

**Solutions:**

1. Reduce `evaluation_max_predictions` in config:
   ```yaml
   retraining:
     evaluation_max_predictions: 10  # Reduce from 20
   ```

2. Increase Render timeout (if possible)

3. Run evaluation in parallel (future enhancement)

---

### Model Not Promoted

**Symptom:** New model trained but not promoted

**Check:** Was improvement sufficient?

```bash
curl http://localhost:8000/models/training-history | jq '.[0]'
```

Look for `improvement_percentage` and `promotion_reason`

**Manual Promotion:**
```bash
# Get version ID
curl http://localhost:8000/models/versions | jq '.versions[0].version_id'

# Promote manually
curl -X POST "http://localhost:8000/models/promote/VERSION_ID"
```

---

### Database Migration

**If you added this system to an existing deployment:**

```bash
# Initialize new tables
python -c "from src.data.database import init_db; init_db()"

# Verify tables created
python -c "from src.data.database import get_session, ModelVersion; print(get_session().query(ModelVersion).count())"
```

---

## Best Practices

### 1. Data Collection

- Ensure continuous data collection is running
- Need at least 100+ posts for good retraining
- More data = better model performance

### 2. Monitoring

- Check training history weekly
- Review promotion decisions
- Monitor MAE and accuracy trends

### 3. Manual Review

- Review models with < 2% improvement
- Check evaluation metrics for anomalies
- Test manually before promoting edge cases

### 4. Rollback

If a promoted model performs poorly:

```bash
# List versions
curl http://localhost:8000/models/versions?model_type=timing

# Promote previous version
curl -X POST "http://localhost:8000/models/promote/PREVIOUS_VERSION_ID"
```

### 5. Archiving

Old versions are automatically archived (keep last 10):
- Archived models remain in database
- Model files can be deleted to save space
- Can be restored if needed

---

## Performance Expectations

### Timing Model

| Metric | Baseline (Random) | Good Model | Excellent Model |
|--------|------------------|------------|----------------|
| MAE | ~24 hours | 8-12 hours | < 8 hours |
| Within 6h | ~25% | 50-65% | > 65% |
| Within 24h | ~50% | 85-90% | > 90% |

### Content Model

| Metric | Baseline | Good Model | Excellent Model |
|--------|----------|------------|----------------|
| Similarity | 0.3 | 0.5-0.6 | > 0.6 |

---

## Future Enhancements

Planned improvements:

1. ✅ Neural Temporal Point Process (timing)
2. ✅ Fine-tuned GPT-2/Phi-3 (content)
3. ✅ BERTScore evaluation
4. ✅ A/B testing framework
5. ✅ Confidence calibration
6. ✅ Feature importance analysis
7. ✅ Hyperparameter optimization

---

## Summary

The automated retraining system provides:

- ✅ **Zero-maintenance** model improvement
- ✅ **Production-ready** with cron automation
- ✅ **Safe** auto-promotion with rollback
- ✅ **Transparent** with full API access
- ✅ **Monitored** with comprehensive logging
- ✅ **Free** on Render.com

Models automatically improve as more data is collected!
