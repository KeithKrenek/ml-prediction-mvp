# Automated Prediction Validation

This document describes the automated prediction validation system that evaluates the accuracy of Trump Post Predictor by comparing predictions against actual posts.

## Overview

The validation system automatically:

1. **Matches predictions to actual posts** within a configurable time window
2. **Calculates accuracy metrics** for both timing and content predictions
3. **Updates the database** with validation results
4. **Provides interactive visualizations** showing predictions vs reality on a dual timeline

This enables continuous monitoring of model performance and helps identify areas for improvement.

---

## How It Works

### 1. Matching Algorithm

The system matches predictions to actual posts using a **time-window matching approach**:

```python
# For each unvalidated prediction:
1. Find all actual posts within ±window_hours of predicted time
2. Select the closest post by time difference
3. Mark this as the matched post
```

**Default window**: 24 hours (configurable via `validation.matching_window_hours`)

### 2. Timing Evaluation

Once matched, timing accuracy is calculated:

```python
timing_error_hours = abs(predicted_time - actual_time).total_seconds() / 3600

is_timing_correct = timing_error_hours <= timing_threshold_hours  # Default: 6 hours
```

**Metrics tracked**:
- **MAE (Mean Absolute Error)**: Average timing error in hours
- **Within 6h accuracy**: Percentage of predictions within 6 hours
- **Within 12h accuracy**: Percentage within 12 hours
- **Within 24h accuracy**: Percentage within 24 hours

### 3. Content Evaluation

Content similarity is calculated using multiple metrics:

```python
def calculate_content_similarity(predicted, actual):
    # 1. Length similarity (how close in character count)
    length_similarity = 1 - abs(len(predicted) - len(actual)) / max(len(predicted), len(actual))

    # 2. Word overlap (Jaccard similarity)
    pred_words = set(predicted.lower().split())
    actual_words = set(actual.lower().split())
    word_overlap = len(pred_words & actual_words) / len(pred_words | actual_words)

    # 3. Character overlap
    pred_chars = set(predicted.lower())
    actual_chars = set(actual.lower())
    char_overlap = len(pred_chars & actual_chars) / len(pred_chars | actual_chars)

    # 4. Composite similarity (weighted average)
    composite = (length_similarity * 0.2) + (word_overlap * 0.5) + (char_overlap * 0.3)

    return composite
```

**Threshold**: Default `content_threshold = 0.3` (predictions with similarity ≥ 0.3 are considered correct)

### 4. Combined Correctness

A prediction is marked as **correct** if BOTH conditions are met:
- Timing error ≤ 6 hours (configurable)
- Content similarity ≥ 0.3 (configurable)

---

## Database Schema

The validation system uses existing tables with additional fields:

### Prediction Table Updates

```sql
-- Updated by validation process:
actual_post_id          VARCHAR    -- ID of matched post (NULL if unmatched)
actual_time             DATETIME   -- Timestamp of actual post
actual_content          TEXT       -- Content of actual post

-- Evaluation metrics:
timing_error_hours      FLOAT      -- Absolute error in hours
bertscore_f1            FLOAT      -- Reserved for future BERTScore integration
was_correct             BOOLEAN    -- Combined success metric
```

### Example Validation Flow

```
Before validation:
┌─────────────────┬──────────────────┬─────────────────┬──────────────┐
│ prediction_id   │ predicted_time   │ actual_post_id  │ was_correct  │
├─────────────────┼──────────────────┼─────────────────┼──────────────┤
│ pred-123        │ 2025-11-07 14:00 │ NULL            │ NULL         │
└─────────────────┴──────────────────┴─────────────────┴──────────────┘

After validation (actual post at 14:32):
┌─────────────────┬──────────────────┬─────────────────┬─────────────────┬──────────────┐
│ prediction_id   │ predicted_time   │ actual_post_id  │ timing_error_h  │ was_correct  │
├─────────────────┼──────────────────┼─────────────────┼─────────────────┼──────────────┤
│ pred-123        │ 2025-11-07 14:00 │ post-456        │ 0.53            │ TRUE         │
└─────────────────┴──────────────────┴─────────────────┴─────────────────┴──────────────┘
```

---

## Configuration

Edit `config/config.yaml`:

```yaml
validation:
  enabled: true  # Enable automatic validation
  matching_window_hours: 24  # Match predictions to posts within ±24h
  timing_threshold_hours: 6  # Prediction correct if within 6h
  content_threshold: 0.3     # Minimum similarity score for correct content
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `true` | Enable/disable automatic validation |
| `matching_window_hours` | `24` | Time window for matching (±hours) |
| `timing_threshold_hours` | `6` | Max error for "correct" timing |
| `content_threshold` | `0.3` | Min similarity for "correct" content |

**Tuning tips**:
- **Increase `matching_window_hours`** if predictions are far from actual times (e.g., 48h)
- **Decrease `timing_threshold_hours`** for stricter evaluation (e.g., 3h)
- **Increase `content_threshold`** for stricter content evaluation (e.g., 0.5)

---

## Dual Timeline Visualization

The dashboard includes an **interactive dual-timeline visualization** that shows predictions vs actual posts.

### Features

#### 1. Dual Timeline Layout

```
┌─────────────────────────────────────────────────────────────┐
│                    Actual Posts (Top)                        │
│  ● ─── ● ─────────── ● ───── ● ──────────── ●              │
│  │     │              :       :               │              │
│  │     │              :       :               │              │
│  ★ ─── ◆ ─────────── ✗ ───── ✗ ──────────── ★              │
│                 Predictions (Bottom)                         │
└─────────────────────────────────────────────────────────────┘
```

#### 2. Color Coding

**Actual Posts (Top timeline)**:
- 🟢 **Green circle**: Matched to a prediction
- ⚫ **Gray circle**: No matching prediction

**Predictions (Bottom timeline)**:
- ⭐ **Green star**: Correct prediction (timing + content)
- 🔶 **Orange diamond**: Matched but not correct (timing or content off)
- ❌ **Red X**: Unmatched (no post within time window)

#### 3. Connecting Lines

Dotted blue lines connect matched prediction-post pairs, making it easy to see timing differences.

#### 4. Interactive Hover

Hover over any marker to see:
- **Actual posts**: Timestamp, content preview, match status
- **Predictions**: Predicted time, content, error metrics, similarity score

### Accessing the Timeline

1. **Dashboard navigation**: Click "🎯 Validation Timeline" in the sidebar
2. **Select time range**: Use the slider to view 7-90 days of history
3. **View statistics**: Top metrics show overall accuracy
4. **Explore timeline**: Hover and click on markers for details
5. **Compare content**: Expand side-by-side comparisons below the timeline

### Example Screenshot Interpretation

```
Timeline view (30 days):

Actual:    ●────────●─────●
            │        :     │
Predicted:  ★────────✗─────◆

Legend:
- First prediction (★): Correct! Within 6h and content matched
- Second prediction (✗): No matching post found (prediction too far off)
- Third prediction (◆): Matched but not correct (timing/content off)
```

---

## API Endpoints

The validation system provides REST API endpoints for programmatic access.

### 1. Get Validation Statistics

```bash
GET /validation/stats
```

**Response:**
```json
{
  "total_predictions": 156,
  "total_validated": 142,
  "unvalidated": 14,
  "correct_predictions": 98,
  "overall_accuracy": 0.690,
  "timing_mae_hours": 4.2,
  "within_6h_accuracy": 0.718,
  "within_12h_accuracy": 0.859,
  "within_24h_accuracy": 0.945,
  "avg_content_similarity": 0.412
}
```

**Use cases**:
- Monitoring dashboards
- Alerting systems
- Performance tracking over time

---

### 2. Run Validation

```bash
POST /validation/validate
```

**Description**: Manually trigger validation for all unvalidated predictions.

**Response:**
```json
{
  "status": "success",
  "message": "Validated 14 predictions",
  "results": {
    "validated": 14,
    "correct": 9,
    "incorrect": 5,
    "accuracy": 0.643
  }
}
```

**Use cases**:
- Manual validation triggers
- Testing validation logic
- Immediate updates after posts

---

### 3. Get Timeline Data

```bash
GET /validation/timeline?days_back=30
```

**Parameters**:
- `days_back` (optional): Days of history to return (default: 30)

**Response:**
```json
{
  "actual_posts": [
    {
      "id": "post-123",
      "time": "2025-11-06T14:32:00",
      "content": "The Fake News Media...",
      "matched_prediction_id": "pred-456"
    }
  ],
  "predictions": [
    {
      "id": "pred-456",
      "time": "2025-11-06T14:00:00",
      "content": "The Fake News...",
      "actual_post_id": "post-123",
      "actual_time": "2025-11-06T14:32:00",
      "timing_error_hours": 0.53,
      "similarity": 0.745,
      "was_correct": true,
      "timing_confidence": 0.82,
      "content_confidence": 0.71
    }
  ]
}
```

**Use cases**:
- Custom visualizations
- Data export for analysis
- External monitoring tools

---

### 4. Get Unvalidated Predictions

```bash
GET /validation/unvalidated
```

**Response:**
```json
{
  "unvalidated_predictions": [
    {
      "prediction_id": "pred-789",
      "predicted_at": "2025-11-06T12:00:00",
      "predicted_time": "2025-11-07T16:00:00",
      "predicted_content": "Democrats are...",
      "timing_confidence": 0.75,
      "content_confidence": 0.68
    }
  ],
  "count": 14
}
```

**Use cases**:
- Monitoring pending validations
- Queue management
- Debugging validation issues

---

## Running Validation

### Option 1: Automated (Cron Job) - Recommended

**For production deployments** on Render.com, validation runs automatically every hour.

#### Configuration in `render.yaml`:

```yaml
services:
  - type: cron
    name: trump-prediction-validator
    runtime: python
    plan: starter
    schedule: "0 * * * *"  # Every hour at :00
    buildCommand: pip install -r requirements.txt
    startCommand: python scripts/cron_validate.py
    envVars:
      - key: DATABASE_URL
        sync: false  # Set in Render dashboard
    autoDeploy: true
```

#### How It Works:

1. Runs hourly (configurable)
2. Checks for predictions whose `predicted_time` has passed
3. Finds matching posts within the time window
4. Calculates metrics and updates database
5. Logs results to Render dashboard

#### Monitoring:

Check **Render.com Dashboard** → **trump-prediction-validator** service → **Logs**:

```
============================================================
Prediction Validation Starting
============================================================
Found 14 unvalidated predictions where predicted time has passed
Validating prediction pred-123...
  → Matched to post-456
  → Timing error: 0.53h
  → Content similarity: 0.745
  → Result: CORRECT ✓
...
============================================================
Validation Summary:
- Total validated: 14
- Correct: 9 (64.3%)
- Timing MAE: 3.8h
- Within 6h: 78.6%
Duration: 2.14 seconds
============================================================
```

---

### Option 2: Manual Script

Run validation manually from the command line:

```bash
# Local development
python scripts/validate_predictions.py

# With custom database
DATABASE_URL=postgresql://user:pass@host/db python scripts/validate_predictions.py
```

**Output:**
```
============================================================
Starting Prediction Validation
============================================================
Loaded configuration from config/config.yaml
Validation settings:
  - Matching window: ±24 hours
  - Timing threshold: 6 hours
  - Content threshold: 0.3
  - Validation enabled: True

Found 14 unvalidated predictions where predicted time has passed

Validating prediction pred-123 (predicted for 2025-11-06 14:00)...
  → Found matching post post-456 at 2025-11-06 14:32
  → Timing error: 0.53 hours
  → Content similarity: 0.745
  → Result: CORRECT ✓

... (more predictions)

============================================================
Validation Complete!
============================================================
Summary:
  - Total validated: 14
  - Correct: 9 (64.3%)
  - Incorrect: 5 (35.7%)
  - Unmatched: 0

Timing Metrics:
  - Mean Absolute Error: 3.8 hours
  - Within 6 hours: 78.6%
  - Within 12 hours: 92.9%
  - Within 24 hours: 100.0%

Content Metrics:
  - Average similarity: 0.412
  - Above threshold (0.3): 71.4%
============================================================
```

---

### Option 3: Via API

Trigger validation programmatically:

```bash
curl -X POST http://localhost:8000/validation/validate
```

**Use cases**:
- CI/CD pipelines
- Custom automation workflows
- Testing after model retraining

---

## Validation Statistics

The system tracks comprehensive statistics accessible via API and dashboard.

### Overall Statistics

```python
{
    'total_predictions': 156,      # All predictions ever made
    'total_validated': 142,        # Predictions that have been validated
    'unvalidated': 14,             # Predictions not yet validated
    'correct_predictions': 98,     # Predictions marked as correct
    'overall_accuracy': 0.690      # correct / validated (69.0%)
}
```

### Timing Statistics

```python
{
    'timing_mae_hours': 4.2,           # Mean absolute error in hours
    'within_6h_accuracy': 0.718,       # 71.8% within 6 hours
    'within_12h_accuracy': 0.859,      # 85.9% within 12 hours
    'within_24h_accuracy': 0.945       # 94.5% within 24 hours
}
```

### Content Statistics

```python
{
    'avg_content_similarity': 0.412,   # Average similarity score
    'content_accuracy': 0.648          # % above content_threshold
}
```

### Accessing Statistics

**1. Dashboard**: Click "🎯 Validation Timeline" → See metrics at top

**2. API**:
```bash
curl http://localhost:8000/validation/stats
```

**3. Database Query**:
```sql
SELECT
    COUNT(*) as total,
    SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct,
    AVG(timing_error_hours) as mae,
    AVG(CASE WHEN timing_error_hours <= 6 THEN 1.0 ELSE 0.0 END) as within_6h
FROM predictions
WHERE actual_post_id IS NOT NULL;
```

---

## Integration with Model Retraining

Validation results feed back into the **automated retraining system**:

### Feedback Loop

```
┌──────────────┐      ┌────────────┐      ┌──────────────┐
│  Prediction  │ ───→ │ Validation │ ───→ │  Evaluation  │
│   System     │      │   System   │      │   Metrics    │
└──────────────┘      └────────────┘      └──────────────┘
                                                  │
                                                  ↓
┌──────────────┐      ┌────────────┐      ┌──────────────┐
│   Better     │ ←─── │ Retraining │ ←─── │  Performance │
│   Models     │      │   System   │      │   Tracking   │
└──────────────┘      └────────────┘      └──────────────┘
```

### How It Works

1. **Validation runs hourly** → Updates predictions with actual outcomes
2. **Metrics accumulate** → Database stores timing errors, similarities
3. **Retraining uses validated data** → Train/test splits include validation results
4. **Model evaluation** → Compare new models against validated predictions
5. **Auto-promotion** → Better models automatically replace worse ones

### Example Workflow

```python
# Weekly retraining (Sunday 3am):

# 1. Load validated predictions
validated_preds = get_validated_predictions()  # 142 predictions with outcomes

# 2. Evaluate current model
current_mae = 4.2 hours
current_accuracy = 69.0%

# 3. Train new model on updated data (includes new posts)
new_model = train_timing_model(all_posts)

# 4. Evaluate new model on validated predictions
new_mae = 3.8 hours
new_accuracy = 72.5%

# 5. Auto-promote if better
improvement = (4.2 - 3.8) / 4.2 * 100 = 9.5%
# 9.5% > 2% threshold → Promote! ✓
```

---

## Monitoring and Troubleshooting

### Health Checks

#### 1. Check Validation Status

```python
# API
GET /validation/stats

# Response should show:
- total_validated > 0
- unvalidated < 50 (assuming hourly validation)
```

#### 2. Check for Stale Predictions

```sql
-- Predictions that should be validated but aren't
SELECT COUNT(*)
FROM predictions
WHERE predicted_time < NOW() - INTERVAL '24 hours'
  AND actual_post_id IS NULL;
```

**Expected**: 0 (all past predictions validated)
**If > 0**: Validation system may be stuck

#### 3. Monitor Cron Job

**Render Dashboard** → **trump-prediction-validator** → **Logs**

Look for:
- ✅ Regular hourly executions
- ✅ "Validation Complete" messages
- ❌ Error messages or missing runs

---

### Common Issues

#### Issue 1: No Validations Happening

**Symptoms**:
- `unvalidated` count keeps growing
- Dashboard shows no validation data

**Diagnosis**:
```bash
# Check if validation is enabled
grep "validation:" config/config.yaml

# Check for unvalidated predictions
curl http://localhost:8000/validation/unvalidated
```

**Solutions**:
1. **Enable validation** in `config/config.yaml`:
   ```yaml
   validation:
     enabled: true
   ```

2. **Check cron job** is running (Render dashboard)

3. **Run manually** to test:
   ```bash
   python scripts/validate_predictions.py
   ```

---

#### Issue 2: All Predictions Show as Incorrect

**Symptoms**:
- Overall accuracy very low (<20%)
- Predictions getting matched but marked incorrect

**Diagnosis**:
```python
# Check thresholds in config
grep "timing_threshold_hours\|content_threshold" config/config.yaml
```

**Solutions**:
1. **Relax timing threshold** (e.g., 6h → 12h):
   ```yaml
   timing_threshold_hours: 12
   ```

2. **Relax content threshold** (e.g., 0.3 → 0.2):
   ```yaml
   content_threshold: 0.2
   ```

3. **Investigate model quality** (may need retraining)

---

#### Issue 3: No Matching Posts Found

**Symptoms**:
- `unmatched` predictions high
- Red X markers dominate timeline

**Diagnosis**:
```bash
# Check if posts are being collected
curl http://localhost:8000/posts/recent?limit=10

# Check matching window
grep "matching_window_hours" config/config.yaml
```

**Solutions**:
1. **Increase matching window** (e.g., 24h → 48h):
   ```yaml
   matching_window_hours: 48
   ```

2. **Verify data collection** is running:
   - Check Render dashboard → trump-prediction-collector logs
   - Ensure posts are being fetched every 30 minutes

3. **Check database** for recent posts:
   ```sql
   SELECT COUNT(*), MAX(created_at) FROM posts;
   ```

---

#### Issue 4: Timeline Not Showing Data

**Symptoms**:
- Dashboard shows "No validation data available yet"
- API returns empty arrays

**Diagnosis**:
```bash
# Check API endpoint
curl http://localhost:8000/validation/timeline?days_back=30

# Check database
SELECT COUNT(*) FROM predictions WHERE actual_post_id IS NOT NULL;
```

**Solutions**:
1. **Make predictions first** (need predictions to validate)
2. **Wait for actual posts** (predictions need time to be validated)
3. **Run validation manually** to test:
   ```bash
   python scripts/validate_predictions.py
   ```
4. **Check database connection** in dashboard

---

### Performance Optimization

#### 1. Database Indexes

Ensure indexes exist for fast validation:

```sql
-- Already created by database.py, but verify:
CREATE INDEX IF NOT EXISTS idx_predictions_predicted_time ON predictions(predicted_time);
CREATE INDEX IF NOT EXISTS idx_predictions_actual_post ON predictions(actual_post_id);
CREATE INDEX IF NOT EXISTS idx_posts_created_at ON posts(created_at);
```

#### 2. Batch Size Tuning

For large datasets, tune validation batch size:

```python
# In scripts/validate_predictions.py
# Validate in smaller batches to avoid timeouts
MAX_BATCH_SIZE = 50  # Validate max 50 predictions per run
```

#### 3. Cron Job Frequency

Adjust validation frequency based on posting rate:

```yaml
# render.yaml

# High frequency (every 30 min) - if posts are frequent
schedule: "*/30 * * * *"

# Normal frequency (hourly) - default
schedule: "0 * * * *"

# Low frequency (every 6 hours) - if posts are infrequent
schedule: "0 */6 * * *"
```

---

## Future Enhancements

### 1. BERTScore Integration

Replace simple similarity with neural embeddings:

```python
# Install: pip install bert-score
from bert_score import score

def calculate_bertscore(predicted, actual):
    P, R, F1 = score([predicted], [actual], lang='en')
    return F1.item()
```

**Benefits**:
- Semantic similarity (understands meaning, not just words)
- Better content evaluation
- Industry-standard metric

**Storage**: `bertscore_f1` column already exists in database

---

### 2. Topic Modeling

Validate topic accuracy, not just exact content:

```python
# Check if prediction captured the same topics/themes
predicted_topics = ["fake news", "media", "democrats"]
actual_topics = ["mainstream media", "democrats", "lies"]

topic_overlap = len(set(predicted_topics) & set(actual_topics)) / len(set(predicted_topics))
```

---

### 3. Confidence Calibration

Analyze if confidence scores match actual accuracy:

```python
# Are high-confidence predictions actually more accurate?
high_conf_preds = predictions[confidence > 0.8]
high_conf_accuracy = sum(was_correct) / len(high_conf_preds)

# Expected: high_conf_accuracy > overall_accuracy
```

**Use case**: Adjust confidence thresholds for auto-posting

---

### 4. Alerting System

Send notifications when accuracy drops:

```python
if overall_accuracy < 0.5:
    send_alert("Model accuracy dropped below 50%! Check system.")
```

**Integration options**:
- Email via SendGrid
- Slack webhooks
- PagerDuty for critical issues

---

## Summary

The validation system provides:

| Feature | Status |
|---------|--------|
| ✅ Automatic matching | Implemented |
| ✅ Timing metrics | Implemented |
| ✅ Content similarity | Implemented |
| ✅ Database persistence | Implemented |
| ✅ API endpoints | Implemented |
| ✅ Dashboard visualization | Implemented |
| ✅ Dual timeline view | Implemented |
| ✅ Hourly cron job | Implemented |
| ✅ Statistics tracking | Implemented |
| ✅ Integration with retraining | Implemented |
| 🔄 BERTScore | Planned |
| 🔄 Topic modeling | Planned |
| 🔄 Alerting | Planned |

---

## Quick Reference

### Configuration File

```yaml
# config/config.yaml
validation:
  enabled: true
  matching_window_hours: 24
  timing_threshold_hours: 6
  content_threshold: 0.3
```

### Manual Validation

```bash
python scripts/validate_predictions.py
```

### API Endpoints

```bash
# Get statistics
GET /validation/stats

# Run validation
POST /validation/validate

# Get timeline data
GET /validation/timeline?days_back=30

# Get unvalidated predictions
GET /validation/unvalidated
```

### Dashboard Access

Navigate to: **🎯 Validation Timeline**

### Monitoring

**Render Dashboard** → **trump-prediction-validator** → **Logs**

---

For more information, see:
- [SCHEDULING.md](SCHEDULING.md) - Automated prediction scheduling
- [RETRAINING.md](RETRAINING.md) - Automated model retraining
- [README.md](README.md) - Main project documentation
