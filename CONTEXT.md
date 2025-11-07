# Real-time Context Integration

This document describes the real-time context integration system that enhances predictions by incorporating current news, trending topics, and market data.

## Overview

The context integration system automatically fetches real-time information from multiple sources to improve both timing and content predictions:

1. **News Headlines** - Latest political and general news (NewsAPI, RSS feeds)
2. **Trending Topics** - What people are searching for (Google Trends)
3. **Stock Market Data** - S&P 500, Dow Jones performance (yfinance)
4. **Intelligent Caching** - 30-minute cache to avoid API rate limits
5. **Database Storage** - Historical context snapshots for analysis

This context is passed to the content generation model to create more relevant, timely predictions.

---

## How It Works

### 1. Context Fetching

```python
# Automatic context gathering during predictions
context_gatherer = RealTimeContextGatherer()
context = context_gatherer.get_full_context(save_to_db=True)

# Context includes:
{
    'top_headlines': [...],       # Recent news headlines
    'political_news': [...],       # Politics-specific news
    'trending_keywords': [...],    # Google Trends data
    'sp500_value': 5234.12,       # Current S&P 500
    'sp500_change_pct': +1.2,     # Daily change %
    'market_sentiment': 'bullish', # Market direction
    ...
}
```

### 2. Integration with Content Generation

The enhanced context is automatically passed to Claude during content generation:

```python
# Content model uses context for more relevant predictions
prompt = f"""
Below are real examples of posting style:
{examples}

Current context:
Time context: Friday, November 08, 2025 at 02:00 PM
Recent news: Trump Rally | Border Security | Economy Report
Trending topics: Election, Economy, Immigration, Tax Reform, Border
Market: bullish (S&P +1.2%)

Now generate a new post relevant to this context...
"""
```

### 3. Caching Strategy

To avoid hitting API rate limits:

- **Cache TTL**: 30 minutes (configurable)
- **Automatic refresh**: Cache expires after TTL
- **Manual refresh**: Available via API or dashboard
- **Fallback mechanisms**: RSS feeds if NewsAPI fails

---

## Data Sources

### 1. NewsAPI (Primary News Source)

**Features**:
- Top headlines from US sources
- Politics-specific news category
- Up to 10 articles per request

**Setup**:
```bash
# Get free API key from https://newsapi.org
export NEWS_API_KEY="your_api_key_here"
```

**Free tier**: 100 requests/day (sufficient with 30min caching)

### 2. RSS Feeds (News Fallback)

**Sources**:
- New York Times Politics
- Washington Post Politics
- Politico

**Advantages**:
- No API key required
- No rate limits
- Always available

**Configuration**:
```yaml
# config/config.yaml
context_integration:
  rss_feeds:
    - https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml
    - https://feeds.washingtonpost.com/rss/politics
    - https://www.politico.com/rss/politicopicks.xml
```

### 3. Google Trends

**Features**:
- Trending searches in USA
- Political keyword interest levels
- Trump, Biden, Election, Congress, Politics tracking

**Setup**: No API key required (uses unofficial pytrends library)

**Limitations**: Rate-limited by Google (caching essential)

### 4. Stock Market Data (yfinance)

**Features**:
- S&P 500 (^GSPC) current value & change
- Dow Jones (^DJI) current value & change
- Automatic sentiment calculation (bullish/bearish/neutral)

**Setup**: No API key required

**Data**: Real-time during market hours, 15min delayed otherwise

---

## Database Schema

Context snapshots are stored for historical analysis:

```sql
CREATE TABLE context_snapshots (
    id                      INTEGER PRIMARY KEY,
    snapshot_id             VARCHAR UNIQUE,
    captured_at             TIMESTAMP,

    -- News
    top_headlines           JSON,
    political_news          JSON,
    news_summary            TEXT,

    -- Trending
    trending_topics         JSON,
    trending_keywords       JSON,
    trend_categories        JSON,

    -- Market
    sp500_value             FLOAT,
    sp500_change_pct        FLOAT,
    dow_value               FLOAT,
    dow_change_pct          FLOAT,
    market_sentiment        VARCHAR,

    -- Events (future)
    upcoming_events         JSON,
    recent_events           JSON,

    -- Metadata
    data_sources            JSON,
    fetch_duration_seconds  FLOAT,
    fetch_errors            JSON,
    completeness_score      FLOAT,
    freshness_score         FLOAT,

    -- Usage tracking
    used_in_predictions     INTEGER,
    prediction_ids          JSON
);
```

---

## Configuration

Edit `config/config.yaml`:

```yaml
context_integration:
  enabled: true
  cache_ttl_minutes: 30           # Cache duration
  save_to_database: true          # Store snapshots

  # News API
  news_api_enabled: true
  news_max_articles: 10

  # Google Trends
  trends_enabled: true
  trends_num_topics: 10

  # Stock Market
  market_data_enabled: true

  # RSS Feeds (fallback)
  rss_feeds_enabled: true
  rss_feeds:
    - https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml
    - https://feeds.washingtonpost.com/rss/politics
    - https://www.politico.com/rss/politicopicks.xml
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `true` | Enable/disable context fetching |
| `cache_ttl_minutes` | `30` | Cache duration in minutes |
| `save_to_database` | `true` | Save snapshots to DB |
| `news_api_enabled` | `true` | Use NewsAPI |
| `news_max_articles` | `10` | Max articles to fetch |
| `trends_enabled` | `true` | Use Google Trends |
| `trends_num_topics` | `10` | Number of trending topics |
| `market_data_enabled` | `true` | Fetch stock market data |
| `rss_feeds_enabled` | `true` | Use RSS as fallback |

---

## API Endpoints

### 1. Get Current Context

```bash
GET /context/current
```

**Response**:
```json
{
  "snapshot_id": "abc123...",
  "captured_at": "2025-11-08T14:30:00",
  "news": {
    "top_headlines": [
      {
        "title": "Trump Rally Draws Thousands",
        "source": "Reuters",
        "url": "https://...",
        "published_at": "2025-11-08T13:00:00"
      }
    ],
    "political_news": [...],
    "summary": "Trump Rally | Border Security | Economy Report"
  },
  "trending": {
    "keywords": ["Trump", "Election", "Economy", "Immigration"],
    "categories": {
      "political": {
        "Trump": 85,
        "Biden": 62,
        "Election": 78
      }
    }
  },
  "market": {
    "sp500": {
      "value": 5234.12,
      "change_pct": 1.2
    },
    "dow": {
      "value": 39456.78,
      "change_pct": 0.9
    },
    "sentiment": "bullish"
  },
  "metadata": {
    "data_sources": ["newsapi", "google_trends", "yfinance"],
    "fetch_duration_seconds": 2.34,
    "completeness_score": 1.0,
    "freshness_score": 1.0,
    "errors": []
  }
}
```

---

### 2. Get Context Summary

```bash
GET /context/summary
```

**Response**:
```json
{
  "summary": "Top news: Trump Rally | Border Security | Economy Report | Trending: Trump, Election, Economy | Market: bullish (S&P +1.2%)",
  "captured_at": "2025-11-08T14:30:00"
}
```

---

### 3. Get Context History

```bash
GET /context/history?limit=10
```

**Response**:
```json
{
  "snapshots": [
    {
      "snapshot_id": "abc123...",
      "captured_at": "2025-11-08T14:30:00",
      "news_summary": "Trump Rally | Border...",
      "trending_keywords": ["Trump", "Election"],
      "market_sentiment": "bullish",
      "sp500_change_pct": 1.2,
      "completeness_score": 1.0,
      "used_in_predictions": 3
    }
  ],
  "count": 10
}
```

---

### 4. Refresh Context (Bypass Cache)

```bash
POST /context/refresh
```

**Response**:
```json
{
  "status": "success",
  "message": "Context refreshed successfully",
  "snapshot_id": "def456...",
  "captured_at": "2025-11-08T14:32:00",
  "completeness_score": 1.0,
  "data_sources": ["newsapi", "google_trends", "yfinance"]
}
```

---

## Dashboard

The **🌐 Real-time Context** dashboard page displays:

### Sections

1. **Summary** - One-line overview of current context
2. **Metrics** - Completeness, Freshness, Fetch Time, Data Sources
3. **News Headlines** - Top 5 news stories with sources
4. **Political News** - Top 3 politics-specific stories
5. **Trending Topics** - Top 10 Google Trends keywords
6. **Political Interest Levels** - Bar chart of political keyword trends
7. **Market Data** - S&P 500, Dow Jones with sentiment
8. **Metadata** - Data sources used and any errors
9. **Context History** - Last 10 context snapshots

### Features

- **Refresh Button**: Clear cache and fetch latest data
- **Expandable Headlines**: Click to see source and URL
- **Visual Charts**: Plotly bar charts for trend data
- **Historical Table**: Track context changes over time

---

## Quality Metrics

### Completeness Score (0.0 - 1.0)

Measures how much context was successfully fetched:

```python
# Scoring: 1 point for each data source
sources = 5  # news, political_news, trending, market, events
filled = 0

if top_headlines:    filled += 1
if political_news:   filled += 1
if trending_keywords: filled += 1
if sp500_value:      filled += 1
if upcoming_events:  filled += 1

completeness = filled / sources
```

- **1.0**: All sources provided data
- **0.8**: 4/5 sources (typical without events)
- **0.6**: 3/5 sources (e.g., no NewsAPI key)
- **<0.5**: Limited context available

### Freshness Score (0.0 - 1.0)

Measures how recent the context is:

```python
fetch_duration = time_to_fetch_all_data

if fetch_duration < 5s:   freshness = 1.0
elif fetch_duration < 15s: freshness = 0.8
elif fetch_duration < 30s: freshness = 0.6
else:                      freshness = 0.4
```

- **High freshness**: APIs responding quickly
- **Low freshness**: Slow APIs or network issues

---

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

**New dependencies added**:
```txt
newsapi-python==0.2.7   # NewsAPI client
pytrends==4.9.2         # Google Trends
feedparser==6.0.11      # RSS feed parsing
yfinance==0.2.36        # Stock market data
```

---

## Environment Variables

### Required for Full Functionality

```bash
# Optional: NewsAPI key (free tier)
export NEWS_API_KEY="your_api_key_here"

# Already required for predictions:
export ANTHROPIC_API_KEY="your_claude_api_key"
export DATABASE_URL="postgresql://..."
```

### Without NEWS_API_KEY

The system will automatically fall back to RSS feeds:
- No signup required
- No rate limits
- Slightly less structured data

**Recommendation**: Use NewsAPI for best results, RSS as backup.

---

## Testing

### Test Context Gathering

```bash
# Run standalone test
python src/context/context_gatherer.py

# Expected output:
============================================================
               CONTEXT GATHERER TEST
============================================================

Snapshot ID: abc123...
Captured at: 2025-11-08 14:30:00
Completeness: 80%
Freshness: 100%
Fetch duration: 2.34s
Data sources: newsapi, google_trends, yfinance

------------------------------------------------------------
NEWS HEADLINES
------------------------------------------------------------
1. Trump Rally Draws Thousands
   Source: Reuters
2. Border Security Bill Advances
   Source: AP News
...

------------------------------------------------------------
TRENDING TOPICS
------------------------------------------------------------
1. Trump
2. Election
3. Economy
...

------------------------------------------------------------
MARKET DATA
------------------------------------------------------------
S&P 500: $5,234.12 (+1.20%)
Dow Jones: $39,456.78 (+0.90%)
Sentiment: BULLISH

============================================================
SUMMARY
============================================================
Top news: Trump Rally | Border Security | Economy Report | Trending: Trump, Election, Economy | Market: bullish (S&P +1.2%)
============================================================
```

---

## Integration with Predictions

Context is automatically integrated into predictions:

```python
# Predictor uses context gatherer (already configured)
predictor = TrumpPostPredictor()
prediction = predictor.predict(save_to_db=True)

# Context is:
# 1. Fetched by context_gatherer.get_full_context()
# 2. Passed to content_model.generate(context=context)
# 3. Incorporated into Claude prompt
# 4. Saved with prediction in database (context_data field)
```

**Benefits**:
- More relevant content predictions
- Context-aware topic selection
- Market sentiment incorporated
- News-driven timing adjustments (future enhancement)

---

## Monitoring and Troubleshooting

### Check Context Health

**Dashboard**: Navigate to 🌐 Real-time Context page

**Metrics to watch**:
- **Completeness < 60%**: Check API keys, network
- **Freshness < 60%**: APIs may be slow
- **Fetch errors**: Review error messages

### Common Issues

#### Issue 1: No News Headlines

**Symptoms**:
- Empty `top_headlines` array
- Warning: "No news headlines available"

**Solutions**:
1. **Set NEWS_API_KEY** environment variable:
   ```bash
   export NEWS_API_KEY="your_key"
   ```
2. **Check API quota**: NewsAPI free tier = 100 req/day
3. **Verify RSS fallback** is enabled in config
4. **Check network**: Can you access newsapi.org?

---

#### Issue 2: No Trending Topics

**Symptoms**:
- Empty `trending_keywords` array
- Error: "Failed to fetch trending topics"

**Solutions**:
1. **Rate limiting**: Google Trends has strict limits
2. **Use cache**: Don't refresh too frequently
3. **Wait 30 minutes**: Cache will retry automatically
4. **Check pytrends version**: Should be 4.9.2

---

#### Issue 3: No Market Data

**Symptoms**:
- `sp500_value` is null
- Error: "Failed to fetch market data"

**Solutions**:
1. **Market hours**: Data may be delayed outside trading hours
2. **Yahoo Finance issues**: yfinance depends on Yahoo
3. **Network timeout**: Increase timeout in code
4. **Check ticker symbols**: ^GSPC (S&P), ^DJI (Dow)

---

#### Issue 4: Low Completeness Score

**Symptoms**:
- Completeness < 60%
- Multiple fetch errors

**Diagnosis**:
```bash
# Check which sources failed
curl http://localhost:8000/context/current | jq '.metadata.errors'

# Expected if healthy:
{
  "errors": []
}

# If errors exist:
{
  "errors": [
    {"source": "newsapi", "error": "API key invalid"},
    {"source": "google_trends", "error": "Rate limit exceeded"}
  ]
}
```

**Solutions**:
- Fix API keys
- Wait for rate limits to reset
- Enable fallback mechanisms

---

## Performance Optimization

### 1. Cache Configuration

Adjust cache TTL based on needs:

```yaml
# config/config.yaml
context_integration:
  cache_ttl_minutes: 30  # Default

  # Options:
  # - 15: Fresher data, more API calls
  # - 30: Balanced (recommended)
  # - 60: Fewer API calls, staler data
```

### 2. Selective Data Sources

Disable sources you don't need:

```yaml
context_integration:
  news_api_enabled: true       # Primary news
  trends_enabled: false        # Disable if rate-limited
  market_data_enabled: true    # Fast, reliable
  rss_feeds_enabled: true      # Always on as backup
```

### 3. Database Cleanup

Context snapshots accumulate over time:

```sql
-- Delete old snapshots (older than 30 days)
DELETE FROM context_snapshots
WHERE captured_at < NOW() - INTERVAL '30 days';

-- Keep only recent or used snapshots
DELETE FROM context_snapshots
WHERE captured_at < NOW() - INTERVAL '30 days'
  AND used_in_predictions = 0;
```

---

## Future Enhancements

### 1. Twitter/X Trends Integration

```python
# Not yet implemented
twitter_trends_enabled: false

# Would add:
- Real-time trending topics from Twitter/X API
- Trump-specific tweet monitoring
- Viral topic detection
```

**Requirements**: Twitter API v2 (paid tier)

### 2. Political Events Calendar

```python
# Not yet implemented
events_calendar_enabled: false

# Would add:
- Upcoming political events (debates, rallies)
- Major announcements
- Legislative calendar
- Event-based timing adjustments
```

**Sources**: Congress.gov API, political event databases

### 3. Advanced NLP on News

```python
# Future enhancement
- Named entity recognition on headlines
- Sentiment analysis of news coverage
- Topic clustering and categorization
- Trump mention frequency in news
```

### 4. Timing Model Integration

Currently, context only affects **content** predictions. Future:

```python
# Use context to adjust timing predictions:
- More posts after major news events?
- Market crashes trigger more posts?
- Trending topics increase posting frequency?
```

---

## Summary

The real-time context integration provides:

| Feature | Status |
|---------|--------|
| ✅ News headlines (NewsAPI) | Implemented |
| ✅ News headlines (RSS fallback) | Implemented |
| ✅ Trending topics (Google Trends) | Implemented |
| ✅ Stock market data (yfinance) | Implemented |
| ✅ Intelligent caching (30min TTL) | Implemented |
| ✅ Database storage | Implemented |
| ✅ API endpoints (4 endpoints) | Implemented |
| ✅ Dashboard page | Implemented |
| ✅ Content generation integration | Implemented |
| 🔄 Twitter/X trends | Planned |
| 🔄 Political events calendar | Planned |
| 🔄 Timing model integration | Planned |

---

## Quick Reference

### Fetch Current Context (Python)

```python
from src.context.context_gatherer import RealTimeContextGatherer

gatherer = RealTimeContextGatherer()
context = gatherer.get_full_context(save_to_db=True)
summary = gatherer.get_context_summary(context)
print(summary)
```

### Fetch Current Context (API)

```bash
curl http://localhost:8000/context/current
curl http://localhost:8000/context/summary
curl http://localhost:8000/context/history?limit=10
curl -X POST http://localhost:8000/context/refresh
```

### Configuration

```yaml
# config/config.yaml
context_integration:
  enabled: true
  cache_ttl_minutes: 30
  news_api_enabled: true
  trends_enabled: true
  market_data_enabled: true
  rss_feeds_enabled: true
```

### Environment Variables

```bash
export NEWS_API_KEY="your_api_key"  # Optional
export ANTHROPIC_API_KEY="your_key" # Required
```

---

For more information, see:
- [SCHEDULING.md](SCHEDULING.md) - Automated prediction scheduling
- [RETRAINING.md](RETRAINING.md) - Automated model retraining
- [VALIDATION.md](VALIDATION.md) - Prediction validation system
- [README.md](README.md) - Main project documentation
