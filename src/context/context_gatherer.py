"""
Real-time Context Gatherer

Fetches real-time context from multiple sources to improve predictions:
- News headlines (NewsAPI, RSS feeds)
- Trending topics (Google Trends)
- Stock market data (yfinance)
- Political events

Includes caching to avoid hitting API rate limits.
"""

import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# News and trends APIs
try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    logger.warning("newsapi-python not installed. News fetching will be limited.")
    NEWSAPI_AVAILABLE = False

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    logger.warning("pytrends not installed. Trending topics will be unavailable.")
    PYTRENDS_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    logger.warning("yfinance not installed. Stock market data will be unavailable.")
    YFINANCE_AVAILABLE = False

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    logger.warning("feedparser not installed. RSS feeds will be unavailable.")
    FEEDPARSER_AVAILABLE = False

import requests
from data.database import get_session, ContextSnapshot


class ContextCache:
    """
    Simple in-memory cache for context data to avoid hitting API limits.

    Cache is valid for a configurable duration (default: 30 minutes).
    """

    def __init__(self, ttl_minutes=30):
        """
        Initialize cache.

        Args:
            ttl_minutes: Time to live for cached data in minutes
        """
        self.cache = {}
        self.ttl_minutes = ttl_minutes
        self.timestamps = {}

    def get(self, key):
        """Get cached value if still valid"""
        if key not in self.cache:
            return None

        # Check if cache is still fresh
        cache_age = (datetime.now() - self.timestamps[key]).total_seconds() / 60

        if cache_age > self.ttl_minutes:
            # Cache expired
            del self.cache[key]
            del self.timestamps[key]
            return None

        logger.debug(f"Cache hit for {key} (age: {cache_age:.1f}m)")
        return self.cache[key]

    def set(self, key, value):
        """Set cached value"""
        self.cache[key] = value
        self.timestamps[key] = datetime.now()
        logger.debug(f"Cached {key}")

    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.timestamps.clear()
        logger.info("Cache cleared")


class RealTimeContextGatherer:
    """
    Gather real-time context from multiple sources.

    Features:
    - News headlines from NewsAPI and RSS feeds
    - Trending topics from Google Trends
    - Stock market data from yfinance
    - Intelligent caching to avoid API limits
    - Fallback mechanisms for failed requests
    - Quality scoring for context completeness
    """

    def __init__(self, config_path="config/config.yaml"):
        """Initialize context gatherer with configuration"""
        self.config = self._load_config(config_path)

        # Initialize cache
        cache_ttl = self.config.get('cache_ttl_minutes', 30)
        self.cache = ContextCache(ttl_minutes=cache_ttl)
        self.rate_limits = self.config.get('rate_limits', {})
        self.last_fetch_times = {}

        # Initialize API clients
        self._init_api_clients()

        # Track data sources used
        self.data_sources_used = []
        self.fetch_errors = []

        logger.info("RealTimeContextGatherer initialized")

    def _load_config(self, config_path):
        """Load configuration from YAML"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('context_integration', {})
        except Exception as e:
            logger.warning(f"Could not load context config: {e}. Using defaults.")
            return {
                'cache_ttl_minutes': 30,
                'news_api_enabled': True,
                'trends_enabled': True,
                'market_data_enabled': True,
                'rss_feeds_enabled': True,
                'save_to_database': True
            }

    def _init_api_clients(self):
        """Initialize API clients for various services"""
        # NewsAPI
        self.newsapi_client = None
        news_api_key = os.getenv('NEWS_API_KEY')
        if NEWSAPI_AVAILABLE and news_api_key and self.config.get('news_api_enabled', True):
            try:
                self.newsapi_client = NewsApiClient(api_key=news_api_key)
                logger.info("NewsAPI client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize NewsAPI: {e}")

        # Google Trends
        self.pytrends_client = None
        if PYTRENDS_AVAILABLE and self.config.get('trends_enabled', True):
            try:
                self.pytrends_client = TrendReq(hl='en-US', tz=360)
                logger.info("PyTrends client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize PyTrends: {e}")

        # yfinance is used directly, no client initialization needed
        self.market_data_enabled = YFINANCE_AVAILABLE and self.config.get('market_data_enabled', True)

        # feedparser is used directly, no client initialization needed
        self.rss_enabled = FEEDPARSER_AVAILABLE and self.config.get('rss_feeds_enabled', True)

    def _rate_limited(self, source: str) -> bool:
        """Return True if the given source should be throttled."""
        minutes = self.rate_limits.get(f"{source}_minutes")
        if not minutes:
            return False
        last_fetch = self.last_fetch_times.get(source)
        if last_fetch and (datetime.now() - last_fetch).total_seconds() < minutes * 60:
            logger.info(f"Rate limit active for {source}; reusing cached data when possible")
            return True
        return False

    def _mark_fetch(self, source: str):
        self.last_fetch_times[source] = datetime.now()

    def get_news_headlines(self, max_articles=10) -> Dict:
        """
        Fetch recent news headlines.

        Args:
            max_articles: Maximum number of articles to fetch

        Returns:
            dict with news data
        """
        # Check cache first
        cached = self.cache.get('news_headlines')
        if cached:
            if self._rate_limited('news'):
                logger.info("Returning cached news headlines due to rate limit")
            return cached

        if self._rate_limited('news'):
            logger.info("News rate limit active but cache expired; refreshing data once")

        headlines_data = {
            'top_headlines': [],
            'political_news': [],
            'news_summary': '',
            'fetched_at': datetime.now(),
            'source': []
        }

        # Try NewsAPI first
        if self.newsapi_client:
            try:
                # Get top headlines
                top_news = self.newsapi_client.get_top_headlines(
                    country='us',
                    page_size=max_articles
                )

                if top_news.get('articles'):
                    headlines_data['top_headlines'] = [
                        {
                            'title': article['title'],
                            'source': article['source']['name'],
                            'url': article['url'],
                            'published_at': article['publishedAt']
                        }
                        for article in top_news['articles'][:max_articles]
                    ]

                # Get politics-specific news
                politics_news = self.newsapi_client.get_top_headlines(
                    country='us',
                    category='politics',
                    page_size=max_articles
                )

                if politics_news.get('articles'):
                    headlines_data['political_news'] = [
                        {
                            'title': article['title'],
                            'source': article['source']['name'],
                            'url': article['url'],
                            'published_at': article['publishedAt']
                        }
                        for article in politics_news['articles'][:max_articles]
                    ]

                headlines_data['source'].append('newsapi')
                self.data_sources_used.append('newsapi')
                logger.info(f"Fetched {len(headlines_data['top_headlines'])} headlines from NewsAPI")

            except Exception as e:
                logger.error(f"Failed to fetch from NewsAPI: {e}")
                self.fetch_errors.append({'source': 'newsapi', 'error': str(e)})

        # Fallback to RSS feeds if NewsAPI failed or not available
        if not headlines_data['top_headlines'] and self.rss_enabled:
            rss_headlines = self._fetch_rss_headlines(max_articles)
            if rss_headlines:
                headlines_data['top_headlines'].extend(rss_headlines)
                headlines_data['source'].append('rss')
                self.data_sources_used.append('rss')

        # Create summary
        if headlines_data['top_headlines']:
            # Take top 3 headlines for summary
            top_3 = [h['title'] for h in headlines_data['top_headlines'][:3]]
            headlines_data['news_summary'] = " | ".join(top_3)

        # Cache the result
        self.cache.set('news_headlines', headlines_data)
        self._mark_fetch('news')

        return headlines_data

    def _fetch_rss_headlines(self, max_articles=10) -> List[Dict]:
        """
        Fetch headlines from RSS feeds as fallback.

        Args:
            max_articles: Maximum number of articles

        Returns:
            list of headline dicts
        """
        if not FEEDPARSER_AVAILABLE:
            return []

        # RSS feeds to check
        feeds = [
            'https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml',
            'https://feeds.washingtonpost.com/rss/politics',
            'https://www.politico.com/rss/politicopicks.xml'
        ]

        headlines = []

        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)

                for entry in feed.entries[:max_articles]:
                    headlines.append({
                        'title': entry.get('title', ''),
                        'source': feed.feed.get('title', 'RSS'),
                        'url': entry.get('link', ''),
                        'published_at': entry.get('published', '')
                    })

                logger.info(f"Fetched {len(feed.entries)} headlines from RSS: {feed_url}")

                if len(headlines) >= max_articles:
                    break

            except Exception as e:
                logger.warning(f"Failed to fetch RSS feed {feed_url}: {e}")
                continue

        return headlines[:max_articles]

    def get_trending_topics(self, num_topics=10) -> Dict:
        """
        Fetch trending topics from Google Trends.

        Args:
            num_topics: Number of trending topics to fetch

        Returns:
            dict with trending data
        """
        # Check cache
        cached = self.cache.get('trending_topics')
        if cached:
            if self._rate_limited('trends'):
                logger.info("Returning cached trending topics due to rate limit")
            return cached

        if self._rate_limited('trends'):
            logger.info("Trends rate limit active but cache expired; refreshing data once")

        trending_data = {
            'trending_topics': [],
            'trending_keywords': [],
            'trend_categories': {},
            'fetched_at': datetime.now(),
            'source': 'google_trends'
        }

        if not self.pytrends_client:
            logger.warning("PyTrends not available")
            return trending_data

        try:
            # Get trending searches (USA)
            trending_searches = self.pytrends_client.trending_searches(pn='united_states')

            if not trending_searches.empty:
                trending_data['trending_keywords'] = trending_searches[0].tolist()[:num_topics]
                logger.info(f"Fetched {len(trending_data['trending_keywords'])} trending topics")

            # Build interest for political keywords
            political_keywords = ['Trump', 'Biden', 'Election', 'Congress', 'Politics']

            try:
                self.pytrends_client.build_payload(
                    political_keywords,
                    timeframe='now 7-d',
                    geo='US'
                )

                interest_over_time = self.pytrends_client.interest_over_time()

                if not interest_over_time.empty:
                    # Get current interest levels
                    latest = interest_over_time.iloc[-1]
                    trending_data['trend_categories']['political'] = {
                        keyword: int(latest[keyword]) if keyword in latest else 0
                        for keyword in political_keywords
                    }

            except Exception as e:
                logger.warning(f"Failed to get political trends: {e}")

            self.data_sources_used.append('google_trends')

        except Exception as e:
            logger.warning(f"Failed to fetch trending topics (this is expected if Google is blocking requests): {e}")
            self.fetch_errors.append({'source': 'google_trends', 'error': str(e)})

        # Cache the result
        self.cache.set('trending_topics', trending_data)
        self._mark_fetch('trends')

        return trending_data

    def get_market_data(self) -> Dict:
        """
        Fetch current stock market data.

        Returns:
            dict with market data
        """
        # Check cache
        cached = self.cache.get('market_data')
        if cached:
            if self._rate_limited('market'):
                logger.info("Returning cached market data due to rate limit")
            return cached

        if self._rate_limited('market'):
            logger.info("Market rate limit active but cache expired; refreshing data once")

        market_data = {
            'sp500_value': None,
            'sp500_change_pct': None,
            'dow_value': None,
            'dow_change_pct': None,
            'market_sentiment': 'neutral',
            'fetched_at': datetime.now(),
            'source': 'yfinance'
        }

        if not self.market_data_enabled:
            logger.warning("yfinance not available")
            return market_data

        try:
            # Fetch S&P 500 data
            sp500 = yf.Ticker("^GSPC")
            sp500_hist = sp500.history(period="2d")

            if not sp500_hist.empty:
                current_price = sp500_hist['Close'].iloc[-1]
                prev_price = sp500_hist['Close'].iloc[-2] if len(sp500_hist) > 1 else current_price
                change_pct = ((current_price - prev_price) / prev_price) * 100

                market_data['sp500_value'] = float(current_price)
                market_data['sp500_change_pct'] = float(change_pct)

            # Fetch Dow Jones data
            dow = yf.Ticker("^DJI")
            dow_hist = dow.history(period="2d")

            if not dow_hist.empty:
                current_price = dow_hist['Close'].iloc[-1]
                prev_price = dow_hist['Close'].iloc[-2] if len(dow_hist) > 1 else current_price
                change_pct = ((current_price - prev_price) / prev_price) * 100

                market_data['dow_value'] = float(current_price)
                market_data['dow_change_pct'] = float(change_pct)

            # Determine market sentiment
            # Handle None values from failed API calls
            sp500_change = market_data['sp500_change_pct']
            dow_change = market_data['dow_change_pct']

            if sp500_change is not None and dow_change is not None:
                avg_change = (sp500_change + dow_change) / 2
            elif sp500_change is not None:
                avg_change = sp500_change
            elif dow_change is not None:
                avg_change = dow_change
            else:
                avg_change = 0.0  # No data available, default to neutral

            if avg_change > 1.0:
                market_data['market_sentiment'] = 'bullish'
            elif avg_change < -1.0:
                market_data['market_sentiment'] = 'bearish'
            else:
                market_data['market_sentiment'] = 'neutral'

            self.data_sources_used.append('yfinance')

            # Only log success if we got at least one data point
            if sp500_change is not None or dow_change is not None:
                s_and_p = f"S&P {sp500_change:.2f}%" if sp500_change is not None else "S&P N/A"
                dow_str = f"Dow {dow_change:.2f}%" if dow_change is not None else "Dow N/A"
                logger.info(f"Fetched market data: {s_and_p}, {dow_str}")
            else:
                logger.warning("Market data fetch returned no values (API may be unavailable)")

        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            self.fetch_errors.append({'source': 'yfinance', 'error': str(e)})

        # Cache the result
        self.cache.set('market_data', market_data)
        self._mark_fetch('market')

        return market_data

    def get_full_context(self, save_to_db=True) -> Dict:
        """
        Gather all context information from all sources.

        Args:
            save_to_db: Whether to save context snapshot to database

        Returns:
            dict with complete context data
        """
        start_time = time.time()

        logger.info("Gathering full context from all sources...")

        # Reset tracking
        self.data_sources_used = []
        self.fetch_errors = []

        # Gather from all sources
        news_data = self.get_news_headlines()
        trending_data = self.get_trending_topics()
        market_data = self.get_market_data()

        # Combine into full context
        full_context = {
            'snapshot_id': str(uuid.uuid4()),
            'captured_at': datetime.now(),

            # News
            'top_headlines': news_data.get('top_headlines', []),
            'political_news': news_data.get('political_news', []),
            'news_summary': news_data.get('news_summary', ''),

            # Trending
            'trending_topics': trending_data.get('trending_topics', []),
            'trending_keywords': trending_data.get('trending_keywords', []),
            'trend_categories': trending_data.get('trend_categories', {}),

            # Market
            'sp500_value': market_data.get('sp500_value'),
            'sp500_change_pct': market_data.get('sp500_change_pct'),
            'dow_value': market_data.get('dow_value'),
            'dow_change_pct': market_data.get('dow_change_pct'),
            'market_sentiment': market_data.get('market_sentiment', 'neutral'),

            # Events (placeholder for future enhancement)
            'upcoming_events': [],
            'recent_events': [],

            # Social media (placeholder)
            'twitter_trends': [],
            'viral_topics': [],

            # Weather (placeholder)
            'major_weather_events': [],

            # Metadata
            'data_sources': self.data_sources_used,
            'fetch_errors': self.fetch_errors,
            'fetch_duration_seconds': time.time() - start_time
        }

        # Calculate quality scores
        full_context['completeness_score'] = self._calculate_completeness(full_context)
        full_context['freshness_score'] = self._calculate_freshness(full_context)

        logger.success(f"Context gathered in {full_context['fetch_duration_seconds']:.2f}s")
        logger.info(f"Completeness: {full_context['completeness_score']:.2%}, Freshness: {full_context['freshness_score']:.2%}")

        # Save to database if requested
        if save_to_db and self.config.get('save_to_database', True):
            self._save_to_database(full_context)

        return full_context

    def _calculate_completeness(self, context: Dict) -> float:
        """
        Calculate how complete the context is (0.0 to 1.0).

        Checks which data sources successfully returned data.
        """
        total_sources = 5  # news, political_news, trending, market, events
        filled_sources = 0

        if context.get('top_headlines'):
            filled_sources += 1
        if context.get('political_news'):
            filled_sources += 1
        if context.get('trending_keywords'):
            filled_sources += 1
        if context.get('sp500_value') is not None:
            filled_sources += 1
        if context.get('upcoming_events') or context.get('recent_events'):
            filled_sources += 1

        return filled_sources / total_sources

    def _calculate_freshness(self, context: Dict) -> float:
        """
        Calculate how fresh the context data is (0.0 to 1.0).

        Based on how recently data was fetched.
        """
        # Since we just fetched it, freshness is always high
        # In the future, could check timestamps of individual news items
        fetch_duration = context.get('fetch_duration_seconds', 0)

        # Penalize if fetch took a long time (stale APIs)
        if fetch_duration < 5:
            return 1.0
        elif fetch_duration < 15:
            return 0.8
        elif fetch_duration < 30:
            return 0.6
        else:
            return 0.4

    def _save_to_database(self, context: Dict):
        """Save context snapshot to database"""
        try:
            session = get_session()

            snapshot = ContextSnapshot(
                snapshot_id=context['snapshot_id'],
                captured_at=context['captured_at'],

                # News
                top_headlines=context.get('top_headlines'),
                political_news=context.get('political_news'),
                news_summary=context.get('news_summary'),

                # Trending
                trending_topics=context.get('trending_topics'),
                trending_keywords=context.get('trending_keywords'),
                trend_categories=context.get('trend_categories'),

                # Market
                sp500_value=context.get('sp500_value'),
                sp500_change_pct=context.get('sp500_change_pct'),
                dow_value=context.get('dow_value'),
                dow_change_pct=context.get('dow_change_pct'),
                market_sentiment=context.get('market_sentiment'),

                # Events
                upcoming_events=context.get('upcoming_events'),
                recent_events=context.get('recent_events'),

                # Social
                twitter_trends=context.get('twitter_trends'),
                viral_topics=context.get('viral_topics'),

                # Weather
                major_weather_events=context.get('major_weather_events'),

                # Metadata
                data_sources=context.get('data_sources'),
                fetch_duration_seconds=context.get('fetch_duration_seconds'),
                fetch_errors=context.get('fetch_errors'),

                # Quality
                completeness_score=context.get('completeness_score'),
                freshness_score=context.get('freshness_score'),

                # Usage
                used_in_predictions=0,
                prediction_ids=[]
            )

            session.add(snapshot)
            session.commit()
            session.close()

            logger.info(f"Context snapshot saved to database: {context['snapshot_id']}")

        except Exception as e:
            logger.error(f"Failed to save context to database: {e}")

    def get_context_summary(self, context: Dict) -> str:
        """
        Generate a human-readable summary of the context.

        Args:
            context: Context dict from get_full_context()

        Returns:
            str summary
        """
        summary_parts = []

        # News summary
        if context.get('news_summary'):
            summary_parts.append(f"Top news: {context['news_summary']}")

        # Trending topics
        if context.get('trending_keywords'):
            top_trends = ', '.join(context['trending_keywords'][:5])
            summary_parts.append(f"Trending: {top_trends}")

        # Market
        if context.get('market_sentiment'):
            sentiment = context['market_sentiment']
            sp_change = context.get('sp500_change_pct', 0)
            summary_parts.append(f"Market: {sentiment} (S&P {sp_change:+.2f}%)")

        return " | ".join(summary_parts) if summary_parts else "Limited context available"


def main():
    """Test the real-time context gatherer"""
    logger.info("="*60)
    logger.info(" "*15 + "CONTEXT GATHERER TEST")
    logger.info("="*60)

    # Initialize
    gatherer = RealTimeContextGatherer()

    # Gather full context
    context = gatherer.get_full_context(save_to_db=False)

    # Display results
    print("\n" + "="*60)
    print("CONTEXT SUMMARY")
    print("="*60)
    print(f"\nSnapshot ID: {context['snapshot_id']}")
    print(f"Captured at: {context['captured_at']}")
    print(f"\nCompleteness: {context['completeness_score']:.1%}")
    print(f"Freshness: {context['freshness_score']:.1%}")
    print(f"Fetch duration: {context['fetch_duration_seconds']:.2f}s")
    print(f"\nData sources: {', '.join(context['data_sources'])}")

    if context['fetch_errors']:
        print(f"\nErrors: {len(context['fetch_errors'])}")
        for error in context['fetch_errors']:
            print(f"  - {error['source']}: {error['error']}")

    print("\n" + "-"*60)
    print("NEWS HEADLINES")
    print("-"*60)
    for i, headline in enumerate(context['top_headlines'][:5], 1):
        print(f"{i}. {headline['title']}")
        print(f"   Source: {headline['source']}")

    print("\n" + "-"*60)
    print("TRENDING TOPICS")
    print("-"*60)
    for i, topic in enumerate(context['trending_keywords'][:10], 1):
        print(f"{i}. {topic}")

    print("\n" + "-"*60)
    print("MARKET DATA")
    print("-"*60)
    print(f"S&P 500: ${context['sp500_value']:.2f} ({context['sp500_change_pct']:+.2f}%)")
    print(f"Dow Jones: ${context['dow_value']:.2f} ({context['dow_change_pct']:+.2f}%)")
    print(f"Sentiment: {context['market_sentiment'].upper()}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(gatherer.get_context_summary(context))
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
