"""
Context-Driven Feature Extractors

Extracts features from real-time context data:
- News sentiment and activity
- Trending topics relevance
- Stock market movements
- Political events
- Context freshness and completeness

These features help detect when Trump is likely to post reactively
(e.g., responding to breaking news or market events).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from loguru import logger


class ContextFeatureExtractor:
    """
    Extracts features from real-time context for timing prediction.
    """

    def __init__(
        self,
        political_keywords: Optional[List[str]] = None,
        market_significance_threshold: float = 2.0
    ):
        """
        Initialize context feature extractor.

        Args:
            political_keywords: Keywords to identify political news
            market_significance_threshold: Threshold for significant market moves (%)
        """
        self.political_keywords = political_keywords or [
            'biden', 'trump', 'election', 'congress', 'senate',
            'white house', 'republican', 'democrat', 'impeach',
            'president', 'campaign', 'vote', 'poll'
        ]
        self.market_significance_threshold = market_significance_threshold

        logger.debug(f"ContextFeatureExtractor initialized")

    def extract_from_context_snapshot(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract features from a context snapshot.

        Args:
            context: Context dict with news, trends, market data

        Returns:
            Dict with context features
        """
        features = {}

        # News features
        if 'news' in context:
            news_features = self._extract_news_features(context['news'])
            features.update(news_features)

        # Trending topics features
        if 'trends' in context:
            trends_features = self._extract_trends_features(context['trends'])
            features.update(trends_features)

        # Market features
        if 'market' in context:
            market_features = self._extract_market_features(context['market'])
            features.update(market_features)

        # Meta features (completeness, freshness)
        meta_features = self._extract_meta_features(context)
        features.update(meta_features)

        return features

    def _extract_news_features(self, news_data: Dict) -> Dict[str, float]:
        """Extract features from news data."""
        features = {}

        # Number of articles
        top_headlines = news_data.get('top_headlines', [])
        political_headlines = news_data.get('political_headlines', [])

        features['news_total_articles'] = len(top_headlines) + len(political_headlines)
        features['news_political_articles'] = len(political_headlines)
        features['news_political_ratio'] = (
            len(political_headlines) / max(len(top_headlines) + len(political_headlines), 1)
        )

        # Keyword density (political keywords in headlines)
        all_headlines = top_headlines + political_headlines
        if all_headlines:
            keyword_matches = sum(
                1 for headline in all_headlines
                if any(keyword.lower() in headline.lower() for keyword in self.political_keywords)
            )
            features['news_political_keyword_density'] = keyword_matches / len(all_headlines)
        else:
            features['news_political_keyword_density'] = 0

        # Breaking news indicator (recency)
        if 'published_at' in news_data:
            try:
                published_time = pd.to_datetime(news_data['published_at'])
                hours_since = (datetime.now() - published_time).total_seconds() / 3600
                features['news_hours_since_latest'] = hours_since
                features['is_breaking_news'] = int(hours_since < 2)  # Within 2 hours
            except:
                features['news_hours_since_latest'] = 24
                features['is_breaking_news'] = 0
        else:
            features['news_hours_since_latest'] = 24
            features['is_breaking_news'] = 0

        # Controversy indicators (certain keywords suggest controversial topics)
        controversy_keywords = ['scandal', 'investigation', 'fraud', 'illegal', 'controversy']
        if all_headlines:
            controversy_matches = sum(
                1 for headline in all_headlines
                if any(keyword in headline.lower() for keyword in controversy_keywords)
            )
            features['news_controversy_score'] = controversy_matches / len(all_headlines)
        else:
            features['news_controversy_score'] = 0

        return features

    def _extract_trends_features(self, trends_data: Dict) -> Dict[str, float]:
        """Extract features from trending topics."""
        features = {}

        trends = trends_data.get('trends', [])

        # Number of trends
        features['trends_total'] = len(trends)

        # Political trend ratio
        if trends:
            political_trends = sum(
                1 for trend in trends
                if any(keyword.lower() in trend.lower() for keyword in self.political_keywords)
            )
            features['trends_political'] = political_trends
            features['trends_political_ratio'] = political_trends / len(trends)
        else:
            features['trends_political'] = 0
            features['trends_political_ratio'] = 0

        # Trump-related trends
        trump_keywords = ['trump', 'donald', 'maga']
        if trends:
            trump_trends = sum(
                1 for trend in trends
                if any(keyword.lower() in trend.lower() for keyword in trump_keywords)
            )
            features['trends_trump_related'] = trump_trends
            features['is_trump_trending'] = int(trump_trends > 0)
        else:
            features['trends_trump_related'] = 0
            features['is_trump_trending'] = 0

        return features

    def _extract_market_features(self, market_data: Dict) -> Dict[str, float]:
        """Extract features from market data."""
        features = {}

        # Market movement magnitude
        sp500_change = market_data.get('sp500_change', 0)
        dow_change = market_data.get('dow_change', 0)

        features['market_sp500_change'] = float(sp500_change)
        features['market_dow_change'] = float(dow_change)
        features['market_avg_change'] = (sp500_change + dow_change) / 2

        # Market movement magnitude (absolute)
        features['market_movement_magnitude'] = abs(features['market_avg_change'])

        # Significant market move indicator
        features['is_significant_market_move'] = int(
            features['market_movement_magnitude'] >= self.market_significance_threshold
        )

        # Market direction
        features['market_direction'] = np.sign(features['market_avg_change'])  # -1, 0, or 1

        # Volatility indicator (large absolute change)
        features['is_market_volatile'] = int(features['market_movement_magnitude'] > 1.0)

        return features

    def _extract_meta_features(self, context: Dict) -> Dict[str, float]:
        """Extract meta-features about context quality."""
        features = {}

        # Context completeness (how many sources have data)
        completeness_score = 0
        if context.get('news'):
            completeness_score += 1
        if context.get('trends'):
            completeness_score += 1
        if context.get('market'):
            completeness_score += 1

        features['context_completeness'] = completeness_score / 3  # Normalize to [0, 1]

        # Context freshness (based on quality scores if available)
        if 'quality_scores' in context:
            quality = context['quality_scores']
            features['context_freshness'] = quality.get('freshness', 0.5)
            features['context_quality'] = quality.get('completeness', 0.5)
        else:
            features['context_freshness'] = 0.5
            features['context_quality'] = 0.5

        return features

    def extract_time_series(
        self,
        context_snapshots: List[Dict],
        timestamps: List[datetime]
    ) -> pd.DataFrame:
        """
        Extract features from a time series of context snapshots.

        Args:
            context_snapshots: List of context dicts
            timestamps: List of timestamps for each snapshot

        Returns:
            DataFrame with time-series context features
        """
        if not context_snapshots or not timestamps:
            return pd.DataFrame()

        features_list = []

        for context in context_snapshots:
            features = self.extract_from_context_snapshot(context)
            features_list.append(features)

        df = pd.DataFrame(features_list, index=timestamps)

        # Add trend features (changes over time)
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[f'{col}_change'] = df[col].diff().fillna(0)

        return df

    def get_context_summary(self, context: Dict) -> Dict:
        """
        Get human-readable summary of context.

        Args:
            context: Context dict

        Returns:
            Dict with context summary
        """
        features = self.extract_from_context_snapshot(context)

        summary = {
            'breaking_news': bool(features.get('is_breaking_news', 0)),
            'trump_trending': bool(features.get('is_trump_trending', 0)),
            'significant_market_move': bool(features.get('is_significant_market_move', 0)),
            'political_news_density': features.get('news_political_ratio', 0),
            'context_quality': features.get('context_completeness', 0),
            'likely_reactive_trigger': (
                features.get('is_breaking_news', 0) or
                features.get('is_significant_market_move', 0)
            )
        }

        return summary


def get_context_features(
    context: Dict,
    prefix: str = 'context_'
) -> Dict[str, float]:
    """
    Convenience function to extract context features.

    Args:
        context: Context dict from ContextGatherer
        prefix: Prefix for feature names

    Returns:
        Dict with context features
    """
    extractor = ContextFeatureExtractor()
    features = extractor.extract_from_context_snapshot(context)

    # Add prefix
    prefixed_features = {f"{prefix}{key}": value for key, value in features.items()}

    return prefixed_features


def test_context_features():
    """Test context feature extraction."""
    print("\n" + "="*80)
    print("CONTEXT FEATURE EXTRACTION TEST")
    print("="*80 + "\n")

    # Create sample context
    context = {
        'news': {
            'top_headlines': [
                'Markets surge on economic data',
                'Trump rally draws thousands',
                'Biden announces new policy'
            ],
            'political_headlines': [
                'Election fraud investigation continues',
                'Congress debates new bill'
            ],
            'published_at': datetime.now() - timedelta(hours=1)
        },
        'trends': {
            'trends': [
                'Trump',
                'Election2024',
                'Breaking News',
                'Stock Market',
                'Biden'
            ]
        },
        'market': {
            'sp500_change': 2.5,
            'dow_change': 3.1
        },
        'quality_scores': {
            'completeness': 0.9,
            'freshness': 0.95
        }
    }

    print("Sample Context:")
    print(f"  News articles: {len(context['news']['top_headlines']) + len(context['news']['political_headlines'])}")
    print(f"  Trending topics: {len(context['trends']['trends'])}")
    print(f"  Market changes: S&P {context['market']['sp500_change']}%, Dow {context['market']['dow_change']}%")
    print()

    # Extract features
    extractor = ContextFeatureExtractor()
    features = extractor.extract_from_context_snapshot(context)

    print("Extracted Features:")
    for key, value in sorted(features.items()):
        print(f"  {key}: {value}")
    print()

    # Get summary
    summary = extractor.get_context_summary(context)
    print("Context Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()

    print("="*80 + "\n")


if __name__ == "__main__":
    test_context_features()
