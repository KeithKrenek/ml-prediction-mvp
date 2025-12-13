"""
Context-Driven Feature Extractors

Extracts features from real-time context data:
- News sentiment and activity
- News urgency scoring for reactive predictions
- Trump mention detection in headlines
- Trending topics relevance
- Stock market movements
- Political events
- Context freshness and completeness
- Time-lagged context features

These features help detect when Trump is likely to post reactively
(e.g., responding to breaking news or market events).

Enhanced with:
- News urgency score combining freshness, controversy, and relevance
- Breaking news detection with confidence scoring
- Reactive trigger probability estimation
- Time-lagged features (news age buckets)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from loguru import logger


@dataclass
class NewsUrgencyAnalysis:
    """Analysis of news urgency for reactive prediction."""
    urgency_score: float  # 0-1, higher = more urgent/reactive trigger
    is_breaking: bool
    trump_mentioned: bool
    controversy_level: float
    freshness_score: float
    reactive_trigger_probability: float
    trigger_keywords: List[str]
    reasoning: str


class ContextFeatureExtractor:
    """
    Extracts features from real-time context for timing prediction.
    
    Enhanced with news urgency scoring and reactive prediction features.
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
        
        # Trump-specific keywords that often trigger reactive posts
        self.trump_trigger_keywords = [
            'trump', 'donald', 'maga', 'mar-a-lago', 'truth social',
            'indictment', 'arraignment', 'charges', 'fraud',
            'investigation', 'witch hunt', 'hoax'
        ]
        
        # High-urgency keywords that indicate breaking/controversial news
        self.urgency_keywords = [
            'breaking', 'urgent', 'just in', 'developing',
            'exclusive', 'alert', 'now', 'live'
        ]
        
        # Controversy keywords that may trigger defensive posts
        self.controversy_keywords = [
            'scandal', 'investigation', 'fraud', 'illegal', 'controversy',
            'lawsuit', 'charges', 'indictment', 'accusation', 'allegation',
            'guilty', 'convicted', 'arrest', 'criminal'
        ]
        
        # Attack keywords that often trigger responses
        self.attack_keywords = [
            'attacks', 'slams', 'blasts', 'criticizes', 'accuses',
            'denies', 'responds', 'fights back', 'defends'
        ]

        logger.debug(f"ContextFeatureExtractor initialized with enhanced urgency detection")

    def extract_from_context_snapshot(
        self,
        context: Dict[str, Any],
        include_reactive_features: bool = True
    ) -> Dict[str, float]:
        """
        Extract features from a context snapshot.

        Args:
            context: Context dict with news, trends, market data
            include_reactive_features: If True, include urgency/reactive features

        Returns:
            Dict with context features
        """
        features = {}

        # News features
        if 'news' in context:
            news_features = self._extract_news_features(context['news'])
            features.update(news_features)
        elif any(k in context for k in ['top_headlines', 'political_news', 'news_summary']):
            # Context has news data in a different format
            news_proxy = {
                'top_headlines': context.get('top_headlines', []),
                'political_headlines': context.get('political_news', [])
            }
            # Extract titles if headlines are dicts
            if news_proxy['top_headlines'] and isinstance(news_proxy['top_headlines'][0], dict):
                news_proxy['top_headlines'] = [h.get('title', '') for h in news_proxy['top_headlines']]
            if news_proxy['political_headlines'] and isinstance(news_proxy['political_headlines'][0], dict):
                news_proxy['political_headlines'] = [h.get('title', '') for h in news_proxy['political_headlines']]
            news_features = self._extract_news_features(news_proxy)
            features.update(news_features)

        # Trending topics features
        if 'trends' in context:
            trends_features = self._extract_trends_features(context['trends'])
            features.update(trends_features)
        elif 'trending_keywords' in context:
            # Context has trends in a different format
            trends_proxy = {'trends': context['trending_keywords']}
            trends_features = self._extract_trends_features(trends_proxy)
            features.update(trends_features)

        # Market features
        if 'market' in context:
            market_features = self._extract_market_features(context['market'])
            features.update(market_features)
        elif any(k in context for k in ['sp500_change_pct', 'dow_change_pct', 'market_sentiment']):
            # Context has market data in a different format (handle None values)
            sp500_val = context.get('sp500_change_pct')
            dow_val = context.get('dow_change_pct')
            market_proxy = {
                'sp500_change': sp500_val if sp500_val is not None else 0,
                'dow_change': dow_val if dow_val is not None else 0
            }
            market_features = self._extract_market_features(market_proxy)
            features.update(market_features)

        # Meta features (completeness, freshness)
        meta_features = self._extract_meta_features(context)
        features.update(meta_features)
        
        # NEW: Reactive prediction features
        if include_reactive_features:
            reactive_features = self.extract_reactive_features(context)
            features.update(reactive_features)

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

        # Market movement magnitude (handle None values)
        sp500_change = market_data.get('sp500_change')
        dow_change = market_data.get('dow_change')
        
        # Default to 0 if None
        sp500_change = float(sp500_change) if sp500_change is not None else 0.0
        dow_change = float(dow_change) if dow_change is not None else 0.0

        features['market_sp500_change'] = sp500_change
        features['market_dow_change'] = dow_change
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
    
    def analyze_news_urgency(self, context: Dict) -> NewsUrgencyAnalysis:
        """
        Analyze news urgency for reactive prediction.
        
        Computes a comprehensive urgency score that indicates how likely
        the current news context is to trigger a reactive post.
        
        Args:
            context: Context dict with news, trends, market data
            
        Returns:
            NewsUrgencyAnalysis with detailed urgency metrics
        """
        # Extract headlines from various sources in context
        headlines = self._gather_all_headlines(context)
        headlines_text = ' '.join(headlines).lower()
        
        # 1. Check for Trump mentions
        trump_mentioned = any(
            kw in headlines_text for kw in self.trump_trigger_keywords
        )
        trump_mention_count = sum(
            1 for kw in self.trump_trigger_keywords if kw in headlines_text
        )
        
        # 2. Check for urgency/breaking indicators
        urgency_matches = [kw for kw in self.urgency_keywords if kw in headlines_text]
        is_breaking = len(urgency_matches) > 0
        
        # 3. Controversy level
        controversy_matches = sum(
            1 for kw in self.controversy_keywords if kw in headlines_text
        )
        controversy_level = min(1.0, controversy_matches / 5.0)  # Normalize
        
        # 4. Attack keywords (often trigger defensive posts)
        attack_matches = sum(
            1 for kw in self.attack_keywords if kw in headlines_text
        )
        
        # 5. Freshness score
        freshness_score = self._calculate_news_freshness(context)
        
        # 6. Calculate overall urgency score
        urgency_components = {
            'trump_mention': 0.35 if trump_mentioned else 0.0,
            'trump_intensity': min(0.15, trump_mention_count * 0.05),
            'breaking': 0.20 if is_breaking else 0.0,
            'controversy': controversy_level * 0.15,
            'attack_keywords': min(0.10, attack_matches * 0.03),
            'freshness': freshness_score * 0.10
        }
        
        urgency_score = sum(urgency_components.values())
        urgency_score = min(1.0, max(0.0, urgency_score))
        
        # 7. Estimate reactive trigger probability
        # Based on urgency score and historical patterns
        if urgency_score > 0.6:
            reactive_probability = 0.7 + (urgency_score - 0.6) * 0.5
        elif urgency_score > 0.3:
            reactive_probability = 0.3 + (urgency_score - 0.3) * 1.0
        else:
            reactive_probability = urgency_score
        
        reactive_probability = min(0.95, max(0.05, reactive_probability))
        
        # 8. Collect trigger keywords found
        trigger_keywords = []
        if trump_mentioned:
            trigger_keywords.extend([kw for kw in self.trump_trigger_keywords if kw in headlines_text])
        trigger_keywords.extend(urgency_matches)
        if controversy_level > 0.2:
            trigger_keywords.extend([kw for kw in self.controversy_keywords if kw in headlines_text][:3])
        
        # 9. Build reasoning
        reasoning_parts = []
        if trump_mentioned:
            reasoning_parts.append(f"Trump mentioned {trump_mention_count}x")
        if is_breaking:
            reasoning_parts.append("Breaking news detected")
        if controversy_level > 0.3:
            reasoning_parts.append(f"High controversy ({controversy_level:.0%})")
        if freshness_score > 0.7:
            reasoning_parts.append("Very fresh news")
        if attack_matches > 0:
            reasoning_parts.append(f"Attack language ({attack_matches} matches)")
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Low urgency context"
        
        return NewsUrgencyAnalysis(
            urgency_score=urgency_score,
            is_breaking=is_breaking,
            trump_mentioned=trump_mentioned,
            controversy_level=controversy_level,
            freshness_score=freshness_score,
            reactive_trigger_probability=reactive_probability,
            trigger_keywords=list(set(trigger_keywords))[:10],
            reasoning=reasoning
        )
    
    def _gather_all_headlines(self, context: Dict) -> List[str]:
        """Gather all headlines from context."""
        headlines = []
        
        # From news section
        if 'news' in context:
            news = context['news']
            headlines.extend(news.get('top_headlines', []))
            headlines.extend(news.get('political_headlines', []))
        
        # From top_headlines (direct)
        if 'top_headlines' in context:
            top_headlines = context['top_headlines']
            for h in top_headlines:
                if isinstance(h, dict):
                    headlines.append(h.get('title', ''))
                else:
                    headlines.append(str(h))
        
        # From political_news
        if 'political_news' in context:
            pol_news = context['political_news']
            for h in pol_news:
                if isinstance(h, dict):
                    headlines.append(h.get('title', ''))
                else:
                    headlines.append(str(h))
        
        # From news_summary
        if 'news_summary' in context:
            headlines.append(context['news_summary'])
        
        # From trending keywords
        if 'trending_keywords' in context:
            headlines.extend(context.get('trending_keywords', []))
        
        return [h for h in headlines if h]  # Filter empty
    
    def _calculate_news_freshness(self, context: Dict) -> float:
        """Calculate how fresh the news is (0-1, higher = fresher)."""
        # Check for explicit freshness score
        if 'freshness_score' in context:
            return context['freshness_score']
        
        # Check for captured_at timestamp
        if 'captured_at' in context:
            try:
                captured = context['captured_at']
                if isinstance(captured, str):
                    captured = datetime.fromisoformat(captured.replace('Z', '+00:00'))
                elif captured.tzinfo is None:
                    captured = captured.replace(tzinfo=timezone.utc)
                
                now = datetime.now(timezone.utc)
                hours_old = (now - captured).total_seconds() / 3600
                
                # Freshness decay: 100% at 0h, 50% at 1h, 25% at 3h, etc.
                freshness = 1.0 / (1.0 + hours_old / 2.0)
                return freshness
            except Exception as e:
                logger.debug(f"Could not parse captured_at: {e}")
        
        # Default to moderate freshness
        return 0.5
    
    def extract_reactive_features(self, context: Dict) -> Dict[str, float]:
        """
        Extract features specifically designed for reactive posting prediction.
        
        These features help predict when a reactive post is likely
        based on news urgency and controversy.
        
        Args:
            context: Context dict
            
        Returns:
            Dict with reactive prediction features
        """
        features = {}
        
        # Get urgency analysis
        urgency = self.analyze_news_urgency(context)
        
        # Core reactive features
        features['news_urgency_score'] = urgency.urgency_score
        features['is_breaking_news'] = int(urgency.is_breaking)
        features['trump_mentioned_in_news'] = int(urgency.trump_mentioned)
        features['controversy_level'] = urgency.controversy_level
        features['reactive_trigger_prob'] = urgency.reactive_trigger_probability
        
        # Freshness features
        features['news_freshness'] = urgency.freshness_score
        
        # Trigger keyword count
        features['trigger_keyword_count'] = len(urgency.trigger_keywords)
        
        # Combined trigger score (for model input)
        features['reactive_trigger_score'] = (
            0.4 * urgency.urgency_score +
            0.3 * urgency.reactive_trigger_probability +
            0.2 * urgency.controversy_level +
            0.1 * urgency.freshness_score
        )
        
        # Time-based urgency expectations
        # Higher urgency = expect post sooner
        if urgency.urgency_score > 0.6:
            features['expected_reaction_window_hours'] = 1.0
        elif urgency.urgency_score > 0.4:
            features['expected_reaction_window_hours'] = 3.0
        elif urgency.urgency_score > 0.2:
            features['expected_reaction_window_hours'] = 6.0
        else:
            features['expected_reaction_window_hours'] = 12.0
        
        return features
    
    def get_reactive_prediction_adjustment(
        self,
        context: Dict,
        base_prediction_hours: float
    ) -> Tuple[float, Dict]:
        """
        Adjust timing prediction based on reactive context.
        
        If high-urgency news is detected, the expected time until
        next post should be reduced.
        
        Args:
            context: Context dict
            base_prediction_hours: Base prediction from timing model
            
        Returns:
            Tuple of (adjusted_hours, adjustment_details)
        """
        urgency = self.analyze_news_urgency(context)
        
        adjustment_factor = 1.0
        details = {
            'urgency_score': urgency.urgency_score,
            'adjustment_applied': False,
            'reason': 'No significant reactive trigger'
        }
        
        if urgency.urgency_score > 0.6:
            # High urgency - significant reduction
            adjustment_factor = 0.3
            details['adjustment_applied'] = True
            details['reason'] = f"High urgency ({urgency.reasoning})"
        elif urgency.urgency_score > 0.4:
            # Medium urgency - moderate reduction
            adjustment_factor = 0.6
            details['adjustment_applied'] = True
            details['reason'] = f"Medium urgency ({urgency.reasoning})"
        elif urgency.urgency_score > 0.2:
            # Low urgency - small reduction
            adjustment_factor = 0.85
            details['adjustment_applied'] = True
            details['reason'] = f"Low urgency ({urgency.reasoning})"
        
        # Ensure minimum prediction time (don't predict < 5 minutes)
        adjusted_hours = max(0.083, base_prediction_hours * adjustment_factor)
        
        details['adjustment_factor'] = adjustment_factor
        details['base_hours'] = base_prediction_hours
        details['adjusted_hours'] = adjusted_hours
        
        return adjusted_hours, details

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
    """Test context feature extraction with enhanced urgency detection."""
    print("\n" + "="*80)
    print("CONTEXT FEATURE EXTRACTION TEST (Enhanced)")
    print("="*80 + "\n")

    # Create sample context with high-urgency news
    context_urgent = {
        'news': {
            'top_headlines': [
                'BREAKING: Trump indictment news develops',
                'Markets surge on economic data',
                'Trump attacks Biden in new statement'
            ],
            'political_headlines': [
                'Investigation into Trump fraud case intensifies',
                'Congress debates new bill'
            ],
            'published_at': datetime.now(timezone.utc) - timedelta(minutes=30)
        },
        'trends': {
            'trends': [
                'Trump',
                'Indictment',
                'Breaking News',
                'MAGA',
                'Biden'
            ]
        },
        'market': {
            'sp500_change': 2.5,
            'dow_change': 3.1
        },
        'captured_at': datetime.now(timezone.utc),
        'quality_scores': {
            'completeness': 0.9,
            'freshness': 0.95
        }
    }

    print("="*60)
    print("TEST 1: HIGH-URGENCY CONTEXT")
    print("="*60)
    print("Sample Context (urgent news about Trump):")
    print(f"  Headlines: {context_urgent['news']['top_headlines']}")
    print()

    extractor = ContextFeatureExtractor()
    
    # Test urgency analysis
    urgency = extractor.analyze_news_urgency(context_urgent)
    print("News Urgency Analysis:")
    print(f"  Urgency Score: {urgency.urgency_score:.2%}")
    print(f"  Is Breaking: {urgency.is_breaking}")
    print(f"  Trump Mentioned: {urgency.trump_mentioned}")
    print(f"  Controversy Level: {urgency.controversy_level:.2%}")
    print(f"  Freshness Score: {urgency.freshness_score:.2%}")
    print(f"  Reactive Trigger Probability: {urgency.reactive_trigger_probability:.2%}")
    print(f"  Trigger Keywords: {urgency.trigger_keywords}")
    print(f"  Reasoning: {urgency.reasoning}")
    print()

    # Extract all features
    features = extractor.extract_from_context_snapshot(context_urgent)
    print("Reactive Features:")
    reactive_keys = ['news_urgency_score', 'reactive_trigger_prob', 'reactive_trigger_score', 
                     'expected_reaction_window_hours', 'trump_mentioned_in_news']
    for key in reactive_keys:
        if key in features:
            print(f"  {key}: {features[key]}")
    print()

    # Test timing adjustment
    base_hours = 6.0
    adjusted, details = extractor.get_reactive_prediction_adjustment(context_urgent, base_hours)
    print("Timing Adjustment:")
    print(f"  Base prediction: {base_hours:.1f}h")
    print(f"  Adjusted prediction: {adjusted:.2f}h")
    print(f"  Adjustment factor: {details['adjustment_factor']:.2f}")
    print(f"  Reason: {details['reason']}")
    print()

    # Test with low-urgency context
    print("="*60)
    print("TEST 2: LOW-URGENCY CONTEXT")
    print("="*60)
    
    context_calm = {
        'news': {
            'top_headlines': [
                'Weather forecast for the weekend',
                'Local sports team wins game',
                'New restaurant opens downtown'
            ],
            'political_headlines': []
        },
        'trends': {
            'trends': ['Weekend', 'Sports', 'Food']
        },
        'market': {
            'sp500_change': 0.2,
            'dow_change': 0.1
        },
        'captured_at': datetime.now(timezone.utc) - timedelta(hours=3)
    }
    
    urgency_calm = extractor.analyze_news_urgency(context_calm)
    print("News Urgency Analysis (calm context):")
    print(f"  Urgency Score: {urgency_calm.urgency_score:.2%}")
    print(f"  Reactive Trigger Probability: {urgency_calm.reactive_trigger_probability:.2%}")
    print(f"  Reasoning: {urgency_calm.reasoning}")
    print()
    
    adjusted_calm, details_calm = extractor.get_reactive_prediction_adjustment(context_calm, base_hours)
    print("Timing Adjustment (calm):")
    print(f"  Adjusted prediction: {adjusted_calm:.2f}h (from {base_hours}h)")
    print(f"  Adjustment applied: {details_calm['adjustment_applied']}")
    print()

    print("="*80 + "\n")


if __name__ == "__main__":
    test_context_features()
