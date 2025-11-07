"""
Feature Engineering Package

Advanced feature extraction for Trump post timing prediction.

This package provides comprehensive feature engineering capabilities:
- Temporal features (hour, day, cyclical encoding)
- Engagement velocity features (rolling averages, momentum)
- Historical pattern features (burst detection, posting frequency)
- Context-driven features (news, trends, market)

Usage:
    from src.features.engineering import FeatureEngineer

    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(posts_df, context=context)
"""

from src.features.engineering import FeatureEngineer
from src.features.temporal import TemporalFeatureExtractor
from src.features.engagement import EngagementFeatureExtractor
from src.features.historical import HistoricalPatternExtractor
from src.features.context import ContextFeatureExtractor

__all__ = [
    'FeatureEngineer',
    'TemporalFeatureExtractor',
    'EngagementFeatureExtractor',
    'HistoricalPatternExtractor',
    'ContextFeatureExtractor'
]

__version__ = '1.0.0'
