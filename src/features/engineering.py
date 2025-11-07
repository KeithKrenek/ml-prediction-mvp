"""
Feature Engineering Coordinator

Main interface for extracting all features for timing prediction models.
Coordinates temporal, engagement, historical, and context-driven feature extractors.

Usage:
    engineer = FeatureEngineer(config)
    features_df = engineer.engineer_features(posts_df, context=context)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from loguru import logger
import yaml
from pathlib import Path

from src.features.temporal import TemporalFeatureExtractor
from src.features.engagement import EngagementFeatureExtractor
from src.features.historical import HistoricalPatternExtractor
from src.features.context import ContextFeatureExtractor


class FeatureEngineer:
    """
    Coordinates all feature extraction for timing prediction.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize feature engineer.

        Args:
            config: Feature engineering configuration dict
            config_path: Path to config file (if config not provided)
        """
        # Load configuration
        if config is None and config_path:
            config = self._load_config(config_path)
        self.config = config or {}

        # Get feature configuration
        feature_config = self.config.get('feature_engineering', {})

        # Initialize feature extractors
        self.temporal_extractor = TemporalFeatureExtractor(
            use_cyclical_encoding=feature_config.get('use_cyclical_encoding', True)
        )

        self.engagement_extractor = EngagementFeatureExtractor(
            rolling_windows=feature_config.get('engagement_windows', [5, 10, 20]),
            normalize=feature_config.get('normalize_engagement', True)
        )

        self.historical_extractor = HistoricalPatternExtractor(
            time_windows_hours=feature_config.get('time_windows_hours', [1, 3, 6, 12, 24]),
            lookback_periods=feature_config.get('lookback_periods', [5, 10, 20]),
            burst_threshold_minutes=feature_config.get('burst_threshold_minutes', 10)
        )

        self.context_extractor = ContextFeatureExtractor(
            market_significance_threshold=feature_config.get('market_threshold', 2.0)
        )

        # Feature selection
        self.enabled_features = feature_config.get('enabled_features', {
            'temporal': True,
            'engagement': True,
            'historical': True,
            'context': True
        })

        logger.info(f"FeatureEngineer initialized with {sum(self.enabled_features.values())} feature groups enabled")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return {}

    def engineer_features(
        self,
        df: pd.DataFrame,
        context: Optional[Dict] = None,
        timestamp_col: str = 'created_at',
        replies_col: str = 'replies_count',
        reblogs_col: str = 'reblogs_count',
        favourites_col: str = 'favourites_count'
    ) -> pd.DataFrame:
        """
        Extract all enabled features from post data.

        Args:
            df: DataFrame with post data (sorted by timestamp)
            context: Optional context dict for context-driven features
            timestamp_col: Column name for timestamp
            replies_col: Column name for replies count
            reblogs_col: Column name for reblogs count
            favourites_col: Column name for favourites count

        Returns:
            DataFrame with all features
        """
        logger.info(f"Engineering features for {len(df)} posts...")

        features_df = df.copy()

        # 1. Temporal features
        if self.enabled_features.get('temporal', True):
            logger.debug("Extracting temporal features...")
            temporal_features = self.temporal_extractor.extract_all(df[timestamp_col])
            features_df = pd.concat([features_df, temporal_features], axis=1)

        # 2. Engagement features
        if self.enabled_features.get('engagement', True):
            logger.debug("Extracting engagement features...")
            engagement_features = self.engagement_extractor.extract_all(
                df,
                replies_col=replies_col,
                reblogs_col=reblogs_col,
                favourites_col=favourites_col,
                timestamp_col=timestamp_col
            )
            features_df = pd.concat([features_df, engagement_features], axis=1)

        # 3. Historical pattern features
        if self.enabled_features.get('historical', True):
            logger.debug("Extracting historical pattern features...")
            historical_features = self.historical_extractor.extract_all(df, timestamp_col)
            features_df = pd.concat([features_df, historical_features], axis=1)

        # 4. Context-driven features (if context provided)
        if self.enabled_features.get('context', True) and context:
            logger.debug("Extracting context-driven features...")
            context_features = self.context_extractor.extract_from_context_snapshot(context)

            # Broadcast context features to all rows
            for key, value in context_features.items():
                features_df[key] = value

        logger.success(f"Feature engineering complete: {len(features_df.columns)} total features")

        return features_df

    def get_feature_names(self, category: Optional[str] = None) -> List[str]:
        """
        Get list of feature names.

        Args:
            category: Optional category filter ('temporal', 'engagement', 'historical', 'context')

        Returns:
            List of feature names
        """
        # This would ideally be populated from actual feature extraction
        # For now, return a representative list

        all_features = {
            'temporal': [
                'hour', 'hour_sin', 'hour_cos',
                'day_of_week', 'day_of_week_sin', 'day_of_week_cos',
                'is_weekend', 'is_business_hours',
                'time_since_midnight_hours', 'time_until_midnight_hours'
            ],
            'engagement': [
                'engagement_total', 'engagement_velocity',
                'engagement_rolling_mean_5', 'engagement_rolling_mean_10',
                'engagement_momentum', 'is_engagement_outlier'
            ],
            'historical': [
                'time_since_last_hours', 'interpost_mean_10',
                'posts_last_1h', 'posts_last_6h', 'posts_last_24h',
                'is_burst_post', 'burst_length', 'posting_regularity'
            ],
            'context': [
                'news_total_articles', 'news_political_ratio', 'is_breaking_news',
                'trends_trump_related', 'is_trump_trending',
                'market_avg_change', 'is_significant_market_move'
            ]
        }

        if category:
            return all_features.get(category, [])
        else:
            # Return all features
            return [f for features in all_features.values() for f in features]

    def get_feature_importance(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get feature importance from a trained model.

        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names

        Returns:
            DataFrame with feature importances sorted by importance
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return pd.DataFrame()

        if feature_names is None:
            feature_names = self.get_feature_names()

        importances = pd.DataFrame({
            'feature': feature_names[:len(model.feature_importances_)],
            'importance': model.feature_importances_
        })

        importances = importances.sort_values('importance', ascending=False)

        return importances

    def select_top_features(
        self,
        df: pd.DataFrame,
        top_n: int = 20,
        importance_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Select top N most important features.

        Args:
            df: DataFrame with all features
            top_n: Number of top features to keep
            importance_df: DataFrame with feature importances

        Returns:
            DataFrame with only top features
        """
        if importance_df is None:
            logger.warning("No importance data provided, returning all features")
            return df

        top_features = importance_df.head(top_n)['feature'].tolist()

        # Keep original columns plus top features
        original_cols = ['post_id', 'created_at', 'content',
                        'replies_count', 'reblogs_count', 'favourites_count']
        keep_cols = [col for col in original_cols if col in df.columns] + top_features

        return df[keep_cols]

    def prepare_for_prophet(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'created_at',
        additional_regressors: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Prepare features for Prophet model.

        Prophet requires:
        - 'ds': datetime column
        - 'y': target variable (1 for each post)
        - Optional additional regressors

        Args:
            df: DataFrame with features
            timestamp_col: Column name for timestamp
            additional_regressors: List of feature names to use as regressors

        Returns:
            DataFrame formatted for Prophet
        """
        prophet_df = pd.DataFrame({
            'ds': df[timestamp_col],
            'y': 1  # Binary: post happened
        })

        # Add additional regressors if specified
        if additional_regressors:
            for regressor in additional_regressors:
                if regressor in df.columns:
                    prophet_df[regressor] = df[regressor]
                else:
                    logger.warning(f"Regressor '{regressor}' not found in DataFrame")

        return prophet_df


def test_feature_engineer():
    """Test feature engineering pipeline."""
    print("\n" + "="*80)
    print("FEATURE ENGINEERING PIPELINE TEST")
    print("="*80 + "\n")

    # Create sample data
    np.random.seed(42)
    n_posts = 30

    df = pd.DataFrame({
        'post_id': [f"post_{i}" for i in range(n_posts)],
        'created_at': pd.date_range(start='2025-01-01', periods=n_posts, freq='3H'),
        'content': [f"Sample post {i}" for i in range(n_posts)],
        'replies_count': np.random.randint(0, 100, n_posts),
        'reblogs_count': np.random.randint(0, 200, n_posts),
        'favourites_count': np.random.randint(0, 500, n_posts)
    })

    # Sample context
    context = {
        'news': {
            'top_headlines': ['Breaking news', 'Trump rally'],
            'political_headlines': ['Election update']
        },
        'trends': {
            'trends': ['Trump', 'Breaking', 'Politics']
        },
        'market': {
            'sp500_change': 1.5,
            'dow_change': 1.8
        }
    }

    print(f"Sample data: {len(df)} posts")
    print(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}")
    print()

    # Initialize feature engineer
    engineer = FeatureEngineer()

    # Engineer features
    features_df = engineer.engineer_features(df, context=context)

    print(f"Features engineered: {len(features_df.columns)} total columns")
    print()

    # Show feature categories
    feature_cols = [col for col in features_df.columns if col not in df.columns]
    print(f"New features added: {len(feature_cols)}")
    print()

    # Sample features
    sample_features = [
        'hour_sin', 'is_weekend', 'engagement_total',
        'time_since_last_hours', 'is_burst_post', 'news_political_ratio'
    ]

    available_samples = [f for f in sample_features if f in features_df.columns]
    if available_samples:
        print("Sample features (last 5 rows):")
        print(features_df[available_samples].tail())
        print()

    # Prepare for Prophet
    prophet_df = engineer.prepare_for_prophet(
        features_df,
        additional_regressors=['engagement_total', 'time_since_last_hours', 'is_weekend']
    )

    print("Prophet-formatted data:")
    print(prophet_df.head())
    print()

    print("="*80 + "\n")


if __name__ == "__main__":
    test_feature_engineer()
