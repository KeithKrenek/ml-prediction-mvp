"""
Engagement Velocity Feature Extractors

Extracts engagement-related features for timing prediction:
- Total engagement (replies + reblogs + favourites)
- Engagement velocity (rate of engagement over time)
- Rolling averages and trends
- Engagement momentum (increasing/decreasing)
- Outlier detection (viral posts)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger


class EngagementFeatureExtractor:
    """
    Extracts engagement velocity and trend features from post data.
    """

    def __init__(
        self,
        rolling_windows: List[int] = [5, 10, 20],
        normalize: bool = True
    ):
        """
        Initialize engagement feature extractor.

        Args:
            rolling_windows: Window sizes for rolling statistics
            normalize: Whether to normalize engagement metrics
        """
        self.rolling_windows = rolling_windows
        self.normalize = normalize

        logger.debug(f"EngagementFeatureExtractor initialized (windows={rolling_windows})")

    def extract_all(
        self,
        df: pd.DataFrame,
        replies_col: str = 'replies',
        reblogs_col: str = 'reblogs',
        favourites_col: str = 'favourites',
        timestamp_col: str = 'created_at'
    ) -> pd.DataFrame:
        """
        Extract all engagement features.

        Args:
            df: DataFrame with post data
            replies_col: Column name for replies count
            reblogs_col: Column name for reblogs count
            favourites_col: Column name for favourites count
            timestamp_col: Column name for timestamp

        Returns:
            DataFrame with engagement features
        """
        features = pd.DataFrame(index=df.index)

        # Handle missing values
        replies = df[replies_col].fillna(0)
        reblogs = df[reblogs_col].fillna(0)
        favourites = df[favourites_col].fillna(0)

        # Total engagement
        features['engagement_total'] = replies + reblogs + favourites
        features['engagement_replies'] = replies
        features['engagement_reblogs'] = reblogs
        features['engagement_favourites'] = favourites

        # Engagement ratios
        total = features['engagement_total']
        features['engagement_replies_ratio'] = np.where(
            total > 0, replies / total, 0
        )
        features['engagement_reblogs_ratio'] = np.where(
            total > 0, reblogs / total, 0
        )
        features['engagement_favourites_ratio'] = np.where(
            total > 0, favourites / total, 0
        )

        # Engagement velocity (if timestamps available)
        if timestamp_col in df.columns:
            features['engagement_velocity'] = self._calculate_engagement_velocity(
                df, features['engagement_total'], timestamp_col
            )

        # Rolling statistics
        for window in self.rolling_windows:
            if len(df) >= window:
                # Rolling mean
                features[f'engagement_rolling_mean_{window}'] = (
                    features['engagement_total'].rolling(window=window, min_periods=1).mean()
                )

                # Rolling std
                features[f'engagement_rolling_std_{window}'] = (
                    features['engagement_total'].rolling(window=window, min_periods=1).std().fillna(0)
                )

                # Rolling max
                features[f'engagement_rolling_max_{window}'] = (
                    features['engagement_total'].rolling(window=window, min_periods=1).max()
                )

        # Engagement relative to rolling average (momentum)
        if 'engagement_rolling_mean_5' in features.columns:
            features['engagement_momentum'] = np.where(
                features['engagement_rolling_mean_5'] > 0,
                (features['engagement_total'] - features['engagement_rolling_mean_5']) / features['engagement_rolling_mean_5'],
                0
            )
        else:
            features['engagement_momentum'] = 0

        # Outlier detection (z-score)
        if len(df) >= 10:
            mean_engagement = features['engagement_total'].mean()
            std_engagement = features['engagement_total'].std()
            if std_engagement > 0:
                features['engagement_z_score'] = (
                    (features['engagement_total'] - mean_engagement) / std_engagement
                )
                features['is_engagement_outlier'] = (
                    np.abs(features['engagement_z_score']) > 2
                ).astype(int)
            else:
                features['engagement_z_score'] = 0
                features['is_engagement_outlier'] = 0
        else:
            features['engagement_z_score'] = 0
            features['is_engagement_outlier'] = 0

        # Engagement trend (increasing/decreasing)
        features['engagement_trend'] = features['engagement_total'].diff().fillna(0)
        features['engagement_trend_direction'] = np.sign(features['engagement_trend'])

        # Normalization (optional)
        if self.normalize:
            features = self._normalize_features(features)

        return features

    def _calculate_engagement_velocity(
        self,
        df: pd.DataFrame,
        engagement: pd.Series,
        timestamp_col: str
    ) -> pd.Series:
        """
        Calculate engagement velocity (engagement per hour since post).

        Args:
            df: DataFrame with timestamps
            engagement: Series with total engagement
            timestamp_col: Name of timestamp column

        Returns:
            Series with engagement velocity
        """
        if timestamp_col not in df.columns:
            return pd.Series(0, index=df.index)

        # Time since post (in hours)
        now = pd.Timestamp.now(tz=df[timestamp_col].dt.tz)
        hours_since_post = (now - df[timestamp_col]).dt.total_seconds() / 3600

        # Avoid division by zero
        hours_since_post = hours_since_post.clip(lower=0.1)  # Minimum 0.1 hour

        # Velocity = engagement / hours
        velocity = engagement / hours_since_post

        return velocity

    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features to [0, 1] range or z-score.

        Args:
            features: DataFrame with features

        Returns:
            DataFrame with normalized features
        """
        normalized = features.copy()

        # Columns to normalize (skip ratios and binary indicators)
        exclude_cols = [
            col for col in features.columns
            if 'ratio' in col or 'is_' in col or 'direction' in col or 'z_score' in col
        ]

        for col in features.columns:
            if col in exclude_cols:
                continue

            col_min = features[col].min()
            col_max = features[col].max()

            if col_max > col_min:
                normalized[col] = (features[col] - col_min) / (col_max - col_min)
            else:
                normalized[col] = 0

        return normalized

    def get_latest_engagement_summary(
        self,
        df: pd.DataFrame,
        n_recent: int = 5
    ) -> Dict:
        """
        Get summary of recent engagement patterns.

        Args:
            df: DataFrame with posts
            n_recent: Number of recent posts to analyze

        Returns:
            Dict with engagement summary statistics
        """
        if len(df) == 0:
            return {}

        recent_df = df.tail(n_recent)
        features = self.extract_all(recent_df)

        summary = {
            'mean_engagement': features['engagement_total'].mean(),
            'median_engagement': features['engagement_total'].median(),
            'max_engagement': features['engagement_total'].max(),
            'std_engagement': features['engagement_total'].std(),
            'recent_trend': features['engagement_trend'].mean(),
            'has_viral_post': (features['is_engagement_outlier'].sum() > 0) if 'is_engagement_outlier' in features else False
        }

        return summary


def get_engagement_features(
    df: pd.DataFrame,
    prefix: str = 'engagement_'
) -> pd.DataFrame:
    """
    Convenience function to extract engagement features from a DataFrame.

    Args:
        df: DataFrame with post data
        prefix: Prefix for feature names (default: engagement_)

    Returns:
        DataFrame with engagement features added
    """
    extractor = EngagementFeatureExtractor()
    engagement_features = extractor.extract_all(df)

    # Add prefix if not already present
    if not all(col.startswith(prefix) for col in engagement_features.columns):
        engagement_features = engagement_features.add_prefix(prefix)

    # Combine with original DataFrame
    result = pd.concat([df, engagement_features], axis=1)

    return result


def test_engagement_features():
    """Test engagement feature extraction."""
    print("\n" + "="*80)
    print("ENGAGEMENT FEATURE EXTRACTION TEST")
    print("="*80 + "\n")

    # Create sample data
    np.random.seed(42)
    n_posts = 20

    df = pd.DataFrame({
        'created_at': pd.date_range(start='2025-01-01', periods=n_posts, freq='6H'),
        'replies': np.random.randint(0, 100, n_posts),
        'reblogs': np.random.randint(0, 200, n_posts),
        'favourites': np.random.randint(0, 500, n_posts)
    })

    print(f"Sample data: {len(df)} posts")
    print(df[['created_at', 'replies', 'reblogs', 'favourites']].head())
    print()

    # Extract features
    extractor = EngagementFeatureExtractor(rolling_windows=[5, 10])
    features = extractor.extract_all(df)

    print("Extracted Features:")
    print(features.head(10))
    print()

    print(f"Total features extracted: {len(features.columns)}")
    print(f"Feature names: {list(features.columns)}")
    print()

    # Get summary
    summary = extractor.get_latest_engagement_summary(df, n_recent=5)
    print("Recent Engagement Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()

    print("="*80 + "\n")


if __name__ == "__main__":
    test_engagement_features()
