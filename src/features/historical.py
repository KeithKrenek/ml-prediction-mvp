"""
Historical Pattern Feature Extractors

Extracts features based on posting history patterns:
- Inter-post times (time between consecutive posts)
- Posting frequency in various time windows
- Burst detection (rapid succession of posts)
- Posting regularity metrics
- Activity level indicators

Critical for high-frequency posters like Trump (20+ posts/day).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger


class HistoricalPatternExtractor:
    """
    Extracts historical posting pattern features for timing prediction.
    """

    def __init__(
        self,
        time_windows_hours: List[float] = [1, 3, 6, 12, 24],
        lookback_periods: List[int] = [5, 10, 20],
        burst_threshold_minutes: float = 10
    ):
        """
        Initialize historical pattern extractor.

        Args:
            time_windows_hours: Time windows for counting posts (in hours)
            lookback_periods: Number of previous posts to analyze
            burst_threshold_minutes: Time threshold for burst detection (in minutes)
        """
        self.time_windows_hours = time_windows_hours
        self.lookback_periods = lookback_periods
        self.burst_threshold_minutes = burst_threshold_minutes

        logger.debug(f"HistoricalPatternExtractor initialized (windows={time_windows_hours})")

    def extract_all(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'created_at'
    ) -> pd.DataFrame:
        """
        Extract all historical pattern features.

        Args:
            df: DataFrame with post data (sorted by timestamp)
            timestamp_col: Column name for timestamp

        Returns:
            DataFrame with historical pattern features
        """
        features = pd.DataFrame(index=df.index)

        timestamps = df[timestamp_col]

        # Inter-post time features
        interpost_features = self._extract_interpost_features(timestamps)
        features = pd.concat([features, interpost_features], axis=1)

        # Posting frequency features
        frequency_features = self._extract_frequency_features(df, timestamp_col)
        features = pd.concat([features, frequency_features], axis=1)

        # Burst detection features
        burst_features = self._extract_burst_features(timestamps)
        features = pd.concat([features, burst_features], axis=1)

        # Regularity features
        regularity_features = self._extract_regularity_features(timestamps)
        features = pd.concat([features, regularity_features], axis=1)

        # Activity level features
        activity_features = self._extract_activity_features(df, timestamp_col)
        features = pd.concat([features, activity_features], axis=1)

        return features

    def _extract_interpost_features(
        self,
        timestamps: pd.Series
    ) -> pd.DataFrame:
        """
        Extract inter-post time features (time between consecutive posts).

        Args:
            timestamps: Series of datetime objects

        Returns:
            DataFrame with inter-post features
        """
        features = pd.DataFrame(index=timestamps.index)

        # Time since last post (in hours)
        time_diff = timestamps.diff()
        features['time_since_last_hours'] = time_diff.dt.total_seconds() / 3600
        features['time_since_last_hours'] = features['time_since_last_hours'].fillna(24)  # Default 24h

        # Time since last post (in minutes)
        features['time_since_last_minutes'] = time_diff.dt.total_seconds() / 60
        features['time_since_last_minutes'] = features['time_since_last_minutes'].fillna(1440)  # Default 24h

        # Rolling statistics on inter-post times
        for period in self.lookback_periods:
            if len(timestamps) >= period:
                rolling_interpost = features['time_since_last_hours'].rolling(
                    window=period, min_periods=1
                )

                features[f'interpost_mean_{period}'] = rolling_interpost.mean()
                features[f'interpost_std_{period}'] = rolling_interpost.std().fillna(0)
                features[f'interpost_min_{period}'] = rolling_interpost.min()
                features[f'interpost_max_{period}'] = rolling_interpost.max()

        # Current vs average inter-post time
        if 'interpost_mean_10' in features.columns:
            features['interpost_relative_to_avg'] = (
                features['time_since_last_hours'] / features['interpost_mean_10'].clip(lower=0.1)
            )
        else:
            features['interpost_relative_to_avg'] = 1.0

        return features

    def _extract_frequency_features(
        self,
        df: pd.DataFrame,
        timestamp_col: str
    ) -> pd.DataFrame:
        """
        Extract posting frequency features (posts per time window).

        Args:
            df: DataFrame with posts
            timestamp_col: Column name for timestamp

        Returns:
            DataFrame with frequency features
        """
        features = pd.DataFrame(index=df.index)

        timestamps = df[timestamp_col]

        # Count posts in various time windows
        for window_hours in self.time_windows_hours:
            window = timedelta(hours=window_hours)
            counts = []

            for i, current_time in enumerate(timestamps):
                # Count posts within window before current post
                window_start = current_time - window
                count = ((timestamps[:i] >= window_start) & (timestamps[:i] < current_time)).sum()
                counts.append(count)

            features[f'posts_last_{window_hours}h'] = counts

        # Posting rate (posts per hour over last 24h)
        if 'posts_last_24h' in features.columns:
            features['posting_rate_per_hour'] = features['posts_last_24h'] / 24

        return features

    def _extract_burst_features(
        self,
        timestamps: pd.Series
    ) -> pd.DataFrame:
        """
        Extract burst detection features (rapid succession of posts).

        Critical for Trump's posting behavior (often posts 5-10 times in an hour).

        Args:
            timestamps: Series of datetime objects

        Returns:
            DataFrame with burst features
        """
        features = pd.DataFrame(index=timestamps.index)

        # Time since last post in minutes
        time_diff_minutes = timestamps.diff().dt.total_seconds() / 60
        time_diff_minutes = time_diff_minutes.fillna(1440)  # Default 24h

        # Is current post part of a burst?
        features['is_burst_post'] = (
            time_diff_minutes <= self.burst_threshold_minutes
        ).astype(int)

        # Burst length (consecutive posts within burst threshold)
        burst_lengths = []
        current_burst = 0

        for is_burst in features['is_burst_post']:
            if is_burst:
                current_burst += 1
            else:
                current_burst = 0
            burst_lengths.append(current_burst)

        features['burst_length'] = burst_lengths

        # Posts in burst over last N posts
        for period in [5, 10]:
            if len(timestamps) >= period:
                features[f'burst_posts_last_{period}'] = (
                    features['is_burst_post'].rolling(window=period, min_periods=1).sum()
                )

        # Max burst size in recent history
        if len(timestamps) >= 20:
            features['max_burst_last_20'] = (
                features['burst_length'].rolling(window=20, min_periods=1).max()
            )
        else:
            features['max_burst_last_20'] = 0

        return features

    def _extract_regularity_features(
        self,
        timestamps: pd.Series
    ) -> pd.DataFrame:
        """
        Extract regularity features (how consistent is posting pattern).

        Args:
            timestamps: Series of datetime objects

        Returns:
            DataFrame with regularity features
        """
        features = pd.DataFrame(index=timestamps.index)

        # Standard deviation of inter-post times
        time_diff_hours = timestamps.diff().dt.total_seconds() / 3600
        time_diff_hours = time_diff_hours.fillna(24)

        # Coefficient of variation (CV) of inter-post times
        for period in [10, 20]:
            if len(timestamps) >= period:
                rolling_mean = time_diff_hours.rolling(window=period, min_periods=1).mean()
                rolling_std = time_diff_hours.rolling(window=period, min_periods=1).std().fillna(0)

                # CV = std / mean (measures relative variability)
                features[f'interpost_cv_{period}'] = np.where(
                    rolling_mean > 0,
                    rolling_std / rolling_mean,
                    0
                )

        # Regularity score (inverse of CV)
        if 'interpost_cv_10' in features.columns:
            features['posting_regularity'] = 1 / (1 + features['interpost_cv_10'])
        else:
            features['posting_regularity'] = 0.5

        return features

    def _extract_activity_features(
        self,
        df: pd.DataFrame,
        timestamp_col: str
    ) -> pd.DataFrame:
        """
        Extract overall activity level features.

        Args:
            df: DataFrame with posts
            timestamp_col: Column name for timestamp

        Returns:
            DataFrame with activity features
        """
        features = pd.DataFrame(index=df.index)

        # Cumulative post count
        features['cumulative_posts'] = range(1, len(df) + 1)

        # Activity level (high/medium/low based on recent posting rate)
        if 'posts_last_24h' in df.columns:
            features['activity_level'] = pd.cut(
                df['posts_last_24h'],
                bins=[-1, 5, 15, 100],
                labels=[0, 1, 2]  # 0=low, 1=medium, 2=high
            ).fillna(1).astype(int)
        else:
            features['activity_level'] = 1  # Default: medium

        return features

    def get_burst_summary(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'created_at',
        n_recent: int = 20
    ) -> Dict:
        """
        Get summary of recent burst activity.

        Args:
            df: DataFrame with posts
            timestamp_col: Column name for timestamp
            n_recent: Number of recent posts to analyze

        Returns:
            Dict with burst summary statistics
        """
        if len(df) == 0:
            return {}

        recent_df = df.tail(n_recent)
        features = self.extract_all(recent_df, timestamp_col)

        summary = {
            'total_burst_posts': features['is_burst_post'].sum(),
            'max_burst_length': features['burst_length'].max(),
            'avg_interpost_time_hours': features['time_since_last_hours'].mean(),
            'min_interpost_time_minutes': features['time_since_last_minutes'].min(),
            'posting_regularity': features['posting_regularity'].mean() if 'posting_regularity' in features else 0,
            'in_active_burst': bool(features['burst_length'].iloc[-1] > 0) if len(features) > 0 else False
        }

        return summary


def get_historical_features(
    df: pd.DataFrame,
    timestamp_col: str = 'created_at',
    prefix: str = 'historical_'
) -> pd.DataFrame:
    """
    Convenience function to extract historical features from a DataFrame.

    Args:
        df: DataFrame with post data
        timestamp_col: Column name for timestamp
        prefix: Prefix for feature names

    Returns:
        DataFrame with historical features added
    """
    extractor = HistoricalPatternExtractor()
    historical_features = extractor.extract_all(df, timestamp_col)

    # Add prefix
    historical_features = historical_features.add_prefix(prefix)

    # Combine with original DataFrame
    result = pd.concat([df, historical_features], axis=1)

    return result


def test_historical_features():
    """Test historical feature extraction."""
    print("\n" + "="*80)
    print("HISTORICAL PATTERN FEATURE EXTRACTION TEST")
    print("="*80 + "\n")

    # Create sample data with bursts
    np.random.seed(42)

    # Simulate Trump-like posting: some bursts (rapid posts) and some gaps
    base_time = pd.Timestamp('2025-01-01')
    timestamps = []

    # Normal posts (every 2-4 hours)
    for i in range(10):
        timestamps.append(base_time + timedelta(hours=i*3))

    # Burst! (5 posts in 30 minutes)
    burst_start = timestamps[-1] + timedelta(hours=2)
    for i in range(5):
        timestamps.append(burst_start + timedelta(minutes=i*5))

    # More normal posts
    for i in range(5):
        timestamps.append(timestamps[-1] + timedelta(hours=2+i))

    df = pd.DataFrame({'created_at': timestamps})

    print(f"Sample data: {len(df)} posts")
    print(f"Time range: {timestamps[0]} to {timestamps[-1]}")
    print()

    # Extract features
    extractor = HistoricalPatternExtractor()
    features = extractor.extract_all(df)

    print("Extracted Features:")
    print(features[['time_since_last_hours', 'is_burst_post', 'burst_length',
                    'posts_last_1h', 'posts_last_6h']].tail(10))
    print()

    print(f"Total features extracted: {len(features.columns)}")
    print()

    # Get burst summary
    summary = extractor.get_burst_summary(df, n_recent=10)
    print("Burst Activity Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()

    # Identify burst period
    burst_posts = features[features['is_burst_post'] == 1]
    print(f"Detected {len(burst_posts)} burst posts (posts within {extractor.burst_threshold_minutes} minutes of previous)")
    print()

    print("="*80 + "\n")


if __name__ == "__main__":
    test_historical_features()
