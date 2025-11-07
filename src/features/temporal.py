"""
Temporal Feature Extractors

Extracts time-based features from post data for timing prediction models.
Includes cyclical encoding for periodic features (hour, day of week).

Key features:
- Hour of day (cyclical: sin/cos encoding)
- Day of week (cyclical: sin/cos encoding)
- Weekend indicator
- Business hours indicator
- Time since midnight / until midnight
- Holiday indicators
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Dict, List, Optional
from loguru import logger


class TemporalFeatureExtractor:
    """
    Extracts temporal features from timestamps for timing prediction.
    """

    def __init__(
        self,
        business_hours_start: time = time(9, 0),
        business_hours_end: time = time(17, 0),
        use_cyclical_encoding: bool = True
    ):
        """
        Initialize temporal feature extractor.

        Args:
            business_hours_start: Start of business hours (default: 9 AM)
            business_hours_end: End of business hours (default: 5 PM)
            use_cyclical_encoding: Whether to use sin/cos encoding for periodic features
        """
        self.business_hours_start = business_hours_start
        self.business_hours_end = business_hours_end
        self.use_cyclical_encoding = use_cyclical_encoding

        logger.debug(f"TemporalFeatureExtractor initialized (cyclical={use_cyclical_encoding})")

    def extract_all(self, timestamps: pd.Series) -> pd.DataFrame:
        """
        Extract all temporal features from timestamps.

        Args:
            timestamps: Series of datetime objects

        Returns:
            DataFrame with all temporal features
        """
        features = pd.DataFrame()

        # Extract basic temporal features
        features['hour'] = timestamps.dt.hour
        features['day_of_week'] = timestamps.dt.dayofweek  # 0=Monday, 6=Sunday
        features['day_of_month'] = timestamps.dt.day
        features['month'] = timestamps.dt.month
        features['is_weekend'] = (timestamps.dt.dayofweek >= 5).astype(int)

        # Cyclical encoding for periodic features
        if self.use_cyclical_encoding:
            # Hour (24-hour cycle)
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)

            # Day of week (7-day cycle)
            features['day_of_week_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['day_of_week_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)

            # Day of month (30-day cycle, approximate)
            features['day_of_month_sin'] = np.sin(2 * np.pi * features['day_of_month'] / 30)
            features['day_of_month_cos'] = np.cos(2 * np.pi * features['day_of_month'] / 30)

            # Month (12-month cycle)
            features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)

        # Business hours indicator
        features['is_business_hours'] = timestamps.apply(
            lambda x: self._is_business_hours(x.time())
        ).astype(int)

        # Time metrics
        features['time_since_midnight_hours'] = timestamps.dt.hour + timestamps.dt.minute / 60
        features['time_until_midnight_hours'] = 24 - features['time_since_midnight_hours']

        # Part of day (morning, afternoon, evening, night)
        features['part_of_day'] = timestamps.dt.hour.apply(self._get_part_of_day)

        return features

    def extract_hour_features(self, timestamps: pd.Series) -> pd.DataFrame:
        """Extract hour-related features."""
        features = pd.DataFrame()
        features['hour'] = timestamps.dt.hour

        if self.use_cyclical_encoding:
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)

        features['is_morning'] = ((timestamps.dt.hour >= 6) & (timestamps.dt.hour < 12)).astype(int)
        features['is_afternoon'] = ((timestamps.dt.hour >= 12) & (timestamps.dt.hour < 18)).astype(int)
        features['is_evening'] = ((timestamps.dt.hour >= 18) & (timestamps.dt.hour < 22)).astype(int)
        features['is_night'] = ((timestamps.dt.hour >= 22) | (timestamps.dt.hour < 6)).astype(int)

        return features

    def extract_day_features(self, timestamps: pd.Series) -> pd.DataFrame:
        """Extract day-related features."""
        features = pd.DataFrame()
        features['day_of_week'] = timestamps.dt.dayofweek

        if self.use_cyclical_encoding:
            features['day_of_week_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['day_of_week_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)

        features['is_weekend'] = (timestamps.dt.dayofweek >= 5).astype(int)
        features['is_monday'] = (timestamps.dt.dayofweek == 0).astype(int)
        features['is_friday'] = (timestamps.dt.dayofweek == 4).astype(int)

        return features

    def _is_business_hours(self, time_obj: time) -> bool:
        """Check if time falls within business hours."""
        return self.business_hours_start <= time_obj <= self.business_hours_end

    def _get_part_of_day(self, hour: int) -> int:
        """
        Get part of day as integer.

        0: Night (22-6)
        1: Morning (6-12)
        2: Afternoon (12-18)
        3: Evening (18-22)
        """
        if 6 <= hour < 12:
            return 1  # Morning
        elif 12 <= hour < 18:
            return 2  # Afternoon
        elif 18 <= hour < 22:
            return 3  # Evening
        else:
            return 0  # Night


def get_temporal_features(
    df: pd.DataFrame,
    timestamp_col: str = 'created_at',
    prefix: str = 'temporal_'
) -> pd.DataFrame:
    """
    Convenience function to extract temporal features from a DataFrame.

    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        prefix: Prefix for feature names

    Returns:
        DataFrame with temporal features added
    """
    extractor = TemporalFeatureExtractor()
    temporal_features = extractor.extract_all(df[timestamp_col])

    # Add prefix to column names
    temporal_features = temporal_features.add_prefix(prefix)

    # Combine with original DataFrame
    result = pd.concat([df, temporal_features], axis=1)

    return result


def test_temporal_features():
    """Test temporal feature extraction."""
    print("\n" + "="*80)
    print("TEMPORAL FEATURE EXTRACTION TEST")
    print("="*80 + "\n")

    # Create sample timestamps
    timestamps = pd.date_range(start='2025-01-01', end='2025-01-07', freq='6H')
    df = pd.DataFrame({'created_at': timestamps})

    print(f"Sample data: {len(df)} timestamps from {timestamps[0]} to {timestamps[-1]}")
    print()

    # Extract features
    extractor = TemporalFeatureExtractor()
    features = extractor.extract_all(df['created_at'])

    print("Extracted Features:")
    print(features.head(10))
    print()

    print(f"Total features extracted: {len(features.columns)}")
    print(f"Feature names: {list(features.columns)}")
    print()

    # Show some statistics
    print("Feature Statistics:")
    print(f"  Weekend posts: {features['is_weekend'].sum()} / {len(features)}")
    print(f"  Business hours posts: {features['is_business_hours'].sum()} / {len(features)}")
    print(f"  Morning posts: {(features['hour'] >= 6).sum() & (features['hour'] < 12).sum()}")
    print()

    print("="*80 + "\n")


if __name__ == "__main__":
    test_temporal_features()
