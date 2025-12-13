"""
Historical Pattern Feature Extractors

Extracts features based on posting history patterns:
- Inter-post times (time between consecutive posts)
- Posting frequency in various time windows
- Burst detection (rapid succession of posts)
- Burst momentum and continuation probability
- Posting regularity metrics
- Activity level indicators

Critical for high-frequency posters like Trump (20+ posts/day).

Enhanced with:
- Burst momentum (acceleration of posting rate)
- Typical burst length statistics
- Probability of burst continuation
- Historical burst pattern analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
from dataclasses import dataclass


@dataclass
class BurstAnalysis:
    """Analysis results for burst detection and prediction."""
    is_in_burst: bool
    current_burst_length: int
    burst_momentum: float  # Rate of acceleration
    prob_continuation: float  # Probability burst continues
    expected_next_interval_minutes: float
    typical_burst_length: float
    burst_intensity: float  # Posts per minute in current burst


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
        
        Enhanced with:
        - Burst momentum (acceleration)
        - Continuation probability
        - Burst intensity metrics

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

        # ============================================================
        # NEW: Burst Momentum Features
        # ============================================================
        
        # Burst momentum: rate of change in inter-post times
        # Negative momentum = accelerating (posts getting faster)
        # Positive momentum = decelerating (posts slowing down)
        time_diff_diff = time_diff_minutes.diff()
        features['burst_momentum'] = time_diff_diff.fillna(0)
        
        # Rolling momentum over last 3-5 posts
        features['burst_momentum_3'] = features['burst_momentum'].rolling(
            window=3, min_periods=1
        ).mean()
        features['burst_momentum_5'] = features['burst_momentum'].rolling(
            window=5, min_periods=1
        ).mean()
        
        # Is accelerating (negative momentum during burst)
        features['is_accelerating'] = (
            (features['is_burst_post'] == 1) & 
            (features['burst_momentum'] < 0)
        ).astype(int)
        
        # ============================================================
        # NEW: Burst Intensity Features
        # ============================================================
        
        # Calculate burst intensity (posts per hour during burst)
        burst_intensities = []
        for i, (is_burst, bl) in enumerate(zip(features['is_burst_post'], features['burst_length'])):
            if is_burst and bl > 0:
                # Look back at burst duration
                burst_start_idx = max(0, i - bl)
                if burst_start_idx < i:
                    burst_duration_minutes = time_diff_minutes.iloc[burst_start_idx:i+1].sum()
                    if burst_duration_minutes > 0:
                        intensity = (bl + 1) * 60 / burst_duration_minutes  # posts per hour
                    else:
                        intensity = 0
                else:
                    intensity = 0
            else:
                intensity = 0
            burst_intensities.append(intensity)
        
        features['burst_intensity'] = burst_intensities
        
        # ============================================================
        # NEW: Burst Continuation Probability
        # ============================================================
        
        # Calculate historical burst statistics for continuation probability
        features['prob_burst_continuation'] = self._compute_continuation_probability(
            features['is_burst_post'].values,
            features['burst_length'].values,
            time_diff_minutes.values
        )
        
        # ============================================================
        # NEW: Typical Burst Statistics
        # ============================================================
        
        # Rolling typical burst length (learned from history)
        burst_end_indices = []
        current_length = 0
        for i, is_burst in enumerate(features['is_burst_post']):
            if is_burst:
                current_length += 1
            elif current_length > 0:
                burst_end_indices.append((i-1, current_length))
                current_length = 0
        
        # If still in burst at end, record it
        if current_length > 0:
            burst_end_indices.append((len(features)-1, current_length))
        
        # Compute rolling typical burst length
        typical_lengths = [3.0] * len(features)  # Default
        if burst_end_indices:
            all_lengths = [bl for _, bl in burst_end_indices]
            cumulative_mean = np.cumsum(all_lengths) / np.arange(1, len(all_lengths) + 1)
            
            # Assign to features based on when each burst ended
            length_idx = 0
            for i in range(len(features)):
                if length_idx < len(burst_end_indices):
                    end_idx, _ = burst_end_indices[length_idx]
                    if i > end_idx:
                        length_idx += 1
                if length_idx > 0 and length_idx <= len(cumulative_mean):
                    typical_lengths[i] = cumulative_mean[length_idx - 1]
        
        features['typical_burst_length'] = typical_lengths
        
        # ============================================================
        # NEW: Expected Next Interval (within burst)
        # ============================================================
        
        # Rolling mean of burst intervals
        burst_intervals = time_diff_minutes.where(features['is_burst_post'] == 1)
        features['expected_burst_interval'] = burst_intervals.expanding().mean().fillna(5.0)  # Default 5 min
        
        # Confidence in burst continuation based on how close to typical length
        features['burst_progress'] = features['burst_length'] / features['typical_burst_length'].clip(lower=1)
        features['burst_completion_pct'] = features['burst_progress'].clip(upper=1.0)

        return features
    
    def _compute_continuation_probability(
        self,
        is_burst: np.ndarray,
        burst_lengths: np.ndarray,
        time_diffs: np.ndarray
    ) -> np.ndarray:
        """
        Compute probability that current burst will continue.
        
        Based on:
        1. Current burst length vs historical typical length
        2. Recent momentum (accelerating or decelerating)
        3. Time since last post (shorter = more likely to continue)
        
        Args:
            is_burst: Binary array indicating burst posts
            burst_lengths: Current burst length at each position
            time_diffs: Inter-post times in minutes
            
        Returns:
            Array of continuation probabilities
        """
        n = len(is_burst)
        probs = np.zeros(n)
        
        # Track historical burst lengths for probability estimation
        historical_bursts = []
        current_burst_length = 0
        
        for i in range(n):
            if is_burst[i]:
                current_burst_length += 1
                
                # Base probability from historical data
                if historical_bursts:
                    # What fraction of historical bursts were longer than current?
                    longer_bursts = sum(1 for bl in historical_bursts if bl > current_burst_length)
                    base_prob = longer_bursts / len(historical_bursts)
                else:
                    # Prior: assume geometric distribution with p=0.3 (mean burst ~3-4)
                    base_prob = 0.7 ** current_burst_length
                
                # Adjust for recent momentum
                if i >= 2:
                    recent_trend = time_diffs[i] - time_diffs[i-1] if time_diffs[i-1] > 0 else 0
                    # Accelerating (negative trend) increases probability
                    momentum_factor = 1.0 - 0.1 * np.sign(recent_trend)
                else:
                    momentum_factor = 1.0
                
                # Adjust for time since last post
                # Shorter intervals = higher continuation probability
                time_factor = np.exp(-time_diffs[i] / 10.0)  # Decay with 10-min half-life
                
                probs[i] = min(0.95, max(0.05, base_prob * momentum_factor * time_factor))
            else:
                probs[i] = 0.0
                
                # Record completed burst
                if current_burst_length > 0:
                    historical_bursts.append(current_burst_length)
                current_burst_length = 0
        
        return probs

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
    
    def analyze_current_burst(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'created_at'
    ) -> BurstAnalysis:
        """
        Analyze current burst state and predict continuation.
        
        This is the main method for burst-aware timing prediction.
        
        Args:
            df: DataFrame with posts (sorted by time)
            timestamp_col: Column name for timestamp
            
        Returns:
            BurstAnalysis with current state and predictions
        """
        if len(df) == 0:
            return BurstAnalysis(
                is_in_burst=False,
                current_burst_length=0,
                burst_momentum=0.0,
                prob_continuation=0.0,
                expected_next_interval_minutes=60.0,
                typical_burst_length=3.0,
                burst_intensity=0.0
            )
        
        # Get full features
        features = self.extract_all(df, timestamp_col)
        
        # Get latest values (handle None/NaN values safely)
        latest = features.iloc[-1]
        
        def safe_float(val, default=0.0):
            """Safely convert to float, handling None and NaN."""
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return default
            try:
                return float(val)
            except (TypeError, ValueError):
                return default
        
        def safe_int(val, default=0):
            """Safely convert to int, handling None and NaN."""
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return default
            try:
                return int(val)
            except (TypeError, ValueError):
                return default
        
        is_in_burst = bool(safe_int(latest.get('is_burst_post', 0)) == 1)
        current_burst_length = safe_int(latest.get('burst_length', 0))
        burst_momentum = safe_float(latest.get('burst_momentum_3', 0.0))
        prob_continuation = safe_float(latest.get('prob_burst_continuation', 0.0))
        typical_burst_length = safe_float(latest.get('typical_burst_length', 3.0), 3.0)
        burst_intensity = safe_float(latest.get('burst_intensity', 0.0))
        
        # Calculate expected next interval
        if is_in_burst:
            expected_interval = safe_float(latest.get('expected_burst_interval', 5.0), 5.0)
            # Adjust based on momentum
            if burst_momentum < 0:  # Accelerating
                expected_interval *= 0.8  # Expect faster
            elif burst_momentum > 0:  # Decelerating
                expected_interval *= 1.2  # Expect slower
        else:
            # Not in burst - use average inter-post time
            expected_interval = safe_float(latest.get('time_since_last_minutes', 60.0), 60.0)
            # If we just ended a burst, expect longer gap
            if len(features) > 1 and safe_int(features.iloc[-2].get('is_burst_post', 0)) == 1:
                expected_interval *= 2.0  # Post-burst cooldown
        
        return BurstAnalysis(
            is_in_burst=is_in_burst,
            current_burst_length=current_burst_length,
            burst_momentum=burst_momentum,
            prob_continuation=prob_continuation,
            expected_next_interval_minutes=expected_interval,
            typical_burst_length=typical_burst_length,
            burst_intensity=burst_intensity
        )
    
    def predict_next_post_timing(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'created_at'
    ) -> Dict:
        """
        Predict timing of next post based on burst analysis.
        
        Returns timing prediction with confidence based on current burst state.
        
        Args:
            df: DataFrame with posts
            timestamp_col: Column name for timestamp
            
        Returns:
            Dict with:
                - expected_minutes: Expected time until next post
                - confidence: Confidence in prediction (0-1)
                - prediction_type: 'burst_continuation', 'burst_ending', or 'regular'
                - reasoning: Human-readable explanation
        """
        burst_analysis = self.analyze_current_burst(df, timestamp_col)
        
        if burst_analysis.is_in_burst:
            if burst_analysis.prob_continuation > 0.5:
                # Burst likely to continue
                prediction_type = 'burst_continuation'
                expected_minutes = burst_analysis.expected_next_interval_minutes
                confidence = burst_analysis.prob_continuation
                reasoning = (
                    f"In active burst (length={burst_analysis.current_burst_length}). "
                    f"Typical burst length={burst_analysis.typical_burst_length:.1f}. "
                    f"Momentum={'accelerating' if burst_analysis.burst_momentum < 0 else 'decelerating'}. "
                    f"Expecting next post in ~{expected_minutes:.0f} minutes."
                )
            else:
                # Burst likely ending
                prediction_type = 'burst_ending'
                # After burst ends, expect longer gap
                features = self.extract_all(df, timestamp_col)
                avg_non_burst = features[features['is_burst_post'] == 0]['time_since_last_hours'].mean()
                expected_minutes = avg_non_burst * 60 if pd.notna(avg_non_burst) else 120
                confidence = 1.0 - burst_analysis.prob_continuation
                reasoning = (
                    f"Burst appears to be ending (length={burst_analysis.current_burst_length}, "
                    f"typical={burst_analysis.typical_burst_length:.1f}). "
                    f"Expecting post-burst gap of ~{expected_minutes:.0f} minutes."
                )
        else:
            # Not in burst
            prediction_type = 'regular'
            expected_minutes = burst_analysis.expected_next_interval_minutes
            
            # Check if conditions favor a new burst starting
            features = self.extract_all(df, timestamp_col)
            recent_activity = features['posts_last_1h'].iloc[-1] if 'posts_last_1h' in features.columns else 0
            
            if recent_activity >= 2:  # Multiple recent posts
                confidence = 0.5  # Moderate confidence - could start burst
                reasoning = f"Regular posting mode with {recent_activity} posts in last hour. Burst could start."
            else:
                confidence = 0.6  # Slightly higher confidence for regular prediction
                reasoning = f"Regular posting mode. Expected ~{expected_minutes:.0f} minutes until next post."
        
        return {
            'expected_minutes': expected_minutes,
            'confidence': confidence,
            'prediction_type': prediction_type,
            'reasoning': reasoning,
            'burst_analysis': burst_analysis
        }


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
    """Test historical feature extraction with enhanced burst analysis."""
    print("\n" + "="*80)
    print("HISTORICAL PATTERN FEATURE EXTRACTION TEST (Enhanced)")
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

    # Add another burst (3 posts rapidly)
    burst2_start = timestamps[-1] + timedelta(hours=1)
    for i in range(3):
        timestamps.append(burst2_start + timedelta(minutes=i*3))

    df = pd.DataFrame({'created_at': timestamps})

    print(f"Sample data: {len(df)} posts")
    print(f"Time range: {timestamps[0]} to {timestamps[-1]}")
    print()

    # Extract features
    extractor = HistoricalPatternExtractor()
    features = extractor.extract_all(df)

    print("Basic Burst Features (last 10 posts):")
    print(features[['time_since_last_hours', 'is_burst_post', 'burst_length',
                    'posts_last_1h', 'posts_last_6h']].tail(10))
    print()

    print("Enhanced Burst Features (last 10 posts):")
    enhanced_cols = ['burst_momentum_3', 'burst_intensity', 'prob_burst_continuation', 
                     'typical_burst_length', 'burst_completion_pct']
    available_cols = [c for c in enhanced_cols if c in features.columns]
    if available_cols:
        print(features[available_cols].tail(10).round(3))
    print()

    print(f"Total features extracted: {len(features.columns)}")
    print(f"Feature columns: {list(features.columns)}")
    print()

    # Get burst summary
    summary = extractor.get_burst_summary(df, n_recent=10)
    print("Burst Activity Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()

    # Test NEW burst analysis
    print("="*60)
    print("BURST ANALYSIS (Current State)")
    print("="*60)
    burst_analysis = extractor.analyze_current_burst(df)
    print(f"  Is in burst: {burst_analysis.is_in_burst}")
    print(f"  Current burst length: {burst_analysis.current_burst_length}")
    print(f"  Burst momentum: {burst_analysis.burst_momentum:.3f}")
    print(f"  Continuation probability: {burst_analysis.prob_continuation:.2%}")
    print(f"  Expected next interval: {burst_analysis.expected_next_interval_minutes:.1f} min")
    print(f"  Typical burst length: {burst_analysis.typical_burst_length:.1f}")
    print(f"  Burst intensity: {burst_analysis.burst_intensity:.1f} posts/hour")
    print()

    # Test NEW timing prediction
    print("="*60)
    print("TIMING PREDICTION (Based on Burst Analysis)")
    print("="*60)
    prediction = extractor.predict_next_post_timing(df)
    print(f"  Prediction type: {prediction['prediction_type']}")
    print(f"  Expected minutes: {prediction['expected_minutes']:.1f}")
    print(f"  Confidence: {prediction['confidence']:.2%}")
    print(f"  Reasoning: {prediction['reasoning']}")
    print()

    # Identify burst periods
    burst_posts = features[features['is_burst_post'] == 1]
    print(f"Detected {len(burst_posts)} burst posts (posts within {extractor.burst_threshold_minutes} minutes of previous)")
    print()

    print("="*80 + "\n")


if __name__ == "__main__":
    test_historical_features()
