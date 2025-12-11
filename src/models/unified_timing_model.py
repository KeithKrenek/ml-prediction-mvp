"""
Unified Timing Prediction Model

Supports both Prophet and Neural Temporal Point Process (NTPP) models.
Provides consistent interface regardless of underlying model type.

Usage:
    from src.models.unified_timing_model import UnifiedTimingPredictor

    predictor = UnifiedTimingPredictor(model_type='ntpp')  # or 'prophet'
    predictor.train(df)
    prediction = predictor.predict_next()
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from loguru import logger
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.database import get_session, Post
from features.engineering import FeatureEngineer
from models.timing_model import TimingPredictor

# Import model implementations
try:
    from prophet import Prophet
except ImportError:
    logger.warning("Prophet not installed")
    Prophet = None

try:
    import torch
    from models.ntpp_model import NTPPPredictor
except ImportError:
    logger.warning("PyTorch not installed. NTPP unavailable.")
    NTPPPredictor = None
    torch = None


class UnifiedTimingPredictor:
    """
    Unified interface for timing prediction models.

    Supports:
    - Prophet: Traditional time series with seasonality
    - NTPP: Neural network for point processes (better for bursts)
    """

    def __init__(
        self,
        model_type: str = 'ntpp',  # 'prophet' or 'ntpp'
        config_path: str = "config/config.yaml"
    ):
        """
        Initialize predictor.

        Args:
            model_type: 'prophet' or 'ntpp'
            config_path: Path to configuration file
        """
        self.model_type = model_type.lower()
        self.config_path = config_path
        self.config = self._load_config(config_path)

        self.feature_engineer = None
        self.model = None
        self.last_trained = None
        self.feature_cols = []
        self.prophet_engine = TimingPredictor(config_path=config_path)
        self.active_model_type = self.model_type

        logger.info(f"UnifiedTimingPredictor initialized with model_type='{self.model_type}'")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('timing_model', {})
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return {}

    def load_data_from_db(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load posts from database."""
        logger.info("Loading posts from database...")

        session = get_session()
        query = session.query(Post).order_by(Post.created_at.asc())

        if limit:
            query = query.limit(limit)

        posts = query.all()
        session.close()

        if not posts:
            logger.error("No posts found in database!")
            return None

        # Convert to DataFrame
        data = [{
            'post_id': p.post_id,
            'created_at': p.created_at,
            'content': p.content,
            'replies_count': p.replies_count,
            'reblogs_count': p.reblogs_count,
            'favourites_count': p.favourites_count
        } for p in posts]

        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} posts from {df['created_at'].min()} to {df['created_at'].max()}")

        return df

    def _get_context_snapshot(self, reference_time=None):
        """Load the closest context snapshot for feature enrichment."""
        from data.database import ContextSnapshot  # Local import to avoid circular deps
        session = get_session()
        try:
            query = session.query(ContextSnapshot).order_by(ContextSnapshot.captured_at.desc())
            if reference_time is not None:
                query = query.filter(ContextSnapshot.captured_at <= reference_time).order_by(ContextSnapshot.captured_at.desc())
            snapshot = query.first()
            if not snapshot:
                return None
            return {
                'top_headlines': snapshot.top_headlines,
                'political_news': snapshot.political_news,
                'news_summary': snapshot.news_summary,
                'trending_topics': snapshot.trending_topics,
                'trending_keywords': snapshot.trending_keywords,
                'trend_categories': snapshot.trend_categories,
                'market_sentiment': snapshot.market_sentiment,
                'sp500_change_pct': snapshot.sp500_change_pct,
                'dow_change_pct': snapshot.dow_change_pct,
                'completeness_score': snapshot.completeness_score,
                'freshness_score': snapshot.freshness_score
            }
        finally:
            session.close()

    def _auto_select_model_type(self, df: pd.DataFrame) -> str:
        """Determine which model to use based on recent posting rate."""
        if df is None or df.empty:
            return 'prophet'
        df_sorted = df.sort_values('created_at')
        window_start = df_sorted['created_at'].max() - timedelta(days=3)
        recent = df_sorted[df_sorted['created_at'] >= window_start]
        if recent.empty:
            recent = df_sorted
        duration_days = max((recent['created_at'].max() - recent['created_at'].min()).total_seconds() / 86400, 1)
        posts_per_day = len(recent) / duration_days
        threshold = self.config.get('auto_switch_threshold', 12)
        selected = 'ntpp' if posts_per_day >= threshold else 'prophet'
        logger.info(
            "Auto-selected timing model: %s (%.1f posts/day, threshold=%s)",
            selected.upper(),
            posts_per_day,
            threshold
        )
        return selected

    def prepare_features(
        self,
        df: pd.DataFrame,
        context: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Prepare features using FeatureEngineer.

        Args:
            df: DataFrame with post data
            context: Optional context dict

        Returns:
            DataFrame with engineered features
        """
        if self.feature_engineer is None:
            self.feature_engineer = FeatureEngineer(config_path=self.config_path)

        features_df = self.feature_engineer.engineer_features(
            df,
            context=context,
            timestamp_col='created_at'
        )

        return features_df

    def train(
        self,
        df: Optional[pd.DataFrame] = None,
        context: Optional[Dict] = None,
        **kwargs
    ) -> bool:
        """
        Train the model.

        Args:
            df: Optional DataFrame (loads from DB if None)
            context: Optional context dict
            **kwargs: Model-specific training parameters

        Returns:
            Success boolean
        """
        logger.info(f"Training {self.model_type.upper()} model...")

        # Load data if not provided
        if df is None:
            df = self.load_data_from_db()

        if df is None or len(df) < 50:
            logger.error("Insufficient data for training!")
            return False

        if self.model_type == 'auto':
            self.active_model_type = self._auto_select_model_type(df)
        else:
            self.active_model_type = self.model_type

        if self.active_model_type == 'prophet':
            trained = self.prophet_engine.train(df=df)
            if trained:
                self.model = self.prophet_engine
                self.last_trained = self.prophet_engine.last_trained
            return trained
        elif self.active_model_type == 'ntpp':
            context = context or self._get_context_snapshot(df['created_at'].max())
            features_df = self.prepare_features(df, context)
            trained = self._train_ntpp(features_df, **kwargs)
            if trained:
                self.last_trained = datetime.now()
            return trained
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _train_ntpp(
        self,
        features_df: pd.DataFrame,
        epochs: int = None,
        **kwargs
    ) -> bool:
        """Train NTPP model."""
        if NTPPPredictor is None or torch is None:
            logger.error("PyTorch/NTPP not available!")
            return False

        # Get NTPP config
        ntpp_config = self.config.get('neural_tpp', {})
        if epochs is None:
            epochs = ntpp_config.get('epochs', 50)

        # Select features for NTPP
        # NTPP can use all features (not limited like Prophet)
        self.feature_cols = [
            col for col in features_df.columns
            if col not in ['post_id', 'created_at', 'content'] and
            features_df[col].dtype in [np.float64, np.int64, np.float32, np.int32]
        ]

        # Filter to most important features (reduce dimensionality)
        priority_features = [
            'hour_sin', 'hour_cos',
            'day_of_week_sin', 'day_of_week_cos',
            'is_weekend', 'is_business_hours',
            'engagement_total', 'engagement_rolling_mean_5',
            'time_since_last_hours',
            'posts_last_1h', 'posts_last_3h', 'posts_last_6h',
            'is_burst_post', 'burst_length',
            'posting_regularity'
        ]

        self.feature_cols = [f for f in priority_features if f in features_df.columns]

        logger.info(f"Using {len(self.feature_cols)} features for NTPP")

        # Initialize NTPP
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = NTPPPredictor(
            feature_dim=len(self.feature_cols),
            hidden_dim=ntpp_config.get('hidden_size', 64),
            num_layers=ntpp_config.get('num_layers', 2),
            learning_rate=ntpp_config.get('learning_rate', 0.001),
            device=device
        )

        # Train
        history = self.model.train(
            features_df,
            feature_cols=self.feature_cols,
            epochs=epochs,
            batch_size=ntpp_config.get('batch_size', 32),
            sequence_length=ntpp_config.get('sequence_length', 20)
        )

        self.last_trained = datetime.now()
        logger.success(f"NTPP training complete! Final val_loss: {history['val_loss'][-1]:.4f}")

        return True

    def predict_next(self, context: Optional[Dict] = None) -> Optional[Dict]:
        """
        Predict next post time.

        Args:
            context: Optional context dict

        Returns:
            Prediction dict or None
        """
        if self.active_model_type == 'prophet':
            return self.prophet_engine.get_next_post_time()

        if self.model is None:
            logger.error("Model not trained!")
            return None

        # Load recent data
        df = self.load_data_from_db(limit=100)  # Last 100 posts
        if df is None or len(df) == 0:
            logger.error("No data available for prediction!")
            return None

        # Engineer features
        features_df = self.prepare_features(df, context)

        if self.active_model_type == 'ntpp':
            return self._predict_ntpp(features_df)

        raise ValueError(f"Unknown model type: {self.active_model_type}")

    def _predict_ntpp(self, features_df: pd.DataFrame) -> Optional[Dict]:
        """Make prediction with NTPP including uncertainty quantification."""
        ntpp_config = self.config.get('neural_tpp', {})
        
        prediction = self.model.predict(
            features_df,
            feature_cols=self.feature_cols,
            sequence_length=ntpp_config.get('sequence_length', 20),
            n_samples=100,
            mc_dropout_samples=5,
            return_structured=False  # Return dict for compatibility
        )
        
        # Add lower/upper bounds for consistency with Prophet output
        if prediction and 'ci_lower_hours' in prediction:
            last_post_time = features_df['created_at'].iloc[-1]
            if hasattr(last_post_time, 'to_pydatetime'):
                last_post_time = last_post_time.to_pydatetime()
            prediction['lower_bound'] = prediction['ci_lower_hours']
            prediction['upper_bound'] = prediction['ci_upper_hours']

        return prediction

    def save(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        target_type = self.active_model_type
        if target_type == 'prophet':
            self.prophet_engine.save(path)
        elif target_type == 'ntpp':
            self.model.save(path)

        logger.success(f"Model saved to {path}")

    def load(self, path: str) -> bool:
        """Load model from disk."""
        try:
            target_type = self.active_model_type
            if target_type == 'prophet':
                loaded = self.prophet_engine.load(path)
                if loaded:
                    self.model = self.prophet_engine
                    self.last_trained = self.prophet_engine.last_trained
                return loaded
            elif target_type == 'ntpp':
                # Initialize model first
                ntpp_config = self.config.get('neural_tpp', {})
                device = 'cuda' if torch and torch.cuda.is_available() else 'cpu'
                self.model = NTPPPredictor(
                    feature_dim=len(self.feature_cols) if self.feature_cols else 15,
                    hidden_dim=ntpp_config.get('hidden_size', 64),
                    device=device
                )
                self.model.load(path)

            logger.success(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


def main():
    """Test unified timing predictor."""
    print("\n" + "="*80)
    print("UNIFIED TIMING PREDICTOR TEST")
    print("="*80 + "\n")

    # Test with NTPP
    print("Testing NTPP model...")
    predictor = UnifiedTimingPredictor(model_type='ntpp')

    # Train (will use synthetic data if no DB)
    print("Training model...")
    success = predictor.train()

    if success:
        print("\nMaking prediction...")
        prediction = predictor.predict_next()

        if prediction:
            print(f"\nPrediction:")
            print(f"  Time: {prediction['predicted_time']}")
            print(f"  Confidence: {prediction['confidence']:.2f}")
            print(f"  Model: {prediction['model_version']}")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
