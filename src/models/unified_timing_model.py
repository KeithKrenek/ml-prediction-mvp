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

        # Engineer features
        features_df = self.prepare_features(df, context)

        if self.model_type == 'prophet':
            return self._train_prophet(features_df, **kwargs)
        elif self.model_type == 'ntpp':
            return self._train_ntpp(features_df, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _train_prophet(
        self,
        features_df: pd.DataFrame,
        **kwargs
    ) -> bool:
        """Train Prophet model."""
        if Prophet is None:
            logger.error("Prophet not available!")
            return False

        # Get regressors from config
        feature_config = self.config.get('feature_engineering', {})
        prophet_regressors = feature_config.get('prophet_regressors', [
            'hour_sin', 'hour_cos',
            'is_weekend',
            'engagement_total',
            'time_since_last_hours',
            'posts_last_1h',
            'is_burst_post'
        ])

        # Prepare Prophet DataFrame
        prophet_df = self.feature_engineer.prepare_for_prophet(
            features_df,
            timestamp_col='created_at',
            additional_regressors=prophet_regressors
        )

        # Fill NaN values
        for regressor in prophet_regressors:
            if regressor in prophet_df.columns:
                prophet_df[regressor] = prophet_df[regressor].fillna(0)

        # Initialize Prophet
        prophet_config = self.config.get('prophet', {})
        self.model = Prophet(
            daily_seasonality=prophet_config.get('daily_seasonality', True),
            weekly_seasonality=prophet_config.get('weekly_seasonality', True),
            yearly_seasonality=prophet_config.get('yearly_seasonality', False),
            changepoint_prior_scale=prophet_config.get('changepoint_prior_scale', 0.05),
            interval_width=prophet_config.get('interval_width', 0.95)
        )

        # Add regressors
        for regressor in prophet_regressors:
            if regressor in prophet_df.columns:
                self.model.add_regressor(regressor)

        # Train
        logger.info(f"Training Prophet with {len(prophet_regressors)} regressors...")
        self.model.fit(prophet_df)

        self.last_trained = datetime.now()
        self.feature_cols = prophet_regressors
        logger.success("Prophet training complete!")

        return True

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

        if self.model_type == 'prophet':
            return self._predict_prophet(features_df)
        elif self.model_type == 'ntpp':
            return self._predict_ntpp(features_df)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _predict_prophet(self, features_df: pd.DataFrame) -> Optional[Dict]:
        """Make prediction with Prophet."""
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=48, freq='H')

        # Add regressor values for future (use last known values)
        for regressor in self.feature_cols:
            if regressor in features_df.columns:
                last_value = features_df[regressor].iloc[-1]
                future[regressor] = last_value

        # Predict
        forecast = self.model.predict(future)

        # Get predictions for future only
        future_forecast = forecast[forecast['ds'] > features_df['created_at'].max()]

        if len(future_forecast) == 0:
            return None

        # Find peak (most likely time)
        peak_idx = future_forecast['yhat'].idxmax()
        next_post = future_forecast.loc[peak_idx]

        confidence = 1.0 - (next_post['yhat_upper'] - next_post['yhat_lower'])
        confidence = max(0.0, min(1.0, confidence))

        return {
            'predicted_time': next_post['ds'],
            'confidence': confidence,
            'model_version': f'prophet_{self.model_type}_v1',
            'trained_at': self.last_trained
        }

    def _predict_ntpp(self, features_df: pd.DataFrame) -> Optional[Dict]:
        """Make prediction with NTPP."""
        prediction = self.model.predict(
            features_df,
            feature_cols=self.feature_cols,
            sequence_length=20
        )

        return prediction

    def save(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        if self.model_type == 'prophet':
            import pickle
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'model_type': self.model_type,
                    'feature_cols': self.feature_cols,
                    'last_trained': self.last_trained
                }, f)
        elif self.model_type == 'ntpp':
            self.model.save(path)

        logger.success(f"Model saved to {path}")

    def load(self, path: str) -> bool:
        """Load model from disk."""
        try:
            if self.model_type == 'prophet':
                import pickle
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                self.model = data['model']
                self.feature_cols = data.get('feature_cols', [])
                self.last_trained = data.get('last_trained')
            elif self.model_type == 'ntpp':
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
