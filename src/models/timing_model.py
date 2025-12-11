"""
Timing Prediction Model using Facebook Prophet.
Simple, effective baseline that handles seasonality automatically.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from pathlib import Path
from loguru import logger
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.database import get_session, Post, ContextSnapshot
from features.engineering import FeatureEngineer
from models.evaluator import ModelEvaluator
from models.model_registry import ModelRegistry


try:
    from prophet import Prophet
except ImportError:
    logger.warning("Prophet not installed. Run: pip install prophet")
    Prophet = None


class TimingPredictor:
    """
    Predicts when Trump will post next using Prophet.
    
    Design: Modular interface that can be swapped with Neural TPP later.
    """
    
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.model = None
        self.last_trained = None
        self.features_df = None
        self.feature_engineer = None
        self.enabled_regressors = []
        self.residual_stats = {'mae': None, 'std': None}
        self.latest_context_features = {}
        self.registry = ModelRegistry()
        
    def _load_config(self, config_path):
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config['timing_model']
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using defaults.")
            return {
                'type': 'prophet',
                'prophet': {
                    'daily_seasonality': True,
                    'weekly_seasonality': True,
                    'yearly_seasonality': False
                }
            }
    
    def load_data_from_db(self, limit=None):
        """Load posts from database"""
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
            'replies': p.replies_count,
            'reblogs': p.reblogs_count,
            'favourites': p.favourites_count
        } for p in posts]
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} posts from {df['created_at'].min()} to {df['created_at'].max()}")
        
        return df
    
    def prepare_features(self, df, context=None):
        """
        Prepare features for timing prediction using FeatureEngineer.
        Prophet needs: ds (datetime), y (target)
        Plus additional regressors from feature engineering.

        Args:
            df: DataFrame with post data
            context: Optional context dict for context features
        """
        # Initialize feature engineer if not already done
        if self.feature_engineer is None:
            self.feature_engineer = FeatureEngineer(config_path=self.config_path)

        # Standardize column names
        df_standardized = df.copy()
        if 'replies' in df.columns:
            df_standardized['replies_count'] = df['replies']
        if 'reblogs' in df.columns:
            df_standardized['reblogs_count'] = df['reblogs']
        if 'favourites' in df.columns:
            df_standardized['favourites_count'] = df['favourites']

        # Engineer all features
        features_df = self.feature_engineer.engineer_features(
            df_standardized,
            context=context,
            timestamp_col='created_at'
        )

        # Get list of regressors to use from config
        feature_config = self.config.get('feature_engineering', {})
        self.enabled_regressors = feature_config.get('prophet_regressors', [
            'hour_sin', 'hour_cos',
            'day_of_week_sin', 'day_of_week_cos',
            'is_weekend',
            'is_business_hours',
            'engagement_total',
            'time_since_last_hours',
            'posts_last_1h',
            'posts_last_6h',
            'is_burst_post'
        ])

        logger.info(f"Prepared features with {len(self.enabled_regressors)} regressors: {self.enabled_regressors[:5]}...")

        return features_df

    def _get_context_snapshot(self, reference_time=None):
        """Fetch the most recent context snapshot relative to the reference time."""
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

    def _compute_residual_stats(self):
        """Calibrate residual distribution for confidence scoring."""
        if self.model is None or self.features_df is None or self.features_df.empty:
            self.residual_stats = {'mae': None, 'std': None}
            return

        regressors = [reg for reg in self.enabled_regressors if reg in self.features_df.columns]
        history = self.features_df[['ds', 'y'] + regressors]
        forecast = self.model.predict(history[['ds'] + regressors])
        residuals = history['y'] - forecast['yhat']
        mae = float(np.mean(np.abs(residuals))) if len(residuals) else None
        std = float(np.std(residuals)) if len(residuals) else None
        self.residual_stats = {'mae': mae, 'std': std}

    def _extract_context_features(self, context_snapshot):
        """Convert a context snapshot into feature columns."""
        if context_snapshot is None or self.feature_engineer is None:
            return {}
        extractor = getattr(self.feature_engineer, 'context_extractor', None)
        if extractor is None:
            return {}
        try:
            return extractor.extract_from_context_snapshot(context_snapshot)
        except Exception as exc:
            logger.warning(f"Failed to extract context features: {exc}")
            return {}

    def _add_future_regressors(self, future_df, context_features=None):
        """Ensure future dataframe contains all regressors."""
        if not self.enabled_regressors:
            return future_df

        ds_series = future_df['ds']
        hours = ds_series.dt.hour
        days = ds_series.dt.dayofweek

        for regressor in self.enabled_regressors:
            if regressor in future_df.columns:
                continue
            if regressor == 'hour_sin':
                future_df[regressor] = np.sin(2 * np.pi * hours / 24)
            elif regressor == 'hour_cos':
                future_df[regressor] = np.cos(2 * np.pi * hours / 24)
            elif regressor == 'day_of_week_sin':
                future_df[regressor] = np.sin(2 * np.pi * days / 7)
            elif regressor == 'day_of_week_cos':
                future_df[regressor] = np.cos(2 * np.pi * days / 7)
            elif regressor == 'is_weekend':
                future_df[regressor] = (days >= 5).astype(int)
            elif regressor == 'is_business_hours':
                future_df[regressor] = ((hours >= 9) & (hours <= 17) & (days < 5)).astype(int)
            else:
                if context_features and regressor in context_features:
                    future_df[regressor] = context_features[regressor]
                elif self.features_df is not None and regressor in self.features_df.columns:
                    future_df[regressor] = self.features_df[regressor].iloc[-1]
                else:
                    future_df[regressor] = 0.0

        return future_df
    
    def train(self, df=None):
        """Train Prophet model on historical data"""
        logger.info("Training timing prediction model...")
        
        if df is None:
            df = self.load_data_from_db()
        
        if df is None or len(df) < 10:
            logger.error("Insufficient data for training!")
            return False
        
        # Prepare features
        context_snapshot = self._get_context_snapshot(df['created_at'].max() if 'created_at' in df.columns else None)
        resample_freq = self.config.get('prophet', {}).get('resample_freq', 'H')
        features_df = self.prepare_features(df, context=context_snapshot)
        self.latest_context_features = self._extract_context_features(context_snapshot)
        prophet_df = self.feature_engineer.prepare_for_prophet(
            features_df,
            timestamp_col='created_at',
            additional_regressors=self.enabled_regressors or None,
            resample_freq=resample_freq
        )
        for regressor in self.enabled_regressors:
            if regressor in prophet_df.columns:
                prophet_df[regressor] = prophet_df[regressor].fillna(method='ffill').fillna(0)
        self.features_df = prophet_df
        
        # Initialize Prophet with configuration
        prophet_config = self.config.get('prophet', {})
        self.model = Prophet(
            daily_seasonality=prophet_config.get('daily_seasonality', True),
            weekly_seasonality=prophet_config.get('weekly_seasonality', True),
            yearly_seasonality=prophet_config.get('yearly_seasonality', False),
            changepoint_prior_scale=prophet_config.get('changepoint_prior_scale', 0.05),
            interval_width=prophet_config.get('interval_width', 0.95)
        )

        # Add all enabled regressors to Prophet model
        for regressor in self.enabled_regressors:
            if regressor in prophet_df.columns:
                self.model.add_regressor(regressor)
                logger.debug(f"Added regressor: {regressor}")

        logger.info(f"Training Prophet with {len(self.enabled_regressors)} regressors...")

        # Fit model
        logger.info(f"Training on {len(prophet_df)} hourly points...")
        self.model.fit(prophet_df[['ds', 'y'] + [reg for reg in self.enabled_regressors if reg in prophet_df.columns]])
        
        self.last_trained = datetime.now()
        self._compute_residual_stats()
        logger.success("Model training complete!")
        
        return True
    
    def predict_next(self, periods_ahead=24):
        """
        Predict next post time(s).

        Args:
            periods_ahead: Number of hours to forecast

        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            logger.error("Model not trained! Call train() first.")
            return None

        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods_ahead, freq='H')
        context_snapshot = self._get_context_snapshot()
        context_features = self._extract_context_features(context_snapshot)
        if not context_features:
            context_features = self.latest_context_features
        future = self._add_future_regressors(future, context_features=context_features)

        # Make predictions
        forecast = self.model.predict(future)

        # Get predictions for future only
        last_training_point = self.features_df['ds'].max() if self.features_df is not None else datetime.now()
        future_forecast = forecast[forecast['ds'] > last_training_point]

        return future_forecast
    
    def get_next_post_time(self):
        """
        Get the most likely time for the next post.
        
        Returns:
            dict with prediction details
        """
        forecast = self.predict_next(periods_ahead=48)
        
        if forecast is None or len(forecast) == 0:
            return None
        
        # Find peak in the forecast (highest yhat value)
        peak_idx = forecast['yhat'].idxmax()
        next_post = forecast.loc[peak_idx]
        
        # Calibrated confidence using residual statistics
        mae = self.residual_stats.get('mae') if self.residual_stats else None
        interval_half_width = max(1e-6, (next_post['yhat_upper'] - next_post['yhat_lower']) / 2)
        if mae is None or mae == 0:
            confidence = 0.5
        else:
            uncertainty_ratio = interval_half_width / (mae + 1e-6)
            confidence = 1 / (1 + uncertainty_ratio)
        confidence = max(0.05, min(0.99, confidence))
        
        result = {
            'predicted_time': next_post['ds'],
            'confidence': confidence,
            'lower_bound': next_post['yhat_lower'],
            'upper_bound': next_post['yhat_upper'],
            'model_version': 'prophet_v1',
            'trained_at': self.last_trained
        }
        
        logger.info(f"Next post predicted at: {result['predicted_time']} (confidence: {confidence:.2f})")
        
        return result
    
    def evaluate(
        self,
        df=None,
        min_train_size: int = 100,
        step_size: int = 1,
        max_evaluations: int = 25
    ):
        """
        Rolling-origin evaluation that retrains on expanding windows and
        persists metrics to the model registry.

        Args:
            df: Optional DataFrame with posts (defaults to DB data)
            min_train_size: Minimum posts before starting evaluation
            step_size: How many samples to skip between evaluations
            max_evaluations: Max number of rolling forecasts

        Returns:
            dict with evaluation metrics
        """
        if df is None:
            df = self.load_data_from_db()
        if df is None or len(df) <= min_train_size:
            logger.warning("Not enough data to run evaluation")
            return None

        predictions = []
        actuals = []
        evaluator = ModelEvaluator()

        evaluation_indices = list(range(min_train_size, len(df), step_size))
        for idx in evaluation_indices:
            if len(predictions) >= max_evaluations or idx >= len(df):
                break
            train_slice = df.iloc[:idx].copy()
            holdout_row = df.iloc[idx]

            temp_model = TimingPredictor(config_path=self.config_path)
            if not temp_model.train(df=train_slice):
                continue
            forecast = temp_model.get_next_post_time()
            if not forecast:
                continue

            predictions.append({
                'predicted_time': forecast['predicted_time'],
                'confidence': forecast['confidence']
            })
            actuals.append({
                'actual_time': holdout_row['created_at'],
                'post_id': holdout_row['post_id']
            })

        if not predictions:
            logger.warning("Evaluation produced no predictions")
            return None

        metrics = evaluator.summarize_timing_predictions(predictions, actuals)
        logger.info(
            "Evaluation metrics: MAE=%.2fh | within_6h=%.1f%% | within_24h=%.1f%%",
            metrics['mae_hours'],
            metrics['within_6h_accuracy'] * 100,
            metrics['within_24h_accuracy'] * 100
        )

        # Persist metrics via model registry if a production model exists
        production_model = self.registry.get_production_model('timing')
        if production_model:
            try:
                self.registry.evaluate_model(
                    production_model.version_id,
                    predictions,
                    actuals,
                    eval_dataset_start=df['created_at'].min(),
                    eval_dataset_end=df['created_at'].max()
                )
            except Exception as exc:
                logger.warning(f"Failed to persist evaluation metrics: {exc}")

        return metrics
    
    def save(self, path="models/timing_model.pkl"):
        """Save trained model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'config': self.config,
                'last_trained': self.last_trained,
                'features_df': self.features_df,
                'residual_stats': self.residual_stats,
                'latest_context_features': self.latest_context_features
            }, f)
        
        logger.success(f"Model saved to {path}")
    
    def load(self, path="models/timing_model.pkl"):
        """Load trained model"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            self.model = data['model']
            self.config = data['config']
            self.last_trained = data['last_trained']
            self.features_df = data['features_df']
            self.residual_stats = data.get('residual_stats', {'mae': None, 'std': None})
            self.latest_context_features = data.get('latest_context_features', {})
            
            logger.success(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


def main():
    """Test the timing predictor"""
    logger.info("Testing Timing Predictor...")
    
    predictor = TimingPredictor()
    
    # Train model
    success = predictor.train()
    
    if success:
        # Get next prediction
        prediction = predictor.get_next_post_time()
        
        if prediction:
            print("\n" + "="*50)
            print("NEXT POST PREDICTION")
            print("="*50)
            print(f"Predicted Time: {prediction['predicted_time']}")
            print(f"Confidence: {prediction['confidence']:.2%}")
            print(f"Model Version: {prediction['model_version']}")
            print("="*50)
        
        # Save model
        predictor.save()


if __name__ == "__main__":
    main()
