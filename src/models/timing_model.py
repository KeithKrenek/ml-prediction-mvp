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
from data.database import get_session, Post
from features.engineering import FeatureEngineer

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

        # Prepare Prophet DataFrame
        prophet_df = self.feature_engineer.prepare_for_prophet(
            features_df,
            timestamp_col='created_at',
            additional_regressors=self.enabled_regressors
        )

        # Fill any NaN values in regressors
        for regressor in self.enabled_regressors:
            if regressor in prophet_df.columns:
                prophet_df[regressor] = prophet_df[regressor].fillna(0)

        logger.info(f"Prepared features with {len(self.enabled_regressors)} regressors: {self.enabled_regressors[:5]}...")

        return prophet_df
    
    def train(self, df=None):
        """Train Prophet model on historical data"""
        logger.info("Training timing prediction model...")
        
        if df is None:
            df = self.load_data_from_db()
        
        if df is None or len(df) < 10:
            logger.error("Insufficient data for training!")
            return False
        
        # Prepare features
        prophet_df = self.prepare_features(df)
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
        logger.info(f"Training on {len(prophet_df)} posts...")
        self.model.fit(prophet_df)
        
        self.last_trained = datetime.now()
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

        # Add regressor values to future dataframe
        # Extract regressors from the model (they were added during training)
        if hasattr(self, 'enabled_regressors') and self.enabled_regressors:
            import numpy as np

            # Add regressor values based on timestamp
            for col in self.enabled_regressors:
                if col in ['hour_sin', 'hour_cos']:
                    # Cyclical hour encoding
                    hours = future['ds'].dt.hour
                    if col == 'hour_sin':
                        future[col] = np.sin(2 * np.pi * hours / 24)
                    else:
                        future[col] = np.cos(2 * np.pi * hours / 24)

                elif col in ['day_of_week_sin', 'day_of_week_cos']:
                    # Cyclical day of week encoding
                    dow = future['ds'].dt.dayofweek
                    if col == 'day_of_week_sin':
                        future[col] = np.sin(2 * np.pi * dow / 7)
                    else:
                        future[col] = np.cos(2 * np.pi * dow / 7)

                elif col == 'is_weekend':
                    future[col] = (future['ds'].dt.dayofweek >= 5).astype(int)

                elif col == 'is_business_hours':
                    hour = future['ds'].dt.hour
                    dow = future['ds'].dt.dayofweek
                    future[col] = ((hour >= 9) & (hour <= 17) & (dow < 5)).astype(int)

                else:
                    # For other regressors (engagement, historical patterns), use mean from training
                    if col in self.features_df.columns:
                        future[col] = self.features_df[col].mean()
                    else:
                        future[col] = 0.0

        # Make predictions
        forecast = self.model.predict(future)

        # Get predictions for future only
        future_forecast = forecast[forecast['ds'] > self.features_df['ds'].max()]

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
        
        # Calculate confidence based on prediction interval width
        confidence = 1.0 - (next_post['yhat_upper'] - next_post['yhat_lower'])
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        
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
    
    def evaluate(self, test_df):
        """
        Evaluate model performance on test data.
        
        Args:
            test_df: DataFrame with actual post times
            
        Returns:
            dict with evaluation metrics
        """
        # Calculate MAE and accuracy within time windows
        predictions = []
        actuals = []
        
        for idx, row in test_df.iterrows():
            # Predict based on data up to this point
            # (In practice, you'd retrain on rolling window)
            pred = self.get_next_post_time()
            if pred:
                predictions.append(pred['predicted_time'])
                actuals.append(row['created_at'])
        
        if not predictions:
            return None
        
        # Calculate metrics
        errors = [abs((p - a).total_seconds() / 3600) for p, a in zip(predictions, actuals)]
        mae = np.mean(errors)
        
        within_6h = sum(1 for e in errors if e <= 6) / len(errors)
        within_24h = sum(1 for e in errors if e <= 24) / len(errors)
        
        metrics = {
            'mae_hours': mae,
            'within_6h_accuracy': within_6h,
            'within_24h_accuracy': within_24h,
            'num_predictions': len(predictions)
        }
        
        logger.info(f"Evaluation metrics: MAE={mae:.2f}h, 6h accuracy={within_6h:.2%}, 24h accuracy={within_24h:.2%}")
        
        return metrics
    
    def save(self, path="models/timing_model.pkl"):
        """Save trained model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'config': self.config,
                'last_trained': self.last_trained,
                'features_df': self.features_df
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
