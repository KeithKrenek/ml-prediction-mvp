"""
Model Evaluator - Comprehensive evaluation for timing and content models.

Provides cross-validation, test set evaluation, and performance tracking.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from loguru import logger

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import get_session, Post, Prediction


class ModelEvaluator:
    """
    Evaluate timing and content models with train/test splits.
    """

    def __init__(self):
        pass

    def get_evaluation_data(
        self,
        train_split: float = 0.8,
        min_samples: int = 50
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get train/test split of historical posts for evaluation.

        Args:
            train_split: Fraction of data for training (default: 0.8)
            min_samples: Minimum number of samples required

        Returns:
            (train_df, test_df) DataFrames
        """
        session = get_session()
        try:
            # Get all posts ordered by time
            posts = session.query(Post).order_by(Post.created_at.asc()).all()

            if len(posts) < min_samples:
                logger.warning(f"Insufficient data: {len(posts)} < {min_samples}")
                return None, None

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

            # Split by time (not random!)
            split_idx = int(len(df) * train_split)
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()

            logger.info(f"Split data: {len(train_df)} train, {len(test_df)} test")
            logger.info(f"Train period: {train_df['created_at'].min()} to {train_df['created_at'].max()}")
            logger.info(f"Test period: {test_df['created_at'].min()} to {test_df['created_at'].max()}")

            return train_df, test_df

        finally:
            session.close()

    def evaluate_timing_model(
        self,
        model,
        test_df: pd.DataFrame,
        max_predictions: int = 20
    ) -> Dict:
        """
        Evaluate timing model on test set.

        Simulates making predictions and compares with actual post times.

        Args:
            model: Trained TimingPredictor instance
            test_df: Test DataFrame with actual posts
            max_predictions: Maximum number of predictions to make

        Returns:
            Dict with evaluation metrics
        """
        logger.info(f"Evaluating timing model on {len(test_df)} test samples...")

        predictions = []
        actuals = []
        errors = []

        # Simplified evaluation: Use the trained model's forecast ability
        # The model was trained on train_df and we're testing on test_df
        # For Prophet, we'll use the forecast it already made during training

        # Since Prophet forecasts from training data, we'll evaluate by:
        # 1. Getting the model's prediction range
        # 2. Comparing test set times to predictions

        try:
            # Get predictions for the test period
            forecast = model.predict_next(periods_ahead=len(test_df) * 2)

            if forecast is None or len(forecast) == 0:
                logger.error("No forecast generated")
                return None

            # For each test post, find closest prediction
            import pandas as pd

            for i in range(min(len(test_df), max_predictions)):
                actual_time = test_df.iloc[i]['created_at']

                # Convert to pandas Timestamp for consistent handling
                if not isinstance(actual_time, pd.Timestamp):
                    actual_time = pd.Timestamp(actual_time)

                # Ensure timezone compatibility between forecast and actual
                # Remove timezone info from both to avoid comparison issues
                if forecast['ds'].dt.tz is not None:
                    forecast_times = forecast['ds'].dt.tz_localize(None)
                else:
                    forecast_times = forecast['ds']

                if hasattr(actual_time, 'tz') and actual_time.tz is not None:
                    actual_time_naive = actual_time.tz_localize(None)
                elif hasattr(actual_time, 'tzinfo') and actual_time.tzinfo is not None:
                    actual_time_naive = actual_time.replace(tzinfo=None)
                else:
                    actual_time_naive = actual_time

                # Find closest forecast time (using naive datetimes)
                time_diffs = abs((forecast_times - actual_time_naive).dt.total_seconds())
                closest_idx = time_diffs.idxmin()
                closest_forecast = forecast.loc[closest_idx]

                # Calculate error (use naive times to avoid timezone issues)
                pred_time_naive = pd.Timestamp(closest_forecast['ds']).tz_localize(None) if hasattr(pd.Timestamp(closest_forecast['ds']), 'tz') else pd.Timestamp(closest_forecast['ds'])
                error_hours = abs((pred_time_naive - actual_time_naive).total_seconds() / 3600)

                predictions.append({
                    'predicted_time': closest_forecast['ds'],
                    'confidence': 0.7,  # Default confidence for now
                    'model_version': 'prophet_v1'
                })
                actuals.append({
                    'actual_time': actual_time,
                    'post_id': test_df.iloc[i]['post_id']
                })
                errors.append(error_hours)

                logger.debug(f"Prediction {i+1}: Error = {error_hours:.2f}h")

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None

        if not errors:
            logger.error("No successful predictions made")
            return None

        # Calculate metrics
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        median_error = np.median(errors)
        std_error = np.std(errors)

        within_6h = sum(1 for e in errors if e <= 6) / len(errors)
        within_12h = sum(1 for e in errors if e <= 12) / len(errors)
        within_24h = sum(1 for e in errors if e <= 24) / len(errors)
        within_48h = sum(1 for e in errors if e <= 48) / len(errors)

        # Calculate overall score (0-1, higher is better)
        # Weighted combination: 70% within_6h accuracy + 30% normalized MAE
        normalized_mae = max(0, 1 - (mae / 24))  # 24h -> 0, 0h -> 1
        overall_score = 0.7 * within_6h + 0.3 * normalized_mae

        metrics = {
            'mae_hours': mae,
            'rmse_hours': rmse,
            'median_error_hours': median_error,
            'std_error_hours': std_error,
            'within_6h_accuracy': within_6h,
            'within_12h_accuracy': within_12h,
            'within_24h_accuracy': within_24h,
            'within_48h_accuracy': within_48h,
            'overall_score': overall_score,
            'num_predictions': len(predictions),
            'predictions': predictions,
            'actuals': actuals
        }

        logger.success(f"Timing model evaluation complete!")
        logger.info(f"MAE: {mae:.2f}h | RMSE: {rmse:.2f}h | Median: {median_error:.2f}h")
        logger.info(f"Within 6h: {within_6h:.1%} | 12h: {within_12h:.1%} | 24h: {within_24h:.1%}")
        logger.info(f"Overall score: {overall_score:.4f}")

        return metrics

    def summarize_timing_predictions(
        self,
        predictions: List[Dict],
        actuals: List[Dict]
    ) -> Dict:
        """
        Compute timing metrics from matched prediction/actual pairs.

        Args:
            predictions: List of dicts with 'predicted_time'
            actuals: List of dicts with 'actual_time'

        Returns:
            Dict with MAE and within-window accuracy metrics.
        """
        paired_errors = []
        for pred, actual in zip(predictions, actuals):
            pred_time = pred.get('predicted_time')
            actual_time = actual.get('actual_time')
            if pred_time is None or actual_time is None:
                continue
            if not isinstance(pred_time, pd.Timestamp):
                pred_time = pd.to_datetime(pred_time)
            if not isinstance(actual_time, pd.Timestamp):
                actual_time = pd.to_datetime(actual_time)
            if getattr(pred_time, 'tzinfo', None):
                pred_time = pred_time.tz_convert(None)
            if getattr(actual_time, 'tzinfo', None):
                actual_time = actual_time.tz_convert(None)
            paired_errors.append(abs((pred_time - actual_time).total_seconds()) / 3600)

        if not paired_errors:
            return {
                'mae_hours': None,
                'within_6h_accuracy': 0,
                'within_24h_accuracy': 0,
                'num_predictions': 0
            }

        mae = float(np.mean(paired_errors))
        within_6h = sum(1 for e in paired_errors if e <= 6) / len(paired_errors)
        within_24h = sum(1 for e in paired_errors if e <= 24) / len(paired_errors)

        return {
            'mae_hours': mae,
            'within_6h_accuracy': within_6h,
            'within_24h_accuracy': within_24h,
            'num_predictions': len(paired_errors)
        }

    def evaluate_content_model(
        self,
        generator,
        test_df: pd.DataFrame,
        max_predictions: int = 20
    ) -> Dict:
        """
        Evaluate content model on test set.

        Args:
            generator: Trained ContentGenerator instance
            test_df: Test DataFrame with actual posts
            max_predictions: Maximum number of predictions to make

        Returns:
            Dict with evaluation metrics
        """
        logger.info(f"Evaluating content model on {len(test_df)} test samples...")

        predictions = []
        actuals = []
        similarities = []

        # For each post in test set, generate content and compare
        for i in range(min(len(test_df), max_predictions)):
            try:
                actual_content = test_df.iloc[i]['content']
                actual_time = test_df.iloc[i]['created_at']

                # Generate content
                generated = generator.generate(
                    context=None,
                    predicted_time=actual_time
                )

                if not generated:
                    continue

                pred_content = generated['content']

                # Calculate simple similarity metrics
                # (In production, would use BERTScore)
                length_sim = 1 - abs(len(pred_content) - len(actual_content)) / max(len(pred_content), len(actual_content))

                # Word overlap similarity
                pred_words = set(pred_content.lower().split())
                actual_words = set(actual_content.lower().split())
                if pred_words or actual_words:
                    word_overlap = len(pred_words & actual_words) / len(pred_words | actual_words)
                else:
                    word_overlap = 0.0

                # Combined similarity
                similarity = (length_sim + word_overlap) / 2

                predictions.append({
                    'predicted_content': pred_content,
                    'predicted_at': datetime.now()
                })
                actuals.append({
                    'actual_content': actual_content,
                    'post_id': test_df.iloc[i]['post_id']
                })
                similarities.append(similarity)

                logger.debug(f"Sample {i+1}: Similarity = {similarity:.3f}")

            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {e}")
                continue

        if not similarities:
            logger.error("No successful evaluations")
            return None

        # Calculate metrics
        avg_similarity = np.mean(similarities)
        median_similarity = np.median(similarities)
        std_similarity = np.std(similarities)

        # Overall score (0-1, higher is better)
        overall_score = avg_similarity

        metrics = {
            'avg_similarity': avg_similarity,
            'median_similarity': median_similarity,
            'std_similarity': std_similarity,
            'min_similarity': min(similarities),
            'max_similarity': max(similarities),
            'overall_score': overall_score,
            'num_predictions': len(predictions),
            'predictions': predictions,
            'actuals': actuals,
            # Placeholder for future metrics
            'bertscore_f1': None,
            'bleu_score': None
        }

        logger.success(f"Content model evaluation complete!")
        logger.info(f"Avg similarity: {avg_similarity:.3f} | Median: {median_similarity:.3f}")
        logger.info(f"Overall score: {overall_score:.4f}")

        return metrics

    def cross_validate(
        self,
        model_class,
        n_folds: int = 5
    ) -> List[Dict]:
        """
        Perform time-series cross-validation.

        Uses expanding window approach (not k-fold, since time series).

        Args:
            model_class: Model class to evaluate
            n_folds: Number of folds

        Returns:
            List of metrics for each fold
        """
        logger.info(f"Performing {n_folds}-fold cross-validation...")

        session = get_session()
        try:
            posts = session.query(Post).order_by(Post.created_at.asc()).all()

            if len(posts) < n_folds * 10:  # Need enough data
                logger.error("Insufficient data for cross-validation")
                return []

            data = [{
                'post_id': p.post_id,
                'created_at': p.created_at,
                'content': p.content,
                'replies': p.replies_count,
                'reblogs': p.reblogs_count,
                'favourites': p.favourites_count
            } for p in posts]

            df = pd.DataFrame(data)

            fold_results = []
            fold_size = len(df) // (n_folds + 1)

            for fold in range(n_folds):
                train_end = fold_size * (fold + 1)
                test_start = train_end
                test_end = train_end + fold_size

                train_df = df.iloc[:train_end].copy()
                test_df = df.iloc[test_start:test_end].copy()

                if len(test_df) == 0:
                    continue

                logger.info(f"Fold {fold + 1}/{n_folds}: Train={len(train_df)}, Test={len(test_df)}")

                # Train model
                model = model_class()
                model.train(train_df)

                # Evaluate
                metrics = self.evaluate_timing_model(model, test_df, max_predictions=10)

                if metrics:
                    fold_results.append({
                        'fold': fold + 1,
                        'train_size': len(train_df),
                        'test_size': len(test_df),
                        **metrics
                    })

            # Aggregate results
            if fold_results:
                logger.info("="*60)
                logger.info("Cross-validation summary:")
                avg_mae = np.mean([f['mae_hours'] for f in fold_results])
                avg_score = np.mean([f['overall_score'] for f in fold_results])
                logger.info(f"Average MAE: {avg_mae:.2f}h")
                logger.info(f"Average score: {avg_score:.4f}")
                logger.info("="*60)

            return fold_results

        finally:
            session.close()


def main():
    """Test the evaluator."""
    logger.info("Testing Model Evaluator...")

    evaluator = ModelEvaluator()

    # Get evaluation data
    train_df, test_df = evaluator.get_evaluation_data(train_split=0.8)

    if train_df is not None and test_df is not None:
        logger.info(f"Train set: {len(train_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")


if __name__ == "__main__":
    main()
