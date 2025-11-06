"""
Prediction Validator - Matches predictions to actual posts and calculates accuracy.

This module handles:
- Time-based matching of predictions to actual posts
- Calculating timing errors (in hours)
- Computing content similarity (word overlap, length, BERTScore)
- Updating prediction records with actual outcomes
- Generating validation reports
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from loguru import logger

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.database import get_session, Prediction, Post


class PredictionValidator:
    """
    Validates predictions against actual posts and calculates accuracy metrics.
    """

    def __init__(self, matching_window_hours: float = 24):
        """
        Initialize validator.

        Args:
            matching_window_hours: Time window for matching predictions to posts (default: 24 hours)
        """
        self.matching_window_hours = matching_window_hours

    def find_unvalidated_predictions(self) -> List[Prediction]:
        """
        Find all predictions that haven't been validated yet.

        Returns:
            List of Prediction objects without actual outcomes
        """
        session = get_session()
        try:
            predictions = (
                session.query(Prediction)
                .filter(Prediction.actual_post_id == None)  # Not yet validated
                .filter(Prediction.predicted_time <= datetime.now(timezone.utc))  # Prediction time has passed
                .all()
            )

            logger.info(f"Found {len(predictions)} unvalidated predictions")
            return predictions

        finally:
            session.close()

    def find_matching_post(
        self,
        predicted_time: datetime,
        window_hours: float = None
    ) -> Optional[Post]:
        """
        Find the closest actual post to a predicted time.

        Args:
            predicted_time: When the post was predicted to occur
            window_hours: Search window in hours (default: use class window)

        Returns:
            Matching Post object or None
        """
        if window_hours is None:
            window_hours = self.matching_window_hours

        window = timedelta(hours=window_hours)
        start_time = predicted_time - window
        end_time = predicted_time + window

        session = get_session()
        try:
            # Find posts within the time window
            posts = (
                session.query(Post)
                .filter(Post.created_at >= start_time)
                .filter(Post.created_at <= end_time)
                .all()
            )

            if not posts:
                return None

            # Find the closest post by time
            closest_post = min(
                posts,
                key=lambda p: abs((p.created_at - predicted_time).total_seconds())
            )

            return closest_post

        finally:
            session.close()

    def calculate_timing_error(
        self,
        predicted_time: datetime,
        actual_time: datetime
    ) -> float:
        """
        Calculate timing error in hours.

        Args:
            predicted_time: Predicted post time
            actual_time: Actual post time

        Returns:
            Error in hours (absolute value)
        """
        error_seconds = abs((actual_time - predicted_time).total_seconds())
        error_hours = error_seconds / 3600
        return error_hours

    def calculate_content_similarity(
        self,
        predicted_content: str,
        actual_content: str
    ) -> Dict[str, float]:
        """
        Calculate content similarity metrics.

        Args:
            predicted_content: Predicted post text
            actual_content: Actual post text

        Returns:
            Dict with similarity metrics
        """
        # Length similarity
        pred_len = len(predicted_content)
        actual_len = len(actual_content)
        length_similarity = 1 - abs(pred_len - actual_len) / max(pred_len, actual_len)

        # Word overlap (Jaccard similarity)
        pred_words = set(predicted_content.lower().split())
        actual_words = set(actual_content.lower().split())

        if pred_words or actual_words:
            intersection = len(pred_words & actual_words)
            union = len(pred_words | actual_words)
            word_overlap = intersection / union if union > 0 else 0
        else:
            word_overlap = 0

        # Character overlap
        pred_chars = set(predicted_content.lower())
        actual_chars = set(actual_content.lower())
        char_overlap = len(pred_chars & actual_chars) / len(pred_chars | actual_chars) if (pred_chars | actual_chars) else 0

        # Composite similarity (average of metrics)
        composite_similarity = (length_similarity + word_overlap + char_overlap) / 3

        return {
            'length_similarity': length_similarity,
            'word_overlap': word_overlap,
            'character_overlap': char_overlap,
            'composite_similarity': composite_similarity,
            # Placeholder for BERTScore (can be added later)
            'bertscore_f1': composite_similarity  # Use composite as proxy for now
        }

    def is_prediction_correct(
        self,
        timing_error_hours: float,
        content_similarity: float,
        timing_threshold: float = 6.0,
        content_threshold: float = 0.3
    ) -> bool:
        """
        Determine if a prediction is considered "correct".

        Args:
            timing_error_hours: Timing error in hours
            content_similarity: Content similarity score (0-1)
            timing_threshold: Maximum acceptable timing error (default: 6 hours)
            content_threshold: Minimum acceptable similarity (default: 0.3)

        Returns:
            True if prediction is correct
        """
        timing_correct = timing_error_hours <= timing_threshold
        content_correct = content_similarity >= content_threshold

        return timing_correct and content_correct

    def validate_prediction(
        self,
        prediction: Prediction,
        save_to_db: bool = True
    ) -> Dict:
        """
        Validate a single prediction against actual posts.

        Args:
            prediction: Prediction object to validate
            save_to_db: Whether to update database

        Returns:
            Dict with validation results
        """
        logger.info(f"Validating prediction {prediction.prediction_id}...")

        # Find matching post
        matching_post = self.find_matching_post(prediction.predicted_time)

        if not matching_post:
            logger.warning(f"No matching post found for prediction {prediction.prediction_id} "
                          f"(predicted time: {prediction.predicted_time})")
            return {
                'prediction_id': prediction.prediction_id,
                'matched': False,
                'reason': 'No post found within matching window'
            }

        # Calculate timing error
        timing_error = self.calculate_timing_error(
            prediction.predicted_time,
            matching_post.created_at
        )

        # Calculate content similarity
        similarity_metrics = self.calculate_content_similarity(
            prediction.predicted_content,
            matching_post.content
        )

        # Determine if correct
        is_correct = self.is_prediction_correct(
            timing_error,
            similarity_metrics['composite_similarity']
        )

        # Update prediction record
        if save_to_db:
            session = get_session()
            try:
                pred = session.query(Prediction).filter_by(
                    prediction_id=prediction.prediction_id
                ).first()

                if pred:
                    pred.actual_post_id = matching_post.post_id
                    pred.actual_time = matching_post.created_at
                    pred.actual_content = matching_post.content
                    pred.timing_error_hours = timing_error
                    pred.bertscore_f1 = similarity_metrics['bertscore_f1']
                    pred.was_correct = is_correct

                    session.commit()
                    logger.success(f"Updated prediction {prediction.prediction_id} with validation results")

            finally:
                session.close()

        result = {
            'prediction_id': prediction.prediction_id,
            'matched': True,
            'actual_post_id': matching_post.post_id,
            'predicted_time': prediction.predicted_time,
            'actual_time': matching_post.created_at,
            'timing_error_hours': timing_error,
            'similarity_metrics': similarity_metrics,
            'is_correct': is_correct,
            'predicted_content': prediction.predicted_content[:100] + '...',
            'actual_content': matching_post.content[:100] + '...'
        }

        logger.info(f"Validation result: Error={timing_error:.2f}h, "
                   f"Similarity={similarity_metrics['composite_similarity']:.3f}, "
                   f"Correct={is_correct}")

        return result

    def validate_all_unvalidated(self) -> Dict:
        """
        Validate all unvalidated predictions.

        Returns:
            Dict with validation summary
        """
        logger.info("Starting validation of all unvalidated predictions...")

        unvalidated = self.find_unvalidated_predictions()

        if not unvalidated:
            logger.info("No unvalidated predictions found")
            return {
                'total_predictions': 0,
                'validated': 0,
                'matched': 0,
                'unmatched': 0,
                'correct': 0,
                'results': []
            }

        results = []
        matched_count = 0
        correct_count = 0

        for prediction in unvalidated:
            try:
                result = self.validate_prediction(prediction, save_to_db=True)
                results.append(result)

                if result['matched']:
                    matched_count += 1
                    if result['is_correct']:
                        correct_count += 1

            except Exception as e:
                logger.error(f"Error validating prediction {prediction.prediction_id}: {e}")
                results.append({
                    'prediction_id': prediction.prediction_id,
                    'matched': False,
                    'error': str(e)
                })

        summary = {
            'total_predictions': len(unvalidated),
            'validated': len(results),
            'matched': matched_count,
            'unmatched': len(results) - matched_count,
            'correct': correct_count,
            'accuracy': correct_count / matched_count if matched_count > 0 else 0,
            'results': results
        }

        logger.success(f"Validation complete: {matched_count}/{len(unvalidated)} matched, "
                      f"{correct_count}/{matched_count} correct "
                      f"(accuracy: {summary['accuracy']:.1%})")

        return summary

    def get_validation_stats(self) -> Dict:
        """
        Get overall validation statistics.

        Returns:
            Dict with aggregate stats
        """
        session = get_session()
        try:
            # Get all validated predictions
            validated_preds = (
                session.query(Prediction)
                .filter(Prediction.actual_post_id != None)
                .all()
            )

            if not validated_preds:
                return {
                    'total_validated': 0,
                    'message': 'No validated predictions yet'
                }

            # Calculate stats
            timing_errors = [p.timing_error_hours for p in validated_preds if p.timing_error_hours is not None]
            similarities = [p.bertscore_f1 for p in validated_preds if p.bertscore_f1 is not None]
            correct_count = sum(1 for p in validated_preds if p.was_correct)

            # Within-window accuracies
            within_6h = sum(1 for e in timing_errors if e <= 6) / len(timing_errors) if timing_errors else 0
            within_12h = sum(1 for e in timing_errors if e <= 12) / len(timing_errors) if timing_errors else 0
            within_24h = sum(1 for e in timing_errors if e <= 24) / len(timing_errors) if timing_errors else 0

            stats = {
                'total_validated': len(validated_preds),
                'total_correct': correct_count,
                'overall_accuracy': correct_count / len(validated_preds),
                'timing_mae_hours': np.mean(timing_errors) if timing_errors else None,
                'timing_median_hours': np.median(timing_errors) if timing_errors else None,
                'timing_std_hours': np.std(timing_errors) if timing_errors else None,
                'within_6h_accuracy': within_6h,
                'within_12h_accuracy': within_12h,
                'within_24h_accuracy': within_24h,
                'avg_content_similarity': np.mean(similarities) if similarities else None,
                'median_content_similarity': np.median(similarities) if similarities else None
            }

            return stats

        finally:
            session.close()

    def get_timeline_data(self, days_back: int = 30) -> Dict:
        """
        Get timeline data for visualization.

        Args:
            days_back: Number of days to look back

        Returns:
            Dict with actual and predicted posts for timeline
        """
        session = get_session()
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)

            # Get actual posts
            actual_posts = (
                session.query(Post)
                .filter(Post.created_at >= cutoff_date)
                .order_by(Post.created_at.asc())
                .all()
            )

            # Get predictions
            predictions = (
                session.query(Prediction)
                .filter(Prediction.predicted_time >= cutoff_date)
                .order_by(Prediction.predicted_time.asc())
                .all()
            )

            actual_data = []
            for post in actual_posts:
                actual_data.append({
                    'id': post.post_id,
                    'time': post.created_at,
                    'content': post.content,
                    'type': 'actual',
                    'matched_prediction_id': None  # Will be filled below
                })

            predicted_data = []
            for pred in predictions:
                predicted_data.append({
                    'id': pred.prediction_id,
                    'time': pred.predicted_time,
                    'content': pred.predicted_content,
                    'type': 'predicted',
                    'actual_time': pred.actual_time,
                    'actual_post_id': pred.actual_post_id,
                    'timing_error_hours': pred.timing_error_hours,
                    'was_correct': pred.was_correct,
                    'similarity': pred.bertscore_f1,
                    'timing_confidence': pred.predicted_time_confidence,
                    'content_confidence': pred.predicted_content_confidence
                })

                # Link predictions to actual posts
                if pred.actual_post_id:
                    for actual in actual_data:
                        if actual['id'] == pred.actual_post_id:
                            actual['matched_prediction_id'] = pred.prediction_id
                            break

            return {
                'actual_posts': actual_data,
                'predictions': predicted_data,
                'date_range': {
                    'start': cutoff_date,
                    'end': datetime.now(timezone.utc)
                }
            }

        finally:
            session.close()


def main():
    """Test the validator."""
    logger.info("Testing Prediction Validator...")

    validator = PredictionValidator(matching_window_hours=24)

    # Get validation stats
    stats = validator.get_validation_stats()
    logger.info(f"Current stats: {stats}")

    # Validate all unvalidated
    summary = validator.validate_all_unvalidated()
    logger.info(f"Validation summary: {summary}")


if __name__ == "__main__":
    main()
