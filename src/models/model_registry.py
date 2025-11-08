"""
Model Registry - Manages model versions, evaluations, and automatic promotion.

This module handles:
- Model versioning and metadata tracking
- Model evaluation and performance comparison
- Automatic model promotion based on performance
- Model rollback capabilities
- Training run tracking
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
import uuid
import shutil
import pickle
import numpy as np
from loguru import logger

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import (
    get_session,
    ModelVersion,
    TrainingRun,
    ModelEvaluation,
    Post,
    Prediction
)


class ModelRegistry:
    """
    Central registry for managing model lifecycle, versions, and promotions.
    """

    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Create versioned subdirectories
        self.timing_models_dir = self.models_dir / "timing"
        self.content_models_dir = self.models_dir / "content"
        self.timing_models_dir.mkdir(parents=True, exist_ok=True)
        self.content_models_dir.mkdir(parents=True, exist_ok=True)

    def generate_version_id(self, model_type: str, algorithm: str) -> str:
        """
        Generate a unique version ID for a model.

        Format: {algorithm}_{model_type}_{timestamp}_{short_uuid}
        Example: prophet_timing_20251106_120000_a3f2
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"{algorithm}_{model_type}_{timestamp}_{short_uuid}"

    def register_model(
        self,
        model_type: str,
        algorithm: str,
        model_file_path: str,
        training_data_start: datetime,
        training_data_end: datetime,
        num_training_samples: int,
        training_duration: float,
        hyperparameters: Dict,
        notes: str = None,
        created_by: str = "system",
        version_id: str = None
    ) -> ModelVersion:
        """
        Register a new trained model in the registry.

        Args:
            model_type: 'timing' or 'content'
            algorithm: 'prophet', 'neural_tpp', 'claude_api', etc.
            model_file_path: Path to saved model file
            training_data_start: Start date of training data
            training_data_end: End date of training data
            num_training_samples: Number of training samples
            training_duration: Training time in seconds
            hyperparameters: Model configuration dict
            notes: Optional notes about this version
            created_by: Who created this ('system', 'manual', 'cron')
            version_id: Optional version ID (if None, one will be generated)

        Returns:
            ModelVersion object
        """
        if version_id is None:
            version_id = self.generate_version_id(model_type, algorithm)

        # Get file size
        file_size = os.path.getsize(model_file_path) if os.path.exists(model_file_path) else 0

        # Create model version record
        model_version = ModelVersion(
            version_id=version_id,
            model_type=model_type,
            algorithm=algorithm,
            trained_at=datetime.now(timezone.utc),
            training_duration_seconds=training_duration,
            training_data_start=training_data_start,
            training_data_end=training_data_end,
            num_training_samples=num_training_samples,
            file_path=model_file_path,
            file_size_bytes=file_size,
            hyperparameters=hyperparameters,
            status='trained',
            is_production=False,
            notes=notes,
            created_by=created_by
        )

        session = get_session()
        try:
            session.add(model_version)
            session.commit()
            session.refresh(model_version)

            logger.success(f"Registered model version: {version_id}")
            return model_version

        finally:
            session.close()

    def get_production_model(self, model_type: str) -> Optional[ModelVersion]:
        """
        Get the current production model for a given type.

        Args:
            model_type: 'timing' or 'content'

        Returns:
            ModelVersion object or None
        """
        session = get_session()
        try:
            model = (
                session.query(ModelVersion)
                .filter_by(model_type=model_type, is_production=True)
                .order_by(ModelVersion.promoted_at.desc())
                .first()
            )
            return model
        finally:
            session.close()

    def get_all_versions(self, model_type: str = None, limit: int = 50) -> List[ModelVersion]:
        """Get all model versions, optionally filtered by type."""
        session = get_session()
        try:
            query = session.query(ModelVersion).order_by(ModelVersion.trained_at.desc())

            if model_type:
                query = query.filter_by(model_type=model_type)

            return query.limit(limit).all()
        finally:
            session.close()

    def promote_to_production(
        self,
        version_id: str,
        reason: str = None
    ) -> bool:
        """
        Promote a model version to production.

        This will:
        1. Demote current production model
        2. Set new model as production
        3. Copy model file to standard location

        Args:
            version_id: Version ID to promote
            reason: Reason for promotion

        Returns:
            True if successful
        """
        session = get_session()
        try:
            # Get the model to promote
            new_model = session.query(ModelVersion).filter_by(version_id=version_id).first()

            if not new_model:
                logger.error(f"Model version {version_id} not found")
                return False

            # Demote current production model (query in SAME session to avoid race condition)
            current_prod = (
                session.query(ModelVersion)
                .filter_by(model_type=new_model.model_type, is_production=True)
                .order_by(ModelVersion.promoted_at.desc())
                .first()
            )
            if current_prod:
                logger.info(f"Demoting current production model: {current_prod.version_id}")
                current_prod.is_production = False
                session.add(current_prod)

            # Promote new model
            new_model.is_production = True
            new_model.promoted_at = datetime.now(timezone.utc)
            new_model.status = 'active'
            if reason:
                new_model.notes = (new_model.notes or "") + f"\n[Promotion] {reason}"

            session.add(new_model)
            session.commit()

            # Copy model file to standard location for easy loading
            if os.path.exists(new_model.file_path):
                standard_path = self.models_dir / f"{new_model.model_type}_model.pkl"
                shutil.copy2(new_model.file_path, standard_path)
                logger.info(f"Copied model to {standard_path}")

            logger.success(f"Promoted {version_id} to production!")
            return True

        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            session.rollback()
            return False

        finally:
            session.close()

    def evaluate_model(
        self,
        version_id: str,
        predictions: List[Dict],
        actuals: List[Dict],
        eval_dataset_start: datetime,
        eval_dataset_end: datetime
    ) -> ModelEvaluation:
        """
        Evaluate a model and store results.

        Args:
            version_id: Model version to evaluate
            predictions: List of prediction dicts
            actuals: List of actual outcome dicts
            eval_dataset_start: Start of evaluation period
            eval_dataset_end: End of evaluation period

        Returns:
            ModelEvaluation object
        """
        session = get_session()
        try:
            model = session.query(ModelVersion).filter_by(version_id=version_id).first()

            if not model:
                logger.error(f"Model {version_id} not found")
                return None

            # Calculate metrics based on model type
            if model.model_type == 'timing':
                metrics = self._calculate_timing_metrics(predictions, actuals)
            else:  # content
                metrics = self._calculate_content_metrics(predictions, actuals)

            # Create evaluation record
            evaluation = ModelEvaluation(
                model_version_id=version_id,
                evaluated_at=datetime.now(timezone.utc),
                eval_dataset_start=eval_dataset_start,
                eval_dataset_end=eval_dataset_end,
                num_samples=len(predictions),
                **metrics,
                predictions_json={
                    'predictions': predictions[:100],  # Store sample for analysis
                    'actuals': actuals[:100]
                }
            )

            session.add(evaluation)
            session.commit()
            session.refresh(evaluation)

            logger.success(f"Evaluated {version_id}: overall_score={evaluation.overall_score:.4f}")
            return evaluation

        finally:
            session.close()

    def _calculate_timing_metrics(self, predictions: List[Dict], actuals: List[Dict]) -> Dict:
        """Calculate timing model evaluation metrics."""
        errors = []

        for pred, actual in zip(predictions, actuals):
            pred_time = pred.get('predicted_time')
            actual_time = actual.get('actual_time')

            if pred_time and actual_time:
                error_hours = abs((pred_time - actual_time).total_seconds() / 3600)
                errors.append(error_hours)

        if not errors:
            return {
                'mae_hours': None,
                'rmse_hours': None,
                'median_error_hours': None,
                'within_6h_accuracy': None,
                'within_12h_accuracy': None,
                'within_24h_accuracy': None,
                'overall_score': 0.0
            }

        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        median_error = np.median(errors)

        within_6h = sum(1 for e in errors if e <= 6) / len(errors)
        within_12h = sum(1 for e in errors if e <= 12) / len(errors)
        within_24h = sum(1 for e in errors if e <= 24) / len(errors)

        # Overall score: weighted combination (lower is better for MAE, higher for accuracy)
        # Normalize and combine: 70% within_6h + 30% (1 - normalized_mae)
        normalized_mae = max(0, 1 - (mae / 24))  # 24h = 0 score, 0h = 1 score
        overall_score = 0.7 * within_6h + 0.3 * normalized_mae

        return {
            'mae_hours': mae,
            'rmse_hours': rmse,
            'median_error_hours': median_error,
            'within_6h_accuracy': within_6h,
            'within_12h_accuracy': within_12h,
            'within_24h_accuracy': within_24h,
            'overall_score': overall_score
        }

    def _calculate_content_metrics(self, predictions: List[Dict], actuals: List[Dict]) -> Dict:
        """Calculate content model evaluation metrics."""
        # Placeholder for content metrics
        # In practice, you'd calculate BERTScore, BLEU, etc.

        length_similarities = []

        for pred, actual in zip(predictions, actuals):
            pred_content = pred.get('predicted_content', '')
            actual_content = actual.get('actual_content', '')

            if pred_content and actual_content:
                pred_len = len(pred_content)
                actual_len = len(actual_content)
                similarity = 1 - abs(pred_len - actual_len) / max(pred_len, actual_len)
                length_similarities.append(similarity)

        avg_length_sim = np.mean(length_similarities) if length_similarities else 0.5

        return {
            'bertscore_precision': None,
            'bertscore_recall': None,
            'bertscore_f1': None,
            'bleu_score': None,
            'avg_length_similarity': avg_length_sim,
            'overall_score': avg_length_sim
        }

    def compare_models(
        self,
        model_a_version: str,
        model_b_version: str
    ) -> Dict:
        """
        Compare two models based on their latest evaluations.

        Returns:
            Dict with comparison results and recommendation
        """
        session = get_session()
        try:
            eval_a = (
                session.query(ModelEvaluation)
                .filter_by(model_version_id=model_a_version)
                .order_by(ModelEvaluation.evaluated_at.desc())
                .first()
            )

            eval_b = (
                session.query(ModelEvaluation)
                .filter_by(model_version_id=model_b_version)
                .order_by(ModelEvaluation.evaluated_at.desc())
                .first()
            )

            if not eval_a or not eval_b:
                return {
                    'comparison': 'incomplete',
                    'reason': 'Missing evaluation data'
                }

            # Compare overall scores
            score_a = eval_a.overall_score or 0
            score_b = eval_b.overall_score or 0

            improvement = ((score_b - score_a) / score_a * 100) if score_a > 0 else 0

            # Determine winner
            if score_b > score_a * 1.02:  # At least 2% better
                winner = model_b_version
                recommendation = 'promote'
                reason = f"New model is {improvement:.1f}% better"
            elif score_b < score_a * 0.98:  # More than 2% worse
                winner = model_a_version
                recommendation = 'reject'
                reason = f"New model is {abs(improvement):.1f}% worse"
            else:
                winner = 'tie'
                recommendation = 'manual_review'
                reason = f"Models are similar ({improvement:.1f}% difference)"

            return {
                'model_a': {
                    'version': model_a_version,
                    'score': score_a,
                    'evaluation': eval_a
                },
                'model_b': {
                    'version': model_b_version,
                    'score': score_b,
                    'evaluation': eval_b
                },
                'winner': winner,
                'improvement_percentage': improvement,
                'recommendation': recommendation,
                'reason': reason
            }

        finally:
            session.close()

    def auto_promote_if_better(
        self,
        new_version_id: str,
        min_improvement_threshold: float = 2.0
    ) -> Tuple[bool, str]:
        """
        Automatically promote a new model if it's better than production.

        Args:
            new_version_id: New model version to evaluate
            min_improvement_threshold: Minimum % improvement required (default: 2%)

        Returns:
            (promoted: bool, reason: str)
        """
        session = get_session()
        try:
            # Get new model
            new_model = session.query(ModelVersion).filter_by(version_id=new_version_id).first()
            if not new_model:
                return False, "Model not found"

            # Get current production model
            prod_model = self.get_production_model(new_model.model_type)

            if not prod_model:
                # No production model exists, promote by default
                logger.info("No production model exists, promoting new model by default")
                self.promote_to_production(new_version_id, "First production model")
                return True, "Promoted as first production model"

            # Compare models
            comparison = self.compare_models(prod_model.version_id, new_version_id)

            if comparison['recommendation'] == 'promote':
                logger.info(f"Auto-promoting {new_version_id}: {comparison['reason']}")
                self.promote_to_production(new_version_id, comparison['reason'])
                return True, comparison['reason']

            elif comparison['recommendation'] == 'reject':
                logger.warning(f"Not promoting {new_version_id}: {comparison['reason']}")
                return False, comparison['reason']

            else:  # manual_review
                logger.info(f"Manual review required for {new_version_id}: {comparison['reason']}")
                return False, comparison['reason']

        finally:
            session.close()

    def create_training_run(
        self,
        model_version_id: str,
        config_snapshot: Dict
    ) -> TrainingRun:
        """Create a new training run record."""
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        training_run = TrainingRun(
            run_id=run_id,
            model_version_id=model_version_id,
            started_at=datetime.now(timezone.utc),
            status='running',
            config_snapshot=config_snapshot
        )

        session = get_session()
        try:
            session.add(training_run)
            session.commit()
            session.refresh(training_run)
            return training_run
        finally:
            session.close()

    def update_training_run(
        self,
        run_id: str,
        status: str = None,
        metrics: Dict = None,
        error_message: str = None
    ):
        """Update a training run with results."""
        session = get_session()
        try:
            run = session.query(TrainingRun).filter_by(run_id=run_id).first()

            if not run:
                logger.error(f"Training run {run_id} not found")
                return

            if status:
                run.status = status

            if status in ['completed', 'failed']:
                run.completed_at = datetime.now(timezone.utc)
                # Ensure both datetimes are timezone-aware for subtraction
                started = run.started_at
                if started.tzinfo is None:
                    started = started.replace(tzinfo=timezone.utc)
                run.duration_seconds = (run.completed_at - started).total_seconds()

            if metrics:
                for key, value in metrics.items():
                    if hasattr(run, key):
                        setattr(run, key, value)

            if error_message:
                run.error_message = error_message

            session.add(run)
            session.commit()

        finally:
            session.close()

    def get_training_history(self, limit: int = 10) -> List[TrainingRun]:
        """Get recent training runs."""
        session = get_session()
        try:
            return (
                session.query(TrainingRun)
                .order_by(TrainingRun.started_at.desc())
                .limit(limit)
                .all()
            )
        finally:
            session.close()

    def rollback_to_version(self, version_id: str) -> bool:
        """
        Rollback to a previous model version.

        This promotes an archived model back to production.
        """
        return self.promote_to_production(
            version_id,
            reason=f"Manual rollback to {version_id}"
        )

    def archive_old_versions(self, keep_recent: int = 10):
        """Archive old model versions to save space."""
        session = get_session()
        try:
            for model_type in ['timing', 'content']:
                # Get all non-production models
                old_models = (
                    session.query(ModelVersion)
                    .filter_by(model_type=model_type, is_production=False)
                    .filter(ModelVersion.status != 'archived')
                    .order_by(ModelVersion.trained_at.desc())
                    .offset(keep_recent)
                    .all()
                )

                for model in old_models:
                    model.status = 'archived'
                    session.add(model)
                    logger.info(f"Archived old model: {model.version_id}")

            session.commit()

        finally:
            session.close()


def main():
    """Test the model registry."""
    logger.info("Testing Model Registry...")

    registry = ModelRegistry()

    # Get production model
    prod_model = registry.get_production_model('timing')
    if prod_model:
        logger.info(f"Current production timing model: {prod_model.version_id}")
    else:
        logger.info("No production timing model found")

    # List all versions
    versions = registry.get_all_versions('timing', limit=5)
    logger.info(f"Found {len(versions)} timing model versions:")
    for v in versions:
        logger.info(f"  - {v.version_id} ({v.status}, production={v.is_production})")


if __name__ == "__main__":
    main()
