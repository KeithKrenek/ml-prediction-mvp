#!/usr/bin/env python3
"""
Automated Model Retraining Script

This script orchestrates the complete retraining workflow:
1. Load latest data from database
2. Split into train/test sets
3. Train new model version
4. Evaluate performance
5. Compare with current production model
6. Auto-promote if better (based on thresholds)
7. Archive old versions
8. Log all metrics and decisions

Designed to run as a cron job (weekly/monthly)
"""

import os
import sys
from pathlib import Path
import time
from datetime import datetime, timezone
import yaml
from loguru import logger

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.models.timing_model import TimingPredictor
from src.models.content_model import ContentGenerator
from src.models.model_registry import ModelRegistry
from src.models.evaluator import ModelEvaluator
from src.data.database import init_db, get_session, Post


def load_retraining_config():
    """Load retraining configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('retraining', {})
    except Exception as e:
        logger.warning(f"Could not load retraining config: {e}. Using defaults.")
        return {
            'enabled': True,
            'auto_promote': True,
            'min_improvement_threshold': 2.0,
            'min_training_samples': 50,
            'test_split': 0.2,
            'evaluation_max_predictions': 20,
            'keep_versions': 10
        }


def retrain_timing_model(registry: ModelRegistry, evaluator: ModelEvaluator, config: dict):
    """
    Retrain timing model and evaluate.

    Returns:
        (version_id, metrics, promoted) tuple
    """
    logger.info("="*80)
    logger.info("TIMING MODEL RETRAINING")
    logger.info("="*80)

    start_time = time.time()

    try:
        # Get train/test split
        logger.info("Loading and splitting data...")
        train_df, test_df = evaluator.get_evaluation_data(
            train_split=1 - config.get('test_split', 0.2),
            min_samples=config.get('min_training_samples', 50)
        )

        if train_df is None or test_df is None:
            logger.error("Insufficient data for training")
            return None, None, False

        if len(train_df) < config.get('min_training_samples', 50):
            logger.error(f"Not enough training samples: {len(train_df)}")
            return None, None, False

        # Train new model
        logger.info(f"Training new timing model on {len(train_df)} samples...")
        model = TimingPredictor()
        train_success = model.train(train_df)

        if not train_success:
            logger.error("Model training failed")
            return None, None, False

        training_duration = time.time() - start_time

        # Save model with version
        version_id = registry.generate_version_id('timing', 'prophet')
        model_dir = registry.timing_models_dir
        model_path = model_dir / f"{version_id}.pkl"

        model.save(str(model_path))
        logger.success(f"Trained model saved: {model_path}")

        # Register model version
        logger.info("Registering model version...")
        model_version = registry.register_model(
            model_type='timing',
            algorithm='prophet',
            model_file_path=str(model_path),
            training_data_start=train_df['created_at'].min(),
            training_data_end=train_df['created_at'].max(),
            num_training_samples=len(train_df),
            training_duration=training_duration,
            hyperparameters=model.config,
            notes=f"Automated retraining - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            created_by='cron'
        )

        logger.success(f"Model version registered: {version_id}")

        # Create training run record
        training_run = registry.create_training_run(
            model_version_id=version_id,
            config_snapshot={
                'algorithm': 'prophet',
                'config': model.config,
                'train_samples': len(train_df),
                'test_samples': len(test_df)
            }
        )

        # Evaluate model
        logger.info("Evaluating model on test set...")
        eval_metrics = evaluator.evaluate_timing_model(
            model,
            test_df,
            max_predictions=config.get('evaluation_max_predictions', 20)
        )

        if not eval_metrics:
            logger.error("Model evaluation failed")
            registry.update_training_run(
                training_run.run_id,
                status='failed',
                error_message='Evaluation failed'
            )
            return version_id, None, False

        # Update training run with metrics
        registry.update_training_run(
            training_run.run_id,
            status='completed',
            metrics={
                'num_training_samples': len(train_df),
                'num_validation_samples': len(test_df),
                'training_data_start': train_df['created_at'].min(),
                'training_data_end': train_df['created_at'].max(),
                'test_mae_hours': eval_metrics['mae_hours'],
                'test_within_6h_accuracy': eval_metrics['within_6h_accuracy'],
                'test_within_24h_accuracy': eval_metrics['within_24h_accuracy']
            }
        )

        # Store evaluation in database
        logger.info("Storing evaluation results...")
        evaluation = registry.evaluate_model(
            version_id=version_id,
            predictions=eval_metrics['predictions'],
            actuals=eval_metrics['actuals'],
            eval_dataset_start=test_df['created_at'].min(),
            eval_dataset_end=test_df['created_at'].max()
        )

        # Auto-promote if configured
        promoted = False
        if config.get('auto_promote', True):
            logger.info("Checking if new model should be promoted...")
            min_improvement = config.get('min_improvement_threshold', 2.0)

            promoted, reason = registry.auto_promote_if_better(
                new_version_id=version_id,
                min_improvement_threshold=min_improvement
            )

            if promoted:
                logger.success(f"✓ Model promoted to production! Reason: {reason}")
                registry.update_training_run(
                    training_run.run_id,
                    metrics={
                        'promoted_to_production': True,
                        'promotion_reason': reason
                    }
                )
            else:
                logger.info(f"✗ Model not promoted. Reason: {reason}")
        else:
            logger.info("Auto-promotion disabled - manual review required")

        logger.info("="*80)
        logger.success("TIMING MODEL RETRAINING COMPLETE")
        logger.info(f"Version: {version_id}")
        logger.info(f"MAE: {eval_metrics['mae_hours']:.2f}h")
        logger.info(f"Within 6h accuracy: {eval_metrics['within_6h_accuracy']:.1%}")
        logger.info(f"Overall score: {eval_metrics['overall_score']:.4f}")
        logger.info(f"Promoted: {promoted}")
        logger.info("="*80)

        return version_id, eval_metrics, promoted

    except Exception as e:
        logger.error(f"Error during timing model retraining: {e}")
        logger.exception("Full traceback:")
        return None, None, False


def retrain_content_model(registry: ModelRegistry, evaluator: ModelEvaluator, config: dict):
    """
    Retrain content model and evaluate.

    Note: For Claude API-based model, we mainly update the example set.
    For fine-tuned models, this would do actual retraining.

    Returns:
        (version_id, metrics, promoted) tuple
    """
    logger.info("="*80)
    logger.info("CONTENT MODEL RETRAINING")
    logger.info("="*80)

    start_time = time.time()

    try:
        # Get train/test split
        logger.info("Loading and splitting data...")
        train_df, test_df = evaluator.get_evaluation_data(
            train_split=1 - config.get('test_split', 0.2),
            min_samples=config.get('min_training_samples', 50)
        )

        if train_df is None or test_df is None:
            logger.error("Insufficient data for training")
            return None, None, False

        # "Train" content model (load new examples)
        logger.info(f"Updating content model with {len(train_df)} training samples...")
        generator = ContentGenerator()
        generator.load_example_posts(num_examples=10)  # Refresh examples

        training_duration = time.time() - start_time

        # Save model state
        version_id = registry.generate_version_id('content', 'claude_api')
        model_dir = registry.content_models_dir
        model_path = model_dir / f"{version_id}.pkl"

        # For Claude API, we don't have much to save, but store metadata
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model_type': 'claude_api',
                'example_posts': generator.example_posts[:5],  # Sample
                'config': generator.config,
                'version_id': version_id
            }, f)

        # Register model version
        logger.info("Registering model version...")
        model_version = registry.register_model(
            model_type='content',
            algorithm='claude_api',
            model_file_path=str(model_path),
            training_data_start=train_df['created_at'].min(),
            training_data_end=train_df['created_at'].max(),
            num_training_samples=len(train_df),
            training_duration=training_duration,
            hyperparameters=generator.config,
            notes=f"Automated example set update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            created_by='cron'
        )

        logger.success(f"Model version registered: {version_id}")

        # Evaluate model
        logger.info("Evaluating content model on test set...")
        eval_metrics = evaluator.evaluate_content_model(
            generator,
            test_df,
            max_predictions=min(10, len(test_df))  # Content eval is expensive
        )

        if not eval_metrics:
            logger.warning("Content model evaluation failed - skipping promotion")
            return version_id, None, False

        # Store evaluation
        evaluation = registry.evaluate_model(
            version_id=version_id,
            predictions=eval_metrics['predictions'],
            actuals=eval_metrics['actuals'],
            eval_dataset_start=test_df['created_at'].min(),
            eval_dataset_end=test_df['created_at'].max()
        )

        # Auto-promote if configured
        promoted = False
        if config.get('auto_promote', True):
            logger.info("Checking if new content model should be promoted...")
            promoted, reason = registry.auto_promote_if_better(
                new_version_id=version_id,
                min_improvement_threshold=config.get('min_improvement_threshold', 2.0)
            )

            if promoted:
                logger.success(f"✓ Content model promoted! Reason: {reason}")
            else:
                logger.info(f"✗ Content model not promoted. Reason: {reason}")
        else:
            logger.info("Auto-promotion disabled")

        logger.info("="*80)
        logger.success("CONTENT MODEL RETRAINING COMPLETE")
        logger.info(f"Version: {version_id}")
        logger.info(f"Average similarity: {eval_metrics['avg_similarity']:.3f}")
        logger.info(f"Overall score: {eval_metrics['overall_score']:.4f}")
        logger.info(f"Promoted: {promoted}")
        logger.info("="*80)

        return version_id, eval_metrics, promoted

    except Exception as e:
        logger.error(f"Error during content model retraining: {e}")
        logger.exception("Full traceback:")
        return None, None, False


def main():
    """Main retraining entry point"""
    start_time = datetime.now()

    logger.info("="*80)
    logger.info("AUTOMATED MODEL RETRAINING")
    logger.info(f"Started at: {start_time}")
    logger.info("="*80)

    # Load configuration
    config = load_retraining_config()

    if not config.get('enabled', True):
        logger.warning("Retraining is disabled in configuration")
        logger.info("Set retraining.enabled: true in config.yaml to enable")
        sys.exit(0)

    # Initialize components
    logger.info("Initializing database...")
    init_db()

    logger.info("Initializing registry and evaluator...")
    registry = ModelRegistry()
    evaluator = ModelEvaluator()

    # Check data availability
    session = get_session()
    num_posts = session.query(Post).count()
    session.close()

    logger.info(f"Total posts in database: {num_posts}")

    if num_posts < config.get('min_training_samples', 50):
        logger.error(f"Insufficient data for retraining: {num_posts} posts")
        logger.info("Collect more data before retraining")
        sys.exit(1)

    # Retrain timing model
    timing_version, timing_metrics, timing_promoted = retrain_timing_model(
        registry, evaluator, config
    )

    # Retrain content model
    content_version, content_metrics, content_promoted = retrain_content_model(
        registry, evaluator, config
    )

    # Archive old versions if configured
    if config.get('archive_old_versions', True):
        logger.info("Archiving old model versions...")
        keep_versions = config.get('keep_versions', 10)
        registry.archive_old_versions(keep_recent=keep_versions)

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("="*80)
    logger.success("RETRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info("")
    logger.info("Timing Model:")
    logger.info(f"  Version: {timing_version or 'FAILED'}")
    logger.info(f"  Promoted: {timing_promoted}")
    if timing_metrics:
        logger.info(f"  MAE: {timing_metrics['mae_hours']:.2f}h")
        logger.info(f"  Within 6h: {timing_metrics['within_6h_accuracy']:.1%}")
    logger.info("")
    logger.info("Content Model:")
    logger.info(f"  Version: {content_version or 'FAILED'}")
    logger.info(f"  Promoted: {content_promoted}")
    if content_metrics:
        logger.info(f"  Avg Similarity: {content_metrics['avg_similarity']:.3f}")
    logger.info("="*80)

    # Exit code
    if timing_version or content_version:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    main()
