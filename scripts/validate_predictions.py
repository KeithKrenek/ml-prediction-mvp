#!/usr/bin/env python3
"""
Prediction Validation Script

Validates predictions against actual posts by:
1. Finding unvalidated predictions
2. Matching them to actual posts within time window
3. Calculating timing error and content similarity
4. Updating prediction records with actual outcomes
5. Generating validation report

Can be run manually or as part of automated workflow.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import yaml
from loguru import logger

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.validation.validator import PredictionValidator
from src.data.database import init_db


def load_config() -> dict:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def main():
    """Main validation entry point"""
    start_time = datetime.now()

    logger.info("="*80)
    logger.info("PREDICTION VALIDATION")
    logger.info(f"Started at: {start_time}")
    logger.info("="*80)

    # Load configuration
    config = load_config()
    validation_config = config.get('validation', {})
    similarity_config = validation_config.get('similarity_metrics', {})

    # Initialize database
    logger.info("Initializing database...")
    init_db()

    # Create validator with enhanced similarity metrics
    logger.info("Initializing validator with enhanced similarity metrics...")
    validator = PredictionValidator(
        matching_window_hours=validation_config.get('matching_window_hours', 24),
        similarity_weights=similarity_config.get('weights'),
        use_bertscore=similarity_config.get('use_bertscore', True),
        use_sentence_embeddings=similarity_config.get('use_sentence_embeddings', True),
        use_entity_matching=similarity_config.get('use_entity_matching', True)
    )

    # Get current stats before validation
    logger.info("Current validation statistics:")
    current_stats = validator.get_validation_stats()
    if current_stats.get('total_validated', 0) > 0:
        logger.info(f"  Total validated: {current_stats['total_validated']}")
        logger.info(f"  Overall accuracy: {current_stats['overall_accuracy']:.1%}")
        logger.info(f"  Timing MAE: {current_stats['timing_mae_hours']:.2f}h")
        logger.info(f"  Within 6h: {current_stats['within_6h_accuracy']:.1%}")
    else:
        logger.info("  No validated predictions yet")

    logger.info("="*80)

    # Validate all unvalidated predictions
    logger.info("Starting validation of unvalidated predictions...")
    summary = validator.validate_all_unvalidated()

    # Calculate duration
    duration = (datetime.now() - start_time).total_seconds()

    # Display results
    logger.info("="*80)
    logger.success("VALIDATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info("")
    logger.info("Results:")
    logger.info(f"  Total predictions checked: {summary['total_predictions']}")
    logger.info(f"  Successfully validated: {summary['validated']}")
    logger.info(f"  Matched to posts: {summary['matched']}")
    logger.info(f"  Unmatched: {summary['unmatched']}")
    logger.info(f"  Correct predictions: {summary['correct']}")
    if summary['matched'] > 0:
        logger.info(f"  Accuracy: {summary['accuracy']:.1%}")
    logger.info("")

    # Show sample results
    if summary['results']:
        logger.info("Sample results:")
        for result in summary['results'][:5]:  # Show first 5
            if result.get('matched'):
                logger.info(f"  ✓ {result['prediction_id'][:8]}... - "
                           f"Error: {result['timing_error_hours']:.2f}h, "
                           f"Similarity: {result['similarity_metrics']['composite_similarity']:.3f}")
            else:
                logger.info(f"  ✗ {result['prediction_id'][:8]}... - {result.get('reason', 'Unknown')}")

    logger.info("="*80)

    # Get updated stats
    logger.info("")
    logger.info("Updated statistics:")
    updated_stats = validator.get_validation_stats()
    if updated_stats.get('total_validated', 0) > 0:
        logger.info(f"  Total validated: {updated_stats['total_validated']}")
        logger.info(f"  Overall accuracy: {updated_stats['overall_accuracy']:.1%}")
        logger.info(f"  Timing MAE: {updated_stats['timing_mae_hours']:.2f}h")
        logger.info(f"  Timing median: {updated_stats['timing_median_hours']:.2f}h")
        logger.info(f"  Within 6h: {updated_stats['within_6h_accuracy']:.1%}")
        logger.info(f"  Within 12h: {updated_stats['within_12h_accuracy']:.1%}")
        logger.info(f"  Within 24h: {updated_stats['within_24h_accuracy']:.1%}")
        logger.info(f"  Avg content similarity: {updated_stats['avg_content_similarity']:.3f}")

    logger.info("="*80)

    # Exit with success - the script completed its job regardless of whether
    # any predictions were validated. Having no predictions to validate is
    # a normal state (e.g., first deployment, all caught up, recent predictions).
    sys.exit(0)


if __name__ == "__main__":
    main()
