#!/usr/bin/env python3
"""
Cron job script for making predictions.
Runs once, makes a prediction, and exits.

Designed for Render.com Cron Jobs (free tier).
Schedule: Configurable via render.yaml (default: every 6 hours)

This script:
1. Initializes the predictor
2. Ensures models are trained
3. Makes one prediction
4. Saves to database
5. Logs results
6. Exits cleanly
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Configure logger for cron (writes to stdout for Render logs)
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)

from src.predictor import TrumpPostPredictor
from src.utils.run_log import CronRunLogger
from src.utils.alerts import get_alert_manager


def main():
    """Main cron job entry point"""
    start_time = datetime.now()

    logger.info("="*60)
    logger.info("Prediction Cron Job Starting")
    logger.info(f"Execution time: {start_time}")
    logger.info("="*60)

    exit_code = 0
    try:
        with CronRunLogger("predict") as run_log:
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            if not anthropic_key or anthropic_key == 'your_anthropic_api_key_here':
                raise RuntimeError("ANTHROPIC_API_KEY not configured - cannot generate content predictions")

            logger.info("Initializing predictor...")
            predictor = TrumpPostPredictor()

            model_path = Path(__file__).parent.parent / "models" / "timing_model.pkl"

            if not model_path.exists():
                logger.info("No existing models found - training new models...")
                success = predictor.train_models()

                if not success:
                    raise RuntimeError("Model training failed - cannot make prediction")

                predictor.save_models()
                logger.success("Models trained and saved successfully")
            else:
                logger.info("Using existing trained models")

            logger.info("Making prediction...")
            prediction = predictor.predict(save_to_db=True)

            if not prediction:
                raise RuntimeError("Prediction failed")

            duration = (datetime.now() - start_time).total_seconds()

            logger.info("="*60)
            logger.success("Prediction completed successfully!")
            logger.info(f"Prediction ID: {prediction['prediction_id']}")
            logger.info(f"Predicted time: {prediction['predicted_time']}")
            logger.info(f"Predicted content: {prediction['predicted_content'][:100]}...")
            logger.info(f"Timing confidence: {prediction['timing_confidence']:.1%}")
            logger.info(f"Content confidence: {prediction['content_confidence']:.1%}")
            logger.info(f"Overall confidence: {prediction['overall_confidence']:.1%}")
            logger.info(f"Duration: {duration:.2f} seconds")
            logger.info("="*60)

            run_log.update(
                records_processed=1,
                api_calls=1,
                extra_metadata={
                    'duration_seconds': duration,
                    'prediction_id': prediction['prediction_id'],
                    'overall_confidence': prediction['overall_confidence'],
                    'timing_confidence': prediction['timing_confidence'],
                    'content_confidence': prediction['content_confidence']
                }
            )
    except Exception as e:
        exit_code = 1
        logger.error(f"Cron job failed: {e}")
        logger.exception("Full traceback:")
        get_alert_manager().check_cron_failures("predict")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
