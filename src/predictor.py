"""
Main Prediction Pipeline - Combines timing and content models.
"""

import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

from loguru import logger
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unified_timing_model import UnifiedTimingPredictor
from src.models.content_model import ContentGenerator, ContextGatherer
from src.data.database import get_session, Prediction, ContextSnapshot
from src.models.model_registry import ModelRegistry
from src.validation.calibration import ConfidenceCalibrator


class TrumpPostPredictor:
    """
    Main prediction system combining timing and content models.
    """
    
    def __init__(self, use_calibration: bool = True):
        self.config = self._load_global_config()
        timing_type = self.config.get('timing_model', {}).get('type', 'prophet')
        self.timing_model = UnifiedTimingPredictor(model_type=timing_type, config_path="config/config.yaml")
        self.content_model = ContentGenerator()
        self.context_gatherer = ContextGatherer()
        self.model_registry = ModelRegistry()
        self.latest_context = None
        
        # Initialize calibrator for confidence calibration
        self.use_calibration = use_calibration
        self.calibrator = None
        if use_calibration:
            self._init_calibrator()
        
        logger.info("Trump Post Predictor initialized (calibration=%s)", use_calibration)
    
    def _init_calibrator(self):
        """Initialize and load confidence calibrator."""
        try:
            self.calibrator = ConfidenceCalibrator()
            loaded = self.calibrator.load()
            if loaded:
                logger.info("Loaded existing calibration model")
            else:
                logger.info("No existing calibration model found - will use raw confidences")
        except Exception as e:
            logger.warning(f"Failed to initialize calibrator: {e}")
            self.calibrator = None

    def _load_global_config(self):
        """Load global configuration for downstream components."""
        config_path = Path("config/config.yaml")
        if not config_path.exists():
            logger.warning("Config file not found at %s", config_path)
            return {}
        try:
            with config_path.open('r') as cfg:
                return yaml.safe_load(cfg) or {}
        except Exception as exc:
            logger.warning(f"Failed to load config: {exc}")
            return {}
    
    def train_models(self):
        """Train both timing and content models"""
        logger.info("Training models...")
        
        logger.info("Gathering latest context before training...")
        context_snapshot = self.context_gatherer.get_full_context(save_to_db=True)
        self.latest_context = context_snapshot

        training_df = self.timing_model.load_data_from_db()
        training_started_at = datetime.now()

        logger.info("Training timing model (%s)...", getattr(self.timing_model, 'active_model_type', 'unknown').upper())
        timing_success = self.timing_model.train(df=training_df, context=context_snapshot)
        
        logger.info("Loading curated example posts for content model...")
        self.content_model.load_example_posts(context=context_snapshot)
        
        if timing_success and training_df is not None:
            try:
                self.timing_model.evaluate(df=training_df)
            except Exception as exc:
                logger.warning(f"Timing evaluation failed: {exc}")
            self._register_timing_model(training_df, training_started_at)
        
        if timing_success:
            logger.success("Models trained successfully!")
            return True
        else:
            logger.error("Model training failed!")
            return False

    def _register_timing_model(self, training_df, training_started_at):
        """Persist the freshly trained timing model via the registry."""
        if self.model_registry is None or training_df is None or training_df.empty:
            return

        algorithm = getattr(self.timing_model, 'active_model_type', self.config.get('timing_model', {}).get('type', 'prophet'))
        version_id = self.model_registry.generate_version_id('timing', algorithm)
        model_dir = Path("models") / "timing"
        model_dir.mkdir(parents=True, exist_ok=True)
        extension = ".pkl" if algorithm == 'prophet' else ".pt"
        model_path = model_dir / f"{version_id}{extension}"

        try:
            self.timing_model.save(str(model_path))
            hyperparameters = self.config.get('timing_model', {}).get('prophet' if algorithm == 'prophet' else 'neural_tpp', {})
            training_duration = (datetime.now() - training_started_at).total_seconds()
            model_version = self.model_registry.register_model(
                model_type='timing',
                algorithm=algorithm,
                model_file_path=str(model_path),
                training_data_start=training_df['created_at'].min(),
                training_data_end=training_df['created_at'].max(),
                num_training_samples=len(training_df),
                training_duration=training_duration,
                hyperparameters=hyperparameters,
                notes=f"Unified timing model ({algorithm})",
                created_by='system'
            )

            if model_version:
                self.model_registry.promote_to_production(
                    model_version.version_id,
                    reason=f"Latest {algorithm} training run"
                )
        except Exception as exc:
            logger.warning(f"Failed to register timing model: {exc}")
    
    def predict(self, save_to_db=True, apply_calibration=True):
        """
        Make a complete prediction: timing + content.
        
        Args:
            save_to_db: Whether to save prediction to database
            apply_calibration: Whether to apply confidence calibration
            
        Returns:
            dict with complete prediction
        """
        logger.info("Making prediction...")

        context = self.context_gatherer.get_full_context()
        if context:
            self.latest_context = context
        context_payload = context or self.latest_context

        # Get timing prediction
        timing_pred = self.timing_model.predict_next(context=context_payload)
        
        if not timing_pred:
            logger.error("Timing prediction failed!")
            return None
        
        # Generate content
        content_pred = self.content_model.generate(
            context=context_payload,
            predicted_time=timing_pred['predicted_time']
        )
        
        if not content_pred:
            logger.error("Content generation failed!")
            return None

        similarity_metrics = content_pred.get('similarity_metrics')
        if context_payload is None:
            context_payload = {}
        elif not isinstance(context_payload, dict):
            context_payload = dict(context_payload)
        if similarity_metrics:
            context_payload['content_similarity_metrics'] = similarity_metrics
        
        # Get raw confidence values
        timing_confidence_raw = timing_pred['confidence']
        content_confidence_raw = content_pred['confidence']
        
        # Apply calibration if available
        timing_confidence = timing_confidence_raw
        content_confidence = content_confidence_raw
        calibration_applied = False
        
        if apply_calibration and self.use_calibration and self.calibrator:
            try:
                timing_confidence = self.calibrator.calibrate_timing(timing_confidence_raw)
                content_confidence = self.calibrator.calibrate_content(content_confidence_raw)
                calibration_applied = True
                logger.info(f"Calibration applied: timing {timing_confidence_raw:.2f}->{timing_confidence:.2f}, "
                           f"content {content_confidence_raw:.2f}->{content_confidence:.2f}")
            except Exception as e:
                logger.warning(f"Calibration failed: {e}. Using raw confidences.")
        
        # Combine predictions
        prediction = {
            'prediction_id': str(uuid.uuid4()),
            'predicted_at': datetime.now(),
            
            # Timing
            'predicted_time': timing_pred['predicted_time'],
            'timing_confidence': timing_confidence,
            'timing_confidence_raw': timing_confidence_raw,
            'timing_model_version': timing_pred['model_version'],
            
            # Content
            'predicted_content': content_pred['content'],
            'content_confidence': content_confidence,
            'content_confidence_raw': content_confidence_raw,
            'content_similarity_metrics': similarity_metrics,
            'content_model_version': content_pred['model_version'],
            
            # Combined confidence
            'overall_confidence': (timing_confidence + content_confidence) / 2,
            
            # Calibration metadata
            'calibration_applied': calibration_applied,
            
            # Additional timing stats if available (from NTPP)
            'timing_stats': {
                k: timing_pred.get(k) for k in 
                ['mean_hours', 'median_hours', 'std_hours', 'ci_lower_hours', 'ci_upper_hours',
                 'prob_within_1h', 'prob_within_3h', 'prob_within_6h']
                if k in timing_pred
            },
            
            # Context
            'context': context_payload
        }
        
        # Save to database if requested
        if save_to_db:
            self._save_prediction(prediction)
        
        logger.success("Prediction complete!")
        return prediction
    
    def calibrate_models(self, min_samples: int = 20) -> bool:
        """
        Run auto-calibration on historical data.
        
        Should be called periodically (e.g., weekly) to keep
        calibration up to date.
        
        Args:
            min_samples: Minimum validated predictions required
            
        Returns:
            True if calibration successful
        """
        if self.calibrator is None:
            self.calibrator = ConfidenceCalibrator()
        
        success = self.calibrator.auto_calibrate(min_samples=min_samples)
        
        if success:
            logger.success("Model calibration complete")
        else:
            logger.warning("Calibration not updated (insufficient data or error)")
        
        return success
    
    def _save_prediction(self, prediction):
        """Save prediction to database"""
        import pandas as pd
        
        # Convert numpy/pandas types to Python native types for PostgreSQL compatibility
        def to_native(val):
            if val is None:
                return None
            if hasattr(val, 'item'):  # numpy types
                return val.item()
            if isinstance(val, pd.Timestamp):
                return val.to_pydatetime()
            return val
        
        session = get_session()
        
        db_prediction = Prediction(
            prediction_id=prediction['prediction_id'],
            predicted_at=to_native(prediction['predicted_at']),
            predicted_time=to_native(prediction['predicted_time']),
            predicted_time_confidence=to_native(prediction['timing_confidence']),
            predicted_content=prediction['predicted_content'],
            predicted_content_confidence=to_native(prediction['content_confidence']),
            timing_model_version=prediction['timing_model_version'],
            content_model_version=prediction['content_model_version'],
            context_data=prediction['context']
        )
        
        session.add(db_prediction)
        session.commit()
        
        # Update context snapshot usage tracking
        if prediction.get('context') and prediction['context'].get('snapshot_id'):
            snapshot_id = prediction['context']['snapshot_id']
            snapshot = session.query(ContextSnapshot).filter_by(snapshot_id=snapshot_id).first()
            if snapshot:
                snapshot.used_in_predictions += 1
                if not snapshot.prediction_ids:
                    snapshot.prediction_ids = []
                snapshot.prediction_ids.append(prediction['prediction_id'])
                session.commit()
                logger.debug(f"Updated context snapshot {snapshot_id} usage count")
        
        session.close()
        
        logger.info(f"Prediction saved to database: {prediction['prediction_id']}")
    
    def display_prediction(self, prediction):
        """Display prediction in a nice format"""
        if not prediction:
            print("No prediction available")
            return
        
        print("\n" + "="*60)
        print(" "*20 + "PREDICTION RESULTS")
        print("="*60)
        print()
        print(f"🕐 PREDICTED TIME:")
        print(f"   {prediction['predicted_time'].strftime('%A, %B %d, %Y at %I:%M %p')}")
        print(f"   Confidence: {prediction['timing_confidence']:.1%}")
        print()
        print(f"📝 PREDICTED CONTENT:")
        print(f"   {prediction['predicted_content']}")
        print(f"   Confidence: {prediction['content_confidence']:.1%}")
        print()
        print(f"📊 OVERALL CONFIDENCE: {prediction['overall_confidence']:.1%}")
        print()
        print(f"ℹ️  METADATA:")
        print(f"   Prediction ID: {prediction['prediction_id']}")
        print(f"   Timing Model: {prediction['timing_model_version']}")
        print(f"   Content Model: {prediction['content_model_version']}")
        print(f"   Predicted at: {prediction['predicted_at'].strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        print()
    
    def save_models(self):
        """Save trained models"""
        logger.info("Saving models...")
        default_path = Path("models") / "timing_model.pkl"
        self.timing_model.save(str(default_path))
        logger.success("Models saved!")


def main():
    """Main entry point for prediction pipeline"""
    logger.info("="*60)
    logger.info(" "*15 + "TRUMP POST PREDICTOR")
    logger.info("="*60)
    
    # Initialize predictor
    predictor = TrumpPostPredictor()
    
    # Train models
    logger.info("\n[1/3] Training models...")
    success = predictor.train_models()
    
    if not success:
        logger.error("Training failed. Cannot make predictions.")
        return
    
    # Save models
    logger.info("\n[2/3] Saving models...")
    predictor.save_models()
    
    # Make prediction
    logger.info("\n[3/3] Making prediction...")
    prediction = predictor.predict(save_to_db=True)
    
    # Display results
    if prediction:
        predictor.display_prediction(prediction)
    
    logger.info("="*60)
    logger.info("Done! Check the database for saved prediction.")
    logger.info("="*60)


if __name__ == "__main__":
    main()
