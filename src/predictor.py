"""
Main Prediction Pipeline - Combines timing and content models.
"""

import os
import sys
from datetime import datetime
from loguru import logger
import uuid

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.timing_model import TimingPredictor
from src.models.content_model import ContentGenerator, ContextGatherer
from src.data.database import get_session, Prediction


class TrumpPostPredictor:
    """
    Main prediction system combining timing and content models.
    """
    
    def __init__(self):
        self.timing_model = TimingPredictor()
        self.content_model = ContentGenerator()
        self.context_gatherer = ContextGatherer()
        
        logger.info("Trump Post Predictor initialized")
    
    def train_models(self):
        """Train both timing and content models"""
        logger.info("Training models...")
        
        # Train timing model
        logger.info("Training timing model...")
        timing_success = self.timing_model.train()
        
        # Load examples for content model
        logger.info("Loading examples for content model...")
        self.content_model.load_example_posts()
        
        if timing_success:
            logger.success("Models trained successfully!")
            return True
        else:
            logger.error("Model training failed!")
            return False
    
    def predict(self, save_to_db=True):
        """
        Make a complete prediction: timing + content.
        
        Args:
            save_to_db: Whether to save prediction to database
            
        Returns:
            dict with complete prediction
        """
        logger.info("Making prediction...")
        
        # Get timing prediction
        timing_pred = self.timing_model.get_next_post_time()
        
        if not timing_pred:
            logger.error("Timing prediction failed!")
            return None
        
        # Gather context
        context = self.context_gatherer.get_full_context()
        
        # Generate content
        content_pred = self.content_model.generate(
            context=context,
            predicted_time=timing_pred['predicted_time']
        )
        
        if not content_pred:
            logger.error("Content generation failed!")
            return None
        
        # Combine predictions
        prediction = {
            'prediction_id': str(uuid.uuid4()),
            'predicted_at': datetime.now(),
            
            # Timing
            'predicted_time': timing_pred['predicted_time'],
            'timing_confidence': timing_pred['confidence'],
            'timing_model_version': timing_pred['model_version'],
            
            # Content
            'predicted_content': content_pred['content'],
            'content_confidence': content_pred['confidence'],
            'content_model_version': content_pred['model_version'],
            
            # Combined confidence
            'overall_confidence': (timing_pred['confidence'] + content_pred['confidence']) / 2,
            
            # Context
            'context': context
        }
        
        # Save to database if requested
        if save_to_db:
            self._save_prediction(prediction)
        
        logger.success("Prediction complete!")
        return prediction
    
    def _save_prediction(self, prediction):
        """Save prediction to database"""
        session = get_session()
        
        db_prediction = Prediction(
            prediction_id=prediction['prediction_id'],
            predicted_at=prediction['predicted_at'],
            predicted_time=prediction['predicted_time'],
            predicted_time_confidence=prediction['timing_confidence'],
            predicted_content=prediction['predicted_content'],
            predicted_content_confidence=prediction['content_confidence'],
            timing_model_version=prediction['timing_model_version'],
            content_model_version=prediction['content_model_version'],
            context_data=prediction['context']
        )
        
        session.add(db_prediction)
        session.commit()
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
        self.timing_model.save()
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
