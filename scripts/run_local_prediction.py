#!/usr/bin/env python3
"""Quick local test for training and prediction."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from dotenv import load_dotenv
load_dotenv()

from src.predictor import TrumpPostPredictor

def main():
    print("=" * 60)
    print("LOCAL PREDICTION TEST")
    print("=" * 60)
    
    # Initialize predictor
    print("\n[1/3] Initializing predictor...")
    predictor = TrumpPostPredictor()
    
    # Train models
    print("\n[2/3] Training models...")
    success = predictor.train_models()
    
    if not success:
        print("ERROR: Training failed!")
        return
    
    predictor.save_models()
    print("Models trained and saved!")
    
    # Make prediction
    print("\n[3/3] Making prediction...")
    result = predictor.predict(save_to_db=True)
    
    if result:
        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(f"Prediction ID: {result['prediction_id']}")
        print(f"Predicted Time: {result['predicted_time']}")
        print(f"Timing Confidence: {result['timing_confidence']:.1%}")
        print(f"\nPredicted Content:")
        print(f"  {result['predicted_content']}")
        print(f"Content Confidence: {result['content_confidence']:.1%}")
        print(f"\nOverall Confidence: {result['overall_confidence']:.1%}")
        print("=" * 60)
    else:
        print("ERROR: Prediction failed!")

if __name__ == "__main__":
    main()

