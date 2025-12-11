"""
Confidence Calibration Module

Implements calibrated confidence scores using isotonic regression.
Maps raw model confidence to calibrated probabilities that match
actual prediction accuracy.

Key features:
- Isotonic regression for monotonic calibration
- Separate calibration for timing and content predictions
- Calibration drift detection
- Reliability diagrams for visualization
"""

import os
import sys
import pickle
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

# Try to import sklearn for isotonic regression
try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import calibration_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available. Using simple calibration fallback.")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.database import get_session, Prediction


class ConfidenceCalibrator:
    """
    Calibrates model confidence scores using isotonic regression.
    
    Isotonic regression ensures monotonic mapping: higher raw confidence
    should always map to higher calibrated confidence.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize calibrator.
        
        Args:
            model_path: Path to load/save calibration model
        """
        self.timing_calibrator = None
        self.content_calibrator = None
        self.model_path = model_path or "models/calibration"
        
        # Calibration statistics
        self.timing_stats = {
            'n_samples': 0,
            'last_calibrated': None,
            'calibration_error': None
        }
        self.content_stats = {
            'n_samples': 0,
            'last_calibrated': None,
            'calibration_error': None
        }
        
        logger.info(f"ConfidenceCalibrator initialized (sklearn={SKLEARN_AVAILABLE})")
    
    def collect_calibration_data(
        self,
        min_samples: int = 20,
        days_back: int = 90
    ) -> Tuple[Dict, Dict]:
        """
        Collect historical prediction data for calibration.
        
        Args:
            min_samples: Minimum samples required for calibration
            days_back: Number of days to look back
            
        Returns:
            Tuple of (timing_data, content_data) dicts
        """
        session = get_session()
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
            
            # Get validated predictions
            predictions = (
                session.query(Prediction)
                .filter(Prediction.actual_post_id != None)
                .filter(Prediction.predicted_at >= cutoff)
                .all()
            )
            
            if len(predictions) < min_samples:
                logger.warning(f"Insufficient calibration data: {len(predictions)} < {min_samples}")
                return {}, {}
            
            # Collect timing data
            timing_confidences = []
            timing_outcomes = []  # 1 if correct (within threshold), 0 otherwise
            
            # Collect content data
            content_confidences = []
            content_outcomes = []  # Actual similarity scores
            
            for pred in predictions:
                # Timing: "correct" if within 6 hours
                if pred.predicted_time_confidence is not None and pred.timing_error_hours is not None:
                    timing_confidences.append(float(pred.predicted_time_confidence))
                    # Binary outcome: 1 if within 6 hours
                    timing_outcomes.append(1.0 if pred.timing_error_hours <= 6 else 0.0)
                
                # Content: Use BERTScore as outcome
                if pred.predicted_content_confidence is not None and pred.bertscore_f1 is not None:
                    content_confidences.append(float(pred.predicted_content_confidence))
                    content_outcomes.append(float(pred.bertscore_f1))
            
            timing_data = {
                'confidences': np.array(timing_confidences),
                'outcomes': np.array(timing_outcomes)
            } if timing_confidences else {}
            
            content_data = {
                'confidences': np.array(content_confidences),
                'outcomes': np.array(content_outcomes)
            } if content_confidences else {}
            
            logger.info(f"Collected {len(timing_confidences)} timing samples, {len(content_confidences)} content samples")
            
            return timing_data, content_data
            
        finally:
            session.close()
    
    def fit(
        self,
        timing_data: Optional[Dict] = None,
        content_data: Optional[Dict] = None,
        min_samples: int = 10
    ) -> bool:
        """
        Fit calibration models on historical data.
        
        Args:
            timing_data: Dict with 'confidences' and 'outcomes' arrays
            content_data: Dict with 'confidences' and 'outcomes' arrays
            min_samples: Minimum samples required per model
            
        Returns:
            True if calibration successful
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available. Skipping calibration.")
            return False
        
        success = False
        
        # Fit timing calibrator
        if timing_data and len(timing_data.get('confidences', [])) >= min_samples:
            try:
                self.timing_calibrator = IsotonicRegression(
                    y_min=0.0, y_max=1.0, out_of_bounds='clip'
                )
                self.timing_calibrator.fit(
                    timing_data['confidences'],
                    timing_data['outcomes']
                )
                
                # Calculate calibration error (ECE - Expected Calibration Error)
                self.timing_stats['calibration_error'] = self._calculate_ece(
                    timing_data['confidences'],
                    timing_data['outcomes']
                )
                self.timing_stats['n_samples'] = len(timing_data['confidences'])
                self.timing_stats['last_calibrated'] = datetime.now()
                
                logger.success(f"Timing calibrator fitted on {len(timing_data['confidences'])} samples")
                success = True
            except Exception as e:
                logger.error(f"Failed to fit timing calibrator: {e}")
        
        # Fit content calibrator
        if content_data and len(content_data.get('confidences', [])) >= min_samples:
            try:
                self.content_calibrator = IsotonicRegression(
                    y_min=0.0, y_max=1.0, out_of_bounds='clip'
                )
                self.content_calibrator.fit(
                    content_data['confidences'],
                    content_data['outcomes']
                )
                
                self.content_stats['calibration_error'] = self._calculate_ece(
                    content_data['confidences'],
                    content_data['outcomes']
                )
                self.content_stats['n_samples'] = len(content_data['confidences'])
                self.content_stats['last_calibrated'] = datetime.now()
                
                logger.success(f"Content calibrator fitted on {len(content_data['confidences'])} samples")
                success = True
            except Exception as e:
                logger.error(f"Failed to fit content calibrator: {e}")
        
        return success
    
    def _calculate_ece(
        self,
        confidences: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error.
        
        ECE measures how well calibrated the confidence scores are.
        Lower is better.
        
        Args:
            confidences: Predicted confidence scores
            outcomes: Actual outcomes (0/1 or continuous)
            n_bins: Number of bins for calibration
            
        Returns:
            ECE value (0-1)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find predictions in this bin
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
            if np.sum(in_bin) > 0:
                bin_confidence = np.mean(confidences[in_bin])
                bin_accuracy = np.mean(outcomes[in_bin])
                bin_weight = np.sum(in_bin) / len(confidences)
                
                ece += bin_weight * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def calibrate_timing(self, raw_confidence: float) -> float:
        """
        Calibrate timing prediction confidence.
        
        Args:
            raw_confidence: Raw model confidence (0-1)
            
        Returns:
            Calibrated confidence (0-1)
        """
        if self.timing_calibrator is None:
            return raw_confidence
        
        try:
            calibrated = self.timing_calibrator.predict([raw_confidence])[0]
            return float(np.clip(calibrated, 0.05, 0.99))
        except Exception as e:
            logger.warning(f"Timing calibration failed: {e}")
            return raw_confidence
    
    def calibrate_content(self, raw_confidence: float) -> float:
        """
        Calibrate content prediction confidence.
        
        Args:
            raw_confidence: Raw model confidence (0-1)
            
        Returns:
            Calibrated confidence (0-1)
        """
        if self.content_calibrator is None:
            return raw_confidence
        
        try:
            calibrated = self.content_calibrator.predict([raw_confidence])[0]
            return float(np.clip(calibrated, 0.05, 0.99))
        except Exception as e:
            logger.warning(f"Content calibration failed: {e}")
            return raw_confidence
    
    def calibrate_prediction(self, prediction: Dict) -> Dict:
        """
        Calibrate all confidence scores in a prediction.
        
        Args:
            prediction: Prediction dict with confidence scores
            
        Returns:
            Prediction with calibrated confidences
        """
        calibrated = prediction.copy()
        
        # Calibrate timing confidence
        if 'timing_confidence' in prediction:
            raw = prediction['timing_confidence']
            calibrated['timing_confidence_raw'] = raw
            calibrated['timing_confidence'] = self.calibrate_timing(raw)
        
        # Calibrate content confidence
        if 'content_confidence' in prediction:
            raw = prediction['content_confidence']
            calibrated['content_confidence_raw'] = raw
            calibrated['content_confidence'] = self.calibrate_content(raw)
        
        # Recalculate overall confidence
        if 'timing_confidence' in calibrated and 'content_confidence' in calibrated:
            calibrated['overall_confidence'] = (
                calibrated['timing_confidence'] + calibrated['content_confidence']
            ) / 2
        
        calibrated['calibration_applied'] = True
        
        return calibrated
    
    def get_reliability_data(
        self,
        prediction_type: str = 'timing',
        n_bins: int = 10
    ) -> Dict:
        """
        Get reliability diagram data for visualization.
        
        A well-calibrated model should have points close to the diagonal.
        
        Args:
            prediction_type: 'timing' or 'content'
            n_bins: Number of bins
            
        Returns:
            Dict with bin data for plotting
        """
        timing_data, content_data = self.collect_calibration_data()
        
        if prediction_type == 'timing':
            data = timing_data
        else:
            data = content_data
        
        if not data:
            return {'bins': [], 'calibration_error': None}
        
        confidences = data['confidences']
        outcomes = data['outcomes']
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bins_data = []
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            bin_center = (bin_lower + bin_upper) / 2
            
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
            if np.sum(in_bin) > 0:
                mean_confidence = float(np.mean(confidences[in_bin]))
                mean_accuracy = float(np.mean(outcomes[in_bin]))
                count = int(np.sum(in_bin))
                
                bins_data.append({
                    'bin_center': bin_center,
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'mean_confidence': mean_confidence,
                    'mean_accuracy': mean_accuracy,
                    'count': count,
                    'gap': abs(mean_accuracy - mean_confidence)
                })
        
        ece = self._calculate_ece(confidences, outcomes, n_bins)
        
        return {
            'bins': bins_data,
            'calibration_error': ece,
            'n_samples': len(confidences),
            'prediction_type': prediction_type
        }
    
    def check_calibration_drift(
        self,
        recent_days: int = 7,
        threshold: float = 0.1
    ) -> Dict:
        """
        Check if calibration has drifted significantly.
        
        Compares recent ECE to historical ECE.
        
        Args:
            recent_days: Number of recent days to check
            threshold: ECE increase threshold to flag drift
            
        Returns:
            Dict with drift analysis
        """
        # Get recent data
        session = get_session()
        try:
            recent_cutoff = datetime.now(timezone.utc) - timedelta(days=recent_days)
            
            recent_preds = (
                session.query(Prediction)
                .filter(Prediction.actual_post_id != None)
                .filter(Prediction.predicted_at >= recent_cutoff)
                .all()
            )
            
            if len(recent_preds) < 5:
                return {
                    'drift_detected': False,
                    'reason': 'Insufficient recent data',
                    'n_recent': len(recent_preds)
                }
            
            # Calculate recent timing ECE
            timing_conf = [p.predicted_time_confidence for p in recent_preds 
                         if p.predicted_time_confidence and p.timing_error_hours is not None]
            timing_outcomes = [1.0 if p.timing_error_hours <= 6 else 0.0 for p in recent_preds
                             if p.predicted_time_confidence and p.timing_error_hours is not None]
            
            if timing_conf:
                recent_timing_ece = self._calculate_ece(
                    np.array(timing_conf),
                    np.array(timing_outcomes)
                )
                
                historical_ece = self.timing_stats.get('calibration_error', 0)
                
                timing_drift = recent_timing_ece - historical_ece if historical_ece else 0
                timing_drifted = timing_drift > threshold
            else:
                recent_timing_ece = None
                timing_drift = 0
                timing_drifted = False
            
            return {
                'drift_detected': timing_drifted,
                'timing_drift': timing_drift,
                'recent_timing_ece': recent_timing_ece,
                'historical_timing_ece': self.timing_stats.get('calibration_error'),
                'n_recent': len(recent_preds),
                'recommendation': 'Recalibrate models' if timing_drifted else 'Calibration OK'
            }
            
        finally:
            session.close()
    
    def save(self, path: Optional[str] = None):
        """Save calibration models to disk."""
        path = path or self.model_path
        Path(path).mkdir(parents=True, exist_ok=True)
        
        data = {
            'timing_calibrator': self.timing_calibrator,
            'content_calibrator': self.content_calibrator,
            'timing_stats': self.timing_stats,
            'content_stats': self.content_stats
        }
        
        filepath = Path(path) / 'calibration_models.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.success(f"Calibration models saved to {filepath}")
    
    def load(self, path: Optional[str] = None) -> bool:
        """Load calibration models from disk."""
        path = path or self.model_path
        filepath = Path(path) / 'calibration_models.pkl'
        
        if not filepath.exists():
            logger.warning(f"No calibration models found at {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.timing_calibrator = data.get('timing_calibrator')
            self.content_calibrator = data.get('content_calibrator')
            self.timing_stats = data.get('timing_stats', self.timing_stats)
            self.content_stats = data.get('content_stats', self.content_stats)
            
            logger.success(f"Calibration models loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load calibration models: {e}")
            return False
    
    def auto_calibrate(self, min_samples: int = 20) -> bool:
        """
        Automatically collect data and calibrate.
        
        Args:
            min_samples: Minimum samples required
            
        Returns:
            True if calibration successful
        """
        logger.info("Running auto-calibration...")
        
        timing_data, content_data = self.collect_calibration_data(min_samples=min_samples)
        
        if not timing_data and not content_data:
            logger.warning("No calibration data available")
            return False
        
        success = self.fit(timing_data, content_data, min_samples=min_samples)
        
        if success:
            self.save()
        
        return success


def test_calibration():
    """Test the calibration module."""
    print("\n" + "="*80)
    print("CONFIDENCE CALIBRATION TEST")
    print("="*80 + "\n")
    
    calibrator = ConfidenceCalibrator()
    
    # Create synthetic calibration data
    np.random.seed(42)
    n_samples = 100
    
    # Simulate overconfident model (predicts high confidence but is often wrong)
    raw_confidences = np.random.beta(5, 2, n_samples)  # Skewed toward high values
    # Actual outcomes are lower than confidence
    true_outcomes = (np.random.random(n_samples) < (raw_confidences * 0.6)).astype(float)
    
    timing_data = {
        'confidences': raw_confidences,
        'outcomes': true_outcomes
    }
    
    print("Synthetic Calibration Data:")
    print(f"  Samples: {n_samples}")
    print(f"  Mean raw confidence: {raw_confidences.mean():.2%}")
    print(f"  Actual accuracy: {true_outcomes.mean():.2%}")
    print()
    
    # Calculate initial ECE
    initial_ece = calibrator._calculate_ece(raw_confidences, true_outcomes)
    print(f"Initial ECE (before calibration): {initial_ece:.4f}")
    print()
    
    # Fit calibrator
    print("Fitting calibrator...")
    if SKLEARN_AVAILABLE:
        success = calibrator.fit(timing_data=timing_data)
        print(f"Calibration success: {success}")
        print()
        
        # Test calibration
        print("Calibration Examples:")
        test_confidences = [0.3, 0.5, 0.7, 0.9]
        for raw in test_confidences:
            calibrated = calibrator.calibrate_timing(raw)
            print(f"  Raw: {raw:.2f} -> Calibrated: {calibrated:.2f}")
        print()
        
        # Test prediction calibration
        print("Full Prediction Calibration:")
        prediction = {
            'timing_confidence': 0.8,
            'content_confidence': 0.7
        }
        calibrated_pred = calibrator.calibrate_prediction(prediction)
        print(f"  Original timing: {prediction['timing_confidence']:.2f}")
        print(f"  Calibrated timing: {calibrated_pred['timing_confidence']:.2f}")
        print(f"  Original content: {prediction['content_confidence']:.2f}")
        print(f"  Calibrated content: {calibrated_pred['content_confidence']:.2f}")
        print()
        
        # Get reliability diagram data
        print("Reliability Diagram Data:")
        reliability = calibrator.get_reliability_data('timing')
        print(f"  ECE: {reliability['calibration_error']:.4f}")
        print(f"  Bins with data: {len(reliability['bins'])}")
        for bin_data in reliability['bins'][:3]:  # First 3 bins
            print(f"    [{bin_data['bin_lower']:.1f}-{bin_data['bin_upper']:.1f}]: "
                  f"conf={bin_data['mean_confidence']:.2f}, acc={bin_data['mean_accuracy']:.2f}, "
                  f"gap={bin_data['gap']:.2f}")
    else:
        print("sklearn not available - skipping calibration tests")
    
    print()
    print("="*80 + "\n")


if __name__ == "__main__":
    test_calibration()

