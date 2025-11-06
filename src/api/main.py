"""
FastAPI REST API for Trump Post Prediction Service.

Provides endpoints for:
- Making predictions
- Retrieving historical predictions
- Getting model status
- Health checks
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
import os
import sys
from pathlib import Path
import yaml
import threading

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

# Load environment
load_dotenv()

# Import prediction system
from src.predictor import TrumpPostPredictor
from src.data.database import get_session, Prediction, Post, ModelVersion, TrainingRun, ModelEvaluation
from src.models.model_registry import ModelRegistry
from src.validation.validator import PredictionValidator

# Initialize FastAPI app
app = FastAPI(
    title="Trump Truth Social Post Prediction API",
    description="Real-time ML pipeline predicting timing and content of Truth Social posts",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance (loaded on startup)
predictor: Optional[TrumpPostPredictor] = None

# Global scheduler instance
scheduler: Optional[BackgroundScheduler] = None

# Global model registry instance
model_registry = ModelRegistry()

# Global validation instance
validator = PredictionValidator()

# Prediction job lock (prevents overlapping predictions)
prediction_job_lock = threading.Lock()
prediction_job_running = False

# Load configuration
config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

scheduling_config = config.get('scheduling', {})


# ============================================================================
# Pydantic Models (API Schemas)
# ============================================================================

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction_id: str
    predicted_at: datetime
    predicted_time: datetime
    predicted_content: str
    timing_confidence: float = Field(ge=0, le=1, description="Confidence for timing (0-1)")
    content_confidence: float = Field(ge=0, le=1, description="Confidence for content (0-1)")
    overall_confidence: float = Field(ge=0, le=1, description="Combined confidence (0-1)")
    timing_model_version: str
    content_model_version: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction_id": "123e4567-e89b-12d3-a456-426614174000",
                "predicted_at": "2025-11-01T10:30:00",
                "predicted_time": "2025-11-04T08:30:00",
                "predicted_content": "The Fake News Media is at it again!",
                "timing_confidence": 0.753,
                "content_confidence": 0.700,
                "overall_confidence": 0.726,
                "timing_model_version": "prophet-v1",
                "content_model_version": "claude-sonnet-4.5"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    models_loaded: bool
    database_connected: bool


class ModelStatus(BaseModel):
    """Model status information"""
    timing_model_loaded: bool
    content_model_loaded: bool
    total_posts_in_db: int
    total_predictions_made: int
    last_post_time: Optional[datetime]
    last_prediction_time: Optional[datetime]


class SchedulerStatus(BaseModel):
    """Scheduler status information"""
    enabled: bool
    running: bool
    next_run_time: Optional[datetime]
    prediction_interval_hours: Optional[float]
    cron_expression: Optional[str]
    total_scheduled_predictions: int
    last_scheduled_prediction_time: Optional[datetime]


class SchedulerControl(BaseModel):
    """Control scheduler (enable/disable)"""
    enabled: bool


# ============================================================================
# Scheduled Job Functions
# ============================================================================

def scheduled_prediction_job():
    """
    Background job that makes predictions on a schedule.
    Uses a lock to prevent overlapping executions.
    """
    global prediction_job_running

    # Check if a job is already running
    if not prediction_job_lock.acquire(blocking=False):
        logger.warning("Scheduled prediction skipped - previous job still running")
        return

    try:
        prediction_job_running = True
        logger.info("="*60)
        logger.info("Starting scheduled prediction job")
        logger.info("="*60)

        # Check if predictor is ready
        if predictor is None:
            logger.error("Predictor not initialized - cannot make scheduled prediction")
            return

        # Ensure models are trained
        if not hasattr(predictor.timing_model, 'model') or predictor.timing_model.model is None:
            logger.info("Training models for first scheduled prediction...")
            success = predictor.train_models()
            if not success:
                logger.error("Model training failed - scheduled prediction aborted")
                return
            predictor.save_models()

        # Make prediction
        logger.info("Making scheduled prediction...")
        prediction = predictor.predict(save_to_db=True)

        if prediction:
            logger.success(f"Scheduled prediction completed successfully!")
            logger.info(f"Prediction ID: {prediction['prediction_id']}")
            logger.info(f"Predicted time: {prediction['predicted_time']}")
            logger.info(f"Overall confidence: {prediction['overall_confidence']:.1%}")
        else:
            logger.error("Scheduled prediction failed")

        logger.info("="*60)

    except Exception as e:
        logger.error(f"Error in scheduled prediction job: {e}")

        # Retry if configured
        if scheduling_config.get('retry_on_failure', True):
            retry_delay = scheduling_config.get('retry_delay_minutes', 30)
            logger.info(f"Will retry in {retry_delay} minutes")
            # Note: APScheduler will continue with normal schedule

    finally:
        prediction_job_running = False
        prediction_job_lock.release()


def setup_scheduler():
    """Initialize and configure the prediction scheduler"""
    global scheduler

    if not scheduling_config.get('enabled', True):
        logger.info("Prediction scheduling is disabled in configuration")
        return

    logger.info("Setting up prediction scheduler...")

    # Create scheduler
    scheduler = BackgroundScheduler()

    # Determine trigger (interval vs cron)
    cron_expression = scheduling_config.get('cron_expression')

    if cron_expression:
        # Use cron expression if provided
        logger.info(f"Using cron expression: {cron_expression}")
        trigger = CronTrigger.from_crontab(cron_expression)
    else:
        # Use interval (default: every 6 hours)
        interval_hours = scheduling_config.get('prediction_interval_hours', 6)
        logger.info(f"Using interval trigger: every {interval_hours} hours")
        trigger = IntervalTrigger(hours=interval_hours)

    # Add job to scheduler
    scheduler.add_job(
        scheduled_prediction_job,
        trigger=trigger,
        id='prediction_job',
        name='Scheduled Trump Post Prediction',
        replace_existing=True,
        max_instances=scheduling_config.get('max_concurrent_predictions', 1)
    )

    # Start scheduler
    scheduler.start()
    logger.success("Prediction scheduler started successfully!")

    # Log next run time
    next_run = scheduler.get_job('prediction_job').next_run_time
    logger.info(f"Next scheduled prediction: {next_run}")


def stop_scheduler():
    """Shutdown the prediction scheduler"""
    global scheduler

    if scheduler is not None:
        logger.info("Stopping prediction scheduler...")
        scheduler.shutdown(wait=True)
        logger.success("Prediction scheduler stopped")


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the predictor on startup"""
    global predictor

    logger.info("Starting Trump Post Prediction API...")

    try:
        # Initialize predictor
        predictor = TrumpPostPredictor()

        # Try to load existing models or train new ones
        try:
            # Check if models exist
            model_path = Path(__file__).parent.parent / "models" / "timing_model.pkl"
            if model_path.exists():
                logger.info("Loading existing models...")
                # Models will be loaded on first prediction
            else:
                logger.info("No existing models found. Training new models...")
                predictor.train_models()
                predictor.save_models()

        except Exception as e:
            logger.warning(f"Could not load/train models on startup: {e}")
            logger.info("Models will be trained on first prediction request")

        # Setup prediction scheduler
        setup_scheduler()

        logger.success("API startup complete!")

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # Don't crash the API, allow it to start even if models aren't loaded


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Trump Post Prediction API...")

    # Stop scheduler
    stop_scheduler()


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Trump Truth Social Post Prediction API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "endpoints": {
            "predict": "/predict",
            "predictions": "/predictions",
            "health": "/health",
            "status": "/status"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    
    # Check database connection
    db_connected = False
    try:
        session = get_session()
        session.execute("SELECT 1")
        session.close()
        db_connected = True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
    
    # Check if models are loaded
    models_loaded = predictor is not None
    
    return {
        "status": "healthy" if (db_connected and models_loaded) else "degraded",
        "timestamp": datetime.now(),
        "models_loaded": models_loaded,
        "database_connected": db_connected
    }


@app.get("/status", response_model=ModelStatus, tags=["General"])
async def get_status():
    """Get detailed system status"""
    
    session = get_session()
    
    try:
        # Get statistics
        total_posts = session.query(Post).count()
        total_predictions = session.query(Prediction).count()
        
        # Get latest post
        latest_post = session.query(Post).order_by(Post.created_at.desc()).first()
        last_post_time = latest_post.created_at if latest_post else None
        
        # Get latest prediction
        latest_prediction = session.query(Prediction).order_by(Prediction.predicted_at.desc()).first()
        last_prediction_time = latest_prediction.predicted_at if latest_prediction else None
        
        return {
            "timing_model_loaded": predictor is not None,
            "content_model_loaded": predictor is not None,
            "total_posts_in_db": total_posts,
            "total_predictions_made": total_predictions,
            "last_post_time": last_post_time,
            "last_prediction_time": last_prediction_time
        }
    
    finally:
        session.close()


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def make_prediction(background_tasks: BackgroundTasks):
    """
    Make a new prediction for Trump's next post.
    
    Returns timing and content prediction with confidence scores.
    """
    
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service not initialized. Please try again in a moment."
        )
    
    try:
        # Ensure models are trained
        if not hasattr(predictor.timing_model, 'model') or predictor.timing_model.model is None:
            logger.info("Training models before first prediction...")
            success = predictor.train_models()
            if not success:
                raise HTTPException(
                    status_code=500,
                    detail="Model training failed. Check logs for details."
                )
            
            # Save models in background
            background_tasks.add_task(predictor.save_models)
        
        # Make prediction
        logger.info("Making prediction via API...")
        prediction = predictor.predict(save_to_db=True)
        
        if not prediction:
            raise HTTPException(
                status_code=500,
                detail="Prediction failed. Check logs for details."
            )
        
        # Return formatted response
        return PredictionResponse(
            prediction_id=prediction['prediction_id'],
            predicted_at=prediction['predicted_at'],
            predicted_time=prediction['predicted_time'],
            predicted_content=prediction['predicted_content'],
            timing_confidence=prediction['timing_confidence'],
            content_confidence=prediction['content_confidence'],
            overall_confidence=prediction['overall_confidence'],
            timing_model_version=prediction['timing_model_version'],
            content_model_version=prediction['content_model_version']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/predictions", tags=["Predictions"])
async def get_predictions(
    limit: int = 10,
    offset: int = 0
):
    """
    Get historical predictions.
    
    Parameters:
    - limit: Number of predictions to return (default: 10, max: 100)
    - offset: Number of predictions to skip (default: 0)
    """
    
    # Validate parameters
    if limit > 100:
        limit = 100
    if offset < 0:
        offset = 0
    
    session = get_session()
    
    try:
        # Query predictions
        predictions = (
            session.query(Prediction)
            .order_by(Prediction.predicted_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )
        
        # Format response
        results = []
        for pred in predictions:
            results.append({
                "prediction_id": pred.prediction_id,
                "predicted_at": pred.predicted_at,
                "predicted_time": pred.predicted_time,
                "predicted_content": pred.predicted_content,
                "timing_confidence": pred.predicted_time_confidence,
                "content_confidence": pred.predicted_content_confidence,
                "timing_model_version": pred.timing_model_version,
                "content_model_version": pred.content_model_version,
                "actual_post_time": pred.actual_post_time,
                "accuracy_score": pred.accuracy_score
            })
        
        return {
            "predictions": results,
            "count": len(results),
            "limit": limit,
            "offset": offset
        }
    
    finally:
        session.close()


@app.get("/predictions/{prediction_id}", response_model=PredictionResponse, tags=["Predictions"])
async def get_prediction(prediction_id: str):
    """Get a specific prediction by ID"""
    
    session = get_session()
    
    try:
        prediction = session.query(Prediction).filter_by(prediction_id=prediction_id).first()
        
        if not prediction:
            raise HTTPException(
                status_code=404,
                detail=f"Prediction {prediction_id} not found"
            )
        
        return PredictionResponse(
            prediction_id=prediction.prediction_id,
            predicted_at=prediction.predicted_at,
            predicted_time=prediction.predicted_time,
            predicted_content=prediction.predicted_content,
            timing_confidence=prediction.predicted_time_confidence,
            content_confidence=prediction.predicted_content_confidence,
            overall_confidence=(
                prediction.predicted_time_confidence + 
                prediction.predicted_content_confidence
            ) / 2,
            timing_model_version=prediction.timing_model_version,
            content_model_version=prediction.content_model_version
        )
    
    finally:
        session.close()


@app.get("/posts/recent", tags=["Data"])
async def get_recent_posts(limit: int = 10):
    """Get recent posts from database"""

    if limit > 100:
        limit = 100

    session = get_session()

    try:
        posts = (
            session.query(Post)
            .order_by(Post.created_at.desc())
            .limit(limit)
            .all()
        )

        results = []
        for post in posts:
            results.append({
                "post_id": post.post_id,
                "content": post.content,
                "created_at": post.created_at,
                "url": post.url,
                "replies_count": post.replies_count,
                "reblogs_count": post.reblogs_count,
                "favourites_count": post.favourites_count
            })

        return {
            "posts": results,
            "count": len(results)
        }

    finally:
        session.close()


@app.get("/scheduler/status", response_model=SchedulerStatus, tags=["Scheduler"])
async def get_scheduler_status():
    """
    Get the current status of the prediction scheduler.

    Returns information about whether scheduling is enabled, running,
    next run time, and scheduling configuration.
    """

    # Check if scheduler exists and is running
    scheduler_running = scheduler is not None and scheduler.running
    next_run = None

    if scheduler_running:
        job = scheduler.get_job('prediction_job')
        if job:
            next_run = job.next_run_time

    # Get count of scheduled predictions from database
    session = get_session()
    try:
        # Note: We don't currently track which predictions were scheduled vs manual
        # This could be added by adding a field to the Prediction model
        total_predictions = session.query(Prediction).count()
        last_prediction = session.query(Prediction).order_by(Prediction.predicted_at.desc()).first()
        last_prediction_time = last_prediction.predicted_at if last_prediction else None
    finally:
        session.close()

    return {
        "enabled": scheduling_config.get('enabled', True),
        "running": scheduler_running,
        "next_run_time": next_run,
        "prediction_interval_hours": scheduling_config.get('prediction_interval_hours'),
        "cron_expression": scheduling_config.get('cron_expression'),
        "total_scheduled_predictions": total_predictions,
        "last_scheduled_prediction_time": last_prediction_time
    }


@app.post("/scheduler/control", tags=["Scheduler"])
async def control_scheduler(control: SchedulerControl):
    """
    Enable or disable the prediction scheduler.

    Note: This controls the scheduler at runtime but does not persist
    the setting. To permanently enable/disable, modify config.yaml.
    """
    global scheduler

    if control.enabled:
        # Enable/start scheduler
        if scheduler is None or not scheduler.running:
            logger.info("Starting scheduler via API request...")
            setup_scheduler()
            return {
                "status": "success",
                "message": "Scheduler started",
                "enabled": True,
                "next_run_time": scheduler.get_job('prediction_job').next_run_time if scheduler else None
            }
        else:
            return {
                "status": "info",
                "message": "Scheduler is already running",
                "enabled": True,
                "next_run_time": scheduler.get_job('prediction_job').next_run_time
            }
    else:
        # Disable/stop scheduler
        if scheduler is not None and scheduler.running:
            logger.info("Stopping scheduler via API request...")
            stop_scheduler()
            return {
                "status": "success",
                "message": "Scheduler stopped",
                "enabled": False,
                "next_run_time": None
            }
        else:
            return {
                "status": "info",
                "message": "Scheduler is already stopped",
                "enabled": False,
                "next_run_time": None
            }


@app.post("/scheduler/trigger", tags=["Scheduler"])
async def trigger_scheduled_prediction():
    """
    Manually trigger a scheduled prediction job immediately.

    This is useful for testing the scheduler or making an immediate
    prediction without waiting for the next scheduled run.
    """

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Predictor not initialized"
        )

    # Check if a job is already running
    if prediction_job_running:
        return {
            "status": "skipped",
            "message": "A prediction job is already running",
            "job_running": True
        }

    try:
        # Run the scheduled job function
        logger.info("Manually triggering scheduled prediction...")
        scheduled_prediction_job()

        return {
            "status": "success",
            "message": "Scheduled prediction triggered successfully",
            "job_running": False
        }

    except Exception as e:
        logger.error(f"Error triggering scheduled prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger prediction: {str(e)}"
        )


# ============================================================================
# Model Management Endpoints
# ============================================================================

@app.get("/models/versions", tags=["Models"])
async def get_model_versions(
    model_type: Optional[str] = None,
    limit: int = 20
):
    """
    Get all model versions.

    Parameters:
    - model_type: Filter by type ('timing' or 'content')
    - limit: Maximum number of versions to return (default: 20, max: 100)
    """

    if limit > 100:
        limit = 100

    try:
        versions = model_registry.get_all_versions(model_type=model_type, limit=limit)

        results = []
        for v in versions:
            results.append({
                "version_id": v.version_id,
                "model_type": v.model_type,
                "algorithm": v.algorithm,
                "trained_at": v.trained_at,
                "status": v.status,
                "is_production": v.is_production,
                "promoted_at": v.promoted_at,
                "num_training_samples": v.num_training_samples,
                "training_duration_seconds": v.training_duration_seconds,
                "file_size_bytes": v.file_size_bytes,
                "notes": v.notes
            })

        return {
            "versions": results,
            "count": len(results),
            "model_type_filter": model_type
        }

    except Exception as e:
        logger.error(f"Error fetching model versions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch model versions: {str(e)}"
        )


@app.get("/models/production", tags=["Models"])
async def get_production_models():
    """Get current production models for timing and content."""

    try:
        timing_model = model_registry.get_production_model('timing')
        content_model = model_registry.get_production_model('content')

        return {
            "timing_model": {
                "version_id": timing_model.version_id if timing_model else None,
                "algorithm": timing_model.algorithm if timing_model else None,
                "trained_at": timing_model.trained_at if timing_model else None,
                "promoted_at": timing_model.promoted_at if timing_model else None,
                "num_training_samples": timing_model.num_training_samples if timing_model else None
            } if timing_model else None,
            "content_model": {
                "version_id": content_model.version_id if content_model else None,
                "algorithm": content_model.algorithm if content_model else None,
                "trained_at": content_model.trained_at if content_model else None,
                "promoted_at": content_model.promoted_at if content_model else None,
                "num_training_samples": content_model.num_training_samples if content_model else None
            } if content_model else None
        }

    except Exception as e:
        logger.error(f"Error fetching production models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch production models: {str(e)}"
        )


@app.get("/models/training-history", tags=["Models"])
async def get_training_history(limit: int = 10):
    """
    Get recent training runs.

    Parameters:
    - limit: Number of runs to return (default: 10, max: 50)
    """

    if limit > 50:
        limit = 50

    try:
        runs = model_registry.get_training_history(limit=limit)

        results = []
        for run in runs:
            results.append({
                "run_id": run.run_id,
                "model_version_id": run.model_version_id,
                "started_at": run.started_at,
                "completed_at": run.completed_at,
                "duration_seconds": run.duration_seconds,
                "status": run.status,
                "num_training_samples": run.num_training_samples,
                "test_mae_hours": run.test_mae_hours,
                "test_within_6h_accuracy": run.test_within_6h_accuracy,
                "promoted_to_production": run.promoted_to_production,
                "promotion_reason": run.promotion_reason,
                "error_message": run.error_message
            })

        return {
            "training_runs": results,
            "count": len(results)
        }

    except Exception as e:
        logger.error(f"Error fetching training history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch training history: {str(e)}"
        )


@app.post("/models/promote/{version_id}", tags=["Models"])
async def promote_model(version_id: str):
    """
    Manually promote a model version to production.

    This will demote the current production model and promote
    the specified version.
    """

    try:
        success = model_registry.promote_to_production(
            version_id=version_id,
            reason="Manual promotion via API"
        )

        if success:
            return {
                "status": "success",
                "message": f"Model {version_id} promoted to production",
                "version_id": version_id
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to promote model {version_id}"
            )

    except Exception as e:
        logger.error(f"Error promoting model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to promote model: {str(e)}"
        )


@app.post("/models/retrain", tags=["Models"])
async def trigger_retraining(background_tasks: BackgroundTasks):
    """
    Manually trigger model retraining.

    This will start a background task to retrain both timing
    and content models with the latest data.

    Note: This is a long-running operation (5-10 minutes).
    Check /models/training-history for progress.
    """

    try:
        # Import here to avoid circular dependency
        import subprocess

        logger.info("Triggering manual model retraining via API...")

        # Run retraining script in background
        def run_retraining():
            try:
                result = subprocess.run(
                    [sys.executable, "scripts/retrain_models.py"],
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout
                )
                logger.info(f"Retraining completed with code {result.returncode}")
                if result.stdout:
                    logger.info(f"Output: {result.stdout}")
                if result.stderr:
                    logger.error(f"Errors: {result.stderr}")
            except Exception as e:
                logger.error(f"Retraining failed: {e}")

        background_tasks.add_task(run_retraining)

        return {
            "status": "started",
            "message": "Model retraining started in background",
            "note": "Check /models/training-history for progress"
        }

    except Exception as e:
        logger.error(f"Error triggering retraining: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger retraining: {str(e)}"
        )


@app.get("/models/compare/{version_a}/{version_b}", tags=["Models"])
async def compare_models(version_a: str, version_b: str):
    """
    Compare two model versions based on their evaluations.

    Returns detailed comparison and recommendation.
    """

    try:
        comparison = model_registry.compare_models(version_a, version_b)

        if comparison['comparison'] == 'incomplete':
            raise HTTPException(
                status_code=400,
                detail=comparison['reason']
            )

        return {
            "version_a": {
                "version_id": version_a,
                "score": comparison['model_a']['score']
            },
            "version_b": {
                "version_id": version_b,
                "score": comparison['model_b']['score']
            },
            "winner": comparison['winner'],
            "improvement_percentage": comparison['improvement_percentage'],
            "recommendation": comparison['recommendation'],
            "reason": comparison['reason']
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compare models: {str(e)}"
        )


# ============================================================================
# Validation Endpoints
# ============================================================================

@app.get("/validation/stats", tags=["Validation"])
async def get_validation_stats():
    """
    Get overall validation statistics.

    Returns aggregate metrics for all validated predictions.
    """

    try:
        stats = validator.get_validation_stats()
        return stats

    except Exception as e:
        logger.error(f"Error fetching validation stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch validation stats: {str(e)}"
        )


@app.post("/validation/validate", tags=["Validation"])
async def trigger_validation(background_tasks: BackgroundTasks):
    """
    Manually trigger validation of unvalidated predictions.

    This will find all predictions whose predicted time has passed,
    match them to actual posts, and calculate accuracy metrics.

    Note: This is a background task that may take a few seconds.
    """

    try:
        import subprocess

        logger.info("Triggering manual validation via API...")

        # Run validation script in background
        def run_validation():
            try:
                result = subprocess.run(
                    [sys.executable, "scripts/validate_predictions.py"],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                logger.info(f"Validation completed with code {result.returncode}")
                if result.stdout:
                    logger.info(f"Output: {result.stdout}")
                if result.stderr:
                    logger.error(f"Errors: {result.stderr}")
            except Exception as e:
                logger.error(f"Validation failed: {e}")

        background_tasks.add_task(run_validation)

        return {
            "status": "started",
            "message": "Validation started in background",
            "note": "Check /validation/stats for updated statistics"
        }

    except Exception as e:
        logger.error(f"Error triggering validation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger validation: {str(e)}"
        )


@app.get("/validation/timeline", tags=["Validation"])
async def get_validation_timeline(days_back: int = 30):
    """
    Get timeline data for visualization.

    Parameters:
    - days_back: Number of days to look back (default: 30, max: 90)

    Returns:
    - actual_posts: List of actual posts
    - predictions: List of predictions
    - Matched pairs indicated by linked IDs
    """

    if days_back > 90:
        days_back = 90

    try:
        timeline_data = validator.get_timeline_data(days_back=days_back)

        return {
            "actual_posts": timeline_data['actual_posts'],
            "predictions": timeline_data['predictions'],
            "date_range": {
                "start": timeline_data['date_range']['start'],
                "end": timeline_data['date_range']['end']
            },
            "summary": {
                "total_actual": len(timeline_data['actual_posts']),
                "total_predicted": len(timeline_data['predictions']),
                "matched": sum(1 for p in timeline_data['predictions'] if p['actual_post_id'])
            }
        }

    except Exception as e:
        logger.error(f"Error fetching timeline data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch timeline data: {str(e)}"
        )


@app.get("/validation/unvalidated", tags=["Validation"])
async def get_unvalidated_predictions():
    """
    Get list of predictions that haven't been validated yet.

    Returns predictions whose predicted time has passed but haven't
    been matched to actual posts.
    """

    try:
        unvalidated = validator.find_unvalidated_predictions()

        results = []
        for pred in unvalidated:
            results.append({
                "prediction_id": pred.prediction_id,
                "predicted_at": pred.predicted_at,
                "predicted_time": pred.predicted_time,
                "predicted_content": pred.predicted_content[:100] + '...',
                "timing_confidence": pred.predicted_time_confidence,
                "content_confidence": pred.predicted_content_confidence
            })

        return {
            "unvalidated_predictions": results,
            "count": len(results)
        }

    except Exception as e:
        logger.error(f"Error fetching unvalidated predictions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch unvalidated predictions: {str(e)}"
        )


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": type(exc).__name__
        }
    )


# ============================================================================
# Development Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    logger.info(f"Starting development server on {host}:{port}")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
