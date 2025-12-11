"""
Database models for Trump prediction system.
Uses SQLAlchemy ORM for database-agnostic design (SQLite -> PostgreSQL).
"""

from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone, date
import os

Base = declarative_base()

import json
from functools import partial

def custom_json_serializer(obj):
    """Custom JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()  # Convert datetime to an ISO 8601 string
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

class Post(Base):
    """Truth Social posts from Trump"""
    __tablename__ = 'posts'
    
    post_id = Column(String, primary_key=True)
    user_id = Column(String, index=True)
    content = Column(Text)
    created_at = Column(DateTime, index=True)
    url = Column(String)
    
    # Engagement metrics
    replies_count = Column(Integer, default=0)
    reblogs_count = Column(Integer, default=0)
    favourites_count = Column(Integer, default=0)
    
    # Metadata
    media_urls = Column(Text)  # JSON string
    fetched_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # For future embedding storage
    content_embedding = Column(Text)  # JSON array as string
    
    def __repr__(self):
        return f"<Post {self.post_id} at {self.created_at}>"


class Prediction(Base):
    """Predictions made by the system"""
    __tablename__ = 'predictions'
    
    prediction_id = Column(String, primary_key=True)
    predicted_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Timing prediction
    predicted_time = Column(DateTime)
    predicted_time_confidence = Column(Float)
    
    # Content prediction
    predicted_content = Column(Text)
    predicted_content_confidence = Column(Float)
    
    # Model versions
    timing_model_version = Column(String)
    content_model_version = Column(String)
    
    # Actual outcome (NULL until verified)
    actual_post_id = Column(String)
    actual_time = Column(DateTime)
    actual_content = Column(Text)
    
    # Evaluation metrics (calculated after actual post)
    timing_error_hours = Column(Float)
    bertscore_f1 = Column(Float)
    was_correct = Column(Boolean)  # Combined success metric
    
    # Context at prediction time
    context_data = Column(JSON)  # Recent news, features, etc.
    
    def __repr__(self):
        return f"<Prediction {self.prediction_id} for {self.predicted_time}>"


class ModelMetrics(Base):
    """Track model performance over time"""
    __tablename__ = 'model_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    recorded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    model_type = Column(String)  # 'timing' or 'content'
    model_version = Column(String)
    
    # Metrics
    metric_name = Column(String)
    metric_value = Column(Float)
    
    # Evaluation window
    eval_start_date = Column(DateTime)
    eval_end_date = Column(DateTime)
    num_samples = Column(Integer)
    
    def __repr__(self):
        return f"<Metrics {self.model_type} {self.metric_name}={self.metric_value}>"


class DataCollectionLog(Base):
    """Log of data collection attempts"""
    __tablename__ = 'data_collection_log'

    id = Column(Integer, primary_key=True, autoincrement=True)
    collected_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    source = Column(String)  # 'apify', 'scrapecreators', 'manual', etc.
    status = Column(String)  # 'success', 'failure', 'partial'
    num_posts_collected = Column(Integer)
    error_message = Column(Text)

    def __repr__(self):
        return f"<Collection {self.source} at {self.collected_at}: {self.status}>"


class CronRunLog(Base):
    """Track cron job executions and metrics"""
    __tablename__ = 'cron_run_log'

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_name = Column(String, index=True)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    completed_at = Column(DateTime)
    status = Column(String, default='running')  # running, success, failed
    records_processed = Column(Integer, default=0)
    api_calls = Column(Integer, default=0)
    error_message = Column(Text)
    extra_metadata = Column(JSON)

    def __repr__(self):
        return f"<CronRunLog {self.job_name} {self.status} at {self.started_at}>"


class ModelVersion(Base):
    """Track all model versions and their metadata"""
    __tablename__ = 'model_versions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    version_id = Column(String, unique=True, index=True)  # e.g., "prophet_v1_20251106_120000"
    model_type = Column(String, index=True)  # 'timing' or 'content'

    # Training metadata
    trained_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    training_duration_seconds = Column(Float)
    training_data_start = Column(DateTime)
    training_data_end = Column(DateTime)
    num_training_samples = Column(Integer)

    # Model file info
    file_path = Column(String)  # Path to saved model file
    file_size_bytes = Column(Integer)

    # Model configuration
    hyperparameters = Column(JSON)  # Model-specific config
    algorithm = Column(String)  # 'prophet', 'neural_tpp', 'claude_api', etc.

    # Status and promotion
    status = Column(String, default='trained')  # 'trained', 'active', 'archived', 'failed'
    is_production = Column(Boolean, default=False, index=True)  # Currently in production
    promoted_at = Column(DateTime)

    # Notes
    notes = Column(Text)
    created_by = Column(String, default='system')  # 'system', 'manual', 'cron'

    def __repr__(self):
        return f"<ModelVersion {self.version_id} ({self.status})>"


class TrainingRun(Base):
    """Track individual model training runs with detailed metrics"""
    __tablename__ = 'training_runs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, unique=True, index=True)
    model_version_id = Column(String, index=True)  # Links to ModelVersion

    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    completed_at = Column(DateTime)
    duration_seconds = Column(Float)

    # Training details
    status = Column(String)  # 'running', 'completed', 'failed'
    error_message = Column(Text)

    # Data used
    num_training_samples = Column(Integer)
    num_validation_samples = Column(Integer)
    training_data_start = Column(DateTime)
    training_data_end = Column(DateTime)

    # Training metrics (during training)
    train_loss = Column(Float)
    val_loss = Column(Float)

    # Evaluation metrics (on test set)
    test_mae_hours = Column(Float)  # For timing models
    test_within_6h_accuracy = Column(Float)
    test_within_24h_accuracy = Column(Float)
    test_bertscore_f1 = Column(Float)  # For content models

    # Comparison with previous model
    previous_model_version_id = Column(String)
    improvement_percentage = Column(Float)  # Positive = better, negative = worse

    # Configuration snapshot
    config_snapshot = Column(JSON)

    # Auto-promotion decision
    promoted_to_production = Column(Boolean, default=False)
    promotion_reason = Column(Text)

    def __repr__(self):
        return f"<TrainingRun {self.run_id} ({self.status})>"


class ModelEvaluation(Base):
    """Store detailed evaluation results for model comparison"""
    __tablename__ = 'model_evaluations'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version_id = Column(String, index=True)
    evaluated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Evaluation dataset info
    eval_dataset_start = Column(DateTime)
    eval_dataset_end = Column(DateTime)
    num_samples = Column(Integer)

    # Timing model metrics
    mae_hours = Column(Float)
    rmse_hours = Column(Float)
    median_error_hours = Column(Float)
    within_6h_accuracy = Column(Float)
    within_12h_accuracy = Column(Float)
    within_24h_accuracy = Column(Float)

    # Content model metrics
    bertscore_precision = Column(Float)
    bertscore_recall = Column(Float)
    bertscore_f1 = Column(Float)
    bleu_score = Column(Float)
    avg_length_similarity = Column(Float)

    # Combined metrics
    overall_score = Column(Float)  # Composite metric for comparison

    # Detailed results
    predictions_json = Column(JSON)  # Store individual predictions for analysis

    def __repr__(self):
        return f"<Evaluation {self.model_version_id} score={self.overall_score}>"


class ContextSnapshot(Base):
    """Store real-time context data for predictions"""
    __tablename__ = 'context_snapshots'

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(String, unique=True, index=True)
    captured_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    # News data
    top_headlines = Column(JSON)  # List of recent news headlines
    political_news = Column(JSON)  # Politics-specific news
    news_summary = Column(Text)  # Brief summary of major news

    # Trending topics
    trending_topics = Column(JSON)  # Google Trends data
    trending_keywords = Column(JSON)  # Top keywords
    trend_categories = Column(JSON)  # Categorized trends

    # Market data
    sp500_value = Column(Float)  # S&P 500 current value
    sp500_change_pct = Column(Float)  # Daily change %
    dow_value = Column(Float)  # Dow Jones current value
    dow_change_pct = Column(Float)  # Daily change %
    market_sentiment = Column(String)  # 'bullish', 'bearish', 'neutral'

    # Events and calendar
    upcoming_events = Column(JSON)  # Political events, debates, etc.
    recent_events = Column(JSON)  # Events from last 24h

    # Social media context
    twitter_trends = Column(JSON)  # Twitter/X trending topics (if available)
    viral_topics = Column(JSON)  # Topics going viral

    # Weather/external factors
    major_weather_events = Column(JSON)  # Hurricanes, storms, etc.

    # Metadata
    data_sources = Column(JSON)  # Which APIs were used
    fetch_duration_seconds = Column(Float)
    fetch_errors = Column(JSON)  # Any errors during fetching

    # Quality metrics
    completeness_score = Column(Float)  # How complete is this context (0-1)
    freshness_score = Column(Float)  # How recent is the data (0-1)

    # Usage tracking
    used_in_predictions = Column(Integer, default=0)  # How many predictions used this
    prediction_ids = Column(JSON)  # List of prediction IDs using this context

    def __repr__(self):
        return f"<ContextSnapshot {self.snapshot_id} at {self.captured_at}>"


# Database initialization functions
def get_engine(database_url=None):
    """Get database engine"""
    if database_url is None:
        # Use absolute path for SQLite to avoid issues with working directory
        default_db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data', 'trump_predictions.db'
        )
        database_url = os.getenv('DATABASE_URL', f'sqlite:///{default_db_path}')

    # Create data directory if using SQLite and directory doesn't exist
    if database_url.startswith('sqlite:///'):
        # Extract the file path from the URL
        db_path = database_url.replace('sqlite:///', '')
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            print(f"Created database directory: {db_dir}")

    custom_dumper = partial(json.dumps, default=custom_json_serializer)

    engine = create_engine(database_url, echo=False, json_serializer=custom_dumper)
    return engine


def init_db(database_url=None):
    """Initialize database with all tables"""
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine=None):
    """Get database session"""
    if engine is None:
        engine = get_engine()
    
    Session = sessionmaker(bind=engine)
    return Session()


if __name__ == "__main__":
    # Test database creation
    print("Creating database...")
    engine = init_db()
    print(f"Database created successfully!")
    
    # Test session
    session = get_session(engine)
    print(f"Session created: {session}")
    session.close()
