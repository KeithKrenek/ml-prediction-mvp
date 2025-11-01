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


# Database initialization functions
def get_engine(database_url=None):
    """Get database engine"""
    if database_url is None:
        database_url = os.getenv('DATABASE_URL', 'sqlite:///./data/trump_predictions.db')

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
