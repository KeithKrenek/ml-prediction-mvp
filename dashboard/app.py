"""
Streamlit Dashboard for Trump Truth Social Post Prediction.

Features:
- Real-time next post prediction
- Historical predictions view
- Model performance metrics
- Posting patterns visualization
- Recent posts timeline
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

# For Streamlit Cloud: Load secrets as environment variables BEFORE any imports
# that might use DATABASE_URL. This MUST happen before importing database modules.
if hasattr(st, 'secrets') and len(st.secrets) > 0:
    try:
        for key in st.secrets:
            value = st.secrets[key]
            # Only set if it's a string (not a nested TOML section)
            if isinstance(value, str):
                os.environ[key] = value
    except Exception as e:
        # Silently handle errors in secrets loading
        pass

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def get_database_info():
    """Get information about which database is being used."""
    db_url = os.getenv('DATABASE_URL', '')
    if not db_url:
        return "SQLite (local)", "⚠️ No DATABASE_URL configured", False
    elif 'neon.tech' in db_url:
        return "Neon PostgreSQL", "✅ Connected to Neon", True
    elif 'postgresql' in db_url or 'postgres' in db_url:
        return "PostgreSQL", "✅ Connected to PostgreSQL", True
    elif 'sqlite' in db_url:
        return "SQLite (local)", "⚠️ Using local SQLite", False
    else:
        return "Unknown", f"Database: {db_url[:30]}...", False


# Import database models (lightweight)
from src.data.database import (
    get_session,
    init_db,
    Post,
    Prediction,
    ContextSnapshot,
    ModelEvaluation,
    ModelVersion,
    CronRunLog,
)

# Initialize database tables on startup (creates tables if they don't exist)
try:
    init_db()
except Exception as e:
    st.error(f"Database initialization failed: {e}")

# LAZY IMPORTS: Heavy ML dependencies are only loaded when needed
# This avoids loading torch/prophet on pages that don't need predictions
_predictor = None
_validator = None
_context_gatherer = None


def get_predictor_lazy():
    """Lazy load the predictor only when needed (avoids torch import on startup)."""
    global _predictor
    if _predictor is None:
        try:
            from src.predictor import TrumpPostPredictor
            _predictor = TrumpPostPredictor()
        except Exception as e:
            st.error(f"Failed to load predictor: {e}")
            return None
    return _predictor


def get_validator_lazy():
    """Lazy load the validator only when needed."""
    global _validator
    if _validator is None:
        try:
            from src.validation.validator import PredictionValidator
            _validator = PredictionValidator()
        except Exception as e:
            st.error(f"Failed to load validator: {e}")
            return None
    return _validator


def get_context_gatherer_lazy():
    """Lazy load the context gatherer only when needed."""
    global _context_gatherer
    if _context_gatherer is None:
        try:
            from src.context.context_gatherer import RealTimeContextGatherer
            _context_gatherer = RealTimeContextGatherer()
        except Exception as e:
            st.error(f"Failed to load context gatherer: {e}")
            return None
    return _context_gatherer


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Trump Post Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# Custom CSS
# ============================================================================

st.markdown("""
<style>
    .big-prediction {
        font-size: 28px;
        font-weight: bold;
        color: #1f77b4;
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .confidence-high {
        color: #2ecc71;
    }
    .confidence-medium {
        color: #f39c12;
    }
    .confidence-low {
        color: #e74c3c;
    }
    .prediction-box {
        border-left: 5px solid #1f77b4;
        padding: 15px;
        margin: 10px 0;
        background-color: #f8f9fa;
        border-radius: 5px;
    }
    .timeline-actual {
        background-color: #2ecc71;
        color: white;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        cursor: pointer;
        transition: all 0.3s;
    }
    .timeline-predicted {
        background-color: #3498db;
        color: white;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        cursor: pointer;
        transition: all 0.3s;
    }
    .timeline-matched {
        background-color: #27ae60;
        border: 3px solid #2ecc71;
    }
    .timeline-unmatched {
        background-color: #e74c3c;
        border: 3px solid #c0392b;
    }
    .timeline-correct {
        box-shadow: 0 0 15px #2ecc71;
    }
    .post-content-preview {
        font-size: 14px;
        font-style: italic;
        margin-top: 5px;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Helper Functions
# ============================================================================

@st.cache_resource
def get_predictor():
    """Initialize and cache the predictor (lazy loaded to avoid torch import on startup)"""
    return get_predictor_lazy()


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_recent_posts(limit=50):
    """Load recent posts from database"""
    session = get_session()
    try:
        posts = (
            session.query(Post)
            .order_by(Post.created_at.desc())
            .limit(limit)
            .all()
        )
        
        data = []
        for post in posts:
            content = post.content or ""
            # Handle potential None values for created_at
            created_at = post.created_at
            if created_at is None:
                continue  # Skip posts without dates
            data.append({
                'Date': created_at,
                'Content': content[:100] + '...' if len(content) > 100 else content,
                'Replies': post.replies_count or 0,
                'Reblogs': post.reblogs_count or 0,
                'Favorites': post.favourites_count or 0,
            })
        
        return pd.DataFrame(data)
    except Exception as e:
        # Log the error for debugging
        import traceback
        st.error(f"Database error: {e}")
        st.code(traceback.format_exc())
        return pd.DataFrame()
    finally:
        session.close()


@st.cache_data(ttl=300)
def load_predictions(limit=20):
    """Load recent predictions from database"""
    session = get_session()
    try:
        predictions = (
            session.query(Prediction)
            .order_by(Prediction.predicted_at.desc())
            .limit(limit)
            .all()
        )
        
        data = []
        for pred in predictions:
            timing_error = pred.timing_error_hours
            status = 'missed'
            if pred.was_correct:
                status = 'correct'
            elif pred.actual_post_id:
                status = 'late'

            data.append({
                'Predicted At': pred.predicted_at,
                'Predicted Time': pred.predicted_time,
                'Actual Time': pred.actual_time,
                'Timing Error (h)': timing_error,
                'Status': status,
                'Content Preview': pred.predicted_content[:50] + '...',
                'Timing Confidence': pred.predicted_time_confidence,
                'Content Confidence': pred.predicted_content_confidence,
                'Content Similarity': pred.bertscore_f1
            })
        
        return pd.DataFrame(data)
    finally:
        session.close()


@st.cache_data(ttl=300)
def load_evaluation_metrics():
    """Load the most recent timing and content evaluation metrics."""
    session = get_session()
    try:
        timing_eval = (
            session.query(ModelEvaluation)
            .join(ModelVersion, ModelEvaluation.model_version_id == ModelVersion.version_id)
            .filter(ModelVersion.model_type == 'timing')
            .order_by(ModelEvaluation.evaluated_at.desc())
            .first()
        )
        content_eval = (
            session.query(ModelEvaluation)
            .join(ModelVersion, ModelEvaluation.model_version_id == ModelVersion.version_id)
            .filter(ModelVersion.model_type == 'content')
            .order_by(ModelEvaluation.evaluated_at.desc())
            .first()
        )

        def serialize(eval_row):
            if not eval_row:
                return None
            return {
                'mae_hours': eval_row.mae_hours,
                'within_6h_accuracy': eval_row.within_6h_accuracy,
                'within_24h_accuracy': eval_row.within_24h_accuracy,
                'bertscore_f1': eval_row.bertscore_f1,
                'evaluated_at': eval_row.evaluated_at
            }

        return {
            'timing': serialize(timing_eval),
            'content': serialize(content_eval)
        }
    finally:
        session.close()


@st.cache_data(ttl=300)
def load_latest_prediction_context():
    """Fetch context metadata for the most recent prediction."""
    session = get_session()
    try:
        latest_prediction = (
            session.query(Prediction)
            .order_by(Prediction.predicted_at.desc())
            .first()
        )
        if latest_prediction and latest_prediction.context_data:
            return latest_prediction.context_data
        return None
    finally:
        session.close()


@st.cache_data(ttl=300)
def load_cron_runs(limit=50):
    """Fetch recent cron job executions."""
    session = get_session()
    try:
        runs = (
            session.query(CronRunLog)
            .order_by(CronRunLog.started_at.desc())
            .limit(limit)
            .all()
        )
        data = []
        for run in runs:
            data.append({
                'Job': run.job_name,
                'Status': run.status,
                'Started': run.started_at,
                'Completed': run.completed_at,
                'Records': run.records_processed,
                'API Calls': run.api_calls,
                'Error': run.error_message,
                'Metadata': run.extra_metadata
            })
        return pd.DataFrame(data)
    except Exception as e:
        # Table might not exist yet - return empty DataFrame
        return pd.DataFrame()
    finally:
        session.close()


@st.cache_data(ttl=300)
def load_prediction_actual_pairs(days_back=30):
    """Return DataFrame linking predictions to actual outcomes."""
    session = get_session()
    cutoff = datetime.now() - timedelta(days=days_back)
    try:
        predictions = (
            session.query(Prediction)
            .filter(Prediction.predicted_at >= cutoff)
            .order_by(Prediction.predicted_at.asc())
            .all()
        )
        data = []
        for pred in predictions:
            actual_time = pred.actual_time
            timing_error = None
            if actual_time and pred.predicted_time:
                timing_error = (actual_time - pred.predicted_time).total_seconds() / 3600
            status = 'open'
            if pred.was_correct:
                status = 'correct'
            elif pred.actual_post_id:
                status = 'matched'
            elif actual_time:
                status = 'late'

            data.append({
                'Prediction ID': pred.prediction_id,
                'Predicted At': pred.predicted_at,
                'Predicted Time': pred.predicted_time,
                'Actual Time': actual_time,
                'Timing Error (h)': timing_error,
                'Status': status,
                'Timing Confidence': pred.predicted_time_confidence,
                'Content Confidence': pred.predicted_content_confidence,
                'Similarity': pred.bertscore_f1,
                'Predicted Content': pred.predicted_content,
                'Actual Content': pred.actual_content,
            })
        return pd.DataFrame(data)
    finally:
        session.close()


def get_posting_stats():
    """Calculate posting statistics"""
    session = get_session()
    try:
        # Total posts
        total_posts = session.query(Post).count()
        
        # Posts in last 7 days
        week_ago = datetime.now() - timedelta(days=7)
        posts_last_week = (
            session.query(Post)
            .filter(Post.created_at >= week_ago)
            .count()
        )
        
        # Latest post
        latest_post = session.query(Post).order_by(Post.created_at.desc()).first()
        
        return {
            'total_posts': total_posts,
            'posts_last_week': posts_last_week,
            'latest_post_time': latest_post.created_at if latest_post else None
        }
    except Exception as e:
        # Table might not exist yet
        return {
            'total_posts': 0,
            'posts_last_week': 0,
            'latest_post_time': None
        }
    finally:
        session.close()


def get_confidence_color(confidence):
    """Return CSS class based on confidence level"""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.5:
        return "confidence-medium"
    else:
        return "confidence-low"


# ============================================================================
# Sidebar
# ============================================================================

st.sidebar.title("🔮 Trump Post Predictor")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigate",
    ["🎯 Make Prediction", "📊 Dashboard", "🎯 Validation Timeline", "📝 Recent Posts", "🌐 Real-time Context", "⚙️ About"]
)

st.sidebar.markdown("---")

# Database connection indicator
db_type, db_status, is_production = get_database_info()
if is_production:
    st.sidebar.success(db_status)
else:
    st.sidebar.warning(db_status)
st.sidebar.caption(f"Database: {db_type}")

st.sidebar.markdown("---")

# Stats in sidebar
stats = get_posting_stats()
st.sidebar.metric("Total Posts in DB", stats['total_posts'])
st.sidebar.metric("Posts Last 7 Days", stats['posts_last_week'])

if stats['latest_post_time']:
    time_since_last = datetime.now() - stats['latest_post_time']
    hours_since = int(time_since_last.total_seconds() / 3600)
    st.sidebar.metric("Hours Since Last Post", hours_since)


# ============================================================================
# Page: Make Prediction
# ============================================================================

if page == "🎯 Make Prediction":
    st.title("🎯 Next Post Prediction")
    st.markdown("Generate a prediction for when Trump will post next and what he'll say.")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
            with st.spinner("Loading prediction models..."):
                try:
                    predictor = get_predictor()
                    
                    if predictor is None:
                        st.error("""
                        ⚠️ **Prediction models not available on Streamlit Cloud**
                        
                        The prediction models require heavy ML dependencies (PyTorch, Prophet) 
                        that aren't compatible with Streamlit Cloud's environment.
                        
                        **Predictions are generated automatically by the backend cron jobs on Render.**
                        
                        Check the **📊 Dashboard** page to see recent predictions!
                        """)
                        st.stop()
                    
                    # Train if needed
                    if not hasattr(predictor.timing_model, 'model') or predictor.timing_model.model is None:
                        st.info("Training models for the first time...")
                        predictor.train_models()
                    
                    # Make prediction
                    prediction = predictor.predict(save_to_db=True)
                    
                    if prediction:
                        st.success("✅ Prediction generated!")
                        
                        # Display timing
                        st.markdown("### 🕐 Predicted Time")
                        predicted_time = prediction['predicted_time']
                        st.markdown(
                            f"<div class='big-prediction'>{predicted_time.strftime('%A, %B %d, %Y at %I:%M %p')}</div>",
                            unsafe_allow_html=True
                        )
                        
                        confidence_class = get_confidence_color(prediction['timing_confidence'])
                        st.markdown(
                            f"<p class='{confidence_class}'>Confidence: {prediction['timing_confidence']:.1%}</p>",
                            unsafe_allow_html=True
                        )
                        
                        # Display content
                        st.markdown("### 📝 Predicted Content")
                        st.markdown(
                            f"<div class='prediction-box'>{prediction['predicted_content']}</div>",
                            unsafe_allow_html=True
                        )
                        
                        confidence_class = get_confidence_color(prediction['content_confidence'])
                        st.markdown(
                            f"<p class='{confidence_class}'>Confidence: {prediction['content_confidence']:.1%}</p>",
                            unsafe_allow_html=True
                        )
                        
                        # Overall stats
                        st.markdown("---")
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Overall Confidence", f"{prediction['overall_confidence']:.1%}")
                        col_b.metric("Timing Model", prediction['timing_model_version'])
                        col_c.metric("Content Model", prediction['content_model_version'])
                    
                    else:
                        st.error("❌ Prediction failed. Check logs for details.")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col1:
        st.markdown("### How It Works")
        st.markdown("""
        **Timing Prediction:**
        - Uses Prophet model trained on historical posting patterns
        - Considers time of day, day of week, and recent posting frequency
        - Typical accuracy: Within 8-12 hours
        
        **Content Generation:**
        - Uses Claude API with few-shot prompting
        - Learns from 10+ example posts
        - Considers current news context and time of day
        - Style similarity: ~60-70% (BERTScore)
        """)


# ============================================================================
# Page: Dashboard
# ============================================================================

elif page == "📊 Dashboard":
    st.title("📊 Prediction Performance Dashboard")

    timeline_days = st.slider("Timeline window (days)", min_value=7, max_value=90, value=30, step=7)
    pairs_df = load_prediction_actual_pairs(days_back=timeline_days)
    cron_df = load_cron_runs(limit=60)

    eval_metrics = load_evaluation_metrics()
    timing_eval = eval_metrics.get('timing') or {}
    content_eval = eval_metrics.get('content') or {}

    st.subheader("Accuracy Overview")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Timing MAE (h)", f"{timing_eval.get('mae_hours', 0):.2f}" if timing_eval.get('mae_hours') else "N/A")
    col_b.metric("Within 6h", f"{timing_eval.get('within_6h_accuracy', 0):.1%}" if timing_eval.get('within_6h_accuracy') is not None else "N/A")
    col_c.metric("Within 24h", f"{timing_eval.get('within_24h_accuracy', 0):.1%}" if timing_eval.get('within_24h_accuracy') is not None else "N/A")
    col_d.metric("Avg Content Similarity", f"{content_eval.get('bertscore_f1', 0):.2f}" if content_eval.get('bertscore_f1') else "N/A")

    if not pairs_df.empty:
        status_options = ['correct', 'matched', 'late', 'open']
        default_status = [s for s in status_options if s in pairs_df['Status'].unique()]
        selected_statuses = st.multiselect("Filter statuses", options=status_options, default=default_status)
        filtered = pairs_df[pairs_df['Status'].isin(selected_statuses)]

        st.subheader("Timeline: Predictions vs Actual Posts")
        if filtered.empty:
            st.info("No predictions in this window with the selected filters.")
        else:
            timeline_fig = go.Figure()
            color_map = {
                'correct': '#2ecc71',
                'matched': '#f1c40f',
                'late': '#e67e22',
                'open': '#e74c3c'
            }

            # Actual posts trace
            actual_points = filtered.dropna(subset=['Actual Time'])
            if not actual_points.empty:
                timeline_fig.add_trace(go.Scatter(
                    x=actual_points['Actual Time'],
                    y=['Actual'] * len(actual_points),
                    mode='markers',
                    marker=dict(size=12, color='#3498db'),
                    name='Actual Post',
                    hovertext=actual_points['Actual Content'].str[:140]
                ))

            # Prediction trace per status
            for status, group in filtered.groupby('Status'):
                timeline_fig.add_trace(go.Scatter(
                    x=group['Predicted Time'],
                    y=['Prediction'] * len(group),
                    mode='markers',
                    marker=dict(size=12, color=color_map.get(status, '#95a5a6'), symbol='diamond'),
                    name=f"Predicted ({status})",
                    hovertext=group['Predicted Content'].str[:140]
                ))

            # Connect matched predictions to actuals
            for _, row in filtered.iterrows():
                if pd.notnull(row['Actual Time']):
                    timeline_fig.add_trace(go.Scatter(
                        x=[row['Predicted Time'], row['Actual Time']],
                        y=['Prediction', 'Actual'],
                        mode='lines',
                        line=dict(color=color_map.get(row['Status'], '#7f8c8d'), dash='dot'),
                        showlegend=False,
                        opacity=0.4
                    ))

            timeline_fig.update_layout(
                yaxis=dict(title="", tickvals=['Prediction', 'Actual']),
                xaxis_title="Time",
                legend=dict(orientation='h'),
                height=400,
                plot_bgcolor='white'
            )
            st.plotly_chart(timeline_fig, use_container_width=True)

            st.subheader("Timing Error Distribution")
            error_df = filtered.dropna(subset=['Timing Error (h)'])
            if error_df.empty:
                st.info("No timing error data yet.")
            else:
                error_df['Abs Error (h)'] = error_df['Timing Error (h)'].abs()
                fig_error = px.histogram(
                    error_df,
                    x='Abs Error (h)',
                    color='Status',
                    nbins=20,
                    title='Absolute Timing Error',
                    labels={'Abs Error (h)': 'Absolute Hours'}
                )
                st.plotly_chart(fig_error, use_container_width=True)

                st.subheader("Content Similarity Trend")
                sim_df = error_df.dropna(subset=['Similarity'])
                if sim_df.empty:
                    st.info("No similarity scores yet.")
                else:
                    fig_similarity = px.line(
                        sim_df,
                        x='Predicted At',
                        y='Similarity',
                        color='Status',
                        title='Content Similarity Over Time'
                    )
                    st.plotly_chart(fig_similarity, use_container_width=True)

            st.subheader("Prediction vs Actual Table")
            display_cols = [
                'Predicted At',
                'Predicted Time',
                'Actual Time',
                'Timing Error (h)',
                'Similarity',
                'Status',
                'Predicted Content',
                'Actual Content'
            ]
            st.dataframe(
                filtered.sort_values('Predicted At', ascending=False)[display_cols],
                use_container_width=True,
                hide_index=True
            )
    else:
        st.warning("No predictions available yet. Run cron jobs to generate data.")

    latest_prediction_context = load_latest_prediction_context()
    if latest_prediction_context:
        st.subheader("Context Signals at Last Prediction")
        c1, c2, c3 = st.columns(3)
        headlines = latest_prediction_context.get('top_headlines') or []
        if headlines:
            c1.markdown("**Top Headlines**")
            for headline in headlines[:3]:
                c1.write(f"- {headline.get('title')}")
        else:
            c1.write("No headlines captured")

        trends = latest_prediction_context.get('trending_keywords') or []
        if trends:
            c2.markdown("**Trending Topics**")
            c2.write(", ".join(trends[:8]))
        else:
            c2.write("No trend data")

        market_sentiment = latest_prediction_context.get('market_sentiment')
        c3.metric("Market Sentiment", market_sentiment.capitalize() if market_sentiment else "Unknown")
        similarity_metrics = latest_prediction_context.get('content_similarity_metrics')
        if similarity_metrics:
            c3.metric("Style Similarity", f"{similarity_metrics.get('composite_similarity', 0):.2f}")

    st.subheader("Cron Activity")
    if not cron_df.empty:
        last_24h = cron_df[cron_df['Started'] >= (datetime.now() - timedelta(hours=24))]
        grouped = last_24h.groupby('Job') if not last_24h.empty else cron_df.groupby('Job')
        cols = st.columns(len(grouped))
        for col, (job, frame) in zip(cols, grouped):
            success_rate = (frame['Status'] == 'success').mean() * 100 if len(frame) else 0
            total_api = frame['API Calls'].fillna(0).sum()
            col.metric(
                job.title(),
                f"{len(frame)} runs",
                f"{success_rate:.0f}% success | {total_api:.0f} API calls"
            )

        st.dataframe(
            cron_df[['Job', 'Status', 'Started', 'Completed', 'Records', 'API Calls', 'Error']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No cron run history found yet.")


# ============================================================================
# Page: Validation Timeline
# ============================================================================

elif page == "🎯 Validation Timeline":
    st.title("🎯 Prediction Validation Timeline")
    st.markdown("Compare predicted vs actual posts side-by-side with accurate timestamps.")

    # Load validation data (lazy loaded to avoid heavy imports)
    validator = get_validator_lazy()
    
    if validator is None:
        st.error("Could not load validator. Check logs for details.")
        st.stop()

    # Get validation stats
    stats = validator.get_validation_stats()

    if stats.get('total_validated', 0) > 0:
        # Display stats at top
        st.markdown("### 📊 Overall Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Validated", stats['total_validated'])
        col2.metric("Overall Accuracy", f"{stats['overall_accuracy']:.1%}")
        col3.metric("Timing MAE", f"{stats['timing_mae_hours']:.1f}h")
        col4.metric("Within 6h", f"{stats['within_6h_accuracy']:.1%}")

        st.markdown("---")

    # Time range selector
    st.markdown("### 📅 Timeline View")
    days_back = st.slider("Days to show", min_value=7, max_value=90, value=30, step=7)

    # Get timeline data
    with st.spinner("Loading timeline data..."):
        timeline_data = validator.get_timeline_data(days_back=days_back)

    actual_posts = timeline_data['actual_posts']
    predictions = timeline_data['predictions']

    if not actual_posts and not predictions:
        st.warning(f"No data found for the last {days_back} days")
        st.info("Make some predictions and wait for actual posts to compare!")
    else:
        # Create dual timeline visualization using Plotly
        st.markdown("### 🕐 Dual Timeline: Actual vs Predicted")

        # Prepare data for Plotly timeline
        fig = go.Figure()

        # Add actual posts (top timeline)
        for post in actual_posts:
            is_matched = post['matched_prediction_id'] is not None
            color = '#2ecc71' if is_matched else '#95a5a6'
            hover_text = f"<b>Actual Post</b><br>Time: {post['time']}<br>Content: {post['content'][:100]}..."
            if is_matched:
                hover_text += "<br>✓ Matched to prediction"

            fig.add_trace(go.Scatter(
                x=[post['time']],
                y=[1],
                mode='markers',
                marker=dict(
                    size=15,
                    color=color,
                    symbol='circle',
                    line=dict(width=2, color='white')
                ),
                name='Actual',
                text=post['content'][:50] + '...',
                hovertext=hover_text,
                hoverinfo='text',
                showlegend=False
            ))

        # Add predictions (bottom timeline)
        for pred in predictions:
            if pred['was_correct']:
                color = '#27ae60'  # Green for correct
                symbol = 'star'
            elif pred['actual_post_id']:
                color = '#f39c12'  # Orange for matched but not correct
                symbol = 'diamond'
            else:
                color = '#e74c3c'  # Red for unmatched
                symbol = 'x'

            hover_text = f"<b>Predicted Post</b><br>Predicted Time: {pred['time']}<br>Content: {pred['content'][:100]}..."
            if pred['actual_post_id']:
                hover_text += f"<br>Actual Time: {pred['actual_time']}<br>Error: {pred['timing_error_hours']:.1f}h"
                hover_text += f"<br>Similarity: {pred['similarity']:.3f}" if pred['similarity'] else ""
                hover_text += "<br>✓ Correct!" if pred['was_correct'] else "<br>✗ Not correct"
            else:
                hover_text += "<br>⏳ Not yet matched"

            fig.add_trace(go.Scatter(
                x=[pred['time']],
                y=[0],
                mode='markers',
                marker=dict(
                    size=15,
                    color=color,
                    symbol=symbol,
                    line=dict(width=2, color='white')
                ),
                name='Predicted',
                text=pred['content'][:50] + '...',
                hovertext=hover_text,
                hoverinfo='text',
                showlegend=False
            ))

        # Add connecting lines for matched pairs
        for pred in predictions:
            if pred['actual_post_id']:
                # Find the corresponding actual post
                actual_post = next((p for p in actual_posts if p['id'] == pred['actual_post_id']), None)
                if actual_post:
                    # Draw line connecting prediction to actual
                    fig.add_trace(go.Scatter(
                        x=[pred['time'], actual_post['time']],
                        y=[0, 1],
                        mode='lines',
                        line=dict(
                            color='rgba(52, 152, 219, 0.3)',
                            width=2,
                            dash='dot'
                        ),
                        hoverinfo='skip',
                        showlegend=False
                    ))

        # Update layout
        fig.update_layout(
            height=400,
            xaxis=dict(
                title="Time",
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title="",
                ticktext=['Predicted', 'Actual'],
                tickvals=[0, 1],
                range=[-0.3, 1.3],
                showgrid=False
            ),
            plot_bgcolor='white',
            hovermode='closest',
            margin=dict(l=100, r=50, t=50, b=50)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Legend
        st.markdown("""
        **Legend:**
        - 🟢 Green circles (top) = Actual posts (matched to prediction)
        - ⚪ Gray circles (top) = Actual posts (no prediction)
        - ⭐ Green stars (bottom) = Correct predictions
        - 🔶 Orange diamonds (bottom) = Matched but not fully correct
        - ❌ Red X (bottom) = Unmatched predictions
        - Dotted lines connect matched prediction-actual pairs
        """)

        st.markdown("---")

        # Detailed comparison table
        st.markdown("### 📋 Detailed Comparisons")

        # Filter to show only matched predictions
        matched_predictions = [p for p in predictions if p['actual_post_id']]

        if matched_predictions:
            comparison_data = []
            for pred in matched_predictions:
                actual_post = next((p for p in actual_posts if p['id'] == pred['actual_post_id']), None)
                if actual_post:
                    comparison_data.append({
                        'Predicted Time': pred['time'],
                        'Actual Time': actual_post['time'],
                        'Error (hours)': f"{pred['timing_error_hours']:.1f}h" if pred['timing_error_hours'] else 'N/A',
                        'Similarity': f"{pred['similarity']:.3f}" if pred['similarity'] else 'N/A',
                        'Correct': '✓' if pred['was_correct'] else '✗',
                        'Predicted Content': pred['content'][:80] + '...',
                        'Actual Content': actual_post['content'][:80] + '...'
                    })

            if comparison_data:
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No matched predictions yet. Predictions will be matched to actual posts automatically.")

        # Individual post comparisons (expandable)
        st.markdown("### 🔍 Side-by-Side Comparison")
        if matched_predictions:
            for pred in matched_predictions[:10]:  # Show first 10
                actual_post = next((p for p in actual_posts if p['id'] == pred['actual_post_id']), None)
                if actual_post:
                    with st.expander(
                        f"{'✓' if pred['was_correct'] else '✗'} Prediction at {pred['time'].strftime('%Y-%m-%d %H:%M')} "
                        f"(Error: {pred['timing_error_hours']:.1f}h)"
                    ):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**🔮 Predicted**")
                            st.markdown(f"**Time:** {pred['time'].strftime('%Y-%m-%d %H:%M:%S')}")
                            st.markdown(f"**Confidence:** Timing: {pred['timing_confidence']:.1%}, Content: {pred['content_confidence']:.1%}")
                            st.markdown("**Content:**")
                            st.info(pred['content'])

                        with col2:
                            st.markdown("**✅ Actual**")
                            st.markdown(f"**Time:** {actual_post['time'].strftime('%Y-%m-%d %H:%M:%S')}")
                            st.markdown(f"**Similarity:** {pred['similarity']:.3f}" if pred['similarity'] else "N/A")
                            st.markdown("**Content:**")
                            st.success(actual_post['content'])

                        # Metrics
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        metric_col1.metric("Timing Error", f"{pred['timing_error_hours']:.1f}h")
                        metric_col2.metric("Content Similarity", f"{pred['similarity']:.3f}" if pred['similarity'] else "N/A")
                        metric_col3.metric("Result", "✓ Correct" if pred['was_correct'] else "✗ Incorrect")
        else:
            st.info("Make some predictions and wait for posts to see detailed comparisons!")


# ============================================================================
# Page: Recent Posts
# ============================================================================

elif page == "📝 Recent Posts":
    st.title("📝 Recent Posts")
    
    # Show database connection info
    db_type, db_status, is_production = get_database_info()
    if not is_production:
        st.warning(f"⚠️ {db_status} - Make sure DATABASE_URL is configured in Streamlit secrets")
    
    posts_df = load_recent_posts(limit=50)
    
    if not posts_df.empty:
        st.markdown(f"**Showing {len(posts_df)} most recent posts**")
        
        # Display each post
        for idx, row in posts_df.iterrows():
            try:
                date_str = row['Date'].strftime('%Y-%m-%d %H:%M') if row['Date'] else "Unknown date"
            except:
                date_str = str(row['Date'])
            with st.expander(f"📅 {date_str}"):
                st.markdown(f"**Content:** {row['Content']}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("💬 Replies", row['Replies'])
                col2.metric("🔄 Reblogs", row['Reblogs'])
                col3.metric("❤️ Favorites", row['Favorites'])
    
    else:
        st.warning("No posts in database yet.")
        
        # Show debugging info
        with st.expander("🔍 Debug Info"):
            st.markdown(f"**Database:** {db_type}")
            st.markdown(f"**Status:** {db_status}")
            
            # Try to get count directly
            try:
                session = get_session()
                count = session.query(Post).count()
                session.close()
                st.markdown(f"**Post count in database:** {count}")
            except Exception as e:
                st.error(f"Could not query database: {e}")
        
        st.info("Run the data collector to fetch historical posts:\n```bash\npython src/data/collector.py\n```")


# ============================================================================
# Page: Real-time Context
# ============================================================================

elif page == "🌐 Real-time Context":
    st.title("🌐 Real-time Context Integration")
    st.markdown("View current real-time context data used for predictions.")

    # Initialize context gatherer (lazy loaded)
    context_gatherer = get_context_gatherer_lazy()
    
    if context_gatherer is None:
        st.error("Could not load context gatherer. Check logs for details.")
        st.stop()

    # Add refresh button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        if st.button("🔄 Refresh Context", use_container_width=True):
            context_gatherer.cache.clear()
            st.rerun()

    # Fetch current context
    with st.spinner("Fetching real-time context..."):
        try:
            context = context_gatherer.get_full_context(save_to_db=True)

            # Display context summary
            st.markdown("### 📝 Summary")
            summary = context_gatherer.get_context_summary(context)
            st.info(summary)

            # Display metadata
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Completeness", f"{context.get('completeness_score', 0):.0%}")
            col2.metric("Freshness", f"{context.get('freshness_score', 0):.0%}")
            col3.metric("Fetch Time", f"{context.get('fetch_duration_seconds', 0):.2f}s")
            col4.metric("Data Sources", len(context.get('data_sources', [])))

            st.markdown("---")

            # News Headlines Section
            st.markdown("### 📰 News Headlines")
            top_headlines = context.get('top_headlines', [])

            if top_headlines:
                for i, headline in enumerate(top_headlines[:5], 1):
                    with st.expander(f"**{i}. {headline['title']}**"):
                        st.markdown(f"**Source:** {headline['source']}")
                        if headline.get('url'):
                            st.markdown(f"**Link:** [{headline['url']}]({headline['url']})")
                        if headline.get('published_at'):
                            st.markdown(f"**Published:** {headline['published_at']}")
            else:
                st.warning("No news headlines available")

            # Political News Section
            st.markdown("### 🏛️ Political News")
            political_news = context.get('political_news', [])

            if political_news:
                for i, news in enumerate(political_news[:3], 1):
                    with st.expander(f"**{i}. {news['title']}**"):
                        st.markdown(f"**Source:** {news['source']}")
                        if news.get('url'):
                            st.markdown(f"**Link:** [{news['url']}]({news['url']})")
            else:
                st.info("No political news available")

            st.markdown("---")

            # Trending Topics Section
            st.markdown("### 📈 Trending Topics")
            trending_keywords = context.get('trending_keywords', [])

            if trending_keywords:
                # Display as tags
                cols = st.columns(5)
                for i, keyword in enumerate(trending_keywords[:10]):
                    col_idx = i % 5
                    cols[col_idx].markdown(f"`{keyword}`")
            else:
                st.info("No trending topics available")

            # Political trends
            trend_categories = context.get('trend_categories', {})
            if trend_categories and 'political' in trend_categories:
                st.markdown("#### Political Interest Levels")
                political_trends = trend_categories['political']

                # Create bar chart
                if political_trends:
                    df_trends = pd.DataFrame([
                        {'Topic': topic, 'Interest': interest}
                        for topic, interest in political_trends.items()
                    ])
                    fig = px.bar(
                        df_trends,
                        x='Topic',
                        y='Interest',
                        title='Google Trends: Political Topics',
                        color='Interest',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Market Data Section
            st.markdown("### 📊 Market Data")

            sp500_value = context.get('sp500_value')
            dow_value = context.get('dow_value')
            market_sentiment = context.get('market_sentiment', 'neutral')

            if sp500_value or dow_value:
                col1, col2, col3 = st.columns(3)

                with col1:
                    if sp500_value:
                        sp_change = context.get('sp500_change_pct', 0)
                        st.metric(
                            "S&P 500",
                            f"${sp500_value:,.2f}",
                            f"{sp_change:+.2f}%",
                            delta_color="normal"
                        )

                with col2:
                    if dow_value:
                        dow_change = context.get('dow_change_pct', 0)
                        st.metric(
                            "Dow Jones",
                            f"${dow_value:,.2f}",
                            f"{dow_change:+.2f}%",
                            delta_color="normal"
                        )

                with col3:
                    sentiment_emoji = {
                        'bullish': '📈',
                        'bearish': '📉',
                        'neutral': '➡️'
                    }
                    st.metric(
                        "Market Sentiment",
                        f"{sentiment_emoji.get(market_sentiment, '➡️')} {market_sentiment.capitalize()}"
                    )
            else:
                st.info("Market data not available")

            st.markdown("---")

            # Data Sources & Errors
            st.markdown("### 🔍 Metadata")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Data Sources Used:**")
                data_sources = context.get('data_sources', [])
                if data_sources:
                    for source in data_sources:
                        st.markdown(f"- ✅ {source}")
                else:
                    st.markdown("- ⚠️ No data sources available")

            with col2:
                st.markdown("**Fetch Errors:**")
                fetch_errors = context.get('fetch_errors', [])
                if fetch_errors:
                    for error in fetch_errors:
                        st.markdown(f"- ❌ {error['source']}: {error['error'][:50]}...")
                else:
                    st.markdown("- ✅ No errors")

            # Context History
            st.markdown("---")
            st.markdown("### 📜 Context History")

            session = get_session()
            snapshots = session.query(ContextSnapshot)\
                .order_by(ContextSnapshot.captured_at.desc())\
                .limit(10)\
                .all()
            session.close()

            if snapshots:
                history_data = []
                for snapshot in snapshots:
                    history_data.append({
                        'Time': snapshot.captured_at.strftime('%Y-%m-%d %H:%M'),
                        'Completeness': f"{snapshot.completeness_score:.0%}" if snapshot.completeness_score else 'N/A',
                        'Freshness': f"{snapshot.freshness_score:.0%}" if snapshot.freshness_score else 'N/A',
                        'Market': snapshot.market_sentiment or 'N/A',
                        'S&P Change': f"{snapshot.sp500_change_pct:+.2f}%" if snapshot.sp500_change_pct else 'N/A',
                        'Used in': snapshot.used_in_predictions or 0
                    })

                df_history = pd.DataFrame(history_data)
                st.dataframe(df_history, use_container_width=True, hide_index=True)
            else:
                st.info("No context history available yet")

        except Exception as e:
            st.error(f"Error fetching context: {str(e)}")
            st.info("Make sure API keys are set: NEWS_API_KEY (optional)")


# ============================================================================
# Page: About
# ============================================================================

elif page == "⚙️ About":
    st.title("⚙️ About This System")
    
    # Database Status Section
    st.markdown("### 🗄️ Database Connection")
    db_type, db_status, is_production = get_database_info()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Database Type", db_type)
    with col2:
        if is_production:
            st.success(db_status)
        else:
            st.warning(db_status)
    
    if not is_production:
        st.error("""
        ⚠️ **Not Connected to Production Database!**
        
        The app is showing data from a local SQLite database, not the production Neon database.
        
        **To fix this:**
        1. Go to Streamlit Cloud → Your App → Settings → Secrets
        2. Add: `DATABASE_URL = "postgresql://user:pass@ep-xxx.neon.tech/db?sslmode=require"`
        3. Save and restart the app
        
        See `STREAMLIT_SETUP.md` for detailed instructions.
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ## Trump Truth Social Post Prediction System
    
    This is a real-time ML pipeline that predicts **when** Trump will post on Truth Social 
    and **what** content he'll post.
    
    ### Architecture
    
    **Data Collection:**
    - Multiple sources: Apify, ScrapeCreators, GitHub Archive
    - Polls every 5 minutes for new posts
    - Stores in PostgreSQL (Neon) database
    
    **Timing Prediction:**
    - Model: Facebook Prophet (time series forecasting)
    - Features: Hour, day of week, time since last post
    - Accuracy: Typically within 8-12 hours
    
    **Content Generation:**
    - Model: Claude API with few-shot prompting
    - Context: Recent news, time of day, posting patterns
    - Quality: ~60-70% style similarity (BERTScore)
    
    ### Technology Stack
    
    - **Backend:** Python, FastAPI, SQLAlchemy
    - **ML:** Prophet, Anthropic Claude API
    - **Frontend:** Streamlit
    - **Database:** Neon PostgreSQL (production)
    - **Deployment:** Render.com, Streamlit Cloud
    
    ### Cost
    
    - Total: **$15-25/month**
    - Claude API: ~$10-20
    - Data collection: $5 (Apify free tier)
    - Hosting: $0 (free tiers)
    
    ### Future Improvements
    
    1. **Neural Temporal Point Process** for timing (better accuracy)
    2. **Fine-tuned Phi-3** for content (higher quality)
    3. **RAG system** with real-time news
    4. **Automated retraining** pipeline
    5. **Model monitoring** and drift detection
    
    ### Repository
    
    [View on GitHub](#) (link to your repo)
    
    ### License
    
    Educational/portfolio project. Use responsibly.
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit and Claude")


# ============================================================================
# Footer
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("Built with [Streamlit](https://streamlit.io)")
st.sidebar.markdown("Powered by [Claude](https://anthropic.com)")
