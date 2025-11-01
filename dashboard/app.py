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

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.predictor import TrumpPostPredictor
from src.data.database import get_session, Post, Prediction


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
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Helper Functions
# ============================================================================

@st.cache_resource
def get_predictor():
    """Initialize and cache the predictor"""
    return TrumpPostPredictor()


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
            data.append({
                'Date': post.created_at,
                'Content': post.content[:100] + '...' if len(post.content) > 100 else post.content,
                'Replies': post.replies_count,
                'Reblogs': post.reblogs_count,
                'Favorites': post.favourites_count,
            })
        
        return pd.DataFrame(data)
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
            data.append({
                'Predicted At': pred.predicted_at,
                'Predicted Time': pred.predicted_time,
                'Content Preview': pred.predicted_content[:50] + '...',
                'Timing Confidence': pred.predicted_time_confidence,
                'Content Confidence': pred.predicted_content_confidence,
                'Actual Time': pred.actual_time,
                'Accuracy': pred.bertscore_f1
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
    ["🎯 Make Prediction", "📊 Dashboard", "📝 Recent Posts", "⚙️ About"]
)

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
            with st.spinner("Training models and generating prediction..."):
                try:
                    predictor = get_predictor()
                    
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
    st.title("📊 Analytics Dashboard")
    
    # Load data
    posts_df = load_recent_posts(limit=100)
    predictions_df = load_predictions()
    
    if not posts_df.empty:
        # Posting frequency over time
        st.markdown("### Posting Frequency")
        
        # Resample by day
        daily_posts = posts_df.set_index('Date').resample('D').size().reset_index(name='Posts')
        
        fig = px.line(
            daily_posts,
            x='Date',
            y='Posts',
            title='Posts Per Day (Last 100 Posts)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Posting by hour of day
        st.markdown("### Posting Patterns by Hour")
        
        posts_df['Hour'] = pd.to_datetime(posts_df['Date']).dt.hour
        hourly = posts_df.groupby('Hour').size().reset_index(name='Count')
        
        fig = px.bar(
            hourly,
            x='Hour',
            y='Count',
            title='Posts by Hour of Day',
            labels={'Hour': 'Hour of Day (EST)', 'Count': 'Number of Posts'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Engagement metrics
        st.markdown("### Engagement Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric(
            "Avg Replies",
            f"{posts_df['Replies'].mean():.0f}",
            f"{posts_df['Replies'].std():.0f} std"
        )
        col2.metric(
            "Avg Reblogs",
            f"{posts_df['Reblogs'].mean():.0f}",
            f"{posts_df['Reblogs'].std():.0f} std"
        )
        col3.metric(
            "Avg Favorites",
            f"{posts_df['Favorites'].mean():.0f}",
            f"{posts_df['Favorites'].std():.0f} std"
        )
    
    else:
        st.warning("No posts in database yet. Run data collection first.")
    
    # Predictions performance
    if not predictions_df.empty:
        st.markdown("### Prediction History")
        st.dataframe(
            predictions_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================================
# Page: Recent Posts
# ============================================================================

elif page == "📝 Recent Posts":
    st.title("📝 Recent Posts")
    
    posts_df = load_recent_posts(limit=50)
    
    if not posts_df.empty:
        st.markdown(f"**Showing {len(posts_df)} most recent posts**")
        
        # Display each post
        for idx, row in posts_df.iterrows():
            with st.expander(f"📅 {row['Date'].strftime('%Y-%m-%d %H:%M')}"):
                st.markdown(f"**Content:** {row['Content']}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("💬 Replies", row['Replies'])
                col2.metric("🔄 Reblogs", row['Reblogs'])
                col3.metric("❤️ Favorites", row['Favorites'])
    
    else:
        st.warning("No posts in database yet.")
        st.info("Run the data collector to fetch historical posts:\n```bash\npython src/data/collector.py\n```")


# ============================================================================
# Page: About
# ============================================================================

elif page == "⚙️ About":
    st.title("⚙️ About This System")
    
    st.markdown("""
    ## Trump Truth Social Post Prediction System
    
    This is a real-time ML pipeline that predicts **when** Trump will post on Truth Social 
    and **what** content he'll post.
    
    ### Architecture
    
    **Data Collection:**
    - Multiple sources: Apify, ScrapeCreators, GitHub Archive
    - Polls every 5 minutes for new posts
    - Stores in SQLite database
    
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
    - **Database:** SQLite (MVP) → PostgreSQL (production)
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
