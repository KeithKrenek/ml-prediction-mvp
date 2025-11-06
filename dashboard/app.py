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

# For Streamlit Cloud: Load secrets as environment variables
if hasattr(st, 'secrets'):
    for key, value in st.secrets.items():
        os.environ[key] = str(value)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.predictor import TrumpPostPredictor
from src.data.database import get_session, Post, Prediction, ContextSnapshot
from src.validation.validator import PredictionValidator
from src.context.context_gatherer import RealTimeContextGatherer


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
    ["🎯 Make Prediction", "📊 Dashboard", "🎯 Validation Timeline", "📝 Recent Posts", "🌐 Real-time Context", "⚙️ About"]
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
# Page: Validation Timeline
# ============================================================================

elif page == "🎯 Validation Timeline":
    st.title("🎯 Prediction Validation Timeline")
    st.markdown("Compare predicted vs actual posts side-by-side with accurate timestamps.")

    # Load validation data
    validator = PredictionValidator()

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
# Page: Real-time Context
# ============================================================================

elif page == "🌐 Real-time Context":
    st.title("🌐 Real-time Context Integration")
    st.markdown("View current real-time context data used for predictions.")

    # Initialize context gatherer
    @st.cache_resource
    def get_context_gatherer():
        return RealTimeContextGatherer()

    context_gatherer = get_context_gatherer()

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
