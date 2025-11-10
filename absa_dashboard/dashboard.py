"""
ABSA Interactive Dashboard
===========================
Multi-page Streamlit dashboard for ABSA analysis with product metadata integration.

Pages:
1. Sentiment Overview
2. Product Explorer
3. Aspect Analysis
4. Product Deep Dive
5. Alerts & Anomalies
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from config import *
from utils import *

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state=SIDEBAR_STATE
)

# ==================== DATA LOADING ====================
@st.cache_data
def load_data():
    """Load preprocessed data."""
    data_path = Path(DATA_DIR) / PREPROCESSED_DATA_FILE
    if not data_path.exists():
        st.error(f"❌ Data file not found: {data_path}")
        st.info("Please run the preprocessing pipeline first (preprocess_data.py)")
        st.stop()
    
    df = pd.read_parquet(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = pd.to_datetime(df['date'])
    return df


# Load data
df = load_data()

# ==================== SIDEBAR ====================
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📊 Sentiment Overview", "🔍 Product Explorer", "🏷️ Aspect Analysis", 
     "📈 Product Deep Dive", "🚨 Alerts & Anomalies"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Global Filters")

# Date range filter
min_date = df['date'].min()
max_date = df['date'].max()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Category filter (if available)
if 'main_category' in df.columns and df['main_category'].notna().any():
    categories = ['All'] + sorted(df['main_category'].dropna().unique().tolist())
    selected_category = st.sidebar.selectbox("Product Category", categories)
else:
    selected_category = 'All'

# Confidence threshold
confidence_threshold = st.sidebar.slider(
    "Minimum Confidence",
    min_value=0.0,
    max_value=1.0,
    value=CONFIDENCE_THRESHOLD,
    step=0.05
)

# Apply filters
df_filtered = df.copy()
if len(date_range) == 2:
    df_filtered = df_filtered[
        (df_filtered['date'] >= pd.to_datetime(date_range[0])) &
        (df_filtered['date'] <= pd.to_datetime(date_range[1]))
    ]
df_filtered = df_filtered[df_filtered['confidence'] >= confidence_threshold]
if selected_category != 'All':
    df_filtered = df_filtered[df_filtered['main_category'] == selected_category]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Filtered Data:** {len(df_filtered):,} reviews")


# ==================== PAGE 1: SENTIMENT OVERVIEW ====================
if page == "📊 Sentiment Overview":
    st.title("📊 Sentiment Overview Dashboard")
    st.markdown("High-level sentiment metrics and trends across all products.")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", f"{len(df_filtered):,}")
        st.metric("Unique Products", f"{df_filtered['parent_asin'].nunique():,}")
    
    with col2:
        st.metric("Unique Aspects", f"{df_filtered['aspect_term_normalized'].nunique():,}")
        avg_conf = df_filtered['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_conf:.2%}")
    
    with col3:
        pct_positive = (df_filtered['sentiment'] == 'Positive').sum() / len(df_filtered) * 100
        pct_negative = (df_filtered['sentiment'] == 'Negative').sum() / len(df_filtered) * 100
        st.metric("Positive Reviews", f"{pct_positive:.1f}%", delta=None)
        st.metric("Negative Reviews", f"{pct_negative:.1f}%", delta=None)
    
    with col4:
        if 'has_metadata' in df_filtered.columns:
            metadata_coverage = df_filtered['has_metadata'].sum() / len(df_filtered) * 100
            st.metric("Metadata Coverage", f"{metadata_coverage:.1f}%")
        pct_neutral = (df_filtered['sentiment'] == 'Neutral').sum() / len(df_filtered) * 100
        st.metric("Neutral Reviews", f"{pct_neutral:.1f}%")
    
    st.markdown("---")
    
    # Sentiment Distribution
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df_filtered['sentiment'].value_counts()
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map=COLOR_PALETTE_SENTIMENT,
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Sentiment Trend Over Time")
        
        # Aggregation selector
        time_agg = st.radio("Time Aggregation", ["Daily", "Weekly", "Monthly"], horizontal=True)
        
        # Aggregate data
        if time_agg == "Daily":
            time_col = 'date'
        elif time_agg == "Weekly":
            df_filtered['time_period'] = df_filtered['timestamp'].dt.to_period('W').dt.to_timestamp()
            time_col = 'time_period'
        else:  # Monthly
            df_filtered['time_period'] = df_filtered['timestamp'].dt.to_period('M').dt.to_timestamp()
            time_col = 'time_period'
        
        sentiment_trend = df_filtered.groupby([time_col, 'sentiment']).size().reset_index(name='count')
        
        fig_line = px.line(
            sentiment_trend,
            x=time_col,
            y='count',
            color='sentiment',
            color_discrete_map=COLOR_PALETTE_SENTIMENT,
            title=f"Sentiment Counts Over Time ({time_agg})"
        )
        fig_line.update_xaxes(title="Date")
        fig_line.update_yaxes(title="Number of Reviews")
        st.plotly_chart(fig_line, use_container_width=True)
    
    st.markdown("---")
    
    # Sentiment Score Trend
    st.subheader("Average Sentiment Score Trend")
    st.caption("Sentiment score: Positive confidence - Negative confidence")
    
    if time_agg == "Daily":
        score_trend = df_filtered.groupby('date')['sentiment_score'].mean().reset_index()
        x_col = 'date'
    elif time_agg == "Weekly":
        score_trend = df_filtered.groupby('time_period')['sentiment_score'].mean().reset_index()
        x_col = 'time_period'
    else:
        score_trend = df_filtered.groupby('time_period')['sentiment_score'].mean().reset_index()
        x_col = 'time_period'
    
    fig_score = go.Figure()
    fig_score.add_trace(go.Scatter(
        x=score_trend[x_col],
        y=score_trend['sentiment_score'],
        mode='lines+markers',
        name='Sentiment Score',
        line=dict(color='#3498db', width=2),
        fill='tozeroy'
    ))
    fig_score.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_score.update_layout(
        xaxis_title="Date",
        yaxis_title="Average Sentiment Score",
        hovermode='x unified'
    )
    st.plotly_chart(fig_score, use_container_width=True)


# ==================== PAGE 2: PRODUCT EXPLORER ====================
elif page == "🔍 Product Explorer":
    st.title("🔍 Product Explorer")
    st.markdown("Search and filter products with detailed sentiment analysis.")
    
    # Search and filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_term = st.text_input("🔎 Search by Product Title", "")
    
    with col2:
        if 'average_rating' in df_filtered.columns:
            rating_range = st.slider(
                "Product Rating Range",
                min_value=1.0,
                max_value=5.0,
                value=(1.0, 5.0),
                step=0.1
            )
        else:
            rating_range = (1.0, 5.0)
    
    with col3:
        min_reviews = st.number_input(
            "Minimum Reviews",
            min_value=1,
            value=MIN_REVIEWS_FOR_ANALYSIS,
            step=1
        )
    
    # Filter products
    product_stats = df_filtered.groupby('parent_asin').agg({
        'review_id': 'count',
        'sentiment_score': 'mean',
        'is_positive': 'sum',
        'is_negative': 'sum',
        'is_neutral': 'sum'
    }).reset_index()
    product_stats.columns = ['parent_asin', 'review_count', 'avg_sentiment_score', 'positive_count', 'negative_count', 'neutral_count']
    
    # Merge with product metadata
    if 'title' in df_filtered.columns:
        product_info = df_filtered.groupby('parent_asin').first()[['title', 'main_category', 'average_rating']].reset_index()
        product_stats = product_stats.merge(product_info, on='parent_asin', how='left')
    
    # Apply filters
    product_stats = product_stats[product_stats['review_count'] >= min_reviews]
    if 'average_rating' in product_stats.columns:
        product_stats = product_stats[
            (product_stats['average_rating'] >= rating_range[0]) &
            (product_stats['average_rating'] <= rating_range[1])
        ]
    if search_term and 'title' in product_stats.columns:
        product_stats = product_stats[product_stats['title'].str.contains(search_term, case=False, na=False)]
    
    st.markdown(f"**Found {len(product_stats)} products**")
    
    # Display products as cards
    for idx, row in product_stats.nlargest(20, 'review_count').iterrows():
        with st.expander(f"📦 {row.get('title', row['parent_asin'])}"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Reviews", f"{int(row['review_count'])}")
                if 'average_rating' in row:
                    st.metric("Product Rating", f"{row['average_rating']:.1f}⭐")
            
            with col2:
                st.metric("Sentiment Score", f"{row['avg_sentiment_score']:.2f}")
                if 'main_category' in row:
                    st.write(f"**Category:** {row['main_category']}")
            
            with col3:
                st.metric("✅ Positive", int(row['positive_count']), delta=None)
                st.metric("❌ Negative", int(row['negative_count']), delta=None)
            
            with col4:
                st.metric("➖ Neutral", int(row['neutral_count']), delta=None)
            
            # Top aspects
            product_aspects = df_filtered[df_filtered['parent_asin'] == row['parent_asin']]
            top_positive = product_aspects[product_aspects['sentiment'] == 'Positive']['aspect_term_normalized'].value_counts().head(3)
            top_negative = product_aspects[product_aspects['sentiment'] == 'Negative']['aspect_term_normalized'].value_counts().head(3)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Top Positive Aspects:**")
                for aspect, count in top_positive.items():
                    st.write(f"  - {aspect} ({count})")
            
            with col_b:
                st.write("**Top Negative Aspects:**")
                for aspect, count in top_negative.items():
                    st.write(f"  - {aspect} ({count})")


# ==================== PAGE 3: ASPECT ANALYSIS ====================
elif page == "🏷️ Aspect Analysis":
    st.title("🏷️ Aspect Analysis")
    st.markdown("Deep dive into aspect mentions and sentiment patterns.")
    
    # Top aspects
    st.subheader(f"Top {TOP_N_ASPECTS} Aspects by Mention Count")
    
    aspect_counts = df_filtered['aspect_term_normalized'].value_counts().head(TOP_N_ASPECTS)
    fig_bar = px.bar(
        x=aspect_counts.values,
        y=aspect_counts.index,
        orientation='h',
        labels={'x': 'Mention Count', 'y': 'Aspect'},
        color=aspect_counts.values,
        color_continuous_scale='Blues'
    )
    fig_bar.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    
    # Aspect sentiment heatmap
    st.subheader("Aspect Sentiment Heatmap")
    
    # Get top aspects and products
    top_aspects_list = df_filtered['aspect_term_normalized'].value_counts().head(15).index.tolist()
    top_products_list = df_filtered['parent_asin'].value_counts().head(10).index.tolist()
    
    # Filter and pivot
    heatmap_data = df_filtered[
        (df_filtered['aspect_term_normalized'].isin(top_aspects_list)) &
        (df_filtered['parent_asin'].isin(top_products_list))
    ]
    
    heatmap_pivot = heatmap_data.groupby(['aspect_term_normalized', 'parent_asin'])['sentiment_score'].mean().reset_index()
    heatmap_matrix = heatmap_pivot.pivot(index='aspect_term_normalized', columns='parent_asin', values='sentiment_score')
    
    fig_heatmap = px.imshow(
        heatmap_matrix,
        labels=dict(x="Product", y="Aspect", color="Sentiment Score"),
        x=heatmap_matrix.columns,
        y=heatmap_matrix.index,
        color_continuous_scale='RdYlGn',
        aspect="auto"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown("---")
    
    # Aspect evolution over time
    st.subheader("Aspect Evolution Over Time")
    
    selected_aspects = st.multiselect(
        "Select Aspects to Track",
        options=aspect_counts.head(10).index.tolist(),
        default=aspect_counts.head(5).index.tolist()
    )
    
    if selected_aspects:
        aspect_evolution = df_filtered[df_filtered['aspect_term_normalized'].isin(selected_aspects)]
        aspect_evolution = aspect_evolution.groupby([pd.Grouper(key='date', freq='W'), 'aspect_term_normalized']).size().reset_index(name='count')
        
        fig_evolution = px.line(
            aspect_evolution,
            x='date',
            y='count',
            color='aspect_term_normalized',
            title="Aspect Mentions Over Time (Weekly)",
            labels={'date': 'Date', 'count': 'Mention Count', 'aspect_term_normalized': 'Aspect'}
        )
        st.plotly_chart(fig_evolution, use_container_width=True)


# ==================== PAGE 4: PRODUCT DEEP DIVE ====================
elif page == "📈 Product Deep Dive":
    st.title("📈 Product Deep Dive")
    st.markdown("Detailed analysis for a specific product.")
    
    # Product selector - always sort products by review count (most reviewed first)
    product_counts = df_filtered['parent_asin'].value_counts()
    ordered_asins = product_counts.index.tolist()

    if 'title' in df_filtered.columns:
        # Build mapping of ASIN -> title (use first occurrence)
        title_map = df_filtered.groupby('parent_asin')['title'].first().to_dict()
        # Ensure options are ordered by review count
        ordered_options = ordered_asins
        selected_product = st.selectbox(
            "Select Product",
            options=ordered_options,
            format_func=lambda x: f"{title_map.get(x, x)} ({x})"
        )
    else:
        # No titles available, just show ASINs ordered by review count
        selected_product = st.selectbox("Select Product (ASIN)", ordered_asins)
    
    # Filter for selected product
    product_data = df_filtered[df_filtered['parent_asin'] == selected_product]
    
    if len(product_data) == 0:
        st.warning("No data available for this product.")
    else:
        # Product metadata card
        st.subheader("Product Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reviews", len(product_data))
            st.metric("ASIN", selected_product)
        
        with col2:
            avg_sentiment = product_data['sentiment_score'].mean()
            st.metric("Avg Sentiment Score", f"{avg_sentiment:.2f}")
            if 'average_rating' in product_data.columns:
                st.metric("Product Rating", f"{product_data['average_rating'].iloc[0]:.1f}⭐")
        
        with col3:
            pct_pos = (product_data['sentiment'] == 'Positive').sum() / len(product_data) * 100
            pct_neg = (product_data['sentiment'] == 'Negative').sum() / len(product_data) * 100
            st.metric("Positive", f"{pct_pos:.1f}%")
            st.metric("Negative", f"{pct_neg:.1f}%")
        
        with col4:
            unique_aspects = product_data['aspect_term_normalized'].nunique()
            st.metric("Unique Aspects", unique_aspects)
            if 'main_category' in product_data.columns:
                st.write(f"**Category:** {product_data['main_category'].iloc[0]}")
        
        st.markdown("---")
        
        # Aspect distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Aspect Distribution")
            aspect_dist = product_data['aspect_term_normalized'].value_counts().head(10)
            fig_pie = px.pie(
                values=aspect_dist.values,
                names=aspect_dist.index,
                title="Top 10 Aspects"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("Sentiment by Aspect")
            aspect_sentiment = product_data.groupby(['aspect_term_normalized', 'sentiment']).size().reset_index(name='count')
            aspect_sentiment = aspect_sentiment[aspect_sentiment['aspect_term_normalized'].isin(aspect_dist.index)]
            
            fig_bar = px.bar(
                aspect_sentiment,
                x='aspect_term_normalized',
                y='count',
                color='sentiment',
                color_discrete_map=COLOR_PALETTE_SENTIMENT,
                title="Sentiment Distribution by Top Aspects",
                barmode='group'
            )
            fig_bar.update_xaxes(tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("---")
        
        # Sentiment trajectory
        st.subheader("Sentiment Trajectory")
        trajectory = product_data.groupby('date')['sentiment_score'].mean().reset_index()
        
        fig_trajectory = go.Figure()
        fig_trajectory.add_trace(go.Scatter(
            x=trajectory['date'],
            y=trajectory['sentiment_score'],
            mode='lines+markers',
            name='Sentiment Score',
            line=dict(color='#3498db', width=2)
        ))
        fig_trajectory.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_trajectory.update_layout(
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            hovermode='x unified'
        )
        st.plotly_chart(fig_trajectory, use_container_width=True)


# ==================== PAGE 5: ALERTS & ANOMALIES ====================
elif page == "🚨 Alerts & Anomalies":
    st.title("🚨 Alerts & Anomalies")
    st.markdown("Real-time alerts for sentiment shifts and emerging issues.")
    
    # Calculate rolling sentiment
    st.subheader("Sentiment Drop Alerts")
    st.caption(f"Products with >{ SENTIMENT_DROP_THRESHOLD*100:.0f}% sentiment drop in past {ROLLING_WINDOW_DAYS} days")
    
    alerts = []
    
    for product in df_filtered['parent_asin'].unique():
        product_data = df_filtered[df_filtered['parent_asin'] == product].sort_values('date')
        
        if len(product_data) < MIN_REVIEWS_FOR_ALERT:
            continue
        
        # Calculate rolling 7-day sentiment
        product_data = product_data.set_index('date')
        rolling_sentiment = product_data['sentiment_score'].rolling(f'{ROLLING_WINDOW_DAYS}D', min_periods=1).mean()
        
        if len(rolling_sentiment) < 2:
            continue
        
        # Compare recent vs previous period
        recent_sentiment = rolling_sentiment.iloc[-1]
        previous_sentiment = rolling_sentiment.iloc[-(ROLLING_WINDOW_DAYS+1)] if len(rolling_sentiment) > ROLLING_WINDOW_DAYS else rolling_sentiment.iloc[0]
        
        if previous_sentiment > 0:  # Avoid division by zero
            sentiment_change = (recent_sentiment - previous_sentiment) / abs(previous_sentiment)
            
            if sentiment_change < -SENTIMENT_DROP_THRESHOLD:
                alerts.append({
                    'product': product,
                    'title': product_data['title'].iloc[0] if 'title' in product_data.columns else product,
                    'category': product_data['main_category'].iloc[0] if 'main_category' in product_data.columns else 'Unknown',
                    'change': sentiment_change,
                    'recent_sentiment': recent_sentiment,
                    'previous_sentiment': previous_sentiment,
                    'review_count': len(product_data)
                })
    
    if alerts:
        alerts_df = pd.DataFrame(alerts)
        alerts_df = alerts_df.sort_values('change')
        
        st.dataframe(
            alerts_df.style.format({
                'change': '{:.1%}',
                'recent_sentiment': '{:.2f}',
                'previous_sentiment': '{:.2f}'
            }),
            use_container_width=True
        )
        
        # Download button
        csv = alerts_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Alerts as CSV",
            data=csv,
            file_name="sentiment_alerts.csv",
            mime="text/csv"
        )
    else:
        st.success("✅ No sentiment drop alerts detected!")
    
    st.markdown("---")
    
    # Emerging aspects
    st.subheader("Emerging Aspects")
    st.caption("New aspects appearing in recent reviews")
    
    # Compare recent vs older aspects
    cutoff_date = df_filtered['date'].max() - pd.Timedelta(days=30)
    recent_aspects = set(df_filtered[df_filtered['date'] > cutoff_date]['aspect_term_normalized'].unique())
    older_aspects = set(df_filtered[df_filtered['date'] <= cutoff_date]['aspect_term_normalized'].unique())
    
    new_aspects = recent_aspects - older_aspects
    
    if new_aspects:
        new_aspect_counts = df_filtered[
            (df_filtered['aspect_term_normalized'].isin(new_aspects)) &
            (df_filtered['date'] > cutoff_date)
        ]['aspect_term_normalized'].value_counts()
        
        st.write(f"**{len(new_aspects)} new aspects detected in past 30 days:**")
        for aspect, count in new_aspect_counts.head(20).items():
            st.write(f"- **{aspect}**: {count} mentions")
    else:
        st.info("No new aspects detected in recent period.")


# ==================== FOOTER ====================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dashboard Info")
st.sidebar.info(f"**Data Range:** {df['date'].min().date()} to {df['date'].max().date()}")
st.sidebar.info(f"**Total Records:** {len(df):,}")
