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
from genai_client import qa_for_product, build_and_persist_index, load_index_and_metadata

# Forecasting module (optional - graceful fallback if not available)
try:
    from forecasting import forecast_aspect_sentiment, batch_forecast_aspects, SentimentForecaster, PROPHET_AVAILABLE, ARIMA_AVAILABLE
    FORECASTING_ENABLED = PROPHET_AVAILABLE or ARIMA_AVAILABLE
    print(f"✅ Forecasting import successful. PROPHET={PROPHET_AVAILABLE}, ARIMA={ARIMA_AVAILABLE}, ENABLED={FORECASTING_ENABLED}")
except ImportError as e:
    FORECASTING_ENABLED = False
    PROPHET_AVAILABLE = False
    ARIMA_AVAILABLE = False
    print(f"❌ Forecasting import failed: {e}")
    # Warning will be shown on the forecasting page itself, not globally
except Exception as e:
    FORECASTING_ENABLED = False
    print(f"❌ Unexpected error in forecasting import: {e}")

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
     "📈 Product Deep Dive", "🔮 Sentiment Forecasting", "🚨 Alerts & Anomalies"]
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

# Product filter (ASIN)
if 'parent_asin' in df.columns:
    product_counts = df['parent_asin'].value_counts()
    product_options = ['All Products'] + [
        f"{asin} ({count} reviews)" 
        for asin, count in product_counts.head(50).items()
    ]
    selected_product = st.sidebar.selectbox(
        "Product (ASIN)",
        product_options,
        help="Filter by specific product or analyze all products"
    )
    # Extract ASIN from selection
    selected_asin = selected_product.split(' ')[0] if selected_product != 'All Products' else 'All'
else:
    selected_asin = 'All'

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
if selected_asin != 'All':
    df_filtered = df_filtered[df_filtered['parent_asin'] == selected_asin]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Filtered Data:** {len(df_filtered):,} reviews")


# ==================== PAGE 1: SENTIMENT OVERVIEW ====================
if page == "📊 Sentiment Overview":
    st.title("📊 Sentiment Overview Dashboard")
    st.markdown("High-level sentiment metrics and trends across all products.")
    

    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)

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
        pct_neutral = (df_filtered['sentiment'] == 'Neutral').sum() / len(df_filtered) * 100
        st.metric("Neutral Reviews", f"{pct_neutral:.1f}%")

    st.markdown("---")
    # Display Aspect Word Cloud and Top Extracted Aspects side by side
    wc_col, bar_col = st.columns(2)
    with wc_col:
        st.subheader("Aspect Word Cloud")
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            aspect_freq = df_filtered['aspect_term_normalized'].value_counts().to_dict()
            wordcloud = WordCloud(width=800, height=300, background_color='white').generate_from_frequencies(aspect_freq)
            fig_wc, ax_wc = plt.subplots(figsize=(6, 3))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)
        except ImportError:
            st.warning("wordcloud package not installed. Run 'pip install wordcloud matplotlib' to enable this feature.")

        # Average Sentiment Score Trend
        st.subheader("Average Sentiment Score Trend")
        if 'sentiment_score' in df_filtered.columns and 'date' in df_filtered.columns:
            df_trend = df_filtered.copy()
            df_trend['date'] = pd.to_datetime(df_trend['date'])
            trend = df_trend.groupby(df_trend['date'].dt.to_period('M'))['sentiment_score'].mean().reset_index()
            trend['date'] = trend['date'].dt.to_timestamp()
            fig_trend = px.line(trend, x='date', y='sentiment_score', markers=True,
                               labels={'date': 'Date', 'sentiment_score': 'Average Sentiment Score'},
                               title='Average Sentiment Score Over Time')
            fig_trend.update_layout(showlegend=False)
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Sentiment score or date column not found in data.")
    with bar_col:
        st.subheader("Top Extracted Aspects")
        aspect_counts = df_filtered['aspect_term_normalized'].value_counts().head(20)
        fig_aspect_bar = px.bar(
            x=aspect_counts.values,
            y=aspect_counts.index,
            orientation='h',
            labels={'x': 'Mention Count', 'y': 'Aspect'},
            color=aspect_counts.values,
            color_continuous_scale='Blues',
            title="Top 20 Extracted Aspects"
        )
        fig_aspect_bar.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_aspect_bar, use_container_width=True)
    
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
    
    st.markdown("---")
    
    # Sentiment vs Rating Scatter Plot
    st.subheader("Sentiment-Rating Coherence Analysis")
    st.caption("Advanced analysis comparing ABSA-detected sentiment with user star ratings")
    
    # Check if rating column exists
    if 'rating' in df_filtered.columns:
        # Aggregate by review to get one point per review
        # Use confidence-weighted sentiment for better accuracy
        def weighted_sentiment(sentiment_scores):
            # This function receives a Series of sentiment_score values
            # We need to access the confidence values from the parent DataFrame
            return sentiment_scores.mean()  # Fallback to mean for now
        
        # Build aggregation with proper access to confidence
        if 'confidence' in df_filtered.columns:
            # Create weighted average directly in aggregation
            review_agg = df_filtered.groupby('review_id').apply(
                lambda group: pd.Series({
                    'sentiment_score': (group['sentiment_score'] * group['confidence']).sum() / group['confidence'].sum(),
                    'rating': group['rating'].iloc[0],
                    'sentiment': group['sentiment'].mode()[0] if len(group['sentiment'].mode()) > 0 else group['sentiment'].iloc[0],
                    'review_length': len(str(group['text'].iloc[0])) if len(group) > 0 else 0,
                    'aspect_count': len(group)
                })
            ).reset_index()
        else:
            # Fallback without confidence weighting
            review_agg = df_filtered.groupby('review_id').agg({
                'sentiment_score': 'mean',
                'rating': 'first',
                'sentiment': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
                'text': lambda x: len(str(x.iloc[0])) if len(x) > 0 else 0,
                'aspect_term_normalized': 'count'
            }).reset_index()
            
            review_agg.rename(columns={
                'text': 'review_length',
                'aspect_term_normalized': 'aspect_count'
            }, inplace=True)
        
        # Visualization type selector
        viz_type = st.radio(
            "Select Visualization Type",
            ["Scatter Plot", "Hexbin Density", "2D Histogram", "Box Plots by Rating", "All Views"],
            horizontal=True
        )
        
        # Calculate correlation coefficient (used in all views)
        correlation = review_agg[['rating', 'sentiment_score']].corr().iloc[0, 1]
        
        # ===== SCATTER PLOT =====
        if viz_type in ["Scatter Plot", "All Views"]:
            st.markdown("#### Interactive Scatter Plot")
            fig_scatter = px.scatter(
                review_agg,
                x='rating',
                y='sentiment_score',
                color='sentiment',
                size='aspect_count',  # Bubble size = number of aspects
                color_discrete_map=COLOR_PALETTE_SENTIMENT,
                opacity=0.6,
                title="Sentiment Score vs Star Rating (bubble size = aspect count)",
                labels={
                    'rating': 'Star Rating (1-5)',
                    'sentiment_score': 'Confidence-Weighted Sentiment Score',
                    'sentiment': 'Dominant Sentiment',
                    'aspect_count': 'Number of Aspects'
                },
                hover_data=['review_id', 'review_length', 'aspect_count']
            )
            
            # Add ideal correlation line
            ideal_x = [1, 2, 3, 4, 5]
            ideal_y = [-1, -0.5, 0, 0.5, 1]
            fig_scatter.add_trace(go.Scatter(
                x=ideal_x,
                y=ideal_y,
                mode='lines',
                name='Ideal Correlation',
                line=dict(color='gray', dash='dash', width=2),
                showlegend=True
            ))
            
            fig_scatter.update_layout(
                xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                yaxis=dict(range=[-1.5, 1.5]),
                height=500
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # ===== HEXBIN DENSITY PLOT =====
        if viz_type in ["Hexbin Density", "All Views"]:
            st.markdown("#### Hexbin Density Plot")
            st.caption("Shows concentration areas - darker colors indicate more reviews")
            
            # Create hexbin using plotly density heatmap
            fig_hexbin = go.Figure()
            
            # Create 2D histogram for hexbin effect
            fig_hexbin = go.Figure(go.Histogram2d(
                x=review_agg['rating'],
                y=review_agg['sentiment_score'],
                colorscale='YlOrRd',
                nbinsx=20,
                nbinsy=30,
                colorbar=dict(title="Review<br>Count")
            ))
            
            # Add ideal line
            fig_hexbin.add_trace(go.Scatter(
                x=ideal_x,
                y=ideal_y,
                mode='lines',
                name='Ideal Correlation',
                line=dict(color='cyan', dash='dash', width=3),
                showlegend=True
            ))
            
            fig_hexbin.update_layout(
                title=f"Review Density Heatmap (Correlation: {correlation:.3f})",
                xaxis_title="Star Rating (1-5)",
                yaxis_title="Confidence-Weighted Sentiment Score",
                xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                yaxis=dict(range=[-1.5, 1.5]),
                height=500
            )
            
            st.plotly_chart(fig_hexbin, use_container_width=True)
        
        # ===== 2D HISTOGRAM WITH COLOR GRADIENT =====
        if viz_type in ["2D Histogram", "All Views"]:
            st.markdown("#### 2D Histogram Density Map")
            st.caption("Color intensity represents review concentration in each bin")
            
            fig_2dhist = px.density_heatmap(
                review_agg,
                x='rating',
                y='sentiment_score',
                nbinsx=10,
                nbinsy=20,
                color_continuous_scale='Viridis',
                title=f"Review Distribution Heatmap (Pearson r = {correlation:.3f})",
                labels={
                    'rating': 'Star Rating (1-5)',
                    'sentiment_score': 'Confidence-Weighted Sentiment Score'
                }
            )
            
            # Add ideal line overlay
            fig_2dhist.add_trace(go.Scatter(
                x=ideal_x,
                y=ideal_y,
                mode='lines+markers',
                name='Expected Pattern',
                line=dict(color='red', dash='dash', width=3),
                marker=dict(size=10, color='red', symbol='x'),
                showlegend=True
            ))
            
            fig_2dhist.update_layout(
                xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                yaxis=dict(range=[-1.5, 1.5]),
                height=500
            )
            
            st.plotly_chart(fig_2dhist, use_container_width=True)
        
        # ===== BOX PLOTS BY RATING =====
        if viz_type in ["Box Plots by Rating", "All Views"]:
            st.markdown("#### Box Plots: Sentiment Distribution by Star Rating")
            st.caption("Shows sentiment score distribution for each rating level - reveals variance and outliers")
            
            fig_box = px.box(
                review_agg,
                x='rating',
                y='sentiment_score',
                color='rating',
                title="Sentiment Score Distribution by Star Rating",
                labels={
                    'rating': 'Star Rating (1-5)',
                    'sentiment_score': 'Confidence-Weighted Sentiment Score'
                },
                points='outliers'  # Show outlier points
            )
            
            # Add horizontal line at y=0
            fig_box.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray",
                opacity=0.5,
                annotation_text="Neutral Sentiment"
            )
            
            # Add expected sentiment zones
            fig_box.add_hrect(
                y0=0.5, y1=1.5,
                fillcolor="green", opacity=0.1,
                layer="below", line_width=0,
                annotation_text="Expected for 4-5★",
                annotation_position="right"
            )
            fig_box.add_hrect(
                y0=-1.5, y1=-0.5,
                fillcolor="red", opacity=0.1,
                layer="below", line_width=0,
                annotation_text="Expected for 1-2★",
                annotation_position="right"
            )
            
            fig_box.update_layout(
                xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                yaxis=dict(range=[-1.5, 1.5]),
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Statistical summary per rating
            with st.expander("📊 Statistical Summary by Rating"):
                stats_by_rating = review_agg.groupby('rating')['sentiment_score'].agg([
                    ('count', 'count'),
                    ('mean', 'mean'),
                    ('median', 'median'),
                    ('std', 'std'),
                    ('min', 'min'),
                    ('max', 'max')
                ]).round(3)
                st.dataframe(stats_by_rating, use_container_width=True)
        
        st.markdown("---")
        
        # ===== METRICS ROW (Common to all visualizations) =====
        st.markdown("#### Coherence Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pearson Correlation", f"{correlation:.3f}")
        with col2:
            coherent = review_agg[
                ((review_agg['rating'] >= 4) & (review_agg['sentiment_score'] > 0)) |
                ((review_agg['rating'] <= 2) & (review_agg['sentiment_score'] < 0)) |
                ((review_agg['rating'] == 3) & (review_agg['sentiment_score'].abs() < 0.3))
            ]
            coherence_rate = len(coherent) / len(review_agg) * 100
            st.metric("Coherence Rate", f"{coherence_rate:.1f}%", 
                     help="% of reviews where sentiment matches rating expectation")
        with col3:
            divergent = review_agg[
                ((review_agg['rating'] >= 4) & (review_agg['sentiment_score'] < -0.3)) |
                ((review_agg['rating'] <= 2) & (review_agg['sentiment_score'] > 0.3))
            ]
            divergence_rate = len(divergent) / len(review_agg) * 100
            st.metric("Divergence Rate", f"{divergence_rate:.1f}%", 
                     delta=f"-{divergence_rate:.1f}%",
                     delta_color="inverse",
                     help="% of reviews with strong rating-sentiment mismatch")
        with col4:
            avg_aspects = review_agg['aspect_count'].mean()
            st.metric("Avg Aspects/Review", f"{avg_aspects:.1f}",
                     help="Average number of aspects extracted per review")
        
        # Show examples of divergent reviews
        if len(divergent) > 0:
            with st.expander("🔍 View Divergent Reviews (Rating-Sentiment Mismatch)"):
                st.caption("Reviews where star rating and detected sentiment strongly disagree")
                divergent_sample = divergent.head(10)
                for idx, row in divergent_sample.iterrows():
                    review_details = df_filtered[df_filtered['review_id'] == row['review_id']].iloc[0]
                    st.markdown(f"""
                    **Review ID:** `{row['review_id']}` | **Rating:** {row['rating']}⭐ | **Sentiment Score:** {row['sentiment_score']:.2f}
                    
                    *{review_details.get('text', 'No text available')[:200]}...*
                    """)
                    st.markdown("---")
    else:
        st.warning("⚠️ Rating column not found in dataset. Cannot generate sentiment-rating coherence plot.")


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

        # New layout: left = sentiment by aspect, right = temporal evolution with aspect breakdown
        left_col, right_col = st.columns(2)

        with left_col:
            st.subheader("Sentiment by Aspect")
            aspect_dist = product_data['aspect_term_normalized'].value_counts().head(10)
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

        with right_col:
            st.subheader("Temporal Evolution by Aspect (Daily)")
            # Aspect selector for breakdown
            available_aspects = product_data['aspect_term_normalized'].value_counts().head(10).index.tolist()
            selected_aspects = st.multiselect(
                "Select Aspects to Show",
                options=available_aspects,
                default=available_aspects[:3],
                help="Choose aspects to visualize their sentiment evolution over time"
            )
            if selected_aspects:
                aspect_evolution = product_data[product_data['aspect_term_normalized'].isin(selected_aspects)]
                aspect_evolution = aspect_evolution.groupby([pd.Grouper(key='date', freq='D'), 'aspect_term_normalized'])['sentiment_score'].mean().reset_index()
                fig_evolution = px.line(
                    aspect_evolution,
                    x='date',
                    y='sentiment_score',
                    color='aspect_term_normalized',
                    title="Aspect Sentiment Evolution Over Time (Daily)",
                    labels={'date': 'Date', 'sentiment_score': 'Avg Sentiment Score', 'aspect_term_normalized': 'Aspect'}
                )
                fig_evolution.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_evolution, use_container_width=True)
            else:
                st.info("Select at least one aspect to view its temporal evolution.")

        st.markdown("---")

        # ==================== GenAI Panel ====================
        st.markdown("---")
        st.subheader("GenAI Insights (Product Q&A & Summary)")

        col_q1, col_q2, col_q3 = st.columns([4, 1, 1])
        with col_q1:
            question = st.text_input("Ask a question about this product (e.g. 'What are top complaints in the last 90 days?')", key=f"qa_{selected_product}")
        with col_q2:
            ask_button = st.button("Ask GenAI", key=f"ask_{selected_product}")
        with col_q3:
            force_refresh = st.checkbox("Force refresh", value=False, help="Bypass local cache and force the model to re-run")

        # Quick chat / summary button (pre-filled example question)
        if st.button("Quick Chat", key=f"summ_{selected_product}"):
            st.info("Generating answer — this may use an external API or local model if configured.")
            with st.spinner('Running GenAI...'):
                q = f"Provide a concise answer about product {selected_product} based on recent reviews. Cite supporting snippets as [n] where applicable."
                resp = qa_for_product(df_filtered, selected_product, q, top_k=8, force_refresh=force_refresh)
                if resp.get('error'):
                    st.error(resp['error'])
                else:
                    st.markdown("**GenAI Response**")
                    # show cache status
                    if resp.get('cached'):
                        try:
                            import datetime
                            cached_at = resp.get('cached_at')
                            if cached_at:
                                st.info(f"Returned from cache (cached at {datetime.datetime.fromtimestamp(int(cached_at)).strftime('%Y-%m-%d %H:%M:%S')})")
                            else:
                                st.info("Returned from cache")
                        except Exception:
                            st.info("Returned from cache")
                    else:
                        if force_refresh:
                            st.warning("Force refresh: bypassed cache and generated a fresh answer")
                    # Prefer structured output when available
                    structured = resp.get('structured')
                    # Debug banner: show parse status and truncation info
                    parse_method = resp.get('structured_parse_method') or resp.get('structured_parse_error')
                    prompt_meta = resp.get('prompt_meta')
                    debug_msg = []
                    if parse_method:
                        debug_msg.append(f"parse={parse_method}")
                    if prompt_meta is not None:
                        was_truncated = (prompt_meta.get('max_snips') < 6) or (prompt_meta.get('snippet_char_limit') < 300)
                        debug_msg.append(f"truncated={was_truncated}")
                        if prompt_meta.get('final_token_len') is not None:
                            debug_msg.append(f"tokens={prompt_meta.get('final_token_len')}/{prompt_meta.get('allowed_input_tokens')}")
                    if debug_msg:
                        st.caption(" · ".join(debug_msg))
                    if structured:
                        st.markdown("**Summary**")
                        st.write(structured.get('summary', ''))
                        if structured.get('pros'):
                            st.markdown("**Pros**")
                            for p in structured.get('pros', []):
                                st.write(f"- {p}")
                        if structured.get('cons'):
                            st.markdown("**Cons**")
                            for c in structured.get('cons', []):
                                st.write(f"- {c}")
                        if structured.get('evidence'):
                            st.markdown("**Evidence**")
                            for ev in structured.get('evidence', []):
                                idx = ev.get('index')
                                excerpt = ev.get('excerpt')
                                if resp.get('snippets') and isinstance(idx, int) and idx < len(resp.get('snippets')):
                                    s = resp['snippets'][idx]
                                    st.write(f"- [{idx}] {excerpt} — {s.get('date')} ({s.get('review_id')})")
                                else:
                                    st.write(f"- [{idx}] {excerpt}")
                        # also offer raw answer if user wants to see original text
                        with st.expander('Show raw model output'):
                            st.code(resp.get('answer') or '')
                    else:
                        st.write(resp.get('answer'))
                        if resp.get('snippets'):
                            with st.expander("Show supporting snippets"):
                                for s in resp['snippets']:
                                    st.write(f"- {s.get('date')} — {s.get('text')[:300]}... (score={s.get('score', 0):.2f})")

        # If user asked a question
        if ask_button and question:
            st.info("Querying GenAI — this may use an external API if configured.")
            with st.spinner('Running GenAI...'):
                resp = qa_for_product(df_filtered, selected_product, question, top_k=8, force_refresh=force_refresh)
                if resp.get('error'):
                    st.error(resp['error'])
                else:
                    st.markdown("**Answer**")
                    if resp.get('cached'):
                        try:
                            import datetime
                            cached_at = resp.get('cached_at')
                            if cached_at:
                                st.info(f"Returned from cache (cached at {datetime.datetime.fromtimestamp(int(cached_at)).strftime('%Y-%m-%d %H:%M:%S')})")
                            else:
                                st.info("Returned from cache")
                        except Exception:
                            st.info("Returned from cache")
                    else:
                        if force_refresh:
                            st.warning("Force refresh: bypassed cache and generated a fresh answer")
                    structured = resp.get('structured')
                    if structured:
                        st.markdown("**Summary**")
                        st.write(structured.get('summary', ''))
                        if structured.get('pros'):
                            st.markdown("**Pros**")
                            for p in structured.get('pros', []):
                                st.write(f"- {p}")
                        if structured.get('cons'):
                            st.markdown("**Cons**")
                            for c in structured.get('cons', []):
                                st.write(f"- {c}")
                        if structured.get('evidence'):
                            st.markdown("**Evidence**")
                            for ev in structured.get('evidence', []):
                                idx = ev.get('index')
                                excerpt = ev.get('excerpt')
                                if resp.get('snippets') and isinstance(idx, int) and idx < len(resp.get('snippets')):
                                    s = resp['snippets'][idx]
                                    st.write(f"- [{idx}] {excerpt} — {s.get('date')} ({s.get('review_id')})")
                                else:
                                    st.write(f"- [{idx}] {excerpt}")
                        with st.expander('Show raw model output'):
                            st.code(resp.get('answer') or '')
                    else:
                        st.write(resp.get('answer'))
                        
                        # Show snippets/sources that were used
                        if resp.get('snippets'):
                            st.markdown("---")
                            st.markdown("**📚 Source Reviews Used**")
                            for i, s in enumerate(resp['snippets'][:8], 1):  # Show top 8
                                with st.expander(f"Review {i} (Score: {s.get('rerank_score', s.get('score', 0)):.2f})"):
                                    st.write(f"**Date:** {s.get('date', 'N/A')}")
                                    st.write(f"**Review ID:** {s.get('review_id', 'N/A')}")
                                    st.write(f"**Text:**")
                                    st.write(s.get('text', '')[:500])
                                    if len(s.get('text', '')) > 500:
                                        st.write("...")
                        elif resp.get('sources'):
                            st.markdown("---")
                            st.markdown("**📚 Cited Sources**")
                            for i, s in enumerate(resp['sources'], 1):
                                st.write(f"{i}. {s.get('text', s.get('snippet', ''))[:300]}...")


# ==================== PAGE 5: SENTIMENT FORECASTING ====================
elif page == "🔮 Sentiment Forecasting":
    st.title("🔮 Sentiment Forecasting")
    st.markdown("Predict future sentiment trends using time series models (Prophet).")
    
    if not FORECASTING_ENABLED:
        st.error("❌ Forecasting is not available. Install dependencies:")
        st.code("pip install prophet statsmodels")
        st.stop()
    
    # Configuration
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # Model selector
        available_models = []
        if PROPHET_AVAILABLE:
            available_models.append("Prophet")
        if ARIMA_AVAILABLE:
            available_models.append("ARIMA")
        
        if len(available_models) == 0:
            st.error("No models available!")
            st.stop()
        
        selected_model = st.selectbox("Model", available_models, index=0)
    
    with col2:
        forecast_days = st.selectbox("Forecast Horizon", [30, 60, 90, 180], index=2)
    with col3:
        aggregation_freq = st.selectbox("Aggregation", ["Daily", "Weekly", "Monthly"], index=0)
        freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
        freq = freq_map[aggregation_freq]
    with col4:
        min_reviews = st.number_input("Min Reviews", min_value=10, max_value=10000, value=20)
    
    # Data quality controls
    st.markdown("### 📊 Data Quality Controls")
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        min_samples_per_period = st.number_input(
            f"Min Reviews per {aggregation_freq[:-2] if aggregation_freq != 'Daily' else 'Day'}",
            min_value=1,
            max_value=50,
            value=5,
            help="Minimum number of reviews required per time period. Higher = better data quality but less data points."
        )
    with col_q2:
        st.metric(
            "Expected Min Data Points",
            value=f"~{min_reviews // min_samples_per_period}",
            help=f"Estimated minimum time periods with >={min_samples_per_period} reviews"
        )
    
    # Advanced Model Tuning
    with st.expander("🔧 Advanced Model Parameters (Fine-tuning)"):
        if selected_model == "Prophet":
            st.markdown("### Prophet Hyperparameters")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                changepoint_scale = st.slider(
                    "Trend Flexibility",
                    min_value=0.001, max_value=0.5, value=0.05, step=0.01,
                    help="Higher = more flexible trend (can overfit). Lower = smoother trend."
                )
            with col_b:
                seasonality_scale = st.slider(
                    "Seasonality Strength", 
                    min_value=0.01, max_value=10.0, value=1.0, step=0.1,
                    help="Higher = stronger seasonal patterns. Lower = smoother forecast."
                )
            with col_c:
                weekly_seasonality = st.checkbox("Weekly Seasonality", value=True)
                yearly_seasonality = st.checkbox("Yearly Seasonality", value=False)
        else:  # ARIMA
            st.markdown("### ARIMA Parameters (p, d, q)")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                p = st.number_input("AR Order (p)", min_value=0, max_value=5, value=1, 
                                   help="Autoregressive: uses past values")
            with col_b:
                d = st.number_input("Differencing (d)", min_value=0, max_value=2, value=1,
                                   help="Makes data stationary (remove trends)")
            with col_c:
                q = st.number_input("MA Order (q)", min_value=0, max_value=5, value=1,
                                   help="Moving average: uses past errors")
    
    # Show model info
    if selected_model == "Prophet":
        st.info("📊 **Prophet**: Additive model with trend + seasonality. Best for data with clear patterns.")
    else:
        st.info("📈 **ARIMA**: AutoRegressive model. Best for stationary time series with linear trends.")
    
    st.markdown("---")
    
    # Show products included in analysis
    if 'parent_asin' in df_filtered.columns:
        product_counts = df_filtered['parent_asin'].value_counts()
        total_products = len(product_counts)
        total_reviews = len(df_filtered)
        
        if selected_asin == 'All':
            with st.expander(f"📦 Products Included in Forecast ({total_products} products, {total_reviews:,} reviews)"):
                st.markdown("**Top 10 Products by Review Count:**")
                top_products = product_counts.head(10)
                for asin, count in top_products.items():
                    pct = (count / total_reviews) * 100
                    st.write(f"- **{asin}**: {count:,} reviews ({pct:.1f}%)")
                if total_products > 10:
                    st.write(f"*...and {total_products - 10} more products*")
        else:
            st.success(f"📦 **Single Product Analysis**: {selected_asin} ({total_reviews:,} reviews)")
    
    # Select aspect to forecast
    aspect_counts = df_filtered['aspect_term_normalized'].value_counts()
    top_aspects = aspect_counts[aspect_counts >= min_reviews].head(20).index.tolist()
    
    if len(top_aspects) == 0:
        st.warning(f"No aspects found with at least {min_reviews} reviews. Lower the minimum threshold.")
        st.stop()
    
    selected_aspect = st.selectbox(
        "Select Aspect to Forecast",
        top_aspects,
        help="Choose an aspect with sufficient historical data"
    )
    
    # Run forecasting
    if st.button(f"🚀 Generate Forecast for '{selected_aspect}'", type="primary"):
        with st.spinner(f"Training {selected_model} model and generating {forecast_days}-day forecast..."):
            # Prepare model parameters
            model_params = {}
            if selected_model == "Prophet":
                model_params = {
                    'changepoint_prior_scale': changepoint_scale,
                    'seasonality_prior_scale': seasonality_scale,
                    'weekly_seasonality': weekly_seasonality,
                    'yearly_seasonality': yearly_seasonality
                }
            else:  # ARIMA
                model_params = {
                    'order': (p, d, q)
                }
            
            result = forecast_aspect_sentiment(
                df_filtered, 
                selected_aspect, 
                forecast_days, 
                freq=freq,
                aspect_col='aspect_term_normalized',
                model_type=selected_model.lower(),
                model_params=model_params,
                min_samples_per_period=min_samples_per_period
            )
        
        if not result['success']:
            st.error(f"❌ Forecasting failed: {result.get('error', 'Unknown error')}")
            st.stop()
        
        # Display results
        st.success(f"✅ Forecast generated using **{result['model']}** with {result['data_points']} historical data points")
        
        # Show scope of analysis
        if 'parent_asin' in df_filtered.columns:
            if selected_asin == 'All':
                product_count = df_filtered['parent_asin'].nunique()
                st.info(f"📊 **Analysis Scope**: Aspect '{selected_aspect}' across **{product_count} products** (aggregated sentiment)")
            else:
                st.info(f"📊 **Analysis Scope**: Aspect '{selected_aspect}' for product **{selected_asin}**")
        
        # Trend Analysis Summary
        st.subheader("📈 Trend Analysis")
        trend_analysis = result['trend_analysis']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Current Sentiment",
                f"{trend_analysis['current_sentiment']:.3f}",
                help="Latest predicted sentiment score"
            )
        with col2:
            st.metric(
                "Trend Direction",
                trend_analysis['trend_direction'],
                f"{trend_analysis['slope']:.4f}/day",
                delta_color="normal" if trend_analysis['slope'] >= 0 else "inverse"
            )
        with col3:
            if trend_analysis['predicted_30d'] is not None:
                delta_30d = trend_analysis['predicted_30d'] - trend_analysis['current_sentiment']
                st.metric(
                    "30-Day Forecast",
                    f"{trend_analysis['predicted_30d']:.3f}",
                    f"{delta_30d:+.3f}",
                    delta_color="normal" if delta_30d >= 0 else "inverse"
                )
        with col4:
            if trend_analysis['predicted_90d'] is not None:
                delta_90d = trend_analysis['predicted_90d'] - trend_analysis['current_sentiment']
                st.metric(
                    "90-Day Forecast",
                    f"{trend_analysis['predicted_90d']:.3f}",
                    f"{delta_90d:+.3f}",
                    delta_color="normal" if delta_90d >= 0 else "inverse"
                )
        
        # Alerts
        if trend_analysis['alerts']:
            st.warning("⚠️ **Alerts Detected:**")
            for alert in trend_analysis['alerts']:
                st.write(f"- {alert['message']}")
        
        st.markdown("---")
        
        # Main Forecast Plot
        st.subheader("📊 Sentiment Forecast with Confidence Intervals")
        
        forecast = result['forecast']
        historical = result['historical']
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical['ds'],
            y=historical['y'],
            mode='markers',
            name='Historical Data',
            marker=dict(size=8, color='darkblue', opacity=0.6)
        ))
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='blue', width=2)
        ))
        
        # 95% Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
            y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,100,200,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence',
            showlegend=True
        ))
        
        # Mark today (boundary between historical and forecast)
        today = historical['ds'].max()
        
        fig.update_layout(
            title=f"Sentiment Forecast: {selected_aspect}",
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            height=500,
            hovermode='x unified',
            shapes=[
                dict(
                    type='line',
                    x0=today,
                    x1=today,
                    y0=0,
                    y1=1,
                    yref='paper',
                    line=dict(color='red', width=2, dash='dash')
                )
            ],
            annotations=[
                dict(
                    x=today,
                    y=1,
                    yref='paper',
                    text='Today',
                    showarrow=False,
                    yshift=10,
                    font=dict(color='red')
                )
            ]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Change Points (Prophet only)
        if result['model'] == 'Prophet' and result['changepoints']:
            st.subheader("📍 Detected Change Points")
            changepoints = result['changepoints']
            if changepoints:
                st.info(f"Detected {len(changepoints)} significant sentiment shifts:")
                for cp in changepoints:
                    st.write(f"- {cp.strftime('%Y-%m-%d')}")
            else:
                st.success("No significant change points detected - sentiment remains stable")
        
        # Anomalies (Prophet only)
        if result['model'] == 'Prophet':
            st.subheader("🔍 Historical Anomalies")
            anomalies = result['anomalies']
            anomaly_df = anomalies[anomalies['is_anomaly'] == True].sort_values('anomaly_score', ascending=False)
            
            if len(anomaly_df) > 0:
                st.warning(f"Found {len(anomaly_df)} anomalous data points:")
                
                # Plot anomalies
                fig_anom = go.Figure()
                
                fig_anom.add_trace(go.Scatter(
                    x=anomalies['ds'],
                    y=anomalies['y'],
                    mode='markers',
                    name='Normal',
                    marker=dict(size=6, color='green'),
                    opacity=0.5
                ))
                
                fig_anom.add_trace(go.Scatter(
                    x=anomaly_df['ds'],
                    y=anomaly_df['y'],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(size=12, color='red', symbol='x'),
                    text=anomaly_df['anomaly_score'].round(2),
                    hovertemplate='<b>Anomaly</b><br>Date: %{x}<br>Sentiment: %{y:.3f}<br>Score: %{text}<extra></extra>'
                ))
                
                fig_anom.add_trace(go.Scatter(
                    x=anomalies['ds'],
                    y=anomalies['yhat'],
                    mode='lines',
                    name='Expected',
                    line=dict(color='blue', dash='dash')
                ))
                
                fig_anom.update_layout(
                    title="Anomaly Detection",
                    xaxis_title="Date",
                    yaxis_title="Sentiment Score",
                    height=400
                )
                
                st.plotly_chart(fig_anom, use_container_width=True)
                
                # Top anomalies table
                with st.expander("📋 View Anomaly Details"):
                    st.dataframe(
                        anomaly_df[['ds', 'y', 'yhat', 'anomaly_score', 'count']].head(20),
                        use_container_width=True
                    )
            else:
                st.success("✅ No anomalies detected - all data points within expected range")
        
        # Decomposition (Prophet only)
        if result['model'] == 'Prophet' and result['decomposition'] is not None:
            st.subheader("🔬 Seasonality Decomposition")
            decomp = result['decomposition']
            
            fig_decomp = go.Figure()
            
            # Plot each component
            fig_decomp.add_trace(go.Scatter(
                x=decomp['ds'], y=decomp['observed'],
                mode='lines', name='Observed',
                line=dict(color='black')
            ))
            fig_decomp.add_trace(go.Scatter(
                x=decomp['ds'], y=decomp['trend'],
                mode='lines', name='Trend',
                line=dict(color='blue')
            ))
            fig_decomp.add_trace(go.Scatter(
                x=decomp['ds'], y=decomp['seasonal'],
                mode='lines', name='Seasonal',
                line=dict(color='green')
            ))
            fig_decomp.add_trace(go.Scatter(
                x=decomp['ds'], y=decomp['residual'],
                mode='lines', name='Residual',
                line=dict(color='red', dash='dot')
            ))
            
            fig_decomp.update_layout(
                title="Time Series Decomposition",
                xaxis_title="Date",
                yaxis_title="Component Value",
                height=500
            )
            
            st.plotly_chart(fig_decomp, use_container_width=True)
            
            st.caption("**Trend**: Long-term direction | **Seasonal**: Repeating patterns | **Residual**: Random noise")
    
    # Batch forecasting option
    st.markdown("---")
    st.subheader("📦 Batch Forecast Multiple Aspects")
    
    if st.button("Generate Forecasts for Top 5 Aspects"):
        top_5_aspects = top_aspects[:5]
        
        with st.spinner(f"Forecasting {len(top_5_aspects)} aspects..."):
            forecasts = {}
            for aspect in top_5_aspects:
                result = forecast_aspect_sentiment(df_filtered, aspect, forecast_days=90, freq='D')
                if result['success']:
                    forecasts[aspect] = result
        
        st.success(f"✅ Generated {len(forecasts)} forecasts")
        
        # Comparative plot
        fig_compare = go.Figure()
        
        for aspect, result in forecasts.items():
            forecast = result['forecast']
            future_forecast = forecast[forecast['ds'] > pd.to_datetime('today')]
            
            fig_compare.add_trace(go.Scatter(
                x=future_forecast['ds'],
                y=future_forecast['yhat'],
                mode='lines',
                name=aspect,
                hovertemplate=f'<b>{aspect}</b><br>Date: %{{x}}<br>Sentiment: %{{y:.3f}}<extra></extra>'
            ))
        
        fig_compare.update_layout(
            title="Comparative Sentiment Forecasts",
            xaxis_title="Date",
            yaxis_title="Predicted Sentiment Score",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Summary table
        summary_data = []
        for aspect, result in forecasts.items():
            trend = result['trend_analysis']
            summary_data.append({
                'Aspect': aspect,
                'Current': f"{trend['current_sentiment']:.3f}",
                'Trend': trend['trend_direction'],
                'Slope': f"{trend['slope']:.4f}",
                '30d Forecast': f"{trend['predicted_30d']:.3f}" if trend['predicted_30d'] else "N/A",
                '90d Forecast': f"{trend['predicted_90d']:.3f}" if trend['predicted_90d'] else "N/A",
                'Alerts': len(trend['alerts'])
            })
        
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)


# ==================== PAGE 6: ALERTS & ANOMALIES ====================
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
