"""
Utility Functions for ABSA Dashboard
=====================================
Helper functions for data processing, visualization, and analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def calculate_sentiment_score(sentiment, confidence):
    """
    Calculate sentiment score from sentiment label and confidence.
    
    Args:
        sentiment: 'Positive', 'Negative', or 'Neutral'
        confidence: float between 0 and 1
        
    Returns:
        float: sentiment score (positive confidence - negative confidence)
    """
    if sentiment == 'Positive':
        return confidence
    elif sentiment == 'Negative':
        return -confidence
    else:
        return 0


def get_rolling_sentiment(df, product_asin, window_days=7):
    """
    Calculate rolling average sentiment for a product.
    
    Args:
        df: DataFrame with columns ['parent_asin', 'date', 'sentiment_score']
        product_asin: Product identifier
        window_days: Rolling window size in days
        
    Returns:
        Series: Rolling sentiment scores indexed by date
    """
    product_data = df[df['parent_asin'] == product_asin].sort_values('date')
    product_data = product_data.set_index('date')
    rolling = product_data['sentiment_score'].rolling(f'{window_days}D', min_periods=1).mean()
    return rolling


def detect_sentiment_shifts(df, threshold=0.20, window_days=7, min_reviews=5):
    """
    Detect significant sentiment shifts for all products.
    
    Args:
        df: DataFrame with ABSA results
        threshold: Minimum percentage drop to flag (default 20%)
        window_days: Rolling window for comparison
        min_reviews: Minimum reviews needed to trigger alert
        
    Returns:
        DataFrame with alerts
    """
    alerts = []
    
    for product in df['parent_asin'].unique():
        rolling = get_rolling_sentiment(df, product, window_days)
        
        if len(rolling) < min_reviews:
            continue
        
        recent_sentiment = rolling.iloc[-1]
        previous_sentiment = rolling.iloc[-(window_days+1)] if len(rolling) > window_days else rolling.iloc[0]
        
        if previous_sentiment > 0:
            change = (recent_sentiment - previous_sentiment) / abs(previous_sentiment)
            
            if change < -threshold:
                product_info = df[df['parent_asin'] == product].iloc[0]
                alerts.append({
                    'product_asin': product,
                    'product_title': product_info.get('title', product),
                    'category': product_info.get('main_category', 'Unknown'),
                    'sentiment_change': change,
                    'recent_sentiment': recent_sentiment,
                    'previous_sentiment': previous_sentiment,
                    'alert_date': rolling.index[-1]
                })
    
    return pd.DataFrame(alerts)


def get_top_aspects_by_product(df, product_asin, n=10):
    """
    Get top N aspects for a specific product.
    
    Args:
        df: ABSA DataFrame
        product_asin: Product identifier
        n: Number of top aspects to return
        
    Returns:
        DataFrame with aspect counts and sentiments
    """
    product_data = df[df['parent_asin'] == product_asin]
    
    aspect_stats = product_data.groupby('aspect_term_normalized').agg({
        'aspect_term': 'count',
        'sentiment_score': 'mean',
        'is_positive': 'sum',
        'is_negative': 'sum',
        'is_neutral': 'sum'
    }).reset_index()
    
    aspect_stats.columns = ['aspect', 'count', 'avg_sentiment', 'positive', 'negative', 'neutral']
    aspect_stats = aspect_stats.sort_values('count', ascending=False).head(n)
    
    return aspect_stats


def get_aspect_sentiment_matrix(df, top_n_aspects=15, top_n_products=10):
    """
    Create aspect-product sentiment matrix for heatmap.
    
    Args:
        df: ABSA DataFrame
        top_n_aspects: Number of top aspects to include
        top_n_products: Number of top products to include
        
    Returns:
        DataFrame: Pivot table with aspects as rows, products as columns
    """
    # Get top aspects and products
    top_aspects = df['aspect_term_normalized'].value_counts().head(top_n_aspects).index
    top_products = df['parent_asin'].value_counts().head(top_n_products).index
    
    # Filter data
    filtered = df[
        (df['aspect_term_normalized'].isin(top_aspects)) &
        (df['parent_asin'].isin(top_products))
    ]
    
    # Create pivot table
    matrix = filtered.groupby(['aspect_term_normalized', 'parent_asin'])['sentiment_score'].mean().reset_index()
    matrix_pivot = matrix.pivot(index='aspect_term_normalized', columns='parent_asin', values='sentiment_score')
    
    return matrix_pivot


def calculate_momentum(df, product_asin, window_days=14):
    """
    Calculate sentiment momentum (rate of change) for a product.
    
    Args:
        df: ABSA DataFrame
        product_asin: Product identifier
        window_days: Window for momentum calculation
        
    Returns:
        float: Momentum score (positive = improving, negative = declining)
    """
    rolling = get_rolling_sentiment(df, product_asin, window_days)
    
    if len(rolling) < window_days:
        return 0
    
    # Calculate slope of recent trend
    recent_values = rolling.iloc[-window_days:].values
    x = np.arange(len(recent_values))
    slope = np.polyfit(x, recent_values, 1)[0]
    
    return slope


def get_product_comparison(df, product_asins):
    """
    Compare multiple products side-by-side.
    
    Args:
        df: ABSA DataFrame
        product_asins: List of product identifiers
        
    Returns:
        DataFrame: Comparison table
    """
    comparison = []
    
    for asin in product_asins:
        product_data = df[df['parent_asin'] == asin]
        
        if len(product_data) == 0:
            continue
        
        comparison.append({
            'product_asin': asin,
            'product_title': product_data['title'].iloc[0] if 'title' in product_data.columns else asin,
            'review_count': len(product_data),
            'avg_sentiment': product_data['sentiment_score'].mean(),
            'pct_positive': (product_data['sentiment'] == 'Positive').sum() / len(product_data) * 100,
            'pct_negative': (product_data['sentiment'] == 'Negative').sum() / len(product_data) * 100,
            'unique_aspects': product_data['aspect_term_normalized'].nunique(),
            'avg_confidence': product_data['confidence'].mean()
        })
    
    return pd.DataFrame(comparison)


def filter_by_date_range(df, start_date, end_date):
    """
    Filter DataFrame by date range.
    
    Args:
        df: DataFrame with 'date' column
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        
    Returns:
        Filtered DataFrame
    """
    df_filtered = df[
        (df['date'] >= pd.to_datetime(start_date)) &
        (df['date'] <= pd.to_datetime(end_date))
    ]
    return df_filtered


def get_category_trends(df):
    """
    Calculate sentiment trends by product category.
    
    Args:
        df: ABSA DataFrame with 'main_category' column
        
    Returns:
        DataFrame with category-level metrics
    """
    if 'main_category' not in df.columns:
        return pd.DataFrame()
    
    category_stats = df.groupby('main_category').agg({
        'review_id': 'count',
        'parent_asin': 'nunique',
        'sentiment_score': 'mean',
        'is_positive': 'sum',
        'is_negative': 'sum',
        'confidence': 'mean'
    }).reset_index()
    
    category_stats.columns = ['category', 'review_count', 'product_count', 
                               'avg_sentiment', 'positive_count', 'negative_count', 
                               'avg_confidence']
    
    # Calculate percentages
    category_stats['pct_positive'] = (
        category_stats['positive_count'] / category_stats['review_count'] * 100
    )
    category_stats['pct_negative'] = (
        category_stats['negative_count'] / category_stats['review_count'] * 100
    )
    
    return category_stats.sort_values('review_count', ascending=False)


def export_to_excel(df, filepath, sheet_name='ABSA Results'):
    """
    Export DataFrame to Excel with formatting.
    
    Args:
        df: DataFrame to export
        filepath: Output file path
        sheet_name: Excel sheet name
    """
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"✓ Exported {len(df)} rows to {filepath}")


def format_percentage(value):
    """Format value as percentage with 1 decimal place."""
    return f"{value:.1f}%"


def format_sentiment_score(value):
    """Format sentiment score with 2 decimal places."""
    return f"{value:.2f}"


def truncate_text(text, max_length=50):
    """Truncate text to max length with ellipsis."""
    if len(text) > max_length:
        return text[:max_length-3] + "..."
    return text
