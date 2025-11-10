"""
ABSA Alert System
==================
Automated detection of sentiment shifts, emerging concerns, and anomalies.

Features:
- Real-time sentiment drop detection
- Category-level monitoring
- Aspect-specific alerts
- Alert enrichment with product metadata
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import *
from utils import *


class AlertSystem:
    """
    Alert system for monitoring ABSA sentiment trends.
    """
    
    def __init__(self, df):
        """
        Initialize alert system with ABSA data.
        
        Args:
            df: Preprocessed ABSA DataFrame
        """
        self.df = df
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df['date'] = pd.to_datetime(self.df['date'])
        
    
    def detect_sentiment_drops(self, threshold=SENTIMENT_DROP_THRESHOLD, 
                                window_days=ROLLING_WINDOW_DAYS, 
                                min_reviews=MIN_REVIEWS_FOR_ALERT):
        """
        Detect products with significant sentiment drops.
        
        Args:
            threshold: Minimum percentage drop (default from config)
            window_days: Rolling window in days
            min_reviews: Minimum reviews to trigger alert
            
        Returns:
            DataFrame with sentiment drop alerts
        """
        print(f"🔍 Detecting sentiment drops (threshold: {threshold*100:.0f}%, window: {window_days} days)...")
        
        alerts = []
        products = self.df['parent_asin'].unique()
        
        for product in products:
            product_data = self.df[self.df['parent_asin'] == product].sort_values('date')
            
            if len(product_data) < min_reviews:
                continue
            
            # Calculate rolling sentiment
            product_data = product_data.set_index('date')
            rolling = product_data['sentiment_score'].rolling(
                f'{window_days}D', 
                min_periods=1
            ).mean()
            
            if len(rolling) < 2:
                continue
            
            # Compare recent vs previous
            recent_sentiment = rolling.iloc[-1]
            previous_idx = -(window_days + 1) if len(rolling) > window_days else 0
            previous_sentiment = rolling.iloc[previous_idx]
            
            if previous_sentiment > 0:
                change_pct = (recent_sentiment - previous_sentiment) / abs(previous_sentiment)
                
                if change_pct < -threshold:
                    # Get product metadata
                    product_info = product_data.iloc[-1]
                    
                    # Get affected aspects
                    recent_date = rolling.index[-1]
                    cutoff_date = recent_date - timedelta(days=window_days)
                    recent_data = product_data[product_data.index >= cutoff_date]
                    top_negative_aspects = recent_data[recent_data['sentiment'] == 'Negative']['aspect_term_normalized'].value_counts().head(3)
                    
                    alerts.append({
                        'alert_type': 'Sentiment Drop',
                        'severity': 'High' if change_pct < -0.5 else 'Medium',
                        'product_asin': product,
                        'product_title': product_info.get('title', product),
                        'category': product_info.get('main_category', 'Unknown'),
                        'sentiment_change': change_pct,
                        'recent_sentiment': recent_sentiment,
                        'previous_sentiment': previous_sentiment,
                        'alert_date': recent_date,
                        'review_count_period': len(recent_data),
                        'top_negative_aspects': ', '.join(top_negative_aspects.index.tolist())
                    })
        
        alerts_df = pd.DataFrame(alerts)
        if len(alerts_df) > 0:
            alerts_df = alerts_df.sort_values('sentiment_change')
        
        print(f"✓ Found {len(alerts_df)} sentiment drop alerts")
        return alerts_df
    
    
    def detect_category_trends(self, min_reviews=50):
        """
        Monitor sentiment trends at category level.
        
        Args:
            min_reviews: Minimum reviews per category
            
        Returns:
            DataFrame with category-level trends
        """
        print("🔍 Analyzing category-level trends...")
        
        if 'main_category' not in self.df.columns:
            print("⚠️  No category information available")
            return pd.DataFrame()
        
        # Calculate recent vs historical sentiment by category
        cutoff_date = self.df['date'].max() - timedelta(days=30)
        
        recent_data = self.df[self.df['date'] > cutoff_date]
        historical_data = self.df[self.df['date'] <= cutoff_date]
        
        category_trends = []
        
        for category in self.df['main_category'].dropna().unique():
            recent_cat = recent_data[recent_data['main_category'] == category]
            historical_cat = historical_data[historical_data['main_category'] == category]
            
            if len(recent_cat) < min_reviews or len(historical_cat) < min_reviews:
                continue
            
            recent_sentiment = recent_cat['sentiment_score'].mean()
            historical_sentiment = historical_cat['sentiment_score'].mean()
            
            change = (recent_sentiment - historical_sentiment) / abs(historical_sentiment) if historical_sentiment != 0 else 0
            
            category_trends.append({
                'category': category,
                'recent_sentiment': recent_sentiment,
                'historical_sentiment': historical_sentiment,
                'change': change,
                'trend': 'Improving' if change > 0.1 else 'Declining' if change < -0.1 else 'Stable',
                'recent_review_count': len(recent_cat),
                'historical_review_count': len(historical_cat),
                'recent_pct_negative': (recent_cat['sentiment'] == 'Negative').sum() / len(recent_cat) * 100
            })
        
        trends_df = pd.DataFrame(category_trends)
        if len(trends_df) > 0:
            trends_df = trends_df.sort_values('change')
        
        print(f"✓ Analyzed {len(trends_df)} categories")
        return trends_df
    
    
    def detect_emerging_aspects(self, lookback_days=30, min_mentions=5):
        """
        Identify new aspects appearing in recent reviews.
        
        Args:
            lookback_days: Days to look back for "recent"
            min_mentions: Minimum mentions to flag
            
        Returns:
            DataFrame with emerging aspects
        """
        print(f"🔍 Detecting emerging aspects (past {lookback_days} days)...")
        
        cutoff_date = self.df['date'].max() - timedelta(days=lookback_days)
        older_date = cutoff_date - timedelta(days=lookback_days)
        
        recent_data = self.df[self.df['date'] > cutoff_date]
        older_data = self.df[
            (self.df['date'] > older_date) & 
            (self.df['date'] <= cutoff_date)
        ]
        
        recent_aspects = set(recent_data['aspect_term_normalized'].unique())
        older_aspects = set(older_data['aspect_term_normalized'].unique())
        
        new_aspects = recent_aspects - older_aspects
        
        emerging = []
        
        for aspect in new_aspects:
            aspect_data = recent_data[recent_data['aspect_term_normalized'] == aspect]
            
            if len(aspect_data) < min_mentions:
                continue
            
            # Calculate sentiment for this aspect
            avg_sentiment = aspect_data['sentiment_score'].mean()
            pct_negative = (aspect_data['sentiment'] == 'Negative').sum() / len(aspect_data) * 100
            
            # Get top products mentioning this aspect
            top_products = aspect_data['parent_asin'].value_counts().head(3)
            
            emerging.append({
                'aspect': aspect,
                'mention_count': len(aspect_data),
                'avg_sentiment': avg_sentiment,
                'pct_negative': pct_negative,
                'unique_products': aspect_data['parent_asin'].nunique(),
                'top_products': ', '.join([
                    aspect_data[aspect_data['parent_asin'] == asin]['title'].iloc[0] 
                    if 'title' in aspect_data.columns 
                    else asin 
                    for asin in top_products.index[:3]
                ]),
                'first_seen': aspect_data['date'].min()
            })
        
        emerging_df = pd.DataFrame(emerging)
        if len(emerging_df) > 0:
            emerging_df = emerging_df.sort_values('mention_count', ascending=False)
        
        print(f"✓ Found {len(emerging_df)} emerging aspects")
        return emerging_df
    
    
    def detect_rating_sentiment_divergence(self, divergence_threshold=1.5):
        """
        Find products where sentiment diverges from product rating.
        
        Args:
            divergence_threshold: Minimum divergence to flag
            
        Returns:
            DataFrame with divergent products
        """
        print("🔍 Detecting rating-sentiment divergence...")
        
        if 'average_rating' not in self.df.columns:
            print("⚠️  No rating information available")
            return pd.DataFrame()
        
        # Normalize sentiment score to 1-5 scale for comparison
        # Sentiment score is -1 to +1, convert to 1-5
        self.df['normalized_sentiment'] = ((self.df['sentiment_score'] + 1) / 2) * 4 + 1
        
        divergent = []
        
        for product in self.df['parent_asin'].unique():
            product_data = self.df[self.df['parent_asin'] == product]
            
            if len(product_data) < MIN_REVIEWS_FOR_ANALYSIS:
                continue
            
            product_rating = product_data['average_rating'].iloc[0]
            avg_sentiment = product_data['normalized_sentiment'].mean()
            
            divergence = abs(product_rating - avg_sentiment)
            
            if divergence >= divergence_threshold:
                divergent.append({
                    'product_asin': product,
                    'product_title': product_data['title'].iloc[0] if 'title' in product_data.columns else product,
                    'category': product_data['main_category'].iloc[0] if 'main_category' in product_data.columns else 'Unknown',
                    'product_rating': product_rating,
                    'sentiment_rating': avg_sentiment,
                    'divergence': divergence,
                    'review_count': len(product_data),
                    'interpretation': 'Rating higher than sentiment' if product_rating > avg_sentiment else 'Sentiment higher than rating'
                })
        
        divergent_df = pd.DataFrame(divergent)
        if len(divergent_df) > 0:
            divergent_df = divergent_df.sort_values('divergence', ascending=False)
        
        print(f"✓ Found {len(divergent_df)} divergent products")
        return divergent_df
    
    
    def generate_alert_report(self, output_path=None):
        """
        Generate comprehensive alert report with all checks.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Dictionary with all alert DataFrames
        """
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE ALERT REPORT")
        print("="*60 + "\n")
        
        report = {
            'sentiment_drops': self.detect_sentiment_drops(),
            'category_trends': self.detect_category_trends(),
            'emerging_aspects': self.detect_emerging_aspects(),
            'rating_divergence': self.detect_rating_sentiment_divergence()
        }
        
        # Summary
        print("\n" + "="*60)
        print("ALERT SUMMARY")
        print("="*60)
        print(f"Sentiment Drop Alerts: {len(report['sentiment_drops'])}")
        print(f"Category Trends: {len(report['category_trends'])}")
        print(f"Emerging Aspects: {len(report['emerging_aspects'])}")
        print(f"Rating Divergences: {len(report['rating_divergence'])}")
        print("="*60 + "\n")
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for sheet_name, df in report.items():
                    if len(df) > 0:
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"✓ Alert report saved to: {output_path}")
        
        return report


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("ABSA Alert System")
    print("-" * 60)
    
    # Load preprocessed data
    data_path = Path(DATA_DIR) / PREPROCESSED_DATA_FILE
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please run preprocess_data.py first.")
        sys.exit(1)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"✓ Loaded {len(df):,} reviews\n")
    
    # Initialize alert system
    alert_system = AlertSystem(df)
    
    # Generate report
    report = alert_system.generate_alert_report(
        output_path='exports/alert_report.xlsx'
    )
    
    print("\n✅ Alert system execution complete!")
