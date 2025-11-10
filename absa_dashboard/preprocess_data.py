"""
ABSA Data Preprocessing Module
================================
This module handles aspect term normalization, DataFrame merging, and data quality validation.

Key Functions:
- normalize_aspect_terms(): Standardizes aspect variations (plurals, synonyms)
- merge_dataframes(): Joins ABSA results with product metadata
- clean_merged_data(): Final validation and derived field creation
- preprocess_pipeline(): Full preprocessing workflow
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def normalize_aspect_terms(df):
    """
    Normalize aspect terms to canonical forms.
    
    Handles:
    - Plural/singular variations (color/colors)
    - Synonyms (smell/scent/fragrance)
    - Case inconsistencies (Color/color)
    
    Args:
        df: DataFrame with 'aspect_term' column
        
    Returns:
        DataFrame with new 'aspect_term_normalized' column
    """
    # Comprehensive aspect normalization mapping
    aspect_mapping = {
        # Color variations
        'color': 'color',
        'colors': 'color',
        'colour': 'color',
        'colours': 'color',
        'shade': 'color',
        'shades': 'color',
        'tint': 'color',
        
        # Quality variations
        'quality': 'quality',
        'qualities': 'quality',
        
        # Smell variations
        'smell': 'smell',
        'smells': 'smell',
        'scent': 'smell',
        'scents': 'smell',
        'fragrance': 'smell',
        'fragrances': 'smell',
        'aroma': 'smell',
        'aromas': 'smell',
        'odor': 'smell',
        'odour': 'smell',
        
        # Price variations
        'price': 'price',
        'prices': 'price',
        'cost': 'price',
        'costs': 'price',
        'pricing': 'price',
        'value': 'price',
        
        # Packaging variations
        'package': 'packaging',
        'packages': 'packaging',
        'packaging': 'packaging',
        'container': 'packaging',
        'containers': 'packaging',
        'box': 'packaging',
        'boxes': 'packaging',
        
        # Texture variations
        'texture': 'texture',
        'textures': 'texture',
        'consistency': 'texture',
        'feel': 'texture',
        
        # Size variations
        'size': 'size',
        'sizes': 'size',
        'amount': 'size',
        'quantity': 'size',
        
        # Bottle variations
        'bottle': 'bottle',
        'bottles': 'bottle',
        'jar': 'bottle',
        'jars': 'bottle',
        'tube': 'bottle',
        'tubes': 'bottle',
        
        # Product variations
        'product': 'product',
        'products': 'product',
        'item': 'product',
        'items': 'product',
        
        # Use/Usage variations
        'use': 'usage',
        'uses': 'usage',
        'usage': 'usage',
        'application': 'usage',
        'applications': 'usage',
        
        # Skin variations
        'skin': 'skin',
        'skins': 'skin',
        'complexion': 'skin',
        
        # Hair variations
        'hair': 'hair',
        'hairs': 'hair',
        
        # Effect variations
        'effect': 'effect',
        'effects': 'effect',
        'result': 'effect',
        'results': 'effect',
        
        # Ingredient variations
        'ingredient': 'ingredient',
        'ingredients': 'ingredient',
        'formula': 'ingredient',
        'formulation': 'ingredient',
        
        # Brand variations
        'brand': 'brand',
        'brands': 'brand',
        'company': 'brand',
        
        # Delivery variations
        'delivery': 'delivery',
        'shipping': 'delivery',
        'shipment': 'delivery',
        
        # Lasting variations
        'lasting': 'lasting',
        'longevity': 'lasting',
        'duration': 'lasting',
        'long-lasting': 'lasting',
        'longlasting': 'lasting',
    }
    
    # Apply normalization (lowercase + mapping)
    df['aspect_term_normalized'] = df['aspect_term'].str.lower().str.strip()
    df['aspect_term_normalized'] = df['aspect_term_normalized'].map(aspect_mapping)
    
    # Keep original (lowercased) for unmapped terms
    df['aspect_term_normalized'] = df['aspect_term_normalized'].fillna(
        df['aspect_term'].str.lower().str.strip()
    )
    
    print(f"✓ Normalized {df['aspect_term'].nunique()} unique aspects to {df['aspect_term_normalized'].nunique()} canonical forms")
    
    return df


def merge_dataframes(df_absa, df_products):
    """
    Merge ABSA results with product metadata.
    
    Args:
        df_absa: DataFrame with ABSA results (must have 'parent_asin')
        df_products: DataFrame with product metadata (must have 'parent_asin')
        
    Returns:
        Merged DataFrame with '_review' and '_product' suffixes for duplicate columns
    """
    print("\n--- Merging DataFrames ---")
    
    # Ensure parent_asin is consistent type (string)
    df_absa['parent_asin'] = df_absa['parent_asin'].astype(str).str.strip()
    df_products['parent_asin'] = df_products['parent_asin'].astype(str).str.strip()
    
    print(f"ABSA DataFrame: {len(df_absa)} rows, {df_absa['parent_asin'].nunique()} unique products")
    print(f"Product DataFrame: {len(df_products)} rows, {df_products['parent_asin'].nunique()} unique products")
    
    # Perform left join (keep all ABSA results)
    df_merged = df_absa.merge(
        df_products,
        on='parent_asin',
        how='left',
        suffixes=('_review', '_product')
    )
    
    # Add metadata coverage flag
    df_merged['has_metadata'] = df_merged['title'].notna()
    
    # Print merge statistics
    matched_products = df_merged['has_metadata'].sum()
    total_reviews = len(df_merged)
    coverage_pct = (matched_products / total_reviews * 100) if total_reviews > 0 else 0
    
    print(f"✓ Merged successfully: {len(df_merged)} rows")
    print(f"  - Reviews with product metadata: {matched_products} ({coverage_pct:.1f}%)")
    print(f"  - Reviews without metadata: {total_reviews - matched_products}")
    
    return df_merged


def clean_merged_data(df, confidence_threshold=0.5):
    """
    Final data cleaning and validation.
    
    Args:
        df: Merged DataFrame with aspect_term_normalized column
        confidence_threshold: Minimum confidence score (default: 0.5)
        
    Returns:
        Cleaned DataFrame with derived fields
    """
    print("\n--- Cleaning Merged Data ---")
    initial_count = len(df)
    
    # Remove duplicates (same review_id + normalized aspect_term)
    df = df.drop_duplicates(subset=['review_id', 'aspect_term_normalized'], keep='first')
    print(f"✓ Removed {initial_count - len(df)} duplicate review-aspect pairs")
    
    # Filter out low-confidence predictions
    before_filter = len(df)
    df = df[df['confidence'] >= confidence_threshold].copy()
    print(f"✓ Filtered {before_filter - len(df)} low-confidence predictions (threshold: {confidence_threshold})")
    
    # Remove aspects that are too generic or meaningless
    exclude_aspects = [
        'thing', 'things', 'stuff', 'something', 'anything', 'everything', 
        'it', 'this', 'that', 'these', 'those', 'one', 'ones'
    ]
    before_generic = len(df)
    df = df[~df['aspect_term_normalized'].isin(exclude_aspects)].copy()
    print(f"✓ Removed {before_generic - len(df)} generic/meaningless aspects")
    
    # Add derived fields
    print("\n--- Adding Derived Fields ---")
    
    # Sentiment score: positive confidence - negative confidence
    def calc_sentiment_score(row):
        if row['sentiment'] == 'Positive':
            return row['confidence']
        elif row['sentiment'] == 'Negative':
            return -row['confidence']
        else:  # Neutral
            return 0
    
    df['sentiment_score'] = df.apply(calc_sentiment_score, axis=1)
    print("✓ Added sentiment_score column")
    
    # Temporal aggregation fields
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['year_month'] = df['timestamp'].dt.to_period('M')
    df['week'] = df['timestamp'].dt.to_period('W')
    df['date'] = df['timestamp'].dt.date
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.day_name()
    print("✓ Added temporal fields (year_month, week, date, year, month, day_of_week)")
    
    # Sentiment category flag
    df['is_positive'] = (df['sentiment'] == 'Positive').astype(int)
    df['is_negative'] = (df['sentiment'] == 'Negative').astype(int)
    df['is_neutral'] = (df['sentiment'] == 'Neutral').astype(int)
    print("✓ Added sentiment flags")
    
    print(f"\n✓ Final cleaned dataset: {len(df)} rows")
    return df


def generate_normalization_report(df_original, df_normalized):
    """
    Generate a report showing aspect normalization statistics.
    
    Args:
        df_original: Original DataFrame (before normalization)
        df_normalized: Normalized DataFrame (after normalization)
        
    Returns:
        Dictionary with normalization statistics
    """
    print("\n" + "="*60)
    print("ASPECT NORMALIZATION REPORT")
    print("="*60)
    
    # Original vs normalized counts
    original_aspects = df_original['aspect_term'].str.lower().nunique()
    normalized_aspects = df_normalized['aspect_term_normalized'].nunique()
    reduction = original_aspects - normalized_aspects
    reduction_pct = (reduction / original_aspects * 100) if original_aspects > 0 else 0
    
    print(f"\n📊 Overall Statistics:")
    print(f"  - Original unique aspects: {original_aspects}")
    print(f"  - Normalized unique aspects: {normalized_aspects}")
    print(f"  - Reduction: {reduction} aspects ({reduction_pct:.1f}%)")
    
    # Top normalized aspects
    print(f"\n🔝 Top 20 Normalized Aspects:")
    top_aspects = df_normalized['aspect_term_normalized'].value_counts().head(20)
    for i, (aspect, count) in enumerate(top_aspects.items(), 1):
        pct = count / len(df_normalized) * 100
        print(f"  {i:2d}. {aspect:20s} - {count:6d} mentions ({pct:5.2f}%)")
    
    # Mapping statistics (how many variants collapsed)
    aspect_groups = df_normalized.groupby('aspect_term_normalized')['aspect_term'].nunique().sort_values(ascending=False)
    multi_variant_aspects = aspect_groups[aspect_groups > 1].head(10)
    
    if len(multi_variant_aspects) > 0:
        print(f"\n🔄 Top Aspects with Multiple Variants:")
        for aspect, variant_count in multi_variant_aspects.items():
            variants = df_normalized[df_normalized['aspect_term_normalized'] == aspect]['aspect_term'].unique()
            variants_str = ', '.join([f'"{v}"' for v in variants[:5]])
            if len(variants) > 5:
                variants_str += f", ... (+{len(variants)-5} more)"
            print(f"  - {aspect}: {variant_count} variants → {variants_str}")
    
    # Unmapped aspects (those that weren't in the mapping dictionary)
    # These are aspects where normalized == original (lowercased)
    if 'aspect_term' in df_normalized.columns:
        df_normalized['_original_lower'] = df_normalized['aspect_term'].str.lower().str.strip()
        unmapped_mask = df_normalized['aspect_term_normalized'] == df_normalized['_original_lower']
        unmapped = df_normalized[unmapped_mask]['aspect_term_normalized'].value_counts().head(20)
        df_normalized.drop('_original_lower', axis=1, inplace=True)
        
        if len(unmapped) > 0:
            print(f"\n⚠️  Top 20 Unmapped Aspects (may need attention):")
            for i, (aspect, count) in enumerate(unmapped.items(), 1):
                print(f"  {i:2d}. {aspect:20s} - {count:6d} mentions")
    
    print("\n" + "="*60 + "\n")
    
    return {
        'original_aspects': original_aspects,
        'normalized_aspects': normalized_aspects,
        'reduction': reduction,
        'reduction_pct': reduction_pct,
        'top_aspects': top_aspects.to_dict(),
        'multi_variant_aspects': multi_variant_aspects.to_dict()
    }


def preprocess_pipeline(df_absa, df_products, confidence_threshold=0.5, output_path=None):
    """
    Complete preprocessing pipeline.
    
    Args:
        df_absa: ABSA results DataFrame
        df_products: Product metadata DataFrame
        confidence_threshold: Minimum confidence score (default: 0.5)
        output_path: Optional path to save preprocessed data
        
    Returns:
        Tuple of (preprocessed_df, normalization_report)
    """
    print("\n" + "="*60)
    print("STARTING ABSA DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # Keep copy for report
    df_original = df_absa.copy()
    
    # Step 1: Normalize aspect terms
    print("\n[1/4] Normalizing aspect terms...")
    df_absa = normalize_aspect_terms(df_absa)
    
    # Step 2: Merge with product metadata
    print("\n[2/4] Merging with product metadata...")
    df_merged = merge_dataframes(df_absa, df_products)
    
    # Step 3: Clean and validate
    print("\n[3/4] Cleaning and validating data...")
    df_cleaned = clean_merged_data(df_merged, confidence_threshold)
    
    # Step 4: Generate normalization report
    print("\n[4/4] Generating normalization report...")
    report = generate_normalization_report(df_original, df_cleaned)
    
    # Save preprocessed data if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.csv':
            df_cleaned.to_csv(output_path, index=False)
        elif output_path.suffix == '.parquet':
            df_cleaned.to_parquet(output_path, index=False)
        elif output_path.suffix in ['.xlsx', '.xls']:
            df_cleaned.to_excel(output_path, index=False)
        
        print(f"\n✓ Saved preprocessed data to: {output_path}")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60 + "\n")
    
    return df_cleaned, report


# Main execution
if __name__ == "__main__":
    print("ABSA Data Preprocessing Module")
    print("-" * 60)
    print("This module provides data preprocessing utilities for ABSA analysis.")
    print("\nUsage:")
    print("  from preprocess_data import preprocess_pipeline")
    print("  df_cleaned, report = preprocess_pipeline(df_absa, df_products)")
    print("\nFor full pipeline, run from Jupyter notebook or import functions.")
