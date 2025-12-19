#!/usr/bin/env python3
"""
ABSA Inference Script - Batch Processing for Amazon Reviews

This script performs Aspect-Based Sentiment Analysis (ABSA) on Amazon Beauty reviews
using PyABSA's pretrained ATEPC-LCF model. It processes reviews by year and saves
results to monthly output files.

Usage:
    python run_inference.py --year 2020 --output absa_output/
    python run_inference.py --year 2020 --month 11 --output absa_output/Nov20/
    python run_inference.py --help

Requirements:
    - PyABSA 2.3.4
    - pandas 2.1.3
    - transformers 4.35.0
    - Data file: data/All_Beauty.jsonl

Author: Amira Mostafa, Yosr Drira
Date: December 2025
"""

import os
import sys
import argparse
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

try:
    from pyabsa import AspectTermExtraction as ATEPC
    from transformers import DebertaV2TokenizerFast
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install dependencies: pip install pyabsa transformers")
    sys.exit(1)


def load_data(review_path: str, year: int, month: Optional[int] = None) -> pd.DataFrame:
    """
    Load and filter Amazon reviews by year and optionally by month.
    
    Args:
        review_path: Path to All_Beauty.jsonl file
        year: Year to filter reviews (e.g., 2020)
        month: Optional month to filter (1-12)
    
    Returns:
        Filtered DataFrame with reviews
    """
    print(f"Loading reviews from {review_path}...")
    
    if not os.path.exists(review_path):
        raise FileNotFoundError(f"Review file not found: {review_path}")
    
    # Load reviews
    df = pd.read_json(review_path, lines=True)
    
    # Convert timestamp to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Filter by year
    df = df[df['timestamp'].dt.year == year]
    print(f"Found {len(df)} reviews for year {year}")
    
    # Filter by month if specified
    if month is not None:
        df = df[df['timestamp'].dt.month == month]
        print(f"Filtered to {len(df)} reviews for month {month}")
    
    if len(df) == 0:
        raise ValueError(f"No reviews found for year={year}, month={month}")
    
    return df


def initialize_model(checkpoint: str = "english") -> ATEPC.AspectExtractor:
    """
    Initialize PyABSA ATEPC model with pretrained checkpoint.
    
    Args:
        checkpoint: Model checkpoint name ("english", "multilingual", or "chinese")
    
    Returns:
        Initialized AspectExtractor
    """
    print(f"Loading PyABSA ATEPC model (checkpoint: {checkpoint})...")
    
    try:
        # Load pretrained model
        extractor = ATEPC.AspectExtractor(checkpoint)
        print("Model loaded successfully")
        return extractor
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to download checkpoint...")
        extractor = ATEPC.AspectExtractor(checkpoint, auto_device=True)
        return extractor


def run_inference_batch(
    extractor: ATEPC.AspectExtractor,
    reviews: List[str],
    batch_size: int = 32,
    show_progress: bool = True
) -> List[Dict]:
    """
    Run ABSA inference on a batch of reviews.
    
    Args:
        extractor: Initialized AspectExtractor
        reviews: List of review texts
        batch_size: Number of reviews to process at once
        show_progress: Whether to print progress updates
    
    Returns:
        List of inference results (dicts with aspects, sentiments, confidence)
    """
    results = []
    total = len(reviews)
    
    print(f"Running inference on {total} reviews (batch_size={batch_size})...")
    
    for i in range(0, total, batch_size):
        batch = reviews[i:i + batch_size]
        
        try:
            # Run inference
            batch_results = extractor.predict(
                batch,
                save_result=False,
                print_result=False,
                ignore_error=True
            )
            results.extend(batch_results)
            
            if show_progress and (i + batch_size) % 1000 == 0:
                progress = min(i + batch_size, total)
                print(f"  Processed {progress}/{total} reviews ({progress/total*100:.1f}%)")
        
        except Exception as e:
            print(f"Warning: Batch {i//batch_size} failed: {e}")
            # Append empty results for failed batch
            results.extend([{"error": str(e)}] * len(batch))
    
    print(f"Inference complete. Processed {len(results)} reviews.")
    return results


def save_results(
    df: pd.DataFrame,
    results: List[Dict],
    output_dir: str,
    year: int,
    month: Optional[int] = None
):
    """
    Save ABSA results to CSV file.
    
    Args:
        df: Original DataFrame with review metadata
        results: ABSA inference results
        output_dir: Output directory
        year: Year for filename
        month: Optional month for filename
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    if month is not None:
        month_name = datetime(year, month, 1).strftime("%b%y")
        output_file = os.path.join(output_dir, f"beauty_absa_results_{month_name}.csv")
    else:
        output_file = os.path.join(output_dir, f"beauty_absa_results_{year}.csv")
    
    # Add results to DataFrame
    df_results = df.copy()
    df_results['absa_results'] = results
    
    # Extract aspects, sentiments, confidence from results
    aspects_list = []
    sentiments_list = []
    confidence_list = []
    
    for result in results:
        if isinstance(result, dict) and 'aspect' in result:
            aspects_list.append(result.get('aspect', []))
            sentiments_list.append(result.get('sentiment', []))
            confidence_list.append(result.get('confidence', []))
        else:
            aspects_list.append([])
            sentiments_list.append([])
            confidence_list.append([])
    
    df_results['aspects'] = aspects_list
    df_results['sentiments'] = sentiments_list
    df_results['confidence'] = confidence_list
    
    # Save to CSV
    print(f"Saving results to {output_file}...")
    df_results.to_csv(output_file, index=False)
    print(f"Results saved successfully. Shape: {df_results.shape}")
    
    # Print summary statistics
    total_aspects = sum(len(a) for a in aspects_list)
    avg_aspects = total_aspects / len(aspects_list) if aspects_list else 0
    print(f"\nSummary:")
    print(f"  Total reviews: {len(df_results)}")
    print(f"  Total aspects extracted: {total_aspects}")
    print(f"  Average aspects per review: {avg_aspects:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run ABSA inference on Amazon Beauty reviews",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all reviews for 2020
  python run_inference.py --year 2020 --output absa_output/
  
  # Process only November 2020
  python run_inference.py --year 2020 --month 11 --output absa_output/Nov20/
  
  # Use custom data path
  python run_inference.py --year 2020 --data-dir /path/to/data --output results/
        """
    )
    
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year to filter reviews (e.g., 2020)"
    )
    
    parser.add_argument(
        "--month",
        type=int,
        default=None,
        choices=range(1, 13),
        help="Optional: Month to filter (1-12)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing All_Beauty.jsonl (default: data/)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="absa_output",
        help="Output directory for results (default: absa_output/)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="english",
        choices=["english", "multilingual", "chinese"],
        help="PyABSA model checkpoint (default: english)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)"
    )
    
    args = parser.parse_args()
    
    # Construct paths
    review_path = os.path.join(args.data_dir, "All_Beauty.jsonl")
    
    try:
        # Load data
        df = load_data(review_path, args.year, args.month)
        
        # Initialize model
        extractor = initialize_model(args.checkpoint)
        
        # Run inference
        reviews = df['text'].tolist()
        results = run_inference_batch(
            extractor,
            reviews,
            batch_size=args.batch_size,
            show_progress=True
        )
        
        # Save results
        save_results(df, results, args.output, args.year, args.month)
        
        print("\n✓ Inference completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
