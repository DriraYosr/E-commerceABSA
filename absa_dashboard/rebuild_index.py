"""
Rebuild FAISS index and metadata from scratch
Run this script when you need to regenerate the embeddings
"""
import pandas as pd
from pathlib import Path
from genai_client import build_and_persist_index, detect_text_column

# Load your data
print("Loading data...")
data_path = Path(__file__).parent.parent / 'data' / 'All_Beauty.jsonl'
df = pd.read_json(data_path, lines=True)

print(f"Loaded {len(df)} reviews")

# Detect text column
text_col = detect_text_column(df)
print(f"Text column: {text_col}")

# Build and persist index
print("\nBuilding FAISS index and metadata...")
print("This may take a few minutes...")

result = build_and_persist_index(df, text_col=text_col, overwrite=True)

print("\n" + "="*60)
print("✅ Index and Metadata Built Successfully!")
print("="*60)
print(f"Status: {result['status']}")
print(f"Index file: {result['index_file']}")
print(f"Metadata file: {result['metadata_file']}")
print(f"Total reviews indexed: {result['n']}")
print("="*60)
