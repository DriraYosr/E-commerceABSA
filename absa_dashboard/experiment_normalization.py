# Quick Experiment: Understanding Aspect Normalization
# =====================================================
# Run this to see how normalization works!

import pandas as pd

# Sample data (like your reviews)
data = {
    'review_id': [1, 2, 3, 4, 5, 6],
    'aspect_term': ['color', 'colors', 'colour', 'smell', 'scent', 'fragrance'],
    'sentiment': ['Positive', 'Positive', 'Negative', 'Positive', 'Negative', 'Negative']
}

df = pd.DataFrame(data)

print("=" * 50)
print("BEFORE NORMALIZATION")
print("=" * 50)
print(df)
print(f"\nUnique aspects: {df['aspect_term'].nunique()}")
print(f"Aspect counts:\n{df['aspect_term'].value_counts()}")

# Apply normalization (like in preprocess_data.py)
aspect_mapping = {
    'color': 'color',
    'colors': 'color',
    'colour': 'color',
    'smell': 'smell',
    'scent': 'smell',
    'fragrance': 'smell'
}

df['aspect_normalized'] = df['aspect_term'].map(aspect_mapping)

print("\n" + "=" * 50)
print("AFTER NORMALIZATION")
print("=" * 50)
print(df)
print(f"\nUnique aspects: {df['aspect_normalized'].nunique()}")
print(f"Aspect counts:\n{df['aspect_normalized'].value_counts()}")

# Now we can analyze properly!
print("\n" + "=" * 50)
print("ANALYSIS (only possible after normalization)")
print("=" * 50)

# Sentiment by aspect
for aspect in df['aspect_normalized'].unique():
    aspect_data = df[df['aspect_normalized'] == aspect]
    positive_count = (aspect_data['sentiment'] == 'Positive').sum()
    negative_count = (aspect_data['sentiment'] == 'Negative').sum()
    
    print(f"\nAspect: {aspect}")
    print(f"  Total mentions: {len(aspect_data)}")
    print(f"  Positive: {positive_count}")
    print(f"  Negative: {negative_count}")
    print(f"  Sentiment ratio: {positive_count / (positive_count + negative_count):.2%}")
