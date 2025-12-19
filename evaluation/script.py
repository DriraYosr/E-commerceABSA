import pandas as pd

# Load predictions and ground truth files
predictions_path = r'predictions.csv'
gt_path = r'GT.csv'
output_path = r'predictions_with_GT.csv'

# Read CSVs
df_pred = pd.read_csv(predictions_path)
df_gt = pd.read_csv(gt_path)

# Merge on review_id (assuming this is the unique key)
df_merged = pd.merge(df_pred, df_gt[['review_id', 'aspects_json']], on='review_id', how='left', suffixes=('', '_GT'))

# Rename the ground truth aspects column
df_merged = df_merged.rename(columns={'aspects_json_y': 'aspects_json_GT'})

# Save to new CSV
df_merged.to_csv(output_path, index=False)

print(f"Merged file with ground truth saved to: {output_path}")
