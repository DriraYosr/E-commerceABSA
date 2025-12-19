from pathlib import Path
p = Path(__file__).resolve().parents[1] / 'data' / 'preprocessed_data.parquet'
print(str(p))
print('exists=', p.exists())
