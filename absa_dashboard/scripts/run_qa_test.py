import sys
sys.path.append('.')
from genai_client import qa_for_product
import pandas as pd

# small synthetic dataset
records = [
    {'parent_asin':'TESTASIN','review_id':'r1','text':'Battery lasts 10 hours on a full charge','date':'2023-10-01'},
    {'parent_asin':'TESTASIN','review_id':'r2','text':'Battery drains quickly after 6 months of use','date':'2023-11-01'},
    {'parent_asin':'TESTASIN','review_id':'r3','text':'Great battery life and fast charging','date':'2023-09-15'},
]

df = pd.DataFrame(records)
# Force TF-IDF reranker fallback and disable local generation for a fast, offline smoke test
import genai_client as g
g.CrossEncoder = None
g._hf_pipeline = None

resp = qa_for_product(df, 'TESTASIN', 'What do customers say about battery life?', top_k=3, force_refresh=True)
import json
print(json.dumps(resp, indent=2, ensure_ascii=False))
