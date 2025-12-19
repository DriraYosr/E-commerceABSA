"""Small test harness to load google/flan-t5-base and run a sample generation.

This will download the model on first run. It's expected to be larger and slower on CPU.
Run from the repo root with your Python environment active.
"""

from transformers import pipeline
import os

MODEL = os.environ.get('LOCAL_GEN_MODEL', 'google/flan-t5-base')

prompt = (
    "You are a concise, evidence-based assistant. Use provided snippets only and cite them as [n].\n"
    "If insufficient information is available, reply 'Insufficient evidence'. Do not hallucinate.\n\n"
    "QUESTION: Does the product charge quickly and hold a charge?\n\n"
    "SNIPPETS:\n"
    "[0] review_id=R1 date=2021-01-01: Battery charges quickly, fast charging works well for me.\n"
    "[1] review_id=R2 date=2021-02-01: Battery life is average; needs daily charging with heavy use.\n"
    "[2] review_id=R3 date=2021-03-01: Excellent battery, lasts all day and charges overnight.\n"
)

print(f"Loading pipeline for model: {MODEL} (this may download weights on first run)")

gen = pipeline('text2text-generation', model=MODEL, device=-1)
print('Model loaded — running generation...')

out = gen(prompt, max_new_tokens=64, do_sample=False)
print('\n=== Generation result ===')
print(out[0].get('generated_text') or str(out[0]))
