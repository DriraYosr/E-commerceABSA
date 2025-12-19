"""Run structured-prompt experiments against the local generator (google/flan-t5-base).

This script loads the local text2text pipeline and runs several prompt variants
against the same question + snippets. It prints raw outputs for comparison.

Run from repo root:
    python absa_dashboard\scripts\structured_prompt_experiments.py
"""

from transformers import pipeline
import os

MODEL = os.environ.get('LOCAL_GEN_MODEL', 'google/flan-t5-base')

QUESTION = "What do users say about the product's durability?"
SNIPPETS = [
    "The build feels very sturdy; I've dropped it a few times and nothing broke.",
    "After a month the finish started to chip, but the frame is still solid.",
    "Excellent durability — been using daily for a year with no issues.",
    "Materials feel cheap and some seams split after light use.",
]

PROMPTS = {
    'json_strict': (
        "You are a concise, evidence-based assistant. Use only the provided snippets as evidence. "
        "Return output EXACTLY as JSON with keys: summary (one sentence), pros (array of short items), "
        "cons (array of short items), evidence (array of {index:int, excerpt:str}). "
        "If insufficient evidence, set summary='Insufficient evidence' and empty lists.\n\n"
        "QUESTION: {q}\n\nSNIPPETS:\n{sn}"
    ),

    'bullets_labeled': (
        "Answer concisely. Use the snippets only. Provide:\n"
        "SUMMARY:\n- one short sentence\n\nPROS:\n- bullet list (short)\n\nCONS:\n- bullet list (short)\n\nEVIDENCE: list snippet indices and short excerpt.\n\n"
        "QUESTION: {q}\n\nSNIPPETS:\n{sn}"
    ),

    'short_summary_then_lists': (
        "Provide a 1-line summary (<=20 words) followed by 'Pros:' and 'Cons:' each with up to 3 short bullets. "
        "Cite supporting snippet indices in parentheses after each bullet like (0) or (2).\n\n"
        "QUESTION: {q}\n\nSNIPPETS:\n{sn}"
    ),

    'table_like': (
        "Provide a compact table-like response with columns: Aspect | Verdict | EvidenceIndex (comma-separated). Keep it brief.\n\n"
        "QUESTION: {q}\n\nSNIPPETS:\n{sn}"
    ),
}

# Prepare snippets as numbered list
sn_text = "\n".join([f"[{i}] {s}" for i, s in enumerate(SNIPPETS)])

print(f"Loading model: {MODEL} (this may take a few seconds to load into memory)")

gen = pipeline('text2text-generation', model=MODEL, device=-1)
print('Model ready. Running experiments...\n')

for name, tpl in PROMPTS.items():
    prompt = tpl.format(q=QUESTION, sn=sn_text)
    print('---')
    print(f'PROMPT VARIANT: {name}\n')
    print('Prompt (truncated):')
    print('\n'.join(prompt.splitlines()[:8]))
    print('...')
    try:
        out = gen(prompt, max_new_tokens=256, do_sample=False)[0]
        text = out.get('generated_text') or out.get('summary_text') or str(out)
        print('\n=== OUTPUT ===')
        print(text.strip())
    except Exception as e:
        print(f'Error running generation for {name}: {e}')

print('\nAll experiments complete.')
