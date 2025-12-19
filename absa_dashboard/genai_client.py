"""
Minimal GenAI client for ABSA Dashboard
- Embeddings & retrieval using sentence-transformers + FAISS
- Optional OpenAI completion wrapper (uses OPENAI_API_KEY env var)

This is intentionally minimal: it provides a retrieval function and a thin
answer wrapper. It scrubs obvious PII before sending snippets to external APIs.
"""
from typing import List, Dict, Optional, Any
import os
import re
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from genai_cache import get_cached_answer, set_cached_answer

# Configure module logger (stream to STDERR when running app)
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

try:
    import importlib
    faiss = importlib.import_module('faiss')
except Exception:
    faiss = None

# Optional OpenAI
try:
    import openai
except Exception:
    openai = None

# Optional Google Gemini
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyADCg85g6qUdsisi_U_KpWyBfQoTl_5G4A')
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info(f"✅ Gemini configured successfully")
    else:
        logger.warning("⚠️  No Gemini API key found")
        genai = None
except Exception as e:
    logger.error(f"❌ Gemini import/configuration failed: {e}")
    genai = None

# Optional local generation via Hugging Face transformers (free to run locally)
try:
    from transformers import pipeline as _hf_pipeline
except Exception:
    _hf_pipeline = None

# Config defaults (kept small; dashboard's config.py contains project settings)
EMBEDDING_MODEL = os.environ.get('SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')
OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-3.5-turbo')
EMBEDDINGS_DIR = os.environ.get('EMBEDDINGS_DIR', 'embeddings')
INDEX_FILE = str(Path(EMBEDDINGS_DIR) / 'faiss.index')
METADATA_FILE = str(Path(EMBEDDINGS_DIR) / 'metadata.parquet')
CACHE_TTL_SECONDS = int(os.environ.get('GENAI_CACHE_TTL', 7 * 24 * 3600))

# Simple PII scrubbing patterns
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+")
PHONE_RE = re.compile(r"\+?\d[\d\-\s]{6,}\d")

# TF-IDF fallback for environments without sentence-transformers (keeps app runnable)
TFIDF_VECTORIZER = None
SVD_MODEL = None
TFIDF_MAX_FEATURES = int(os.environ.get('TFIDF_MAX_FEATURES', 1024))
SVD_DIM = int(os.environ.get('TFIDF_SVD_DIM', 384))
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
except Exception:
    TfidfVectorizer = None
    TruncatedSVD = None

# Cross-encoder (optional, used for reranking retrieved candidates)
CROSS_ENCODER = None
CROSS_ENCODER_MODEL = os.environ.get('CROSS_ENCODER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')


def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ['review_body', 'review_text', 'content', 'review', 'text', 'review_body_clean']
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: try any string column that's long enough
    for c in df.select_dtypes(include='object').columns:
        if df[c].dropna().astype(str).map(len).median() > 20:
            return c
    return None


def scrub_pii(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = EMAIL_RE.sub('[REDACTED_EMAIL]', text)
    text = PHONE_RE.sub('[REDACTED_PHONE]', text)
    return text


def sanitize_snippets_for_cache(snippets: List[Dict]) -> List[Dict]:
    """Convert pandas Timestamps and other non-JSON-serializable types to strings."""
    sanitized = []
    for snippet in snippets:
        clean_snippet = {}
        for key, value in snippet.items():
            # Convert pandas Timestamp to ISO string
            if pd.api.types.is_datetime64_any_dtype(type(value)) or hasattr(value, 'isoformat'):
                try:
                    clean_snippet[key] = value.isoformat() if hasattr(value, 'isoformat') else str(value)
                except Exception:
                    clean_snippet[key] = str(value)
            # Convert numpy types to native Python types
            elif hasattr(value, 'item'):
                clean_snippet[key] = value.item()
            else:
                clean_snippet[key] = value
        sanitized.append(clean_snippet)
    return sanitized


def parse_structured_json(text: str) -> Optional[Dict]:
    """Attempt to extract a JSON object from model output and parse it.

    Returns parsed dict on success or None on failure.
    """
    if not isinstance(text, str):
        return None
    # Quick attempt: whole text is JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find the largest braces-enclosed substring
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    return None


def parse_answer_evidence(text: str, prepared: List[str]) -> Optional[Dict]:
    """Parse human-friendly 'Answer:' / 'Evidence:' outputs into a structured dict.

    Returns a dict with keys: summary (one-line), pros (empty list), cons (empty list),
    evidence (array of {index:int, excerpt:str}) on success, otherwise None.
    """
    if not isinstance(text, str):
        return None

    # Look for sections 'Answer:' and 'Evidence:' (case-insensitive)
    m = re.search(r"Answer:\s*(.*?)\s*Evidence:\s*(.*)$", text, flags=re.I | re.S)
    if not m:
        # Try looser match: everything after 'Answer:' as answer, try to find citations
        if 'Answer:' in text:
            parts = text.split('Answer:', 1)[1].strip()
            # no explicit Evidence: section
            answer_text = parts.strip()
            evidence_text = ''
        else:
            return None
    else:
        answer_text = m.group(1).strip()
        evidence_text = m.group(2).strip()

    # Short summary: first line of answer_text
    summary = answer_text.splitlines()[0].strip() if answer_text else ''

    # Find cited indices anywhere in the output
    cited = re.findall(r"\[(\d+)\]", text)
    cited_idx = sorted(set(int(c) for c in cited if c.isdigit()))

    evidence = []
    for i in cited_idx:
        if 0 <= i < len(prepared):
            # prepared entries are like "[0] review_id=...: snippet"
            entry = prepared[i]
            # try to extract snippet text after the first colon
            if ':' in entry:
                _, snippet = entry.split(':', 1)
                excerpt = snippet.strip()
            else:
                excerpt = entry
            evidence.append({'index': i, 'excerpt': excerpt})

    return {'summary': summary, 'pros': [], 'cons': [], 'evidence': evidence}


def get_sentence_transformer(model_name: str = EMBEDDING_MODEL) -> Any:
    """Return a SentenceTransformer instance or None if unavailable.

    Note: we avoid raising here so callers can fall back to TF-IDF embeddings.
    """
    if SentenceTransformer is None:
        return None
    return SentenceTransformer(model_name)


def embed_texts(texts: List[str], model: Optional[Any] = None) -> np.ndarray:
    texts_clean = [str(t) for t in texts]

    # Use SentenceTransformer when available
    if model is None:
        model = get_sentence_transformer()

    if model is not None:
        embeddings = model.encode(texts_clean, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.astype('float32')

    # Fallback: TF-IDF + optional SVD to produce dense, fixed-size vectors
    if TfidfVectorizer is None:
        raise ImportError('Neither sentence-transformers nor scikit-learn are available for embeddings')

    global TFIDF_VECTORIZER, SVD_MODEL
    if TFIDF_VECTORIZER is None:
        TFIDF_VECTORIZER = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words='english')
        # fit on the provided texts only (online fallback) — callers should precompute index for best results
        TFIDF_VECTORIZER.fit(texts_clean)

    X = TFIDF_VECTORIZER.transform(texts_clean)

    # If SVD is available, reduce to SVD_DIM for compact embeddings
    if TruncatedSVD is not None:
        if SVD_MODEL is None:
            target_dim = min(SVD_DIM, X.shape[1])
            SVD_MODEL = TruncatedSVD(n_components=target_dim)
            try:
                SVD_MODEL.fit(X)
            except Exception:
                SVD_MODEL = None

        if SVD_MODEL is not None:
            X_reduced = SVD_MODEL.transform(X)
            embeddings = X_reduced.astype('float32')
        else:
            embeddings = X.toarray().astype('float32')
    else:
        embeddings = X.toarray().astype('float32')

    # normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (embeddings / norms).astype('float32')


def build_faiss_index(embeddings: np.ndarray):
    if faiss is None:
        raise ImportError('faiss is required for nearest neighbor search')
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine via normalized vectors; we'll normalize
    # normalize embeddings to unit vectors for inner-product as cosine
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_norm = embeddings / norms
    index.add(embeddings_norm.astype('float32'))
    return index


def ensure_embeddings_dir():
    Path(EMBEDDINGS_DIR).mkdir(parents=True, exist_ok=True)


def build_and_persist_index(df: pd.DataFrame, text_col: Optional[str] = None, model: Optional[Any] = None, overwrite: bool = False) -> Dict:
    """Compute embeddings for all reviews in df, build a FAISS index and persist it along with metadata.
    Returns a dict with paths and summary info.
    """
    ensure_embeddings_dir()
    if Path(INDEX_FILE).exists() and Path(METADATA_FILE).exists() and not overwrite:
        return {'status': 'exists', 'index_file': INDEX_FILE, 'metadata_file': METADATA_FILE}

    if text_col is None:
        text_col = detect_text_column(df)
        if text_col is None:
            raise ValueError('No text column detected in DataFrame')

    texts = df[text_col].fillna('').astype(str).tolist()
    texts_scrubbed = [scrub_pii(t) for t in texts]

    if model is None:
        model = get_sentence_transformer()

    embeddings = embed_texts(texts_scrubbed, model=model)
    # normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_norm = embeddings / norms

    # Build FAISS index
    index = build_faiss_index(embeddings_norm)

    # Persist index and metadata
    if faiss is not None:
        try:
            faiss.write_index(index, INDEX_FILE)
        except Exception:
            # fallback: try float32 save via numpy
            np.save(str(Path(EMBEDDINGS_DIR) / 'embeddings.npy'), embeddings_norm.astype('float32'))
    else:
        np.save(str(Path(EMBEDDINGS_DIR) / 'embeddings.npy'), embeddings_norm.astype('float32'))

    # Save metadata (keep a mapping of row -> parent_asin, review_id, date, text length)
    meta = df.reset_index(drop=True)[['parent_asin']].copy()
    # attempt to include review_id and date if present
    if 'review_id' in df.columns:
        meta['review_id'] = df['review_id'].astype(str).values
    else:
        meta['review_id'] = [f'row_{i}' for i in range(len(df))]
    if 'date' in df.columns:
        meta['date'] = pd.to_datetime(df['date']).astype(str).values
    else:
        meta['date'] = [None] * len(df)
    meta['text_preview'] = [t[:300] for t in texts_scrubbed]
    meta.to_parquet(METADATA_FILE, index=False)

    return {'status': 'saved', 'index_file': INDEX_FILE, 'metadata_file': METADATA_FILE, 'n': len(df)}


def rerank_snippets(question: str, candidates: List[Dict], top_k: int = 8, model_name: str = None) -> List[Dict]:
    """Rerank a list of candidate snippets using a CrossEncoder when available.

    candidates: list of dicts with at least a 'text' key.
    Returns the top_k candidates sorted by reranker score (added as 'rerank_score').
    If CrossEncoder is not available or loading fails, return the input list truncated to top_k.
    """
    if model_name is None:
        model_name = CROSS_ENCODER_MODEL

    # Prefer CrossEncoder when available; if it fails (e.g., network/download issues)
    # fall back to a lightweight TF-IDF based reranker (no external downloads).
    texts = [c.get('text', '') for c in candidates]

    if CrossEncoder is not None:
        global CROSS_ENCODER
        try:
            if CROSS_ENCODER is None:
                CROSS_ENCODER = CrossEncoder(model_name)
            pairs = [[question, t] for t in texts]
            scores = CROSS_ENCODER.predict(pairs)
            scored = sorted(((float(s), c) for s, c in zip(scores, candidates)), key=lambda x: -x[0])
            top = []
            for score, cand in scored[:top_k]:
                cand = dict(cand)
                cand['rerank_score'] = float(score)
                top.append(cand)
            return top
        except Exception:
            # proceed to TF-IDF fallback
            pass

    # TF-IDF fallback (fast, local). Requires scikit-learn.
    if TfidfVectorizer is not None:
        try:
            vec = TfidfVectorizer(stop_words='english')
            docs = [question] + texts
            X = vec.fit_transform(docs)
            qv = X[0].toarray()
            docv = X[1:].toarray()
            # cosine similarity
            norms_q = (qv ** 2).sum(axis=1, keepdims=True) ** 0.5
            norms_d = (docv ** 2).sum(axis=1, keepdims=True) ** 0.5
            norms_q[norms_q == 0] = 1.0
            norms_d[norms_d == 0] = 1.0
            sims = (docv @ qv.T).reshape(-1) / (norms_d.reshape(-1) * norms_q.reshape(-1))
            scored = sorted(((float(s), c) for s, c in zip(sims, candidates)), key=lambda x: -x[0])
            top = []
            for score, cand in scored[:top_k]:
                cand = dict(cand)
                cand['rerank_score'] = float(score)
                top.append(cand)
            return top
        except Exception:
            pass

    # Last resort: return first top_k candidates (original order)
    return candidates[:top_k]


def rerank_snippets(question: str, candidates: List[Dict], top_k: int = 8, model_name: str = None) -> List[Dict]:
    """Rerank a list of candidate snippets using a CrossEncoder when available.

    candidates: list of dicts with at least a 'text' key.
    Returns the top_k candidates (possibly fewer) sorted by reranker score (added as 'rerank_score').
    If CrossEncoder is not available, return the input list truncated to top_k (preserving original order).
    """
    if model_name is None:
        model_name = CROSS_ENCODER_MODEL

    # Quick fallback: if no CrossEncoder installed, return first top_k candidates
    if CrossEncoder is None:
        return candidates[:top_k]

    global CROSS_ENCODER
    try:
        if CROSS_ENCODER is None:
            CROSS_ENCODER = CrossEncoder(model_name)
    except Exception:
        # If loading the cross-encoder fails, gracefully fallback
        return candidates[:top_k]

    texts = [c.get('text', '') for c in candidates]
    pairs = [[question, t] for t in texts]

    try:
        scores = CROSS_ENCODER.predict(pairs)
    except Exception:
        return candidates[:top_k]

    scored = sorted(((float(s), c) for s, c in zip(scores, candidates)), key=lambda x: -x[0])
    top = []
    for score, cand in scored[:top_k]:
        cand = dict(cand)  # copy
        cand['rerank_score'] = float(score)
        top.append(cand)
    return top


def load_index_and_metadata():
    """Load persisted FAISS index and metadata; returns (index, metadata_df) or (None, None)."""
    if not Path(EMBEDDINGS_DIR).exists():
        return None, None
    meta = None
    idx = None
    if Path(METADATA_FILE).exists():
        try:
            meta = pd.read_parquet(METADATA_FILE)
            print(f"   ├─ Metadata loaded: {len(meta)} rows")
        except Exception as e:
            print(f"   ├─ Metadata load failed: {e}")
            meta = None
    else:
        print(f"   ├─ Metadata file not found: {METADATA_FILE}")
    if Path(INDEX_FILE).exists() and faiss is not None:
        try:
            idx = faiss.read_index(INDEX_FILE)
            print(f"   ├─ Index loaded: {idx.ntotal} vectors")
        except Exception as e:
            print(f"   ├─ Index load failed: {e}")
            idx = None
    else:
        if not Path(INDEX_FILE).exists():
            print(f"   ├─ Index file not found: {INDEX_FILE}")
        if faiss is None:
            print(f"   ├─ FAISS not available")
    # If index missing but embeddings.npy exists, we can load embeddings as fallback
    return idx, meta


def retrieve_top_k(df: pd.DataFrame, asin: str, k: int = 8) -> List[Dict]:
    """Return top-k review snippets for a product ASIN.
    Each item: {review_id, text, date, score}
    """
    product_rows = df[df['parent_asin'] == asin].copy()
    if product_rows.empty:
        return []

    text_col = detect_text_column(product_rows)
    if text_col is None:
        return []

    product_rows = product_rows.reset_index(drop=True)
    texts = product_rows[text_col].fillna('').astype(str).tolist()
    # Scrub PII before embedding
    texts_scrubbed = [scrub_pii(t) for t in texts]

    model = get_sentence_transformer()
    embeddings = embed_texts(texts_scrubbed, model=model)

    # normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_norm = embeddings / norms

    # Build index and query using the product "corpus" itself by default (we'll return highest-scoring docs by length)
    # For retrieval for a question, caller should compute query embedding and search; this helper returns the corpus with embeddings.
    results = []
    for i, row in product_rows.iterrows():
        results.append({
            'idx': i,
            'review_id': row.get('review_id', f'{asin}_{i}'),
            'text': texts_scrubbed[i],
            'date': row.get('date', None),
            'embedding': embeddings_norm[i]
        })
    return results


def answer_with_openai(question: str, snippets: List[Dict], model_name: str = OPENAI_MODEL) -> Dict:
    """Call OpenAI ChatCompletion with a constructed prompt using snippets.
    Returns dict with 'answer' and 'sources' list.
    """
    if openai is None:
        raise ImportError('openai package is required for this function')
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if not openai_api_key:
        raise EnvironmentError('OPENAI_API_KEY not set in environment')
    openai.api_key = openai_api_key

    # Build a compact context: use up to 6 snippets, shortened
    max_snips = 6
    prepared = []
    for i, s in enumerate(snippets[:max_snips]):
        text = s['text']
        # truncate to ~300 chars per snippet
        snippet_short = (text[:300] + '...') if len(text) > 300 else text
        prepared.append(f"[{i}] review_id={s.get('review_id')} date={s.get('date')}: {snippet_short}")

    # Strong instruction to return strict JSON matching schema; provide a short example
    system = (
        "You are a concise, evidence-based assistant. Return output EXACTLY as a single valid JSON object and nothing else. "
        "Do not add any commentary outside the JSON. Always base claims only on the provided snippets."
    )

    schema = (
        "Respond with JSON keys: summary (one short sentence), pros (array of short strings), "
        "cons (array of short strings), evidence (array of {index:int, excerpt:str}). "
        "If insufficient evidence, set summary='Insufficient evidence' and empty arrays."
    )

    example = (
        "Example output:\n{\n  \"summary\": \"Mostly durable with minor cosmetic issues.\",\n  \"pros\": [\"Sturdy build\"],\n  \"cons\": [\"Finish may chip\"],\n  \"evidence\": [{\"index\":0, \"excerpt\":\"The build feels very sturdy...\"}]\n}"
    )

    user = schema + "\n\n" + example + "\n\nQUESTION: " + question + "\n\nSNIPPETS:\n" + "\n".join(prepared)

    resp = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.0,
        max_tokens=512
    )
    answer = resp['choices'][0]['message']['content'].strip()

    # Try to parse structured JSON from the assistant's output
    structured = parse_structured_json(answer)
    structured_parse_method = None
    if structured is not None:
        structured_parse_method = 'json'
    else:
        try:
            parsed = parse_answer_evidence(answer, prepared)
            if parsed:
                structured = parsed
                structured_parse_method = 'answer_evidence'
            else:
                logger.info('OpenAI: failed to parse JSON or Answer/Evidence format')
        except Exception as e:
            logger.exception('Error parsing OpenAI output for structured data: %s', e)

    # Simple heuristic to extract cited indices like [0], [1]
    cited = re.findall(r"\[(\d+)\]", answer)
    cited_idx = sorted(set(int(c) for c in cited if c.isdigit()))
    sources = []
    for i in cited_idx:
        if i < len(prepared):
            sources.append({'index': i, 'snippet': prepared[i]})

    result = {'answer': answer, 'sources': sources, 'raw': resp, 'structured': structured, 'structured_parse_method': structured_parse_method}
    if structured is None:
        result['structured_parse_error'] = 'no_json_found'
    return result


def answer_with_gemini(question: str, snippets: List[Dict], model_name: str = 'gemini-2.0-flash') -> Dict:
    """Generate an answer using Google Gemini API (free tier available).
    
    Gemini 2.0 Flash is fast, free (with limits), and excellent for RAG tasks.
    """
    if genai is None:
        raise ImportError('google-generativeai is required for Gemini')
    
    # Build context from snippets
    context_parts = []
    for i, s in enumerate(snippets[:8]):  # Use top 8 snippets
        text = s.get('text', '')
        if text:
            context_parts.append(f"Review {i+1}: {text[:500]}")
    
    context = "\n\n".join(context_parts)
    
    # Build prompt
    prompt = f"""You are an expert product review analyst. Answer the question based ONLY on the customer reviews provided below.

Customer Reviews:
{context}

Question: {question}

Instructions:
- Synthesize information from multiple reviews
- Be specific and cite examples from the reviews
- If the reviews don't contain enough information, say so
- Keep your answer concise and factual

Answer:"""
    
    try:
        # Try primary model first
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        answer = response.text
        
        return {
            'answer': answer,
            'sources': snippets,
            'raw': response,
            'model': model_name
        }
    except Exception as e:
        error_msg = str(e)
        # If quota exceeded on gemini-2.0-flash, try gemini-2.5-flash
        if 'quota' in error_msg.lower() and model_name == 'gemini-2.0-flash':
            try:
                logger.info("Trying fallback model: gemini-2.5-flash")
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(prompt)
                answer = response.text
                return {
                    'answer': answer,
                    'sources': snippets,
                    'raw': response,
                    'model': 'gemini-2.5-flash'
                }
            except Exception as e2:
                return {'error': f'Gemini API call failed: {e2}', 'answer': None}
        return {'error': f'Gemini API call failed: {e}', 'answer': None}


def answer_with_local_model(question: str, snippets: List[Dict], model_name: str = None) -> Dict:
    """Generate an answer using a local Hugging Face seq2seq model (free).

    This downloads a larger model on first run (e.g. 'google/flan-t5-base' or 'sshleifer/distilbart-cnn-12-6').
    If transformers is not available, raises ImportError.
    """
    if _hf_pipeline is None:
        raise ImportError('transformers is required for local generation')

    if model_name is None:
        # Use Phi-2 (2.7B parameters) - fits on 4GB GPU
        model_name = 'microsoft/phi-2'

    # Build a compact context. We must ensure the tokenized prompt fits the model's max input length.
    max_new_tokens = int(os.environ.get('GENAI_MAX_NEW_TOKENS', 256))


    # Choose pipeline type based on model family
    if any(x in model_name.lower() for x in ['llama', 'deepseek', 'mistral', 'chat', 'alpaca', 'phi', 'gpt']):
        pipe_type = 'text-generation'
    else:
        pipe_type = 'text2text-generation'

    # Use GPU if available
    import torch
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"
    print(f"   ├─ Device: {device_name}")
    
    # Create pipeline with proper device and dtype parameters
    if device == 0:
        gen = _hf_pipeline(pipe_type, model=model_name, device=device, model_kwargs={"torch_dtype": torch.float16})
    else:
        gen = _hf_pipeline(pipe_type, model=model_name, device=device)
    tokenizer = getattr(gen, 'tokenizer', None)
    # model_max_length fallback
    try:
        model_max = int(getattr(tokenizer, 'model_max_length', 512) or 512)
    except Exception:
        model_max = 512

    # Reserve tokens for generation and a small buffer
    buffer_tokens = 32
    reserve = max_new_tokens + buffer_tokens
    allowed_input_tokens = max(128, model_max - reserve)

    # Start with defaults and iteratively shrink until encoded length <= allowed_input_tokens
    max_snips = 6
    snippet_char_limit = 300
    prepared = []
    json_instruction = (
        "You are a concise, evidence-based assistant specialized in analyzing customer reviews. "
        "Use only the information from the snippets provided below. Do not use any external knowledge or assumptions.\n\n"
        "Your goal:\n1. Answer the QUESTION directly and concisely.\n2. Base every claim on the snippets, citing sources as [n].\n"
        "3. After your answer, list the relevant evidence snippets that support it.\n4. If the snippets don’t contain enough information, reply exactly: \"Insufficient evidence.\"\n\n"
        "Output format (follow exactly):\nAnswer:\n<your concise, factual answer citing snippets as [n]>\n\nEvidence:\n<copy or briefly summarize the relevant snippet parts with their [n] reference>\n\n"
    )


    # Iteratively reduce prompt size, but always keep at least 2 snippets
    min_snips = 2
    while True:
        prepared = []
        for i, s in enumerate(snippets[:max(max_snips, min_snips)]):
            text = s.get('text', '')
            snippet_short = (text[:snippet_char_limit] + '...') if len(text) > snippet_char_limit else text
            prepared.append(f"[{i}] review_id={s.get('review_id')} date={s.get('date')}: {snippet_short}")

        # Refined prompt instruction for complaints
        refined_instruction = (
            "Summarize the main complaints or negative feedback from the following product reviews. "
            "List the most common issues and cite supporting snippets as [n]. If insufficient evidence, reply 'Insufficient evidence.'\n\n"
        )
        prompt_body = refined_instruction + f"QUESTION: {question}\n\nSNIPPETS:\n" + "\n".join(prepared)

        toks = None
        try:
            toks = tokenizer.encode(prompt_body, add_special_tokens=False)
            if len(toks) <= allowed_input_tokens:
                prompt = prompt_body
                final_token_len = len(toks)
                break
            else:
                logger.info(f"Prompt token length {len(toks)} exceeds allowed {allowed_input_tokens}; shrinking...")
        except Exception:
            if len(prompt_body) <= allowed_input_tokens * 4:
                prompt = prompt_body
                final_token_len = None
                break
            else:
                logger.info(f"Prompt char length {len(prompt_body)} exceeds heuristic allowed {allowed_input_tokens*4}; shrinking...")

        # Reduce snippet length first, then number of snippets, but keep at least min_snips
        if snippet_char_limit > 80:
            old = snippet_char_limit
            snippet_char_limit = max(80, snippet_char_limit // 2)
            logger.info(f"Reducing snippet_char_limit from {old} -> {snippet_char_limit}")
            continue
        if max_snips > min_snips:
            old_snips = max_snips
            max_snips -= 1
            logger.info(f"Reducing max_snips from {old_snips} -> {max_snips}")
            continue
        prompt = prompt_body
        final_token_len = toks and len(toks)
        logger.warning(f"Could not shrink prompt to fit model limits; final_token_len={final_token_len}, allowed={allowed_input_tokens}")
        break

    # Log if we reduced prompt size from defaults
    try:
        if max_snips < 6 or snippet_char_limit < 300:
            logger.info(f"Final prompt composition: max_snips={max_snips}, snippet_char_limit={snippet_char_limit}, final_token_len={locals().get('final_token_len')}")
    except Exception:
        pass
    # Use deterministic generation (do_sample=False) for consistency
    # Use `max_new_tokens` per HuggingFace Transformers recommendations.
    # `max_new_tokens` controls the number of tokens generated and avoids the
    # ambiguity/warning when both `max_length` and `max_new_tokens` are set.


    # For text-generation models, output is a list of dicts with 'generated_text'
    if pipe_type == 'text-generation':
        out = gen(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]
        answer = out.get('generated_text') or str(out)
    else:
        out = gen(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]
        answer = out.get('generated_text') or out.get('summary_text') or str(out)

    # Try to parse structured JSON from the model output
    structured = parse_structured_json(answer)
    structured_parse_method = None
    if structured is not None:
        structured_parse_method = 'json'
    else:
        try:
            parsed = parse_answer_evidence(answer, prepared)
            if parsed:
                structured = parsed
                structured_parse_method = 'answer_evidence'
            else:
                logger.info('Local model: failed to parse JSON or Answer/Evidence format')
        except Exception as e:
            logger.exception('Error parsing model output for structured data: %s', e)

    # If the answer is just a repeated snippet, fallback to concatenating top complaint snippets
    fallback_trigger = False
    if answer.strip() == prepared[0].strip() or len(answer.strip()) < 40:
        fallback_trigger = True
    # Extract cited indices heuristically
    cited = re.findall(r"\[(\d+)\]", answer)
    cited_idx = sorted(set(int(c) for c in cited if c.isdigit()))
    sources = []
    for i in cited_idx:
        if i < len(prepared):
            sources.append({'index': i, 'snippet': prepared[i]})

    prompt_meta = {
        'max_snips': max_snips,
        'snippet_char_limit': snippet_char_limit,
        'final_token_len': locals().get('final_token_len'),
        'allowed_input_tokens': locals().get('allowed_input_tokens')
    }

    if fallback_trigger:
        combined = "\n---\n".join([s['text'] for s in snippets[:3]])
        fallback_answer = f"[Fallback summary] Top {min(3, len(snippets))} complaint excerpts:\n" + combined
        return {
            'answer': fallback_answer,
            'sources': sources,
            'raw': out,
            'structured': None,
            'structured_parse_method': None,
            'prompt_meta': prompt_meta,
            'structured_parse_error': 'fallback_triggered',
            'snippets': snippets
        }

    result = {
        'answer': answer.strip(),
        'sources': sources,
        'raw': out,
        'structured': structured,
        'structured_parse_method': structured_parse_method,
        'prompt_meta': prompt_meta,
        'snippets': snippets
    }
    if structured is None:
        result['structured_parse_error'] = 'no_json_found'
    return result


def qa_for_product(df: pd.DataFrame, asin: str, question: str, top_k: int = 12, force_refresh: bool = False) -> Dict:
    """End-to-end: retrieve top-k relevant snippets (via cosine similarity to question) and call Llama model.
    Uses the new ABSARAGAgent class for cleaner, more maintainable code.
    """
    print(f"\n{'='*60}")
    print(f"🤖 GenAI Pipeline Started (New RAG Agent)")
    print(f"{'='*60}")
    print(f"📦 Product ASIN: {asin}")
    print(f"❓ Question: {question}")
    print(f"🎯 Top-K: {top_k}")
    print(f"🔄 Force Refresh: {force_refresh}")
    print(f"{'='*60}\n")
    
    # Check cache first (unless force_refresh is True)
    # Cache disabled
    print("⏭️  Cache disabled - generating fresh answer")
    
    # Try to use the new RAG Agent class if metadata exists, otherwise use legacy
    print("\n🤖 Step 2: Checking for metadata file...")
    metadata_path = Path(EMBEDDINGS_DIR) / 'metadata.parquet'
    
    if metadata_path.exists():
        print("✅ Metadata file found - using new RAG Agent")
        try:
            from rag_agent import ABSARAGAgent
            
            # Initialize agent with Llama model
            model_name = os.environ.get('LOCAL_GEN_MODEL', 'meta-llama/Llama-3.1-8B-Instruct')
            agent = ABSARAGAgent(
                model_name=model_name,
                embedding_model=EMBEDDING_MODEL,
                cross_encoder_model=CROSS_ENCODER_MODEL,
                embeddings_dir=EMBEDDINGS_DIR,
                max_new_tokens=int(os.environ.get('GENAI_MAX_NEW_TOKENS', 256))
            )
            print(f"✅ RAG Agent initialized with model: {model_name}")
            
            # Get response from agent
            print("\n🚀 Step 3: Getting response from RAG Agent...")
            result = agent.get_response(question, asin, top_k=top_k)
            
            # Add cached flag
            result['cached'] = False
            
            # Cache disabled
            
            # Format result to match expected structure
            formatted_result = {
                'answer': result.get('answer'),
                'snippets': result.get('sources', []),
                'cached': False,
                'method': result.get('method'),
                'model': result.get('model')
            }
            
            if 'error' in result:
                formatted_result['error'] = result['error']
            
            print(f"\n{'='*60}")
            print(f"✅ Pipeline Complete - {result.get('method', 'unknown')}")
            print(f"{'='*60}\n")
            
            return formatted_result
            
        except Exception as e:
            print(f"❌ RAG Agent failed: {e}")
            print("   └─ Falling back to legacy implementation...")
    else:
        print("⚠️  Metadata file not found - using legacy implementation")
        print(f"   └─ To use new RAG Agent, run: python absa_dashboard/rebuild_index.py")
        # Fall back to legacy implementation below

    # Prefer a persisted global index for retrieval if available
    print("\n📊 Step 2: Loading global FAISS index...")
    idx, meta = load_index_and_metadata()
    text_col_global = detect_text_column(df)

    if idx is not None and meta is not None and text_col_global is not None:
        print(f"✅ Global index loaded successfully!")
        print(f"   ├─ Index vectors: {idx.ntotal}")
        print(f"   ├─ Metadata rows: {len(meta)}")
        print(f"   └─ Text column: {text_col_global}")
        
        try:
            print("\n🔢 Step 3: Computing question embedding...")
            model = get_sentence_transformer()
            q_emb = embed_texts([question], model=model)
            q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
            print(f"✅ Question embedded: shape={q_emb.shape}")

            candidate_k = max(500, top_k * 50)
            print(f"\n🔎 Step 4: Vector search (searching {candidate_k} candidates)...")
            qvec = q_emb.astype('float32')
            D, I = idx.search(qvec, candidate_k)
            D = D[0]
            I = I[0]
            print(f"✅ Vector search complete")
            print(f"   ├─ Top similarity scores: {D[:5].tolist()}")
            print(f"   └─ Top indices: {I[:5].tolist()}")
            
            # Collect candidate snippets from the persisted index (do not stop at top_k yet)
            print(f"\n🎯 Step 5: Filtering by ASIN={asin}...")
            candidates = []
            for score, pos in zip(D, I):
                if pos < 0:
                    continue
                try:
                    row = meta.iloc[int(pos)]
                except Exception:
                    continue
                if str(row.get('parent_asin')) != str(asin):
                    continue
                candidates.append({
                    'review_id': row.get('review_id', f'{asin}_{pos}'),
                    'text': row.get('text_preview', '')[:1000],
                    'date': row.get('date', None),
                    'score': float(score)
                })
            print(f"✅ Found {len(candidates)} matching reviews for this product")

            # Rerank candidates using a CrossEncoder if available, then keep top_k
            print(f"\n🔄 Step 6: Reranking top {top_k} snippets...")
            if candidates:
                snippets = rerank_snippets(question, candidates, top_k=top_k)
                print(f"✅ Reranking complete: {len(snippets)} snippets selected")
                if snippets:
                    print(f"   └─ Top snippet score: {snippets[0]['score']:.4f}")
            else:
                snippets = []
                print("⚠️  No candidates to rerank")

            # after collecting candidates, if we have snippets, call model or return fallback
            if len(snippets) > 0:
                print(f"\n🤖 Step 7: Generating answer...")
                
                # Try Gemini first (free, fast, excellent quality)
                if genai is not None:
                    print("   ├─ Using: Google Gemini 1.5 Flash (free API)")
                    try:
                        result = answer_with_gemini(question, snippets)
                        result['snippets'] = snippets
                        result['cached'] = False
                        print("   └─ ✅ Answer generated successfully")
                        # save to cache
                        try:
                            set_cached_answer(asin, question, result['answer'], sanitize_snippets_for_cache(snippets))
                            print("   └─ ✅ Answer cached for future use")
                        except Exception as e:
                            print(f"   └─ ⚠️  Cache save failed: {e}")
                        print(f"\n{'='*60}")
                        print("✅ Pipeline Complete - Gemini Answer")
                        print(f"{'='*60}\n")
                        return result
                    except Exception as e:
                        print(f"   └─ ⚠️  Gemini failed: {e}")
                        # Fall through to next option
                
                # Try OpenAI if available
                if openai is not None and os.environ.get('OPENAI_API_KEY'):
                    print("   ├─ Using: OpenAI GPT-3.5-turbo")
                    try:
                        result = answer_with_openai(question, snippets)
                        result['snippets'] = snippets
                        result['cached'] = False
                        print("   └─ ✅ Answer generated successfully")
                        # save to cache
                        try:
                            set_cached_answer(asin, question, result['answer'], sanitize_snippets_for_cache(snippets))
                            print("   └─ ✅ Answer cached for future use")
                        except Exception as e:
                            print(f"   └─ ⚠️  Cache save failed: {e}")
                        print(f"\n{'='*60}")
                        print("✅ Pipeline Complete - OpenAI Answer")
                        print(f"{'='*60}\n")
                        return result
                    except Exception as e:
                        print(f"   └─ ❌ OpenAI failed: {e}")
                        # Fall through to next option

                # If OpenAI not available, try local transformers-based generator (free)
                if _hf_pipeline is not None:
                    model_name_display = 'microsoft/phi-2'
                    print(f"   ├─ Using: Local HuggingFace model ({model_name_display})")
                    try:
                        result = answer_with_local_model(question, snippets, model_name=model_name_display)
                        result['snippets'] = snippets
                        result['cached'] = False
                        print("   └─ ✅ Answer generated successfully")
                        print(f"\n{'='*60}")
                        print("✅ Pipeline Complete - Local Model Answer")
                        print(f"{'='*60}\n")
                        return result
                    except Exception as e:
                        # fall through to extractive fallback
                        print(f"   └─ ⚠️  Local model failed: {e}")

                print("   ├─ Using: Extractive fallback (concatenate top snippets)")
                combined = "\n---\n".join([s['text'] for s in snippets[:3]])
                fallback_answer = f"[Fallback summary] Top {min(3, len(snippets))} review excerpts:\n" + combined
                print("   └─ ✅ Fallback answer created")
                print(f"\n{'='*60}")
                print("✅ Pipeline Complete - Extractive Fallback")
                print(f"{'='*60}\n")
                return {'answer': fallback_answer, 'snippets': snippets, 'cached': False}

            # else fall through to per-ASIN compute
            print("⚠️  No snippets found from global index, falling back to per-ASIN search")
        except Exception as e:
            # If index usage fails, fall back to original per-ASIN method
            print(f"❌ Global index search failed: {e}")
            print("   └─ Falling back to per-ASIN embedding search...")
    else:
        print("⚠️  Global index not available")
        print(f"   ├─ Index: {idx is not None}")
        print(f"   ├─ Metadata: {meta is not None}")
        print(f"   └─ Text column: {text_col_global}")
        print("   └─ Using per-ASIN embedding search instead...")

    # Per-ASIN embedding search (fallback)
    print("\n📍 Step 2b: Per-ASIN embedding search (fallback path)...")
    rows = df[df['parent_asin'] == asin].reset_index(drop=True)
    if rows.empty:
        print(f"❌ No reviews found for ASIN: {asin}")
        print(f"\n{'='*60}")
        print("❌ Pipeline Failed - No Data")
        print(f"{'='*60}\n")
        return {'error': 'No data for product', 'answer': None}
    
    print(f"✅ Found {len(rows)} reviews for this product")

    text_col = detect_text_column(rows)
    if text_col is None:
        print("❌ No text column found in data")
        print(f"\n{'='*60}")
        print("❌ Pipeline Failed - No Text Column")
        print(f"{'='*60}\n")
        return {'error': 'No text column found', 'answer': None}
    
    print(f"✅ Text column detected: {text_col}")

    texts = rows[text_col].fillna('').astype(str).tolist()
    texts_scrubbed = [scrub_pii(t) for t in texts]
    print(f"✅ Texts extracted and scrubbed for PII")

    print("\n🔢 Step 3b: Computing embeddings for reviews...")
    model = get_sentence_transformer()
    corpus_emb = embed_texts(texts_scrubbed, model=model)
    corpus_norm = corpus_emb / (np.linalg.norm(corpus_emb, axis=1, keepdims=True) + 1e-12)
    print(f"✅ Corpus embeddings computed: shape={corpus_emb.shape}")

    print("\n🔢 Step 4b: Computing question embedding...")
    q_emb = embed_texts([question], model=model)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
    print(f"✅ Question embedded: shape={q_emb.shape}")

    print("\n🔎 Step 5b: Computing similarity scores...")
    sims = (corpus_norm @ q_emb[0]).astype(float)
    candidate_k = max(200, top_k * 20)
    top_candidate_idx = list(np.argsort(-sims)[:candidate_k])
    print(f"✅ Similarity computed, selected top {candidate_k} candidates")
    print(f"   ├─ Max similarity: {sims.max():.4f}")
    print(f"   ├─ Mean similarity: {sims.mean():.4f}")
    print(f"   └─ Top 5 scores: {[sims[i] for i in top_candidate_idx[:5]]}")

    candidates = []
    for i in top_candidate_idx:
        candidates.append({
            'review_id': rows.loc[i].get('review_id', f'{asin}_{i}'),
            'text': texts_scrubbed[i],
            'date': rows.loc[i].get('date', None),
            'score': float(sims[i])
        })

    # Rerank locally retrieved candidates as well
    print(f"\n🔄 Step 6b: Reranking top {top_k} snippets...")
    snippets = rerank_snippets(question, candidates, top_k=top_k)
    print(f"✅ Reranking complete: {len(snippets)} snippets selected")
    if snippets:
        print(f"   └─ Top snippet score: {snippets[0]['score']:.4f}")

    print(f"\n🤖 Step 7b: Generating answer...")
    
    # Try Gemini first (free, fast, excellent quality)
    if genai is not None:
        print("   ├─ Using: Google Gemini 1.5 Flash (free API)")
        try:
            result = answer_with_gemini(question, snippets)
            result['snippets'] = snippets
            result['cached'] = False
            print("   └─ ✅ Answer generated successfully")
            print(f"\n{'='*60}")
            print("✅ Pipeline Complete - Gemini Answer (per-ASIN path)")
            print(f"{'='*60}\n")
            return result
        except Exception as e:
            print(f"   └─ ⚠️  Gemini failed: {e}")
            # Fall through to next option
    
    # Try OpenAI if available
    if openai is not None and os.environ.get('OPENAI_API_KEY'):
        print("   ├─ Using: OpenAI GPT-3.5-turbo")
        try:
            result = answer_with_openai(question, snippets)
            result['snippets'] = snippets
            result['cached'] = False
            print("   └─ ✅ Answer generated successfully")
            print(f"\n{'='*60}")
            print("✅ Pipeline Complete - OpenAI Answer (per-ASIN path)")
            print(f"{'='*60}\n")
            return result
        except Exception as e:
            print(f"   └─ ❌ OpenAI failed: {e}")
            # Fall through to next option

    # If OpenAI not available, try local transformers-based generator (free)
    if _hf_pipeline is not None:
        print("   ├─ Using: Local HuggingFace model (Phi-2 with GPU - optimized for 4GB)")
        try:
            result = answer_with_local_model(question, snippets)
            result['snippets'] = snippets
            result['cached'] = False
            print("   └─ ✅ Answer generated successfully")
            print(f"\n{'='*60}")
            print("✅ Pipeline Complete - Local Model Answer (per-ASIN path)")
            print(f"{'='*60}\n")
            return result
        except Exception as e:
            # fall through to extractive fallback
            print(f"   └─ ⚠️  Local model failed: {e}")

    print("   ├─ Using: Extractive fallback (concatenate top snippets)")
    combined = "\n---\n".join([s['text'] for s in snippets[:3]])
    fallback_answer = f"[Fallback summary] Top {min(3, len(snippets))} review excerpts:\n" + combined
    print("   └─ ✅ Fallback answer created")
    print(f"\n{'='*60}")
    print("✅ Pipeline Complete - Extractive Fallback (per-ASIN path)")
    print(f"{'='*60}\n")
    return {'answer': fallback_answer, 'snippets': snippets, 'cached': False}
