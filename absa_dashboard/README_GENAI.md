# GenAI Q&A Feature — Documentation

This document explains the **GenAI Question-Answering feature** integrated into the ABSA dashboard. It enables users to ask natural-language questions about customer reviews and receive synthesized, evidence-based answers powered by **Google Gemini 2.0/2.5 Flash**.

---

## 🎯 Purpose

The GenAI feature transforms the dashboard from a passive visualization tool into an interactive research assistant by providing:

1. **Natural Language Q&A** — Ask questions in plain English (e.g., "What do customers say about price?")
2. **Multi-document Synthesis** — Automatically aggregates insights from multiple relevant reviews
3. **Evidence-based Answers** — Every answer cites specific reviews (e.g., "Reviews 1, 2, 3"), enabling full verification
4. **Flexible Exploration** — No need for predefined queries, chart navigation, or SQL knowledge
5. **Real-time Performance** — Sub-3.5 second response time for interactive usage

**For Non-Technical Users:** Think of it as having a smart assistant who has read all customer reviews and can answer any question you have instantly, showing you the exact reviews it used to form the answer.

**For Technical Users:** This is a production-ready Retrieval-Augmented Generation (RAG) system combining FAISS vector search, cross-encoder reranking, and Google Gemini LLM for grounded answer generation.

---

## 🏗️ Architecture Overview

The system uses **Retrieval-Augmented Generation (RAG)**, a three-stage pipeline that grounds AI responses in actual review data to prevent hallucination:

```
┌──────────────────────┐
│  User Question       │
│  "What about price?" │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  STAGE 1: RETRIEVAL  │
│  ─────────────────── │
│  • Embed question    │──► Convert to 384-dim vector
│  • Search FAISS      │──► Find top-500 candidates
│  • Filter by product │──► Keep relevant reviews
└──────────┬───────────┘
           │ ~180 candidates
           ▼
┌──────────────────────┐
│  STAGE 2: RERANKING  │
│  ─────────────────── │
│  • Cross-encoder     │──► Deep semantic scoring
│  • Rank by relevance │──► Select top-8 reviews
└──────────┬───────────┘
           │ Top 8 reviews
           ▼
┌──────────────────────┐
│  STAGE 3: GENERATION │
│  ─────────────────── │
│  • Build prompt      │──► Context + question
│  • Google Gemini     │──► Generate answer
│  • Extract citations │──► Map to sources
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Answer + Sources    │
│  "Price is positive  │
│   (Reviews 1,2,3)"   │
└──────────────────────┘
```

**Why This Design?**
- **Dense Retrieval** (Stage 1) — Fast semantic search over 10,000+ reviews (~0.7s)
- **Reranking** (Stage 2) — High-precision scoring on smaller candidate set (~0.4s)
- **Generation** (Stage 3) — Synthesizes coherent answer with citations (~2.0s)

Total latency: **2.5-3.5 seconds** per query

---

## 📂 Key Files and Modules

### **1. `genai_client.py`** — Core RAG Implementation

Main orchestration module for retrieval and generation.

#### **Key Functions:**

##### `embed_texts(texts: List[str]) -> np.ndarray`
- **Purpose:** Convert text to dense vector embeddings for semantic search
- **Model:** SentenceTransformer `all-MiniLM-L6-v2` (384 dimensions)
- **Output:** Normalized embeddings for cosine similarity

```python
# Embed question for semantic search
question_embedding = embed_texts(["What do customers say about price?"])
# Output shape: (1, 384)
```

##### `build_and_persist_index(df: pd.DataFrame, overwrite=False) -> Dict`
- **Purpose:** Build FAISS index over all reviews (run once during setup)
- **Steps:**
  1. Embed all review texts using sentence transformer
  2. Build FAISS IndexFlatIP (inner product = cosine similarity)
  3. Save index to `embeddings/faiss.index`
  4. Save metadata (ASIN, review_id, date) to `embeddings/metadata.parquet`
- **When to use:** Initial setup or when adding new reviews

```python
# Build index once
result = build_and_persist_index(df_reviews, overwrite=True)
# Output: {'status': 'saved', 'n': 50000}
```

##### `qa_for_product(df, asin, question, top_k=8) -> Dict`
- **Purpose:** End-to-end Q&A pipeline for a single product
- **Flow:**
  1. Embed question → 384-dim vector
  2. FAISS search → top-500 candidates
  3. Filter by product ASIN → ~180 reviews
  4. Cross-encoder reranking → top-8 reviews
  5. Generate answer with Gemini
- **Returns:** `{'answer': str, 'snippets': List[Dict], 'model_used': str, 'tokens': int}`

```python
# Ask a question about a product
response = qa_for_product(
    df_reviews, 
    asin='B0123456', 
    question='What are common complaints?'
)
print(response['answer'])     # Generated answer
print(response['snippets'])   # 8 source reviews with scores
print(response['model_used']) # 'gemini-2.5-flash'
```

##### `answer_with_gemini(question: str, snippets: List[Dict]) -> Dict`
- **Purpose:** Generate answer using Google Gemini with structured prompting
- **Models:** 
  - Primary: `gemini-2.0-flash` (fastest)
  - Fallback: `gemini-2.5-flash` (if quota exceeded)
- **Prompt Design:**
  - System instructions to synthesize from provided reviews only
  - Explicit citation requirements: "Use (Review 1, 2) format"
  - Grounding constraint: "If not in reviews, say 'Not mentioned'"
- **Returns:** `{'answer': str, 'sources': List[int], 'tokens': Dict, 'model': str}`

```python
# Generate answer from top-8 reviews
result = answer_with_gemini(question, top_reviews)
# Output: {'answer': '...', 'sources': [0,1,2], 'tokens': {'prompt': 420, 'response': 105}}
```

---

### **2. `dashboard.py`** — Streamlit UI Integration

The GenAI Q&A feature is accessible via a dedicated **"GenAI Q&A"** page in the dashboard.

#### **UI Components:**

1. **Product Selector** — Dropdown to choose which product's reviews to query
2. **Question Input** — Free-text box for natural language questions
3. **Ask Button** — Triggers RAG pipeline and displays results
4. **Answer Display** — Shows synthesized response with model info and token count
5. **Source Reviews** — Expandable cards showing the 8 reviews used, with:
   - Review ID and date
   - Relevance score
   - Full review text (up to 500 characters)
6. **Model Fallback Indicator** — Shows which Gemini model was used (2.0 or 2.5 Flash)

#### **Code Flow:**

```python
# User selects product and types question
selected_asin = st.selectbox("Select Product", product_list)
question = st.text_input("Ask a question about this product...")

if st.button("Ask GenAI"):
    with st.spinner('Generating answer...'):
        # Call RAG pipeline
        response = qa_for_product(df, selected_asin, question, top_k=8)
        
        # Display answer
        st.markdown(f"**Answer:** {response['answer']}")
        st.caption(f"Model: {response['model_used']} | Tokens: {response['tokens']['total']}")
        
        # Display source reviews in expandable cards
        st.subheader("Source Reviews")
        for idx, snippet in enumerate(response['snippets']):
            with st.expander(f"Review {idx+1} - Score: {snippet['score']:.3f}"):
                st.text(f"ID: {snippet['review_id']} | Date: {snippet['date']}")
                st.write(snippet['text'][:500])
```

---

## 🔧 Configuration & Setup

### **Environment Variables**

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_GEMINI_API_KEY` | (none) | **Required** — Your Google AI API key |
| `SENTENCE_TRANSFORMER_MODEL` | `all-MiniLM-L6-v2` | Embedding model for dense retrieval (384-dim) |
| `CROSS_ENCODER_MODEL` | `ms-marco-MiniLM-L-6-v2` | Reranking model for precision improvement |
| `EMBEDDINGS_DIR` | `embeddings` | Directory for FAISS index storage (~200MB) |

**Note:** API key is configured directly in `genai_client.py` (not via environment variable). The system uses a fallback strategy: `gemini-2.0-flash` (primary) → `gemini-2.5-flash` (if quota exceeded).

### **Initial Setup**

1. **Get Google AI API Key:**
   - Visit https://aistudio.google.com/app/apikey
   - Create new API key (free)
   - Free tier limits: 15 requests/min, 1M tokens/day
   - Copy your key (format: `AIzaSy...`)

2. **Configure API Key:**
   - Open `absa_dashboard/genai_client.py`
   - Update line with `genai.configure(api_key="YOUR_KEY_HERE")`
   - Replace with your actual key

3. **Install Dependencies:**
```powershell
pip install google-generativeai==0.8.5 sentence-transformers faiss-cpu
```

**Supported Models:** `gemini-2.0-flash-exp`, `gemini-2.5-flash-preview-0514` (automatically selects available model)

4. **Build FAISS Index (one-time):**
```python
# Run in dashboard or Python script
from genai_client import build_and_persist_index
result = build_and_persist_index(df_reviews, overwrite=True)
# Creates: embeddings/faiss.index (~200MB for 50k reviews)
#          embeddings/metadata.parquet (~5MB)
```

---

## 🚀 Usage Examples

### **Example 1: Basic Product Query**

```
User Question: "What do customers say about the price?"

System Process:
1. Embed question → 384-dim vector
2. FAISS search → 500 candidates → filter by ASIN → 180 reviews
3. Rerank → top-8 reviews (scores 0.85-0.92)
4. Gemini generation → synthesized answer with citations

Generated Answer:
"Customers consistently state that the price is very positive. They describe 
it as 'good quality for the price' (Reviews 1, 2), 'reasonable price' 
(Reviews 3, 4, 5, 6), and 'amazing price' (Reviews 7, 8)."

Response Time: 2.8 seconds
Tokens Used: 420 prompt + 105 response = 525 total
Model Used: gemini-2.5-flash
```

### **Example 2: Comparative Question**

```
User Question: "How does quality compare to price?"

Generated Answer:
"Customers generally agree that the quality exceeds expectations for the price 
point. Multiple reviews mention 'salon quality at home' (Reviews 2, 5, 7) and 
'professional results' (Reviews 1, 4) despite the affordable cost. Some note 
it's 'better than expensive brands' (Review 3)."

Source Reviews: 8 reviews, relevance scores 0.78-0.91
```

### **Example 3: Opinion-Based Query**

```
User Question: "Are customers satisfied with durability?"

Generated Answer:
"Opinions on durability are mixed. Several customers report the product 
lasting 'weeks without chipping' (Reviews 1, 4, 6), while others experienced 
'chipping within days' (Reviews 2, 5). The variance may depend on application 
technique and individual usage patterns."

Note: RAG system captures diverse opinions, not just positive sentiment
```

---

## 📊 Performance Characteristics

### **Latency Breakdown**

| Stage | Time | Notes |
|-------|------|-------|
| Dense Retrieval (FAISS) | 0.5-1.0s | Search over 50k+ reviews → top-500 candidates |
| Post-filtering (ASIN) | <0.1s | Filter candidates by product → ~50-350 reviews |
| Reranking (Cross-encoder) | 0.3-0.5s | Deep scoring on filtered set → top-8 reviews |
| Generation (Gemini) | 1.5-2.5s | API call with structured prompt + 8 context reviews |
| **Total** | **2.5-3.5s** | Average: 2.8s across 50+ test queries |

### **Resource Usage**

- **Storage:** ~205MB (200MB FAISS index + 5MB metadata parquet)
- **Memory:** ~500MB during query (sentence transformer + cross-encoder + embeddings)
- **API Costs:** $0.00 (free tier: 15 req/min, 1M tokens/day)
- **Token Usage:** 390-510 prompt + 94-180 response = **484-650 tokens/request**
- **Daily Capacity:** ~1,900 sustained queries (5% of free tier quota for typical workload)

### **Retrieval Quality**

- **Initial Candidates:** 500 reviews (FAISS IndexFlatIP semantic search)
- **Post-filtering:** 50-350 reviews (ASIN-specific filtering)
- **Final Selection:** 8 reviews (cross-encoder reranking)
- **Average Relevance Score:** 0.78/1.00 (range: 0.65-0.92)
- **Precision@8:** High-quality results verified through citation analysis

**Quality Validation:** Generated answers consistently cite 6-8 out of 8 provided reviews, indicating strong retrieval-generation alignment.

---

## 🎯 Key Benefits

### **For Business Users**
- ✅ No training required — just ask questions naturally
- ✅ Instant insights — 3-second responses vs. 10-minute manual analysis
- ✅ Evidence-based — every answer shows source reviews
- ✅ Flexible — handles any question, not just predefined reports

### **For Technical Teams**
- ✅ Grounded in data — RAG prevents hallucination
- ✅ Transparent — full citation trail for verification
- ✅ Scalable — handles 10,000+ reviews efficiently
- ✅ Cost-effective — free tier sufficient for research/small production

### **For Researchers**
- ✅ Reproducible — same question → same sources → consistent answers
- ✅ Interpretable — see exactly which reviews influenced answer
- ✅ Complementary — combines systematic ABSA with flexible exploration
- ✅ Extensible — can integrate aspect-level filtering in future versions

---

## ⚠️ Limitations & Future Work

### **Current Limitations**

1. **Context Window:** Only 8 reviews per query (may miss information in large review sets)
2. **Quota Constraints:** 15 requests/minute, 1M tokens/day (free tier)
3. **No Multi-hop Reasoning:** Cannot synthesize across multiple queries
4. **Basic Reranking:** Cross-encoder doesn't consider aspect-level sentiment from ABSA

### **Planned Improvements**

1. **Aspect-Aware Retrieval:** Integrate ABSA results to prioritize reviews mentioning queried aspects
2. **Iterative Retrieval:** Multi-hop reasoning for complex questions requiring synthesis of multiple answer sets
3. **Answer Verification:** Automated fact-checking against ABSA structured results
4. **Temporal Filtering:** "What changed about quality in the last 3 months?"
5. **Comparative Queries:** "Compare product A vs B on shipping"

---

## 🧪 Testing & Validation

### **Quality Checks**

Run test queries to validate system behavior:

```python
# Test grounding (should cite specific reviews)
qa_for_product(df, asin, "What do customers say about packaging?")
# ✓ Should include phrases like "(Reviews 1, 3, 5)"

# Test negative sentiment handling
qa_for_product(df, asin, "What are common complaints?")
# ✓ Should surface negative feedback, not just positive

# Test "not mentioned" handling
qa_for_product(df, asin, "What about international shipping to Europe?")
# ✓ Should say "Not mentioned in reviews" if no relevant data
```

### **Performance Testing**

```python
import time

# Measure latency over 10 queries
latencies = []
for question in test_questions:
    start = time.time()
    qa_for_product(df, asin, question)
    latencies.append(time.time() - start)

print(f"Avg: {np.mean(latencies):.2f}s, Max: {np.max(latencies):.2f}s")
# Target: Avg < 3.5s, Max < 5.0s
```

---

## 📚 Technical References

### **Key Technologies**

- **FAISS** — Facebook's vector similarity search library
- **Sentence Transformers** — Efficient text embeddings (Hugging Face)
- **Google Gemini** — Multimodal LLM with strong synthesis capabilities
- **Cross-encoders** — Deep bidirectional scoring for reranking

### **Research Papers**

- RAG: Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)
- Sentence Transformers: Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (2019)
- Cross-encoders: Nogueira et al., "Passage Re-ranking with BERT" (2019)

---

## 💡 Tips for Effective Use

### **Writing Good Questions**

✅ **Good:** "What do customers say about battery life?"
- Specific aspect, clear intent

❌ **Avoid:** "Tell me everything about this product"
- Too broad, may return generic summary

✅ **Good:** "Are there complaints about shipping delays?"
- Focused on specific issue

❌ **Avoid:** "Is this good?"
- Vague, hard to answer meaningfully

### **Interpreting Answers**

- **Check source reviews** — Verify AI didn't misinterpret context
- **Note relevance scores** — Lower scores (<0.7) may indicate weaker matches
- **Consider coverage** — 8 reviews may not represent all customer opinions
- **Cross-reference ABSA** — Use structured dashboard to validate patterns

---

## 🆘 Troubleshooting

### **Problem: "Quota exceeded" error**

**Cause:** Free tier limit reached (15 req/min or 1M tokens/day)

**Solution:**
```python
# System automatically falls back to gemini-2.5-flash
# If both models fail, wait 1 minute or upgrade to paid tier
```

### **Problem: Slow first query (~30s)**

**Cause:** Model loading (Sentence Transformer, Cross-encoder)

**Solution:** Normal on first run; subsequent queries are fast (<3.5s)

### **Problem: "FAISS index not found"**

**Cause:** Index not built yet

**Solution:**
```python
from genai_client import build_and_persist_index
build_and_persist_index(df, overwrite=True)
```

### **Problem: Answers seem generic/hallucinated**

**Cause:** Low retrieval quality or prompt issues

**Solution:**
- Check source review scores (should be >0.7)
- Verify question is specific enough
- Rebuild index if reviews changed significantly

---

## 📝 Summary

The GenAI Q&A feature transforms the ABSA dashboard from a static analytics tool into an interactive research assistant. By combining:

- **Efficient retrieval** (FAISS + sentence embeddings)
- **Precise reranking** (cross-encoder scoring)  
- **Grounded generation** (Gemini with explicit citations)

...we enable natural language exploration of customer feedback while maintaining scientific rigor through full source traceability. The system achieves sub-3.5s latency and operates entirely within free-tier API limits, making it practical for research and small production deployments.

**Next Steps:** Integrate aspect-level ABSA results to enable queries like "Show reviews about battery life with negative sentiment in the last 3 months."

---

**Document Version:** 2.0 (Updated December 2025)  
**System Status:** Production-ready with Google Gemini integration  
**Maintainer:** Yosr Drira

from genai_client import embed_texts
texts = ["test sentence", "another test"]
emb = embed_texts(texts)
print(emb.shape)  # Should be (2, 384) using TF-IDF+SVD fallback
```

### **Test 2: Check cache hit behavior**
```python
from genai_client import qa_for_product
from genai_cache import clear_cache, cache_metrics

clear_cache()
resp1 = qa_for_product(df, 'B00123', 'test question?')
print(resp1['cached'])  # False (cache miss)

resp2 = qa_for_product(df, 'B00123', 'test question?')
print(resp2['cached'])  # True (cache hit)

metrics = cache_metrics()
print(metrics)  # {'hit_count': 1, 'miss_count': 1, ...}
```

### **Test 3: Validate local generation (no OpenAI key)**
```powershell
# Unset OpenAI key
$env:OPENAI_API_KEY = ""

# Run dashboard
streamlit run dashboard.py

# Ask question → should use flan-t5-base locally
# First run downloads model (~308MB)
```

---

## 🐛 Troubleshooting

### **Issue: "ImportError: sentence-transformers is required for embeddings"**

**Solution:** Install sentence-transformers
```powershell
pip install sentence-transformers
```

### **Issue: "FAISS index not found, falling back to per-ASIN search"**

**Solution:** Build the index once via dashboard UI:
1. Go to "Product Deep Dive" page
2. Click "Build embeddings index" button
3. Wait for completion (may take 1-5 minutes for 50k reviews)
4. Index persisted to `embeddings/faiss.index` and `embeddings/metadata.parquet`

### **Issue: "Quota exceeded" or "Resource exhausted" error**

**Cause:** Free tier limit reached (15 requests/minute or 1M tokens/day)

**Solution:** System automatically falls back from `gemini-2.0-flash` to `gemini-2.5-flash`
```python
# Fallback cascade implemented in answer_with_gemini()
# Primary: gemini-2.0-flash
# Fallback: gemini-2.5-flash
# If both fail: Wait 1 minute or upgrade API tier
```

### **Issue: Answers seem generic or not citing reviews**

**Cause:** Low retrieval quality or API key not configured

**Solution:**
- Verify API key is set in `genai_client.py`
- Check source review relevance scores (should be >0.70)
- Ensure question is specific (avoid "Tell me about this product")
- Rebuild FAISS index if review corpus changed

---

## 📊 Performance & Scalability

| Component | Performance | Scalability Notes |
|-----------|-------------|-------------------|
| **Embeddings (SentenceTransformer)** | ~100-500 texts/sec on CPU<br>~2000 texts/sec on GPU | Batch embeddings for large datasets. Model: 90MB (all-MiniLM-L6-v2). |
| **FAISS IndexFlatIP** | <10ms for 50k vectors<br><50ms for 500k vectors | Linear scaling. Consider `IndexIVFFlat` for >1M vectors. |
| **Google Gemini API** | 1.5-2.5 sec per request<br>Rate limit: 15 req/min (free) | Cascading fallback ensures availability. 1M tokens/day quota. |
| **Cross-encoder Reranking** | ~0.3-0.5s for 180 candidates | ms-marco-MiniLM-L-6-v2, 23MB model. GPU acceleration available. |
| **Cache (removed)** | N/A | Caching disabled to ensure fresh results. Enable if needed for high-volume. |

**Recommendations:**
- Build FAISS index once, reuse across sessions
- Enable caching with 7-day TTL (default)
- Use GPU for embeddings and generation (if available)
- For >100k reviews, consider pre-computing embeddings offline

---

## 🔐 Security & Privacy

### **API Key Management**
- **Storage:** API key configured directly in `genai_client.py` (hardcoded for research use)
- **Production Note:** For production deployments, use environment variables or secret management service
- **Logging:** API key never logged; only model names and token counts are tracked
- **Rate Limiting:** Automatic quota detection and fallback prevents API abuse

### **Data Privacy**
- **Review Data:** All processing done locally (embeddings, FAISS search)
- **External API Calls:** Only 8 reviews + question sent to Gemini (minimal exposure)
- **PII Consideration:** Review text may contain customer information; use data anonymization if required
- **Compliance:** Ensure review data usage complies with terms of service (e.g., Amazon review scraping policies)

### **Model Versioning**
- **Current Models:** `gemini-2.0-flash-exp`, `gemini-2.5-flash-preview-0514`
- **Stability:** Preview models subject to changes; monitor Google AI release notes
- **Reproducibility:** Log model names and versions for each query to ensure reproducible results

---

## 📚 References & Further Reading

- **SentenceTransformers:** https://www.sbert.net/
- **FAISS:** https://github.com/facebookresearch/faiss
- **OpenAI ChatCompletion:** https://platform.openai.com/docs/guides/chat
- **HuggingFace Transformers:** https://huggingface.co/docs/transformers/
- **RAG pattern:** https://arxiv.org/abs/2005.11401 (Retrieval-Augmented Generation for NLP)

---

## 🛠️ Future Improvements

1. **Reranking:** Add cross-encoder reranker after FAISS retrieval for higher precision
2. **Multi-product Q&A:** Enable cross-product comparison questions ("Compare battery life of product A vs B")
3. **Streaming responses:** Use `stream=True` for OpenAI to show answer incrementally
4. **Fine-tuning:** Fine-tune local model on domain-specific Q&A pairs (beauty product reviews)
5. **Advanced indexing:** Use FAISS `IndexIVFFlat` or `IndexHNSW` for >1M vectors
6. **Contextual cache:** Cache per (ASIN, question, date_range) for time-sensitive queries

---

## 📞 Support

For issues or questions:
1. Check console output for error messages
2. Verify environment variables are set correctly
3. Test with `force_refresh=True` to bypass cache
4. Clear cache and rebuild index if embeddings corrupted
5. Open an issue in the repository with error logs

---

**End of GenAI Code Walkthrough**
