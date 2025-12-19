"""
ABSA RAG Agent - Class-based RAG implementation for ABSA Dashboard
Uses HuggingFace Llama model for local generation with FAISS vector retrieval
"""
from typing import List, Dict, Optional, Any
import os
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
import logging

# Configure logger
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:
    SentenceTransformer = None
    CrossEncoder = None

try:
    import importlib
    faiss = importlib.import_module('faiss')
except ImportError:
    faiss = None

try:
    from transformers import pipeline as hf_pipeline
except ImportError:
    hf_pipeline = None


class ABSARAGAgent:
    """
    RAG Agent for ABSA product review question answering.
    Uses HuggingFace models (Llama) for generation and FAISS for retrieval.
    """
    
    def __init__(
        self,
        model_name: str = 'microsoft/phi-2',  # 2.7B params, fits on 4GB GPU
        embedding_model: str = 'all-MiniLM-L6-v2',
        cross_encoder_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        embeddings_dir: str = 'embeddings',
        max_new_tokens: int = 256
    ):
        """
        Initialize the RAG Agent.
        
        Args:
            model_name: HuggingFace model for text generation
            embedding_model: Sentence transformer model for embeddings
            cross_encoder_model: Cross-encoder model for reranking
            embeddings_dir: Directory containing FAISS index and metadata
            max_new_tokens: Maximum tokens to generate in response
        """
        self.model_name = model_name
        self.embedding_model_name = embedding_model
        self.cross_encoder_model_name = cross_encoder_model
        self.embeddings_dir = embeddings_dir
        self.max_new_tokens = max_new_tokens
        
        # Initialize paths - use absolute path if embeddings_dir is relative
        embeddings_path = Path(embeddings_dir)
        if not embeddings_path.is_absolute():
            # Get the directory where this file is located
            base_dir = Path(__file__).parent
            embeddings_path = base_dir / embeddings_dir
        
        self.index_file = str(embeddings_path / 'faiss.index')
        self.metadata_file = str(embeddings_path / 'metadata.parquet')
        
        # Lazy-loaded components
        self._generator = None
        self._embedding_model = None
        self._cross_encoder = None
        self._faiss_index = None
        self._metadata = None
        
        logger.info(f"✅ ABSARAGAgent initialized with model: {model_name}")
    
    @property
    def generator(self):
        """Lazy load the HuggingFace text generation pipeline."""
        if self._generator is None:
            if hf_pipeline is None:
                raise ImportError("transformers library is required for text generation")
            
            logger.info(f"Loading generation model: {self.model_name}")
            
            # Use GPU if available
            import torch
            device = 0 if torch.cuda.is_available() else -1
            device_name = "GPU (CUDA)" if device == 0 else "CPU"
            logger.info(f"Device: {device_name}")
            
            # Use text-generation for Llama models with GPU acceleration
            if device == 0:
                self._generator = hf_pipeline(
                    'text-generation', 
                    model=self.model_name,
                    device=device,
                    model_kwargs={"torch_dtype": torch.float16}
                )
            else:
                self._generator = hf_pipeline(
                    'text-generation', 
                    model=self.model_name,
                    device=device
                )
            logger.info("✅ Generation model loaded")
        
        return self._generator
    
    @property
    def embedding_model(self):
        """Lazy load the sentence transformer embedding model."""
        if self._embedding_model is None:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers library is required")
            
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("✅ Embedding model loaded")
        
        return self._embedding_model
    
    @property
    def cross_encoder(self):
        """Lazy load the cross-encoder reranking model."""
        if self._cross_encoder is None:
            if CrossEncoder is None:
                logger.warning("CrossEncoder not available, skipping reranking")
                return None
            
            logger.info(f"Loading cross-encoder: {self.cross_encoder_model_name}")
            self._cross_encoder = CrossEncoder(self.cross_encoder_model_name)
            logger.info("✅ Cross-encoder loaded")
        
        return self._cross_encoder
    
    @property
    def faiss_index(self):
        """Lazy load the FAISS index."""
        if self._faiss_index is None:
            if faiss is None:
                raise ImportError("faiss library is required")
            
            if not os.path.exists(self.index_file):
                raise FileNotFoundError(f"FAISS index not found: {self.index_file}")
            
            logger.info(f"Loading FAISS index from: {self.index_file}")
            self._faiss_index = faiss.read_index(self.index_file)
            logger.info(f"✅ FAISS index loaded: {self._faiss_index.ntotal} vectors")
        
        return self._faiss_index
    
    @property
    def metadata(self):
        """Lazy load the metadata DataFrame."""
        if self._metadata is None:
            if not os.path.exists(self.metadata_file):
                raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
            
            logger.info(f"Loading metadata from: {self.metadata_file}")
            self._metadata = pd.read_parquet(self.metadata_file)
            logger.info(f"✅ Metadata loaded: {len(self._metadata)} rows")
        
        return self._metadata
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Normalized embedding vector
        """
        embedding = self.embedding_model.encode([text], show_progress_bar=False)
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-12)
        return embedding.astype('float32')
    
    def retrieve_candidates(
        self,
        question: str,
        asin: str,
        top_k: int = 8,
        search_k: int = 500
    ) -> List[Dict]:
        """
        Retrieve relevant review snippets using FAISS vector search.
        
        Args:
            question: User question
            asin: Product ASIN to filter by
            top_k: Number of top results to return after reranking
            search_k: Number of candidates to retrieve before filtering
            
        Returns:
            List of candidate snippets with scores
        """
        # Generate question embedding
        logger.info(f"Embedding question: {question[:50]}...")
        q_emb = self.get_embedding(question)
        
        # Vector search
        logger.info(f"Searching FAISS index for top {search_k} candidates...")
        D, I = self.faiss_index.search(q_emb, search_k)
        D = D[0]
        I = I[0]
        logger.info(f"Vector search returned {len(I)} results")
        logger.info(f"Top 5 scores: {D[:5].tolist()}")
        logger.info(f"Top 5 indices: {I[:5].tolist()}")
        
        # Filter by ASIN and collect candidates
        logger.info(f"Filtering candidates by ASIN={asin}")
        logger.info(f"Metadata has {len(self.metadata)} rows")
        logger.info(f"Metadata columns: {self.metadata.columns.tolist()}")
        
        # Check unique ASINs in metadata
        unique_asins = self.metadata['parent_asin'].unique()
        logger.info(f"Unique ASINs in metadata: {len(unique_asins)}")
        logger.info(f"Sample ASINs: {unique_asins[:5].tolist() if len(unique_asins) > 0 else 'None'}")
        logger.info(f"Target ASIN: {asin} (type: {type(asin)})")
        
        candidates = []
        checked = 0
        for score, pos in zip(D, I):
            if pos < 0:
                continue
            checked += 1
            try:
                row = self.metadata.iloc[int(pos)]
                row_asin = str(row.get('parent_asin', ''))
                
                if checked <= 5:  # Log first 5 for debugging
                    logger.info(f"  Candidate {checked}: pos={pos}, asin={row_asin}, score={score:.4f}")
                
                if row_asin != str(asin):
                    continue
                    
                text_content = row.get('text_preview', '')
                if not text_content or len(text_content.strip()) == 0:
                    logger.warning(f"  Empty text_preview at position {pos}")
                    continue
                    
                candidates.append({
                    'review_id': row.get('review_id', f'{asin}_{pos}'),
                    'text': text_content[:1000],
                    'date': row.get('date', None),
                    'score': float(score)
                })
            except Exception as e:
                logger.warning(f"  Error processing position {pos}: {e}")
                continue
        
        logger.info(f"✅ Found {len(candidates)} matching reviews out of {checked} checked")
        
        # Rerank using cross-encoder
        if candidates and self.cross_encoder:
            logger.info(f"Reranking top {min(len(candidates), top_k)} candidates...")
            snippets = self._rerank_snippets(question, candidates, top_k)
            logger.info(f"✅ Reranking complete: {len(snippets)} snippets selected")
        else:
            snippets = candidates[:top_k]
            logger.info(f"No reranking - using top {len(snippets)} candidates")
        
        if snippets:
            logger.info(f"Final snippets preview:")
            for i, s in enumerate(snippets[:3]):
                logger.info(f"  Snippet {i+1}: score={s['score']:.4f}, text_len={len(s.get('text', ''))}")
        
        return snippets
    
    def _rerank_snippets(
        self,
        question: str,
        candidates: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            question: User question
            candidates: List of candidate snippets
            top_k: Number of top results to return
            
        Returns:
            Reranked list of snippets
        """
        if not self.cross_encoder:
            return candidates[:top_k]
        
        pairs = [(question, c['text']) for c in candidates]
        scores = self.cross_encoder.predict(pairs)
        
        for i, c in enumerate(candidates):
            c['rerank_score'] = float(scores[i])
        
        # Sort by rerank score and take top_k
        candidates.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        return candidates[:top_k]
    
    def generate_answer(
        self,
        question: str,
        snippets: List[Dict]
    ) -> Dict[str, Any]:
        """
        Generate an answer using retrieved snippets and Llama model.
        
        Args:
            question: User question
            snippets: List of retrieved review snippets
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not snippets:
            return {
                'answer': 'No relevant reviews found for this product.',
                'sources': [],
                'method': 'no_context'
            }
        
        # Build context from snippets
        source_knowledge = "\n\n".join([
            f"Review {i+1} (Score: {s['score']:.3f}):\n{s['text']}"
            for i, s in enumerate(snippets[:5])
        ])
        
        # Construct prompt
        system_prompt = """You are an expert product review analyst. Your task is to answer questions about products based on customer reviews.
Use ONLY the information provided in the reviews below. Synthesize information from multiple reviews to give a comprehensive answer.
If the reviews don't contain enough information to answer the question, say so clearly.
Do not add information that is not in the reviews."""
        
        user_prompt = f"""Using the customer reviews below, answer the following question:

Customer Reviews:
{source_knowledge}

Question: {question}

Answer:"""
        
        # Generate response using Llama
        logger.info(f"Generating answer with {self.model_name}...")
        try:
            # Tokenizer
            tokenizer = self.generator.tokenizer
            model_max_length = getattr(tokenizer, 'model_max_length', 4096)
            
            # Calculate token budget
            prompt_tokens = len(tokenizer.encode(system_prompt + user_prompt))
            available_tokens = model_max_length - prompt_tokens - self.max_new_tokens - 50
            
            if available_tokens < 0:
                # Truncate context if needed
                logger.warning(f"Context too long, truncating...")
                snippets = snippets[:3]
                source_knowledge = "\n\n".join([
                    f"Review {i+1}:\n{s['text'][:300]}"
                    for i, s in enumerate(snippets)
                ])
                user_prompt = f"""Using the customer reviews below, answer the following question:

Customer Reviews:
{source_knowledge}

Question: {question}

Answer:"""
            
            # Generate
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            output = self.generator(
                full_prompt,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = output[0]['generated_text']
            
            # Remove the prompt from the output
            if generated_text.startswith(full_prompt):
                answer = generated_text[len(full_prompt):].strip()
            else:
                answer = generated_text.strip()
            
            logger.info("✅ Answer generated successfully")
            
            return {
                'answer': answer,
                'sources': snippets,
                'method': 'llama_generation',
                'model': self.model_name
            }
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            # Fallback to extractive summary
            return self._fallback_answer(question, snippets, error=str(e))
    
    def _fallback_answer(
        self,
        question: str,
        snippets: List[Dict],
        error: str = None
    ) -> Dict[str, Any]:
        """
        Fallback to extractive summary if generation fails.
        
        Args:
            question: User question
            snippets: List of retrieved snippets
            error: Optional error message
            
        Returns:
            Dictionary with fallback answer
        """
        combined = "\n---\n".join([s['text'][:300] for s in snippets[:3]])
        answer = f"[Extractive Summary] Top {min(3, len(snippets))} relevant review excerpts:\n\n{combined}"
        
        result = {
            'answer': answer,
            'sources': snippets,
            'method': 'extractive_fallback'
        }
        
        if error:
            result['error'] = error
        
        return result
    
    def get_response(
        self,
        question: str,
        asin: str,
        top_k: int = 8
    ) -> Dict[str, Any]:
        """
        Main method to get a response for a question about a product.
        
        Args:
            question: User question
            asin: Product ASIN
            top_k: Number of snippets to retrieve
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Processing Question for ASIN: {asin}")
        logger.info(f"Question: {question}")
        logger.info(f"{'='*60}\n")
        
        try:
            # Step 1: Retrieve relevant snippets
            logger.info("📊 Step 1: Retrieving relevant review snippets...")
            snippets = self.retrieve_candidates(question, asin, top_k=top_k)
            
            # Step 2: Generate answer
            logger.info("\n🤖 Step 2: Generating answer...")
            result = self.generate_answer(question, snippets)
            
            logger.info(f"\n{'='*60}")
            logger.info("✅ Pipeline Complete")
            logger.info(f"Method: {result.get('method')}")
            logger.info(f"{'='*60}\n")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return {
                'error': str(e),
                'answer': f"An error occurred: {e}",
                'sources': [],
                'method': 'error'
            }
