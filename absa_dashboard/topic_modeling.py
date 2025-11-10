"""
ABSA Topic Modeling Module
============================
Advanced topic extraction using LDA and BERT embeddings.

Features:
- LDA topic extraction for negative reviews
- BERT-based clustering
- Category-specific analysis
- Cross-category comparison
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))
from config import *

# NLP imports
try:
    from gensim import corpora, models
    from gensim.models import LdaModel
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("⚠️  gensim not installed. LDA functionality will be limited.")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. BERT functionality will be limited.")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
    STOP_WORDS = set(stopwords.words('english'))
except ImportError:
    NLTK_AVAILABLE = False
    STOP_WORDS = set()
    print("⚠️  nltk not installed. Text preprocessing will be limited.")


class TopicModeler:
    """
    Topic modeling for ABSA data analysis.
    """
    
    def __init__(self, df):
        """
        Initialize topic modeler with ABSA data.
        
        Args:
            df: Preprocessed ABSA DataFrame
        """
        self.df = df
        self.lda_model = None
        self.dictionary = None
        self.corpus = None
        self.bert_model = None
        
    
    def preprocess_text(self, text):
        """
        Preprocess text for topic modeling.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if not isinstance(text, str):
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        if NLTK_AVAILABLE:
            tokens = word_tokenize(text)
        else:
            tokens = text.split()
        
        # Remove stopwords and short tokens
        tokens = [
            token for token in tokens 
            if token not in STOP_WORDS and len(token) > 2 and token.isalnum()
        ]
        
        return tokens
    
    
    def extract_lda_topics(self, sentiment_filter='Negative', num_topics=NUM_TOPICS_LDA, 
                           category=None, min_reviews=20):
        """
        Extract topics using LDA on reviews.
        
        Args:
            sentiment_filter: Filter by sentiment ('Positive', 'Negative', 'Neutral', or None for all)
            num_topics: Number of topics to extract
            category: Optional category filter
            min_reviews: Minimum reviews needed
            
        Returns:
            Dictionary with topics and metadata
        """
        if not GENSIM_AVAILABLE:
            print("❌ gensim not installed. Cannot perform LDA.")
            return {}
        
        print(f"🔍 Extracting {num_topics} LDA topics...")
        if sentiment_filter:
            print(f"   Filtering for: {sentiment_filter} sentiment")
        if category:
            print(f"   Category: {category}")
        
        # Filter data
        data = self.df.copy()
        if sentiment_filter:
            data = data[data['sentiment'] == sentiment_filter]
        if category:
            data = data[data['main_category'] == category]
        
        if len(data) < min_reviews:
            print(f"⚠️  Insufficient reviews ({len(data)}) for topic modeling")
            return {}
        
        print(f"   Processing {len(data)} reviews...")
        
        # Preprocess texts
        documents = data['text'].apply(self.preprocess_text).tolist()
        documents = [doc for doc in documents if len(doc) > 0]
        
        if len(documents) < min_reviews:
            print("⚠️  Insufficient valid documents after preprocessing")
            return {}
        
        # Create dictionary and corpus
        self.dictionary = corpora.Dictionary(documents)
        self.dictionary.filter_extremes(no_below=2, no_above=0.5)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in documents]
        
        # Train LDA model
        print("   Training LDA model...")
        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=PASSES_LDA,
            iterations=ITERATIONS_LDA,
            alpha='auto',
            per_word_topics=True
        )
        
        # Extract topics
        topics = []
        for idx in range(num_topics):
            topic_words = self.lda_model.show_topic(idx, topn=10)
            words = [word for word, prob in topic_words]
            probs = [prob for word, prob in topic_words]
            
            # Get top aspects for this topic
            topic_docs = []
            for doc_idx, doc_bow in enumerate(self.corpus):
                doc_topics = self.lda_model.get_document_topics(doc_bow)
                for topic_id, prob in doc_topics:
                    if topic_id == idx and prob > 0.3:
                        topic_docs.append(data.iloc[doc_idx])
            
            if topic_docs:
                topic_df = pd.DataFrame(topic_docs)
                top_aspects = topic_df['aspect_term_normalized'].value_counts().head(5)
            else:
                top_aspects = pd.Series()
            
            topics.append({
                'topic_id': idx,
                'words': words,
                'probabilities': probs,
                'top_aspects': top_aspects.to_dict() if len(top_aspects) > 0 else {},
                'document_count': len(topic_docs)
            })
        
        print(f"✓ Extracted {len(topics)} topics")
        
        return {
            'topics': topics,
            'model': self.lda_model,
            'dictionary': self.dictionary,
            'corpus': self.corpus,
            'filter': sentiment_filter,
            'category': category,
            'num_documents': len(documents)
        }
    
    
    def extract_bert_clusters(self, num_clusters=NUM_CLUSTERS, category=None, 
                              sentiment_filter='Negative', max_reviews=1000):
        """
        Cluster reviews using BERT embeddings.
        
        Args:
            num_clusters: Number of clusters
            category: Optional category filter
            sentiment_filter: Optional sentiment filter
            max_reviews: Maximum reviews to process (for performance)
            
        Returns:
            Dictionary with clusters and metadata
        """
        if not BERT_AVAILABLE:
            print("❌ sentence-transformers not installed. Cannot perform BERT clustering.")
            return {}
        
        print(f"🔍 Clustering reviews with BERT ({num_clusters} clusters)...")
        
        # Filter data
        data = self.df.copy()
        if sentiment_filter:
            data = data[data['sentiment'] == sentiment_filter]
        if category:
            data = data[data['main_category'] == category]
        
        # Sample if too many reviews
        if len(data) > max_reviews:
            print(f"   Sampling {max_reviews} reviews from {len(data)}")
            data = data.sample(n=max_reviews, random_state=42)
        
        if len(data) < num_clusters:
            print(f"⚠️  Insufficient reviews ({len(data)}) for clustering")
            return {}
        
        print(f"   Processing {len(data)} reviews...")
        
        # Load BERT model
        if self.bert_model is None:
            print(f"   Loading BERT model: {BERT_MODEL}")
            self.bert_model = SentenceTransformer(BERT_MODEL)
        
        # Generate embeddings
        print("   Generating embeddings...")
        texts = data['text'].fillna('').tolist()
        embeddings = self.bert_model.encode(texts, show_progress_bar=True)
        
        # Perform clustering
        print(f"   Clustering into {num_clusters} groups...")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Analyze clusters
        data['cluster'] = clusters
        cluster_info = []
        
        for cluster_id in range(num_clusters):
            cluster_data = data[data['cluster'] == cluster_id]
            
            # Get representative aspects
            top_aspects = cluster_data['aspect_term_normalized'].value_counts().head(5)
            
            # Get most common words
            all_words = []
            for text in cluster_data['text']:
                if isinstance(text, str):
                    all_words.extend(self.preprocess_text(text))
            word_freq = Counter(all_words).most_common(10)
            
            cluster_info.append({
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'avg_sentiment': cluster_data['sentiment_score'].mean(),
                'top_aspects': top_aspects.to_dict(),
                'top_words': dict(word_freq),
                'sample_reviews': cluster_data['text'].head(3).tolist()
            })
        
        print(f"✓ Created {len(cluster_info)} clusters")
        
        return {
            'clusters': cluster_info,
            'embeddings': embeddings,
            'cluster_labels': clusters,
            'category': category,
            'sentiment_filter': sentiment_filter,
            'num_reviews': len(data)
        }
    
    
    def compare_categories(self, num_topics=5):
        """
        Compare topics across different product categories.
        
        Args:
            num_topics: Number of topics per category
            
        Returns:
            Dictionary with category comparisons
        """
        if 'main_category' not in self.df.columns:
            print("⚠️  No category information available")
            return {}
        
        print("🔍 Comparing topics across categories...")
        
        categories = self.df['main_category'].dropna().unique()
        category_topics = {}
        
        for category in categories:
            print(f"\n   Analyzing category: {category}")
            topics = self.extract_lda_topics(
                sentiment_filter='Negative',
                num_topics=num_topics,
                category=category,
                min_reviews=20
            )
            
            if topics:
                category_topics[category] = topics
        
        # Find common vs unique aspects
        all_aspects = {}
        for category, topics in category_topics.items():
            for topic in topics.get('topics', []):
                for aspect in topic.get('top_aspects', {}).keys():
                    if aspect not in all_aspects:
                        all_aspects[aspect] = []
                    all_aspects[aspect].append(category)
        
        # Universal issues (appear in all categories)
        universal = {
            aspect: cats 
            for aspect, cats in all_aspects.items() 
            if len(cats) >= len(category_topics) * 0.7
        }
        
        # Category-specific (appear in only one category)
        specific = {
            aspect: cats[0] 
            for aspect, cats in all_aspects.items() 
            if len(cats) == 1
        }
        
        print(f"\n✓ Found {len(universal)} universal concerns")
        print(f"✓ Found {len(specific)} category-specific concerns")
        
        return {
            'category_topics': category_topics,
            'universal_concerns': universal,
            'category_specific': specific,
            'all_aspects': all_aspects
        }
    
    
    def generate_topic_report(self, output_path=None):
        """
        Generate comprehensive topic modeling report.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Dictionary with all topic analyses
        """
        print("\n" + "="*60)
        print("GENERATING TOPIC MODELING REPORT")
        print("="*60 + "\n")
        
        report = {}
        
        # LDA on negative reviews
        print("[1/3] LDA Topic Extraction...")
        report['lda_negative'] = self.extract_lda_topics(sentiment_filter='Negative')
        
        # BERT clustering
        print("\n[2/3] BERT Clustering...")
        report['bert_clusters'] = self.extract_bert_clusters(sentiment_filter='Negative')
        
        # Category comparison
        print("\n[3/3] Category Comparison...")
        report['category_comparison'] = self.compare_categories()
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON (since it contains complex nested structures)
            import json
            
            # Convert non-serializable objects
            serializable_report = {}
            for key, value in report.items():
                if key == 'lda_negative' and 'model' in value:
                    # Remove model objects, keep only topics
                    serializable_report[key] = {
                        'topics': value.get('topics', []),
                        'filter': value.get('filter'),
                        'category': value.get('category'),
                        'num_documents': value.get('num_documents')
                    }
                elif key == 'bert_clusters' and 'embeddings' in value:
                    # Remove embeddings, keep only cluster info
                    serializable_report[key] = {
                        'clusters': value.get('clusters', []),
                        'category': value.get('category'),
                        'sentiment_filter': value.get('sentiment_filter'),
                        'num_reviews': value.get('num_reviews')
                    }
                else:
                    serializable_report[key] = value
            
            with open(output_path, 'w') as f:
                json.dump(serializable_report, f, indent=2, default=str)
            
            print(f"\n✓ Topic report saved to: {output_path}")
        
        print("\n" + "="*60)
        print("TOPIC MODELING COMPLETE")
        print("="*60 + "\n")
        
        return report


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("ABSA Topic Modeling Module")
    print("-" * 60)
    
    # Load preprocessed data
    data_path = Path(DATA_DIR) / PREPROCESSED_DATA_FILE
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please run preprocess_data.py first.")
        sys.exit(1)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"✓ Loaded {len(df):,} reviews\n")
    
    # Initialize topic modeler
    topic_modeler = TopicModeler(df)
    
    # Generate report
    report = topic_modeler.generate_topic_report(
        output_path='exports/topic_report.json'
    )
    
    print("\n✅ Topic modeling execution complete!")
