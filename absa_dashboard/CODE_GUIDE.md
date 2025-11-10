# ABSA Dashboard Code Guide

## 📚 Complete Walkthrough of the System

Welcome! This guide will walk you through every part of the ABSA dashboard system, explaining **what** each piece does, **why** it's needed, and **how** it works.

---

## 🎯 The Big Picture

### What Problem Are We Solving?

You have thousands of product reviews with detected **aspects** (features mentioned) and **sentiments** (positive/negative/neutral). The challenges are:

1. **Messy aspect data**: "color", "colors", "colour" are all the same thing
2. **No product context**: Reviews lack product names, categories, prices
3. **Hard to spot trends**: Which products are declining? Which aspects are problematic?
4. **Need visualizations**: Can't easily explore the data

### Our Solution: 4 Modules

```
┌─────────────────────────────────────────────────────────────┐
│                    ABSA DASHBOARD SYSTEM                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │  Preprocess  │───▶│  Dashboard   │───▶│   Insights   │ │
│  │    Data      │    │   (Streamlit)│    │   Reports    │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                                        ▲         │
│         │                                        │         │
│         ▼                                        │         │
│  ┌──────────────┐    ┌──────────────┐          │         │
│  │  Config &    │    │ Alert System │──────────┘         │
│  │   Utils      │    │ Topic Model  │                    │
│  └──────────────┘    └──────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 File Structure Overview

Let's understand what each file does:

| File | Purpose | You'll Learn |
|------|---------|--------------|
| `preprocess_data.py` | Cleans and merges data | Data normalization, joining tables |
| `config.py` | Settings and constants | Configuration management |
| `utils.py` | Helper functions | Reusable utilities |
| `dashboard.py` | Interactive UI | Streamlit, visualizations |
| `alert_system.py` | Monitors trends | Time-series analysis, alerting |
| `topic_modeling.py` | Discovers themes | NLP, clustering |

---

## 🔍 Module 1: `preprocess_data.py`

### Purpose
Transform raw ABSA results into clean, analysis-ready data.

### Key Concepts

#### 1. **Aspect Normalization** (Lines 36-125)

**Problem**: Reviews say "color", "colors", "colour" - all mean the same thing.

**Solution**: Map variations to canonical forms.

```python
def normalize_aspect_terms(df):
    """
    Converts aspect variations to standard forms.
    
    Example:
    Before: ['color', 'colors', 'smell', 'scent', 'price', 'cost']
    After:  ['color', 'color', 'smell', 'smell', 'price', 'price']
    """
    
    # Step 1: Create mapping dictionary
    aspect_mapping = {
        'color': 'color',      # Already canonical
        'colors': 'color',     # Plural → singular
        'colour': 'color',     # British spelling
        'smell': 'smell',
        'scent': 'smell',      # Synonym
        'fragrance': 'smell',  # Synonym
        # ... 50+ mappings
    }
    
    # Step 2: Apply mapping
    # .str.lower() → convert to lowercase first
    # .map() → look up in dictionary
    df['aspect_term_normalized'] = df['aspect_term'].str.lower().map(aspect_mapping)
    
    # Step 3: Keep original if not in mapping
    # .fillna() → for unmapped terms, use original (lowercased)
    df['aspect_term_normalized'] = df['aspect_term_normalized'].fillna(
        df['aspect_term'].str.lower().str.strip()
    )
    
    return df
```

**Why This Matters**:
- Before: 500 unique aspects (confusing)
- After: 300 unique aspects (clearer patterns)

---

#### 2. **DataFrame Merging** (Lines 128-177)

**Problem**: ABSA results only have product IDs (ASINs), not names/categories.

**Solution**: Join with product metadata table.

```python
def merge_dataframes(df_absa, df_products):
    """
    Combines review data with product info.
    
    Input 1 (df_absa):
    review_id | parent_asin | aspect_term | sentiment
    123       | B08X1Y2Z3   | color       | Positive
    
    Input 2 (df_products):
    parent_asin | title              | main_category | price
    B08X1Y2Z3   | Blue Nail Polish   | Beauty        | $9.99
    
    Output (merged):
    review_id | parent_asin | aspect_term | sentiment | title           | category
    123       | B08X1Y2Z3   | color       | Positive  | Blue Nail Pol.. | Beauty
    """
    
    # Step 1: Ensure same data type (both must be strings)
    df_absa['parent_asin'] = df_absa['parent_asin'].astype(str).str.strip()
    df_products['parent_asin'] = df_products['parent_asin'].astype(str).str.strip()
    
    # Step 2: Perform LEFT JOIN (keep all reviews, add product info where available)
    df_merged = df_absa.merge(
        df_products,
        on='parent_asin',        # Join key
        how='left',              # Keep all df_absa rows
        suffixes=('_review', '_product')  # Handle duplicate column names
    )
    
    # Step 3: Add flag for coverage tracking
    df_merged['has_metadata'] = df_merged['title'].notna()
    
    return df_merged
```

**SQL Equivalent**:
```sql
SELECT a.*, p.title, p.main_category, p.price
FROM df_absa a
LEFT JOIN df_products p ON a.parent_asin = p.parent_asin
```

---

#### 3. **Data Quality Checks** (Lines 180-255)

**Problem**: Low-confidence predictions, generic aspects, duplicates.

**Solution**: Filter and validate data.

```python
def clean_merged_data(df, confidence_threshold=0.5):
    """
    Removes noise and adds derived fields.
    """
    
    # Step 1: Remove duplicate review-aspect pairs
    # Example: Review 123 mentioned "color" twice → keep only first
    df = df.drop_duplicates(subset=['review_id', 'aspect_term_normalized'], keep='first')
    
    # Step 2: Filter low-confidence predictions
    # Only keep predictions where model is >50% confident
    df = df[df['confidence'] >= confidence_threshold]
    
    # Step 3: Remove meaningless aspects
    exclude = ['thing', 'stuff', 'it', 'this']  # Too vague
    df = df[~df['aspect_term_normalized'].isin(exclude)]
    
    # Step 4: Calculate sentiment score
    # Convert sentiment label to numeric score
    def calc_sentiment_score(row):
        if row['sentiment'] == 'Positive':
            return row['confidence']    # 0 to +1
        elif row['sentiment'] == 'Negative':
            return -row['confidence']   # -1 to 0
        else:
            return 0                    # Neutral
    
    df['sentiment_score'] = df.apply(calc_sentiment_score, axis=1)
    
    # Step 5: Add temporal fields for analysis
    df['year_month'] = df['timestamp'].dt.to_period('M')  # 2020-01
    df['week'] = df['timestamp'].dt.to_period('W')         # 2020-W01
    df['date'] = df['timestamp'].dt.date                   # 2020-01-15
    
    return df
```

**Why Sentiment Score?**
- Easy to aggregate: `mean(sentiment_score)` shows overall sentiment
- Accounts for confidence: High-confidence negative is worse than low-confidence negative
- Enables time-series analysis: Track score over time

---

#### 4. **The Full Pipeline** (Lines 341-409)

**Orchestrates everything**:

```python
def preprocess_pipeline(df_absa, df_products, confidence_threshold=0.5, output_path=None):
    """
    Complete preprocessing workflow.
    
    Flow:
    1. Normalize aspects (color/colors → color)
    2. Merge with product metadata
    3. Clean and validate
    4. Generate report
    5. Save preprocessed data
    """
    
    # Keep original for report
    df_original = df_absa.copy()
    
    # Execute steps
    df_absa = normalize_aspect_terms(df_absa)
    df_merged = merge_dataframes(df_absa, df_products)
    df_cleaned = clean_merged_data(df_merged, confidence_threshold)
    report = generate_normalization_report(df_original, df_cleaned)
    
    # Save result
    if output_path:
        df_cleaned.to_parquet(output_path, index=False)
    
    return df_cleaned, report
```

---

## ⚙️ Module 2: `config.py`

### Purpose
Central configuration file - change settings without editing code.

### Key Concepts

```python
# ===== ASPECT MAPPINGS =====
# Add new mappings here instead of editing preprocess_data.py
ASPECT_MAPPING = {
    'color': 'color',
    'colors': 'color',
    # Add more...
}

# ===== THRESHOLDS =====
CONFIDENCE_THRESHOLD = 0.5          # Minimum confidence to include
MIN_REVIEWS_FOR_ANALYSIS = 10      # Minimum reviews per product
SENTIMENT_DROP_THRESHOLD = 0.20    # 20% drop triggers alert

# ===== COLORS =====
COLOR_PALETTE_SENTIMENT = {
    'Positive': '#2ecc71',  # Green
    'Negative': '#e74c3c',  # Red
    'Neutral': '#95a5a6'    # Gray
}
```

**Why This Matters**:
- **Easy customization**: Change thresholds without touching complex code
- **Consistency**: All files use same colors/thresholds
- **Maintainability**: Settings in one place

---

## 🛠️ Module 3: `utils.py`

### Purpose
Reusable helper functions to keep code DRY (Don't Repeat Yourself).

### Key Functions

#### 1. **Sentiment Score Calculation**

```python
def calculate_sentiment_score(sentiment, confidence):
    """
    Convert sentiment label to numeric score.
    
    Examples:
    - ('Positive', 0.9) → +0.9
    - ('Negative', 0.8) → -0.8
    - ('Neutral', 0.5) → 0
    """
    if sentiment == 'Positive':
        return confidence
    elif sentiment == 'Negative':
        return -confidence
    else:
        return 0
```

#### 2. **Rolling Sentiment Analysis**

```python
def get_rolling_sentiment(df, product_asin, window_days=7):
    """
    Calculate moving average sentiment over time.
    
    Example:
    Day 1: 0.5
    Day 2: 0.6
    Day 3: 0.4
    Day 4: 0.3  ← 7-day average = 0.45
    Day 5: 0.2  ← 7-day average = 0.40 (declining!)
    
    This smooths out daily noise and reveals trends.
    """
    product_data = df[df['parent_asin'] == product_asin].sort_values('date')
    product_data = product_data.set_index('date')
    
    # Calculate rolling mean over 7-day window
    rolling = product_data['sentiment_score'].rolling(
        f'{window_days}D',    # Window size
        min_periods=1         # Allow smaller windows at start
    ).mean()
    
    return rolling
```

**Visual Example**:
```
Daily scores:  ★★★★★ ★★★ ★★★★ ★★ ★ (volatile)
Rolling avg:   ★★★★ ★★★★ ★★★ ★★ ★  (smooth trend)
```

#### 3. **Top Aspects by Product**

```python
def get_top_aspects_by_product(df, product_asin, n=10):
    """
    Get most-mentioned aspects for a product.
    
    Returns DataFrame:
    aspect   | count | avg_sentiment | positive | negative
    color    | 150   | 0.45          | 120      | 30
    smell    | 100   | -0.20         | 40       | 60
    quality  | 80    | 0.60          | 70       | 10
    """
    product_data = df[df['parent_asin'] == product_asin]
    
    # Group by aspect and calculate stats
    aspect_stats = product_data.groupby('aspect_term_normalized').agg({
        'aspect_term': 'count',          # Total mentions
        'sentiment_score': 'mean',       # Average sentiment
        'is_positive': 'sum',            # Count positives
        'is_negative': 'sum',            # Count negatives
        'is_neutral': 'sum'              # Count neutrals
    }).reset_index()
    
    # Sort by most mentioned and take top N
    return aspect_stats.sort_values('count', ascending=False).head(n)
```

---

## 📊 Module 4: `dashboard.py` (Streamlit UI)

### Purpose
Interactive web interface for exploring the data.

### Streamlit Basics

**What is Streamlit?**
- Python library that turns Python scripts into web apps
- No HTML/CSS/JavaScript needed
- Automatic UI generation from Python code

**Basic Example**:
```python
import streamlit as st

# Create title
st.title("My Dashboard")

# Create slider
threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Create chart
st.line_chart(data)
```

### Dashboard Architecture

```python
# ===== PAGE SETUP =====
st.set_page_config(
    page_title="ABSA Dashboard",
    page_icon="📊",
    layout="wide"              # Use full screen width
)

# ===== DATA LOADING (with caching) =====
@st.cache_data                 # Only load once, then cache in memory
def load_data():
    df = pd.read_parquet('data/preprocessed_data.parquet')
    return df

df = load_data()

# ===== SIDEBAR NAVIGATION =====
page = st.sidebar.radio("Select Page", [
    "📊 Sentiment Overview",
    "🔍 Product Explorer",
    "🏷️ Aspect Analysis",
    "📈 Product Deep Dive",
    "🚨 Alerts & Anomalies"
])

# ===== SHOW SELECTED PAGE =====
if page == "📊 Sentiment Overview":
    # Show overview page
elif page == "🔍 Product Explorer":
    # Show product explorer
# ... etc
```

---

### Page 1: Sentiment Overview

**Purpose**: High-level KPIs and trends.

```python
# ===== KPI CARDS =====
col1, col2, col3, col4 = st.columns(4)  # Create 4 columns

with col1:
    st.metric("Total Reviews", f"{len(df):,}")
    #         ↑ Label        ↑ Value (formatted with commas)

with col2:
    st.metric("Unique Products", f"{df['parent_asin'].nunique():,}")
    #                              ↑ Count unique ASINs

# ===== PIE CHART =====
sentiment_counts = df['sentiment'].value_counts()
#                  ↑ Count occurrences: Positive=5000, Negative=2000, Neutral=1000

fig_pie = px.pie(
    values=sentiment_counts.values,      # [5000, 2000, 1000]
    names=sentiment_counts.index,        # ['Positive', 'Negative', 'Neutral']
    color=sentiment_counts.index,        # Color by sentiment
    color_discrete_map={                 # Custom colors
        'Positive': '#2ecc71',           # Green
        'Negative': '#e74c3c',           # Red
        'Neutral': '#95a5a6'             # Gray
    }
)
st.plotly_chart(fig_pie)

# ===== LINE CHART =====
# Group by date and sentiment, count reviews
trend = df.groupby(['date', 'sentiment']).size().reset_index(name='count')
#       ↑ Result:
#       date       | sentiment | count
#       2020-01-01 | Positive  | 100
#       2020-01-01 | Negative  | 50
#       2020-01-02 | Positive  | 120
#       2020-01-02 | Negative  | 40

fig_line = px.line(
    trend,
    x='date',
    y='count',
    color='sentiment',              # Separate line per sentiment
    title="Sentiment Trend Over Time"
)
st.plotly_chart(fig_line)
```

**Result**: Beautiful interactive charts without writing HTML!

---

### Page 2: Product Explorer

**Purpose**: Search and filter products.

**Key Feature: The Product Selector (What I Just Changed)**

```python
# ===== BUILD PRODUCT LIST (SORTED BY REVIEW COUNT) =====

# Count reviews per product
product_counts = df_filtered['parent_asin'].value_counts()
# Result:
# B08X1Y2Z3    500 reviews
# B09A2B3C4    350 reviews
# B07D5E6F7    200 reviews

# Get ASINs in order of review count
ordered_asins = product_counts.index.tolist()
# Result: ['B08X1Y2Z3', 'B09A2B3C4', 'B07D5E6F7', ...]

# Get product titles (for display)
title_map = df_filtered.groupby('parent_asin')['title'].first().to_dict()
# Result: {'B08X1Y2Z3': 'Blue Nail Polish', ...}

# ===== CREATE DROPDOWN =====
selected_product = st.selectbox(
    "Select Product",
    options=ordered_asins,                    # ASINs in review count order
    format_func=lambda x: f"{title_map.get(x, x)} ({x})"
    #            ↑ Display as: "Blue Nail Polish (B08X1Y2Z3)"
)
```

**Why This Matters**:
- **Before**: Products in random order → hard to find popular ones
- **After**: Most-reviewed products appear first → easier navigation

---

### Page 3: Aspect Analysis

**Purpose**: Analyze which aspects are mentioned most.

```python
# ===== TOP ASPECTS BAR CHART =====

# Count aspect mentions
aspect_counts = df['aspect_term_normalized'].value_counts().head(20)
# Result:
# color     5000
# quality   3000
# price     2500
# ...

# Create horizontal bar chart
fig = px.bar(
    x=aspect_counts.values,       # Counts on x-axis
    y=aspect_counts.index,        # Aspect names on y-axis
    orientation='h',              # Horizontal bars
    color=aspect_counts.values,   # Color by count (gradient)
    color_continuous_scale='Blues'
)

# ===== ASPECT SENTIMENT HEATMAP =====

# Get top 15 aspects and top 10 products
top_aspects = df['aspect_term_normalized'].value_counts().head(15).index
top_products = df['parent_asin'].value_counts().head(10).index

# Filter to these combinations
heatmap_data = df[
    (df['aspect_term_normalized'].isin(top_aspects)) &
    (df['parent_asin'].isin(top_products))
]

# Calculate average sentiment for each aspect-product pair
heatmap_pivot = heatmap_data.groupby([
    'aspect_term_normalized',
    'parent_asin'
])['sentiment_score'].mean().reset_index()

# Convert to matrix format (aspects × products)
matrix = heatmap_pivot.pivot(
    index='aspect_term_normalized',  # Rows
    columns='parent_asin',           # Columns
    values='sentiment_score'         # Cell values
)

# Create heatmap
fig = px.imshow(
    matrix,
    color_continuous_scale='RdYlGn',  # Red-Yellow-Green
    # Red = negative, Green = positive
)
```

**Result**:
```
              Product A  Product B  Product C
color         +0.8       +0.5       -0.2
quality       +0.6       +0.4       +0.3
price         -0.3       +0.1       -0.5
```

---

## 🚨 Module 5: `alert_system.py`

### Purpose
Automatically detect problems in the data.

### Alert Type 1: Sentiment Drops

**Problem**: Product sentiment declining but you don't notice until too late.

**Solution**: Calculate rolling averages and detect drops.

```python
def detect_sentiment_drops(self, threshold=0.20, window_days=7):
    """
    Find products with >20% sentiment drop in past week.
    """
    
    alerts = []
    
    for product in self.df['parent_asin'].unique():
        # Get product's reviews, sorted by date
        product_data = self.df[self.df['parent_asin'] == product].sort_values('date')
        
        # Calculate 7-day rolling average
        product_data = product_data.set_index('date')
        rolling = product_data['sentiment_score'].rolling('7D').mean()
        
        # Compare recent vs previous
        recent = rolling.iloc[-1]              # Latest 7-day average
        previous = rolling.iloc[-(7+1)]        # 7 days ago
        
        # Calculate % change
        change = (recent - previous) / abs(previous)
        
        # Alert if dropped >20%
        if change < -0.20:
            alerts.append({
                'product': product,
                'change': change,
                'recent_sentiment': recent,
                'previous_sentiment': previous
            })
    
    return pd.DataFrame(alerts)
```

**Example**:
```
Week 1 avg: 0.60 (good)
Week 2 avg: 0.45 (declining)
Week 3 avg: 0.35 (declining)
Week 4 avg: 0.20 (dropped 67%!) ← ALERT!
```

---

### Alert Type 2: Emerging Aspects

**Problem**: New complaints appearing but buried in data.

**Solution**: Compare recent vs historical aspects.

```python
def detect_emerging_aspects(self, lookback_days=30):
    """
    Find aspects that appeared in last 30 days but not before.
    """
    
    cutoff = self.df['date'].max() - timedelta(days=30)
    
    # Aspects mentioned in last 30 days
    recent_aspects = set(
        self.df[self.df['date'] > cutoff]['aspect_term_normalized'].unique()
    )
    
    # Aspects mentioned before that
    older_aspects = set(
        self.df[self.df['date'] <= cutoff]['aspect_term_normalized'].unique()
    )
    
    # New aspects = in recent but not in older
    new_aspects = recent_aspects - older_aspects
    
    return new_aspects
```

**Example**:
```
Before: color, quality, price, smell
Recent: color, quality, price, smell, leak, broken, defective

New aspects detected: leak, broken, defective ← ALERT!
```

---

### Alert Type 3: Rating-Sentiment Divergence

**Problem**: Product has 4.5★ rating but negative sentiments → mismatched signals.

**Solution**: Compare rating vs sentiment score.

```python
def detect_rating_sentiment_divergence(self):
    """
    Find products where rating ≠ sentiment.
    """
    
    for product in self.df['parent_asin'].unique():
        product_data = self.df[self.df['parent_asin'] == product]
        
        # Get product rating (1-5 stars)
        product_rating = product_data['average_rating'].iloc[0]  # e.g., 4.5
        
        # Calculate sentiment rating (convert -1 to +1 → 1 to 5)
        sentiment_score = product_data['sentiment_score'].mean()  # e.g., -0.2
        sentiment_rating = ((sentiment_score + 1) / 2) * 4 + 1    # → 3.2 stars
        
        # Calculate divergence
        divergence = abs(product_rating - sentiment_rating)       # |4.5 - 3.2| = 1.3
        
        # Alert if divergence >1.5 stars
        if divergence > 1.5:
            alerts.append({
                'product': product,
                'rating': product_rating,
                'sentiment_rating': sentiment_rating,
                'divergence': divergence,
                'interpretation': 'Rating higher than sentiment'
            })
```

**Why This Matters**:
- High rating + negative sentiment = Recent decline (old good reviews, new bad reviews)
- Low rating + positive sentiment = Improving product

---

## 🧠 Module 6: `topic_modeling.py`

### Purpose
Discover hidden themes in reviews using machine learning.

### Method 1: LDA (Latent Dirichlet Allocation)

**What is LDA?**
- Unsupervised learning algorithm
- Finds groups of words that frequently appear together
- These groups = "topics"

**Example**:
```
Topic 1: color, fade, vibrant, bright, pigment
  → About color quality

Topic 2: price, expensive, cheap, worth, value
  → About pricing

Topic 3: bottle, leak, pump, cap, broken
  → About packaging defects
```

**How It Works**:

```python
def extract_lda_topics(self, num_topics=10):
    """
    Extract topics from negative reviews.
    """
    
    # Step 1: Get negative reviews
    negative = self.df[self.df['sentiment'] == 'Negative']
    
    # Step 2: Preprocess text (tokenize, remove stopwords)
    def preprocess(text):
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
        return tokens
    
    documents = negative['text'].apply(preprocess).tolist()
    # Result: [['color', 'bad', 'fade'], ['price', 'expensive'], ...]
    
    # Step 3: Create dictionary (word → ID mapping)
    dictionary = corpora.Dictionary(documents)
    # Result: {'color': 0, 'bad': 1, 'fade': 2, ...}
    
    # Step 4: Convert documents to bag-of-words
    corpus = [dictionary.doc2bow(doc) for doc in documents]
    # Result: [[(0, 2), (1, 1), (2, 1)], ...]
    #          ↑ word_id=0 appears 2 times, word_id=1 appears 1 time
    
    # Step 5: Train LDA model
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=10              # Number of training passes
    )
    
    # Step 6: Extract topics
    topics = []
    for idx in range(num_topics):
        # Get top 10 words for this topic
        words = lda_model.show_topic(idx, topn=10)
        # Result: [('color', 0.05), ('fade', 0.04), ...]
        topics.append(words)
    
    return topics
```

---

### Method 2: BERT Clustering

**What is BERT?**
- Pre-trained neural network that understands text
- Converts text to vectors (embeddings)
- Similar texts → similar vectors

**How It Works**:

```python
def extract_bert_clusters(self, num_clusters=8):
    """
    Group similar reviews together.
    """
    
    # Step 1: Load pre-trained BERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Step 2: Convert reviews to embeddings
    texts = self.df['text'].tolist()
    embeddings = model.encode(texts)
    # Result: [[0.1, 0.5, -0.3, ...], [0.2, 0.4, -0.2, ...], ...]
    #         ↑ 384-dimensional vectors
    
    # Step 3: Cluster embeddings
    kmeans = KMeans(n_clusters=8)
    clusters = kmeans.fit_predict(embeddings)
    # Result: [0, 0, 1, 2, 1, 0, 2, ...] ← cluster assignments
    
    # Step 4: Analyze each cluster
    self.df['cluster'] = clusters
    
    for cluster_id in range(8):
        cluster_data = self.df[self.df['cluster'] == cluster_id]
        
        # Get top aspects in this cluster
        top_aspects = cluster_data['aspect_term_normalized'].value_counts().head(5)
        
        # Get sample reviews
        samples = cluster_data['text'].head(3).tolist()
        
        print(f"Cluster {cluster_id}:")
        print(f"  Top aspects: {top_aspects.index.tolist()}")
        print(f"  Sample: {samples[0][:100]}...")
```

**Example Result**:
```
Cluster 0 (250 reviews):
  Top aspects: color, fade, vibrant
  Theme: Color quality issues

Cluster 1 (180 reviews):
  Top aspects: price, expensive, value
  Theme: Pricing complaints

Cluster 2 (320 reviews):
  Top aspects: bottle, leak, pump
  Theme: Packaging defects
```

---

## 🎯 Putting It All Together

### The Complete Workflow

```
1. RAW DATA
   ├─ ABSA results: review_id, aspect_term, sentiment
   └─ Product data: parent_asin, title, category

        ↓ (preprocess_data.py)

2. CLEAN DATA
   ├─ Normalized aspects (color/colors → color)
   ├─ Merged with product info
   ├─ Quality filtered (confidence >0.5)
   └─ Saved to preprocessed_data.parquet

        ↓ (dashboard.py)

3. INTERACTIVE DASHBOARD
   ├─ Overview: KPIs, trends
   ├─ Product Explorer: Search, filter
   ├─ Aspect Analysis: Heatmaps, charts
   └─ Product Deep Dive: Individual analysis

        ↓ (alert_system.py)

4. AUTOMATED MONITORING
   ├─ Sentiment drop alerts
   ├─ Emerging aspect detection
   └─ Rating divergence warnings

        ↓ (topic_modeling.py)

5. DEEP INSIGHTS
   ├─ LDA topics (word groups)
   ├─ BERT clusters (similar reviews)
   └─ Category comparison
```

---

## 💡 Key Concepts Summary

### 1. **Data Normalization**
- **Problem**: Messy, inconsistent data
- **Solution**: Map variations to canonical forms
- **Benefit**: Clearer patterns, easier analysis

### 2. **DataFrame Operations**
- **Groupby**: Group rows by column, calculate stats per group
- **Merge**: Join two tables on common key
- **Pivot**: Reshape data from long to wide format

### 3. **Time-Series Analysis**
- **Rolling windows**: Smooth noisy data, reveal trends
- **Period conversion**: Group by week/month/year
- **Change detection**: Compare periods to spot shifts

### 4. **Visualization**
- **Streamlit**: Turn Python scripts into web apps
- **Plotly**: Interactive charts (zoom, hover, filter)
- **Layout**: Columns, expanders, tabs for organization

### 5. **Machine Learning**
- **LDA**: Unsupervised topic discovery
- **BERT**: Pre-trained text understanding
- **Clustering**: Group similar items automatically

---

## 🎓 Learning Path

If you want to understand the code even deeper:

### Level 1: Python Basics
1. Learn pandas DataFrames (groupby, merge, pivot)
2. Understand dictionaries and list comprehensions
3. Practice with datetime operations

### Level 2: Data Analysis
1. Study data normalization techniques
2. Learn SQL-like operations in pandas
3. Explore time-series analysis

### Level 3: Visualization
1. Try Streamlit tutorials
2. Learn Plotly for interactive charts
3. Understand dashboard design principles

### Level 4: Machine Learning
1. Study topic modeling (LDA)
2. Learn about embeddings and BERT
3. Explore clustering algorithms

---

## 🔗 Resources

- **Pandas Docs**: https://pandas.pydata.org/docs/
- **Streamlit Docs**: https://docs.streamlit.io/
- **Plotly Docs**: https://plotly.com/python/
- **Gensim (LDA)**: https://radimrehurek.com/gensim/
- **Sentence Transformers**: https://www.sbert.net/

---

## ❓ Common Questions

### Q: Why use parquet instead of CSV?
**A**: Parquet is faster and smaller (compressed). For 68K rows:
- CSV: ~15 MB, ~2 seconds to load
- Parquet: ~3 MB, ~0.5 seconds to load

### Q: What is `@st.cache_data`?
**A**: Streamlit decorator that caches function results. First run loads data, subsequent runs use cached version (faster).

### Q: Why normalize aspects?
**A**: Without normalization:
- "color" appears 1000 times
- "colors" appears 500 times
- "colour" appears 200 times
You'd think there are 3 aspects, but it's really 1 with 1700 mentions!

### Q: What is a rolling window?
**A**: Moving average over time. Like a "sliding calculator" that averages the past N days. Smooths noise, reveals trends.

### Q: How does LDA find topics?
**A**: LDA assumes:
1. Documents are mixtures of topics
2. Topics are mixtures of words
3. Uses probability to discover these mixtures

### Q: What is BERT?
**A**: Deep learning model trained on massive text corpus. It "understands" language by learning patterns. We use it to convert text to vectors.

---

## 🚀 Next Steps

Now that you understand the code, you can:

1. **Customize mappings**: Add your own aspect normalizations in `config.py`
2. **Adjust thresholds**: Fine-tune alert sensitivity
3. **Add features**: Create new visualizations or alert types
4. **Extend analysis**: Add more pages to dashboard
5. **Optimize performance**: Profile code, add caching

---

**Questions?** Check the code comments or the README.md for more details!
