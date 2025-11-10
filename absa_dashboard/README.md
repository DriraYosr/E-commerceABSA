# ABSA Interactive Dashboard

Comprehensive Aspect-Based Sentiment Analysis (ABSA) dashboard with preprocessing, visualization, alerts, and topic modeling.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Preprocessing](#data-preprocessing)
- [Dashboard Usage](#dashboard-usage)
- [Alert System](#alert-system)
- [Topic Modeling](#topic-modeling)
- [Configuration](#configuration)
- [Project Structure](#project-structure)

---

## 🎯 Overview

This dashboard analyzes customer reviews using Aspect-Based Sentiment Analysis (ABSA) to provide:
- **Product-level insights**: Identify top products, sentiment trends, and review patterns
- **Aspect analysis**: Track specific product features (color, quality, smell, etc.) mentioned in reviews
- **Real-time alerts**: Detect sentiment drops, emerging concerns, and anomalies
- **Topic modeling**: Discover hidden themes in negative reviews using LDA and BERT

---

## ✨ Features

### 1. Data Preprocessing
- **Aspect normalization**: Handles synonyms (smell/scent/fragrance → smell), plurals (color/colors), and case variations
- **DataFrame merging**: Joins ABSA results with product metadata (title, category, ratings, price)
- **Data quality checks**: Filters low-confidence predictions, removes generic aspects, handles duplicates
- **Derived fields**: Sentiment scores, temporal aggregations, sentiment flags

### 2. Interactive Dashboard (Streamlit)
- **Page 1 - Sentiment Overview**: KPIs, sentiment distribution, trend charts
- **Page 2 - Product Explorer**: Search/filter products, product cards with top aspects
- **Page 3 - Aspect Analysis**: Top aspects, sentiment heatmaps, aspect evolution over time
- **Page 4 - Product Deep Dive**: Detailed analysis for individual products
- **Page 5 - Alerts & Anomalies**: Real-time sentiment drop alerts, emerging aspects

### 3. Automated Alert System
- **Sentiment shift detection**: Rolling 7-day sentiment analysis with configurable thresholds
- **Category-level monitoring**: Track trends at product category level
- **Emerging aspect detection**: Identify new concerns appearing in recent reviews
- **Rating-sentiment divergence**: Find products where sentiment differs from ratings

### 4. Topic Modeling
- **LDA topic extraction**: Discover latent topics in negative reviews
- **BERT clustering**: Group similar reviews using sentence embeddings
- **Category comparison**: Identify universal vs category-specific issues
- **Aspect integration**: Link topics to detected aspects for actionable insights

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip

### Install Dependencies

```bash
cd absa_dashboard
pip install -r requirements.txt
```

### Download NLTK Data (for topic modeling)

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## ⚡ Quick Start

### Step 1: Prepare Your Data

Ensure you have:
1. **ABSA results** (`df`): DataFrame with columns:
   - `review_id`, `rating`, `text`, `timestamp`, `user_id`, `parent_asin`
   - `aspect_term`, `sentiment`, `confidence`

2. **Product metadata** (`product_df`): DataFrame with:
   - `parent_asin` (JOIN KEY)
   - `title`, `main_category`, `average_rating`, `price`, etc.

### Step 2: Run Preprocessing

```python
from preprocess_data import preprocess_pipeline

# Execute preprocessing
df_cleaned, report = preprocess_pipeline(
    df_absa=df,
    df_products=product_df,
    confidence_threshold=0.5,
    output_path='data/preprocessed_data.parquet'
)
```

### Step 3: Launch Dashboard

```bash
streamlit run dashboard.py
```

Open your browser to `http://localhost:8501`

---

## 🔧 Data Preprocessing

### Aspect Normalization Rules

The preprocessing module applies comprehensive normalization:

```python
# Example mappings
'color', 'colors', 'colour' → 'color'
'smell', 'scent', 'fragrance' → 'smell'
'price', 'cost', 'pricing' → 'price'
'bottle', 'jar', 'tube' → 'bottle'
```

**Update mappings** in `config.py`:

```python
ASPECT_MAPPING = {
    'your_variant': 'canonical_form',
    # Add more as needed
}
```

### DataFrame Merging

Joins ABSA results with product metadata:

```python
def merge_dataframes(df_absa, df_products):
    # Ensures parent_asin consistency
    df_merged = df_absa.merge(
        df_products,
        on='parent_asin',
        how='left',  # Keeps all ABSA results
        suffixes=('_review', '_product')
    )
    return df_merged
```

### Data Quality Filters

Applied automatically:
- ✅ Confidence threshold (default: 0.5)
- ✅ Remove duplicates (same review_id + aspect)
- ✅ Exclude generic aspects ('thing', 'stuff', 'it', etc.)
- ✅ Handle missing timestamps

---

## 📊 Dashboard Usage

### Global Filters (Sidebar)

All pages respect these filters:
- **Date Range**: Filter by review date
- **Product Category**: Filter by main_category (if available)
- **Minimum Confidence**: Adjust confidence threshold

### Page-Specific Features

#### 1. Sentiment Overview
- **KPIs**: Total reviews, products, aspects, confidence
- **Pie Chart**: Overall sentiment distribution
- **Line Chart**: Sentiment trends (daily/weekly/monthly)
- **Area Chart**: Sentiment score evolution

#### 2. Product Explorer
- **Search Box**: Find products by title
- **Rating Range**: Filter by product rating (1-5 stars)
- **Minimum Reviews**: Set review count threshold
- **Product Cards**: Expandable cards showing:
  - Review count, rating, sentiment score
  - Top 3 positive/negative aspects
  - Sample reviews

#### 3. Aspect Analysis
- **Bar Chart**: Top 20 aspects by mention count
- **Heatmap**: Aspect sentiment by product (normalized aspects)
- **Line Chart**: Aspect evolution over time
- **Multi-select**: Track multiple aspects simultaneously

#### 4. Product Deep Dive
- **Product Selector**: Autocomplete search by title
- **Metadata Card**: Product info, category, ratings
- **Pie Chart**: Aspect distribution (normalized)
- **Bar Chart**: Sentiment by aspect
- **Line Chart**: Daily sentiment trajectory
- **Comparison**: Product vs category averages

#### 5. Alerts & Anomalies
- **Sentiment Drop Table**: Products with >20% drops
- **Emerging Aspects**: New aspects in past 30 days
- **Divergence Detection**: Rating vs sentiment mismatches
- **Export**: Download alerts as CSV

---

## 🚨 Alert System

### Running Alerts Manually

```python
from alert_system import AlertSystem

# Initialize
alert_system = AlertSystem(df_preprocessed)

# Generate comprehensive report
report = alert_system.generate_alert_report(
    output_path='exports/alert_report.xlsx'
)

# Access individual alert types
sentiment_drops = report['sentiment_drops']
category_trends = report['category_trends']
emerging_aspects = report['emerging_aspects']
rating_divergence = report['rating_divergence']
```

### Alert Types

| Alert Type | Description | Threshold |
|------------|-------------|-----------|
| **Sentiment Drop** | Product sentiment declined >20% in past 7 days | Configurable in `config.py` |
| **Category Trend** | Entire category shows negative trend | 30-day comparison |
| **Emerging Aspect** | New aspect appearing in recent reviews | Min 5 mentions |
| **Rating Divergence** | Sentiment differs from product rating by >1.5 stars | Normalized 1-5 scale |

### Customizing Thresholds

Edit `config.py`:

```python
SENTIMENT_DROP_THRESHOLD = 0.20  # 20% drop
ROLLING_WINDOW_DAYS = 7  # 7-day window
MIN_REVIEWS_FOR_ALERT = 5  # Minimum reviews
```

---

## 🧠 Topic Modeling

### Running Topic Modeling

```python
from topic_modeling import TopicModeler

# Initialize
topic_modeler = TopicModeler(df_preprocessed)

# Generate comprehensive report
report = topic_modeler.generate_topic_report(
    output_path='exports/topic_report.json'
)
```

### LDA Topics

Extracts topics from negative reviews:

```python
topics = topic_modeler.extract_lda_topics(
    sentiment_filter='Negative',
    num_topics=10,
    category='Beauty',  # Optional
    min_reviews=20
)

# Access topics
for topic in topics['topics']:
    print(f"Topic {topic['topic_id']}: {topic['words']}")
    print(f"Top aspects: {topic['top_aspects']}")
```

### BERT Clustering

Groups similar reviews using embeddings:

```python
clusters = topic_modeler.extract_bert_clusters(
    num_clusters=8,
    sentiment_filter='Negative',
    max_reviews=1000  # For performance
)

# Access clusters
for cluster in clusters['clusters']:
    print(f"Cluster {cluster['cluster_id']} (size: {cluster['size']})")
    print(f"Avg sentiment: {cluster['avg_sentiment']:.2f}")
    print(f"Top aspects: {cluster['top_aspects']}")
```

### Category Comparison

Find universal vs category-specific issues:

```python
comparison = topic_modeler.compare_categories(num_topics=5)

# Universal concerns (appear across categories)
print(comparison['universal_concerns'])

# Category-specific issues
print(comparison['category_specific'])
```

---

## ⚙️ Configuration

All settings in `config.py`:

### Aspect Mapping

```python
ASPECT_MAPPING = {
    'color': 'color',
    'colors': 'color',
    # Add your mappings
}
```

### Thresholds

```python
CONFIDENCE_THRESHOLD = 0.5
MIN_REVIEWS_FOR_ANALYSIS = 10
SENTIMENT_DROP_THRESHOLD = 0.20
ROLLING_WINDOW_DAYS = 7
```

### Visualization Settings

```python
COLOR_PALETTE_SENTIMENT = {
    'Positive': '#2ecc71',  # Green
    'Negative': '#e74c3c',  # Red
    'Neutral': '#95a5a6'    # Gray
}

TOP_N_ASPECTS = 20
CHART_HEIGHT = 500
```

### Data Paths

```python
DATA_DIR = 'data'
PREPROCESSED_DATA_FILE = 'preprocessed_data.parquet'
ABSA_RESULTS_FILE = 'absa_results.csv'
PRODUCT_METADATA_FILE = 'full-00000-of-00001.parquet'
```

---

## 📁 Project Structure

```
absa_dashboard/
├── preprocess_data.py       # Data normalization and merging
├── dashboard.py              # Main Streamlit app
├── alert_system.py           # Alert detection logic
├── topic_modeling.py         # LDA and BERT analysis
├── utils.py                  # Helper functions
├── config.py                 # Configuration (mappings, thresholds)
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── data/
│   ├── absa_results.csv          # Input: ABSA results
│   ├── full-00000-of-00001.parquet  # Input: Product metadata
│   └── preprocessed_data.parquet    # Output: Preprocessed data
└── exports/
    ├── alert_report.xlsx         # Alert system output
    └── topic_report.json         # Topic modeling output
```

---

## 🔍 Troubleshooting

### Issue: "Data file not found"
**Solution**: Run `preprocess_data.py` first to generate `preprocessed_data.parquet`

### Issue: "No category information available"
**Solution**: Ensure `product_df` contains `main_category` column when merging

### Issue: Topic modeling fails
**Solution**: Install optional dependencies:
```bash
pip install gensim sentence-transformers nltk spacy
python -m spacy download en_core_web_sm
```

### Issue: Dashboard won't start
**Solution**: Check port availability:
```bash
streamlit run dashboard.py --server.port 8502
```

---

## 📚 Additional Resources

- **ABSA Research**: [Survey on ABSA](https://arxiv.org/abs/1910.00883)
- **Streamlit Docs**: [https://docs.streamlit.io](https://docs.streamlit.io)
- **PyABSA Library**: [https://github.com/yangheng95/PyABSA](https://github.com/yangheng95/PyABSA)

---

## 📄 License

This project is provided as-is for educational and research purposes.

---

## 🤝 Contributing

To add new features:

1. **New aspect mappings**: Update `ASPECT_MAPPING` in `config.py`
2. **Custom alerts**: Add methods to `AlertSystem` class in `alert_system.py`
3. **Dashboard pages**: Add new pages in `dashboard.py` following existing patterns
4. **Visualization functions**: Add to `utils.py`

---

## 📧 Support

For questions or issues:
1. Check this README and code comments
2. Review configuration in `config.py`
3. Examine normalization report for unmapped aspects
4. Check alert reports for data quality issues

---

**Happy Analyzing! 📊✨**
