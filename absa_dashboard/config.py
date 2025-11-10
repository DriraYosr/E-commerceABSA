"""
Configuration file for ABSA Dashboard
======================================
Contains aspect mapping, thresholds, and other configurable parameters.
"""

# ==================== ASPECT NORMALIZATION MAPPING ====================
# Extended mapping for aspect term normalization
# Add new mappings here as needed

ASPECT_MAPPING = {
    # Color variations
    'color': 'color',
    'colors': 'color',
    'colour': 'color',
    'colours': 'color',
    'shade': 'color',
    'shades': 'color',
    'tint': 'color',
    
    # Quality variations
    'quality': 'quality',
    'qualities': 'quality',
    
    # Smell variations
    'smell': 'smell',
    'smells': 'smell',
    'scent': 'smell',
    'scents': 'smell',
    'fragrance': 'smell',
    'fragrances': 'smell',
    'aroma': 'smell',
    'aromas': 'smell',
    'odor': 'smell',
    'odour': 'smell',
    
    # Price variations
    'price': 'price',
    'prices': 'price',
    'cost': 'price',
    'costs': 'price',
    'pricing': 'price',
    'value': 'price',
    
    # Packaging variations
    'package': 'packaging',
    'packages': 'packaging',
    'packaging': 'packaging',
    'container': 'packaging',
    'containers': 'packaging',
    'box': 'packaging',
    'boxes': 'packaging',
    
    # Texture variations
    'texture': 'texture',
    'textures': 'texture',
    'consistency': 'texture',
    'feel': 'texture',
    
    # Size variations
    'size': 'size',
    'sizes': 'size',
    'amount': 'size',
    'quantity': 'size',
    
    # Bottle variations
    'bottle': 'bottle',
    'bottles': 'bottle',
    'jar': 'bottle',
    'jars': 'bottle',
    'tube': 'bottle',
    'tubes': 'bottle',
    
    # Product variations
    'product': 'product',
    'products': 'product',
    'item': 'product',
    'items': 'product',
    
    # Use/Usage variations
    'use': 'usage',
    'uses': 'usage',
    'usage': 'usage',
    'application': 'usage',
    'applications': 'usage',
    
    # Skin variations
    'skin': 'skin',
    'skins': 'skin',
    'complexion': 'skin',
    
    # Hair variations
    'hair': 'hair',
    'hairs': 'hair',
    
    # Effect variations
    'effect': 'effect',
    'effects': 'effect',
    'result': 'effect',
    'results': 'effect',
    
    # Ingredient variations
    'ingredient': 'ingredient',
    'ingredients': 'ingredient',
    'formula': 'ingredient',
    'formulation': 'ingredient',
    
    # Brand variations
    'brand': 'brand',
    'brands': 'brand',
    'company': 'brand',
    
    # Delivery variations
    'delivery': 'delivery',
    'shipping': 'delivery',
    'shipment': 'delivery',
    
    # Lasting variations
    'lasting': 'lasting',
    'longevity': 'lasting',
    'duration': 'lasting',
    'long-lasting': 'lasting',
    'longlasting': 'lasting',
}


# ==================== GENERIC/MEANINGLESS ASPECTS ====================
# Aspects to exclude from analysis (too vague or uninformative)

EXCLUDE_ASPECTS = [
    'thing', 'things', 'stuff', 'something', 'anything', 'everything',
    'it', 'this', 'that', 'these', 'those', 'one', 'ones',
    'part', 'parts', 'piece', 'pieces'
]


# ==================== THRESHOLDS ====================

# Data Quality Thresholds
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for aspect predictions
MIN_REVIEWS_FOR_ANALYSIS = 10  # Minimum reviews to include a product in analysis

# Alert System Thresholds
SENTIMENT_DROP_THRESHOLD = 0.20  # 20% drop triggers alert
ROLLING_WINDOW_DAYS = 7  # 7-day rolling window for trend detection
MIN_REVIEWS_FOR_ALERT = 5  # Minimum reviews in period to trigger alert

# Aspect Frequency Thresholds
TOP_N_ASPECTS = 20  # Number of top aspects to show in visualizations
MIN_ASPECT_MENTIONS = 3  # Minimum mentions to include aspect in analysis


# ==================== VISUALIZATION SETTINGS ====================

# Color Palettes
COLOR_PALETTE_SENTIMENT = {
    'Positive': '#2ecc71',  # Green
    'Negative': '#e74c3c',  # Red
    'Neutral': '#95a5a6'    # Gray
}

COLOR_PALETTE_CATEGORY = 'plotly'  # Plotly default palette

# Chart Defaults
CHART_HEIGHT = 500
CHART_WIDTH = 800
FONT_SIZE = 12


# ==================== DATA PATHS ====================

DATA_DIR = 'data'
PREPROCESSED_DATA_FILE = 'preprocessed_data.parquet'
ABSA_RESULTS_FILE = 'absa_results.csv'
PRODUCT_METADATA_FILE = 'full-00000-of-00001.parquet'


# ==================== STREAMLIT SETTINGS ====================

PAGE_TITLE = "ABSA Dashboard"
PAGE_ICON = "📊"
LAYOUT = "wide"
SIDEBAR_STATE = "expanded"


# ==================== TOPIC MODELING SETTINGS ====================

# LDA Parameters
NUM_TOPICS_LDA = 10
PASSES_LDA = 10
ITERATIONS_LDA = 100

# BERT Parameters
BERT_MODEL = 'all-MiniLM-L6-v2'
CLUSTERING_METHOD = 'kmeans'
NUM_CLUSTERS = 8


# ==================== EXPORT SETTINGS ====================

EXPORT_FORMATS = ['CSV', 'Excel', 'JSON']
EXPORT_DIR = 'exports'
