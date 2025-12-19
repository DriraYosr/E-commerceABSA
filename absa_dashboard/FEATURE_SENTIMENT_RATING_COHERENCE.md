# Sentiment-Rating Coherence Analysis Feature

## Overview
Comprehensive multi-view visualization suite comparing ABSA-detected sentiment scores against user-provided star ratings (1-5 scale). Now includes **four visualization types** plus an "All Views" mode.

## Purpose
This feature validates the coherence between:
- **ABSA Sentiment Score**: Confidence-weighted sentiment computed from aspect-level analysis (-1 to +1 scale)
- **Star Rating**: User-provided rating (1 to 5 stars)

The visualization helps identify:
1. Whether sentiment analysis results align with user ratings
2. Reviews where rating and sentiment strongly disagree (potential review quality issues, sarcasm, or model errors)
3. Overall correlation strength and distribution patterns
4. Statistical variance and outliers per rating level

## Visualization Types

### 1. **Interactive Scatter Plot**
- **Purpose**: Show individual reviews as points with interactive hover details
- **X-axis**: Star rating (1-5)
- **Y-axis**: Confidence-weighted sentiment score (-1 to +1)
- **Color**: Dominant sentiment category (Positive/Negative/Neutral)
- **Bubble Size**: Number of aspects extracted from the review
- **Ideal Correlation Line**: Gray dashed line showing expected sentiment-rating relationship
- **Best For**: Identifying individual outliers and understanding aspect count impact

**Key Improvements**:
- Uses **confidence-weighted sentiment** instead of simple average
- Bubble size shows aspect count (more aspects = more reliable sentiment)
- Hover data includes review_id, review_length, and aspect_count

### 2. **Hexbin Density Plot** (NEW)
- **Purpose**: Visualize concentration areas for large datasets
- **Visualization**: 2D histogram with warm color gradient (Yellow → Orange → Red)
- **Darker colors**: More reviews in that rating-sentiment combination
- **Overlay**: Cyan dashed line showing ideal correlation
- **Best For**: Large datasets (1000+ reviews) where scatter plots become cluttered

**Advantages**:
- Reveals density patterns invisible in scatter plots
- Identifies most common rating-sentiment combinations
- Shows if reviews cluster around expected values or deviate

### 3. **2D Histogram Density Map** (NEW)
- **Purpose**: Alternative density visualization with different binning
- **Visualization**: Continuous heatmap with Viridis color scale
- **Bins**: 10 horizontal (rating) × 20 vertical (sentiment) bins
- **Overlay**: Red dashed line with markers showing expected pattern
- **Best For**: Academic presentations and reports

**Advantages**:
- Smoother color gradients than hexbin
- Better for screenshots and figures in papers
- Correlation coefficient displayed in title

### 4. **Box Plots by Rating** (NEW)
- **Purpose**: Statistical distribution analysis per rating level
- **Shows**: Median, quartiles, and outliers for each star rating
- **Colored zones**:
  - Green shaded area (0.5 to 1.5): Expected for 4-5★ reviews
  - Red shaded area (-1.5 to -0.5): Expected for 1-2★ reviews
  - Neutral line at y=0
- **Outlier points**: Shows individual reviews far from median
- **Best For**: Understanding variance and detecting systematic biases

**Statistical Summary Included**:
Expandable table showing per-rating statistics:
- Count: Number of reviews
- Mean: Average sentiment score
- Median: Middle value
- Std: Standard deviation (spread)
- Min/Max: Range of sentiment scores

### 5. **All Views Mode** (NEW)
- Shows all four visualizations stacked vertically
- Allows comprehensive analysis in single scroll
- Ideal for exploratory data analysis sessions

## Key Metrics (Enhanced)

Four calculated metrics displayed below all visualizations:

#### 1. Pearson Correlation
- Measures linear relationship between rating and sentiment
- Range: -1 to +1
- **Interpretation**:
  - 0.7-1.0: Strong positive correlation (good model performance)
  - 0.4-0.7: Moderate correlation (acceptable)
  - <0.4: Weak correlation (model or data issues)

#### 2. Coherence Rate
Percentage of reviews where sentiment matches rating expectations:
- 4-5 stars → positive sentiment (score > 0)
- 1-2 stars → negative sentiment (score < 0)
- 3 stars → neutral sentiment (|score| < 0.3)

#### 3. Divergence Rate
Percentage of reviews with **strong** mismatches:
- 4-5 stars but negative sentiment (score < -0.3)
- 1-2 stars but positive sentiment (score > 0.3)
- Lower is better (indicates fewer errors)

#### 4. Avg Aspects/Review (NEW)
- Average number of aspects extracted per review
- Higher values typically mean more detailed reviews
- Useful for assessing extraction coverage

## Features

### Confidence-Weighted Sentiment
```python
def weighted_sentiment(group):
    if 'confidence' in group.columns:
        weights = group['confidence']
        return (group['sentiment_score'] * weights).sum() / weights.sum()
    return group['sentiment_score'].mean()
```
Gives more weight to high-confidence aspect predictions.

### Review Metadata Tracking
Each aggregated review includes:
- `review_length`: Character count of review text
- `aspect_count`: Number of aspects extracted
- Dominant sentiment category

### Divergent Review Inspector
Expandable section showing up to 10 examples where rating and sentiment strongly disagree:
- Shows review ID, rating, and sentiment score
- Displays first 200 characters of review text
- Useful for:
  - Quality control of ABSA model
  - Identifying sarcastic or complex reviews
  - Finding potential fake/suspicious reviews
  - Understanding edge cases

## Implementation Details

### Data Aggregation
```python
review_agg = df_filtered.groupby('review_id').agg({
    'sentiment_score': weighted_sentiment,  # Confidence-weighted average
    'rating': 'first',                      # Star rating (same for all aspects)
    'sentiment': lambda x: x.mode()[0],     # Most common sentiment category
    'text': lambda x: len(str(x.iloc[0])), # Review length
    'aspect_term_normalized': 'count'       # Number of aspects
}).reset_index()
```

### Visualization Selection
User can switch between views using radio buttons:
- Each visualization optimized for different use cases
- Common metrics displayed below all views
- Statistical summary available for box plots

## Usage

1. Navigate to **📊 Sentiment Overview** page in the dashboard
2. Scroll down past the sentiment trend charts
3. Select visualization type:
   - **Scatter Plot**: Individual review details
   - **Hexbin Density**: Concentration patterns
   - **2D Histogram**: Smooth density map
   - **Box Plots by Rating**: Statistical distribution
   - **All Views**: See everything
4. Review coherence metrics
5. Expand "View Divergent Reviews" to inspect mismatches
6. For box plots, expand "Statistical Summary by Rating" for detailed stats

## Requirements
- `rating` column must exist in preprocessed data
- Reviews must have both rating and sentiment_score values
- `confidence` column recommended for weighted aggregation

## Location
- **File**: `absa_dashboard/dashboard.py`
- **Section**: Sentiment Overview page (after sentiment score trend)
- **Lines**: ~217-490 (after existing time-series visualizations)

## Interpretation Guide

### What Good Coherence Looks Like
- Pearson correlation > 0.6
- Coherence rate > 70%
- Box plot medians follow expected pattern (5★→+1, 1★→-1)
- Density concentrated along ideal line
- Low variance within each rating level

### Warning Signs
- Correlation < 0.4 (model may need retraining)
- High divergence rate (>15%)
- Box plots show overlapping distributions
- Density concentrated away from ideal line
- High variance for 4-5★ reviews (inconsistent positive detection)

### Common Patterns
1. **Inflation at 5★**: Many 5★ reviews with neutral/negative sentiment
   - Possible causes: Default 5★ ratings, brief reviews, spam
2. **Deflation at 1★**: Many 1★ reviews with neutral/positive sentiment
   - Possible causes: Shipping issues (not product quality), misclassification
3. **3★ variance**: Wide spread of sentiment for 3★ reviews
   - Expected: 3★ is genuinely mixed/neutral feedback

## Future Enhancements
- [ ] Add filtering by divergence type (high rating + negative vs low rating + positive)
- [ ] Export divergent reviews for manual labeling/validation
- [ ] Time-based coherence tracking to detect model drift
- [ ] Product-level coherence breakdown (which products have poor coherence?)
- [ ] Aspect-category coherence (quality aspects vs delivery aspects)
- [ ] Interactive threshold adjustment for coherence/divergence definitions
- [ ] Confidence interval bands on box plots
- [ ] Violin plots as alternative to box plots
- [ ] Download buttons for each visualization (PNG/SVG export)
