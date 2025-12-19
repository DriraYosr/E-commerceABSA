# Sentiment Forecasting Feature
## Time Series Prediction for Aspect-Level Sentiment

### Overview
The Sentiment Forecasting module uses **Facebook Prophet** to predict future sentiment trends for specific product aspects. This enables proactive decision-making by identifying potential issues before they escalate.

---

## Features

### 1. **Time Series Forecasting**
- **Multi-horizon predictions**: 30, 60, 90, or 180 days ahead
- **Confidence intervals**: 80% and 95% uncertainty bands
- **Automatic seasonality detection**: Weekly, monthly, yearly patterns
- **Trend decomposition**: Separate long-term trend from seasonal effects

### 2. **Change Point Detection**
- **CUSUM algorithm**: Detects significant sentiment shifts
- **Automatic change point identification**: Highlights dates when sentiment changed rapidly
- **Use cases**:
  - Product recalls or quality issues
  - Competitive product launches
  - Supply chain disruptions
  - Marketing campaign impacts

### 3. **Anomaly Detection**
- **Statistical outlier identification**: Points outside 95% confidence interval
- **Anomaly scoring**: Quantifies how unusual each data point is
- **Visual flagging**: Red markers on timeline for easy identification
- **Use cases**:
  - Fake review detection (sudden sentiment spikes)
  - Data quality issues
  - One-time events (viral posts, media coverage)

### 4. **Trend Analysis**
- **Slope calculation**: Quantifies rate of sentiment change
- **Trend classification**: Improving / Declining / Stable
- **Volatility metrics**: Measures sentiment consistency
- **Threshold alerts**: Warns when sentiment predicted to turn negative

### 5. **Seasonality Decomposition**
- **Additive decomposition**: Separates time series into components:
  - **Trend**: Long-term direction
  - **Seasonal**: Repeating patterns (weekly/monthly cycles)
  - **Residual**: Random noise
- **Interpretation**:
  - Strong seasonality → day-of-week or monthly patterns
  - High residual variance → unpredictable sentiment

---

## Usage Guide

### Basic Forecasting Workflow

1. **Navigate** to "🔮 Sentiment Forecasting" page
2. **Configure parameters**:
   - Forecast horizon: 30-180 days
   - Aggregation: Daily, Weekly, or Monthly
   - Minimum reviews: Data quality threshold
3. **Select aspect** from top aspects (sorted by frequency)
4. **Click "Generate Forecast"**
5. **Interpret results** using visualizations and metrics

### Interpreting Forecast Plots

#### Main Forecast Chart
- **Blue dots**: Historical sentiment data
- **Blue line**: Predicted sentiment trajectory
- **Light blue band**: 95% confidence interval (wider = more uncertainty)
- **Red dashed line**: Today's date (separates history from forecast)

#### Key Metrics Dashboard
1. **Current Sentiment**: Latest predicted value
2. **Trend Direction**: "Improving", "Declining", or "Stable"
3. **30-Day Forecast**: Predicted sentiment in 1 month
4. **90-Day Forecast**: Predicted sentiment in 3 months

#### Alerts
- **Negative Threshold**: Sentiment predicted to drop below 0
- **Rapid Decline**: Slope exceeds -0.02/day (accelerating problems)

---

## Technical Details

### Model: Facebook Prophet
Prophet is designed for business time series with:
- **Strong seasonal effects**: Handles weekly/monthly patterns
- **Multiple seasonality**: Can model both short-term and long-term cycles
- **Robustness to missing data**: Handles gaps in review timestamps
- **Interpretable parameters**: Changepoint and seasonality controls

### Hyperparameters

```python
SentimentForecaster(
    changepoint_prior_scale=0.05,  # Flexibility of trend changes (0.001-0.5)
    seasonality_prior_scale=10.0,   # Strength of seasonality (0.01-10)
    interval_width=0.95             # Confidence interval width (0-1)
)
```

**Tuning guidance**:
- **High changepoint_prior_scale**: Model adapts quickly to trend changes (risk: overfitting)
- **Low changepoint_prior_scale**: Smooth trend, less reactive (risk: missing real shifts)
- **High seasonality_prior_scale**: Strong repeating patterns
- **Low seasonality_prior_scale**: Minimal seasonal effects

### Data Preprocessing

1. **Aggregation**: Reviews grouped by date (daily/weekly/monthly)
2. **Outlier removal**: 3-sigma rule removes extreme values
3. **Minimum sample filter**: Dates with <3 reviews excluded
4. **Missing value handling**: Prophet interpolates gaps automatically

### Requirements

```bash
pip install prophet>=1.1.5
pip install statsmodels>=0.14.0
```

**Note**: Prophet requires C++ compiler on Windows. Alternative:
```bash
conda install -c conda-forge prophet
```

---

## Use Cases & Business Value

### 1. **Proactive Quality Management**
- **Scenario**: "Packaging" sentiment declining steadily
- **Action**: Contact supplier before complaints escalate
- **Value**: Reduce negative reviews, improve customer satisfaction

### 2. **Inventory & Logistics Planning**
- **Scenario**: "Delivery" sentiment predicted to drop during holidays
- **Action**: Increase warehouse capacity proactively
- **Value**: Maintain service quality during peak demand

### 3. **Product Development Prioritization**
- **Scenario**: "Battery life" sentiment improving post-update
- **Action**: Validate engineering changes are working
- **Value**: Data-driven product roadmap decisions

### 4. **Competitive Analysis**
- **Scenario**: "Price" sentiment dropping after competitor discount
- **Action**: Adjust pricing strategy or improve value messaging
- **Value**: Stay competitive in dynamic markets

### 5. **Marketing Campaign Timing**
- **Scenario**: "Quality" sentiment at seasonal peak
- **Action**: Launch promotional campaign when perception is highest
- **Value**: Maximize campaign ROI

---

## Limitations & Considerations

### Data Requirements
- **Minimum samples**: 10 data points (preferably 30+)
- **Time coverage**: At least 2 weeks for meaningful patterns
- **Review frequency**: Inconsistent reviews reduce forecast accuracy

### Model Assumptions
- **Linearity**: Assumes trends are piecewise linear
- **Additive seasonality**: Seasonal effects add to trend (not multiplicative)
- **Stationarity**: Long-term behavior doesn't fundamentally change

### When Forecasts May Fail
1. **Black swan events**: COVID-19, natural disasters (unpredictable)
2. **Product discontinuation**: No future data to learn from
3. **Viral reviews**: Sudden spikes don't predict long-term trends
4. **Insufficient data**: <10 reviews or <2 weeks of history

### Best Practices
- **Validate regularly**: Compare predictions to actuals weekly
- **Update frequently**: Retrain models as new data arrives
- **Combine with domain expertise**: Use forecasts as one input, not sole decision driver
- **Monitor confidence intervals**: Wide bands = high uncertainty
- **Check residuals**: Large residuals suggest missing variables

---

## API Reference

### `forecast_aspect_sentiment()`
```python
result = forecast_aspect_sentiment(
    df=reviews_df,           # DataFrame with review data
    aspect='quality',         # Aspect term to forecast
    forecast_days=90,         # Prediction horizon
    freq='D'                  # 'D'=daily, 'W'=weekly, 'M'=monthly
)
```

**Returns**: Dictionary with:
- `success`: Boolean
- `forecast`: Prophet forecast DataFrame
- `historical`: Original time series data
- `trend_analysis`: Trend metrics and alerts
- `changepoints`: Detected shift dates
- `anomalies`: Outlier-flagged data
- `decomposition`: Seasonality breakdown

### `SentimentForecaster` Class
```python
forecaster = SentimentForecaster(
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    interval_width=0.95
)

# Prepare data
ts_data = forecaster.prepare_data(df, aspect='delivery', freq='D')

# Train model
forecaster.fit(ts_data, aspect='delivery')

# Generate predictions
forecast = forecaster.predict('delivery', periods=90, freq='D')

# Analyze results
analysis = forecaster.analyze_trend(forecast)
anomalies = forecaster.detect_anomalies(forecast, ts_data)
```

---

## Advanced Features

### Batch Forecasting
Generate forecasts for multiple aspects simultaneously:

```python
from forecasting import batch_forecast_aspects

results = batch_forecast_aspects(
    df=reviews_df,
    aspects=['quality', 'delivery', 'price', 'packaging'],
    forecast_days=90
)
```

### Custom Seasonality
Add domain-specific seasonal patterns:

```python
model = Prophet()
model.add_seasonality(
    name='quarterly',
    period=91.25,  # ~3 months
    fourier_order=5
)
```

### External Regressors
Include external events (e.g., promotions, holidays):

```python
model = Prophet()
model.add_regressor('is_holiday')
model.add_regressor('discount_active')
```

---

## Troubleshooting

### "Insufficient data" Error
- **Cause**: <10 reviews for selected aspect/timeframe
- **Fix**: Lower minimum review threshold or select more popular aspect

### Prophet Installation Fails
- **Windows**: Use conda instead of pip
  ```bash
  conda install -c conda-forge prophet
  ```
- **Linux/Mac**: Install build tools
  ```bash
  sudo apt-get install python3-dev build-essential
  ```

### Forecast Confidence Intervals Too Wide
- **Cause**: High data variability or insufficient samples
- **Fix**: 
  - Increase aggregation (daily → weekly)
  - Collect more data over time
  - Lower `interval_width` parameter (e.g., 0.80)

### Seasonal Pattern Not Detected
- **Cause**: Insufficient history (need 2+ full cycles)
- **Fix**: Wait for more data or disable seasonality manually

---

## Future Enhancements

1. **Multi-variate forecasting**: Predict sentiment based on review volume, rating, etc.
2. **Ensemble models**: Combine Prophet with ARIMA, LSTM
3. **Cross-aspect dependencies**: Model how "delivery" affects "quality" perception
4. **Automated retraining**: Schedule model updates daily/weekly
5. **Alert automation**: Email/Slack notifications for critical forecasts
6. **Forecast comparison**: A/B test different model configurations

---

## References

- [Prophet Documentation](https://facebook.github.io/prophet/)
- [Time Series Analysis in Python](https://otexts.com/fpp3/)
- [CUSUM Change Point Detection](https://en.wikipedia.org/wiki/CUSUM)
- [Seasonal Decomposition](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html)

---

## Support

For issues or questions:
1. Check minimum data requirements (10+ reviews, 2+ weeks)
2. Verify Prophet installation: `python -c "import prophet; print(prophet.__version__)"`
3. Review console logs for detailed error messages
4. Consult documentation above for common scenarios
