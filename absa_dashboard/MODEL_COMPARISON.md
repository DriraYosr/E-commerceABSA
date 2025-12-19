# Forecasting Model Comparison

## Available Models

### 1. Prophet (Facebook's Time Series Model)
**Type**: Additive decomposition model  
**Equation**: `y(t) = g(t) + s(t) + h(t) + ε`

**Strengths:**
- ✅ Handles missing data automatically
- ✅ Detects weekly and yearly seasonality
- ✅ Robust to outliers
- ✅ Provides change point detection
- ✅ Includes anomaly detection
- ✅ Seasonality decomposition available
- ✅ Intuitive hyperparameters

**Best For:**
- Data with clear seasonal patterns
- Business forecasting (reviews over time)
- When you need to understand **why** sentiment changes
- Irregular time series with gaps

**Limitations:**
- Slower training (~2-5 seconds)
- Requires more data for good seasonality detection
- Can overfit with strong seasonality settings

---

### 2. ARIMA (AutoRegressive Integrated Moving Average)
**Type**: Statistical linear model  
**Equation**: `y(t) = φ₁y(t-1) + ... + θ₁ε(t-1) + ... + ε(t)`

**Strengths:**
- ✅ Fast training (<1 second)
- ✅ Works well with linear trends
- ✅ Good for stationary data
- ✅ Simpler, more interpretable
- ✅ Less prone to overfitting

**Best For:**
- Smooth, stationary time series
- When you want quick forecasts
- Simple linear trends
- Smaller datasets

**Limitations:**
- ❌ No automatic change point detection
- ❌ No built-in anomaly detection
- ❌ Limited seasonality handling (needs SARIMA)
- ❌ Assumes stationarity
- ❌ Less robust to missing data

---

## Implementation Details

### Current Configuration

**Prophet:**
- `changepoint_prior_scale=0.05` (moderate trend flexibility)
- `seasonality_prior_scale=1.0` (reduced for smoother forecasts)
- `interval_width=0.95` (95% confidence intervals)
- Weekly seasonality: Auto-enabled if data > 14 days
- Yearly seasonality: Auto-enabled if data > 365 days

**ARIMA:**
- `order=(1,1,1)` - AR(1), first-order differencing, MA(1)
- `seasonal_order=(0,0,0,0)` - No seasonal ARIMA components
- 95% confidence intervals via get_forecast()

---

## Feature Availability Matrix

| Feature | Prophet | ARIMA |
|---------|---------|-------|
| Forecasting | ✅ | ✅ |
| Confidence Intervals | ✅ | ✅ |
| Trend Analysis | ✅ | ✅ |
| Alerts | ✅ | ✅ |
| Change Point Detection | ✅ | ❌ |
| Anomaly Detection | ✅ | ❌ |
| Seasonality Decomposition | ✅ | ❌ |
| Training Speed | Slower | Faster |
| Handles Missing Data | ✅ | ⚠️ |

---

## Usage in Dashboard

### Model Selection
1. Navigate to **🔮 Sentiment Forecasting** page
2. Select model from dropdown (first column)
3. Configure forecast horizon, aggregation, min reviews
4. Choose aspect and click "Generate Forecast"

### When to Use Prophet:
- You have ≥30 days of data
- Reviews show weekly patterns (weekend vs weekday)
- You need to identify when sentiment changed
- You want to understand seasonal effects

### When to Use ARIMA:
- You want quick results
- Data is relatively smooth
- You prioritize speed over detail
- You have a simple trending pattern

---

## Performance Benchmarks

**Dataset**: 1000 reviews aggregated daily over 6 months

| Model | Training Time | Prediction Time | Memory |
|-------|--------------|-----------------|---------|
| Prophet | ~2-5 sec | <1 sec | ~200 MB |
| ARIMA | <1 sec | <0.5 sec | ~50 MB |

---

## Future Enhancements

### Planned:
- [ ] SARIMA support (seasonal ARIMA)
- [ ] Auto-ARIMA for automatic parameter selection
- [ ] Ensemble forecasting (combine models)
- [ ] LSTM neural network option
- [ ] Model performance comparison view

### Advanced Options:
- Custom Prophet seasonality
- External regressors (holidays, promotions)
- Multi-step ahead optimization
- Rolling window validation

---

## References

**Prophet:**
- Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37-45.
- https://facebook.github.io/prophet/

**ARIMA:**
- Box, G. E. P., & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*
- https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html

---

**Last Updated**: November 21, 2025  
**Version**: 1.1.0
