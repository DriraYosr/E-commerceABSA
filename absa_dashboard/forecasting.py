"""
Sentiment Forecasting Module
=============================
Time series forecasting for aspect-level sentiment trends using Prophet and ARIMA.

Features:
- Multiple models: Prophet, ARIMA
- Aspect-specific sentiment forecasting
- Confidence intervals (80%, 95%)
- Anomaly detection on historical data
- Change point detection
- Seasonality decomposition
- Multi-step ahead predictions (1, 3, 6 months)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not installed. Install with: pip install prophet")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("Warning: ARIMA not available. Install with: pip install statsmodels")

from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats


class SentimentForecaster:
    """
    Forecast aspect-level sentiment using Prophet time series model.
    """
    
    def __init__(self, changepoint_prior_scale: float = 0.05, 
                 seasonality_prior_scale: float = 1.0,
                 interval_width: float = 0.95):
        """
        Initialize forecaster with Prophet hyperparameters.
        
        Args:
            changepoint_prior_scale: Flexibility of trend changes (0.001-0.5)
            seasonality_prior_scale: Strength of seasonality component (0.01-10)
            interval_width: Width of uncertainty intervals (0-1)
        """
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.interval_width = interval_width
        self.models = {}  # Store trained models per aspect
        
    def prepare_data(self, df: pd.DataFrame, aspect: str, 
                     date_col: str = 'timestamp', 
                     sentiment_col: str = 'sentiment_score',
                     aspect_col: str = 'aspect',
                     freq: str = 'D',
                     min_samples_per_period: int = 5) -> pd.DataFrame:
        """
        Prepare time series data for Prophet (requires 'ds' and 'y' columns).
        
        Args:
            df: DataFrame with review data
            aspect: Aspect term to filter
            date_col: Name of datetime column
            sentiment_col: Name of sentiment score column
            aspect_col: Name of aspect column
            freq: Aggregation frequency ('D'=daily, 'W'=weekly, 'M'=monthly)
            min_samples_per_period: Minimum reviews required per period (default 5)
            
        Returns:
            DataFrame with columns ['ds', 'y'] suitable for Prophet
        """
        # Filter by aspect
        aspect_df = df[df[aspect_col] == aspect].copy()
        
        if len(aspect_df) == 0:
            raise ValueError(f"No data found for aspect: {aspect}")
        
        # Ensure datetime
        aspect_df[date_col] = pd.to_datetime(aspect_df[date_col])
        
        # Aggregate by date
        ts_data = aspect_df.groupby(pd.Grouper(key=date_col, freq=freq))[sentiment_col].agg([
            ('y', 'mean'),  # Average sentiment
            ('count', 'count')  # Number of reviews
        ]).reset_index()
        
        # Rename for Prophet
        ts_data = ts_data.rename(columns={date_col: 'ds'})
        
        # Filter out dates with too few samples
        ts_data = ts_data[ts_data['count'] >= min_samples_per_period].copy()
        
        if len(ts_data) < 10:
            raise ValueError(f"Insufficient data after volume filtering: only {len(ts_data)} periods with >={min_samples_per_period} reviews")
        
        # Remove outliers (3-sigma rule)
        mean_sent = ts_data['y'].mean()
        std_sent = ts_data['y'].std()
        ts_data = ts_data[
            (ts_data['y'] >= mean_sent - 3*std_sent) & 
            (ts_data['y'] <= mean_sent + 3*std_sent)
        ]
        
        # Ensure temporal continuity: find the longest consecutive sequence
        ts_data = ts_data.sort_values('ds').reset_index(drop=True)
        
        # Detect gaps in the time series
        if freq == 'D':
            max_gap = pd.Timedelta(days=7)  # Allow up to 7 day gaps for daily data
        elif freq == 'W':
            max_gap = pd.Timedelta(weeks=4)  # Allow up to 4 week gaps for weekly data
        elif freq == 'M':
            max_gap = pd.Timedelta(days=60)  # Allow up to 2 month gaps for monthly data
        else:
            max_gap = pd.Timedelta(days=7)
        
        # Find largest continuous segment
        if len(ts_data) > 1:
            gaps = ts_data['ds'].diff()
            large_gaps = gaps > max_gap
            
            if large_gaps.any():
                # Split into segments at large gaps
                gap_indices = large_gaps[large_gaps].index.tolist()
                segments = []
                start_idx = 0
                
                for gap_idx in gap_indices:
                    segment = ts_data.iloc[start_idx:gap_idx]
                    if len(segment) >= 10:  # Only keep segments with enough data
                        segments.append(segment)
                    start_idx = gap_idx
                
                # Add final segment
                final_segment = ts_data.iloc[start_idx:]
                if len(final_segment) >= 10:
                    segments.append(final_segment)
                
                if not segments:
                    raise ValueError("No continuous time segment found with sufficient data")
                
                # Select the longest segment
                ts_data = max(segments, key=len).copy()
                print(f"⚠️ Time series had gaps. Using longest continuous segment: {len(ts_data)} periods from {ts_data['ds'].min()} to {ts_data['ds'].max()}")
        
        return ts_data[['ds', 'y', 'count']].reset_index(drop=True)
    
    def detect_changepoints(self, ts_data: pd.DataFrame) -> List[datetime]:
        """
        Detect significant change points in sentiment trend using CUSUM algorithm.
        
        Args:
            ts_data: Time series data with 'ds' and 'y' columns
            
        Returns:
            List of datetime objects where change points detected
        """
        if len(ts_data) < 10:
            return []
        
        # CUSUM algorithm
        values = ts_data['y'].values
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return []
        
        # Normalized data
        z_scores = (values - mean) / std
        
        # Cumulative sum
        cusum_pos = np.zeros(len(z_scores))
        cusum_neg = np.zeros(len(z_scores))
        threshold = 3.0  # Detection threshold
        
        changepoints = []
        
        for i in range(1, len(z_scores)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + z_scores[i] - 0.5)
            cusum_neg[i] = max(0, cusum_neg[i-1] - z_scores[i] - 0.5)
            
            if cusum_pos[i] > threshold or cusum_neg[i] > threshold:
                changepoints.append(ts_data.iloc[i]['ds'])
                cusum_pos[i] = 0
                cusum_neg[i] = 0
        
        return changepoints
    
    def fit(self, ts_data: pd.DataFrame, aspect: str, 
            weekly_seasonality: bool = None, 
            yearly_seasonality: bool = None) -> 'SentimentForecaster':
        """
        Train Prophet model on historical sentiment data.
        
        Args:
            ts_data: Time series data with 'ds' and 'y' columns
            aspect: Aspect name for model storage
            weekly_seasonality: Override weekly seasonality setting
            yearly_seasonality: Override yearly seasonality setting
            
        Returns:
            Self for method chaining
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not available. Install with: pip install prophet")
        
        if len(ts_data) < 10:
            raise ValueError(f"Insufficient data for training. Need at least 10 points, got {len(ts_data)}")
        
        # Determine seasonality settings
        if weekly_seasonality is None:
            weekly_seasonality = len(ts_data) > 14
        if yearly_seasonality is None:
            yearly_seasonality = len(ts_data) > 365
        
        # Initialize Prophet model
        model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            interval_width=self.interval_width,
            daily_seasonality=False,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality
        )
        
        # Fit model
        model.fit(ts_data[['ds', 'y']])
        
        # Store trained model
        self.models[aspect] = model
        
        return self
    
    def predict(self, aspect: str, periods: int = 90, 
                freq: str = 'D') -> pd.DataFrame:
        """
        Generate future sentiment predictions.
        
        Args:
            aspect: Aspect name (must have trained model)
            periods: Number of periods to forecast
            freq: Frequency ('D'=daily, 'W'=weekly, 'M'=monthly)
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        if aspect not in self.models:
            raise ValueError(f"No trained model for aspect: {aspect}. Call fit() first.")
        
        model = self.models[aspect]
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq=freq)
        
        # Generate predictions
        forecast = model.predict(future)
        
        # Add aspect column
        forecast['aspect'] = aspect
        
        return forecast
    
    def analyze_trend(self, forecast: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze forecast trend and provide insights.
        
        Args:
            forecast: Prophet forecast output
            
        Returns:
            Dictionary with trend analysis metrics
        """
        # Get trend component
        trend = forecast['trend'].values
        
        # Calculate trend direction
        recent_trend = trend[-30:]  # Last 30 points
        if len(recent_trend) > 1:
            slope, _, _, _, _ = stats.linregress(range(len(recent_trend)), recent_trend)
        else:
            slope = 0
        
        # Classify trend
        if slope > 0.01:
            trend_direction = "Improving"
        elif slope < -0.01:
            trend_direction = "Declining"
        else:
            trend_direction = "Stable"
        
        # Calculate volatility
        volatility = forecast['yhat'].std()
        
        # Predict crossing thresholds
        future_forecast = forecast[forecast['ds'] > datetime.now()]
        
        # Check if sentiment will cross critical thresholds
        alerts = []
        if any(future_forecast['yhat'] < 0):
            first_negative = future_forecast[future_forecast['yhat'] < 0].iloc[0]
            alerts.append({
                'type': 'negative_threshold',
                'date': first_negative['ds'],
                'message': f"Sentiment predicted to turn negative by {first_negative['ds'].strftime('%Y-%m-%d')}"
            })
        
        if slope < -0.02:  # Rapid decline
            alerts.append({
                'type': 'rapid_decline',
                'slope': slope,
                'message': f"Sentiment declining rapidly (slope: {slope:.4f}/day)"
            })
        
        return {
            'trend_direction': trend_direction,
            'slope': slope,
            'volatility': volatility,
            'current_sentiment': forecast['yhat'].iloc[-1],
            'predicted_30d': future_forecast['yhat'].iloc[min(30, len(future_forecast)-1)] if len(future_forecast) > 0 else None,
            'predicted_90d': future_forecast['yhat'].iloc[min(90, len(future_forecast)-1)] if len(future_forecast) > 0 else None,
            'alerts': alerts
        }
    
    def detect_anomalies(self, forecast: pd.DataFrame, 
                         historical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in historical data based on forecast model.
        
        Args:
            forecast: Prophet forecast output
            historical_data: Original time series data
            
        Returns:
            DataFrame with anomaly flags
        """
        # Merge historical with predictions
        merged = historical_data.merge(
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
            on='ds', 
            how='left'
        )
        
        # Flag anomalies (outside 95% confidence interval)
        merged['is_anomaly'] = (
            (merged['y'] < merged['yhat_lower']) | 
            (merged['y'] > merged['yhat_upper'])
        )
        
        # Calculate anomaly score
        merged['anomaly_score'] = np.abs(merged['y'] - merged['yhat']) / (
            merged['yhat_upper'] - merged['yhat_lower'] + 1e-6
        )
        
        return merged
    
    def decompose_seasonality(self, ts_data: pd.DataFrame, 
                               freq: int = 7) -> pd.DataFrame:
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Args:
            ts_data: Time series data
            freq: Seasonal frequency (7=weekly, 30=monthly)
            
        Returns:
            DataFrame with decomposition components
        """
        if len(ts_data) < 2 * freq:
            return None
        
        # Set datetime index
        ts_indexed = ts_data.set_index('ds')['y']
        
        # Decompose
        try:
            decomposition = seasonal_decompose(
                ts_indexed, 
                model='additive', 
                period=freq,
                extrapolate_trend='freq'
            )
            
            result = pd.DataFrame({
                'ds': ts_data['ds'],
                'observed': decomposition.observed,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid
            })
            
            return result
        except:
            return None


class ARIMAForecaster:
    """
    Forecast aspect-level sentiment using ARIMA/SARIMA model.
    """
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)):
        """
        Initialize ARIMA forecaster.
        
        Args:
            order: (p, d, q) for ARIMA - (AR order, differencing, MA order)
            seasonal_order: (P, D, Q, s) for SARIMA - seasonal components
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.models = {}
        self.ts_data_cache = {}
    
    def prepare_data(self, df: pd.DataFrame, aspect: str,
                     date_col: str = 'timestamp',
                     sentiment_col: str = 'sentiment_score',
                     aspect_col: str = 'aspect',
                     freq: str = 'D',
                     min_samples_per_period: int = 5) -> pd.DataFrame:
        """
        Prepare time series data for ARIMA with proper volume filtering.
        
        Args:
            min_samples_per_period: Minimum reviews required per period (default 5)
        """
        aspect_df = df[df[aspect_col] == aspect].copy()
        
        if len(aspect_df) == 0:
            raise ValueError(f"No data found for aspect: {aspect}")
        
        aspect_df[date_col] = pd.to_datetime(aspect_df[date_col])
        
        # Aggregate by date
        ts_data = aspect_df.groupby(pd.Grouper(key=date_col, freq=freq))[sentiment_col].agg([
            ('y', 'mean'),
            ('count', 'count')
        ]).reset_index()
        
        ts_data = ts_data.rename(columns={date_col: 'ds'})
        
        # Filter by minimum volume per period
        ts_data = ts_data[ts_data['count'] >= min_samples_per_period].copy()
        
        if len(ts_data) < 10:
            raise ValueError(f"Insufficient data after volume filtering: only {len(ts_data)} periods with >={min_samples_per_period} reviews")
        
        # Remove outliers (3-sigma rule)
        z_scores = np.abs(stats.zscore(ts_data['y']))
        ts_data = ts_data[z_scores < 3].copy()
        
        # Ensure temporal continuity: find the longest consecutive sequence
        ts_data = ts_data.sort_values('ds').reset_index(drop=True)
        
        # Detect gaps in the time series
        if freq == 'D':
            max_gap = pd.Timedelta(days=7)  # Allow up to 7 day gaps for daily data
        elif freq == 'W':
            max_gap = pd.Timedelta(weeks=4)  # Allow up to 4 week gaps for weekly data
        elif freq == 'M':
            max_gap = pd.Timedelta(days=60)  # Allow up to 2 month gaps for monthly data
        else:
            max_gap = pd.Timedelta(days=7)
        
        # Find largest continuous segment
        if len(ts_data) > 1:
            gaps = ts_data['ds'].diff()
            large_gaps = gaps > max_gap
            
            if large_gaps.any():
                # Split into segments at large gaps
                gap_indices = large_gaps[large_gaps].index.tolist()
                segments = []
                start_idx = 0
                
                for gap_idx in gap_indices:
                    segment = ts_data.iloc[start_idx:gap_idx]
                    if len(segment) >= 10:  # Only keep segments with enough data
                        segments.append(segment)
                    start_idx = gap_idx
                
                # Add final segment
                final_segment = ts_data.iloc[start_idx:]
                if len(final_segment) >= 10:
                    segments.append(final_segment)
                
                if not segments:
                    raise ValueError("No continuous time segment found with sufficient data")
                
                # Select the longest segment
                ts_data = max(segments, key=len).copy()
                print(f"⚠️ Time series had gaps. Using longest continuous segment: {len(ts_data)} periods from {ts_data['ds'].min()} to {ts_data['ds'].max()}")
        
        # Set datetime index for ARIMA
        ts_data = ts_data.set_index('ds')
        
        return ts_data
    
    def fit(self, ts_data: pd.DataFrame, aspect: str):
        """Train ARIMA model."""
        if not ARIMA_AVAILABLE:
            raise ImportError("ARIMA not available. Install statsmodels.")
        
        # Use SARIMAX if seasonal order is specified
        if sum(self.seasonal_order) > 0:
            model = SARIMAX(
                ts_data['y'],
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
        else:
            model = ARIMA(
                ts_data['y'],
                order=self.order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
        
        # Fit model (suppress warnings)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted_model = model.fit()
        
        self.models[aspect] = fitted_model
        self.ts_data_cache[aspect] = ts_data
        
        return self
    
    def predict(self, aspect: str, periods: int = 90) -> pd.DataFrame:
        """Generate ARIMA predictions."""
        if aspect not in self.models:
            raise ValueError(f"No trained model for aspect: {aspect}")
        
        model = self.models[aspect]
        ts_data = self.ts_data_cache[aspect]
        
        # Forecast
        forecast_result = model.get_forecast(steps=periods)
        forecast_mean = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int(alpha=0.05)  # 95% CI
        
        # Create future dates
        last_date = ts_data.index[-1]
        freq = ts_data.index.freq or pd.infer_freq(ts_data.index)
        future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
        
        # Build forecast dataframe
        forecast_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_mean.values,
            'yhat_lower': forecast_ci.iloc[:, 0].values,
            'yhat_upper': forecast_ci.iloc[:, 1].values
        })
        
        return forecast_df


def forecast_aspect_sentiment(df: pd.DataFrame, aspect: str, 
                              forecast_days: int = 90,
                              freq: str = 'D',
                              aspect_col: str = 'aspect',
                              model_type: str = 'prophet',
                              model_params: Dict = None,
                              min_samples_per_period: int = 5) -> Dict[str, any]:
    """
    Complete workflow: prepare data, train model, forecast, and analyze.
    
    Args:
        df: DataFrame with review data
        aspect: Aspect to forecast
        forecast_days: Number of days to forecast ahead
        freq: Aggregation frequency
        aspect_col: Name of the aspect column in df
        model_type: 'prophet' or 'arima'
        model_params: Dictionary of model-specific hyperparameters
        min_samples_per_period: Minimum reviews per time period for data quality
        
    Returns:
        Dictionary with forecast results and analysis
    """
    
    # Set default parameters
    if model_params is None:
        model_params = {}
    
    try:
        # Select model
        if model_type.lower() == 'arima':
            if not ARIMA_AVAILABLE:
                return {
                    'success': False,
                    'error': 'ARIMA not available. Install statsmodels.'
                }
            
            # Get ARIMA parameters
            order = model_params.get('order', (1, 1, 1))
            forecaster = ARIMAForecaster(order=order)
            ts_data = forecaster.prepare_data(
                df, aspect, freq=freq, aspect_col=aspect_col, 
                min_samples_per_period=min_samples_per_period
            )
            
            if len(ts_data) < 10:
                return {
                    'success': False,
                    'error': f'Insufficient data: only {len(ts_data)} data points (need at least 10)'
                }
            
            # Train ARIMA
            forecaster.fit(ts_data, aspect)
            
            # Generate forecast
            forecast = forecaster.predict(aspect, periods=forecast_days)
            
            # Prepare historical data for plotting
            historical = ts_data.reset_index()[['ds', 'y']]
            
            # Simple trend analysis for ARIMA
            slope = (forecast['yhat'].iloc[-1] - forecast['yhat'].iloc[0]) / len(forecast)
            current_sentiment = historical['y'].iloc[-1]
            
            if slope > 0.01:
                trend_direction = "Improving ⬆️"
            elif slope < -0.01:
                trend_direction = "Declining ⬇️"
            else:
                trend_direction = "Stable ➡️"
            
            trend_analysis = {
                'trend_direction': trend_direction,
                'slope': slope,
                'current_sentiment': current_sentiment,
                'predicted_30d': forecast['yhat'].iloc[min(29, len(forecast)-1)],
                'predicted_90d': forecast['yhat'].iloc[min(89, len(forecast)-1)],
                'volatility': forecast['yhat'].std(),
                'alerts': []
            }
            
            # Add alerts
            if forecast['yhat'].iloc[-1] < 0:
                trend_analysis['alerts'].append({
                    'severity': 'high',
                    'message': f'Sentiment predicted to turn negative'
                })
            
            if slope < -0.02:
                trend_analysis['alerts'].append({
                    'severity': 'medium',
                    'message': f'Sentiment declining rapidly (slope: {slope:.4f}/day)'
                })
            
            return {
                'success': True,
                'model': 'ARIMA',
                'aspect': aspect,
                'forecast': forecast,
                'historical': historical,
                'trend_analysis': trend_analysis,
                'changepoints': [],  # ARIMA doesn't detect changepoints
                'anomalies': pd.DataFrame(),  # Simplified for ARIMA
                'decomposition': None,
                'data_points': len(ts_data),
                'forecast_periods': forecast_days
            }
            
        else:  # Prophet (default)
            if not PROPHET_AVAILABLE:
                return {
                    'success': False,
                    'error': 'Prophet not available. Install prophet.'
                }
            
            # Get Prophet parameters
            changepoint_prior_scale = model_params.get('changepoint_prior_scale', 0.05)
            seasonality_prior_scale = model_params.get('seasonality_prior_scale', 1.0)
            
            forecaster = SentimentForecaster(
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale
            )
            
            # Prepare data
            ts_data = forecaster.prepare_data(
                df, aspect, freq=freq, aspect_col=aspect_col,
                min_samples_per_period=min_samples_per_period
            )
            
            if len(ts_data) < 10:
                return {
                    'success': False,
                    'error': f'Insufficient data: only {len(ts_data)} data points (need at least 10)'
                }
            
            # Detect change points
            changepoints = forecaster.detect_changepoints(ts_data)
        
        # Train model with custom seasonality settings
        weekly = model_params.get('weekly_seasonality', True if len(ts_data) > 14 else False)
        yearly = model_params.get('yearly_seasonality', True if len(ts_data) > 365 else False)
        forecaster.fit(ts_data, aspect, weekly_seasonality=weekly, yearly_seasonality=yearly)
        
        # Generate forecast
        forecast = forecaster.predict(aspect, periods=forecast_days, freq=freq)
        
        # Analyze trend
        trend_analysis = forecaster.analyze_trend(forecast)
        
        # Detect anomalies
        anomalies = forecaster.detect_anomalies(forecast, ts_data)
        
        # Decompose seasonality
        decomposition = forecaster.decompose_seasonality(ts_data, freq=7 if freq == 'D' else 4)
        
        return {
            'success': True,
            'model': 'Prophet',
            'aspect': aspect,
            'forecast': forecast,
            'historical': ts_data,
            'trend_analysis': trend_analysis,
            'changepoints': changepoints,
            'anomalies': anomalies,
            'decomposition': decomposition,
            'data_points': len(ts_data),
            'forecast_periods': forecast_days
        }
        
    except Exception as e:
        return {
            'success': False,
            'aspect': aspect,
            'error': str(e)
        }


def batch_forecast_aspects(df: pd.DataFrame, aspects: List[str], 
                           forecast_days: int = 90,
                           aspect_col: str = 'aspect') -> Dict[str, Dict]:
    """
    Forecast multiple aspects in batch.
    
    Args:
        df: DataFrame with review data
        aspects: List of aspects to forecast
        forecast_days: Number of days to forecast
        aspect_col: Name of the aspect column in df
        
    Returns:
        Dictionary mapping aspect names to forecast results
    """
    results = {}
    
    for aspect in aspects:
        print(f"Forecasting: {aspect}...")
        result = forecast_aspect_sentiment(df, aspect, forecast_days, aspect_col=aspect_col)
        results[aspect] = result
    
    return results
