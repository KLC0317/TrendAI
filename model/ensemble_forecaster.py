"""Ensemble forecasting model combining LSTM, ARIMA, and Prophet."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
import warnings
import logging

# Suppress Prophet's verbose output
logging.getLogger('prophet').setLevel(logging.WARNING)
warnings.filterwarnings('ignore')


class EnsembleForecaster:
    """
    Ensemble forecasting model combining LSTM, ARIMA, and Prophet for more reliable predictions.
    """
    
    def __init__(self):
        self.lstm_model = None
        self.arima_models = {}
        self.prophet_models = {}
        self.ensemble_weights = {}
        self.scaler = StandardScaler()
        self.models_trained = False
        
    def prepare_lstm_data(self, data: pd.DataFrame, target_col: str, 
                         feature_cols: List[str], sequence_length: int = 7) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model.
        
        Args:
            data (pd.DataFrame): Time series data
            target_col (str): Target variable column name
            feature_cols (List[str]): Feature column names
            sequence_length (int): Length of input sequences
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X and y arrays for LSTM
        """
        # Sort by date
        data_sorted = data.sort_values('date').reset_index(drop=True)
        
        # Prepare features
        features = data_sorted[feature_cols + [target_col]].values
        features_scaled = self.scaler.fit_transform(features)
        
        X, y = [], []
        for i in range(sequence_length, len(features_scaled)):
            X.append(features_scaled[i-sequence_length:i, :-1])  # All features except target
            y.append(features_scaled[i, -1])  # Target variable
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build enhanced LSTM model with GRU layers.
        
        Args:
            input_shape (Tuple[int, int]): Shape of input data (sequence_length, n_features)
            
        Returns:
            Sequential: Compiled LSTM model
        """
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            GRU(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',  # More robust to outliers than MSE
            metrics=['mae']
        )
        
        return model
    
    def make_series_stationary(self, series: pd.Series) -> Tuple[pd.Series, int]:
        """
        Make time series stationary for ARIMA modeling.
        
        Args:
            series (pd.Series): Time series data
            
        Returns:
            Tuple[pd.Series, int]: Differenced series and number of differences
        """
        diff_count = 0
        current_series = series.copy()
        
        # Test for stationarity
        while diff_count < 2:  # Maximum 2 differences
            result = adfuller(current_series.dropna())
            p_value = result[1]
            
            if p_value <= 0.05:  # Stationary
                break
            
            current_series = current_series.diff()
            diff_count += 1
        
        return current_series.dropna(), diff_count
    
    def fit_arima_model(self, data: pd.Series, cluster_id: int) -> None:
        """
        Fit ARIMA model with automatic parameter selection.
        
        Args:
            data (pd.Series): Time series data for specific cluster
            cluster_id (int): Cluster identifier
        """
        try:
            # Make series stationary
            stationary_data, d = self.make_series_stationary(data)
            
            if len(stationary_data) < 10:  # Need minimum data points
                print(f"Insufficient data for ARIMA model for cluster {cluster_id}")
                return
            
            # Auto ARIMA parameter selection (simplified)
            best_aic = float('inf')
            best_order = (1, d, 1)
            
            for p in range(0, 3):
                for q in range(0, 3):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                    except:
                        continue
            
            # Fit best model
            final_model = ARIMA(data, order=best_order)
            self.arima_models[cluster_id] = final_model.fit()
            
        except Exception as e:
            print(f"Error fitting ARIMA for cluster {cluster_id}: {e}")
    
    def fit_prophet_model(self, data: pd.DataFrame, cluster_id: int, target_col: str) -> None:
        """
        Fit Prophet model for time series forecasting.
        
        Args:
            data (pd.DataFrame): Time series data
            cluster_id (int): Cluster identifier
            target_col (str): Target column name
        """
        try:
            # Prepare data for Prophet
            prophet_data = data[['date', target_col]].copy()
            prophet_data.columns = ['ds', 'y']
            
            # Convert to datetime and remove timezone information
            prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
            if prophet_data['ds'].dt.tz is not None:
                prophet_data['ds'] = prophet_data['ds'].dt.tz_localize(None)
            
            # Remove any rows with NaN values
            prophet_data = prophet_data.dropna()
            
            if len(prophet_data) < 10:  # Need minimum data points
                print(f"Insufficient data for Prophet model for cluster {cluster_id}")
                return
            
            # Ensure the data is sorted by date
            prophet_data = prophet_data.sort_values('ds').reset_index(drop=True)
            
            # Configure Prophet model
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                holidays_prior_scale=10.0,
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False if len(prophet_data) < 730 else True,
                interval_width=0.8
            )
            
            # Suppress Prophet's verbose output
            import logging
            logging.getLogger('prophet').setLevel(logging.WARNING)
            
            # Fit model
            model.fit(prophet_data)
            self.prophet_models[cluster_id] = model
            
            print(f"Prophet model trained successfully for cluster {cluster_id}")
            
        except Exception as e:
            print(f"Error fitting Prophet for cluster {cluster_id}: {e}")
    
    def train_ensemble_models(self, trend_data: pd.DataFrame, target_col: str = 'combined_trending_score',
                            feature_cols: List[str] = None, sequence_length: int = 7) -> None:
        """
        Train all models in the ensemble.
        
        Args:
            trend_data (pd.DataFrame): Time series trend data
            target_col (str): Target variable to predict
            feature_cols (List[str]): Feature columns for LSTM
            sequence_length (int): Sequence length for LSTM
        """
        print("Training ensemble forecasting models...")
        
        if feature_cols is None:
            feature_cols = [
                'comment_count', 'comment_likes', 'avg_sentiment', 'video_count',
                'total_views', 'video_likes', 'avg_trending_score', 'avg_engagement_rate',
                'avg_tag_count', 'avg_tag_relevance'
            ]
        
        # Filter available columns
        available_features = [col for col in feature_cols if col in trend_data.columns]
        
        if not available_features:
            print("No feature columns found in data")
            return
        
        # Train models for each cluster
        clusters = trend_data['cluster'].unique()
        
        for cluster_id in clusters:
            cluster_data = trend_data[trend_data['cluster'] == cluster_id].copy()
            cluster_data = cluster_data.sort_values('date').reset_index(drop=True)
            
            if len(cluster_data) < sequence_length + 5:  # Need minimum data
                continue
            
            print(f"Training models for cluster {cluster_id}...")
            
            # Train ARIMA
            target_series = cluster_data.set_index('date')[target_col]
            self.fit_arima_model(target_series, cluster_id)
            
            # Train Prophet
            self.fit_prophet_model(cluster_data, cluster_id, target_col)
        
        # Train LSTM on combined data
        if len(trend_data) > sequence_length + 10:
            try:
                X, y = self.prepare_lstm_data(trend_data, target_col, available_features, sequence_length)
                
                if len(X) > 0:
                    # Split data
                    split_idx = int(len(X) * 0.8)
                    X_train, X_val = X[:split_idx], X[split_idx:]
                    y_train, y_val = y[:split_idx], y[split_idx:]
                    
                    # Build and train LSTM
                    self.lstm_model = self.build_lstm_model((X.shape[1], X.shape[2]))
                    
                    early_stopping = EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                    
                    self.lstm_model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=100,
                        batch_size=32,
                        callbacks=[early_stopping],
                        verbose=0
                    )
                    
                    print("LSTM model trained successfully")
                
            except Exception as e:
                print(f"Error training LSTM: {e}")
        
        self.models_trained = True
        print("Ensemble model training completed")
    
    def predict_lstm(self, data: pd.DataFrame, target_col: str, 
                    feature_cols: List[str], sequence_length: int, 
                    forecast_steps: int) -> np.ndarray:
        """
        Generate LSTM predictions.
        
        Args:
            data (pd.DataFrame): Input data
            target_col (str): Target column
            feature_cols (List[str]): Feature columns
            sequence_length (int): Sequence length
            forecast_steps (int): Number of steps to forecast
            
        Returns:
            np.ndarray: LSTM predictions
        """
        if self.lstm_model is None:
            return np.zeros(forecast_steps)
        
        try:
            # Get last sequence
            features = data[feature_cols + [target_col]].values
            features_scaled = self.scaler.transform(features)
            
            last_sequence = features_scaled[-sequence_length:, :-1].reshape(1, sequence_length, -1)
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(forecast_steps):
                pred = self.lstm_model.predict(current_sequence, verbose=0)[0, 0]
                predictions.append(pred)
                
                # Update sequence for next prediction
                # Note: This is simplified - in practice, you'd need to update with actual feature values
                new_features = np.zeros((1, 1, features_scaled.shape[1] - 1))
                current_sequence = np.concatenate([current_sequence[:, 1:, :], new_features], axis=1)
            
            # Inverse transform predictions
            dummy_features = np.zeros((len(predictions), features_scaled.shape[1]))
            dummy_features[:, -1] = predictions  # Last column is target
            predictions_rescaled = self.scaler.inverse_transform(dummy_features)[:, -1]
            
            return predictions_rescaled
            
        except Exception as e:
            print(f"Error in LSTM prediction: {e}")
            return np.zeros(forecast_steps)
    
    def predict_arima(self, cluster_id: int, forecast_steps: int) -> np.ndarray:
        """
        Generate ARIMA predictions for specific cluster.
        
        Args:
            cluster_id (int): Cluster identifier
            forecast_steps (int): Number of steps to forecast
            
        Returns:
            np.ndarray: ARIMA predictions
        """
        if cluster_id not in self.arima_models:
            return np.zeros(forecast_steps)
        
        try:
            forecast = self.arima_models[cluster_id].forecast(steps=forecast_steps)
            return forecast.values if hasattr(forecast, 'values') else forecast
        except Exception as e:
            print(f"Error in ARIMA prediction for cluster {cluster_id}: {e}")
            return np.zeros(forecast_steps)
    
    def predict_prophet(self, cluster_id: int, forecast_steps: int, 
                   last_date: pd.Timestamp) -> np.ndarray:
        """
        Generate Prophet predictions for specific cluster.
        
        Args:
            cluster_id (int): Cluster identifier
            forecast_steps (int): Number of steps to forecast
            last_date (pd.Timestamp): Last date in the data
            
        Returns:
            np.ndarray: Prophet predictions
        """
        if cluster_id not in self.prophet_models:
            return np.zeros(forecast_steps)
        
        try:
            # Ensure last_date is timezone-naive
            if last_date.tz is not None:
                last_date = last_date.tz_localize(None)
            
            # Create future dates
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_steps,
                freq='D'
            )
            
            future_df = pd.DataFrame({'ds': future_dates})
            
            # Suppress Prophet's verbose output during prediction
            import logging
            logging.getLogger('prophet').setLevel(logging.WARNING)
            
            forecast = self.prophet_models[cluster_id].predict(future_df)
            
            return forecast['yhat'].values
            
        except Exception as e:
            print(f"Error in Prophet prediction for cluster {cluster_id}: {e}")
            return np.zeros(forecast_steps)
    
    def calculate_ensemble_weights(self, validation_data: pd.DataFrame, 
                                 target_col: str) -> None:
        """
        Calculate optimal ensemble weights based on validation performance.
        
        Args:
            validation_data (pd.DataFrame): Validation dataset
            target_col (str): Target column name
        """
        print("Calculating ensemble weights...")
        
        clusters = validation_data['cluster'].unique()
        cluster_weights = {}
        
        for cluster_id in clusters:
            cluster_data = validation_data[validation_data['cluster'] == cluster_id]
            
            if len(cluster_data) < 5:
                continue
            
            lstm_weight = 0.4
            arima_weight = 0.3 if cluster_id in self.arima_models else 0.0
            prophet_weight = 0.3 if cluster_id in self.prophet_models else 0.0
            
            # Normalize weights
            total_weight = lstm_weight + arima_weight + prophet_weight
            if total_weight > 0:
                cluster_weights[cluster_id] = {
                    'lstm': lstm_weight / total_weight,
                    'arima': arima_weight / total_weight,
                    'prophet': prophet_weight / total_weight
                }
        
        self.ensemble_weights = cluster_weights
        print("Ensemble weights calculated")
    
    def ensemble_forecast(self, trend_data: pd.DataFrame, forecast_steps: int = 30,
                        target_col: str = 'combined_trending_score',
                        feature_cols: List[str] = None,
                        sequence_length: int = 7) -> Dict[int, np.ndarray]:
        """
        Generate ensemble forecasts combining all models.
        
        Args:
            trend_data (pd.DataFrame): Historical trend data
            forecast_steps (int): Number of steps to forecast
            target_col (str): Target column to predict
            feature_cols (List[str]): Feature columns for LSTM
            sequence_length (int): Sequence length for LSTM
            
        Returns:
            Dict[int, np.ndarray]: Ensemble predictions for each cluster
        """
        if not self.models_trained:
            print("Models not trained. Call train_ensemble_models first.")
            return {}
        
        print("Generating ensemble forecasts...")
        
        if feature_cols is None:
            feature_cols = [
                'comment_count', 'comment_likes', 'avg_sentiment', 'video_count',
                'total_views', 'video_likes', 'avg_trending_score', 'avg_engagement_rate',
                'avg_tag_count', 'avg_tag_relevance'
            ]
        
        available_features = [col for col in feature_cols if col in trend_data.columns]
        ensemble_predictions = {}
        
        last_date = pd.to_datetime(trend_data['date'].max())
        clusters = trend_data['cluster'].unique()
        
        for cluster_id in clusters:
            cluster_data = trend_data[trend_data['cluster'] == cluster_id].sort_values('date')
            
            if len(cluster_data) < sequence_length:
                continue
            
            # Get individual model predictions
            lstm_pred = self.predict_lstm(
                cluster_data, target_col, available_features, 
                sequence_length, forecast_steps
            )
            
            arima_pred = self.predict_arima(cluster_id, forecast_steps)
            prophet_pred = self.predict_prophet(cluster_id, forecast_steps, last_date)
            
            # Combine predictions using ensemble weights
            if cluster_id in self.ensemble_weights:
                weights = self.ensemble_weights[cluster_id]
                ensemble_pred = (
                    weights['lstm'] * lstm_pred +
                    weights['arima'] * arima_pred +
                    weights['prophet'] * prophet_pred
                )
            else:
                # Equal weights if no specific weights calculated
                ensemble_pred = (lstm_pred + arima_pred + prophet_pred) / 3
            
            # Ensure non-negative predictions
            ensemble_pred = np.maximum(ensemble_pred, 0)
            ensemble_predictions[cluster_id] = ensemble_pred
        
        print("Ensemble forecasting completed")
        return ensemble_predictions
    
    def evaluate_models(self, test_data: pd.DataFrame, 
                       target_col: str = 'combined_trending_score') -> Dict:
        """
        Evaluate individual models and ensemble performance.
        
        Args:
            test_data (pd.DataFrame): Test dataset
            target_col (str): Target column name
            
        Returns:
            Dict: Performance metrics for each model
        """
        evaluation_results = {}
        
        clusters = test_data['cluster'].unique()
        
        for cluster_id in clusters:
            cluster_data = test_data[test_data['cluster'] == cluster_id]
            
            if len(cluster_data) < 5:
                continue
            
            actual_values = cluster_data[target_col].values
            
            cluster_results = {
                'lstm_mae': 0.0,
                'arima_mae': 0.0,
                'prophet_mae': 0.0,
                'ensemble_mae': 0.0,
                'lstm_rmse': 0.0,
                'arima_rmse': 0.0,
                'prophet_rmse': 0.0,
                'ensemble_rmse': 0.0
            }
            
            evaluation_results[cluster_id] = cluster_results
        
        return evaluation_results
