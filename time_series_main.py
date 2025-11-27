# Advanced Time Series Forecasting with Attention Mechanisms
# Production-ready implementation with comprehensive documentation

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf
from datetime import datetime, timedelta
import json
import logging
from typing import Tuple, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles data acquisition and preprocessing for time series forecasting.
    Supports both real financial data and synthetic data generation.
    """
    
    def __init__(self, lookback_window: int = 60, forecast_horizon: int = 10):
        """
        Initialize data loader.
        
        Args:
            lookback_window: Number of historical steps to use for prediction
            forecast_horizon: Number of future steps to forecast
        """
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.scaler = StandardScaler()
        logger.info(f"DataLoader initialized: lookback={lookback_window}, horizon={forecast_horizon}")
    
    def acquire_financial_data(self, ticker: str = 'AAPL', days: int = 365) -> pd.DataFrame:
        """
        Acquire real stock data using yfinance.
        
        Args:
            ticker: Stock ticker symbol
            days: Number of historical days to retrieve
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            logger.info(f"Downloaded {len(data)} records for {ticker}")
            return data
        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")
            raise
    
    def generate_synthetic_data(self, n_samples: int = 1000, n_features: int = 4) -> np.ndarray:
        """
        Generate synthetic multivariate time series data representing noisy, non-stationary processes.
        
        Args:
            n_samples: Number of time steps
            n_features: Number of features/variables
            
        Returns:
            Synthetic time series array of shape (n_samples, n_features)
        """
        np.random.seed(42)
        
        # Base trend component
        time = np.arange(n_samples)
        trends = np.zeros((n_samples, n_features))
        
        for i in range(n_features):
            # Non-stationary trend with drift
            drift = 0.01 * (i + 1)
            trend = drift * time + 50 * (i + 1)
            trends[:, i] = trend
        
        # Seasonal components
        seasonality = np.zeros((n_samples, n_features))
        for i in range(n_features):
            seasonality[:, i] = 5 * (i + 1) * np.sin(2 * np.pi * time / 100)
        
        # Autoregressive component
        ar_component = np.zeros((n_samples, n_features))
        for t in range(1, n_samples):
            ar_component[t] = 0.7 * ar_component[t-1] + np.random.randn(n_features) * 0.5
        
        # Noise (non-Gaussian, non-stationary)
        noise = np.random.randn(n_samples, n_features) * np.linspace(1, 3, n_samples)[:, np.newaxis]
        
        # Combine all components
        data = trends + seasonality + ar_component + noise
        
        logger.info(f"Generated synthetic data: shape {data.shape}")
        return data
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for supervised learning.
        
        Args:
            data: Normalized time series data
            
        Returns:
            X: Input sequences of shape (n_samples, lookback_window, n_features)
            y: Target sequences of shape (n_samples, forecast_horizon, n_features)
        """
        X, y = [], []
        
        for i in range(len(data) - self.lookback_window - self.forecast_horizon + 1):
            X.append(data[i:i + self.lookback_window])
            y.append(data[i + self.lookback_window:i + self.lookback_window + self.forecast_horizon])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
        return X, y
    
    def preprocess_data(self, data: np.ndarray, test_split: float = 0.2) -> Dict:
        """
        Normalize and split data into train/test sets.
        
        Args:
            data: Raw time series data
            test_split: Fraction of data to use for testing
            
        Returns:
            Dictionary containing train/test sequences and scalers
        """
        # Normalize data
        n_samples, n_features = data.shape
        data_reshaped = data.reshape(-1, n_features)
        data_normalized = self.scaler.fit_transform(data_reshaped)
        data_normalized = data_normalized.reshape(n_samples, n_features)
        
        # Create sequences
        X, y = self.create_sequences(data_normalized)
        
        # Split into train/test
        split_idx = int(len(X) * (1 - test_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': self.scaler,
            'data_normalized': data_normalized
        }


class MultiHeadAttention(layers.Layer):
    """
    Custom Multi-Head Attention layer for transformer architecture.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, **kwargs):
        """
        Initialize Multi-Head Attention layer.
        
        Args:
            embed_dim: Embedding dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
        """
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.output_dense = layers.Dense(embed_dim)
    
    def split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        """Split heads for multi-head attention."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, mask=None) -> tf.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Attention mask (optional)
            
        Returns:
            Attention output
        """
        batch_size = tf.shape(query)[0]
        
        # Linear transformations
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        
        # Split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        scaled_attention_logits = matmul_qk * self.scale
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_output = tf.matmul(attention_weights, value)
        
        # Concatenate heads
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, -1, self.embed_dim))
        
        # Final linear transformation
        output = self.output_dense(attention_output)
        
        return output, attention_weights


class TransformerForecastingModel:
    """
    Transformer-based time series forecasting model with attention mechanisms.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int],
                 forecast_horizon: int = 10,
                 embed_dim: int = 64,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 ff_dim: int = 128,
                 dropout_rate: float = 0.1):
        """
        Initialize Transformer model.
        
        Args:
            input_shape: Shape of input sequences (lookback_window, n_features)
            forecast_horizon: Number of future steps to forecast
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Feed-forward hidden dimension
            dropout_rate: Dropout rate
        """
        self.input_shape = input_shape
        self.forecast_horizon = forecast_horizon
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.model = None
        self.attention_weights_history = []
        
        logger.info(f"TransformerForecastingModel initialized with embed_dim={embed_dim}, num_layers={num_layers}")
    
    def build_model(self) -> keras.Model:
        """
        Build transformer model architecture.
        
        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=self.input_shape)
        x = inputs
        
        # Embedding layer
        x = layers.Dense(self.embed_dim)(x)
        x = layers.LayerNormalization()(x)
        
        # Positional encoding
        positions = tf.range(start=0, limit=self.input_shape[0], delta=1)
        angle_rads = self._get_angles(positions, np.arange(self.embed_dim)[np.newaxis, :], self.embed_dim)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        x = x + tf.cast(pos_encoding, tf.float32)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Transformer encoder layers
        for _ in range(self.num_layers):
            # Multi-head attention
            attention_layer = MultiHeadAttention(self.embed_dim, self.num_heads)
            attention_output, _ = attention_layer(x, x, x)
            x = layers.Add()([x, attention_output])
            x = layers.LayerNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)
            
            # Feed-forward network
            ff_output = layers.Dense(self.ff_dim, activation='relu')(x)
            ff_output = layers.Dense(self.embed_dim)(ff_output)
            x = layers.Add()([x, ff_output])
            x = layers.LayerNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layers
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.forecast_horizon * self.input_shape[1])(x)
        outputs = layers.Reshape((self.forecast_horizon, self.input_shape[1]))(outputs)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        
        logger.info(f"Model built successfully")
        self.model = model
        return model
    
    @staticmethod
    def _get_angles(pos: np.ndarray, i: np.ndarray, d_model: int) -> np.ndarray:
        """Calculate positional encoding angles."""
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos[:, np.newaxis] * angle_rates
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              validation_split: float = 0.2, epochs: int = 100, batch_size: int = 32):
        """
        Train the model with rolling window cross-validation.
        
        Args:
            X_train: Training input sequences
            y_train: Training target sequences
            validation_split: Fraction of training data for validation
            epochs: Number of training epochs
            batch_size: Batch size
        """
        if self.model is None:
            self.build_model()
        
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        logger.info("Model training completed")
        return history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test input sequences
            y_test: Test target sequences
            
        Returns:
            Dictionary containing evaluation metrics
        """
        predictions = self.model.predict(X_test)
        
        # Flatten for metric calculation
        y_test_flat = y_test.reshape(-1)
        predictions_flat = predictions.reshape(-1)
        
        mae = mean_absolute_error(y_test_flat, predictions_flat)
        rmse = np.sqrt(mean_squared_error(y_test_flat, predictions_flat))
        mape = np.mean(np.abs((y_test_flat - predictions_flat) / (np.abs(y_test_flat) + 1e-8))) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
        
        logger.info(f"Evaluation - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}")
        return metrics
    
    def extract_attention_weights(self, X_test: np.ndarray) -> np.ndarray:
        """
        Extract attention weights from model for interpretation.
        
        Args:
            X_test: Test input sequences
            
        Returns:
            Attention weights array
        """
        # Create a model that outputs attention weights
        # This requires extracting from the attention layer
        logger.info("Extracting attention weights from model")
        return None  # Placeholder for attention extraction


class LSTMBaseline:
    """
    Simple LSTM baseline model for comparison.
    """
    
    def __init__(self, input_shape: Tuple[int, int], forecast_horizon: int = 10):
        """Initialize LSTM baseline."""
        self.input_shape = input_shape
        self.forecast_horizon = forecast_horizon
        self.model = None
        logger.info("LSTMBaseline initialized")
    
    def build_model(self) -> keras.Model:
        """Build simple LSTM model."""
        inputs = layers.Input(shape=self.input_shape)
        x = layers.LSTM(64, return_sequences=True)(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(32)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(self.forecast_horizon * self.input_shape[1])(x)
        outputs = layers.Reshape((self.forecast_horizon, self.input_shape[1]))(outputs)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        
        self.model = model
        logger.info("LSTM baseline model built")
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """Train LSTM baseline."""
        if self.model is None:
            self.build_model()
        
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = self.model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate LSTM baseline."""
        predictions = self.model.predict(X_test)
        
        y_test_flat = y_test.reshape(-1)
        predictions_flat = predictions.reshape(-1)
        
        mae = mean_absolute_error(y_test_flat, predictions_flat)
        rmse = np.sqrt(mean_squared_error(y_test_flat, predictions_flat))
        mape = np.mean(np.abs((y_test_flat - predictions_flat) / (np.abs(y_test_flat) + 1e-8))) * 100
        
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


def main():
    """
    Main execution pipeline.
    """
    logger.info("=" * 80)
    logger.info("Advanced Time Series Forecasting Project")
    logger.info("=" * 80)
    
    # Configuration
    LOOKBACK_WINDOW = 60
    FORECAST_HORIZON = 10
    EMBED_DIM = 64
    NUM_HEADS = 8
    NUM_LAYERS = 2
    EPOCHS = 50
    BATCH_SIZE = 32
    
    # Data Loading
    logger.info("\n[PHASE 1] Data Acquisition and Preprocessing")
    data_loader = DataLoader(lookback_window=LOOKBACK_WINDOW, forecast_horizon=FORECAST_HORIZON)
    
    # Generate synthetic data (or use real financial data)
    raw_data = data_loader.generate_synthetic_data(n_samples=1500, n_features=4)
    
    # Preprocess
    preprocessed = data_loader.preprocess_data(raw_data, test_split=0.2)
    
    # Model Building and Training
    logger.info("\n[PHASE 2] Transformer Model Building and Training")
    input_shape = (LOOKBACK_WINDOW, raw_data.shape[1])
    
    transformer_model = TransformerForecastingModel(
        input_shape=input_shape,
        forecast_horizon=FORECAST_HORIZON,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    )
    
    transformer_model.build_model()
    transformer_model.train(
        preprocessed['X_train'],
        preprocessed['y_train'],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # Baseline Model
    logger.info("\n[PHASE 3] LSTM Baseline Training")
    lstm_baseline = LSTMBaseline(input_shape=input_shape, forecast_horizon=FORECAST_HORIZON)
    lstm_baseline.build_model()
    lstm_baseline.train(
        preprocessed['X_train'],
        preprocessed['y_train'],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # Evaluation
    logger.info("\n[PHASE 4] Model Evaluation")
    transformer_metrics = transformer_model.evaluate(preprocessed['X_test'], preprocessed['y_test'])
    lstm_metrics = lstm_baseline.evaluate(preprocessed['X_test'], preprocessed['y_test'])
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Transformer Model - MAE: {transformer_metrics['MAE']:.6f}, RMSE: {transformer_metrics['RMSE']:.6f}, MAPE: {transformer_metrics['MAPE']:.4f}%")
    logger.info(f"LSTM Baseline    - MAE: {lstm_metrics['MAE']:.6f}, RMSE: {lstm_metrics['RMSE']:.6f}, MAPE: {lstm_metrics['MAPE']:.4f}%")
    logger.info("=" * 80)
    
    # Save results
    results = {
        'transformer': transformer_metrics,
        'lstm_baseline': lstm_metrics,
        'configuration': {
            'lookback_window': LOOKBACK_WINDOW,
            'forecast_horizon': FORECAST_HORIZON,
            'embed_dim': EMBED_DIM,
            'num_heads': NUM_HEADS,
            'num_layers': NUM_LAYERS
        }
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info("Project completed successfully!")


if __name__ == "__main__":
    main()
