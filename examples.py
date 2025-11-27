# Configuration and Example Usage
# Advanced Time Series Forecasting Project

"""
This file demonstrates various usage patterns and configurations
for the time series forecasting project.
"""

import json
from time_series_main import (
    DataLoader,
    TransformerForecastingModel,
    LSTMBaseline
)
from evaluation_utils import ModelEvaluator, ComparisonAnalyzer


# ============================================================================
# CONFIGURATION PRESETS
# ============================================================================

CONFIG_LIGHTWEIGHT = {
    'lookback_window': 30,
    'forecast_horizon': 5,
    'embed_dim': 32,
    'num_heads': 4,
    'num_layers': 1,
    'ff_dim': 64,
    'dropout_rate': 0.05,
    'epochs': 20,
    'batch_size': 64
}

CONFIG_BALANCED = {
    'lookback_window': 60,
    'forecast_horizon': 10,
    'embed_dim': 64,
    'num_heads': 8,
    'num_layers': 2,
    'ff_dim': 128,
    'dropout_rate': 0.1,
    'epochs': 100,
    'batch_size': 32
}

CONFIG_HEAVY = {
    'lookback_window': 120,
    'forecast_horizon': 20,
    'embed_dim': 128,
    'num_heads': 16,
    'num_layers': 4,
    'ff_dim': 256,
    'dropout_rate': 0.15,
    'epochs': 200,
    'batch_size': 16
}

CONFIG_FINANCIAL = {
    'lookback_window': 60,  # 60 trading days ~3 months
    'forecast_horizon': 10,  # 10 days ahead
    'embed_dim': 64,
    'num_heads': 8,
    'num_layers': 3,
    'ff_dim': 128,
    'dropout_rate': 0.1,
    'epochs': 150,
    'batch_size': 32
}


# ============================================================================
# EXAMPLE 1: Quick Start with Synthetic Data
# ============================================================================

def example_quick_start():
    """
    Minimal example to get started quickly.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Quick Start")
    print("="*60)
    
    # Data
    data_loader = DataLoader(lookback_window=60, forecast_horizon=10)
    synthetic_data = data_loader.generate_synthetic_data(n_samples=1000)
    preprocessed = data_loader.preprocess_data(synthetic_data)
    
    # Model
    model = TransformerForecastingModel(
        input_shape=(60, synthetic_data.shape[1]),
        forecast_horizon=10,
        embed_dim=64,
        num_heads=8,
        num_layers=2
    )
    
    # Train
    model.build_model()
    model.train(
        preprocessed['X_train'],
        preprocessed['y_train'],
        epochs=50,
        batch_size=32
    )
    
    # Evaluate
    metrics = model.evaluate(preprocessed['X_test'], preprocessed['y_test'])
    print(f"\nResults:")
    print(f"  MAE:  {metrics['MAE']:.6f}")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  MAPE: {metrics['MAPE']:.4f}%")


# ============================================================================
# EXAMPLE 2: Using Real Financial Data
# ============================================================================

def example_financial_data():
    """
    Using real stock market data for forecasting.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Real Financial Data")
    print("="*60)
    
    data_loader = DataLoader(
        lookback_window=60,
        forecast_horizon=10
    )
    
    try:
        # Download real data
        stock_data = data_loader.acquire_financial_data(
            ticker='AAPL',
            days=365
        )
        
        # Extract closing prices
        prices = stock_data['Close'].values.reshape(-1, 1)
        
        # Preprocess
        preprocessed = data_loader.preprocess_data(prices)
        
        print(f"Successfully loaded {len(prices)} trading days of AAPL data")
        print(f"Training set size: {preprocessed['X_train'].shape[0]}")
        print(f"Test set size: {preprocessed['X_test'].shape[0]}")
        
    except Exception as e:
        print(f"Note: Could not fetch real data ({e})")
        print("Using synthetic data instead...")
        synthetic_data = data_loader.generate_synthetic_data(n_samples=1500, n_features=1)
        preprocessed = data_loader.preprocess_data(synthetic_data)


# ============================================================================
# EXAMPLE 3: Lightweight Configuration (Fast Training)
# ============================================================================

def example_lightweight():
    """
    For testing/debugging with minimal training time.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Lightweight Configuration (Fast Training)")
    print("="*60)
    
    config = CONFIG_LIGHTWEIGHT
    
    # Data
    data_loader = DataLoader(
        lookback_window=config['lookback_window'],
        forecast_horizon=config['forecast_horizon']
    )
    synthetic_data = data_loader.generate_synthetic_data(n_samples=500)
    preprocessed = data_loader.preprocess_data(synthetic_data)
    
    # Model
    model = TransformerForecastingModel(
        input_shape=(config['lookback_window'], synthetic_data.shape[1]),
        forecast_horizon=config['forecast_horizon'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        ff_dim=config['ff_dim'],
        dropout_rate=config['dropout_rate']
    )
    
    print(f"Model Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Train
    model.build_model()
    print("\nTraining started...")
    model.train(
        preprocessed['X_train'],
        preprocessed['y_train'],
        epochs=config['epochs'],
        batch_size=config['batch_size']
    )
    
    # Evaluate
    metrics = model.evaluate(preprocessed['X_test'], preprocessed['y_test'])
    print(f"\nResults:")
    print(f"  MAE:  {metrics['MAE']:.6f}")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  MAPE: {metrics['MAPE']:.4f}%")


# ============================================================================
# EXAMPLE 4: Comparing Multiple Models
# ============================================================================

def example_model_comparison():
    """
    Train both Transformer and LSTM, compare performance.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Model Comparison")
    print("="*60)
    
    # Data
    config = CONFIG_BALANCED
    data_loader = DataLoader(
        lookback_window=config['lookback_window'],
        forecast_horizon=config['forecast_horizon']
    )
    synthetic_data = data_loader.generate_synthetic_data(n_samples=1000)
    preprocessed = data_loader.preprocess_data(synthetic_data)
    
    # Transformer
    print("\nTraining Transformer Model...")
    transformer = TransformerForecastingModel(
        input_shape=(config['lookback_window'], synthetic_data.shape[1]),
        forecast_horizon=config['forecast_horizon'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers']
    )
    transformer.build_model()
    transformer.train(
        preprocessed['X_train'],
        preprocessed['y_train'],
        epochs=50,  # Reduced for demonstration
        batch_size=config['batch_size']
    )
    transformer_metrics = transformer.evaluate(
        preprocessed['X_test'],
        preprocessed['y_test']
    )
    
    # LSTM Baseline
    print("\nTraining LSTM Baseline...")
    lstm = LSTMBaseline(
        input_shape=(config['lookback_window'], synthetic_data.shape[1]),
        forecast_horizon=config['forecast_horizon']
    )
    lstm.build_model()
    lstm.train(
        preprocessed['X_train'],
        preprocessed['y_train'],
        epochs=50
    )
    lstm_metrics = lstm.evaluate(
        preprocessed['X_test'],
        preprocessed['y_test']
    )
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"\nTransformer:")
    print(f"  MAE:  {transformer_metrics['MAE']:.6f}")
    print(f"  RMSE: {transformer_metrics['RMSE']:.6f}")
    print(f"  MAPE: {transformer_metrics['MAPE']:.4f}%")
    
    print(f"\nLSTM Baseline:")
    print(f"  MAE:  {lstm_metrics['MAE']:.6f}")
    print(f"  RMSE: {lstm_metrics['RMSE']:.6f}")
    print(f"  MAPE: {lstm_metrics['MAPE']:.4f}%")
    
    # Calculate improvement
    mae_improvement = ((lstm_metrics['MAE'] - transformer_metrics['MAE']) / lstm_metrics['MAE']) * 100
    rmse_improvement = ((lstm_metrics['RMSE'] - transformer_metrics['RMSE']) / lstm_metrics['RMSE']) * 100
    mape_improvement = ((lstm_metrics['MAPE'] - transformer_metrics['MAPE']) / lstm_metrics['MAPE']) * 100
    
    print(f"\nImprovement (Transformer vs LSTM):")
    print(f"  MAE:  {mae_improvement:.2f}% ↓")
    print(f"  RMSE: {rmse_improvement:.2f}% ↓")
    print(f"  MAPE: {mape_improvement:.2f}% ↓")


# ============================================================================
# EXAMPLE 5: Multi-Feature Time Series
# ============================================================================

def example_multivariate():
    """
    Working with multiple correlated time series features.
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Multivariate Time Series (4 Features)")
    print("="*60)
    
    config = CONFIG_BALANCED
    
    # Generate 4-feature time series
    data_loader = DataLoader(
        lookback_window=config['lookback_window'],
        forecast_horizon=config['forecast_horizon']
    )
    synthetic_data = data_loader.generate_synthetic_data(
        n_samples=1500,
        n_features=4  # Multiple features
    )
    preprocessed = data_loader.preprocess_data(synthetic_data)
    
    print(f"Data shape: {synthetic_data.shape}")
    print(f"  Timesteps: {synthetic_data.shape[0]}")
    print(f"  Features: {synthetic_data.shape[1]}")
    
    # Model
    model = TransformerForecastingModel(
        input_shape=(config['lookback_window'], 4),
        forecast_horizon=config['forecast_horizon'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers']
    )
    
    # Train
    model.build_model()
    model.train(
        preprocessed['X_train'],
        preprocessed['y_train'],
        epochs=50,
        batch_size=config['batch_size']
    )
    
    # Evaluate per-feature
    predictions = model.model.predict(preprocessed['X_test'])
    evaluator = ModelEvaluator(predictions, preprocessed['y_test'])
    metrics = evaluator.compute_metrics()
    
    print(f"\nOverall Metrics:")
    print(f"  MAE:  {metrics['overall_mae']:.6f}")
    print(f"  RMSE: {metrics['overall_rmse']:.6f}")
    print(f"  MAPE: {metrics['overall_mape']:.4f}%")
    
    print(f"\nPer-Feature Performance:")
    for feat_idx, mae in enumerate(metrics['per_feature_mae']):
        print(f"  Feature {feat_idx}: MAE = {mae:.6f}")


# ============================================================================
# EXAMPLE 6: Saving and Loading Models
# ============================================================================

def example_save_load():
    """
    Train model and save for later use.
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Model Persistence")
    print("="*60)
    
    # Train
    data_loader = DataLoader(60, 10)
    synthetic_data = data_loader.generate_synthetic_data(n_samples=1000)
    preprocessed = data_loader.preprocess_data(synthetic_data)
    
    model = TransformerForecastingModel(
        input_shape=(60, 4),
        forecast_horizon=10,
        embed_dim=64,
        num_heads=8,
        num_layers=2
    )
    model.build_model()
    model.train(
        preprocessed['X_train'],
        preprocessed['y_train'],
        epochs=30,
        batch_size=32
    )
    
    # Save
    print("\nSaving model...")
    model.model.save('transformer_model.h5')
    print("  ✓ Model weights saved to 'transformer_model.h5'")
    
    # Save config
    config = {
        'lookback_window': 60,
        'forecast_horizon': 10,
        'embed_dim': 64,
        'num_heads': 8,
        'num_layers': 2,
        'ff_dim': 128
    }
    with open('model_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    print("  ✓ Configuration saved to 'model_config.json'")
    
    # Evaluate
    metrics = model.evaluate(preprocessed['X_test'], preprocessed['y_test'])
    results = {
        'mae': float(metrics['MAE']),
        'rmse': float(metrics['RMSE']),
        'mape': float(metrics['MAPE'])
    }
    with open('model_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("  ✓ Results saved to 'model_results.json'")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("Advanced Time Series Forecasting - Example Usage")
    print("="*60)
    
    examples = {
        '1': ('Quick Start', example_quick_start),
        '2': ('Financial Data', example_financial_data),
        '3': ('Lightweight Config', example_lightweight),
        '4': ('Model Comparison', example_model_comparison),
        '5': ('Multivariate Series', example_multivariate),
        '6': ('Save/Load Models', example_save_load),
        'all': ('Run All Examples', None)
    }
    
    print("\nAvailable Examples:")
    for key, (name, _) in examples.items():
        if key != 'all':
            print(f"  {key}. {name}")
    print(f"  all. Run All Examples")
    
    # Get user input or use command line argument
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nSelect example (1-6 or 'all'): ").strip()
    
    if choice == 'all':
        example_quick_start()
        example_lightweight()
        example_multivariate()
    elif choice in examples and choice != 'all':
        _, func = examples[choice]
        if func:
            func()
    else:
        print(f"Invalid choice: {choice}")
        print(f"Valid options: {', '.join(examples.keys())}")
