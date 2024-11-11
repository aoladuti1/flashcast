
# FlashCast: RNN Forecasting with Time Series in Mind

This module provides a programmer tool for prediction with RNNs. It is ideal for those who want to experiment on time series data with a PyTorch abstraction.

## Major Features

-**LSTM Model**: The Magic class is a basic Torch LSTM neural network for time series forecasting.
  
-**Data Processing**: Prepares features, handles holidays, splits data, and fills missing values for robustness.

-**Model Training**: Supports training with sliding lookback windows (sampling), gradient clipping, custom loss functions, batch handling, and evaluation options.

-**Prediction & Forecasting**: Allows forecasting with options for holiday filtering, binary classification, and multiple model support.

-**Evaluation & Visualization**: Includes historical validation, visualization, and mutliple prediction configurations + output aggregation.

-**Utilities**: Provides functions to save, load, and interpret models.

-**Broadcasting**: When providing multiple configurations for prediction on a dataset, they don't have to be fully specified. One-dimensional or single-entry values in the configuration will be broadcast to all configurations.

## Requirements

- **Python**: 3.10+
- **Dependencies**: Run ```pip install -r requirements.txt``` in a terminal pointed to the requirements.txt directory

## Example Usage

Can be found in the demo Jupyter notebook: `flashcast_demo.ipynb`.
