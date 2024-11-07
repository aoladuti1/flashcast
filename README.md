
# PyTorch LSTM Time-Series Forecasting Tool

This module provides an LSTM-based tool for (time-series) prediction and is especially suited for financial and trading data.

## Key Features

1. **LSTM Model**: A basic customizable LSTM neural network for time-series forecasting.
  
2. **Data Processing**: Prepares features, handles holidays, splits data, and fills missing values for robust data handling.

3. **Model Training**: Supports training with gradient clipping, custom loss functions, batch handling, and evaluation options.

4. **Prediction & Forecasting**: Allows forecasting with options for holiday filtering, binary classification, and multiple model support.

5. **Evaluation & Visualization**: Includes validation, prediction aggregation, and visualization for model evaluation.

6. **Utilities**: Provides functions to save, load, and interpret models effectively.

## Requirements

- **Python**: 3.10+
- **Dependencies**: `torch`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `holidays`

## Example Usage

Can be found in the demo Jupyter notebook: `flashcaster_demo.ipynb`.
