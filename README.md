# ChronoPredict 📈
 
**ChronoPredict** is a lightweight, C++ based time-series forecasting engine built from scratch. It implements financial data processing, Linear Regression, and a Multi-Layer Perceptron (MLP) Neural Network without relying on external machine learning libraries like PyTorch or TensorFlow.

## 🚀 Features

*   **Zero Dependencies:** Built using only the C++ Standard Library (STL).
*   **Custom Architecture:** Implements a sliding-window mechanism for time-series data.
*   **Dual Models:**
    *   **Linear Regression:** For trend-based analysis.
    *   **MLP Neural Network:** Uses Backpropagation and Sigmoid activation for non-linear patterns.
*   **Data Pipeline:** Handles CSV loading, normalization (Z-Score), and splitting (Train/Test).
*   **Visualization:** Renders ASCII-based charts in the terminal for immediate feedback.

## 📂 Project Structure

```text
ChronoPredict/
├── CMakeLists.txt       # Build configuration
├── README.md            # Documentation
├── main.cpp             # Entry point
├── data/                # Data files
│   └── financial.csv
└── src/
    ├── utils.hpp / .cpp            # Helper functions & Random engine
    ├── data_loader.hpp / .cpp      # CSV parsing
    ├── preprocessor.hpp / .cpp     # Normalization & Windowing
    ├── metrics.hpp / .cpp          # MSE, RMSE, Accuracy calcs
    ├── linear_regression.hpp / .cpp
    └── mlp.hpp / .cpp              # Neural Network implementation

```
