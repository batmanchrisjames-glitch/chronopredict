

# ChronoPredict Setup Script
# This script provides instructions to compile and run the ChronoPredict program.

Multi-file compilation (recommended):
    g++ -std=c++11 -O2 -c utils.cpp
    g++ -std=c++11 -O2 -c data_loader.cpp
    g++ -std=c++11 -O2 -c preprocessor.cpp
    g++ -std=c++11 -O2 -c linear_regression.cpp
    g++ -std=c++11 -O2 -c mlp.cpp
    g++ -std=c++11 -O2 -c metrics.cpp
    g++ -std=c++11 -O2 -c main.cpp
    g++ -o chronopredict *.o

Run:
    ./chronopredict

# The program will:
        1. Create a sample financial dataset (or load existing CSV)
        2. Preprocess data with normalization and sliding windows
        3. Train both Linear Regression and MLP models
        4. Evaluate with MSE, RMSE, and direction accuracy
        5. Display ASCII charts comparing actual vs predicted values

        CSV Format:
            Date,Close
            2024-1-1,100.5
            2024-1-2,102.3
            ...
        