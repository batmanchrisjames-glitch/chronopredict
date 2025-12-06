#include <iostream>
#include <string>

#include "DataLoader.h"
#include "Preprocessor.h"
#include "LinearRegression.h"
#include "MLP.h"
#include "Metrics.h"

using namespace chrono;

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           ChronoPredict v1.0                               ║\n";
    std::cout << "║     Time-Series Forecasting Engine for Financial Data     ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    std::cout << std::endl;
    
    // Configuration
    const std::string dataFile = "financial_data.csv";
    const int windowSize = 10;
    const int trainTestSplit = 80; // 80% train, 20% test
    
    // Step 1: Create or load data
    std::cout << "[1/6] Loading Data..." << std::endl;
    DataLoader loader;
    
    // Create sample data if file doesn't exist
    loader.createSampleData(dataFile, 500);
    
    TimeSeriesData data = loader.loadCSV(dataFile);
    if (data.prices.empty()) {
        std::cerr << "Error: No data loaded. Exiting." << std::endl;
        return 1;
    }
    
    // Step 2: Preprocess data
    std::cout << "\n[2/6] Preprocessing Data..." << std::endl;
    Preprocessor preprocessor;
    auto normalized = preprocessor.normalize(data.prices);
    auto windows = preprocessor.createWindows(normalized, windowSize);
    
    // Split into train and test sets
    int trainSize = (windows.X.size() * trainTestSplit) / 100;
    std::vector<std::vector<double>> X_train(windows.X.begin(), windows.X.begin() + trainSize);
    std::vector<double> y_train(windows.y.begin(), windows.y.begin() + trainSize);
    std::vector<std::vector<double>> X_test(windows.X.begin() + trainSize, windows.X.end());
    std::vector<double> y_test(windows.y.begin() + trainSize, windows.y.end());
    
    std::cout << "  Train samples: " << X_train.size() << std::endl;
    std::cout << "  Test samples:  " << X_test.size() << std::endl;
    //load existing model, otherwise train
    std::ifstream f(modelFile);
    if (f.good()) {
        std::cout << "  Found saved model. Loading..." << std::endl;
        mlpModel.loadModel(modelFile);
    } else {
        std::cout << "  No saved model found. Training from scratch..." << std::endl;
        mlpModel.train(X_train, y_train, 200, 0.01);
        mlpModel.saveModel(modelFile); // Save for next time
    }
    
    // Step 3: Train Linear Regression
    std::cout << "\n[3/6] Training Linear Regression Model..." << std::endl;
    LinearRegression linearModel;
    linearModel.train(X_train, y_train, 100, 0.01);
    
    // Step 4: Train MLP
    std::cout << "\n[4/6] Training MLP Neural Network..." << std::endl;
    MLP mlpModel(windowSize, 16, 1);
    mlpModel.train(X_train, y_train, 200, 0.01);

    
    // Step 5: Make predictions
    std::cout << "\n[5/6] Making Predictions..." << std::endl;
    auto lr_predictions = linearModel.predict(X_test);
    auto mlp_predictions = mlpModel.predict(X_test);
    
    // Denormalize predictions
    auto y_test_denorm = preprocessor.denormalize(y_test);
    auto lr_pred_denorm = preprocessor.denormalize(lr_predictions);
    auto mlp_pred_denorm = preprocessor.denormalize(mlp_predictions);
    
    // Step 6: Evaluate and display results
    std::cout << "\n[6/6] Evaluation Results..." << std::endl;
    
    Metrics::printMetrics("Linear Regression", y_test_denorm, lr_pred_denorm);
    Metrics::printMetrics("MLP Neural Network", y_test_denorm, mlp_pred_denorm);
    
    // Display ASCII chart
    std::cout << "\n\n=== LINEAR REGRESSION ===" << std::endl;
    Metrics::plotComparison(y_test_denorm, lr_pred_denorm, 60);
    
    std::cout << "\n\n=== MLP NEURAL NETWORK ===" << std::endl;
    Metrics::plotComparison(y_test_denorm, mlp_pred_denorm, 60);
    
    std::cout << "\n\n✓ ChronoPredict execution completed successfully!\n" << std::endl;
    
    return 0;
}
