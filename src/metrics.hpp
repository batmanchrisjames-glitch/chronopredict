#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace chrono {

class Metrics {
public:
    static double mse(const std::vector<double>& actual, 
                     const std::vector<double>& predicted);
    
    static double rmse(const std::vector<double>& actual,
                      const std::vector<double>& predicted);
    
    static double directionAccuracy(const std::vector<double>& actual,
                                   const std::vector<double>& predicted);
    
    static void printMetrics(const std::string& modelName,
                           const std::vector<double>& actual,
                           const std::vector<double>& predicted);
    
    static void plotComparison(const std::vector<double>& actual,
                             const std::vector<double>& predicted,
                             int numPoints = 50);
};

} 
namespace chrono {

double Metrics::mse(const std::vector<double>& actual,
                   const std::vector<double>& predicted) {
    if (actual.size() != predicted.size() || actual.empty()) return 0.0;
    
    double sum = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        double diff = actual[i] - predicted[i];
        sum += diff * diff;
    }
    return sum / actual.size();
}

double Metrics::rmse(const std::vector<double>& actual,
                    const std::vector<double>& predicted) {
    return std::sqrt(mse(actual, predicted));
}

double Metrics::directionAccuracy(const std::vector<double>& actual,
                                 const std::vector<double>& predicted) {
    if (actual.size() < 2 || predicted.size() < 2) return 0.0;
    
    int correct = 0;
    int total = 0;
    
    for (size_t i = 1; i < std::min(actual.size(), predicted.size()); ++i) {
        double actualDir = actual[i] - actual[i-1];
        double predDir = predicted[i] - predicted[i-1];
        
        if ((actualDir >= 0 && predDir >= 0) || (actualDir < 0 && predDir < 0)) {
            correct++;
        }
        total++;
    }
    
    return total > 0 ? (100.0 * correct / total) : 0.0;
}

void Metrics::printMetrics(const std::string& modelName,
                          const std::vector<double>& actual,
                          const std::vector<double>& predicted) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  " << modelName << " - Evaluation Metrics" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  MSE:                " << mse(actual, predicted) << std::endl;
    std::cout << "  RMSE:               " << rmse(actual, predicted) << std::endl;
    std::cout << "  Direction Accuracy: " << directionAccuracy(actual, predicted) << "%" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void Metrics::plotComparison(const std::vector<double>& actual,
                            const std::vector<double>& predicted,
                            int numPoints) {
    if (actual.empty() || predicted.empty()) return;
    
    numPoints = std::min(numPoints, (int)actual.size());
    
    // Find min and max for scaling
    double minVal = actual[0], maxVal = actual[0];
    for (int i = 0; i < numPoints; ++i) {
        minVal = std::min({minVal, actual[i], predicted[i]});
        maxVal = std::max({maxVal, actual[i], predicted[i]});
    }
    
    const int height = 20;
    const int width = numPoints;
    
    std::cout << "\n" << std::string(width + 20, '=') << std::endl;
    std::cout << "  Actual vs Predicted (Last " << numPoints << " points)" << std::endl;
    std::cout << "  Legend: * = Actual, + = Predicted" << std::endl;
    std::cout << std::string(width + 20, '=') << std::endl;
    
    // Create plot grid
    for (int row = 0; row < height; ++row) {
        double threshold = maxVal - (maxVal - minVal) * row / (height - 1);
        
        std::cout << std::setw(8) << std::fixed << std::setprecision(1) << threshold << " |";
        
        for (int col = 0; col < numPoints; ++col) {
            int idx = actual.size() - numPoints + col;
            char ch = ' ';
            
            double actualVal = actual[idx];
            double predVal = predicted[idx];
            
            bool actualClose = std::abs(actualVal - threshold) < (maxVal - minVal) / (height * 2);
            bool predClose = std::abs(predVal - threshold) < (maxVal - minVal) / (height * 2);
            
            if (actualClose && predClose) ch = 'X';
            else if (actualClose) ch = '*';
            else if (predClose) ch = '+';
            
            std::cout << ch;
        }
        std::cout << std::endl;
    }
    
    std::cout << "         +" << std::string(width, '-') << std::endl;
    std::cout << std::string(width + 20, '=') << std::endl;
}

}