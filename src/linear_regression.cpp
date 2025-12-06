#include "linear_regression.h"
#include "progress_bar.h"
#include <iostream>


namespace chrono {

void LinearRegression::train(const std::vector<std::vector<double>>& X,
                             const std::vector<double>& y,
                             int epochs,
                             double learningRate) {
    if (X.empty() || y.empty()) return;
    
    int numFeatures = X[0].size();
    int numSamples = X.size();
    
    // Initialize weights
    weights_.resize(numFeatures, 0.0);
    bias_ = 0.0;
    
    std::cout << "\nTraining Linear Regression..." << std::endl;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<double> gradWeights(numFeatures, 0.0);
        double gradBias = 0.0;
        double loss = 0.0;
        
        // Calculate gradients
        for (int i = 0; i < numSamples; ++i) {
            double pred = bias_;
            for (int j = 0; j < numFeatures; ++j) {
                pred += weights_[j] * X[i][j];
            }
            
            double error = pred - y[i];
            loss += error * error;
            
            gradBias += error;
            for (int j = 0; j < numFeatures; ++j) {
                gradWeights[j] += error * X[i][j];
            }
        }
        
        // Update weights
        bias_ -= learningRate * gradBias / numSamples;
        for (int j = 0; j < numFeatures; ++j) {
            weights_[j] -= learningRate * gradWeights[j] / numSamples;
        }
        
        if (epoch % 10 == 0) {
            printProgress(epoch + 1, epochs, "  Epoch ");
        }
    }
    std::cout << "\n  Training complete!" << std::endl;
}

std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& X) {
    std::vector<double> predictions;
    predictions.reserve(X.size());
    
    for (const auto& x : X) {
        predictions.push_back(predict(x));
    }
    
    return predictions;
}

double LinearRegression::predict(const std::vector<double>& x) {
    double pred = bias_;
    for (size_t i = 0; i < x.size() && i < weights_.size(); ++i) {
        pred += weights_[i] * x[i];
    }
    return pred;
}

} 