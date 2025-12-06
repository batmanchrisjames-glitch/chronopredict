
#include "mlp.hpp"
#include <iostream>
#include "core/ChRandom.h"
#include "core/ChProgressBar.h"

namespace chrono {

MLP::MLP(int inputSize, int hiddenSize, int outputSize)
    : inputSize_(inputSize), hiddenSize_(hiddenSize), outputSize_(outputSize) {
    
    auto& rng = Random::instance();
    
    // Initialize weights with Xavier initialization
    double limitIH = std::sqrt(6.0 / (inputSize + hiddenSize));
    weightsInputHidden_.resize(inputSize, std::vector<double>(hiddenSize));
    for (auto& row : weightsInputHidden_) {
        for (auto& w : row) {
            w = rng.uniform(-limitIH, limitIH);
        }
    }
    
    double limitHO = std::sqrt(6.0 / (hiddenSize + outputSize));
    weightsHiddenOutput_.resize(hiddenSize, std::vector<double>(outputSize));
    for (auto& row : weightsHiddenOutput_) {
        for (auto& w : row) {
            w = rng.uniform(-limitHO, limitHO);
        }
    }
    
    biasHidden_.resize(hiddenSize, 0.0);
    biasOutput_.resize(outputSize, 0.0);
}

std::vector<double> MLP::forward(const std::vector<double>& x) {
    // Hidden layer
    hiddenActivations_.resize(hiddenSize_);
    for (int i = 0; i < hiddenSize_; ++i) {
        double sum = biasHidden_[i];
        for (int j = 0; j < inputSize_; ++j) {
            sum += x[j] * weightsInputHidden_[j][i];
        }
        hiddenActivations_[i] = sigmoid(sum);
    }
    
    // Output layer
    outputActivations_.resize(outputSize_);
    for (int i = 0; i < outputSize_; ++i) {
        double sum = biasOutput_[i];
        for (int j = 0; j < hiddenSize_; ++j) {
            sum += hiddenActivations_[j] * weightsHiddenOutput_[j][i];
        }
        outputActivations_[i] = sum; // Linear output for regression
    }
    
    return outputActivations_;
}

void MLP::train(const std::vector<std::vector<double>>& X,
               const std::vector<double>& y,
               int epochs,
               double learningRate) {
    
    std::cout << "\nTraining MLP Neural Network..." << std::endl;
    int numSamples = X.size();
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalLoss = 0.0;
        
        for (int s = 0; s < numSamples; ++s) {
            // Forward pass
            forward(X[s]);
            
            // Calculate loss
            double error = outputActivations_[0] - y[s];
            totalLoss += error * error;
            
            // Backpropagation
            std::vector<double> outputGrad(outputSize_);
            outputGrad[0] = 2.0 * error; // MSE derivative
            
            std::vector<double> hiddenGrad(hiddenSize_, 0.0);
            for (int i = 0; i < hiddenSize_; ++i) {
                for (int j = 0; j < outputSize_; ++j) {
                    hiddenGrad[i] += outputGrad[j] * weightsHiddenOutput_[i][j];
                }
                hiddenGrad[i] *= sigmoidDerivative(hiddenActivations_[i]);
            }
            
            // Update weights hidden->output
            for (int i = 0; i < hiddenSize_; ++i) {
                for (int j = 0; j < outputSize_; ++j) {
                    weightsHiddenOutput_[i][j] -= learningRate * outputGrad[j] * hiddenActivations_[i] / numSamples;
                }
            }
            
            // Update weights input->hidden
            for (int i = 0; i < inputSize_; ++i) {
                for (int j = 0; j < hiddenSize_; ++j) {
                    weightsInputHidden_[i][j] -= learningRate * hiddenGrad[j] * X[s][i] / numSamples;
                }
            }
            
            // Update biases
            for (int i = 0; i < outputSize_; ++i) {
                biasOutput_[i] -= learningRate * outputGrad[i] / numSamples;
            }
            for (int i = 0; i < hiddenSize_; ++i) {
                biasHidden_[i] -= learningRate * hiddenGrad[i] / numSamples;
            }
        }
        
        if (epoch % 20 == 0) {
            printProgress(epoch + 1, epochs, "  Epoch ");
        }
    }
    std::cout << "\n  Training complete!" << std::endl;
}

std::vector<double> MLP::predict(const std::vector<std::vector<double>>& X) {
    std::vector<double> predictions;
    predictions.reserve(X.size());
    
    for (const auto& x : X) {
        predictions.push_back(predict(x));
    }
    
    return predictions;
}

double MLP::predict(const std::vector<double>& x) {
    return forward(x)[0];
}

} 