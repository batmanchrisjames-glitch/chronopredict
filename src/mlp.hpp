#pragma once
#include <vector>
#include <cmath>

namespace chrono {

class MLP {
public:
    MLP(int inputSize, int hiddenSize, int outputSize = 1);
    
    void train(const std::vector<std::vector<double>>& X,
              const std::vector<double>& y,
              int epochs = 200,
              double learningRate = 0.01);
    bool saveModel(const std::string& filename) const;
    bool loadModel(const std::string& filename);          
    
    std::vector<double> predict(const std::vector<std::vector<double>>& X);
    double predict(const std::vector<double>& x);
    
private:
    double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
    double sigmoidDerivative(double x) { double s = sigmoid(x); return s * (1 - s); }
    
    std::vector<double> forward(const std::vector<double>& x);
    
    std::vector<std::vector<double>> weightsInputHidden_;
    std::vector<std::vector<double>> weightsHiddenOutput_;
    std::vector<double> biasHidden_;
    std::vector<double> biasOutput_;
    
    int inputSize_, hiddenSize_, outputSize_;
    std::vector<double> hiddenActivations_;
    std::vector<double> outputActivations_;
};
} 