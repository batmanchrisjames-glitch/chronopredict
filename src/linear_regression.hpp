#pragma once
#include <vector>
#include <numeric>

namespace chrono {

class LinearRegression {
public:
    void train(const std::vector<std::vector<double>>& X, 
              const std::vector<double>& y,
              int epochs = 100,
              double learningRate = 0.01);
    
    std::vector<double> predict(const std::vector<std::vector<double>>& X);
    double predict(const std::vector<double>& x);
    
private:
    std::vector<double> weights_;
    double bias_ = 0.0;
};

} 
