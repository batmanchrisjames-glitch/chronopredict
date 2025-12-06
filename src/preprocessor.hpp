#pragma once
#include <vector>
#include <algorithm>
#include <cmath>

namespace chrono {

struct DataStats {
    double mean = 0.0;
    double std = 0.0;
    double min = 0.0;
    double max = 0.0;
};

class Preprocessor {
public:
    // Normalize data using z-score normalization
    std::vector<double> normalize(const std::vector<double>& data);
    std::vector<double> denormalize(const std::vector<double>& data);
    
    // Create sliding window dataset
    struct WindowData {
        std::vector<std::vector<double>> X; // Input sequences
        std::vector<double> y;               // Target values
    };
    
    WindowData createWindows(const std::vector<double>& data, int windowSize);
    
    DataStats getStats() const { return stats_; }
    
private:
    DataStats stats_;
};

} 