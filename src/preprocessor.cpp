#include "preprocessor.hpp" 
#include <iostream>  
#include <sstream>
   

namespace chrono {

std::vector<double> Preprocessor::normalize(const std::vector<double>& data) {
    if (data.empty()) return {};
    
    // Calculate statistics
    stats_.mean = 0.0;
    for (double val : data) stats_.mean += val;
    stats_.mean /= data.size();
    
    stats_.std = 0.0;
    for (double val : data) {
        stats_.std += (val - stats_.mean) * (val - stats_.mean);
    }
    stats_.std = std::sqrt(stats_.std / data.size());
    
    stats_.min = *std::min_element(data.begin(), data.end());
    stats_.max = *std::max_element(data.begin(), data.end());
    
    // Normalize
    std::vector<double> normalized;
    normalized.reserve(data.size());
    for (double val : data) {
        normalized.push_back((val - stats_.mean) / (stats_.std + 1e-8));
    }
    
    return normalized;
}

std::vector<double> Preprocessor::denormalize(const std::vector<double>& data) {
    std::vector<double> denormalized;
    denormalized.reserve(data.size());
    for (double val : data) {
        denormalized.push_back(val * stats_.std + stats_.mean);
    }
    return denormalized;
}

Preprocessor::WindowData Preprocessor::createWindows(const std::vector<double>& data, 
                                                     int windowSize) {
    WindowData windows;
    
    for (size_t i = 0; i + windowSize < data.size(); ++i) {
        std::vector<double> window(data.begin() + i, data.begin() + i + windowSize);
        windows.X.push_back(window);
        windows.y.push_back(data[i + windowSize]);
    }
    
    std::cout << "Created " << windows.X.size() << " windows of size " 
              << windowSize << std::endl;
    return windows;
}

}