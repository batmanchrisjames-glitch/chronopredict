
#include "data_loader.hpp"
#include "random.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include "utils.hpp"

namespace chrono {

TimeSeriesData DataLoader::loadCSV(const std::string& filename, 
                                   const std::string& priceColumn,
                                   bool hasHeader) {
    TimeSeriesData data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }
    
    std::string line;
    int priceColIdx = -1;
    int dateColIdx = 0;
    
    // Parse header if exists
    if (hasHeader && std::getline(file, line)) {
        auto headers = split(line, ',');
        for (size_t i = 0; i < headers.size(); ++i) {
            if (headers[i] == priceColumn || headers[i] == "\"" + priceColumn + "\"") {
                priceColIdx = i;
                break;
            }
        }
        if (priceColIdx == -1) priceColIdx = headers.size() - 1; // Default to last column
    } else {
        priceColIdx = 1; // Default to second column
    }
    
    // Parse data rows
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        auto values = split(line, ',');
        
        if (values.size() > priceColIdx) {
            try {
                data.prices.push_back(std::stod(values[priceColIdx]));
                data.dates.push_back(values[dateColIdx]);
            } catch (...) {
                continue; // Skip malformed rows
            }
        }
    }
    
    file.close();
    std::cout << "Loaded " << data.prices.size() << " data points from " << filename << std::endl;
    return data;
}

void DataLoader::createSampleData(const std::string& filename, int numPoints) {
    std::ofstream file(filename);
    file << "Date,Close\n";
    
    double price = 100.0;
    auto& rng = Random::instance();
    
    for (int i = 0; i < numPoints; ++i) {
        // Random walk with trend
        double change = rng.normal(0.05, 1.5);
        price += change;
        price = std::max(10.0, price); // Keep positive
        
        file << "2024-" << (i % 12 + 1) << "-" << (i % 28 + 1) << "," << price << "\n";
    }
    
    file.close();
    std::cout << "Created sample data file: " << filename << std::endl;
}

}