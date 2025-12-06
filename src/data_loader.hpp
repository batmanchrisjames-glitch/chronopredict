#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

namespace chrono {

struct TimeSeriesData {
    std::vector<double> prices;
    std::vector<std::string> dates;
};

class DataLoader {
public:
    TimeSeriesData loadCSV(const std::string& filename, 
                          const std::string& priceColumn = "Close",
                          bool hasHeader = true);
    
    void createSampleData(const std::string& filename, int numPoints = 1000);
};

}
