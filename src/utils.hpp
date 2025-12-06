#pragma once
#include <vector>
#include <string>
#include <random>

namespace chrono {

class Random {
public:
    static Random& instance() {
        static Random inst;
        return inst;
    }
    
    double uniform(double min = 0.0, double max = 1.0) {
        std::uniform_real_distribution<double> dist(min, max);
        return dist(gen);
    }
    
    double normal(double mean = 0.0, double stddev = 1.0) {
        std::normal_distribution<double> dist(mean, stddev);
        return dist(gen);
    }
    
private:
    Random() : gen(42) {}
    std::mt19937 gen;
};

std::vector<std::string> split(const std::string& str, char delim);
void printProgress(int current, int total, const std::string& prefix = "");

} 
namespace chrono {

std::vector<std::string> split(const std::string& str, char delim) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream stream(str);
    while (std::getline(stream, token, delim)) {
        tokens.push_back(token);
    }
    return tokens;
}

void printProgress(int current, int total, const std::string& prefix) {
    int barWidth = 50;
    float progress = (float)current / total;
    int pos = barWidth * progress;
    
    std::cout << prefix << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "%\r";
    std::cout.flush();
}

} 