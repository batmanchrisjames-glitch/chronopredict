#include "utils.hpp" 
#include <iostream>  
#include <sstream>
   


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