#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

class ATSPInstance {
public:
    std::string                   name;
    int                           dimension;
    std::vector<std::vector<double>> distance_matrix;

    ATSPInstance(const std::string &filename) { parse_instance(filename); }

    void parse_instance(const std::string &filename) {
        std::ifstream file(filename);
        if (!file.is_open()) { throw std::runtime_error("Could not open file"); }

        std::string      line;
        bool             reading_edge_weights = false;
        std::vector<int> edge_weights;

        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string        key;
            if (line.find(':') != std::string::npos) {
                iss >> key;
                if (key == "NAME:") {
                    iss >> name;
                } else if (key == "DIMENSION:") {
                    iss >> dimension;
                }
            } else if (line == "EDGE_WEIGHT_SECTION") {
                reading_edge_weights = true;
            } else if (reading_edge_weights) {
                if (line == "EOF") { break; }
                int weight;
                while (iss >> weight) { edge_weights.push_back(weight); }
            }
        }

        file.close();

        distance_matrix.resize(dimension, std::vector<double>(dimension));
        for (int i = 0; i < dimension; ++i) {
            for (int j = 0; j < dimension; ++j) {
                distance_matrix[i][j] = edge_weights[i * dimension + j];
                // Treat 0 (except diagonal) as non-travelable paths
                // if (i != j && distance_matrix[i][j] == 0) { distance_matrix[i][j] = GRB_INFINITY; }
            }
        }
    }

    void print_instance() const {
        std::cout << "ATSP Instance: " << name << "\n";
        std::cout << "Dimension: " << dimension << "\n";
        std::cout << "Distance Matrix:\n";
        for (const auto &row : distance_matrix) {
            for (const auto &weight : row) { std::cout << weight << " "; }
            std::cout << "\n";
        }
    }
};