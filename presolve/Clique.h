#pragma once

#include "../include/Definitions.h"
#include <algorithm>
#include <execution>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/**
 * @class CliqueManager
 * @brief Manages cliques and conflict graphs for binary variables in a model.
 *
 * This class is responsible for detecting and managing cliques within a given model.
 * It also builds and updates conflict graphs for binary variables to facilitate
 * efficient clique detection and management.
 *
 * @details
 * The CliqueManager class provides functionalities to:
 * - Add cliques to the internal storage.
 * - Initialize and update conflict graphs for binary variables.
 * - Detect cliques in a sparse matrix representation of the model.
 * - Print detected cliques and conflict graphs.
 *
 * @note This class is designed to be thread-safe.
 */
class CliqueManager {
private:
    const ModelData &modelData;
    std::vector<std::vector<int>> cliques; // Stores all detected cliques
    std::vector<std::unordered_set<int>> conflictGraph; // Conflict graph for binary variables
    std::mutex clique_mutex; // For thread safety when adding cliques
    std::mutex conflictGraph_mutex; // For thread safety in conflict graph updates

public:
    explicit CliqueManager(const ModelData &mData) : modelData(mData) {}

    void addClique(const std::vector<int> &clique) {
        std::lock_guard<std::mutex> lock(clique_mutex);
        cliques.push_back(clique);
    }

    // Initialize the conflict graph for binary variables
    void initCg(int binary_number) {
        conflictGraph.clear();
        conflictGraph.resize(binary_number); // Resize only for binary variables
    }

    // Build and update the conflict graph using binary variables
    void buildUpdateCg(const std::vector<std::vector<int>> &set, int binary_number) {
        initCg(binary_number); // Initialize conflict graph for binary variables

        // Parallel processing of cliques to build the conflict graph
        std::for_each(std::execution::par_unseq, set.begin(), set.end(), [&](const std::vector<int> &clique) {
            for (size_t i = 0; i < clique.size(); ++i) {
                for (size_t j = i + 1; j < clique.size(); ++j) {
                    int var_i = clique[i];
                    int var_j = clique[j];

                    // Only process binary variables
                    if (var_i < binary_number && var_j < binary_number) {
                        std::lock_guard<std::mutex> lock(conflictGraph_mutex);
                        conflictGraph[var_i].insert(var_j);
                        conflictGraph[var_j].insert(var_i);
                    }
                }
            }
        });
    }

    // Function to find cliques in the sparse matrix
    void findCliques() {
        cliques.clear(); // Clear previous cliques
        std::unordered_map<int, std::vector<std::pair<double, int>>> rowToElements;

        // Populate rowToElements for efficient row-wise access
        for (size_t nz_idx = 0; nz_idx < modelData.A_sparse.row_indices.size(); ++nz_idx) {
            int row = modelData.A_sparse.row_indices[nz_idx];
            int col = modelData.A_sparse.col_indices[nz_idx];
            double coeff = modelData.A_sparse.values[nz_idx];

            rowToElements[row].emplace_back(coeff, col);
        }

        // Parallel processing of rows
        std::for_each(std::execution::par, modelData.b.begin(), modelData.b.end(), [&](const auto &rhs_value) {
            int row = &rhs_value - &modelData.b[0]; // Get row index
            double rhs = rhs_value;

            std::vector<std::pair<double, int>> binCoeffs;
            binCoeffs.reserve(100); // Reserve space for binary coefficients

            auto &rowElements = rowToElements[row];

            // Adjust RHS and collect binary variable coefficients
            for (const auto &elem : rowElements) {
                double coeff = elem.first;
                int col = elem.second;

                if (modelData.vtype[col] != 'B') {
                    rhs -= (coeff > 0 ? coeff * modelData.ub[col] : 0);
                } else {
                    binCoeffs.emplace_back(coeff, col);
                }
            }

            if (binCoeffs.size() < 2) return; // Skip rows with fewer than 2 binary coefficients

            std::sort(binCoeffs.begin(), binCoeffs.end());

            if (binCoeffs.back().first + binCoeffs[binCoeffs.size() - 2].first < rhs) return;

            int k = findFirstClique(binCoeffs, rhs);
            if (k == -1) return;

            addCliqueFromIndex(binCoeffs, k);

            processRemainingCliques(binCoeffs, k, rhs);
        });
    }

    // Helper function to find the first clique index
    int findFirstClique(const std::vector<std::pair<double, int>> &binCoeffs, double rhs) {
        int left = 0, right = binCoeffs.size() - 2, k = -1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (binCoeffs[mid].first + binCoeffs[mid + 1].first > rhs) {
                k = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return k;
    }

    // Add cliques from index k
    void addCliqueFromIndex(const std::vector<std::pair<double, int>> &binCoeffs, int k) {
        std::vector<int> clique;
        clique.reserve(binCoeffs.size() - k);
        for (int j = k; j < binCoeffs.size(); ++j) {
            clique.push_back(binCoeffs[j].second);
        }
        addClique(clique);
    }

    // Process remaining cliques after the first one is added
    void processRemainingCliques(const std::vector<std::pair<double, int>> &binCoeffs, int k, double rhs) {
        for (int o = k - 1; o >= 0; --o) {
            int f = findSmallestF(binCoeffs, o, rhs);
            if (f != -1) {
                std::vector<int> clique_f;
                clique_f.push_back(binCoeffs[o].second);
                for (int j = f; j < binCoeffs.size(); ++j) {
                    clique_f.push_back(binCoeffs[j].second);
                }
                addClique(clique_f);
            }
        }
    }

    // Helper function to find the smallest f index
    int findSmallestF(const std::vector<std::pair<double, int>> &binCoeffs, int o, double rhs) {
        int left = o + 1, right = binCoeffs.size() - 1, f = -1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (binCoeffs[o].first + binCoeffs[mid].first > rhs) {
                f = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return f;
    }

    // Function to print cliques
    void printCliques() {
        for (const auto &clique : cliques) {
            fmt::print("Clique: ");
            for (const auto &c : clique) {
                fmt::print("{} ", c);
            }
            fmt::print("\n");
        }
    }

    // Function to print the conflict graph
    void printCg() {
        for (size_t i = 0; i < conflictGraph.size(); ++i) {
            fmt::print("Variable {} conflicts with: ", i);
            for (const auto &c : conflictGraph[i]) {
                fmt::print("{} ", c);
            }
            fmt::print("\n");
        }
    }
};
