/*
 * @file Clique.h
 * @brief Header file for clique detection and conflict graph management.
 *
 * This file provides functions for detecting cliques in the conflict graph and managing the conflict graph for binary
 * variables. The CliqueManager class contains methods for adding cliques, building and updating the conflict graph,
 * and finding cliques using the Bron-Kerbosch algorithm. The class also provides options to set parameters such as
 * minimum fractional value, minimum violation, and maximum number of recursive calls.
 *
 */

#pragma once

#include "Definitions.h"

#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

#include <algorithm>
#include <execution>
#include <mutex>
#include <vector>

#include "ankerl/unordered_dense.h"

class CliqueManager {
private:
    const ModelData              &modelData;
    std::vector<std::vector<int>> cliques; // Stores all detected cliques
    std::vector<std::vector<bool>>
               conflictGraph;       // Conflict graph for binary variables (using bools for memory efficiency)
    std::mutex clique_mutex;        // For thread safety when adding cliques
    std::mutex conflictGraph_mutex; // For thread safety in conflict graph updates
    size_t     maxCallsBK;          // Limit for clique detection calls
    double     minFrac;             // Minimum value to consider in clique detection
    double     minViol;             // Minimum violation to store a clique
    double     BKCLQ_MULTIPLIER;    // Multiplier for scaling
    double     BKCLQ_EPS;           // Small tolerance for numerical stability
    size_t     cap_;                // Capacity for clique storage (dynamic resizing)
    double    *vertexWeight_;       // Vertex weights for cliques
    double    *rc_;                 // Reduced costs of variables

public:
    explicit CliqueManager(const ModelData &mData)
        : modelData(mData), maxCallsBK(1000), minFrac(0.001), minViol(0.02), BKCLQ_MULTIPLIER(1000.0), BKCLQ_EPS(1e-6),
          cap_(0), vertexWeight_(nullptr), rc_(nullptr) {
        preallocateMemory(modelData.b.size());
    }

    void addClique(const std::vector<int> &clique) {
        std::lock_guard<std::mutex> lock(clique_mutex);
        cliques.push_back(clique);
    }

    // Preallocate memory for conflictGraph and cliques
    void preallocateMemory(size_t size) {
        conflictGraph.resize(size, std::vector<bool>(size, false)); // Initialize conflict graph as a matrix of bools
        cliques.reserve(4096);                                      // Preallocate space for 4096 cliques initially
    }

    // Set the minimum fractional value to consider variables in clique detection
    void setMinFrac(const double minFrac) { this->minFrac = minFrac; }

    // Set the minimum violation for cliques to be stored
    void setMinViol(const double minViol) { this->minViol = minViol; }

    // Set the maximum number of recursive calls in Bron-Kerbosch algorithm
    void setMaxCallsBK(size_t maxCallsBK) { this->maxCallsBK = maxCallsBK; }

    // Build and update the conflict graph using binary variables
    void buildUpdateCg(const std::vector<std::vector<int>> &set, int binary_number) {
        initCg(binary_number); // Initialize conflict graph for binary variables

        // Create tasks for each combination of clique[i] and clique[j] for all cliques in `set`
        std::vector<std::tuple<int, int>> tasks;
        tasks.reserve(set.size() * (binary_number * (binary_number - 1)) / 2); // Preallocate task space for conflicts

        for (const auto &clique : set) {
            for (size_t i = 0; i < clique.size(); ++i) {
                for (size_t j = i + 1; j < clique.size(); ++j) {
                    int var_i = clique[i];
                    int var_j = clique[j];

                    // Only process binary variables
                    if (var_i < binary_number && var_j < binary_number) { tasks.emplace_back(var_i, var_j); }
                }
            }
        }

        // Define chunk size for processing tasks in batches
        const int chunk_size = 100; // Adjust based on performance needs

        // Define a bulk sender to process tasks in parallel
        auto bulk_sender =
            stdexec::bulk(stdexec::just(), (tasks.size() + chunk_size - 1) / chunk_size,
                          [this, &tasks, chunk_size](std::size_t chunk_idx) {
                              size_t start_idx = chunk_idx * chunk_size;
                              size_t end_idx   = std::min(start_idx + chunk_size, tasks.size());

                              // Process a chunk of tasks
                              for (size_t task_idx = start_idx; task_idx < end_idx; ++task_idx) {
                                  const auto &[var_i, var_j] = tasks[task_idx];

                                  // Mark conflict between var_i and var_j (and vice versa) in the conflict graph
                                  {
                                      std::lock_guard<std::mutex> lock(conflictGraph_mutex);
                                      conflictGraph[var_i][var_j] = true; // Mark conflict between var_i and var_j
                                      conflictGraph[var_j][var_i] = true; // Mark conflict between var_j and var_i
                                  }
                              }
                          });
        exec::static_thread_pool pool(std::thread::hardware_concurrency());
        auto                     sched = pool.get_scheduler();
        // Submit work to the scheduler for parallel execution
        auto work = stdexec::starts_on(sched, bulk_sender);
        stdexec::sync_wait(std::move(work));
    }

    void findCliques() {
        cliques.clear(); // Clear previous cliques
        ankerl::unordered_dense::map<int, std::vector<std::pair<double, int>>> rowToElements;

        // Populate rowToElements for efficient row-wise access from SparseMatrix elements
        for (const auto &elem : modelData.A_sparse.elements) {
            int    row   = elem.row;
            int    col   = elem.col;
            double coeff = elem.value;

            rowToElements[row].emplace_back(coeff, col);
        }

        // Prepare tasks for parallel processing. Each task represents a row to process.
        std::vector<int> tasks(modelData.b.size());
        std::iota(tasks.begin(), tasks.end(), 0); // Populate tasks with row indices

        // Define chunk size for processing tasks in batches
        const int chunk_size = 100; // Adjust based on performance needs

        // Define a bulk sender to process tasks in parallel
        auto bulk_sender =
            stdexec::bulk(stdexec::just(), (tasks.size() + chunk_size - 1) / chunk_size,
                          [this, &tasks, &rowToElements, chunk_size](std::size_t chunk_idx) {
                              size_t start_idx = chunk_idx * chunk_size;
                              size_t end_idx   = std::min(start_idx + chunk_size, tasks.size());

                              // Process a chunk of tasks (i.e., rows)
                              for (size_t task_idx = start_idx; task_idx < end_idx; ++task_idx) {
                                  int    row = tasks[task_idx]; // Get the row index
                                  double rhs = modelData.b[row];

                                  std::vector<std::pair<double, int>> binCoeffs;
                                  binCoeffs.reserve(100); // Reserve space for binary coefficients

                                  auto &rowElements = rowToElements[row];

                                  // Adjust RHS and collect binary variable coefficients
                                  for (const auto &elem : rowElements) {
                                      double coeff = elem.first;
                                      int    col   = elem.second;

                                      if (modelData.vtype[col] != 'B') {
                                          rhs -= (coeff > 0 ? coeff * modelData.ub[col] : 0);
                                      } else {
                                          binCoeffs.emplace_back(coeff, col);
                                      }
                                  }

                                  if (binCoeffs.size() < 2) continue; // Skip rows with fewer than 2 binary coefficients

                                  std::sort(binCoeffs.begin(), binCoeffs.end());

                                  if (binCoeffs.back().first + binCoeffs[binCoeffs.size() - 2].first < rhs) continue;

                                  int k = findFirstClique(binCoeffs, rhs);
                                  if (k == -1) continue;

                                  addCliqueFromIndex(binCoeffs, k);

                                  processRemainingCliques(binCoeffs, k, rhs);
                              }
                          });

        exec::static_thread_pool pool(std::thread::hardware_concurrency());
        auto                     sched = pool.get_scheduler();
        // Submit work to the scheduler for parallel execution
        auto work = stdexec::starts_on(sched, bulk_sender);
        stdexec::sync_wait(std::move(work));
    }

    void initCg(int binary_number) {
        // Ensure that the conflict graph has the appropriate size based on the number of binary variables
        conflictGraph.clear(); // Clear any existing graph
        conflictGraph.resize(binary_number, std::vector<bool>(binary_number, false));
    }

    int findFirstClique(const std::vector<std::pair<double, int>> &binCoeffs, double rhs) {
        int left  = 0;
        int right = binCoeffs.size() - 2; // We are comparing two elements, hence size - 2.
        int k     = -1;

        // Binary search for the first index where the sum of binCoeffs[mid] and binCoeffs[mid + 1] > rhs
        while (left <= right) {
            int mid = left + (right - left) / 2;

            // Check if the sum of two consecutive coefficients exceeds rhs
            if (binCoeffs[mid].first + binCoeffs[mid + 1].first > rhs) {
                k     = mid;     // Found a valid clique start index
                right = mid - 1; // Narrow down the search to find the earliest such index
            } else {
                left = mid + 1; // Move the search to the right
            }
        }

        return k; // If no valid clique is found, k will remain -1.
    }

    void processRemainingCliques(const std::vector<std::pair<double, int>> &binCoeffs, int k, double rhs) {
        // Process cliques starting from earlier indices before k
        for (int o = k - 1; o >= 0; --o) {
            // Find the smallest index f such that binCoeffs[o] + binCoeffs[f] > rhs
            int left  = o + 1;
            int right = binCoeffs.size() - 1;
            int f     = -1;

            while (left <= right) {
                int mid = left + (right - left) / 2;

                // If the sum of the two coefficients exceeds rhs, try to find the smallest valid f
                if (binCoeffs[o].first + binCoeffs[mid].first > rhs) {
                    f     = mid;
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }

            // If a valid f was found, we can form another clique
            if (f != -1) {
                std::vector<int> clique;
                clique.push_back(binCoeffs[o].second); // Add the element at index o
                for (int j = f; j < binCoeffs.size(); ++j) {
                    clique.push_back(binCoeffs[j].second); // Add elements starting from index f
                }

                // Add this new clique (assuming addClique is a function that stores the clique)
                addClique(clique);
            }
        }
    }

    // Add cliques from index k
    void addCliqueFromIndex(const std::vector<std::pair<double, int>> &binCoeffs, int k) {
        std::vector<int> clique;
        clique.reserve(binCoeffs.size() - k);
        for (int j = k; j < binCoeffs.size(); ++j) { clique.push_back(binCoeffs[j].second); }
        addClique(clique);
    }

    // Function to print cliques
    void printCliques() {
        for (const auto &clique : cliques) {
            fmt::print("Clique: ");
            for (const auto &c : clique) { fmt::print("{} ", c); }
            fmt::print("\n");
        }
    }

    // Function to print the conflict graph
    void printCg() {
        for (size_t i = 0; i < conflictGraph.size(); ++i) {
            fmt::print("Variable {} conflicts with: ", i);
            for (size_t j = 0; j < conflictGraph[i].size(); ++j) {
                if (conflictGraph[i][j]) { fmt::print("{} ", j); }
            }
            fmt::print("\n");
        }
    }

    void printCg(std::map<int, std::string> varIndex2Name) {
        for (size_t i = 0; i < conflictGraph.size(); ++i) {
            fmt::print("Variable {} conflicts with: ", varIndex2Name[i]);
            for (size_t j = 0; j < conflictGraph[i].size(); ++j) {
                if (conflictGraph[i][j]) { fmt::print("{} ", varIndex2Name[j]); }
            }
            fmt::print("\n");
        }
    }

    void extendCliques(const std::vector<std::vector<int>> &initialCliques, std::vector<std::vector<int>> &extCliques) {
        // Iterate over each initial clique and attempt to extend it
        for (const auto &clique : initialCliques) {
            ankerl::unordered_dense::set<int> cliqueSet(clique.begin(),
                                                        clique.end()); // Store the clique for easy lookup
            std::vector<int>                  extendedClique = clique; // Start with the current clique

            bool extended = false; // Keep track if we successfully extend the clique

            // Create a list of candidate vertices (not in the clique) and sort them by degree
            std::vector<int> candidates;
            for (size_t v = 0; v < conflictGraph.size(); ++v) {
                if (cliqueSet.find(v) == cliqueSet.end()) { candidates.push_back(v); }
            }

            // Sort candidates by their degree (most connections to other vertices first)
            std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
                return std::count(conflictGraph[a].begin(), conflictGraph[a].end(), true) >
                       std::count(conflictGraph[b].begin(), conflictGraph[b].end(), true);
            });

            // Try to extend the clique with vertices sorted by their degree
            for (int v : candidates) {
                bool canExtend = true;
                for (int u : clique) {
                    if (!conflictGraph[v][u]) {
                        canExtend = false; // If v is not connected to any member of the clique, break out
                        break;
                    }
                }

                if (canExtend) {
                    extendedClique.push_back(v);
                    cliqueSet.insert(v); // Add v to the clique set
                    extended = true;
                }
            }

            // Add the extended clique to the result
            if (extended) {
                extCliques.push_back(extendedClique);
            } else {
                extCliques.push_back(clique); // If no extension was possible, keep the original clique
            }
        }
    }

    auto getCliques() const -> const std::vector<std::vector<int>> & { return cliques; }
};
