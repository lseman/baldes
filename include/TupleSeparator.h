#pragma once

#include "config.h"

#include "Path.h"
#include <algorithm>
#include <cmath>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexec/execution.hpp>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <vector>

constexpr double violation_tolerance = 0.02;

class TupleBasedSeparator {
public:
    using RowPair = std::pair<int, int>;
    using Rowset  = std::set<int>;

    TupleBasedSeparator(const std::vector<Path> &routes) : routes(routes) {
        std::unordered_set<int> rows;
        for (int i = 0; i < N_SIZE; ++i) { rows.insert(i); }
        rowPairs = generateAllRowPairs(rows);
    }
    int                 chunk_size = 10;
    std::vector<Rowset> separate4R1Cs();

private:
    std::vector<Path>                routes;          // LP solution routes λ*
    std::vector<RowPair>             rowPairs;        // All row pairs P = {(r1, r2) ∈ V^2 | r1 < r2}
    std::set<Rowset>                 exploredRowsets; // Set of explored rowsets C
    std::vector<std::vector<double>> multiplierSets;  // Set of multipliers for 4-row R1Cs

    // Generates all unique pairs of rows (r1, r2) where r1 < r2
    std::vector<RowPair> generateAllRowPairs(const std::unordered_set<int> &rows) {
        std::vector<RowPair> pairs;
        for (auto it1 = rows.begin(); it1 != rows.end(); ++it1) {
            for (auto it2 = std::next(it1); it2 != rows.end(); ++it2) { pairs.emplace_back(*it1, *it2); }
        }
        return pairs;
    }

    // Combines two pairs of rows to form a rowset C with four unique rows
    Rowset combineRowPairs(const RowPair &p1, const RowPair &p2) {
        Rowset C = {p1.first, p1.second, p2.first, p2.second};
        return C;
    }

    // Filter Ω* to get Ω*2 - routes visiting at least two rows in C
    std::vector<Path> getOmega2(const Rowset &C) {
        std::vector<Path> omega2;
        for (const auto &route : routes) {
            int count = 0;
            for (int node : route.route) {
                if (C.count(node)) {
                    count++;
                    if (count >= 2) {
                        omega2.push_back(route);
                        break;
                    }
                }
            }
        }
        return omega2;
    }

    // Computes the violation for a given rowset C, multiplier m, and routes in Ω*2
    double computeViolation(const Rowset &C, const std::vector<double> &multipliers, const std::vector<Path> &omega2) {
        double violation = 0.0;

        for (const auto &route : omega2) {
            double route_contribution = 0.0;
            int    multiplierIndex    = 0;

            for (int node : route.route) {
                if (C.count(node)) { route_contribution += multipliers[multiplierIndex++] * route.frac_x; }
            }

            violation += std::floor(route_contribution);
        }

        double rhs = std::accumulate(multipliers.begin(), multipliers.end(), 0.0);
        return violation - rhs;
    }

    // Remove duplicate cuts from the result
    std::vector<Rowset> removeDuplicateCuts(const std::vector<Rowset> &cuts) {
        std::set<Rowset> uniqueCuts(cuts.begin(), cuts.end());
        return std::vector<Rowset>(uniqueCuts.begin(), uniqueCuts.end());
    }
};
