/**
 * @file Cut.h
 * @brief This file contains the definition of the Cut struct and CutStorage class.
 *
 * This file contains the definition of the Cut struct, which represents a cut in the optimization problem.
 * The Cut struct holds information about a specific cut, including its base set, neighbors, coefficients, multipliers,
 * and other properties. The CutStorage class manages the storage and operations related to cuts in a solver.
 *
 */
#pragma once
#include "Common.h"
#include "Definitions.h"
#include "MIPHandler/Constraint.h"

/**
 * @struct Cut
 * @brief Represents a cut in the optimization problem.
 *
 * The Cut structure holds information about a specific cut, including its base set,
 * neighbors, coefficients, multipliers, and other properties.
 *
 */
struct Cut {
    int                             cutMaster;
    std::array<uint64_t, num_words> baseSet;      // Bit-level baseSet
    std::array<uint64_t, num_words> neighbors;    // Bit-level neighbors
    std::vector<int>                baseSetOrder; // Order for baseSet
    std::vector<double>             coefficients; // Cut coefficients
    std::vector<double>             multipliers = {0.5, 0.5, 0.5};
    double                          rhs         = 1;
    int                             id          = -1;
    bool                            added       = false;
    bool                            updated     = false;
    CutType                         type        = CutType::ThreeRow;
    Constraint                       grbConstr;
    size_t                          key;
    // Default constructor
    Cut() = default;

    // constructor to receive array
    Cut(const std::array<uint64_t, num_words> baseSetInput, const std::array<uint64_t, num_words> &neighborsInput,
        const std::vector<double> &coefficients)
        : baseSet(baseSetInput), neighbors(neighborsInput), coefficients(coefficients) {}

    Cut(const std::array<uint64_t, num_words> baseSetInput, const std::array<uint64_t, num_words> &neighborsInput,
        const std::vector<double> &coefficients, const std::vector<double> &multipliers)
        : baseSet(baseSetInput), neighbors(neighborsInput), coefficients(coefficients), multipliers(multipliers) {}

    // Define size of the cut
    size_t size() const { return coefficients.size(); }
};

using Cuts = std::vector<Cut>;

/**
 * @class CutStorage
 * @brief Manages the storage and operations related to cuts in a solver.
 *
 * The CutStorage class provides functionalities to add, manage, and query cuts.
 * It also allows setting dual values and computing coefficients with limited memory.
 *
 */
class CutStorage {
public:
    int latest_column = 0;

    Cut &operator[](std::size_t index) { return cuts[index]; }

    // Add a cut to the storage
    /**
     * @brief Adds a cut to the current collection of cuts.
     *
     * This function takes a reference to a Cut object and adds it to the
     * collection of cuts maintained by the solver. The cut is used to
     * refine the solution space and improve the efficiency of the solver.
     *
     */
    void addCut(Cut &cut);

    void reset() {
        cuts.clear();
        cutMaster_to_cut_map.clear();
        indexCuts.clear();
        SRCDuals = {};
    }

    Cut &getCut(int cutIndex) { return cuts[cutIndex]; }

    int getID(int cutIndex) { return cuts[cutIndex].id; }

    void removeCut(int cutIndex) {
        // Ensure the cutIndex is within bounds
        if (cutIndex < 0 || cutIndex >= cuts.size()) {
            std::cerr << "Cut index " << cutIndex << " is out of bounds." << std::endl;
            return;
        }

        // Erase the cut from the cuts vector
        cuts.erase(cuts.begin() + cutIndex);

        for (auto &entry : cuts) {
            if (entry.id > cutIndex) {
                entry.id--; // Decrement index to account for the removed cut
            }
        }

        // Find the corresponding entry in the map
        auto it = std::find_if(cutMaster_to_cut_map.begin(), cutMaster_to_cut_map.end(),
                               [cutIndex](const auto &pair) { return pair.second == cutIndex; });

        if (it != cutMaster_to_cut_map.end()) {
            // Remove the entry for the removed cut
            cutMaster_to_cut_map.erase(it);
        } else {
            std::cerr << "Cut index " << cutIndex << " not found in cutMaster_to_cut_map." << std::endl;
            return; // No need to continue if the cut wasn't found in the map
        }

        // Reorganize the indices in the map for all subsequent cuts
        for (auto &entry : cutMaster_to_cut_map) {
            if (entry.second > cutIndex) {
                entry.second--; // Decrement index to account for the removed cut
            }
        }
    }

    std::vector<double> SRCDuals = {};

    /**
     * @brief Sets the dual values for the SRC.
     *
     * This function assigns the provided vector of dual values to the SRCDuals member.
     *
     */
    void setDuals(const std::vector<double> &duals) { SRCDuals = duals; }

    // Define size method
    size_t size() noexcept { return cuts.size(); }

    // Define begin and end
    auto begin() const noexcept { return cuts.begin(); }
    auto end() const noexcept { return cuts.end(); }
    auto begin() noexcept { return cuts.begin(); }
    auto end() noexcept { return cuts.end(); }

    // Define empty method
    bool empty() const noexcept { return cuts.empty(); }

    /**
     * @brief Checks if a cut exists for the given cut key and returns its size and coefficients.
     *
     * This function searches for the specified cut key in the cutMaster_to_cut_map. If the cut key
     * is found, it retrieves the size and coefficients of the corresponding cut from the cuts vector.
     * If the cut key is not found, it returns a pair with -1 and an empty vector.
     *
     */
    std::pair<int, std::vector<double>> cutExists(const std::size_t &cut_key) const {
        auto it = cutMaster_to_cut_map.find(cut_key);
        if (it != cutMaster_to_cut_map.end()) {
            auto tam    = cuts[it->second].size();
            auto coeffs = cuts[it->second].coefficients;
            return {tam, coeffs};
        }
        return {-1, {}};
    }

    /**
     * @brief Retrieves the constraint at the specified index.
     *
     * This function returns the Gurobi constraint object associated with the
     * cut at the given index.
     *
     */
    auto getCtr(int i) const { return cuts[i].grbConstr; }

    /**
     * @brief Computes limited memory coefficients for a given set of cuts.
     *
     * This function iterates over a collection of cuts and computes a set of coefficients
     * based on the provided vector P. The computation involves checking membership of nodes
     * in specific sets and updating coefficients accordingly.
     *
     */
    auto computeLimitedMemoryCoefficients(const std::vector<int> &P) {
        // iterate over cuts
        std::vector<double> alphas;
        alphas.reserve(cuts.size());
        for (auto c : cuts) {
            double alpha = 0.0;
            double S     = 0;
            auto   AM    = c.neighbors;
            auto   C     = c.baseSet;
            auto   p     = c.multipliers;
            auto   order = c.baseSetOrder;

            for (size_t j = 1; j < P.size() - 1; ++j) {
                int vj = P[j];

                // Check if the node vj is in AM (bitwise check)
                if (!(AM[vj / 64] & (1ULL << (vj % 64)))) {
                    S = 0; // Reset S if vj is not in AM
                }
                if (C[vj / 64] & (1ULL << (vj % 64))) {
                    // Get the position of vj in C by counting the set bits up to vj
                    int pos = order[vj];
                    S += p[pos];
                    if (S >= 1) {
                        S -= 1;
                        alpha += 1;
                    }
                }
            }

            alphas.push_back(alpha);
        }
        return alphas;
    }

    std::size_t generateCutKey(const int &cutMaster, const std::vector<bool> &baseSetStr) const;

private:
    ankerl::unordered_dense::map<std::size_t, int>              cutMaster_to_cut_map;
    Cuts                                                        cuts;
    ankerl::unordered_dense::map<std::size_t, std::vector<int>> indexCuts;
};
