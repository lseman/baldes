/**
 * @file Cut.h
 * @brief This file contains the definition of the Cut struct and CutStorage
 * class.
 *
 * This file contains the definition of the Cut struct, which represents a cut
 * in the optimization problem. The Cut struct holds information about a
 * specific cut, including its base set, neighbors, coefficients, multipliers,
 * and other properties. The CutStorage class manages the storage and operations
 * related to cuts in a solver.
 *
 */
#pragma once
#include "Common.h"
#include "Definitions.h"
#include "miphandler/Constraint.h"

struct SRCPermutation {
    std::vector<int> num;
    std::vector<double> frac;
    int den;
    // Default constructor
    SRCPermutation() = default;

    // Constructor with two vectors
    SRCPermutation(std::vector<int> num, int den) : num(num), den(den) {}

    SRCPermutation(std::vector<double> frac) : frac(frac) {}
    // Function to compute RHS
    double getRHS() const {
        double rhs = 0;
        for (size_t i = 0; i < num.size(); ++i) {
            rhs += static_cast<double>(num[i]) / static_cast<double>(den);
        }
        return std::floor(rhs);
    }

    // Begin and end (non-const to allow modification)
    auto begin() noexcept { return num.begin(); }
    auto end() noexcept { return num.end(); }

    // Swap function for SRCPermutation
    // Swap function for SRCPermutation
    void swap(SRCPermutation &other) {
        // Swap each corresponding element in the num and den vectors
        for (size_t i = 0; i < num.size(); ++i) {
            std::swap(num[i], other.num[i]);
        }
        std::swap(den, other.den);
        // std::swap(frac, other.frac);
    }

    // Support next_permutation for the 'num' vector
    bool next_permutation() {
        return std::next_permutation(num.begin(), num.end());
    }
};

/**
 * @struct Cut
 * @brief Represents a cut in the optimization problem.
 *
 * The Cut structure holds information about a specific cut, including its base
 * set, neighbors, coefficients, multipliers, and other properties.
 *
 */
struct Cut {
    int cutMaster;
    std::array<uint64_t, num_words> baseSet;    // Bit-level baseSet
    std::array<uint64_t, num_words> neighbors;  // Bit-level neighbors
    std::vector<int> baseSetOrder;              // Order for baseSet
    std::vector<double> coefficients;           // Cut coefficients
    SRCPermutation p = {{1, 1, 1, 1}, 2};
    // SRCPermutation p = {{1.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0}};

    void printCut() const {
        fmt::print("BaseSet: ");
        // print the bit position where the bit is set
        for (int i = 0; i < N_SIZE; ++i) {
            if (baseSet[i / 64] & (1ULL << (i % 64))) {
                fmt::print("{} ", i);
            }
        }
        fmt::print("\n");
        fmt::print("Neighbors: ");
        for (int i = 0; i < N_SIZE; ++i) {
            if (neighbors[i / 64] & (1ULL << (i % 64))) {
                fmt::print("{} ", i);
            }
        }
        fmt::print("\n");
        // print the p.num and p.den
        fmt::print("Permutation: ");
        for (auto &i : p.num) {
            fmt::print("{} ", i);
        }
        fmt::print("| {}\n", p.den);
    }
    double rhs = 1;
    int id = -1;
    bool added = false;
    bool updated = false;
    CutType type = CutType::ThreeRow;
    baldesCtrPtr grbConstr;
    size_t key;
    // Default constructor
    Cut() = default;

    // constructor to receive array
    Cut(const std::array<uint64_t, num_words> baseSetInput,
        const std::array<uint64_t, num_words> &neighborsInput,
        const std::vector<double> &coefficients)
        : baseSet(baseSetInput),
          neighbors(neighborsInput),
          coefficients(coefficients) {
        rhs = p.getRHS();
    }

    Cut(const std::array<uint64_t, num_words> baseSetInput,
        const std::array<uint64_t, num_words> &neighborsInput,
        const std::vector<double> &coefficients,
        const SRCPermutation &multipliers)
        : baseSet(baseSetInput),
          neighbors(neighborsInput),
          coefficients(coefficients),
          p(multipliers) {
        rhs = p.getRHS();
    }

    // Define size of the cut
    size_t size() const { return coefficients.size(); }
};

using Cuts = std::vector<Cut>;

/**
 * @class CutStorage
 * @brief Manages the storage and operations related to cuts in a solver.
 *
 * The CutStorage class provides functionalities to add, manage, and query cuts.
 * It also allows setting dual values and computing coefficients with limited
 * memory.
 *
 */
class CutStorage {
   public:
    int latest_column = 0;

    struct ActiveCutInfo {
        size_t index;        // Original index in cuts vector
        const Cut *cut_ptr;  // Pointer to the cut
        double dual_value;   // Cached dual value
        CutType type;        // Type of the cut

        ActiveCutInfo(size_t idx, const Cut *ptr, double dual, CutType type)
            : index(idx), cut_ptr(ptr), dual_value(dual), type(type) {}

        bool isSRCset(int i, int j) const {
            return cut_ptr->baseSet[i / 64] & (1ULL << (i % 64)) &&
                   cut_ptr->baseSet[j / 64] & (1ULL << (j % 64));
        }
        bool isSRCset(int i) const {
            return cut_ptr->baseSet[i / 64] & (1ULL << (i % 64));
        }
    };

    struct SegmentMasks {
        constexpr static size_t n_segments = (N_SIZE + 63) / 64;
        constexpr static size_t n_bit_positions = 64;
        std::vector<std::vector<uint64_t>>
            neighbor_masks;  // [segment][bit_position]
        std::vector<std::vector<uint64_t>>
            base_set_masks;       // [segment][bit_position]
        size_t n_cuts;            // Keep track of number of cuts for safety
        uint64_t cut_limit_mask;  // Mask to limit the number of cuts

        std::vector<std::vector<uint64_t>>
            valid_cut_masks;  // [segment][bit_position] - NEW!

        SegmentMasks() : n_cuts(0) {
            neighbor_masks.resize(n_segments);
            base_set_masks.resize(n_segments);
            valid_cut_masks.resize(n_segments);
            for (size_t i = 0; i < n_segments; ++i) {
                neighbor_masks[i].resize(n_bit_positions, 0);
                base_set_masks[i].resize(n_bit_positions, 0);
                valid_cut_masks[i].resize(n_bit_positions, 0);
            }
        }
        void precompute(const std::vector<ActiveCutInfo> &active_cuts) {
            n_cuts = active_cuts.size();
            cut_limit_mask = (n_cuts >= 64) ? ~0ULL : ((1ULL << n_cuts) - 1);

            // Precompute bit masks once
            std::vector<uint64_t> cut_bit_masks(n_cuts);
            for (size_t i = 0; i < n_cuts; ++i) {
                cut_bit_masks[i] = 1ULL << i;
            }

            // Resize and zero-initialize
            neighbor_masks.assign(n_segments,
                                  std::vector<uint64_t>(n_bit_positions, 0));
            base_set_masks.assign(n_segments,
                                  std::vector<uint64_t>(n_bit_positions, 0));
            valid_cut_masks.assign(n_segments,
                                   std::vector<uint64_t>(n_bit_positions, 0));

            // Iterate over segments and bit positions
            for (size_t segment = 0; segment < n_segments; ++segment) {
                size_t bit_limit =
                    (segment == n_segments - 1) ? (N_SIZE % 64) : 64;

                for (size_t bit_pos = 0; bit_pos < bit_limit; ++bit_pos) {
                    const uint64_t bit_mask = 1ULL << bit_pos;
                    uint64_t neighbor_bits = 0;
                    uint64_t base_bits = 0;

                    for (size_t cut_idx = 0; cut_idx < n_cuts; ++cut_idx) {
                        const auto &cut = *active_cuts[cut_idx].cut_ptr;

                        if (cut.neighbors[segment] & bit_mask) {
                            neighbor_bits |= cut_bit_masks[cut_idx];
                        }
                        if (cut.baseSet[segment] & bit_mask) {
                            base_bits |= cut_bit_masks[cut_idx];
                        }
                    }

                    // Store with cut limit applied
                    neighbor_masks[segment][bit_pos] =
                        (neighbor_bits & cut_limit_mask);
                    base_set_masks[segment][bit_pos] =
                        (base_bits & cut_limit_mask);
                    valid_cut_masks[segment][bit_pos] =
                        (neighbor_bits & base_bits & cut_limit_mask) &
                        cut_limit_mask;
                }
            }
        }

        // Safe accessors
        uint64_t get_neighbor_mask(size_t segment, size_t bit_pos) const {
            if (segment >= n_segments || bit_pos >= n_bit_positions) return 0;
            return neighbor_masks[segment][bit_pos];
        }

        uint64_t get_base_mask(size_t segment, size_t bit_pos) const {
            if (segment >= n_segments || bit_pos >= n_bit_positions) return 0;
            return base_set_masks[segment][bit_pos];
        }

        uint64_t get_valid_cut_mask(size_t segment, size_t bit_pos) const {
            if (segment >= n_segments || bit_pos >= n_bit_positions) return 0;
            return valid_cut_masks[segment][bit_pos];
        }

        uint64_t get_cut_limit_mask() { return cut_limit_mask; }
    };

    SegmentMasks segment_masks;

    auto &getSegmentMasks() { return segment_masks; }

    // Access a cut by index
    Cut &operator[](std::size_t index) {
        assert(index < cuts.size());
        return cuts[index];
    }

    // Add a cut to the storage
    void addCut(Cut &cut);

    // Reset the storage
    void reset() {
        cuts.clear();
        cutMaster_to_cut_map.clear();
        indexCuts.clear();
        SRCDuals.clear();
    }

    // Get a cut by index
    Cut &getCut(int cutIndex) { return cuts[cutIndex]; }

    Cut &get_cut(int cutIndex) { return cuts[cutIndex]; }

    std::span<const ActiveCutInfo> getActiveCuts() const noexcept {
        return std::span<const ActiveCutInfo>(active_cuts);
    }

    const Cut &getCut(int cutIndex) const { return cuts[cutIndex]; }

    // Get the ID of a cut
    int getID(int cutIndex) const { return cuts[cutIndex].id; }

    // Print all cuts
    void printCuts() const {
        for (const auto &cut : cuts) {
            cut.printCut();
        }
    }

    // Remove a cut by index
    void removeCut(int cutIndex) {
        if (cutIndex < 0 || cutIndex >= cuts.size()) {
            std::cerr << "Cut index " << cutIndex << " is out of bounds."
                      << std::endl;
            return;
        }

        // Erase the cut from the cuts vector
        cuts.erase(cuts.begin() + cutIndex);

        // Update IDs for remaining cuts
        for (auto &cut : cuts) {
            if (cut.id > cutIndex) {
                cut.id--;
            }
        }

        // Remove the corresponding entry from the map
        auto it = std::find_if(
            cutMaster_to_cut_map.begin(), cutMaster_to_cut_map.end(),
            [cutIndex](const auto &pair) { return pair.second == cutIndex; });

        if (it != cutMaster_to_cut_map.end()) {
            cutMaster_to_cut_map.erase(it);
        } else {
            std::cerr << "Cut index " << cutIndex
                      << " not found in cutMaster_to_cut_map." << std::endl;
            return;
        }

        // Update indices in the map for all subsequent cuts
        for (auto &entry : cutMaster_to_cut_map) {
            if (entry.second > cutIndex) {
                entry.second--;
            }
        }
    }

    // Set dual values for SRC
    void setDuals(const std::vector<double> &duals) {
        SRCDuals = duals;
        updateActiveCuts();
    }

    // Get the number of cuts
    size_t size() const noexcept { return cuts.size(); }
    size_t activeSize() const noexcept { return active_cuts.size(); }

    // Iterators
    auto begin() const noexcept { return cuts.begin(); }
    auto end() const noexcept { return cuts.end(); }
    auto begin() noexcept { return cuts.begin(); }
    auto end() noexcept { return cuts.end(); }

    // Check if the storage is empty
    bool empty() const noexcept { return cuts.empty(); }

    // Check if a cut exists for the given cut key
    std::pair<int, std::vector<double>> cutExists(
        const std::size_t &cut_key) const {
        auto it = cutMaster_to_cut_map.find(cut_key);
        if (it != cutMaster_to_cut_map.end()) {
            auto tam = cuts[it->second].size();
            auto coeffs = cuts[it->second].coefficients;
            return {tam, coeffs};
        }
        return {-1, {}};
    }

    // Get the constraint at the specified index
    auto getCtr(int i) const {
        assert(i >= 0 && i < cuts.size());
        return cuts[i].grbConstr;
    }

    // Compute limited memory coefficients for a given set of cuts
    std::vector<double> computeLimitedMemoryCoefficients(
        const std::vector<uint16_t> &P) const {
        std::vector<double> alphas;
        alphas.reserve(cuts.size());

        for (const auto &c : cuts) {
            double alpha = 0.0;
            int S = 0;
            auto den = c.p.den;
            auto AM = c.neighbors;
            auto C = c.baseSet;
            auto p = c.p;
            auto order = c.baseSetOrder;

            for (size_t j = 1; j < P.size() - 1; ++j) {
                int vj = P[j];

                // Check if the node vj is in AM (bitwise check)
                if (!(AM[vj / 64] & (1ULL << (vj % 64)))) {
                    S = 0;  // Reset S if vj is not in AM
                } else if (C[vj / 64] & (1ULL << (vj % 64))) {
                    // Get the position of vj in C by counting the set
                    // bits up to vj
                    int pos = order[vj];
                    S += p.num[pos];
                    if (S >= den) {
                        S -= den;
                        alpha += 1;
                    }
                }
            }

            alphas.push_back(alpha);
        }

        return alphas;
    }

    // Generate a cut key
    std::size_t generateCutKey(const int &cutMaster,
                               const std::vector<bool> &baseSetStr) const {
        std::size_t key = cutMaster;
        for (bool b : baseSetStr) {
            key = (key << 1) | b;
        }
        return key;
    }
    std::vector<double> SRCDuals;  // Dual values for SRC

   private:
    std::vector<Cut> cuts;  // Storage for cuts
    ankerl::unordered_dense::map<std::size_t, int>
        cutMaster_to_cut_map;  // Map from cut key to cut index
    ankerl::unordered_dense::map<std::size_t, std::vector<int>>
        indexCuts;  // Additional index for cuts (if needed)
    std::vector<ActiveCutInfo> active_cuts;  // Cuts with positive duals

    void updateActiveCuts() {
        active_cuts.clear();
        active_cuts.reserve(cuts.size());  // Pre-allocate for efficiency

        for (size_t i = 0; i < cuts.size(); ++i) {
            if (i < SRCDuals.size() && std::abs(SRCDuals[i]) > 1e-2) {
                active_cuts.emplace_back(i, &cuts[i], SRCDuals[i],
                                         cuts[i].type);
            }
        }

        // Pre-compute bitmasks for all cuts in the all the segments
        segment_masks.precompute(active_cuts);
    }
};
