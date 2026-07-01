/**
 * @file Cut.h
 * @brief Defines cut structures and storage for solver separation.
 *
 */
#pragma once
#include "Common.h"
#include "Definitions.h"
#include "Label.h"
#include "Path.h"
#include "Serializer.h"
#include "miphandler/Constraint.h"
struct SRCPermutation {
    std::vector<int>    num;
    std::vector<double> frac;
    int                 den;

    REFLECT(num, den, frac);
    // Default constructor
    SRCPermutation() = default;

    // Constructor with two vectors
    SRCPermutation(std::vector<int> num, int den) : num(num), den(den) {}

    SRCPermutation(std::vector<double> frac) : frac(frac) {}
    // Function to compute RHS
    double getRHS() const {
        double rhs = 0;
        for (size_t i = 0; i < num.size(); ++i) { rhs += static_cast<double>(num[i]) / static_cast<double>(den); }
        return std::floor(rhs);
    }

    // Begin and end (non-const to allow modification)
    auto begin() noexcept { return num.begin(); }
    auto end() noexcept { return num.end(); }

    // Swap function for SRCPermutation
    // Swap function for SRCPermutation
    void swap(SRCPermutation &other) {
        // Swap each corresponding element in the num and den vectors
        for (size_t i = 0; i < num.size(); ++i) { std::swap(num[i], other.num[i]); }
        std::swap(den, other.den);
        // std::swap(frac, other.frac);
    }

    // Support next_permutation for the 'num' vector
    bool next_permutation() { return std::next_permutation(num.begin(), num.end()); }
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
    int                             cutMaster;
    std::array<uint64_t, num_words> baseSet;             // Bit-level baseSet
    std::array<uint64_t, num_words> neighbors;           // Bit-level neighbors
    std::vector<int>                baseSetOrder;        // Order for baseSet
    std::vector<double>             coefficients;        // Cut coefficients
    std::vector<int>                coefficient_indices; // Sparse coefficient indices
    std::vector<double>             coefficient_values;  // Sparse coefficient values
    SRCPermutation                  p = {{1, 1, 1, 1}, 2};
    // SRCPermutation p = {{1.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0}};

    void printCut() const {
        fmt::print("BaseSet: ");
        // print the bit position where the bit is set
        for (int i = 0; i < N_SIZE; ++i) {
            if (baseSet[i / 64] & (1ULL << (i % 64))) { fmt::print("{} ", i); }
        }
        fmt::print("\n");
        fmt::print("Neighbors: ");
        for (int i = 0; i < N_SIZE; ++i) {
            if (neighbors[i / 64] & (1ULL << (i % 64))) { fmt::print("{} ", i); }
        }
        fmt::print("\n");
        // print the p.num and p.den
        fmt::print("Permutation: ");
        for (auto &i : p.num) { fmt::print("{} ", i); }
        fmt::print("| {}\n", p.den);
    }

    bool isSRCset(int i, int j) const {
        const uint64_t mask_i = 1ULL << (i % 64);
        const uint64_t mask_j = 1ULL << (j % 64);
        return (baseSet[i / 64] & mask_i) && (baseSet[j / 64] & mask_j) && (neighbors[i / 64] & mask_i) &&
               (neighbors[j / 64] & mask_j);
    }

    bool isSRCset(int i) const { return baseSet[i / 64] & (1ULL << (i % 64)); }
    bool isSRCMemoryNode(int i) const { return neighbors[i / 64] & (1ULL << (i % 64)); }
    bool isSRCMemoryArc(int i, int j) const { return i >= 0 && j >= 0 && isSRCMemoryNode(i) && isSRCMemoryNode(j); }
    int  srcMultiplier(int node) const { return p.num[baseSetOrder[node]]; }

    double rhs = 1;
    int    id  = -1;

    // Cut pool management metadata
    int64_t separation_count        = 0; // Times this cut was generated/separated
    int64_t inclusion_count         = 0; // Times this cut was selected into active set
    int64_t active_count            = 0; // Pricing rounds with a nonzero dual
    int64_t creation_epoch          = 0; // B&B node epoch when cut was created
    int64_t last_separation_epoch   = 0; // Last epoch where separation found this cut
    int64_t last_active_epoch       = 0; // Last epoch where the cut was pricing-active
    int64_t last_nonzero_dual_epoch = 0; // Last epoch with nonzero dual
    double  last_violation          = 0; // Most recent violation observed at separation
    double  max_violation           = 0; // Peak violation observed at separation
    double  total_violation         = 0; // Sum of observed violations
    double  avg_dual_magnitude      = 0; // EMA of absolute dual values

    REFLECT(baseSet, neighbors, coefficients, p, baseSetOrder, id)

    bool         added   = false;
    bool         updated = false;
    CutType      type    = CutType::ThreeRow;
    baldesCtrPtr grbConstr;
    size_t       key;
    // Default constructor
    Cut() = default;

    // constructor to receive array
    Cut(const std::array<uint64_t, num_words> baseSetInput, const std::array<uint64_t, num_words> &neighborsInput,
        const std::vector<double> &coefficients)
        : baseSet(baseSetInput), neighbors(neighborsInput), coefficients(coefficients) {
        rhs = p.getRHS();
    }

    Cut(const std::array<uint64_t, num_words> baseSetInput, const std::array<uint64_t, num_words> &neighborsInput,
        const std::vector<double> &coefficients, const SRCPermutation &multipliers)
        : baseSet(baseSetInput), neighbors(neighborsInput), coefficients(coefficients), p(multipliers) {
        rhs = p.getRHS();
    }

    // Define size of the cut
    size_t size() const { return coefficients.size(); }
    bool   hasSparseCoefficients() const noexcept {
        return !coefficient_indices.empty() && coefficient_indices.size() == coefficient_values.size();
    }
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

struct ActiveCutInfo {
    size_t     index;      // Compact pricing-active index
    size_t     cut_index;  // Original index in cuts vector
    const Cut *cut_ptr;    // Pointer to the cut
    double     dual_value; // Cached dual value
    CutType    type;       // Type of the cut

    ActiveCutInfo(size_t active_idx, size_t cut_idx, const Cut *ptr, double dual, CutType type)
        : index(active_idx), cut_index(cut_idx), cut_ptr(ptr), dual_value(dual), type(type) {}

    bool isSRCset(int i, int j) const {
        const uint64_t mask_i = 1ULL << (i % 64);
        const uint64_t mask_j = 1ULL << (j % 64);
        return (cut_ptr->baseSet[i / 64] & mask_i) && (cut_ptr->baseSet[j / 64] & mask_j) &&
               (cut_ptr->neighbors[i / 64] & mask_i) && (cut_ptr->neighbors[j / 64] & mask_j);
    }
    bool isSRCset(int i) const { return cut_ptr->baseSet[i / 64] & (1ULL << (i % 64)); };
    bool isSRCMemoryNode(int i) const { return cut_ptr->neighbors[i / 64] & (1ULL << (i % 64)); }
    bool isSRCMemoryArc(int i, int j) const { return i >= 0 && j >= 0 && isSRCMemoryNode(i) && isSRCMemoryNode(j); }
};
class CutStorage {
public:
    int                     latest_column      = 0;
    bool                    busy               = false;
    static constexpr double SRC_DUAL_TOLERANCE = 1e-6;

    struct SRCNodeUpdate {
        uint16_t active_idx;
        uint16_t add;
        uint16_t den;
        double   dual;
    };

    struct SegmentMasks {
        constexpr static size_t            n_segments      = (N_SIZE + 63) / 64;
        constexpr static size_t            n_bit_positions = 64;
        std::vector<std::vector<uint64_t>> neighbor_masks; // [segment][bit_position]
        std::vector<std::vector<uint64_t>> base_set_masks; // [segment][bit_position]
        size_t                             n_cuts;         // Keep track of number of cuts for safety
        uint64_t                           cut_limit_mask; // Mask to limit the number of cuts

        std::vector<std::vector<uint64_t>> valid_cut_masks; // [segment][bit_position] - NEW!

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
            // Determine the number of cuts and compute the overall cut limit
            // mask.
            n_cuts         = active_cuts.size();
            cut_limit_mask = (n_cuts >= 64) ? ~0ULL : ((1ULL << n_cuts) - 1);

            // Precompute individual bit masks for each cut.
            std::vector<uint64_t> cut_bit_masks(n_cuts);
            for (size_t i = 0; i < n_cuts; ++i) { cut_bit_masks[i] = 1ULL << i; }

            // Resize and initialize mask matrices for all segments and bit
            // positions.
            neighbor_masks.assign(n_segments, std::vector<uint64_t>(n_bit_positions, 0));
            base_set_masks.assign(n_segments, std::vector<uint64_t>(n_bit_positions, 0));
            valid_cut_masks.assign(n_segments, std::vector<uint64_t>(n_bit_positions, 0));

            // Iterate over each segment.
            for (size_t segment = 0; segment < n_segments; ++segment) {
                // Determine the number of bits to process in this segment.
                const size_t bit_limit = (segment == n_segments - 1 && (N_SIZE % 64) != 0) ? (N_SIZE % 64) : 64;

                // Cache references to the current segment's mask vectors.
                auto &nbr_mask_vec   = neighbor_masks[segment];
                auto &base_mask_vec  = base_set_masks[segment];
                auto &valid_mask_vec = valid_cut_masks[segment];

                // Process each bit position within the segment.
                for (size_t bit_pos = 0; bit_pos < bit_limit; ++bit_pos) {
                    const uint64_t bit_mask      = 1ULL << bit_pos;
                    uint64_t       neighbor_bits = 0;
                    uint64_t       base_bits     = 0;

                    // Loop over each active cut.
                    for (size_t cut_idx = 0; cut_idx < n_cuts; ++cut_idx) {
                        // Cache pointer to current cut.
                        const auto &cut = *active_cuts[cut_idx].cut_ptr;
                        if (cut.neighbors[segment] & bit_mask) neighbor_bits |= cut_bit_masks[cut_idx];
                        if (cut.baseSet[segment] & bit_mask) base_bits |= cut_bit_masks[cut_idx];
                    }

                    // Apply the cut limit mask and store computed values.
                    nbr_mask_vec[bit_pos]   = neighbor_bits & cut_limit_mask;
                    base_mask_vec[bit_pos]  = base_bits & cut_limit_mask;
                    valid_mask_vec[bit_pos] = (neighbor_bits & base_bits) & cut_limit_mask;
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

    const auto &getSRCNodeUpdates(int node_id) const noexcept { return src_node_updates[node_id]; }
    const auto &getSRCNodeClears(int node_id) const noexcept { return src_node_clears[node_id]; }

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
        active_cuts.clear();
        segment_masks.precompute(active_cuts);
        precomputeSRCNodeTables();
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
        for (const auto &cut : cuts) { cut.printCut(); }
    }

    void removeCut(Cut *cut) {
        if (!cut) return;
        removeCut(cut->id);
    }

    // Remove a cut by index
    void removeCut(int cutIndex) {
        // Validate cutIndex.
        if (cutIndex < 0 || cutIndex >= static_cast<int>(cuts.size())) {
            std::cerr << "Cut index " << cutIndex << " is out of bounds." << std::endl;
            return;
        }

        // Remove the cut from the 'cuts' vector.
        cuts.erase(cuts.begin() + cutIndex);

        // Remove the associated dual if it exists.
        if (cutIndex < static_cast<int>(SRCDuals.size())) { SRCDuals.erase(SRCDuals.begin() + cutIndex); }

        cutMaster_to_cut_map.clear();
        indexCuts.clear();
        for (int idx = 0; idx < static_cast<int>(cuts.size()); ++idx) {
            cuts[idx].id                        = idx;
            cutMaster_to_cut_map[cuts[idx].key] = idx;
            indexCuts[cuts[idx].key].push_back(idx);
        }
        updateActiveCuts();
    }

    // Set dual values for SRC
    void setDuals(const std::vector<double> &duals) {
        SRCDuals = duals;
        updateActiveCuts();
    }

    void pruneLowDualCuts(double threshold) {
        if (cuts.empty() || SRCDuals.empty()) return;
        bool removed = false;
        for (int i = static_cast<int>(cuts.size()) - 1; i >= 0; --i) {
            if (i < static_cast<int>(SRCDuals.size()) && std::abs(SRCDuals[i]) < threshold) {
                removeCut(i);
                removed = true;
            }
        }
        if (removed) updateActiveCuts();
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
    std::pair<int, std::vector<double>> cutExists(const std::size_t &cut_key) const {
        auto it = cutMaster_to_cut_map.find(cut_key);
        if (it != cutMaster_to_cut_map.end()) {
            auto tam    = cuts[it->second].size();
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
    std::vector<double> computeLimitedMemoryCoefficients(const std::vector<uint16_t> &P) const {
        const size_t        P_size = P.size();
        std::vector<double> alphas;
        alphas.reserve(cuts.size());

        const uint16_t *P_data = P.data();

        for (const auto &cut : cuts) {
            double          alpha           = 0.0;
            int             runningSum      = 0;
            const int       denominator     = cut.p.den;
            const uint64_t *neighborsBitset = cut.neighbors.data();
            const uint64_t *baseSetBitset   = cut.baseSet.data();
            const auto     &pValues         = cut.p.num;
            const auto     &baseSetOrder    = cut.baseSetOrder;

#if defined(SRC_MEMORY_MODE_ARC)
            for (size_t j = 1; j < P_size - 1; ++j) {
                const int      nodeId    = P_data[j];
                const int      prevNode  = P_data[j - 1];
                const size_t   word      = nodeId >> 6;
                const uint64_t bit_mask  = 1ULL << (nodeId & 63);
                const size_t   prev_word = prevNode >> 6;
                const uint64_t prev_mask = 1ULL << (prevNode & 63);

                if (!(neighborsBitset[word] & bit_mask) || !(neighborsBitset[prev_word] & prev_mask)) {
                    runningSum = 0;
                }
                if (baseSetBitset[word] & bit_mask) {
                    const int pos = baseSetOrder[nodeId];
                    runningSum += pValues[pos];
                    if (runningSum >= denominator) {
                        runningSum -= denominator;
                        alpha += 1.0;
                    }
                }
            }
#else
            for (size_t j = 1; j < P_size - 1; ++j) {
                const int      nodeId   = P_data[j];
                const size_t   word     = nodeId >> 6;
                const uint64_t bit_mask = 1ULL << (nodeId & 63);

                if (!(neighborsBitset[word] & bit_mask)) {
                    runningSum = 0;
                } else if (baseSetBitset[word] & bit_mask) {
                    const int pos = baseSetOrder[nodeId];
                    runningSum += pValues[pos];
                    if (runningSum >= denominator) {
                        runningSum -= denominator;
                        alpha += 1.0;
                    }
                }
            }
#endif
            alphas.push_back(alpha);
        }
        return alphas;
    }

    /**
     * @brief Computes a unique cut key based on the provided base set and
     * multipliers.
     *
     * This function generates a hash value that uniquely identifies the
     * combination of the base set and multipliers. The base set is an array of
     * uint64_t values, and the multipliers are a vector of double values. The
     * order of elements in both the base set and multipliers is preserved
     * during hashing.
     *
     */
    std::size_t compute_cut_key(const std::array<uint64_t, num_words> &baseSet, const std::vector<int> &perm_num,
                                const int perm_den) {
        XXH3_state_t *state = XXH3_createState();
        assert(state != nullptr);
        XXH3_64bits_reset(state);

        // Hash baseSet (array of uint64_t)
        XXH3_64bits_update(state, baseSet.data(), baseSet.size() * sizeof(uint64_t));
        // Hash perm_num
        XXH3_64bits_update(state, perm_num.data(), perm_num.size() * sizeof(int));
        // Hash perm_den
        XXH3_64bits_update(state, &perm_den, sizeof(int));

        std::size_t cut_key = XXH3_64bits_digest(state);
        XXH3_freeState(state);
        return cut_key;
    }

    double computeLimitedMemoryCoefficient(const std::vector<uint16_t> &P, const Cut &cut) const {
        const size_t P_size = P.size();
        if (P_size < 3) return 0.0;

        const uint16_t *P_data      = P.data();
        double          alpha       = 0.0;
        int             runningSum  = 0;
        const int       denominator = cut.p.den;

        const uint64_t *neighborsBitset = cut.neighbors.data();
        const uint64_t *baseSetBitset   = cut.baseSet.data();
        const auto     &pValues         = cut.p.num;
        const auto     &baseSetOrder    = cut.baseSetOrder;

#if defined(SRC_MEMORY_MODE_ARC)
        for (size_t j = 1; j < P_size - 1; ++j) {
            const int      nodeId    = P_data[j];
            const int      prevNode  = P_data[j - 1];
            const size_t   word      = nodeId >> 6;
            const uint64_t bit_mask  = 1ULL << (nodeId & 63);
            const size_t   prev_word = prevNode >> 6;
            const uint64_t prev_mask = 1ULL << (prevNode & 63);

            if (!(neighborsBitset[word] & bit_mask) || !(neighborsBitset[prev_word] & prev_mask)) { runningSum = 0; }
            if (baseSetBitset[word] & bit_mask) {
                const int pos = baseSetOrder[nodeId];
                runningSum += pValues[pos];
                if (runningSum >= denominator) {
                    runningSum -= denominator;
                    alpha += 1.0;
                }
            }
        }
#else
        for (size_t j = 1; j < P_size - 1; ++j) {
            const int      nodeId   = P_data[j];
            const size_t   word     = nodeId >> 6;
            const uint64_t bit_mask = 1ULL << (nodeId & 63);

            if (!(neighborsBitset[word] & bit_mask)) {
                runningSum = 0;
            } else if (baseSetBitset[word] & bit_mask) {
                const int pos = baseSetOrder[nodeId];
                runningSum += pValues[pos];
                if (runningSum >= denominator) {
                    runningSum -= denominator;
                    alpha += 1.0;
                }
            }
        }
#endif
        return alpha;
    }

    // Compute coefficients for a single cut given a vector of paths
    std::vector<double> loadCoefficients(const std::vector<Path> &allPaths, Cut &cut, bool loaded = false) {
        std::vector<double> coefficients;
        coefficients.assign(allPaths.size(), 0.0);
        for (size_t idx = 0; idx < allPaths.size(); ++idx) {
            coefficients[idx] = computeLimitedMemoryCoefficient(allPaths[idx].route, cut);
        }

        // adjust cut stuff
        if (loaded) {
            cut.key                       = compute_cut_key(cut.baseSet, cut.p.num, cut.p.den);
            cutMaster_to_cut_map[cut.key] = cut.id;
        }
        return coefficients;
    }

    void materializeSparseCoefficients(Cut &cut, size_t n_paths) const {
        if (!cut.hasSparseCoefficients()) return;
        cut.coefficients.assign(n_paths, 0.0);
        for (size_t idx = 0; idx < cut.coefficient_indices.size(); ++idx) {
            const int path_idx = cut.coefficient_indices[idx];
            if (path_idx >= 0 && static_cast<size_t>(path_idx) < n_paths) {
                cut.coefficients[static_cast<size_t>(path_idx)] = cut.coefficient_values[idx];
            }
        }
    }

    void syncCoefficientCacheForPath(const Path &path, size_t path_idx) {
        for (auto &cut : cuts) {
            if (cut.coefficients.size() == path_idx) {
                cut.coefficients.push_back(computeLimitedMemoryCoefficient(path.route, cut));
            }
        }
        latest_column = static_cast<int>(std::max<size_t>(latest_column, path_idx + 1));
    }

    void eraseCoefficientIndices(const std::vector<int> &descending_indices) {
        for (auto &cut : cuts) {
            if (cut.coefficients.empty()) continue;
            for (int idx : descending_indices) {
                if (idx >= 0 && static_cast<size_t>(idx) < cut.coefficients.size()) {
                    cut.coefficients.erase(cut.coefficients.begin() + idx);
                }
            }
        }
    }
    // Generate a cut key
    std::size_t generateCutKey(const int &cutMaster, const std::vector<bool> &baseSetStr) const {
        std::size_t key = cutMaster;
        for (bool b : baseSetStr) { key = (key << 1) | b; }
        return key;
    }
    std::vector<double>                            SRCDuals;             // Dual values for SRC
    ankerl::unordered_dense::map<std::size_t, int> cutMaster_to_cut_map; // Map from cut key to cut index

    // Cut pool management configuration
    size_t  max_pool_size_         = 500;  // Max cuts in storage pool
    int64_t current_epoch_         = 0;    // B&B node epoch counter for aging
    double  age_decay_alpha_       = 0.97; // Exponential decay factor per epoch (0-1)
    double  dual_ema_alpha_        = 0.20; // EMA weight for current dual magnitude
    double  selection_temperature_ = 2.0;  // Temperature for probabilistic selection (higher = more random)

private:
    std::vector<Cut> cuts; // Storage for cuts

    ankerl::unordered_dense::map<std::size_t, std::vector<int>> indexCuts;   // Additional index for cuts (if needed)
    std::vector<ActiveCutInfo>                                  active_cuts; // Cuts with positive duals
    std::array<std::vector<SRCNodeUpdate>, N_SIZE>              src_node_updates;
    std::array<std::vector<uint16_t>, N_SIZE>                   src_node_clears;

    void precomputeSRCNodeTables() {
        for (auto &updates : src_node_updates) { updates.clear(); }
        for (auto &clears : src_node_clears) { clears.clear(); }

        for (int node_id = 0; node_id < N_SIZE; ++node_id) {
            const size_t   segment = static_cast<size_t>(node_id) >> 6;
            const uint64_t bit     = 1ULL << (node_id & 63);

            auto &updates = src_node_updates[node_id];
            auto &clears  = src_node_clears[node_id];
            updates.reserve(active_cuts.size());
            clears.reserve(active_cuts.size());

            for (const auto &active_cut : active_cuts) {
                const auto &cut = *active_cut.cut_ptr;
                if (!(cut.neighbors[segment] & bit)) {
                    clears.push_back(static_cast<uint16_t>(active_cut.index));
                    continue;
                }
                if (cut.baseSet[segment] & bit) {
                    updates.push_back({static_cast<uint16_t>(active_cut.index),
                                       static_cast<uint16_t>(cut.p.num[cut.baseSetOrder[node_id]]),
                                       static_cast<uint16_t>(cut.p.den), active_cut.dual_value});
                }
            }
        }
    }

    /**
     * @brief Compute probabilistic selection score for a cut.
     *
     * Score combines current dual strength, historical violation strength, age,
     * and prior activity. Current nonzero duals dominate selection, but useful
     * recently-separated cuts survive even before the solver gives them a large
     * dual.
     */
    double computeSelectionScore(size_t cut_idx) const {
        if (cut_idx >= cuts.size()) return 0.0;

        const Cut   &cut      = cuts[cut_idx];
        const double abs_dual = (cut_idx < SRCDuals.size()) ? std::abs(SRCDuals[cut_idx]) : 0.0;

        const double current_dual_strength    = std::log1p(abs_dual);
        const double historical_dual_strength = std::log1p(cut.avg_dual_magnitude);
        const double violation_strength =
            std::log1p(std::max({0.0, cut.last_violation, cut.max_violation, cut.total_violation * 0.1}));
        const double activity_bonus = 1.0 + 0.08 * std::log1p(static_cast<double>(cut.active_count)) +
                                      0.04 * std::log1p(static_cast<double>(cut.inclusion_count));

        const int64_t last_use_epoch = std::max(
            {cut.creation_epoch, cut.last_separation_epoch, cut.last_nonzero_dual_epoch, cut.last_active_epoch});
        const int64_t age         = std::max<int64_t>(0, current_epoch_ - last_use_epoch);
        const double  age_penalty = (age == 0) ? 1.0 : std::pow(age_decay_alpha_, static_cast<double>(age));

        return (4.0 * current_dual_strength + historical_dual_strength + violation_strength) * activity_bonus *
               age_penalty;
    }

    /**
     * @brief Probabilistic selection using weighted sampling.
     *
     * Instead of deterministic top-K by dual magnitude, uses softmax-like
     * weighted selection where score includes dual magnitude, age, and
     * historical usefulness. Higher temperature = more exploration (random).
     */
    void updateActiveCutsProbabilistic() {
        constexpr size_t max_active_cuts = std::min<size_t>(MAX_SRC_CUTS, 64);

        active_cuts.clear();
        active_cuts.reserve(std::min(cuts.size(), max_active_cuts));

        // Collect candidates with scores
        std::vector<std::pair<size_t, double>> scored_candidates;
        scored_candidates.reserve(cuts.size());

        for (size_t i = 0; i < cuts.size(); ++i) {
            if (i < SRCDuals.size() && std::abs(SRCDuals[i]) > SRC_DUAL_TOLERANCE) {
                double score = computeSelectionScore(i);
                scored_candidates.emplace_back(i, score);
            }
        }

        if (scored_candidates.empty()) {
            segment_masks.precompute(active_cuts);
            precomputeSRCNodeTables();
            return;
        }

        // Sort by score descending
        pdqsort(scored_candidates.begin(), scored_candidates.end(),
                [](const auto &a, const auto &b) { return a.second > b.second; });

        // Truncate to active cut limit
        if (scored_candidates.size() > max_active_cuts) { scored_candidates.resize(max_active_cuts); }

        // Build active cuts from top scored
        for (auto &[cut_idx, score] : scored_candidates) {
            active_cuts.emplace_back(active_cuts.size(), cut_idx, &cuts[cut_idx], SRCDuals[cut_idx],
                                     cuts[cut_idx].type);
            auto &cut = cuts[cut_idx];
            cut.inclusion_count++;
            cut.active_count++;
            cut.last_active_epoch       = current_epoch_;
            cut.last_nonzero_dual_epoch = current_epoch_;
            const double abs_dual       = std::abs(SRCDuals[cut_idx]);
            cut.avg_dual_magnitude =
                (cut.avg_dual_magnitude <= 0.0)
                    ? abs_dual
                    : (1.0 - dual_ema_alpha_) * cut.avg_dual_magnitude + dual_ema_alpha_ * abs_dual;
        }

        segment_masks.precompute(active_cuts);
        precomputeSRCNodeTables();
    }

    /**
     * @brief Original deterministic update (dual-magnitude only).
     * Kept for backward compatibility and fallback.
     */
    void updateActiveCutsDeterministic() {
        constexpr size_t max_active_cuts = std::min<size_t>(MAX_SRC_CUTS, 64);

        active_cuts.clear();
        active_cuts.reserve(std::min(cuts.size(), max_active_cuts));

        std::vector<size_t> candidates;
        candidates.reserve(std::min(cuts.size(), SRCDuals.size()));

        for (size_t i = 0; i < cuts.size(); ++i) {
            if (i < SRCDuals.size() && std::abs(SRCDuals[i]) > SRC_DUAL_TOLERANCE) { candidates.push_back(i); }
        }

        pdqsort(candidates.begin(), candidates.end(),
                [&](size_t a, size_t b) { return std::abs(SRCDuals[a]) > std::abs(SRCDuals[b]); });
        if (candidates.size() > max_active_cuts) { candidates.resize(max_active_cuts); }

        for (const size_t cut_idx : candidates) {
            active_cuts.emplace_back(active_cuts.size(), cut_idx, &cuts[cut_idx], SRCDuals[cut_idx],
                                     cuts[cut_idx].type);
            auto &cut = cuts[cut_idx];
            cut.inclusion_count++;
            cut.active_count++;
            cut.last_active_epoch       = current_epoch_;
            cut.last_nonzero_dual_epoch = current_epoch_;
        }

        segment_masks.precompute(active_cuts);
        precomputeSRCNodeTables();
    }

public:
    /**
     * @brief Enforce pool size limit for cuts that are not in the LP yet.
     *
     * Cuts with an LP row must be removed together with their model constraint
     * and the caller-owned SRCconstraints vector. This storage-only eviction is
     * therefore limited to pending cuts; active/model cuts are cleaned by
     * LimitedMemoryRank1Cuts::cutCleaner.
     */
    void enforcePoolSizeLimit() {
        if (cuts.size() <= max_pool_size_) return;

        // Compute scores for all cuts.
        std::vector<std::pair<size_t, double>> scored;
        scored.reserve(cuts.size());
        for (size_t i = 0; i < cuts.size(); ++i) {
            if (!cuts[i].added) { scored.emplace_back(i, computeSelectionScore(i)); }
        }
        if (scored.empty()) return;

        // Sort by score ascending (lowest first = candidates for removal)
        pdqsort(scored.begin(), scored.end(), [](const auto &a, const auto &b) { return a.second < b.second; });

        // Remove excess pending cuts (lowest scoring first). Erase in
        // descending index order so vector indices remain valid.
        const size_t     excess = std::min(cuts.size() - max_pool_size_, scored.size());
        std::vector<int> remove_indices;
        remove_indices.reserve(excess);
        for (auto &[cut_idx, score] : scored) {
            if (remove_indices.size() >= excess) break;
            remove_indices.push_back(static_cast<int>(cut_idx));
        }
        std::sort(remove_indices.begin(), remove_indices.end(), std::greater<int>());
        for (int idx : remove_indices) { removeCut(idx); }
    }

    void markCutSeparatedByKey(size_t cut_key, double violation) {
        auto it = cutMaster_to_cut_map.find(cut_key);
        if (it == cutMaster_to_cut_map.end()) return;
        auto &stored = cuts[it->second];
        stored.separation_count++;
        stored.last_violation = std::max(0.0, violation);
        stored.total_violation += stored.last_violation;
        stored.max_violation         = std::max(stored.max_violation, stored.last_violation);
        stored.last_separation_epoch = current_epoch_;
        if (stored.creation_epoch == 0) { stored.creation_epoch = current_epoch_; }
        if (stored.added) { stored.updated = true; }
    }

    void markCutSeparated(Cut &cut, double violation) {
        if (cut.key == 0) { cut.key = compute_cut_key(cut.baseSet, cut.p.num, cut.p.den); }
        cut.separation_count++;
        cut.last_violation = std::max(0.0, violation);
        cut.total_violation += cut.last_violation;
        cut.max_violation         = std::max(cut.max_violation, cut.last_violation);
        cut.last_separation_epoch = current_epoch_;
        if (cut.creation_epoch == 0) { cut.creation_epoch = current_epoch_; }

        auto it = cutMaster_to_cut_map.find(cut.key);
        if (it != cutMaster_to_cut_map.end()) {
            auto &stored = cuts[it->second];
            stored.separation_count++;
            stored.last_violation = cut.last_violation;
            stored.total_violation += cut.last_violation;
            stored.max_violation         = std::max(stored.max_violation, cut.last_violation);
            stored.last_separation_epoch = current_epoch_;
            if (stored.creation_epoch == 0) { stored.creation_epoch = current_epoch_; }
            if (stored.added) { stored.updated = true; }
        }
    }

    void updateActiveCuts() {
        // Increment epoch at each update call for aging
        current_epoch_++;
        updateActiveCutsProbabilistic();
    }

public:
    REFLECT(cuts)

    void printCuts() {
        for (auto &cut : cuts) { cut.printCut(); }
    }

    void updateRedCost(std::vector<Label *> labels) {
        const auto active_cuts = getActiveCuts();

        const auto n_cuts = activeSize();

        for (auto label : labels) {
            label->SRCmap.assign(n_cuts, 0);
            if (n_cuts == 0) continue;
            int prev_node = -1;
            for (auto node_id : label->getRoute()) {
                double total_cost_update = 0.0;
#if defined(SRC_MEMORY_MODE_ARC)
                for (const auto &active_cut : active_cuts) {
                    const auto &cut           = *active_cut.cut_ptr;
                    auto       &src_map_value = label->SRCmap[active_cut.index];
                    if (!cut.isSRCMemoryArc(prev_node, node_id)) { src_map_value = 0; }
                    if (cut.isSRCset(node_id)) {
                        src_map_value += cut.srcMultiplier(node_id);
                        if (src_map_value >= cut.p.den) {
                            src_map_value -= cut.p.den;
                            total_cost_update -= active_cut.dual_value;
                        }
                    }
                }
#else
                for (const auto &update : getSRCNodeUpdates(node_id)) {
                    auto &src_map_value = label->SRCmap[update.active_idx];
                    src_map_value += update.add;
                    if (src_map_value >= update.den) {
                        src_map_value -= update.den;
                        total_cost_update -= update.dual;
                    }
                }

                for (const auto active_idx : getSRCNodeClears(node_id)) { label->SRCmap[active_idx] = 0; }
#endif
                label->cost += total_cost_update;
                prev_node = node_id;
            }
        }
    }
};
