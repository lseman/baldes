/**
 * @file Label.h
 * @brief Defines the Label state and comparator used by the solver.
 *
 */

#pragma once
#include "math/Common.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef SRC
/**
 * Rank-1 cut memory state of a label, stored inline.
 *
 * Every stored value is reduced modulo its cut denominator, so one byte per
 * cut is enough. Keeping the bytes inline makes copying a label's state a
 * fixed-size copy instead of a heap allocation, and lets dominance compare
 * whole 32-byte chunks at once (see src_greater_mask32).
 */
struct SRCMap {
    static constexpr std::size_t kStorage = (MAX_SRC_CUTS + 31) / 32 * 32;

    std::array<uint8_t, kStorage> values{};
    uint8_t                       logical_size = 0;

    SRCMap() = default;

    explicit SRCMap(std::size_t n, uint16_t value = 0) { assign(n, value); }

    SRCMap(const std::vector<uint16_t> &src) { *this = src; }

    SRCMap &operator=(const std::vector<uint16_t> &src) {
        ensure_capacity(src.size());
        values.fill(0);
        logical_size = static_cast<uint8_t>(src.size());
        for (std::size_t i = 0; i < src.size(); ++i) values[i] = narrow(src[i]);
        return *this;
    }

    static void ensure_capacity(std::size_t n) {
        if (unlikely(n > MAX_SRC_CUTS)) { throw std::length_error("SRCMap capacity exceeded; increase MAX_SRC_CUTS"); }
    }

    static uint8_t narrow(uint16_t value) {
        if (unlikely(value > UINT8_MAX)) { throw std::out_of_range("SRCMap value does not fit in one byte"); }
        return static_cast<uint8_t>(value);
    }

    void clear() noexcept {
        values.fill(0);
        logical_size = 0;
    }

    void resize(std::size_t n, uint16_t value = 0) {
        ensure_capacity(n);
        const uint8_t v = narrow(value);
        for (std::size_t i = logical_size; i < n; ++i) values[i] = v;
        for (std::size_t i = n; i < logical_size; ++i) values[i] = 0;
        logical_size = static_cast<uint8_t>(n);
    }

    void assign(std::size_t n, uint16_t value) {
        ensure_capacity(n);
        values.fill(0);
        std::fill_n(values.begin(), n, narrow(value));
        logical_size = static_cast<uint8_t>(n);
    }

    [[nodiscard]] std::size_t    size() const noexcept { return logical_size; }
    [[nodiscard]] bool           empty() const noexcept { return logical_size == 0; }
    [[nodiscard]] const uint8_t *data() const noexcept { return values.data(); }

    uint8_t &operator[](std::size_t idx) noexcept {
        assert(idx < MAX_SRC_CUTS);
        return values[idx];
    }

    uint8_t operator[](std::size_t idx) const noexcept {
        assert(idx < MAX_SRC_CUTS);
        return values[idx];
    }

    [[nodiscard]] std::vector<uint16_t> to_vector() const {
        return std::vector<uint16_t>(values.begin(), values.begin() + logical_size);
    }
};

/**
 * Bit i is set when a[i] > b[i] (unsigned), for the 32 bytes at a and b.
 */
inline uint32_t src_greater_mask32(const uint8_t *a, const uint8_t *b) noexcept {
#if defined(__AVX2__)
    const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a));
    const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b));
    // a > b  <=>  min(a, b) != a
    const __m256i not_greater = _mm256_cmpeq_epi8(_mm256_min_epu8(va, vb), va);
    return ~static_cast<uint32_t>(_mm256_movemask_epi8(not_greater));
#else
    uint32_t mask = 0;
    for (int i = 0; i < 32; ++i) mask |= static_cast<uint32_t>(a[i] > b[i]) << i;
    return mask;
#endif
}
#endif

/**
 * @struct Label
 * @brief Represents a label used in a solver.
 *
 * This struct contains various properties and methods related to a label used
 * in a solver. It stores information such as the set of Fj, id, is_extended
 * flag, vertex, cost, real_cost, SRC_cost, resources, predecessor, is_dominated
 * flag, nodes_covered, nodes_ordered, node_id, cut_storage, parent, children,
 * status, visited, and SRCmap.
 *
 * The struct provides constructors to initialize the label with or without a
 * node_id. It also provides methods to set the covered nodes, add a node to the
 * covered nodes, check if a node is already covered, and initialize the label
 * with new values.
 *
 * The struct overloads the equality and greater than operators for comparison.
 */
struct Label {
    static constexpr std::size_t bitmap_words = (N_SIZE + 63) / 64;

    // Hot dominance/extension data. Keep the fields most frequently read by
    // dominance scans close together; route reconstruction stays below.
    double                          cost           = 0.0;
    double                          real_cost      = 0.0;
    std::array<double, R_SIZE>      resources      = {};
    std::array<uint64_t, bitmap_words> visited_bitmap = {0};
    // 64-bit summary of visited_bitmap used by dominance; see
    // visited_signature(). Zero is always safe (it never causes a wrong
    // dominance, at most a missed one).
    uint64_t visited_sig = 0;
#ifdef UNREACHABLE_DOMINANCE
    std::array<uint64_t, bitmap_words> unreachable_bitmap = {0};
#endif
#ifdef SRC
    SRCMap SRCmap;
#endif

    int    vertex       = -1;
    int    node_id      = -1;
    int    path_len     = 0;
    const Label *parent = nullptr;
    bool   is_extended  = false;
    bool   is_dominated = false;
    bool   fresh        = true;

    // Cold route materialization data. Partial labels pay for one nullable
    // pointer; accepted columns allocate and own the actual route vector.
    std::unique_ptr<std::vector<uint16_t>> route_storage;
    // Constructor with node_id
    Label(int v, double c, const std::vector<double> &res, int pred, int node_id)
        : vertex(v), cost(c), resources({res[0]}), node_id(node_id) {}

    // Constructor without node_id
    Label(int v, double c, const std::vector<double> &res, int pred)
        : vertex(v), cost(c), resources({res[0]}), node_id(-1) {}

    // Default constructor
    Label() : vertex(-1), cost(0), resources({0.0}), node_id(-1) {}

    void set_extended(bool extended) { is_extended = extended; }
    void set_dominated(bool dominated) { is_dominated = dominated; }

    Label(const Label &)            = delete;
    Label &operator=(const Label &) = delete;
    Label(Label &&)                 = default;
    Label &operator=(Label &&)      = default;

    // Only materialized output labels expose their owned route directly.
    const std::vector<uint16_t> &getRoute() const noexcept {
        static const std::vector<uint16_t> empty_route;
        return route_storage ? *route_storage : empty_route;
    }

    std::vector<uint16_t> &mutableRoute() {
        if (!route_storage) route_storage = std::make_unique<std::vector<uint16_t>>();
        return *route_storage;
    }

    void materializeRoute(std::vector<uint16_t> &route) const {
        if (parent == nullptr) {
            const auto &stored_route = getRoute();
            route.assign(stored_route.begin(), stored_route.end());
            return;
        }

        const Label *root  = this;
        std::size_t  depth = 0;
        while (root->parent != nullptr) {
            ++depth;
            root = root->parent;
        }

        const auto       &root_route  = root->getRoute();
        const std::size_t prefix_size = root_route.size();
        route.resize(prefix_size + depth);
        std::copy(root_route.begin(), root_route.end(), route.begin());

        const Label *current = this;
        std::size_t  write   = route.size();
        while (current->parent != nullptr) {
            route[--write] = static_cast<uint16_t>(current->node_id);
            current        = current->parent;
        }
    }

    void clearRoute() noexcept {
        route_storage.reset();
        path_len = 0;
        parent   = nullptr;
    }

    void addRoute(const std::vector<int> &route) {
        auto &stored_route = mutableRoute();
        stored_route.insert(stored_route.end(), route.begin(), route.end());
        path_len = static_cast<int>(stored_route.size());
        parent   = nullptr;
    }

    void addRoute(const std::vector<uint16_t> &route) {
        auto &stored_route = mutableRoute();
        stored_route.insert(stored_route.end(), route.begin(), route.end());
        path_len = static_cast<int>(stored_route.size());
        parent   = nullptr;
    }
    /**
     * @brief Checks if a node has been visited.
     *
     * This function determines whether a node, identified by its node_id, has
     * been visited. It uses a bitmask (visited_bitmap) where each bit
     * represents the visit status of a node.
     *
     */
    bool visits(int node_id) const noexcept { return visited_bitmap[node_id / 64] & (1ULL << (node_id % 64)); }

    /**
     * Visited-set signature used by dominance. Two encodings, told apart by
     * kExactNgBit:
     *  - exact: the ng-memory in the label vertex's local coordinates (bit p
     *    set iff the p-th node of N(v), in increasing node id, is memorised),
     *    plus kExactNgBit. For two labels at the same vertex the subset test
     *    on this word is exactly the subset test on visited_bitmap, so the
     *    bitmap need not be read.
     *  - conservative: OR-fold of visited_bitmap with kExactNgBit cleared.
     *    If A is a subset of B then sig(A) is a subset of sig(B); a passing
     *    test still requires the full bitmap check.
     * Set by BucketGraph::finalize_visited_signature().
     */
    static constexpr uint64_t kExactNgBit = 1ULL << 63;

    [[nodiscard]] uint64_t visited_signature() const noexcept { return visited_sig; }

    [[nodiscard]] uint64_t conservative_visited_signature() const noexcept {
        uint64_t signature = 0;
        for (uint64_t word : visited_bitmap) signature |= word;
        return signature & ~kExactNgBit;
    }

    // True when both signatures are exact local ng words, i.e. the
    // signature subset test is the complete visited-set subset test.
    [[nodiscard]] static constexpr bool exact_visited_pair(uint64_t lhs, uint64_t rhs) noexcept {
        return ((lhs & rhs) & kExactNgBit) != 0;
    }

    /**
     * @brief Resets the state of the object to its initial values.
     *
     */
    inline void reset() noexcept {
        // Reset basic properties
        vertex       = -1;
        cost         = 0.0;
        node_id      = -1;
        real_cost    = 0.0;
        path_len     = 0;
        parent       = nullptr;
        is_extended  = false;
        is_dominated = false;
        fresh        = true;
        // Reset resources container (assuming operator= clears properly)
        resources = {};

        // Release cold output storage before this label returns to the hot
        // partial-label pool.
        route_storage.reset();

        // Zero out the bitmaps efficiently.
        std::memset(visited_bitmap.data(), 0, visited_bitmap.size() * sizeof(uint64_t));
        visited_sig = 0;
#ifdef UNREACHABLE_DOMINANCE
        std::memset(unreachable_bitmap.data(), 0, unreachable_bitmap.size() * sizeof(uint64_t));
#endif

        // If using source mode mapping, clear it.
#ifdef SRC
        SRCmap.clear();
#endif
    }

    void addNode(int node) {
        auto &stored_route = mutableRoute();
        stored_route.push_back(node);
        path_len = static_cast<int>(stored_route.size());
        parent   = nullptr;
    }

    /**
     * @brief Initializes the object with the given parameters.
     *
     */
    inline void initialize(int vertex, double cost, std::span<const double> resources, int node_id) {
        this->vertex = vertex;
        this->cost   = cost;

        // Assuming `resources` is a vector or array-like structure with the
        // same size as the input
        std::copy(resources.begin(), resources.end(), this->resources.begin());

        this->node_id  = node_id;
        this->path_len = 0;
    }

    bool operator>(const Label &other) const { return cost > other.cost; }

    bool operator<(const Label &other) const { return cost < other.cost; }
};

/**
 * @class LabelComparator
 * @brief Comparator class for comparing two Label objects based on their cost.
 *
 * This class provides an overloaded operator() that allows for comparison
 * between two Label pointers. The comparison is based on the cost attribute
 * of the Label objects, with the comparison being in descending order.
 */
class LabelComparator {
public:
    bool operator()(Label *a, Label *b) { return a->cost > b->cost; }
};
