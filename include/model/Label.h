/**
 * @file Label.h
 * @brief Defines the Label state and comparator used by the solver.
 *
 */

#pragma once
#include "math/Common.h"

#include <cassert>
#include <stdexcept>

#ifdef SRC
struct SRCMap {
    std::array<uint16_t, MAX_SRC_CUTS> values       = {};
    std::size_t                        logical_size = 0;

    SRCMap() = default;

    explicit SRCMap(std::size_t n, uint16_t value = 0) { assign(n, value); }

    SRCMap(const std::vector<uint16_t> &src) { *this = src; }

    SRCMap &operator=(const std::vector<uint16_t> &src) {
        ensure_capacity(src.size());
        std::copy(src.begin(), src.end(), values.begin());
        logical_size = src.size();
        return *this;
    }

    static void ensure_capacity(std::size_t n) {
        if (unlikely(n > MAX_SRC_CUTS)) { throw std::length_error("SRCMap capacity exceeded; increase MAX_SRC_CUTS"); }
    }

    void clear() noexcept { logical_size = 0; }

    void resize(std::size_t n, uint16_t value = 0) {
        ensure_capacity(n);
        assert(n <= MAX_SRC_CUTS);
        if (n > logical_size) {
            std::fill(values.begin() + static_cast<std::ptrdiff_t>(logical_size),
                      values.begin() + static_cast<std::ptrdiff_t>(n), value);
        }
        logical_size = n;
    }

    void assign(std::size_t n, uint16_t value) {
        ensure_capacity(n);
        assert(n <= MAX_SRC_CUTS);
        std::fill(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(n), value);
        logical_size = n;
    }

    [[nodiscard]] std::size_t size() const noexcept { return logical_size; }
    [[nodiscard]] bool        empty() const noexcept { return logical_size == 0; }

    uint16_t       *data() noexcept { return values.data(); }
    const uint16_t *data() const noexcept { return values.data(); }

    uint16_t &operator[](std::size_t idx) noexcept {
        assert(idx < MAX_SRC_CUTS);
        return values[idx];
    }

    const uint16_t &operator[](std::size_t idx) const noexcept {
        assert(idx < MAX_SRC_CUTS);
        return values[idx];
    }

    [[nodiscard]] std::vector<uint16_t> to_vector() const {
        return {values.begin(), values.begin() + static_cast<std::ptrdiff_t>(logical_size)};
    }
};
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
    // Hot dominance/extension data. Keep the fields most frequently read by
    // dominance scans close together; route reconstruction stays below.
    double                          cost           = 0.0;
    double                          real_cost      = 0.0;
    std::array<double, R_SIZE>      resources      = {};
    std::array<uint64_t, num_words> visited_bitmap = {0};
#ifdef UNREACHABLE_DOMINANCE
    std::array<uint64_t, num_words> unreachable_bitmap = {0};
#endif
    SRC_MODE_BLOCK(SRCMap SRCmap;)

    int    vertex       = -1;
    int    node_id      = -1;
    int    path_len     = 0;
    Label *parent       = nullptr;
    bool   is_extended  = false;
    bool   is_dominated = false;
    bool   fresh        = true;

    // Cold route materialization data.
    std::vector<uint16_t> nodes_covered = {};
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

    const auto &getRoute() const { return nodes_covered; }

    void clearRoute() noexcept {
        nodes_covered.clear();
        path_len = 0;
        parent   = nullptr;
    }

    void addRoute(const std::vector<int> &route) {
        nodes_covered.insert(nodes_covered.end(), route.begin(), route.end());
        path_len = static_cast<int>(nodes_covered.size());
        parent   = nullptr;
    }

    void addRoute(const std::vector<uint16_t> &route) {
        nodes_covered.insert(nodes_covered.end(), route.begin(), route.end());
        path_len = static_cast<int>(nodes_covered.size());
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
     * Conservative 64-bit summary used to reject impossible visited-set
     * subset tests before reading the complete bitmap. If A is a subset of B,
     * signature(A) is necessarily a subset of signature(B); collisions only
     * cause a full check, never an incorrect dominance result.
     */
    [[nodiscard]] uint64_t visited_signature() const noexcept {
        uint64_t signature = 0;
        for (uint64_t word : visited_bitmap) signature |= word;
        return signature;
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

        // Clear route storage; keep capacity for amortized reuse.
        nodes_covered.clear();

        // Zero out the bitmaps efficiently.
        std::memset(visited_bitmap.data(), 0, visited_bitmap.size() * sizeof(uint64_t));
#ifdef UNREACHABLE_DOMINANCE
        std::memset(unreachable_bitmap.data(), 0, unreachable_bitmap.size() * sizeof(uint64_t));
#endif

        // If using source mode mapping, clear it.
        SRC_MODE_BLOCK(SRCmap.clear();)
    }

    void addNode(int node) {
        nodes_covered.push_back(node);
        path_len = static_cast<int>(nodes_covered.size());
        parent   = nullptr;
    }

    /**
     * @brief Initializes the object with the given parameters.
     *
     */
    inline void initialize(int vertex, double cost, const std::vector<double> &resources, int node_id) {
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
