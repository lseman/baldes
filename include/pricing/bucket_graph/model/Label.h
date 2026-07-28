/**
 * @file Label.h
 * @brief Defines the Label state and comparator used by the solver.
 *
 */

#pragma once
#include "math/Common.h"

#include <cassert>
#include <memory>
#include <stdexcept>

#ifdef SRC
struct SRCMap {
    class Reference {
    public:
        Reference(SRCMap &owner, std::size_t index) noexcept : owner_(owner), index_(index) {}

        operator uint16_t() const noexcept { return owner_.get(index_); }

        Reference &operator=(uint16_t value) {
            owner_.set(index_, value);
            return *this;
        }

        Reference &operator+=(uint16_t value) {
            owner_.set(index_, static_cast<uint16_t>(owner_.get(index_) + value));
            return *this;
        }

        Reference &operator-=(uint16_t value) {
            owner_.set(index_, static_cast<uint16_t>(owner_.get(index_) - value));
            return *this;
        }

    private:
        SRCMap    &owner_;
        std::size_t index_;
    };

    std::array<uint8_t, MAX_SRC_CUTS> compact_values = {};
    std::unique_ptr<std::array<uint16_t, MAX_SRC_CUTS>> wide_values;
    uint8_t logical_size = 0;

    SRCMap() = default;

    explicit SRCMap(std::size_t n, uint16_t value = 0) { assign(n, value); }

    SRCMap(const std::vector<uint16_t> &src) { *this = src; }

    SRCMap(const SRCMap &other)
        : compact_values(other.compact_values), logical_size(other.logical_size) {
        if (other.wide_values) {
            wide_values = std::make_unique<std::array<uint16_t, MAX_SRC_CUTS>>(*other.wide_values);
        }
    }

    SRCMap &operator=(const SRCMap &other) {
        if (this == &other) return *this;
        compact_values = other.compact_values;
        logical_size   = other.logical_size;
        wide_values = other.wide_values
                          ? std::make_unique<std::array<uint16_t, MAX_SRC_CUTS>>(*other.wide_values)
                          : nullptr;
        return *this;
    }

    SRCMap(SRCMap &&) noexcept            = default;
    SRCMap &operator=(SRCMap &&) noexcept = default;

    SRCMap &operator=(const std::vector<uint16_t> &src) {
        ensure_capacity(src.size());
        clear();
        logical_size = static_cast<uint8_t>(src.size());
        for (std::size_t i = 0; i < src.size(); ++i) set(i, src[i]);
        return *this;
    }

    static void ensure_capacity(std::size_t n) {
        if (unlikely(n > MAX_SRC_CUTS)) { throw std::length_error("SRCMap capacity exceeded; increase MAX_SRC_CUTS"); }
    }

    void clear() noexcept {
        logical_size = 0;
        wide_values.reset();
    }

    void resize(std::size_t n, uint16_t value = 0) {
        ensure_capacity(n);
        assert(n <= MAX_SRC_CUTS);
        if (n > logical_size) {
            for (std::size_t i = logical_size; i < n; ++i) set(i, value);
        }
        logical_size = static_cast<uint8_t>(n);
    }

    void assign(std::size_t n, uint16_t value) {
        ensure_capacity(n);
        assert(n <= MAX_SRC_CUTS);
        wide_values.reset();
        logical_size = static_cast<uint8_t>(n);
        if (value <= UINT8_MAX) {
            std::fill_n(compact_values.begin(), n, static_cast<uint8_t>(value));
        } else {
            promote();
            std::fill_n(wide_values->begin(), n, value);
        }
    }

    [[nodiscard]] std::size_t size() const noexcept { return logical_size; }
    [[nodiscard]] bool        empty() const noexcept { return logical_size == 0; }
    [[nodiscard]] bool        is_compact() const noexcept { return !wide_values; }

    [[nodiscard]] const void *storage_data() const noexcept {
        return wide_values ? static_cast<const void *>(wide_values->data())
                           : static_cast<const void *>(compact_values.data());
    }

    Reference operator[](std::size_t idx) noexcept {
        assert(idx < MAX_SRC_CUTS);
        return Reference(*this, idx);
    }

    uint16_t operator[](std::size_t idx) const noexcept {
        assert(idx < MAX_SRC_CUTS);
        return get(idx);
    }

    [[nodiscard]] std::vector<uint16_t> to_vector() const {
        std::vector<uint16_t> result(logical_size);
        for (std::size_t i = 0; i < logical_size; ++i) result[i] = get(i);
        return result;
    }

private:
    [[nodiscard]] uint16_t get(std::size_t idx) const noexcept {
        return wide_values ? (*wide_values)[idx] : compact_values[idx];
    }

    void set(std::size_t idx, uint16_t value) {
        if (value > UINT8_MAX && !wide_values) promote();
        if (wide_values) {
            (*wide_values)[idx] = value;
        } else {
            compact_values[idx] = static_cast<uint8_t>(value);
        }
    }

    void promote() {
        wide_values = std::make_unique<std::array<uint16_t, MAX_SRC_CUTS>>();
        std::copy(compact_values.begin(), compact_values.end(), wide_values->begin());
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
    static constexpr std::size_t bitmap_words = (N_SIZE + 63) / 64;

    // Hot dominance/extension data. Keep the fields most frequently read by
    // dominance scans close together; route reconstruction stays below.
    double                          cost           = 0.0;
    double                          real_cost      = 0.0;
    std::array<double, R_SIZE>      resources      = {};
    std::array<uint64_t, bitmap_words> visited_bitmap = {0};
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
    bool   bucket_dominance_checked = false;
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
        bucket_dominance_checked = false;
        fresh        = true;
        // Reset resources container (assuming operator= clears properly)
        resources = {};

        // Release cold output storage before this label returns to the hot
        // partial-label pool.
        route_storage.reset();

        // Zero out the bitmaps efficiently.
        std::memset(visited_bitmap.data(), 0, visited_bitmap.size() * sizeof(uint64_t));
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
