/*
 * @file Bucket.h
 * @brief This file contains the definition of the Bucket struct.
 *
 * This file contains the definition of the Bucket struct, which represents a
 * bucket in a solver. The Bucket struct contains information about the node ID,
 * lower bounds, upper bounds, forward arcs, backward arcs, forward jump arcs,
 * backward jump arcs, and labels associated with the bucket. It provides
 * methods to add arcs, add jump arcs, get arcs, get jump arcs, add labels,
 * remove labels, get labels, clear labels, reset labels, and clear arcs.
 *
 */
#pragma once

#include "Arc.h"
#include "Definitions.h"
#include "Label.h"
#include "config.h"

enum class BucketState { Split, Unsplit };

/**
 * @struct Bucket
 * @brief Represents a bucket.
 *
 * A bucket is a data structure that contains labels, node ID, lower bounds,
 * upper bounds, forward arcs, backward arcs, forward jump arcs, and backward
 * jump arcs. It provides methods to add arcs, add jump arcs, get arcs, get jump
 * arcs, add labels, remove labels, get labels, clear labels, reset labels, and
 * clear arcs.
 */
struct alignas(64) Bucket {
    std::vector<Label *> labels_vec;
    int                  depth{0};
    int                  node_id{-1};
    bool                 is_split{false};
    double               min_cost{std::numeric_limits<double>::max()};
    char                 padding[32];

    std::vector<double>    lb;
    std::vector<double>    ub;
    std::vector<Arc>       fw_arcs;
    std::vector<Arc>       bw_arcs;
    std::vector<BucketArc> fw_bucket_arcs;
    std::vector<BucketArc> bw_bucket_arcs;
    std::vector<JumpArc>   fw_jump_arcs;
    std::vector<JumpArc>   bw_jump_arcs;

    size_t               split_point{0};
    double               split_midpoint{0.0};
    std::vector<Label *> labels_flush;
    bool                 shall_split{false};

    static constexpr size_t RESOURCE_INDEX        = 0;
    static constexpr int    MAX_RECURSION_DEPTH   = 1;
    static constexpr int    MAX_BUCKET_DEPTH      = 10;
    static constexpr double MIN_SPLIT_RANGE       = 0.5;
    static constexpr size_t SMALL_BATCH_THRESHOLD = 16;
    static constexpr size_t MIN_RESERVE_SIZE      = 64;

    // Core operations with template dispatch
    void add_label(Label *label) noexcept {
        is_split ? add_label<BucketState::Split>(label) : add_label<BucketState::Unsplit>(label);
    }

    void remove_label(Label *label) noexcept {
        if (!label) return;
        remove_label_impl(label, 0);
    }

    void remove_labels_batch(const std::vector<Label *> &to_remove) noexcept {
        is_split ? remove_labels_batch<BucketState::Split>(to_remove)
                 : remove_labels_batch<BucketState::Unsplit>(to_remove);
    }

    Label *get_best_label() noexcept {
        return is_split ? get_best_label<BucketState::Split>() : get_best_label<BucketState::Unsplit>();
    }

    double get_cb() const noexcept {
        if (labels_vec.empty() && !is_split) { return std::numeric_limits<double>::max(); }
        return is_split ? get_cb<BucketState::Split>() : get_cb<BucketState::Unsplit>();
    }

    bool check_dominance(const Label *new_label, const auto &dominance_func, int &stat_n_dom) const noexcept {
        if (get_cb() > new_label->cost) return false;
        return is_split ? check_dominance<BucketState::Split>(new_label, dominance_func, stat_n_dom)
                        : check_dominance<BucketState::Unsplit>(new_label, dominance_func, stat_n_dom);
    }

    void lazy_flush(Label *new_label) noexcept { labels_flush.push_back(new_label); }

    void add_sorted_label(Label *label) noexcept {
        is_split ? add_sorted_label<BucketState::Split>(label) : add_sorted_label<BucketState::Unsplit>(label);
    }

private:
    auto find_insert_position(Label *label, size_t start, size_t end) const {
        if (!label || start >= labels_vec.size() || end > labels_vec.size() || start >= end) {
            return labels_vec.end();
        }

        return std::ranges::lower_bound(labels_vec.begin() + start, labels_vec.begin() + end, label,
                                        [](const Label *a, const Label *b) {
                                            if (!a || !b) return false;
                                            return a->cost < b->cost;
                                        });
    }

    // Template implementations
    template <BucketState State>
    void add_label(Label *label) noexcept {
        if (!label) return;

        if constexpr (State == BucketState::Split) {
            if (labels_vec.empty()) {
                is_split = false;
                add_label<BucketState::Unsplit>(label);
                return;
            }

            const bool   goes_to_first = label->resources[RESOURCE_INDEX] <= split_midpoint;
            const size_t start         = goes_to_first ? 0 : split_point;
            const size_t end           = goes_to_first ? split_point : labels_vec.size();

            ensure_capacity();
            auto insert_pos = find_insert_position(label, start, end);
            labels_vec.insert(insert_pos, label);
            if (goes_to_first) split_point++;
        } else {
            ensure_capacity();
            labels_vec.push_back(label);
            if (should_split()) split_into_virtual_sub_buckets();
        }
    }

    template <BucketState State>
    void add_sorted_label(Label *label) noexcept {
        if (!label) return;

        if constexpr (State == BucketState::Split) {
            if (labels_vec.empty()) {
                is_split = false;
                add_sorted_label<BucketState::Unsplit>(label);
                return;
            }

            const bool goes_to_first = label->resources[RESOURCE_INDEX] <= split_midpoint;
            ensure_capacity();
            const size_t start = goes_to_first ? 0 : split_point;
            const size_t end   = goes_to_first ? split_point : labels_vec.size();

            auto insert_pos = find_insert_position(label, start, end);
            labels_vec.insert(insert_pos, label);
            if (goes_to_first) split_point++;
        } else {
            ensure_capacity();
            if (labels_vec.empty()) {
                labels_vec.push_back(label);
                return;
            }

            if (label->cost >= labels_vec.back()->cost) {
                labels_vec.push_back(label);
            } else if (label->cost <= labels_vec.front()->cost) {
                labels_vec.insert(labels_vec.begin(), label);
            } else {
                auto insert_pos = find_insert_position(label, 0, labels_vec.size());
                labels_vec.insert(insert_pos, label);
            }

            if (should_split()) {
                split_into_virtual_sub_buckets();
            }
        }
    }

    template <BucketState State>
    void remove_labels_batch(const std::vector<Label *> &to_remove) noexcept {
        if (labels_vec.empty() || to_remove.empty()) return;

        if constexpr (State == BucketState::Split) {
            auto [bucket0_removes, bucket1_removes] = partition_removals(to_remove);
            if (!bucket0_removes.empty()) { remove_from_virtual_bucket(bucket0_removes, 0, split_point); }
            if (!bucket1_removes.empty()) {
                remove_from_virtual_bucket(bucket1_removes, split_point, labels_vec.size());
            }
        } else {
#ifdef SORTED_LABELS
            remove_batch_sorted_range(to_remove, 0, labels_vec.size());
#else
            to_remove.size() > SMALL_BATCH_THRESHOLD ? remove_batch_hash_range(to_remove, 0, labels_vec.size())
                                                     : remove_batch_linear_range(to_remove, 0, labels_vec.size());
#endif
        }
    }

    template <BucketState State>
    Label *get_best_label() noexcept {
        if constexpr (State == BucketState::Split) {
            Label *first_best = split_point > 0 ? get_min_label(0, split_point) : nullptr;
            Label *second_best =
                split_point < labels_vec.size() ? get_min_label(split_point, labels_vec.size()) : nullptr;

            if (!first_best) return second_best;
            if (!second_best) return first_best;
            return (first_best->cost <= second_best->cost) ? first_best : second_best;
        } else {
            if (labels_vec.empty()) return nullptr;
#ifdef SORTED_LABELS
            return labels_vec.front();
#else
            return get_min_label(0, labels_vec.size());
#endif
        }
    }

    template <BucketState State>
    double get_cb() const noexcept {
        if constexpr (State == BucketState::Split) {
            double min_cb = std::numeric_limits<double>::max();
            if (split_point > 0) { min_cb = std::min(min_cb, get_min_cost(0, split_point)); }
            if (split_point < labels_vec.size()) {
                min_cb = std::min(min_cb, get_min_cost(split_point, labels_vec.size()));
            }
            return min_cb;
        } else {
#ifdef SORTED_LABELS
            return labels_vec.front()->cost;
#else
            return get_min_cost(0, labels_vec.size());
#endif
        }
    }

    template <BucketState State>
    bool check_dominance(const Label *new_label, const auto &dominance_func, int &stat_n_dom) const noexcept {
        if constexpr (State == BucketState::Split) {
            if (labels_vec.empty() || split_point >= labels_vec.size()) { return false; }

            std::array<std::pair<size_t, size_t>, 2> ranges;
            size_t                                   num_ranges = 0;

            if (new_label->resources[RESOURCE_INDEX] <= split_midpoint) { ranges[num_ranges++] = {0, split_point}; }
            if (new_label->resources[RESOURCE_INDEX] >= split_midpoint) {
                ranges[num_ranges++] = {split_point, labels_vec.size()};
            }

            for (size_t i = 0; i < num_ranges; ++i) {
                const auto [start, end] = ranges[i];
                std::vector<Label *> sub_labels(labels_vec.begin() + start, labels_vec.begin() + end);
                if (dominance_func(sub_labels)) {
                    ++stat_n_dom;
                    return true;
                }
            }
            return false;
        } else {
            if (dominance_func(labels_vec)) {
                ++stat_n_dom;
                return true;
            }
            return false;
        }
    }

    // Utility functions
    void ensure_capacity() noexcept {
        if (labels_vec.size() == labels_vec.capacity()) {
            labels_vec.reserve(std::max<size_t>(MIN_RESERVE_SIZE, labels_vec.capacity() * 2));
        }
    }

    bool should_split() const noexcept {
        return depth < MAX_BUCKET_DEPTH && labels_vec.size() >= BUCKET_CAPACITY && !is_split;
    }

    double get_min_cost(size_t start, size_t end) const noexcept {
        if (start >= end || end > labels_vec.size() || labels_vec.empty()) {
            return std::numeric_limits<double>::max();
        }

        auto min_it = std::ranges::min_element(labels_vec.begin() + start, labels_vec.begin() + end,
                                               [](const Label *a, const Label *b) {
                                                   if (!a || !b) return false;
                                                   return a->cost < b->cost;
                                               });

        if (min_it == labels_vec.begin() + end || !*min_it) { return std::numeric_limits<double>::max(); }

        return (*min_it)->cost;
    }

    Label *get_min_label(size_t start, size_t end) noexcept {
        if (start >= end || end > labels_vec.size() || labels_vec.empty()) { return nullptr; }

        auto min_it = std::ranges::min_element(labels_vec.begin() + start, labels_vec.begin() + end,
                                               [](const Label *a, const Label *b) {
                                                   if (!a || !b) return false;
                                                   return a->cost < b->cost;
                                               });

        return (min_it != labels_vec.begin() + end && *min_it) ? *min_it : nullptr;
    }
    std::pair<std::vector<Label *>, std::vector<Label *>>
    partition_removals(const std::vector<Label *> &to_remove) const noexcept {
        std::vector<Label *> bucket0_removes, bucket1_removes;
        bucket0_removes.reserve(to_remove.size());
        bucket1_removes.reserve(to_remove.size());

        for (Label *label : to_remove) {
            if (!label) continue;
            if (label->resources[RESOURCE_INDEX] <= split_midpoint) {
                bucket0_removes.push_back(label);
            } else {
                bucket1_removes.push_back(label);
            }
        }
        return {bucket0_removes, bucket1_removes};
    }

    // Removal helpers
    void remove_from_virtual_bucket(const std::vector<Label *> &to_remove, size_t start, size_t end) noexcept {
        if (labels_vec.empty() || to_remove.empty() || start >= end) return;

#ifdef SORTED_LABELS
        remove_batch_sorted_range(to_remove, start, end);
#else
        to_remove.size() > SMALL_BATCH_THRESHOLD ? remove_batch_hash_range(to_remove, start, end)
                                                 : remove_batch_linear_range(to_remove, start, end);
#endif
    }

    void remove_batch_sorted_range(const std::vector<Label *> &to_remove, size_t start, size_t end) noexcept {
        if (start >= labels_vec.size() || end > labels_vec.size() || start >= end) { return; }

        std::vector<bool> should_remove(end - start, false);

        for (Label *label : to_remove) {
            if (!label) continue;
            auto it = std::ranges::lower_bound(labels_vec.begin() + start, labels_vec.begin() + end, label,
                                               [](const Label *a, const Label *b) { return a->cost < b->cost; });
            if (it != labels_vec.begin() + end && *it == label) {
                size_t idx = std::distance(labels_vec.begin() + start, it);
                if (idx < should_remove.size()) { should_remove[idx] = true; }
            }
        }

        // Compact elements in place
        size_t write = start;
        for (size_t read = start; read < end && read < labels_vec.size(); read++) {
            if (read - start >= should_remove.size() || !should_remove[read - start]) {
                if (write != read && write < labels_vec.size()) { labels_vec[write] = labels_vec[read]; }
                write++;
            }
        }

        // Shift remaining elements and resize
        if (write < end) {
            std::move(labels_vec.begin() + end, labels_vec.end(), labels_vec.begin() + write);
            labels_vec.resize(labels_vec.size() - (end - write));
        }
    }

    void remove_batch_hash_range(const std::vector<Label *> &to_remove, size_t start, size_t end) noexcept {
        ankerl::unordered_dense::set<Label *> remove_set(to_remove.begin(), to_remove.end());
        size_t                                write = start;
        for (size_t read = start; read < end; read++) {
            if (!remove_set.contains(labels_vec[read])) {
                if (write != read) { labels_vec[write] = labels_vec[read]; }
                write++;
            }
        }

        if (write < end) {
            std::move(labels_vec.begin() + end, labels_vec.end(), labels_vec.begin() + write);
            labels_vec.resize(labels_vec.size() - (end - write));
        }
    }

    void remove_batch_linear_range(const std::vector<Label *> &to_remove, size_t start, size_t end) noexcept {
        for (Label *label : to_remove) {
            if (!label) continue;
            auto it = std::find(labels_vec.begin() + start, labels_vec.begin() + end, label);
            if (it != labels_vec.begin() + end) {
                *it = std::move(labels_vec[--end]);
                labels_vec.pop_back();
            }
        }
    }

    void split_into_virtual_sub_buckets() noexcept {
        if (labels_vec.empty() || is_split || depth >= MAX_BUCKET_DEPTH) return;

        const double total_range = ub[RESOURCE_INDEX] - lb[RESOURCE_INDEX];
        if (total_range < MIN_SPLIT_RANGE) return;

        const size_t mid_idx = labels_vec.size() / 2;
        split_midpoint       = labels_vec[mid_idx]->resources[RESOURCE_INDEX];

        auto partition_point =
            std::partition(labels_vec.begin(), labels_vec.end(), [this](const Label *label) noexcept {
                return label->resources[RESOURCE_INDEX] <= split_midpoint;
            });

        split_point = std::distance(labels_vec.begin(), partition_point);

        if (split_point == 0 || split_point == labels_vec.size()) return;

        is_split    = true;
        shall_split = true;
    }

    void remove_label_impl(Label *label, int recursion_depth) noexcept {
        if (recursion_depth > MAX_RECURSION_DEPTH) {
            auto it = std::find(labels_vec.begin(), labels_vec.end(), label);
            if (it != labels_vec.end()) labels_vec.erase(it);
            return;
        }

        if (is_split) {
            if (labels_vec.empty() || split_point >= labels_vec.size()) {
                is_split = false;
                remove_label_impl(label, recursion_depth + 1);
                return;
            }

            if (label->resources[RESOURCE_INDEX] <= split_midpoint) {
                auto it = std::find(labels_vec.begin(), labels_vec.begin() + split_point, label);
                if (it != labels_vec.begin() + split_point) {
                    labels_vec.erase(it);
                    split_point--;
                }
            } else {
                auto it = std::find(labels_vec.begin() + split_point, labels_vec.end(), label);
                if (it != labels_vec.end()) { labels_vec.erase(it); }
            }
            return;
        }

        auto it = std::find(labels_vec.begin(), labels_vec.end(), label);
        if (it != labels_vec.end()) labels_vec.erase(it);
    }

    /////////////////////////////////////////////////
public:
    // create default constructor
    Bucket() {}

    inline const std::vector<Label *> &get_labels() const { return labels_vec; }

    // Update flush to use batch removal
    void flush() {
        remove_labels_batch(labels_flush);
        labels_flush.clear(); // Clear after successful removal
    }

    void clear() {
        labels_vec.clear();
        is_split    = false;
        shall_split = false;
        split_point = 0;
    }

    Bucket(const Bucket &other) : labels_vec(other.labels_vec) {
        // Deep copy or other operations, if needed
        // Perform deep copy of all relevant members
        labels_vec     = other.labels_vec;
        node_id        = other.node_id;
        lb             = other.lb;
        ub             = other.ub;
        fw_arcs        = other.fw_arcs;
        bw_arcs        = other.bw_arcs;
        fw_bucket_arcs = other.fw_bucket_arcs;
        bw_bucket_arcs = other.bw_bucket_arcs;
        fw_jump_arcs   = other.fw_jump_arcs;
        bw_jump_arcs   = other.bw_jump_arcs;
    }

    Bucket &operator=(const Bucket &other) {
        if (this == &other) return *this; // Handle self-assignment

        // Perform deep copy of all relevant members
        labels_vec     = other.labels_vec;
        node_id        = other.node_id;
        lb             = other.lb;
        ub             = other.ub;
        fw_arcs        = other.fw_arcs;
        bw_arcs        = other.bw_arcs;
        fw_bucket_arcs = other.fw_bucket_arcs;
        bw_bucket_arcs = other.bw_bucket_arcs;
        fw_jump_arcs   = other.fw_jump_arcs;
        bw_jump_arcs   = other.bw_jump_arcs;

        return *this;
    }

    /**
     * @brief Adds an arc between two buckets.
     *
     * This function adds an arc from one bucket to another, either in the
     * forward or backward direction. The arc is characterized by resource
     * increments, cost increment, and whether it is fixed or not.
     *
     */
    template <Direction D>
    void add_bucket_arc(int from_bucket, int to_bucket, const std::vector<double> &res_inc, double cost_inc,
                        bool fixed) {
        if constexpr (D == Direction::Forward) {
            fw_bucket_arcs.emplace_back(from_bucket, to_bucket, res_inc, cost_inc, fixed);
        } else {
            bw_bucket_arcs.emplace_back(from_bucket, to_bucket, res_inc, cost_inc, fixed);
        }
    }

    /**
     * @brief Adds a jump arc between two buckets.
     *
     * This function adds a jump arc from one bucket to another with the
     * specified resource increment and cost increment. The direction of the
     * jump arc is determined by the `fw` parameter.
     *
     */
    template <Direction D>

    void add_jump_arc(int from_bucket, int to_bucket, const std::vector<double> &res_inc, double cost_inc) {
        if constexpr (D == Direction::Forward) {
            fw_jump_arcs.emplace_back(from_bucket, to_bucket, res_inc, cost_inc);
        } else {
            bw_jump_arcs.emplace_back(from_bucket, to_bucket, res_inc, cost_inc);
        }
    }

    Bucket(int node_id, std::vector<double> lb, std::vector<double> ub)
        : node_id(node_id), lb(std::move(lb)), ub(std::move(ub)) {
        labels_vec.reserve(256);
    }

    /**
     * @brief Retrieves a reference to the vector of arcs based on the
     * specified direction.
     *
     * This function template returns a reference to either the forward arcs
     * (fw_arcs) or the backward arcs (bw_arcs) depending on the template
     * parameter `dir`.
     *
     */
    template <Direction dir>
    std::vector<Arc> &get_arcs() {
        if constexpr (dir == Direction::Forward) {
            return fw_arcs;
        } else {
            return bw_arcs;
        }
    }

    template <Direction dir>
    std::vector<BucketArc> &get_bucket_arcs() {
        if constexpr (dir == Direction::Forward) {
            return fw_bucket_arcs;
        } else {
            return bw_bucket_arcs;
        }
    }

    template <Direction dir>
    std::vector<JumpArc> &get_jump_arcs() {
        if constexpr (dir == Direction::Forward) {
            return fw_jump_arcs;
        } else {
            return bw_jump_arcs;
        }
    }

    [[nodiscard]] bool empty() const { return labels_vec.empty(); }

    ~Bucket() {
        reset(); // Ensure all elements are cleared
        // pool.release(); // Explicitly release the memory pool resources
    }

    /**
     * @brief Clears the arcs in the specified direction.
     *
     * This function clears the arcs in either the forward or backward
     * direction based on the input parameter.
     *
     */
    void clear_arcs(bool fw) {
        if (fw) {
            fw_bucket_arcs.clear();
            fw_jump_arcs.clear();
        } else {
            bw_bucket_arcs.clear();
            bw_jump_arcs.clear();
        }
    }

    // define reset method
    void reset() {
        fw_bucket_arcs.clear();
        bw_bucket_arcs.clear();
        fw_jump_arcs.clear();
        bw_jump_arcs.clear();
        labels_vec.clear();
        split_point = 0;
        is_split    = false;
        shall_split = false;
    }
};
