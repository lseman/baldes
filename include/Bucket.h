/*
 * @file Bucket.h
 * @brief Defines the Bucket storage structure for bucket-based labeling.
 *
 */
#pragma once

#include <algorithm>
#include <memory_resource>
#include <span>

#include "Arc.h"
#include "Definitions.h"
#include "Label.h"
#include "NumericUtils.h"
#include "config.h"
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
    // Hot data - frequently accessed (kept in first cache line)
    int    depth{0};
    int    node_id{-1};
    bool   is_split{false};
    double min_cost{std::numeric_limits<double>::max()};
    char   padding[32]; // Ensure alignment

    // Committed labels are kept sorted by reduced cost. Fresh labels are
    // staged separately so insertion stays amortized O(1); they are merged
    // into labels only when a fully sorted view is required.
    mutable std::pmr::vector<Label *> labels;
    mutable std::pmr::vector<Label *> extra_labels;

    // Virtual split: an index that logically partitions the labels vector.
    mutable size_t virtual_split_index = 0;
    mutable bool   is_virtual_split    = false;

    // Cold data - less frequently accessed
    std::vector<double>     lb;
    std::vector<double>     ub;
    std::vector<Arc>        fw_arcs;
    std::vector<Arc>        bw_arcs;
    std::vector<BucketArc>  fw_bucket_arcs;
    std::vector<BucketArc>  bw_bucket_arcs;
    std::vector<JumpArc>    fw_jump_arcs;
    std::vector<JumpArc>    bw_jump_arcs;
    std::array<Bucket *, 2> sub_buckets;
    std::vector<Label *>    labels_flush;
    mutable bool            shall_split = false;

    double get_lb() const { return lb[0]; }
    double get_ub() const { return ub[0]; }

    Bucket(const Bucket &other)                = default;
    Bucket &operator=(const Bucket &other)     = default;
    Bucket(Bucket &&other) noexcept            = default;
    Bucket &operator=(Bucket &&other) noexcept = default;

    double               min_split_range  = 0.5;
    static constexpr int MAX_BUCKET_DEPTH = 1;

    inline void activate_virtual_split_assume_sorted() const noexcept {
        if (labels.size() < 2 || is_virtual_split) return;
        virtual_split_index = labels.size() / 2;
        if (virtual_split_index >= labels.size()) { virtual_split_index = labels.size() - 1; }
        is_virtual_split = true;
    }

    // --- Virtual Splitting ---
    // When the bucket reaches capacity, we virtually split it.
    void virtual_split() noexcept {
        flush_extra_labels();
        if (labels.size() < 2 || is_virtual_split) return;

        // Ensure labels are sorted by cost.
        pdqsort(labels.begin(), labels.end(), [](const Label *a, const Label *b) { return a->cost < b->cost; });
        // Set the virtual split index at the median.
        virtual_split_index = labels.size() / 2;
        if (virtual_split_index >= labels.size()) { virtual_split_index = labels.size() - 1; }
        is_virtual_split = true;
    }

    void flush_extra_labels() {
        if (extra_labels.empty()) return;

        const auto old_size = labels.size();
        labels.insert(labels.end(), extra_labels.begin(), extra_labels.end());
        pdqsort(labels.begin() + static_cast<std::ptrdiff_t>(old_size), labels.end(),
                [](const Label *a, const Label *b) { return a->cost < b->cost; });
        std::inplace_merge(labels.begin(), labels.begin() + static_cast<std::ptrdiff_t>(old_size), labels.end(),
                           [](const Label *a, const Label *b) { return a->cost < b->cost; });
        extra_labels.clear();
        min_cost = std::numeric_limits<double>::max();

        is_virtual_split    = false;
        virtual_split_index = 0;
        if (labels.size() >= BUCKET_CAPACITY) {
            shall_split = true;
            activate_virtual_split_assume_sorted();
        }
    }

    // --- Insertion ---
    // Stage fresh labels in an unsorted tier; sorted access materializes them
    // lazily through flush_extra_labels().
    void add_sorted_label(Label *label) noexcept {
        if (!label) return;

        extra_labels.push_back(label);
        if (label->cost < min_cost) min_cost = label->cost;
        if (labels.size() + extra_labels.size() >= BUCKET_CAPACITY) { shall_split = true; }
    }

    // For cases where sorted order isn’t required on insertion.
    void add_label(Label *label) noexcept { add_sorted_label(label); }

    // --- Retrieval ---
    // Returns the best (i.e. minimum cost) label cost in this bucket.
    double get_cb() const noexcept {
        const double best_sorted = labels.empty() ? std::numeric_limits<double>::max() : labels.front()->cost;
        return std::min(best_sorted, min_cost);
    }

    // Returns the best label pointer.
    Label *get_best_label() noexcept {
        Label *best = labels.empty() ? nullptr : labels.front();
        for (Label *label : extra_labels) {
            if (best == nullptr || label->cost < best->cost) best = label;
        }
        return best;
    }

    std::pmr::vector<Label *> &get_labels() {
        flush_extra_labels();
        return labels;
    }

    const std::pmr::vector<Label *> &get_labels() const { return labels; }

    const std::pmr::vector<Label *> &get_sorted_labels() const noexcept { return labels; }
    const std::pmr::vector<Label *> &get_extra_labels() const noexcept { return extra_labels; }

    [[nodiscard]] size_t size() const noexcept { return labels.size() + extra_labels.size(); }

    // --- Dominance Check ---
    // Checks whether a new label is dominated by any labels already in the
    // bucket. The dominance_func is a lambda or function that performs the
    // actual check on a given set of labels.
    template <typename SortedDominanceFunc, typename ExtraDominanceFunc>
    bool check_dominance(const Label *new_label, SortedDominanceFunc &&sorted_dominance_func,
                         ExtraDominanceFunc &&extra_dominance_func, uint &stat_n_dom) const noexcept {
        if (!new_label) return false;
        if (labels.empty() && extra_labels.empty()) return false;
        // Early exit: if the bucket's best cost is higher than the new label's
        // cost, it cannot be dominated.
        if (get_cb() > new_label->cost) return false;

        if (!labels.empty() && !is_virtual_split) {
            if (sorted_dominance_func(std::span<Label *const>(labels.data(), labels.size()), stat_n_dom)) {
                return true;
            }
        } else if (!labels.empty()) {
            // For a virtual split, check each half separately without copying.
            auto        *label_ptr = labels.data();
            const size_t split     = std::min(virtual_split_index, labels.size());
            if (split == 0 || split >= labels.size()) {
                if (sorted_dominance_func(std::span<Label *const>(label_ptr, labels.size()), stat_n_dom)) {
                    return true;
                }
            } else {
                if (sorted_dominance_func(std::span<Label *const>(label_ptr, split), stat_n_dom)) { return true; }
                if (sorted_dominance_func(std::span<Label *const>(label_ptr + split, labels.size() - split),
                                          stat_n_dom)) {
                    return true;
                }
            }
        }
        if (extra_labels.empty()) return false;
        return extra_dominance_func(std::span<Label *const>(extra_labels.data(), extra_labels.size()), stat_n_dom);
    }

    bool is_empty() const noexcept { return labels.empty() && extra_labels.empty(); }

    void sort() {
        flush_extra_labels();
        pdqsort(labels.begin(), labels.end(), [](const Label *a, const Label *b) { return a->cost < b->cost; });
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

    template <Direction D>
    void remove_bucket_arc(int from_bucket, int to_bucket) {
        if constexpr (D == Direction::Forward) {
            auto it = std::ranges::find_if(fw_bucket_arcs, [&](const BucketArc &arc) {
                return arc.from_bucket == from_bucket && arc.to_bucket == to_bucket && arc.jump == false;
            });
            if (it != fw_bucket_arcs.end()) { fw_bucket_arcs.erase(it); }
        } else {
            auto it = std::ranges::find_if(bw_bucket_arcs, [&](const BucketArc &arc) {
                return arc.from_bucket == from_bucket && arc.to_bucket == to_bucket && arc.jump == false;
            });
            if (it != bw_bucket_arcs.end()) { bw_bucket_arcs.erase(it); }
        }
    }

    template <Direction dir>
    std::vector<BucketArc> get_jump_arcs() {
        std::vector<BucketArc> jump_arcs;
        if constexpr (dir == Direction::Forward) {
            // populate jump_arcs with fw_bucket_arcs in which jump is true
            for (const auto &arc : fw_bucket_arcs) {
                if (arc.jump) { jump_arcs.push_back(arc); }
            }
        } else {
            // populate jump_arcs with bw_bucket_arcs in which jump is true
            for (const auto &arc : bw_bucket_arcs) {
                if (arc.jump) { jump_arcs.push_back(arc); }
            }
        }
        return jump_arcs;
    }

    template <Direction dir>
    std::vector<BucketArc> &get_bucket_arcs() {
        if constexpr (dir == Direction::Forward) {
            return fw_bucket_arcs;
        } else {
            return bw_bucket_arcs;
        }
    }

    Bucket(int node_id, std::vector<double> lb, std::vector<double> ub)
        : node_id(node_id), lb(std::move(lb)), ub(std::move(ub)) {
        labels.reserve(256);
        extra_labels.reserve(64);
    }

    // create default constructor
    Bucket() {}

    bool contains(const std::vector<double> &resource_values_vec) const noexcept {
        for (size_t i = 0; i < resource_values_vec.size(); ++i) {
            if (numericutils::lt(resource_values_vec[i], lb[i]) || numericutils::gt(resource_values_vec[i], ub[i])) {
                return false;
            }
        }
        return true;
    }

    // Returns non-dominated labels.
    // --- Utility Methods ---
    // Returns all non-dominated labels from the bucket.
    std::vector<Label *> get_non_dominated_labels() const noexcept {
        std::vector<Label *> result;
        result.reserve(labels.size() + extra_labels.size());
        for (Label *l : labels) {
            if (!l->is_dominated) result.push_back(l);
        }
        for (Label *l : extra_labels) {
            if (!l->is_dominated) result.push_back(l);
        }
        return result;
    }

    constexpr static const auto is_unextended = [](Label *label) { return !label->is_extended; };

    void clear() {
        sub_buckets      = {};
        is_split         = false;
        shall_split      = false;
        is_virtual_split = false;
        labels.clear();
        extra_labels.clear();
        virtual_split_index = 0;
        min_cost            = std::numeric_limits<double>::max();
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
        sub_buckets = {};
        labels.clear();
        extra_labels.clear();
        min_cost = std::numeric_limits<double>::max();

        is_split            = false;
        shall_split         = false;
        is_virtual_split    = false;
        virtual_split_index = 0;
    }

    [[nodiscard]] bool empty() const { return labels.empty() && extra_labels.empty(); }

    template <Direction D>
    void add_jump_arc(int from_bucket, int to_bucket, const std::vector<double> &res_inc, double cost_inc) {
        if constexpr (D == Direction::Forward) {
            fw_jump_arcs.emplace_back(from_bucket, to_bucket, res_inc, cost_inc);
        } else {
            bw_jump_arcs.emplace_back(from_bucket, to_bucket, res_inc, cost_inc);
        }
    }

    ~Bucket() = default;
};
