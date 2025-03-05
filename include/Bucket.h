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

#include <memory_resource>

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
    int depth{0};
    int node_id{-1};
    bool is_split{false};
    double min_cost{std::numeric_limits<double>::max()};
    char padding[32];  // Ensure alignment

    // Container for labels; this will always be kept sorted by cost.
    std::pmr::vector<Label *> labels;

    // Virtual split: an index that logically partitions the labels vector.
    size_t virtual_split_index = 0;
    bool is_virtual_split = false;

    // Cold data - less frequently accessed
    std::vector<double> lb;
    std::vector<double> ub;
    std::vector<Arc> fw_arcs;
    std::vector<Arc> bw_arcs;
    std::vector<BucketArc> fw_bucket_arcs;
    std::vector<BucketArc> bw_bucket_arcs;
    std::vector<JumpArc> fw_jump_arcs;
    std::vector<JumpArc> bw_jump_arcs;
    std::array<Bucket *, 2> sub_buckets;
    std::vector<Label *> labels_flush;
    bool shall_split = false;

    double get_lb() const { return lb[0]; }
    double get_ub() const { return ub[0]; }

    Bucket(const Bucket &other) : labels(other.labels) {
        // Deep copy or other operations, if needed
        // Perform deep copy of all relevant members
        labels = other.labels;
        node_id = other.node_id;
        lb = other.lb;
        ub = other.ub;
        fw_arcs = other.fw_arcs;
        bw_arcs = other.bw_arcs;
        fw_bucket_arcs = other.fw_bucket_arcs;
        bw_bucket_arcs = other.bw_bucket_arcs;
        fw_jump_arcs = other.fw_jump_arcs;
        bw_jump_arcs = other.bw_jump_arcs;
    }

    Bucket &operator=(const Bucket &other) {
        if (this == &other) return *this;  // Handle self-assignment

        // Perform deep copy of all relevant members
        node_id = other.node_id;
        lb = other.lb;
        ub = other.ub;
        fw_arcs = other.fw_arcs;
        bw_arcs = other.bw_arcs;
        fw_bucket_arcs = other.fw_bucket_arcs;
        bw_bucket_arcs = other.bw_bucket_arcs;
        fw_jump_arcs = other.fw_jump_arcs;
        bw_jump_arcs = other.bw_jump_arcs;
        labels = other.labels;

        return *this;
    }

    double min_split_range = 0.5;
    static constexpr int MAX_BUCKET_DEPTH = 1;

    // --- Virtual Splitting ---
    // When the bucket reaches capacity, we virtually split it.
    void virtual_split() noexcept {
        // if (labels.empty() || is_virtual_split) return;
        // if (depth >= MAX_BUCKET_DEPTH) return;
        // Optionally, you could check whether the resource range is large
        // enough.
        // if (ub[0] - lb[0] < min_split_range) return;

        // Ensure labels are sorted by cost.
        pdqsort(
            labels.begin(), labels.end(),
            [](const Label *a, const Label *b) { return a->cost < b->cost; });
        // Set the virtual split index at the median.
        virtual_split_index = labels.size() / 2;
        is_virtual_split = true;
    }

    // --- Insertion ---
    // Adds a label into the bucket in sorted order.
    // This method will update the virtual split when needed.
    void add_sorted_label(Label *label) noexcept {
        if (!label) return;

        // Find the correct insertion point using binary search.
        auto it = std::lower_bound(
            labels.begin(), labels.end(), label,
            [](const Label *a, const Label *b) { return a->cost < b->cost; });

        // Calculate the position before inserting.
        size_t pos = std::distance(labels.begin(), it);
        labels.insert(it, label);

        if (is_virtual_split) {
            // If the new label is inserted before or at the split point,
            // then shift the virtual split index to maintain the median.
            if (pos <= virtual_split_index) {
                ++virtual_split_index;
            }
        } else {
            // Not split yet: Check if we have reached capacity to trigger a
            // split.
            if (labels.size() >= BUCKET_CAPACITY && !is_virtual_split)
                virtual_split();
        }
    }

    // For cases where sorted order isn’t required on insertion.
    void add_label(Label *label) noexcept {
        if (!label) return;

        if (is_virtual_split) {
            // Since the bucket is virtually split, we assume labels are kept
            // sorted. If the new label's cost is less than or equal to the
            // label at the split, insert it into the lower half.
            if (label->cost <= labels[virtual_split_index]->cost) {
                labels.insert(labels.begin() + virtual_split_index, label);
                ++virtual_split_index;  // Adjust split index as lower half
                                        // grows.
            } else {
                // Otherwise, append to the upper half.
                labels.push_back(label);
            }
        } else {
            // When not split, simply push back.
            labels.push_back(label);
            // Trigger virtual split if capacity is reached.
            if (labels.size() >= BUCKET_CAPACITY && depth < MAX_BUCKET_DEPTH)
                virtual_split();
        }
    }

    // --- Retrieval ---
    // Returns the best (i.e. minimum cost) label cost in this bucket.
    double get_cb() const noexcept {
        if (labels.empty()) return std::numeric_limits<double>::max();
        // Since labels are kept sorted, the first element has the minimal cost.
        return labels.front()->cost;
    }

    // Returns the best label pointer.
    Label *get_best_label() noexcept {
        if (labels.empty()) return nullptr;
        return labels.front();
    }

    std::pmr::vector<Label *> &get_labels() noexcept { return labels; }

    // --- Dominance Check ---
    // Checks whether a new label is dominated by any labels already in the
    // bucket. The dominance_func is a lambda or function that performs the
    // actual check on a given set of labels.
    bool check_dominance(
        const Label *new_label,
        const std::function<bool(const std::pmr::vector<Label *> &, uint &)>
            &dominance_func,
        uint &stat_n_dom) const noexcept {
        if (!new_label) return false;
        // Early exit: if the bucket's best cost is higher than the new label's
        // cost, it cannot be dominated.
        if (get_cb() > new_label->cost) return false;
        // print stat_n_dom

        if (!is_virtual_split) {
            return dominance_func(labels, stat_n_dom);
        } else {
            // For a virtual split, check each half separately.
            std::pmr::vector<Label *> first_half(
                labels.begin(), labels.begin() + virtual_split_index);
            if (dominance_func(first_half, stat_n_dom)) return true;
            std::pmr::vector<Label *> second_half(
                labels.begin() + virtual_split_index, labels.end());
            return dominance_func(second_half, stat_n_dom);
        }
    }

    bool is_empty() const noexcept { return labels.empty(); }

    void sort() {
        pdqsort(
            labels.begin(), labels.end(),
            [](const Label *a, const Label *b) { return a->cost < b->cost; });
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
    void add_bucket_arc(int from_bucket, int to_bucket,
                        const std::vector<double> &res_inc, double cost_inc,
                        bool fixed) {
        if constexpr (D == Direction::Forward) {
            fw_bucket_arcs.emplace_back(from_bucket, to_bucket, res_inc,
                                        cost_inc, fixed);
        } else {
            bw_bucket_arcs.emplace_back(from_bucket, to_bucket, res_inc,
                                        cost_inc, fixed);
        }
    }

    template <Direction D>
    void remove_bucket_arc(int from_bucket, int to_bucket) {
        if constexpr (D == Direction::Forward) {
            auto it =
                std::ranges::find_if(fw_bucket_arcs, [&](const BucketArc &arc) {
                    return arc.from_bucket == from_bucket &&
                           arc.to_bucket == to_bucket && arc.jump == false;
                });
            if (it != fw_bucket_arcs.end()) {
                fw_bucket_arcs.erase(it);
            }
        } else {
            auto it =
                std::ranges::find_if(bw_bucket_arcs, [&](const BucketArc &arc) {
                    return arc.from_bucket == from_bucket &&
                           arc.to_bucket == to_bucket && arc.jump == false;
                });
            if (it != bw_bucket_arcs.end()) {
                bw_bucket_arcs.erase(it);
            }
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
    }

    // create default constructor
    Bucket() {}

    bool contains(std::vector<double> &resource_values_vec) const noexcept {
        for (size_t i = 0; i < resource_values_vec.size(); ++i) {
            if (numericutils::lt(resource_values_vec[i], lb[i]) ||
                numericutils::gt(resource_values_vec[i], ub[i])) {
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
        for (Label *l : labels) {
            if (!l->is_dominated) result.push_back(l);
        }
        return result;
    }

    constexpr static const auto is_unextended = [](Label *label) {
        return !label->is_extended;
    };

    void clear() {
        sub_buckets = {};
        is_split = false;
        shall_split = false;
        is_virtual_split = false;
        labels.clear();
        virtual_split_index = 0;
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

        is_split = false;
        shall_split = false;
        is_virtual_split = false;
        virtual_split_index = 0;
    }

    [[nodiscard]] bool empty() const { return labels.empty(); }

    template <Direction D>
    void add_jump_arc(int from_bucket, int to_bucket,
                      const std::vector<double> &res_inc, double cost_inc) {
        if constexpr (D == Direction::Forward) {
            fw_jump_arcs.emplace_back(from_bucket, to_bucket, res_inc,
                                      cost_inc);
        } else {
            bw_jump_arcs.emplace_back(from_bucket, to_bucket, res_inc,
                                      cost_inc);
        }
    }

    ~Bucket() {
        reset();  // Ensure all elements are cleared
        // pool.release(); // Explicitly release the memory pool resources
    }
};
