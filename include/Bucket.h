/*
 * @file Bucket.h
 * @brief This file contains the definition of the Bucket struct.
 *
 * This file contains the definition of the Bucket struct, which represents a bucket in a solver.
 * The Bucket struct contains information about the node ID, lower bounds, upper bounds, forward arcs,
 * backward arcs, forward jump arcs, backward jump arcs, and labels associated with the bucket.
 * It provides methods to add arcs, add jump arcs, get arcs, get jump arcs, add labels, remove labels,
 * get labels, clear labels, reset labels, and clear arcs.
 *
 */
#pragma once

#include "Definitions.h"

#include "Arc.h"
#include "Label.h"
#include "config.h"
/**
 * @struct Bucket
 * @brief Represents a bucket.
 *
 * A bucket is a data structure that contains labels, node ID, lower bounds, upper bounds, forward arcs, backward
 * arcs, forward jump arcs, and backward jump arcs. It provides methods to add arcs, add jump arcs, get arcs,
 * get jump arcs, add labels, remove labels, get labels, clear labels, reset labels, and clear arcs.
 */
struct Bucket {
    std::vector<Label *> labels_vec;

    int depth = 0;

    int                    node_id = -1;
    std::vector<double>    lb;
    std::vector<double>    ub;
    std::vector<Arc>       fw_arcs;
    std::vector<Arc>       bw_arcs;
    std::vector<BucketArc> fw_bucket_arcs;
    std::vector<BucketArc> bw_bucket_arcs;
    std::vector<JumpArc>   fw_jump_arcs;
    std::vector<JumpArc>   bw_jump_arcs;
    std::vector<Bucket>    sub_buckets; // Holds sub-buckets within the "mother bucket"
    bool                   is_split = false;

    double get_cb() const {
        if (is_split) {
            double min_cb = std::numeric_limits<double>::max();
            for (const auto &sub_bucket : sub_buckets) { min_cb = std::min(min_cb, sub_bucket.get_cb()); }
            return min_cb;
        } else {
#ifdef SORTED_LABELS
            return labels_vec.empty() ? std::numeric_limits<double>::max() : labels_vec.front()->cost;
#else
            if (labels_vec.empty()) return std::numeric_limits<double>::max();

            return (*std::min_element(labels_vec.begin(), labels_vec.end(),
                                      [](const Label *a, const Label *b) { return a->cost < b->cost; }))
                ->cost;
#endif
        }
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
     * @brief Deletes a bucket arc from the specified direction.
     *
     * This function removes a bucket arc from either the forward or backward bucket arcs list,
     * depending on the value of the `fw` parameter. The arc to be removed is identified by the
     * `from_bucket` and `to_bucket` parameters.
     *
     */
    void delete_bucket_arc(int from_bucket, int to_bucket, bool fw) {
        if (fw) {
            fw_bucket_arcs.erase(std::remove_if(fw_bucket_arcs.begin(), fw_bucket_arcs.end(),
                                                [from_bucket, to_bucket](const BucketArc &arc) {
                                                    return arc.from_bucket == from_bucket && arc.to_bucket == to_bucket;
                                                }),
                                 fw_bucket_arcs.end());
        } else {
            bw_bucket_arcs.erase(std::remove_if(bw_bucket_arcs.begin(), bw_bucket_arcs.end(),
                                                [from_bucket, to_bucket](const BucketArc &arc) {
                                                    return arc.from_bucket == from_bucket && arc.to_bucket == to_bucket;
                                                }),
                                 bw_bucket_arcs.end());
        }
    }

    // Dominance check method for Bucket
    bool check_dominance(const Label                                             *new_label,
                         const std::function<bool(const std::vector<Label *> &)> &dominance_func, int &stat_n_dom) {
        if (!is_split) {
            if (get_cb() > new_label->cost) return false;
            if (dominance_func(get_labels())) {
                ++stat_n_dom;
                return true;
            }
            return false;
        }

        std::atomic<bool> dominance_found{false};
        std::for_each(std::execution::par_unseq, sub_buckets.begin(), sub_buckets.end(), [&](Bucket &bucket) {
            if (dominance_found || bucket.get_cb() > new_label->cost) return;
            if (bucket.check_dominance(new_label, dominance_func, stat_n_dom)) { dominance_found = true; }
        });

        return dominance_found;
    }

    double min_split_range = 0.5;
    void   split_into_sub_buckets() noexcept {
        if (labels_vec.empty()) return;

        // If range is too small, don't split
        double total_range = ub[0] - lb[0];
        if (total_range < min_split_range) return;

        if (is_split) return;
        

        // Sort labels by their values
        std::vector<double> label_values;
        label_values.reserve(labels_vec.size());
        if (label_values.empty()) { return; }

        is_split = true;
        for (const auto *label : labels_vec) { label_values.push_back(label->resources[0]); }
        std::sort(label_values.begin(), label_values.end());

        // Find median value
        size_t mid_idx = label_values.size() / 2;
        double midpoint;
        if (label_values.size() % 2 == 0) {
            // If even number of labels, take average of middle two
            midpoint = roundToTwoDecimalPlaces((label_values[mid_idx - 1] + label_values[mid_idx]) / 2.0);
        } else {
            // If odd number of labels, take middle value
            midpoint = roundToTwoDecimalPlaces(label_values[mid_idx]);
        }

        // Ensure midpoint is within bucket bounds
        midpoint = std::max(midpoint, lb[0]);
        midpoint = std::min(midpoint, ub[0]);

        sub_buckets.clear();
        sub_buckets.reserve(2);

        // Create first sub-bucket
        Bucket first_sub_bucket;
        first_sub_bucket.node_id = node_id;
        first_sub_bucket.lb      = {roundToTwoDecimalPlaces(lb[0])};
        first_sub_bucket.ub      = {midpoint};
        first_sub_bucket.depth   = depth + 1;
        sub_buckets.push_back(first_sub_bucket);

        // Create second sub-bucket
        Bucket second_sub_bucket;
        second_sub_bucket.node_id = node_id;
        second_sub_bucket.lb      = {midpoint};
        second_sub_bucket.ub      = {roundToTwoDecimalPlaces(ub[0])};
        second_sub_bucket.depth   = depth + 1;
        sub_buckets.push_back(second_sub_bucket);

        // Assign labels to sub-buckets
        for (auto *label : labels_vec) { assign_label_to_sub_bucket(label); }
        labels_vec.clear();
    }
    void sort() {
        pdqsort(labels_vec.begin(), labels_vec.end(), [](const Label *a, const Label *b) { return a->cost < b->cost; });
    }

    void assign_label_to_sub_bucket(Label *label) noexcept {
        for (auto &bucket : sub_buckets) {
            if (bucket.is_contained(label)) {
                bucket.add_label(label);
                return;
            }
        }

        print_info("Warning: Label {:.2f} outside bucket range [{:.2f}, {:.2f}]\n", label->resources[0], lb[0], ub[0]);
    }
    /**
     * @brief Adds an arc between two buckets.
     *
     * This function adds an arc from one bucket to another, either in the forward or backward direction.
     * The arc is characterized by resource increments, cost increment, and whether it is fixed or not.
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
     * This function adds a jump arc from one bucket to another with the specified resource increment and cost
     * increment. The direction of the jump arc is determined by the `fw` parameter.
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

    /**
     * @brief Retrieves a reference to the vector of arcs based on the specified direction.
     *
     * This function template returns a reference to either the forward arcs (fw_arcs)
     * or the backward arcs (bw_arcs) depending on the template parameter `dir`.
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

    Bucket(int node_id, std::vector<double> lb, std::vector<double> ub)
        : node_id(node_id), lb(std::move(lb)), ub(std::move(ub)) {
        labels_vec.reserve(256);
    }

    // create default constructor
    Bucket() {}

    /**
     * @brief Adds a label to the labels vector.
     *
     * This function adds a label to the labels vector. The label is currently added to the end of the vector.
     *
     */
    // void add_label(Label *label) noexcept { labels_vec.push_back(label); }
    void add_label(Label *label) noexcept {
        if (is_split) {
            assign_label_to_sub_bucket(label);
            return;
        }

        labels_vec.push_back(label);
        if (labels_vec.size() >= BUCKET_CAPACITY) { split_into_sub_buckets(); }
    }

    void add_sorted_label(Label *label) noexcept {
        if (is_split) {
            for (auto &bucket : sub_buckets) {
                if (bucket.is_contained(label)) {
                    bucket.add_sorted_label(label);
                    return;
                }
            }
        }

        // Insert into current level's labels_vec
        if (labels_vec.empty() || label->cost >= labels_vec.back()->cost) {
            labels_vec.push_back(label);
        } else if (label->cost <= labels_vec.front()->cost) {
            labels_vec.insert(labels_vec.begin(), label);
        } else {
            auto it = std::lower_bound(labels_vec.begin(), labels_vec.end(), label,
                                       [](const Label *a, const Label *b) { return a->cost < b->cost; });
            labels_vec.insert(it, label);
        }

        if (!is_split && labels_vec.size() >= BUCKET_CAPACITY) { split_into_sub_buckets(); }
    }

    bool is_contained(const Label *label) {
        // Check if each resource in label is within the bounds of sub_bucket
        for (size_t i = 0; i < 1; ++i) {
            if (label->resources[i] < lb[i] || label->resources[i] > ub[i]) {
                return false; // Resource is out of bounds
            }
        }
        return true; // All resources are within bounds
    }

    /**
     * @brief Adds a label to the labels vector with a limit on the number of labels.
     *
     * This function attempts to add a given label to the labels vector. If the vector
     * has not yet reached the specified limit, the label is simply added. If the vector
     * has reached the limit, the function will replace the label with the highest cost
     * if the new label has a lower cost.
     *
     */
    void add_label_lim(Label *label, size_t limit) noexcept {
        if (labels_vec.size() < limit) {
            labels_vec.push_back(label);
        } else {
            auto it = std::max_element(labels_vec.begin(), labels_vec.end(),
                                       [](const Label *a, const Label *b) { return a->cost > b->cost; });
            if (label->cost < (*it)->cost) { *it = label; }
        }
    }

    void add_sorted_with_limit(Label *label, size_t limit) noexcept {
        if (labels_vec.empty() || label->cost >= labels_vec.back()->cost) {
            labels_vec.push_back(label);
        } else if (label->cost <= labels_vec.front()->cost) {
            labels_vec.insert(labels_vec.begin(), label);
        } else {
            auto it = std::lower_bound(labels_vec.begin(), labels_vec.end(), label,
                                       [](const Label *a, const Label *b) { return a->cost < b->cost; });

            // Insert only if within limit or smaller than current max element
            if (labels_vec.size() < limit || label->cost < labels_vec.back()->cost) {
                labels_vec.insert(it, label);

                // Remove last if we exceed the limit
                if (labels_vec.size() > limit) { labels_vec.pop_back(); }
            }
        }
    }

    /**
     * @brief Removes a label from the labels vector.
     *
     * This function searches for the specified label in the labels vector.
     * If found, it replaces the label with the last element in the vector
     * and then removes the last element, effectively removing the specified label.
     *
     */
    void remove_label(Label *label) noexcept {
        if (!is_split) {
            auto it = std::find(labels_vec.begin(), labels_vec.end(), label);
            if (it != labels_vec.end()) {
                *it = std::move(labels_vec.back());
                labels_vec.pop_back();
            }
        } else {
            for (auto &bucket : sub_buckets) { bucket.remove_label(label); }
        }
    }

    mutable std::vector<Label *> all_labels; // Mutable to allow modification in const function

    inline const std::vector<Label *> &get_labels() const {
        if (!is_split) return labels_vec;

        all_labels.clear();
        all_labels.reserve(
            std::accumulate(sub_buckets.begin(), sub_buckets.end(), size_t{0},
                            [](size_t sum, const auto &bucket) { return sum + bucket.get_labels().size(); }));

        for (const auto &bucket : sub_buckets) {
            const auto &sub_labels = bucket.get_labels();
            all_labels.insert(all_labels.end(), sub_labels.begin(), sub_labels.end());
        }
        return all_labels;
    }

    inline auto &get_sorted_labels() {
        pdqsort(labels_vec.begin(), labels_vec.end(), [](const Label *a, const Label *b) { return a->cost < b->cost; });
        return labels_vec;
    }

    constexpr static const auto is_unextended = [](Label *label) { return !label->is_extended; };

    inline auto get_unextended_labels() const {
        // Directly use get_labels() to retrieve the appropriate vector, whether split or not
        const auto &all_labels_ref = get_labels();

        // Return a filtered view of the retrieved labels
        return std::ranges::filter_view{
            std::ranges::views::all(static_cast<const std::vector<Label *> &>(all_labels_ref)), is_unextended};
    }

    void clear() {
        for (auto &sub_bucket : sub_buckets) { sub_bucket.clear(); }
        sub_buckets.clear();
        labels_vec.clear();
        is_split = false;
    }

    /**
     * @brief Clears the arcs in the specified direction.
     *
     * This function clears the arcs in either the forward or backward direction
     * based on the input parameter.
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
        for (auto &sub_bucket : sub_buckets) { sub_bucket.reset(); }
        sub_buckets.clear();
        labels_vec.clear();

        is_split = false;
    }
    /**
     * @brief Retrieves the best label from the labels vector.
     *
     * This function returns the first label in the labels vector if it is not empty.
     * If the vector is empty, it returns a nullptr.
     *
     */
    Label *get_best_label() {
        if (!is_split) {
            if (labels_vec.empty()) return nullptr;
#ifdef SORTED_LABELS
            return labels_vec[0];
#else
            return *std::min_element(labels_vec.begin(), labels_vec.end(),
                                     [](const Label *a, const Label *b) { return a->cost < b->cost; });
#endif
        }

        Label *best     = nullptr;
        float  min_cost = std::numeric_limits<float>::max();

        for (auto &bucket : sub_buckets) {
            if (Label *label = bucket.get_best_label()) {
                if (label->cost < min_cost) {
                    min_cost = label->cost;
                    best     = label;
                }
            }
        }
        return best;
    }

    [[nodiscard]] bool empty() const { return labels_vec.empty(); }

    ~Bucket() {
        reset(); // Ensure all elements are cleared
        // pool.release(); // Explicitly release the memory pool resources
    }
};
