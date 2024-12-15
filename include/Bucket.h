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
        if (labels_vec.empty() && !is_split) { return std::numeric_limits<double>::max(); }

        if (is_split) {
            double min_cb = std::numeric_limits<double>::max();
            for (const auto &sub_bucket : sub_buckets) { min_cb = std::min(min_cb, sub_bucket.get_cb()); }
            return min_cb;
        }

#ifdef SORTED_LABELS
        return labels_vec.front()->cost;
#else
        return (*std::min_element(labels_vec.begin(), labels_vec.end(),
                                  [](const Label *a, const Label *b) { return a->cost < b->cost; }))
            ->cost;
#endif
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

    double               min_split_range  = 0.5;
    static constexpr int MAX_BUCKET_DEPTH = 4;

    // Dominance check method for Bucket
    bool check_dominance(const Label                                             *new_label,
                         const std::function<bool(const std::vector<Label *> &)> &dominance_func, int &stat_n_dom) {
        // Early return if cost bound exceeds new label's cost
        if (get_cb() > new_label->cost) { return false; }

        // Handle non-split case directly
        if (!is_split) {
            if (dominance_func(get_labels())) {
                ++stat_n_dom;
                return true;
            }
            return false;
        }

        // For small number of sub-buckets, use sequential processing
        constexpr size_t PARALLEL_THRESHOLD = 4;
        if (sub_buckets.size() <= PARALLEL_THRESHOLD) {
            for (auto &bucket : sub_buckets) {
                if (bucket.get_cb() <= new_label->cost &&
                    bucket.check_dominance(new_label, dominance_func, stat_n_dom)) {
                    return true;
                }
            }
            return false;
        }

        // For larger sets, use parallel processing with optimizations
        std::atomic<bool> dominance_found{false};
        std::atomic<int>  local_stat_n_dom{0};

        // Pre-filter buckets that could potentially dominate
        std::vector<Bucket *> candidate_buckets;
        candidate_buckets.reserve(sub_buckets.size());

        for (auto &bucket : sub_buckets) {
            if (bucket.get_cb() <= new_label->cost) { candidate_buckets.push_back(&bucket); }
        }

        // If no candidates, return early
        if (candidate_buckets.empty()) { return false; }

        // Process candidates in parallel
        std::for_each(std::execution::par_unseq, candidate_buckets.begin(), candidate_buckets.end(),
                      [&](Bucket *bucket) {
                          // Skip if dominance already found
                          if (dominance_found.load(std::memory_order_relaxed)) { return; }

                          int local_dom = 0;
                          if (bucket->check_dominance(new_label, dominance_func, local_dom)) {
                              dominance_found.store(true, std::memory_order_release);
                              local_stat_n_dom.fetch_add(local_dom, std::memory_order_relaxed);
                          }
                      });

        // Update stats counter
        stat_n_dom += local_stat_n_dom.load(std::memory_order_relaxed);

        return dominance_found.load(std::memory_order_acquire);
    }

    void split_into_sub_buckets() noexcept {
        // Early returns for invalid states
        if (labels_vec.empty() || is_split || depth >= MAX_BUCKET_DEPTH) { return; }

        constexpr size_t RESOURCE_INDEX = 0;
        const double     total_range    = ub[RESOURCE_INDEX] - lb[RESOURCE_INDEX];

        if (total_range < min_split_range) { return; }

        // Pre-allocate vectors with exact sizes needed
        size_t              label_count = labels_vec.size();
        std::vector<double> label_values;
        label_values.reserve(label_count);

        // Extract resource values
        std::transform(labels_vec.begin(), labels_vec.end(), std::back_inserter(label_values),
                       [](const Label *label) { return label->resources[RESOURCE_INDEX]; });

        // Find median using nth_element
        const size_t mid_idx = label_count / 2;
        std::nth_element(label_values.begin(), label_values.begin() + mid_idx, label_values.end());

        const double midpoint         = roundToTwoDecimalPlaces(label_values[mid_idx]);
        const double clamped_midpoint = std::clamp(midpoint, lb[RESOURCE_INDEX], ub[RESOURCE_INDEX]);

        // Pre-allocate sub-buckets
        sub_buckets.clear();
        sub_buckets.reserve(2);

        // Initialize both sub-buckets at once
        for (int i = 0; i < 2; ++i) {
            sub_buckets.emplace_back();
            auto &bucket           = sub_buckets.back();
            bucket.node_id         = node_id;
            bucket.depth           = depth + 1;
            bucket.min_split_range = min_split_range;

            if (i == 0) {
                bucket.lb = {roundToTwoDecimalPlaces(lb[RESOURCE_INDEX])};
                bucket.ub = {clamped_midpoint};
            } else {
                bucket.lb = {clamped_midpoint};
                bucket.ub = {roundToTwoDecimalPlaces(ub[RESOURCE_INDEX])};
            }
        }

        // Partition labels more efficiently
        auto &first  = sub_buckets[0];
        auto &second = sub_buckets[1];

        // Reserve approximate space based on median
        first.labels_vec.reserve(label_count / 2 + label_count / 10); // Add 10% buffer
        second.labels_vec.reserve(label_count / 2 + label_count / 10);

        // Use stable_partition for better cache utilization
        auto partition_point = std::stable_partition(labels_vec.begin(), labels_vec.end(),
                                                     [RESOURCE_INDEX, clamped_midpoint](const Label *label) {
                                                         return label->resources[RESOURCE_INDEX] <= clamped_midpoint;
                                                     });

        // Move labels to sub-buckets
        first.labels_vec.assign(std::make_move_iterator(labels_vec.begin()), std::make_move_iterator(partition_point));
        second.labels_vec.assign(std::make_move_iterator(partition_point), std::make_move_iterator(labels_vec.end()));

        // Clear parent bucket's labels
        labels_vec.clear();
        labels_vec.shrink_to_fit();
        is_split = true;

#ifdef SORTED_LABELS
        // Maintain sorting in sub-buckets if needed
        // first.sort();
        // second.sort();
        pdqsort(first.labels_vec.begin(), first.labels_vec.end(),
                [](const Label *a, const Label *b) { return a->cost < b->cost; });
        pdqsort(second.labels_vec.begin(), second.labels_vec.end(),
                [](const Label *a, const Label *b) { return a->cost < b->cost; });
#endif
    }

    void add_label(Label *label) noexcept {
        // Early return for null label
        if (!label) { return; }

        if (is_split) {
            // There should always be exactly 2 sub-buckets when split
            assert(sub_buckets.size() == 2);

            // Direct index access is faster than iteration for 2 buckets
            if (sub_buckets[0].is_contained(label)) {
                if (sub_buckets[0].depth >= MAX_BUCKET_DEPTH) {
                    sub_buckets[0].labels_vec.push_back(label);
                } else {
                    sub_buckets[0].add_label(label);
                }
                return;
            }

            if (sub_buckets[1].is_contained(label)) {
                if (sub_buckets[1].depth >= MAX_BUCKET_DEPTH) {
                    sub_buckets[1].labels_vec.push_back(label);
                } else {
                    sub_buckets[1].add_label(label);
                }
                return;
            }

// Handle out-of-bounds case
#ifdef DEBUG
            print_info("Warning: Label {:.2f} outside bucket range [{:.2f}, {:.2f}]\n", label->resources[0], lb[0],
                       ub[0]);
#endif

            // Add to the closest sub-bucket instead of parent
            const double mid_point = (sub_buckets[0].ub[0] + sub_buckets[1].lb[0]) / 2.0;
            if (label->resources[0] <= mid_point) {
                sub_buckets[0].labels_vec.push_back(label);
            } else {
                sub_buckets[1].labels_vec.push_back(label);
            }
            return;
        }

        // Pre-reserve space if needed
        if (labels_vec.size() == labels_vec.capacity()) {
            labels_vec.reserve(std::max(size_t(64), labels_vec.capacity() * 2));
        }

        labels_vec.push_back(label);

        // Check for split condition
        if (depth < MAX_BUCKET_DEPTH && labels_vec.size() >= BUCKET_CAPACITY) { split_into_sub_buckets(); }
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

    void add_sorted_label(Label *label) noexcept {
        // Check for invalid label
        if (!label) return;

        if (is_split) {
            // Find appropriate sub-bucket
            for (auto &bucket : sub_buckets) {
                if (bucket.is_contained(label)) {
                    // Only recurse if we haven't hit maximum depth
                    if (bucket.depth >= MAX_BUCKET_DEPTH) {
                        // Find insertion point even at max depth
                        auto insert_pos =
                            std::lower_bound(bucket.labels_vec.begin(), bucket.labels_vec.end(), label,
                                             [](const Label *a, const Label *b) { return a->cost < b->cost; });
                        bucket.labels_vec.insert(insert_pos, label);
                    } else {
                        bucket.add_sorted_label(label);
                    }
                    return;
                }
            }
            // Fallback: add to parent if label doesn't fit in any sub-bucket
            auto insert_pos = std::lower_bound(labels_vec.begin(), labels_vec.end(), label,
                                               [](const Label *a, const Label *b) { return a->cost < b->cost; });
            labels_vec.insert(insert_pos, label);
            return;
        }

        // Pre-reserve space if needed
        if (labels_vec.size() == labels_vec.capacity()) {
            labels_vec.reserve(std::max(size_t(64), labels_vec.capacity() * 2));
        }

        // Handle empty vector case
        if (labels_vec.empty()) {
            labels_vec.push_back(label);
            return;
        }

        // Fast path for common cases
        if (label->cost >= labels_vec.back()->cost) {
            labels_vec.push_back(label);
            return;
        }

        if (label->cost <= labels_vec.front()->cost) {
            labels_vec.insert(labels_vec.begin(), label);
            return;
        }

        // Binary search for insertion point
        auto insert_pos = std::lower_bound(labels_vec.begin(), labels_vec.end(), label,
                                           [](const Label *a, const Label *b) { return a->cost < b->cost; });
        labels_vec.insert(insert_pos, label);

        // Only split if we're not at max depth and have enough labels
        if (depth < MAX_BUCKET_DEPTH && labels_vec.size() >= BUCKET_CAPACITY) { split_into_sub_buckets(); }
    }

    bool is_contained(const Label *label) const noexcept {
        if (!label) { return false; }

        constexpr size_t RESOURCE_INDEX = 0;
        const double     resource_value = label->resources[RESOURCE_INDEX];

        return resource_value >= lb[RESOURCE_INDEX] && resource_value <= ub[RESOURCE_INDEX];
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
        if (!label) { return; }

        if (!is_split) {
            // For small vectors, linear search is faster than binary search
            if (labels_vec.size() <= 16) {
                auto it = std::find(labels_vec.begin(), labels_vec.end(), label);
                if (it != labels_vec.end()) {
                    *it = std::move(labels_vec.back());
                    labels_vec.pop_back();
                }
                return;
            }

#ifdef SORTED_LABELS
            // Use binary search for larger sorted vectors
            auto it = std::lower_bound(labels_vec.begin(), labels_vec.end(), label,
                                       [](const Label *a, const Label *b) { return a->cost < b->cost; });

            if (it != labels_vec.end() && *it == label) { labels_vec.erase(it); }
#else
            // Use standard find for unsorted
            auto it = std::find(labels_vec.begin(), labels_vec.end(), label);
            if (it != labels_vec.end()) {
                *it = std::move(labels_vec.back());
                labels_vec.pop_back();
            }
#endif
            return;
        }

        // When split, check only the bucket that could contain the label
        constexpr size_t RESOURCE_INDEX = 0;
        const double     resource_value = label->resources[RESOURCE_INDEX];

        // Direct index access is faster than iteration for 2 buckets
        assert(sub_buckets.size() == 2);

        if (resource_value <= sub_buckets[0].ub[RESOURCE_INDEX]) {
            sub_buckets[0].remove_label(label);
        } else {
            sub_buckets[1].remove_label(label);
        }
    }

    mutable std::vector<Label *> all_labels; // Mutable to allow modification in const function

    inline const std::vector<Label *> &get_labels() const {
        if (!is_split) { return labels_vec; }

        // Pre-calculate total size to avoid multiple allocations
        assert(sub_buckets.size() == 2); // We always split into exactly 2 buckets
        const size_t total_size = sub_buckets[0].labels_vec.size() + sub_buckets[1].labels_vec.size();

        all_labels.clear();
        if (all_labels.capacity() < total_size) { all_labels.reserve(total_size); }

        // Direct insertion for each sub-bucket
        // Using sub_bucket.labels_vec directly instead of get_labels() to avoid recursion
        for (const auto &bucket : sub_buckets) {
            if (!bucket.is_split) {
                all_labels.insert(all_labels.end(), bucket.labels_vec.begin(), bucket.labels_vec.end());
            } else {
                // Recursive case - but should be rare due to MAX_BUCKET_DEPTH
                const auto &sub_labels = bucket.get_labels();
                all_labels.insert(all_labels.end(), sub_labels.begin(), sub_labels.end());
            }
        }

#ifdef SORTED_LABELS
        // If labels need to be sorted, sort the combined result
        // std::sort(all_labels.begin(), all_labels.end(),
        // [](const Label *a, const Label *b) { return a->cost < b->cost; });
#endif

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
    Label *get_best_label() noexcept {
        if (!is_split) {
            if (labels_vec.empty()) { return nullptr; }

#ifdef SORTED_LABELS
            return labels_vec[0];
#else
            return *std::min_element(labels_vec.begin(), labels_vec.end(),
                                     [](const Label *a, const Label *b) { return a->cost < b->cost; });
#endif
        }

        // We always split into exactly 2 buckets
        assert(sub_buckets.size() == 2);

        // Get best labels from both buckets
        Label *first_best  = sub_buckets[0].get_best_label();
        Label *second_best = sub_buckets[1].get_best_label();

        // Handle cases where one or both buckets might be empty
        if (!first_best) { return second_best; }
        if (!second_best) { return first_best; }

        // Return the label with lower cost
        return (first_best->cost <= second_best->cost) ? first_best : second_best;
    }

    [[nodiscard]] bool empty() const { return labels_vec.empty(); }

    ~Bucket() {
        reset(); // Ensure all elements are cleared
        // pool.release(); // Explicitly release the memory pool resources
    }
};
