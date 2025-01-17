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
    std::vector<Label *> labels_vec;  // Most accessed member
    int depth{0};
    int node_id{-1};
    bool is_split{false};
    double min_cost{std::numeric_limits<double>::max()};
    char padding[32];  // Ensure alignment

    // Cold data - less frequently accessed
    std::vector<double> lb;
    std::vector<double> ub;
    std::vector<Arc> fw_arcs;
    std::vector<Arc> bw_arcs;
    std::vector<BucketArc> fw_bucket_arcs;
    std::vector<BucketArc> bw_bucket_arcs;
    std::vector<JumpArc> fw_jump_arcs;
    std::vector<JumpArc> bw_jump_arcs;
    std::vector<Bucket> sub_buckets;
    std::vector<Label *> labels_flush;

    double get_cb() const {
        if (labels_vec.empty() && !is_split) {
            return std::numeric_limits<double>::max();
        }

        if (is_split) {
            double min_cb = std::numeric_limits<double>::max();
            for (const auto &sub_bucket : sub_buckets) {
                min_cb = std::min(min_cb, sub_bucket.get_cb());
            }
            return min_cb;
        }

#ifdef SORTED_LABELS
        return labels_vec.front()->cost;
#else
        return (*std::min_element(labels_vec.begin(), labels_vec.end(),
                                  [](const Label *a, const Label *b) {
                                      return a->cost < b->cost;
                                  }))
            ->cost;
#endif
    }

    Bucket(const Bucket &other) : labels_vec(other.labels_vec) {
        // Deep copy or other operations, if needed
        // Perform deep copy of all relevant members
        labels_vec = other.labels_vec;
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
        labels_vec = other.labels_vec;
        node_id = other.node_id;
        lb = other.lb;
        ub = other.ub;
        fw_arcs = other.fw_arcs;
        bw_arcs = other.bw_arcs;
        fw_bucket_arcs = other.fw_bucket_arcs;
        bw_bucket_arcs = other.bw_bucket_arcs;
        fw_jump_arcs = other.fw_jump_arcs;
        bw_jump_arcs = other.bw_jump_arcs;

        return *this;
    }

    /**
     * @brief Deletes a bucket arc from the specified direction.
     *
     * This function removes a bucket arc from either the forward or backward
     * bucket arcs list, depending on the value of the `fw` parameter. The arc
     * to be removed is identified by the `from_bucket` and `to_bucket`
     * parameters.
     *
     */
    void delete_bucket_arc(int from_bucket, int to_bucket, bool fw) {
        auto &arcs =
            fw ? fw_bucket_arcs
               : bw_bucket_arcs;  // Use a reference to avoid code duplication

        // Use std::erase_if (C++20) for cleaner and more efficient removal
        std::erase_if(arcs, [from_bucket, to_bucket](const BucketArc &arc) {
            return arc.from_bucket == from_bucket && arc.to_bucket == to_bucket;
        });
    }

    void lazy_flush(Label *new_label) { labels_flush.push_back(new_label); }

    void flush() {
        // remove from labels_vec all labels in labels_flush
        for (auto label : labels_flush) {
            remove_label(label);
        }
    }

    double min_split_range = 0.5;
    static constexpr int MAX_BUCKET_DEPTH = 4;

    bool check_dominance(
        const Label *new_label,
        const std::function<bool(const std::vector<Label *> &)> &dominance_func,
        int &stat_n_dom) {
        // Early return if cost bound exceeds new label's cost
        if (get_cb() > new_label->cost) {
            return false;
        }

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
                    bucket.check_dominance(new_label, dominance_func,
                                           stat_n_dom)) {
                    return true;
                }
            }
            return false;
        }

        // For larger sets, use parallel processing with optimizations
        std::atomic<bool> dominance_found{false};
        std::atomic<int> local_stat_n_dom{0};

        // Pre-filter buckets that could potentially dominate
        std::vector<Bucket *> candidate_buckets;
        candidate_buckets.reserve(sub_buckets.size());

        for (auto &bucket : sub_buckets) {
            if (bucket.get_cb() <= new_label->cost) {
                candidate_buckets.push_back(&bucket);
            }
        }

        // If no candidates, return early
        if (candidate_buckets.empty()) {
            return false;
        }

        // Process candidates in parallel
        std::for_each(candidate_buckets.begin(), candidate_buckets.end(),
                      [&](Bucket *bucket) {
                          // Skip if dominance already found
                          if (dominance_found.load(std::memory_order_relaxed)) {
                              return;
                          }

                          int local_dom = 0;
                          if (bucket->check_dominance(new_label, dominance_func,
                                                      local_dom)) {
                              dominance_found.store(true,
                                                    std::memory_order_release);
                              local_stat_n_dom.fetch_add(
                                  local_dom, std::memory_order_relaxed);
                          }
                      });

        // Update stats counter
        stat_n_dom += local_stat_n_dom.load(std::memory_order_relaxed);

        return dominance_found.load(std::memory_order_acquire);
    }

    void split_into_sub_buckets() noexcept {
        // Early returns for invalid states
        if (labels_vec.empty() || is_split || depth >= MAX_BUCKET_DEPTH) {
            return;
        }

        constexpr size_t RESOURCE_INDEX = 0;
        const double total_range = ub[RESOURCE_INDEX] - lb[RESOURCE_INDEX];

        if (total_range < min_split_range) {
            return;
        }

        // Pre-allocate vectors with exact sizes needed
        const size_t label_count = labels_vec.size();
        std::vector<double> label_values;
        label_values.reserve(label_count);

        // Extract resource values
        std::transform(labels_vec.begin(), labels_vec.end(),
                       std::back_inserter(label_values),
                       [](const Label *label) {
                           return label->resources[RESOURCE_INDEX];
                       });

        // Find median using nth_element
        const size_t mid_idx = label_count / 2;
        std::nth_element(label_values.begin(), label_values.begin() + mid_idx,
                         label_values.end());

        const double midpoint = roundToTwoDecimalPlaces(label_values[mid_idx]);
        const double clamped_midpoint =
            std::clamp(midpoint, lb[RESOURCE_INDEX], ub[RESOURCE_INDEX]);

        // Pre-allocate sub-buckets
        sub_buckets.clear();
        sub_buckets.reserve(2);

        // Initialize both sub-buckets at once
        for (int i = 0; i < 2; ++i) {
            sub_buckets.emplace_back();
            auto &bucket = sub_buckets.back();
            bucket.node_id = node_id;
            bucket.depth = depth + 1;
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
        auto &first = sub_buckets[0];
        auto &second = sub_buckets[1];

        // Reserve approximate space based on median
        first.labels_vec.reserve(label_count / 2 +
                                 label_count / 10);  // Add 10% buffer
        second.labels_vec.reserve(label_count / 2 + label_count / 10);

        // Use stable_partition for better cache utilization
        auto partition_point = std::stable_partition(
            labels_vec.begin(), labels_vec.end(),
            [RESOURCE_INDEX, clamped_midpoint](const Label *label) {
                return label->resources[RESOURCE_INDEX] <= clamped_midpoint;
            });

        // Move labels to sub-buckets
        first.labels_vec.assign(std::make_move_iterator(labels_vec.begin()),
                                std::make_move_iterator(partition_point));
        second.labels_vec.assign(std::make_move_iterator(partition_point),
                                 std::make_move_iterator(labels_vec.end()));

        // Clear parent bucket's labels
        labels_vec.clear();
        labels_vec.shrink_to_fit();
        is_split = true;

#ifdef SORTED_LABELS
        // Maintain sorting in sub-buckets if needed
        pdqsort(
            first.labels_vec.begin(), first.labels_vec.end(),
            [](const Label *a, const Label *b) { return a->cost < b->cost; });
        pdqsort(
            second.labels_vec.begin(), second.labels_vec.end(),
            [](const Label *a, const Label *b) { return a->cost < b->cost; });
#endif
    }

    void add_label(Label *label) noexcept {
        // Early return for null label
        if (!label) {
            return;
        }

        if (is_split) {
            // There should always be exactly 2 sub-buckets when split
            assert(sub_buckets.size() == 2);

            // Direct index access is faster than iteration for 2 buckets
            auto &first_bucket = sub_buckets[0];
            auto &second_bucket = sub_buckets[1];

            if (first_bucket.is_contained(label)) {
                if (first_bucket.depth >= MAX_BUCKET_DEPTH) {
                    first_bucket.labels_vec.push_back(label);
                } else {
                    first_bucket.add_label(label);
                }
                return;
            }

            if (second_bucket.is_contained(label)) {
                if (second_bucket.depth >= MAX_BUCKET_DEPTH) {
                    second_bucket.labels_vec.push_back(label);
                } else {
                    second_bucket.add_label(label);
                }
                return;
            }

            // Handle out-of-bounds case
#ifdef DEBUG
            print_info(
                "Warning: Label {:.2f} outside bucket range [{:.2f}, "
                "{:.2f}]\n",
                label->resources[0], lb[0], ub[0]);
#endif

            // Add to the closest sub-bucket instead of parent
            const double mid_point =
                (first_bucket.ub[0] + second_bucket.lb[0]) / 2.0;
            if (label->resources[0] <= mid_point) {
                first_bucket.labels_vec.push_back(label);
            } else {
                second_bucket.labels_vec.push_back(label);
            }
            return;
        }

        // Pre-reserve space if needed
        if (labels_vec.size() == labels_vec.capacity()) {
            labels_vec.reserve(std::max(size_t(64), labels_vec.capacity() * 2));
        }

        labels_vec.push_back(label);

        // Check for split condition
        if (depth < MAX_BUCKET_DEPTH && labels_vec.size() >= BUCKET_CAPACITY) {
            split_into_sub_buckets();
        }
    }

    void sort() {
        pdqsort(
            labels_vec.begin(), labels_vec.end(),
            [](const Label *a, const Label *b) { return a->cost < b->cost; });
    }

    void assign_label_to_sub_bucket(Label *label) noexcept {
        // Early return for null label
        if (!label) {
            return;
        }

        // Use a range-based for loop for clarity
        for (auto &bucket : sub_buckets) {
            if (bucket.is_contained(label)) {
                bucket.add_label(label);
                return;
            }
        }

        // Handle out-of-bounds case
#ifdef DEBUG
        print_info(
            "Warning: Label {:.2f} outside bucket range [{:.2f}, {:.2f}]\n",
            label->resources[0], lb[0], ub[0]);
#endif
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

    /**
     * @brief Adds a jump arc between two buckets.
     *
     * This function adds a jump arc from one bucket to another with the
     * specified resource increment and cost increment. The direction of the
     * jump arc is determined by the `fw` parameter.
     *
     */
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

    Bucket(int node_id, std::vector<double> lb, std::vector<double> ub)
        : node_id(node_id), lb(std::move(lb)), ub(std::move(ub)) {
        labels_vec.reserve(256);
    }

    // create default constructor
    Bucket() {}

    void add_sorted_label(Label *label) noexcept {
        if (!label) return;

        // Split bucket case
        if (is_split) {
            // Try to add to appropriate sub-bucket
            for (auto &bucket : sub_buckets) {
                if (bucket.is_contained(label)) {
                    if (bucket.depth >= MAX_BUCKET_DEPTH) {
                        // At max depth, insert directly into bucket's
                        // vector
                        insert_sorted_label(bucket.labels_vec, label);
                    } else {
                        bucket.add_sorted_label(label);
                    }
                    return;
                }
            }
            // Fallback: add to parent bucket if label doesn't fit in
            // sub-buckets
            insert_sorted_label(labels_vec, label);
            return;
        }

        // Non-split bucket case
        // Pre-reserve space if needed
        if (labels_vec.size() == labels_vec.capacity()) {
            labels_vec.reserve(std::max<size_t>(64, labels_vec.capacity() * 2));
        }

        // Handle empty vector case
        if (labels_vec.empty()) {
            labels_vec.push_back(label);
            return;
        }

        // Fast path checks
        if (label->cost >= labels_vec.back()->cost) {
            labels_vec.push_back(label);
            return;
        }
        if (label->cost <= labels_vec.front()->cost) {
            labels_vec.insert(labels_vec.begin(), label);
            return;
        }

        // Standard case: binary search and insert
        insert_sorted_label(labels_vec, label);

        // Check if we need to split
        if (depth < MAX_BUCKET_DEPTH && labels_vec.size() >= BUCKET_CAPACITY) {
            split_into_sub_buckets();
        }
    }

    // Helper function to avoid code duplication
    static void insert_sorted_label(std::vector<Label *> &vec,
                                    Label *label) noexcept {
        auto insert_pos = std::lower_bound(
            vec.begin(), vec.end(), label,
            [](const Label *a, const Label *b) { return a->cost < b->cost; });
        vec.insert(insert_pos, label);
    }

    bool is_contained(const Label *label) const noexcept {
        if (!label) {
            return false;
        }

        constexpr size_t RESOURCE_INDEX = 0;
        const double resource_value = label->resources[RESOURCE_INDEX];

        // Directly check if the resource value is within the bounds
        return resource_value >= lb[RESOURCE_INDEX] &&
               resource_value <= ub[RESOURCE_INDEX];
    }
    /**
     * @brief Adds a label to the labels vector with a limit on the number
     * of labels.
     *
     * This function attempts to add a given label to the labels vector. If
     * the vector has not yet reached the specified limit, the label is
     * simply added. If the vector has reached the limit, the function will
     * replace the label with the highest cost if the new label has a lower
     * cost.
     *
     */
    void add_label_lim(Label *label, size_t limit) noexcept {
        if (labels_vec.size() < limit) {
            labels_vec.push_back(label);
        } else {
            auto it = std::max_element(labels_vec.begin(), labels_vec.end(),
                                       [](const Label *a, const Label *b) {
                                           return a->cost > b->cost;
                                       });
            if (label->cost < (*it)->cost) {
                *it = label;
            }
        }
    }

    void add_sorted_with_limit(Label *label, size_t limit) noexcept {
        if (labels_vec.empty() || label->cost >= labels_vec.back()->cost) {
            labels_vec.push_back(label);
        } else if (label->cost <= labels_vec.front()->cost) {
            labels_vec.insert(labels_vec.begin(), label);
        } else {
            auto it =
                std::lower_bound(labels_vec.begin(), labels_vec.end(), label,
                                 [](const Label *a, const Label *b) {
                                     return a->cost < b->cost;
                                 });

            // Insert only if within limit or smaller than current max
            // element
            if (labels_vec.size() < limit ||
                label->cost < labels_vec.back()->cost) {
                labels_vec.insert(it, label);

                // Remove last if we exceed the limit
                if (labels_vec.size() > limit) {
                    labels_vec.pop_back();
                }
            }
        }
    }

    /**
     * @brief Removes a label from the labels vector.
     *
     * This function searches for the specified label in the labels vector.
     * If found, it replaces the label with the last element in the vector
     * and then removes the last element, effectively removing the specified
     * label.
     *
     */
    void remove_label(Label *label) noexcept {
        if (!label) return;

        if (is_split) {
            // When split, directly access appropriate bucket based on resource
            // value
            constexpr size_t RESOURCE_INDEX = 0;
            sub_buckets[label->resources[RESOURCE_INDEX] >
                        sub_buckets[0].ub[RESOURCE_INDEX]]
                .remove_label(label);
            return;
        }

        // For unsplit case, use a single optimized removal approach
        auto &vec = labels_vec;
        auto it = vec.size() <= 16
                      ? std::find(vec.begin(), vec.end(), label)
                      :
#ifdef SORTED_LABELS
                      std::lower_bound(vec.begin(), vec.end(), label,
                                       [](const Label *a, const Label *b) {
                                           return a->cost < b->cost;
                                       });
#else
                      std::find(vec.begin(), vec.end(), label);
#endif

        if (it != vec.end() &&
            (!std::is_sorted(vec.begin(), vec.end()) || *it == label)) {
#ifdef SORTED_LABELS
            vec.erase(it);
#else
            *it = std::move(vec.back());
            vec.pop_back();
#endif
        }
    }

    mutable std::vector<Label *>
        all_labels;  // Mutable to allow modification in const function

    inline const std::vector<Label *> &get_labels() const {
        // Fast path - return direct vector if not split
        if (!is_split) {
            return labels_vec;
        }

        // We're split - need to combine sub-buckets
        assert(sub_buckets.size() == 2);  // Always split into 2 buckets

        // Calculate total size once
        const size_t total_size =
            std::accumulate(sub_buckets.begin(), sub_buckets.end(), size_t{0},
                            [](size_t sum, const auto &bucket) {
                                return sum + bucket.labels_vec.size();
                            });

        // Prepare all_labels vector
        all_labels.clear();
        all_labels.reserve(total_size);

        // Insert all labels from sub-buckets
        for (const auto &bucket : sub_buckets) {
            const auto &source_labels =
                bucket.is_split ? bucket.get_labels() : bucket.labels_vec;
            all_labels.insert(all_labels.end(), source_labels.begin(),
                              source_labels.end());
        }

        return all_labels;
    }

    inline auto &get_sorted_labels() {
        pdqsort(
            labels_vec.begin(), labels_vec.end(),
            [](const Label *a, const Label *b) { return a->cost < b->cost; });
        return labels_vec;
    }

    constexpr static const auto is_unextended = [](Label *label) {
        return !label->is_extended;
    };

    inline auto get_unextended_labels() const {
        // Directly use get_labels() to retrieve the appropriate vector,
        // whether split or not
        const auto &all_labels_ref = get_labels();

        // Return a filtered view of the retrieved labels
        return std::ranges::filter_view{
            std::ranges::views::all(
                static_cast<const std::vector<Label *> &>(all_labels_ref)),
            is_unextended};
    }

    void clear() {
        for (auto &sub_bucket : sub_buckets) {
            sub_bucket.clear();
        }
        sub_buckets.clear();
        labels_vec.clear();
        is_split = false;
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
        for (auto &sub_bucket : sub_buckets) {
            sub_bucket.reset();
        }
        sub_buckets.clear();
        labels_vec.clear();

        is_split = false;
    }
    /**
     * @brief Retrieves the best label from the labels vector.
     *
     * This function returns the first label in the labels vector if it is
     * not empty. If the vector is empty, it returns a nullptr.
     *
     */
    Label *get_best_label() noexcept {
        if (!is_split) {
            if (labels_vec.empty()) {
                return nullptr;
            }

#ifdef SORTED_LABELS
            return labels_vec[0];
#else
            return *std::min_element(labels_vec.begin(), labels_vec.end(),
                                     [](const Label *a, const Label *b) {
                                         return a->cost < b->cost;
                                     });
#endif
        }

        // We always split into exactly 2 buckets
        assert(sub_buckets.size() == 2);

        // Get best labels from both buckets
        Label *first_best = sub_buckets[0].get_best_label();
        Label *second_best = sub_buckets[1].get_best_label();

        // Handle cases where one or both buckets might be empty
        if (!first_best) {
            return second_best;
        }
        if (!second_best) {
            return first_best;
        }

        // Return the label with lower cost
        return (first_best->cost <= second_best->cost) ? first_best
                                                       : second_best;
    }

    [[nodiscard]] bool empty() const { return labels_vec.empty(); }

    ~Bucket() {
        reset();  // Ensure all elements are cleared
        // pool.release(); // Explicitly release the memory pool resources
    }
};
