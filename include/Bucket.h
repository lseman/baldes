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
    std::vector<Label *> labels_vec;
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
    std::array<Bucket *, 2> sub_buckets;
    std::vector<Label *> labels_flush;
    bool shall_split = false;

    double get_cb() const {
        // Early return for empty unsplit buckets
        if (labels_vec.empty() && !is_split) {
            return std::numeric_limits<double>::max();
        }

        // Handle split buckets
        if (is_split) {
            // Use std::ranges::min for cleaner and potentially faster min
            // computation
            auto min_cb = std::ranges::min(
                sub_buckets | std::views::transform([](const auto &sub_bucket) {
                    return sub_bucket->get_cb();
                }));
            return min_cb;
        }

#ifdef SORTED_LABELS
        // Directly return the first element's cost if labels are sorted
        return labels_vec.front()->cost;
#else
        // Use std::ranges::min_element for cleaner and potentially faster min
        // computation
        auto min_label = std::ranges::min_element(
            labels_vec,
            [](const Label *a, const Label *b) { return a->cost < b->cost; });
        return (*min_label)->cost;
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

    double min_split_range = 0.5;
    static constexpr int MAX_BUCKET_DEPTH = 4;

    bool check_dominance(const Label *new_label, const auto &dominance_func,
                         int &stat_n_dom) const noexcept {
        if (get_cb() > new_label->cost) {
            return false;
        }
        if (!is_split) {
            if (dominance_func(get_labels())) {
                ++stat_n_dom;
                return true;
            }
            return false;
        }
        std::array<const Bucket *, 2> candidate_buckets;
        size_t num_candidates = 0;
        for (const auto &bucket : sub_buckets) {
            if (bucket->get_cb() <= new_label->cost &&
                num_candidates < candidate_buckets.size()) {
                candidate_buckets[num_candidates++] = bucket;
            }
        }
        std::ranges::sort(
            std::ranges::subrange(candidate_buckets.begin(),
                                  candidate_buckets.begin() + num_candidates),
            [](const auto *a, const auto *b) noexcept {
                return a->get_cb() < b->get_cb();
            });
        for (size_t i = 0; i < num_candidates; ++i) {
            if (candidate_buckets[i]->check_dominance(new_label, dominance_func,
                                                      stat_n_dom)) {
                return true;
            }
        }
        return false;
    }

    void split_into_sub_buckets() noexcept {
        if (labels_vec.empty() || is_split || depth >= MAX_BUCKET_DEPTH) {
            return;
        }
        static constexpr size_t RESOURCE_INDEX = 0;
        const double total_range = ub[RESOURCE_INDEX] - lb[RESOURCE_INDEX];
        if (total_range < min_split_range) {
            return;
        }

        std::span labels_span{labels_vec};
        const size_t label_count = labels_span.size();
        const size_t mid_idx = label_count / 2;

        // Initialize sub-buckets on the stack
        std::array<Bucket *, 2> new_buckets;
        for (size_t i = 0; i < 2; ++i) {
            auto bucket = new Bucket();
            bucket->node_id = node_id;
            bucket->depth = depth + 1;
            bucket->min_split_range = min_split_range;

            // Calculate bounds based on actual values at split point
            if (i == 0) {
                bucket->lb = {std::round(lb[RESOURCE_INDEX] * 100.0) / 100.0};
                // Upper bound for first bucket is the resource value at mid_idx
                bucket->ub = {
                    std::round(
                        labels_vec[mid_idx - 1]->resources[RESOURCE_INDEX] *
                        100.0) /
                    100.0};
            } else {
                // Lower bound for second bucket is the resource value at
                // mid_idx
                bucket->lb = {
                    std::round(labels_vec[mid_idx]->resources[RESOURCE_INDEX] *
                               100.0) /
                    100.0};
                bucket->ub = {std::round(ub[RESOURCE_INDEX] * 100.0) / 100.0};
            }
            new_buckets[i] = bucket;
        }

#ifdef SORTED_LABELS
        // Since we're splitting purely by position, we can directly split the
        // vector
        new_buckets[0]->labels_vec.reserve(mid_idx);
        new_buckets[1]->labels_vec.reserve(label_count - mid_idx);

        // Copy first half to first bucket
        std::copy(labels_vec.begin(), labels_vec.begin() + mid_idx,
                  std::back_inserter(new_buckets[0]->labels_vec));

        // Copy second half to second bucket
        std::copy(labels_vec.begin() + mid_idx, labels_vec.end(),
                  std::back_inserter(new_buckets[1]->labels_vec));
#else
        // For unsorted labels, we still split purely by position
        new_buckets[0]->labels_vec.reserve(mid_idx);
        new_buckets[1]->labels_vec.reserve(label_count - mid_idx);

        // Move first half to first bucket
        std::ranges::move(std::ranges::subrange(labels_vec.begin(),
                                                labels_vec.begin() + mid_idx),
                          std::back_inserter(new_buckets[0]->labels_vec));

        // Move second half to second bucket
        std::ranges::move(std::ranges::subrange(labels_vec.begin() + mid_idx,
                                                labels_vec.end()),
                          std::back_inserter(new_buckets[1]->labels_vec));
#endif

        // Move new buckets into place
        sub_buckets[0] = new_buckets[0];
        sub_buckets[1] = new_buckets[1];

        // Clear parent bucket's labels efficiently
        labels_vec = std::vector<Label *>();
        is_split = true;
        shall_split = true;
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

            if (first_bucket->is_contained(label)) {
                if (first_bucket->depth >= MAX_BUCKET_DEPTH) {
                    first_bucket->labels_vec.push_back(label);
                } else {
                    first_bucket->add_label(label);
                }
                return;
            }

            if (second_bucket->is_contained(label)) {
                if (second_bucket->depth >= MAX_BUCKET_DEPTH) {
                    second_bucket->labels_vec.push_back(label);
                } else {
                    second_bucket->add_label(label);
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
                (first_bucket->ub[0] + second_bucket->lb[0]) / 2.0;
            if (label->resources[0] <= mid_point) {
                first_bucket->labels_vec.push_back(label);
            } else {
                second_bucket->labels_vec.push_back(label);
            }
            return;
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
            if (bucket->is_contained(label)) {
                bucket->add_label(label);
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
                if (bucket->is_contained(label)) {
                    if (bucket->depth >= MAX_BUCKET_DEPTH) {
                        // At max depth, insert directly into bucket's vector
                        insert_sorted_label(bucket->labels_vec, label);
                    } else {
                        bucket->add_sorted_label(label);
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
        // Use std::ranges::lower_bound for cleaner and potentially faster
        // search
        auto insert_pos = std::ranges::lower_bound(
            vec, label,
            [](const Label *a, const Label *b) { return a->cost < b->cost; });

        // Use emplace to avoid unnecessary copying of pointers
        vec.emplace(insert_pos, label);
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
     * @brief Removes a label from the labels vector.
     *
     * This function searches for the specified label in the labels vector.
     * If found, it replaces the label with the last element in the vector
     *
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
                        sub_buckets[0]->ub[RESOURCE_INDEX]]
                ->remove_label(label);
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

        // Calculate total size once using std::ranges::fold_left (C++23)
        const size_t total_size = std::ranges::fold_left(
            sub_buckets, size_t{0}, [](size_t sum, const auto &bucket) {
                return sum + bucket->labels_vec.size();
            });

        // Prepare all_labels vector
        all_labels.clear();
        all_labels.reserve(total_size);

        // Insert all labels from sub-buckets
        for (const auto &bucket : sub_buckets) {
            const auto &source_labels =
                bucket->is_split ? bucket->get_labels() : bucket->labels_vec;
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
        sub_buckets = {};
        labels_vec.clear();
        is_split = false;
        shall_split = false;
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
        labels_vec.clear();

        is_split = false;
        shall_split = false;
    }
    /**
     * @brief Retrieves the best label from the labels vector.
     *
     * This function returns the first label in the labels vector if it is
     * not empty. If the vector is empty, it returns a nullptr.
     *
     */
    Label *get_best_label() noexcept {
        // Fast path for non-split buckets
        if (!is_split) {
            if (labels_vec.empty()) {
                return nullptr;
            }

#ifdef SORTED_LABELS
            return labels_vec.front();  // Direct access for sorted labels
#else
            // Use std::ranges::min_element for cleaner and potentially faster
            // min computation
            return *std::ranges::min_element(
                labels_vec, [](const Label *a, const Label *b) {
                    return a->cost < b->cost;
                });
#endif
        }

        // Get best labels from both buckets
        Label *first_best = sub_buckets[0]->get_best_label();
        Label *second_best = sub_buckets[1]->get_best_label();

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
