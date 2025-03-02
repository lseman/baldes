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

    double get_lb() const { return lb[0]; }
    double get_ub() const { return ub[0]; }

    double get_cb() const {
        // For empty unsplit buckets, return a high cost.
        if (labels_vec.empty() && !is_split) {
            return std::numeric_limits<double>::max();
        }

        if (!is_split) {
#ifdef SORTED_LABELS
            // For sorted labels, the first label holds the minimum cost.
            return labels_vec.front()->cost;
#else
            // For unsorted labels, use std::ranges::min_element to find the
            // label with the minimum cost.
            auto min_label = std::ranges::min_element(
                labels_vec, [](const Label *a, const Label *b) {
                    return a->cost < b->cost;
                });
            return (*min_label)->cost;
#endif
        }

        // For split buckets, compute the minimum cost among sub-buckets.
        return std::ranges::min(
            sub_buckets | std::views::transform([](const auto &sub_bucket) {
                return sub_bucket->get_cb();
            }));
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
        // Early exit if the bucket's current best cost (cb) is greater than the
        // new label's cost.
        // if (numericutils::gt(get_cb(), new_label->cost)) {
        //     return false;
        // }

        // For non-split buckets, simply use the provided dominance function.
        if (!is_split) {
            return dominance_func(get_labels(), stat_n_dom);
        }

        // For split buckets, collect candidate sub-buckets whose best cost (cb)
        // is not greater than the new label's cost.
        std::array<const Bucket *, 2> candidate_buckets{};
        size_t num_candidates = 0;
        for (const auto &bucket : sub_buckets) {
            // if (bucket->get_cb() <= new_label->cost &&
            // num_candidates < candidate_buckets.size()) {
            candidate_buckets[num_candidates++] = bucket;
        }
        // }

        // Sort candidates by their best cost (cb) for early termination.
        std::ranges::sort(
            std::ranges::subrange(candidate_buckets.begin(),
                                  candidate_buckets.begin() + num_candidates),
            [](const Bucket *a, const Bucket *b) noexcept {
                return a->get_cb() < b->get_cb();
            });

        // Check dominance recursively in candidate sub-buckets.
        for (size_t i = 0; i < num_candidates; ++i) {
            if (candidate_buckets[i]->check_dominance(new_label, dominance_func,
                                                      stat_n_dom)) {
                return true;
            }
        }
        return false;
    }

    // Splits the bucket into two sub-buckets based on the median resource
    // value.
    void split_into_sub_buckets() noexcept {
        // Early exit if no labels, already split, or at max depth.
        if (labels_vec.empty() || is_split || depth >= MAX_BUCKET_DEPTH) {
            return;
        }
        static constexpr size_t RESOURCE_INDEX = 0;
        const double total_range = ub[RESOURCE_INDEX] - lb[RESOURCE_INDEX];
        if (total_range < min_split_range) {
            return;
        }

        // Create a span over the labels vector.
        std::span<Label *> labels_span{labels_vec};
        const size_t label_count = labels_span.size();
        const size_t mid_idx = label_count / 2;

        // Initialize two new sub-buckets on the heap.
        std::array<Bucket *, 2> new_buckets;
        for (size_t i = 0; i < 2; ++i) {
            auto bucket = new Bucket();
            bucket->node_id = node_id;
            bucket->depth = depth + 1;
            bucket->min_split_range = min_split_range;
            bucket->shall_split = false;  // Default value

            // Calculate new bounds based on the resource at the split point.
            if (i == 0) {
                // First bucket: lower bound from parent's lower bound,
                // upper bound based on the resource value of the last label in
                // the first half.
                bucket->lb = {std::round(lb[RESOURCE_INDEX] * 100.0) / 100.0};
                bucket->ub = {
                    std::round(
                        labels_vec[mid_idx - 1]->resources[RESOURCE_INDEX] *
                        100.0) /
                    100.0};
            } else {
                // Second bucket: lower bound based on the first label in the
                // second half, upper bound from parent's upper bound.
                bucket->lb = {
                    std::round(labels_vec[mid_idx]->resources[RESOURCE_INDEX] *
                               100.0) /
                    100.0};
                bucket->ub = {std::round(ub[RESOURCE_INDEX] * 100.0) / 100.0};
            }
            new_buckets[i] = bucket;
        }

#ifdef SORTED_LABELS
        // For sorted labels, reserve capacity and copy by position.
        new_buckets[0]->labels_vec.reserve(mid_idx);
        new_buckets[1]->labels_vec.reserve(label_count - mid_idx);
        std::copy(labels_vec.begin(), labels_vec.begin() + mid_idx,
                  std::back_inserter(new_buckets[0]->labels_vec));
        std::copy(labels_vec.begin() + mid_idx, labels_vec.end(),
                  std::back_inserter(new_buckets[1]->labels_vec));
#else
        // For unsorted labels, move by position.
        new_buckets[0]->labels_vec.reserve(mid_idx);
        new_buckets[1]->labels_vec.reserve(label_count - mid_idx);
        std::ranges::move(std::ranges::subrange(labels_vec.begin(),
                                                labels_vec.begin() + mid_idx),
                          std::back_inserter(new_buckets[0]->labels_vec));
        std::ranges::move(std::ranges::subrange(labels_vec.begin() + mid_idx,
                                                labels_vec.end()),
                          std::back_inserter(new_buckets[1]->labels_vec));
#endif

        // Store the new sub-buckets.
        sub_buckets[0] = new_buckets[0];
        sub_buckets[1] = new_buckets[1];

        // Clear the parent's labels vector efficiently.
        labels_vec.clear();
        is_split = true;
        // shall_split = true;
    }

    bool is_empty() const noexcept {
        if (is_split) {
            return sub_buckets[0]->is_empty() && sub_buckets[1]->is_empty();
        }
        return labels_vec.empty();
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
        if (!label) {
            return;
        }

        // Iterate over each sub-bucket and assign the label if it belongs
        // there.
        for (auto &bucket : sub_buckets) {
            if (bucket->is_contained(label)) {
                bucket->add_label(label);
                return;
            }
        }

#ifdef DEBUG
        // Warn if the label doesn't fit in any sub-bucket.
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

    bool contains(std::vector<double> &resource_values_vec) const noexcept {
        for (size_t i = 0; i < resource_values_vec.size(); ++i) {
            if (numericutils::lt(resource_values_vec[i], lb[i]) ||
                numericutils::gt(resource_values_vec[i], ub[i])) {
                return false;
            }
        }
        return true;
    }

    // Adds a label into the bucket (or its sub-buckets) in sorted order.
    void add_sorted_label(Label *label) noexcept {
        if (!label) return;

        if (is_split) {
            // For a split bucket, we know there are exactly two sub-buckets.
            auto &first_bucket = sub_buckets[0];
            auto &second_bucket = sub_buckets[1];

            // Quick check for containment
            if (label->resources[0] >= first_bucket->lb[0] &&
                label->resources[0] <= first_bucket->ub[0]) {
                if (first_bucket->depth >= MAX_BUCKET_DEPTH)
                    first_bucket->labels_vec.push_back(label);
                else
                    first_bucket->add_sorted_label(label);
                return;
            }

            if (label->resources[0] >= second_bucket->lb[0] &&
                label->resources[0] <= second_bucket->ub[0]) {
                if (second_bucket->depth >= MAX_BUCKET_DEPTH)
                    second_bucket->labels_vec.push_back(label);
                else
                    second_bucket->add_sorted_label(label);
                return;
            }

            // Fallback: add to the closest sub-bucket
            const double mid_point =
                (first_bucket->ub[0] + second_bucket->lb[0]) * 0.5;
            if (label->resources[0] <= mid_point)
                first_bucket->labels_vec.push_back(label);
            else
                second_bucket->labels_vec.push_back(label);
            return;
        }

        // Non-split bucket case: optimize insertion by avoiding unnecessary
        // shifting

        // Reserve more capacity if needed, with a 50% growth factor to reduce
        // reallocations
        const size_t current_size = labels_vec.size();
        if (current_size == labels_vec.capacity()) {
            labels_vec.reserve(std::max<size_t>(
                64, labels_vec.capacity() + (labels_vec.capacity() >> 1)));
        }

        // Fast paths for insertion at beginning or end
        if (current_size == 0 || label->cost >= labels_vec.back()->cost) {
            labels_vec.push_back(label);
            return;
        }

        // Optimization: avoid calling front() which can be slower than direct
        // access
        if (label->cost <= labels_vec[0]->cost) {
            // For insertion at beginning, use resize + shift instead of insert
            // This is faster because it avoids one memory allocation and only
            // does one shift
            labels_vec.resize(current_size + 1);

            // Shift all elements right by one position
            for (size_t i = current_size; i > 0; --i) {
                labels_vec[i] = labels_vec[i - 1];
            }

            // Insert at beginning
            labels_vec[0] = label;
            return;
        }

        // Use binary search to find the insertion point
        auto it = std::lower_bound(
            labels_vec.begin(), labels_vec.end(), label,
            [](const Label *a, const Label *b) { return a->cost < b->cost; });

        // If insertion is near the end, consider whether push_back + rotate is
        // faster
        const size_t pos = std::distance(labels_vec.begin(), it);
        if (pos > current_size - 8) {  // Heuristic: if inserting near the end
            // Add to the end and rotate into position
            labels_vec.push_back(label);
            std::rotate(labels_vec.begin() + pos,
                        labels_vec.begin() + current_size, labels_vec.end());
        } else {
            // Use standard insert for positions not near the end
            labels_vec.insert(it, label);
        }

        // Check if bucket size forces a split
        if (depth < MAX_BUCKET_DEPTH && labels_vec.size() >= BUCKET_CAPACITY) {
            // shall_split = true;
            split_into_sub_buckets();
        }
    }

    // Helper function to avoid code duplication
    static void insert_sorted_label(std::vector<Label *> &vec,
                                    Label *label) noexcept {
        // Find the insertion point using std::ranges::lower_bound with a custom
        // comparator.
        auto insert_pos = std::ranges::lower_bound(
            vec, label,
            [](const Label *a, const Label *b) { return a->cost < b->cost; });

        // Insert the label at the correct sorted position.
        vec.emplace(insert_pos, label);
    }

    bool is_contained(const Label *label) const noexcept {
        if (!label) {
            return false;
        }

        constexpr size_t RESOURCE_INDEX = 0;
        const double resource_value = label->resources[RESOURCE_INDEX];

        // Check if the resource value is within the bucket's lower and upper
        // bounds.
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

    inline std::vector<Label *> get_labels() const {
        if (!is_split) {
            return labels_vec;
        }
        // Calculate total size (flattening recursively if needed)
        size_t total_size = 0;
        for (const auto &bucket : sub_buckets) {
            if (bucket) {
                total_size += bucket->is_split ? bucket->get_labels().size()
                                               : bucket->labels_vec.size();
            }
        }
        std::vector<Label *> combined;
        combined.reserve(total_size);
        // Insert labels from each sub-bucket directly
        for (const auto &bucket : sub_buckets) {
            if (bucket) {
                if (bucket->is_split) {
                    // Call recursively; you might inline this loop if the
                    // recursion depth is low.
                    auto sub = bucket->get_labels();
                    combined.insert(combined.end(), sub.begin(), sub.end());
                } else {
                    combined.insert(combined.end(), bucket->labels_vec.begin(),
                                    bucket->labels_vec.end());
                }
            }
        }
        return combined;
    }

    inline std::vector<Label *> get_non_dominated_labels() const {
        std::vector<Label *> result;
        size_t estimated = 0;
        if (!is_split) {
            estimated = labels_vec.size();
        } else {
            for (const auto &bucket : sub_buckets) {
                estimated += bucket->is_split ? bucket->get_labels().size()
                                              : bucket->labels_vec.size();
            }
        }
        result.reserve(estimated);

        auto add_labels = [&](const std::vector<Label *> &src) {
            for (Label *l : src) {
                if (!l->is_dominated) {
                    result.push_back(l);
                }
            }
        };

        if (!is_split) {
            add_labels(labels_vec);
        } else {
            for (const auto &bucket : sub_buckets) {
                if (bucket) {
                    if (bucket->is_split) {
                        add_labels(bucket->get_labels());
                    } else {
                        add_labels(bucket->labels_vec);
                    }
                }
            }
        }
        return result;
    }

    inline std::vector<Label *> get_active_labels() const {
        std::vector<Label *> result;
        size_t estimated = 0;
        if (!is_split) {
            estimated = labels_vec.size();
        } else {
            for (const auto &bucket : sub_buckets) {
                estimated += bucket->is_split ? bucket->get_labels().size()
                                              : bucket->labels_vec.size();
            }
        }
        result.reserve(estimated);

        auto add_labels = [&](const std::vector<Label *> &src) {
            for (Label *l : src) {
                if (!l->is_dominated && !l->is_extended) {
                    result.push_back(l);
                }
            }
        };

        if (!is_split) {
            add_labels(labels_vec);
        } else {
            for (const auto &bucket : sub_buckets) {
                if (bucket) {
                    if (bucket->is_split) {
                        add_labels(bucket->get_labels());
                    } else {
                        add_labels(bucket->labels_vec);
                    }
                }
            }
        }
        return result;
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
        // Return a view filtering get_labels() to only include unextended
        // labels.
        return get_labels() | std::views::filter(is_unextended);
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
        // Fast path: non-split bucket.
        if (!is_split) {
            if (labels_vec.empty()) {
                return nullptr;
            }
#ifdef SORTED_LABELS
            // For sorted labels, the first element is the best.
            return labels_vec.front();
#else
            // For unsorted labels, use std::ranges::min_element to find the
            // best label.
            return *std::ranges::min_element(
                labels_vec, [](const Label *a, const Label *b) {
                    return a->cost < b->cost;
                });
#endif
        }

        // For split buckets, retrieve the best label from each sub-bucket.
        Label *first_best = sub_buckets[0]->get_best_label();
        Label *second_best = sub_buckets[1]->get_best_label();

        // Return the non-null best label, or if both are non-null, the one with
        // lower cost.
        if (!first_best) return second_best;
        if (!second_best) return first_best;
        return (first_best->cost <= second_best->cost) ? first_best
                                                       : second_best;
    }

    [[nodiscard]] bool empty() const { return labels_vec.empty(); }

    ~Bucket() {
        reset();  // Ensure all elements are cleared
        // pool.release(); // Explicitly release the memory pool resources
    }
};
