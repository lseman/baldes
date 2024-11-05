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
    // std::vector<Label *> labels_vec;
    // std::vector<Label *, PoolAllocator<Label *>> labels_vec;
    // std::pmr::pool_options                 pool_opts;
    // std::pmr::unsynchronized_pool_resource pool;
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
    bool check_dominance(const Label *new_label, std::function<bool(const std::vector<Label *> &)> dominance_func,
                         int &stat_n_dom) {
        if (!is_split) {
            // Directly check dominance in the mother bucket
            if (dominance_func(get_labels())) {
                stat_n_dom++; // Increment dominated labels count
                return true;
            }
        } else {
            // Find the current sub-bucket containing the new_label
            Bucket *current_sub_bucket       = nullptr;
            size_t  current_sub_bucket_index = sub_buckets.size(); // Invalid index if not found

            for (size_t i = 0; i < sub_buckets.size(); ++i) {
                if (sub_buckets[i].is_contained(new_label)) {
                    current_sub_bucket       = &sub_buckets[i];
                    current_sub_bucket_index = i;
                    break;
                }
            }

            // Check dominance recursively in the current sub-bucket
            if (current_sub_bucket && current_sub_bucket->check_dominance(new_label, dominance_func, stat_n_dom)) {
                return true;
            }

            // Check only the sub-buckets that come before the current sub-bucket
            for (size_t i = 0; i < current_sub_bucket_index; ++i) {
                if (sub_buckets[i].check_dominance(new_label, dominance_func, stat_n_dom)) { return true; }
            }
        }
        return false;
    }

    bool                    is_split        = false;
    static constexpr double min_split_range = 0.5; // Define a threshold for minimum range to split
    void                    split_into_sub_buckets(size_t num_sub_buckets) {
        double total_range = ub[0] - lb[0]; // Assuming 1D bounds here
        if (total_range < min_split_range) { return; }

        // if (depth > 0) { return; } // Do not split beyond a certain depth

        is_split = true;
        sub_buckets.clear();
        sub_buckets.reserve(num_sub_buckets);

        // Calculate the range for each sub-bucket
        double range_per_bucket = total_range / num_sub_buckets;

        for (size_t i = 0; i < num_sub_buckets; ++i) {
            Bucket sub_bucket;
            sub_bucket.node_id = node_id;
            sub_bucket.lb      = {lb[0] + i * range_per_bucket};
            sub_bucket.ub      = {lb[0] + (i + 1) * range_per_bucket};

            // For the last sub-bucket, set ub to the mother's ub[0] to ensure full range coverage
            if (i == num_sub_buckets - 1) { sub_bucket.ub[0] = ub[0]; }

            sub_bucket.depth = depth + 1;
            sub_buckets.push_back(std::move(sub_bucket));
        }

        // Distribute existing labels into sub-buckets based on their resources
        for (auto label : labels_vec) { assign_label_to_sub_bucket(label); }

        // iterate over subbuckets and sort labels
        for (auto &sub_bucket : sub_buckets) { sub_bucket.sort(); }
        labels_vec.clear();
    }

    void sort() {
        pdqsort(labels_vec.begin(), labels_vec.end(), [](const Label *a, const Label *b) { return a->cost < b->cost; });
    }

    void assign_label_to_sub_bucket(Label *label) {
        // Attempt to assign the label to the correct sub-bucket based on resources[0]
        for (auto &sub_bucket : sub_buckets) {
            // print sub_bucket lb and ub
            if (sub_bucket.is_contained(label)) {
                sub_bucket.add_label(label);
                return;
            }
        }

        // Print warning if the label does not fit into any sub-bucket
        print_info("Warning: Label did not fit into any sub-bucket\n");
        print_info("Label resources[0]: {}, Mother lb: {}, Mother ub: {}\n", label->resources[0], lb[0], ub[0]);
    }

    /**
     * @brief Adds an arc between two buckets.
     *
     * This function adds an arc from one bucket to another, either in the forward or backward direction.
     * The arc is characterized by resource increments, cost increment, and whether it is fixed or not.
     *
     */
    void add_bucket_arc(int from_bucket, int to_bucket, const std::vector<double> &res_inc, double cost_inc, bool fw,
                        bool fixed) {
        if (fw) {
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
    void add_jump_arc(int from_bucket, int to_bucket, const std::vector<double> &res_inc, double cost_inc, bool fw) {
        if (fw) {
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
    void add_label(Label *label) {

        if (!is_split) {
            labels_vec.push_back(label);
            if (labels_vec.size() >= BUCKET_CAPACITY) {
                // print_info("Bucket capacity reached\n");
                split_into_sub_buckets(2); // Example: Split into 4 sub-buckets
            }
        } else {
            // fmt::print("Warning: Adding label to split bucket\n");
            assign_label_to_sub_bucket(label); // Directly add to the correct sub-bucket
        }
    }

    /**
     * @brief Adds a label to the labels_vec in sorted order based on the cost.
     *
     * This function inserts the given label into the labels_vec such that the vector
     * remains sorted in ascending order of the label's cost. The insertion is done
     * using binary search to find the appropriate position, ensuring efficient insertion.
     *
     */
    /*
   void add_sorted_label(Label *label) noexcept {
       if (labels_vec.empty() || label->cost >= labels_vec.back()->cost) {
           labels_vec.push_back(label); // Direct insertion at the end
       } else if (label->cost <= labels_vec.front()->cost) {
           labels_vec.insert(labels_vec.begin(), label); // Direct insertion at the beginning
       } else {
           auto it = std::lower_bound(labels_vec.begin(), labels_vec.end(), label,
                                      [](const Label *a, const Label *b) { return a->cost < b->cost; });
           labels_vec.insert(it, label); // Insertion in the middle
       }
   }
   */

    void add_sorted_label(Label *label) noexcept {
        // Lambda to add label in sorted order within a vector of labels
        auto add_sorted_to_vector = [&](std::vector<Label *> &labels) {
            if (labels.empty() || label->cost >= labels.back()->cost) {
                labels.push_back(label); // Direct insertion at the end
            } else if (label->cost <= labels.front()->cost) {
                labels.insert(labels.begin(), label); // Direct insertion at the beginning
            } else {
                auto it = std::lower_bound(labels.begin(), labels.end(), label,
                                           [](const Label *a, const Label *b) { return a->cost < b->cost; });
                labels.insert(it, label); // Insertion in the middle
            }
        };

        if (!is_split) {
            // Add label to the mother bucket's labels in sorted order
            add_sorted_to_vector(labels_vec);

            // Check if splitting is needed
            if (labels_vec.size() >= BUCKET_CAPACITY) { split_into_sub_buckets(2); }
        } else {
            // If already split, find the correct sub-bucket and add the label there
            for (auto &sub_bucket : sub_buckets) {
                if (sub_bucket.is_contained(label)) {
                    add_sorted_to_vector(sub_bucket.labels_vec);
                    return;
                }
            }
        }
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
        if (is_split) {
            // Recursively call remove_label on each sub-bucket
            for (auto &sub_bucket : sub_buckets) { sub_bucket.remove_label(label); }
        } else {
            // If not split, remove the label from the main labels_vec
            auto it = std::find(labels_vec.begin(), labels_vec.end(), label);
            if (it != labels_vec.end()) {
                // Only move last element if there are multiple elements
                if (labels_vec.size() > 1) {
                    *it = labels_vec.back(); // Move last element to the found position
                }
                labels_vec.pop_back(); // Remove the last element
            }
        }
    }

    mutable std::vector<Label *> all_labels; // Mutable to allow modification in const function

    inline const std::vector<Label *> &get_labels() const {
        if (is_split) {
            // Reserve space only once, based on the size of all sub-buckets
            all_labels.clear();
            size_t total_size = 0;
            for (const auto &sub_bucket : sub_buckets) { total_size += sub_bucket.get_labels().size(); }
            all_labels.reserve(total_size);

            // Insert labels from each sub-bucket
            for (const auto &sub_bucket : sub_buckets) {
                const auto &sub_labels = sub_bucket.get_labels();
                all_labels.insert(all_labels.end(), sub_labels.begin(), sub_labels.end());
            }
            return all_labels;
        } else {
            return labels_vec;
        }
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
        labels_vec.clear();
        for (auto &sub_bucket : sub_buckets) { sub_bucket.clear(); }
        sub_buckets.clear();
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
        labels_vec.clear();
        for (auto &sub_bucket : sub_buckets) { sub_bucket.reset(); }
        sub_buckets.clear();
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
        if (labels_vec.empty()) return nullptr;
        return labels_vec.front();
    }

    [[nodiscard]] bool empty() const { return labels_vec.empty(); }

    ~Bucket() {
        reset(); // Ensure all elements are cleared
        // pool.release(); // Explicitly release the memory pool resources
    }
};
