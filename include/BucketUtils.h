/**
 * @file BucketUtils.h
 * @brief Header file for utilities related to the Bucket Graph in the Vehicle Routing Problem (VRP).
 *
 * This file contains various template functions and algorithms for managing buckets in the Bucket Graph. The
 * Bucket Graph is a representation of the VRP problem, where jobs are assigned to "buckets" based on resource
 * intervals, and arcs represent feasible transitions between buckets. The utilities provided include adding arcs,
 * defining buckets, generating arcs, extending labels, and managing strongly connected components (SCCs).
 *
 * Key components:
 * - `add_arc`: Adds directed arcs between buckets based on the direction and resource increments.
 * - `get_bucket_number`: Computes the bucket number for a given job and resource values.
 * - `define_buckets`: Defines the structure and intervals for the buckets based on resource bounds.
 * - `generate_arcs`: Generates arcs between buckets based on resource constraints and feasibility.
 * - `labeling_algorithm`: Solves labeling algorithms in both forward and backward directions for resource constraints.
 * - `SCC_handler`: Identifies and processes SCCs in the bucket graph.
 * - `Extend`: Extends a label with a given arc, checking for feasibility based on resources.
 *
 * The utilities use template parameters for direction (Forward or Backward), stages, and other configurations,
 * allowing flexible handling of the bucket graph in both directions.
 */

#pragma once

#include "BucketJump.h"
#include "Definitions.h"

#include "../cuts/SRC.h"
#include <cstring>

/**
 * Adds an arc to the bucket graph.
 *
 * @tparam D The direction of the arc (Forward or Backward).
 * @param from_bucket The index of the source bucket.
 * @param to_bucket The index of the destination bucket.
 * @param res_inc The vector of resource increments.
 * @param cost_inc The cost increment.
 */
template <Direction D>
void BucketGraph::add_arc(int from_bucket, int to_bucket, const std::vector<double> &res_inc, double cost_inc) {

    if constexpr (D == Direction::Forward) {
        fw_arcs.emplace_back(from_bucket, to_bucket, res_inc, cost_inc);
        fw_bucket_graph[from_bucket].push_back(to_bucket);

    } else if constexpr (D == Direction::Backward) {

        bw_arcs.emplace_back(from_bucket, to_bucket, res_inc, cost_inc);
        bw_bucket_graph[from_bucket].push_back(to_bucket);
    }
}

/**
 * @brief Get the bucket number for a given job and value.
 *
 * This function returns the bucket number for a given job and value in the bucket graph.
 * The bucket graph is represented by the `buckets` vector, which contains intervals of buckets.
 * The direction of the bucket graph is specified by the template parameter `D`.
 *
 * @tparam D The direction of the bucket graph.
 * @param job The job index.
 * @param value The value to search for in the bucket graph.
 * @return The bucket number if found, -1 otherwise.
 */
template <Direction D>
inline int BucketGraph::get_bucket_number(int job, const std::vector<double> &resource_values_vec) noexcept {
    const auto &num_buckets_index = assign_buckets<D>(num_buckets_index_fw, num_buckets_index_bw);
    const int   start_index       = num_buckets_index[job];

    if constexpr (R_SIZE > 1) {
        auto            &vrpjob = jobs[job];
        std::vector<int> base_intervals(intervals.size());
        std::vector<int> first_lb(intervals.size());
        std::vector<int> first_ub(intervals.size());
        std::vector<int> remainders(intervals.size());

        // Compute base intervals, remainders, and bounds for each dimension
        for (int r = 0; r < intervals.size(); ++r) {
            base_intervals[r] = (R_max[r] - R_min[r] + 1) / intervals[r].interval;
            remainders[r]     = static_cast<int>(R_max[r] - R_min[r] + 1) % intervals[r].interval;
            first_lb[r]       = vrpjob.lb[r]; // Lower bound for each dimension
            first_ub[r]       = vrpjob.ub[r]; // Upper bound for each dimension
        }

        int bucket_index = 0;
        int multiplier   = 1; // This will be used to compute the final bucket index across dimensions

        if constexpr (D == Direction::Forward) {
            // Iterate over each resource value and compute the bucket index for forward direction
            for (int r = resource_values_vec.size() - 1; r >= 0; --r) {
                int resource_value_int = static_cast<int>(resource_values_vec[r]);
                int bucket_dim_index;

                // Handle uneven divisions by accounting for remainders
                if (resource_value_int < first_lb[r] + base_intervals[r] * (remainders[r] > 0)) {
                    bucket_dim_index = (resource_value_int - first_lb[r]) / base_intervals[r];
                } else {
                    bucket_dim_index = (resource_value_int - first_lb[r] - remainders[r]) / base_intervals[r];
                }

                bucket_index += bucket_dim_index * multiplier;
                multiplier *=
                    (R_max[r] - R_min[r] + 1) / base_intervals[r]; // Update the multiplier for the next dimension
            }
        } else if constexpr (D == Direction::Backward) {
            // Iterate over each resource value and compute the bucket index for backward direction
            for (int r = resource_values_vec.size() - 1; r >= 0; --r) {
                int resource_value_int = static_cast<int>(resource_values_vec[r]);
                int bucket_dim_index;

                // Handle uneven divisions by accounting for remainders
                if (resource_value_int > first_ub[r] - base_intervals[r] * (remainders[r] > 0)) {
                    bucket_dim_index = (first_ub[r] - resource_value_int) / base_intervals[r];
                } else {
                    bucket_dim_index = (first_ub[r] - resource_value_int - remainders[r]) / base_intervals[r];
                }

                bucket_index += bucket_dim_index * multiplier;
                multiplier *=
                    (R_max[r] - R_min[r] + 1) / base_intervals[r]; // Update the multiplier for the next dimension
            }
        }
        return bucket_index + start_index; // Add the starting index to get the final bucket index
    } else {
        const auto &num_buckets_index = assign_buckets<D>(num_buckets_index_fw, num_buckets_index_bw);
        const int   start_index       = num_buckets_index[job];
        const int   base_interval     = (T_max - 0.0 + 1) / static_cast<int>(intervals[0].interval);
        int         value             = static_cast<int>(resource_values_vec[0]);
        // print number of buckets for job

        auto &vrpjob = jobs[job];
        if constexpr (D == Direction::Forward) {
            const int first_lb     = vrpjob.lb[0];
            int       bucket_index = (value - first_lb) / base_interval + start_index;
            return bucket_index; // Return the computed bucket index
        } else if constexpr (D == Direction::Backward) {
            const int first_ub     = vrpjob.ub[0];
            int       bucket_index = (first_ub - value) / base_interval + start_index;
            return bucket_index; // Correct bucket
        }
        std::throw_with_nested(std::runtime_error("Invalid direction"));
        return -1; // If no bucket is found
    }
}

/**
 * @brief Defines the buckets for the BucketGraph.
 *
 * This function determines the number of buckets based on the time intervals and assigns buckets to the graph.
 * It computes resource bounds for each vertex and defines the bounds of each bucket.
 *
 * @tparam D The direction of the buckets (Forward or Backward).
 */
template <Direction D>
void BucketGraph::define_buckets() {
    int              num_intervals = intervals.size(); // Determine how many resources we have (number of intervals)
    std::vector<int> total_ranges(num_intervals);
    std::vector<int> base_intervals(num_intervals);
    std::vector<int> remainders(num_intervals);

    // Determine the base interval and other relevant values for each resource
    for (int r = 0; r < intervals.size(); ++r) {
        total_ranges[r]   = R_max[r] - R_min[r] + 1;
        base_intervals[r] = total_ranges[r] / intervals[r].interval;
        remainders[r]     = total_ranges[r] % intervals[r].interval; // Use std::fmod for floating-point modulo
    }
    auto &buckets           = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &num_buckets       = assign_buckets<D>(num_buckets_fw, num_buckets_bw);
    auto &num_buckets_index = assign_buckets<D>(num_buckets_index_fw, num_buckets_index_bw);
    num_buckets.resize(jobs.size());
    num_buckets_index.resize(jobs.size());

    int cum_sum      = 0; // Keeps track of the global bucket index
    int bucket_index = 0; // Keeps track of where to insert the next bucket

    // Loop through each job to define its specific buckets
    for (const auto &VRPJob : jobs) {
        std::vector<int> job_total_ranges(num_intervals);
        for (int r = 0; r < num_intervals; ++r) { job_total_ranges[r] = VRPJob.ub[r] - VRPJob.lb[r]; }
        // Check if the job's range fits within a single base interval for all dimensions
        bool fits_single_bucket = true;
        for (int r = 0; r < num_intervals; ++r) {
            if (job_total_ranges[r] > base_intervals[r]) {
                fits_single_bucket = false;
                break;
            }
        }

        if (fits_single_bucket) {
            // Single bucket case for all resources
            std::vector<int> lb(num_intervals), ub(num_intervals);
            for (int r = 0; r < num_intervals; ++r) {
                lb[r] = VRPJob.lb[r];
                ub[r] = VRPJob.ub[r];
            }
            buckets[bucket_index]        = Bucket(VRPJob.id, lb, ub);
            num_buckets[VRPJob.id]       = 1;
            num_buckets_index[VRPJob.id] = cum_sum;
            bucket_index++;
            cum_sum++;
        } else {
            // Multiple bucket case, need to consider intervals for each resource dimension
            int              n_buckets = 0;
            std::vector<int> current_pos(num_intervals, 0);

            // Nested loop over all intervals in each resource dimension
            while (true) {
                std::vector<int> interval_start(num_intervals), interval_end(num_intervals);

                for (int r = 0; r < num_intervals; ++r) {
                    interval_start[r] = VRPJob.lb[r] + current_pos[r] * base_intervals[r];
                    interval_end[r]   = VRPJob.ub[r] - current_pos[r] * base_intervals[r];

                    if constexpr (D == Direction::Forward) {
                        interval_end[r] = VRPJob.ub[r];
                    } else if constexpr (D == Direction::Backward) {
                        interval_start[r] = VRPJob.lb[r];
                    }
                }
                buckets[bucket_index] = Bucket(VRPJob.id, interval_start, interval_end);
                bucket_index++;
                n_buckets++;
                cum_sum++;

                // Update the position of intervals
                bool done = true;
                for (int r = num_intervals - 1; r >= 0; --r) {
                    current_pos[r]++;
                    if (current_pos[r] * base_intervals[r] < job_total_ranges[r]) {
                        done = false;
                        break;
                    } else {
                        current_pos[r] = 0;
                    }
                }
                if (done) break;
            }

            num_buckets[VRPJob.id]       = n_buckets;
            num_buckets_index[VRPJob.id] = cum_sum - n_buckets;
        }
    }

    if constexpr (D == Direction::Forward) {
        fw_buckets_size = cum_sum;
    } else {
        bw_buckets_size = cum_sum;
    }
}

/**
 * Generates arcs in the bucket graph based on the specified direction.
 *
 * @tparam D The direction of the arcs (Forward or Backward).
 */
template <Direction D>
void BucketGraph::generate_arcs() {
    auto buckets_mutex = std::mutex();
    num_buckets_per_interval.clear();

    // Clear the appropriate bucket graph
    if constexpr (D == Direction::Forward) {
        fw_bucket_graph.clear();
    } else {
        bw_bucket_graph.clear();
    }

    auto fixed_buckets = assign_buckets<D>(fw_fixed_buckets, bw_fixed_buckets);

    auto &buckets           = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &num_buckets       = assign_buckets<D>(num_buckets_fw, num_buckets_bw);
    auto &num_buckets_index = assign_buckets<D>(num_buckets_index_fw, num_buckets_index_bw);

    // Determine base intervals for each resource dimension
    std::vector<int> base_intervals(intervals.size());
    for (int r = 0; r < intervals.size(); ++r) {
        base_intervals[r] = std::floor((R_max[r] - R_min[r] + 1) / static_cast<int>(intervals[r].interval));
    }

    // Clear arcs in each bucket
    std::for_each(std::execution::par_unseq, buckets.begin(), buckets.end(), [&](auto &bucket) {
        bucket.clear();
        bucket.clear_arcs(D == Direction::Forward);
    });

    // Function to add arcs for a specific job and bucket
    auto add_arcs_for_job = [&](const VRPJob &job, int from_bucket, std::vector<double> &res_inc,
                                std::vector<std::pair<int, int>> &local_arcs) {
        auto arcs = job.get_arcs<D>();

        for (const auto &arc : arcs) {
            auto &next_job = jobs[arc.to];

            if (job.id == next_job.id) continue;

            auto   travel_cost = getcij(job.id, next_job.id);
            double cost_inc    = travel_cost - next_job.cost;
            res_inc[0]         = travel_cost + job.duration; // Update based on job duration

            for (int j = 0; j < num_buckets[next_job.id]; ++j) {
                int to_bucket = j + num_buckets_index[next_job.id];
                if (from_bucket == to_bucket) continue;

                if (fixed_buckets[from_bucket][to_bucket] == 1) continue;

                bool valid_arc = true;
                for (int r = 0; r < res_inc.size(); ++r) {
                    if constexpr (D == Direction::Forward) {
                        if (buckets[from_bucket].lb[r] + res_inc[r] > buckets[to_bucket].ub[r]) {
                            valid_arc = false;
                            break;
                        }
                    } else if constexpr (D == Direction::Backward) {
                        if (buckets[from_bucket].ub[r] - res_inc[r] < buckets[to_bucket].lb[r]) {
                            valid_arc = false;
                            break;
                        }
                    }
                }
                if (!valid_arc) continue;

                if constexpr (D == Direction::Forward) {
                    for (int r = 0; r < res_inc.size(); ++r) {
                        double max_calc =
                            std::max(buckets[from_bucket].lb[r] + res_inc[r], static_cast<double>(next_job.lb[r]));
                        if (max_calc < buckets[to_bucket].lb[r] ||
                            max_calc >= buckets[to_bucket].lb[r] + base_intervals[r]) {
                            valid_arc = false;
                            break;
                        }
                    }
                } else if constexpr (D == Direction::Backward) {
                    for (int r = 0; r < res_inc.size(); ++r) {
                        double min_calc =
                            std::min(buckets[from_bucket].ub[r] - res_inc[r], static_cast<double>(next_job.ub[r]));
                        if (min_calc > buckets[to_bucket].ub[r] ||
                            min_calc <= buckets[to_bucket].ub[r] - base_intervals[r]) {
                            valid_arc = false;
                            break;
                        }
                    }
                }
                if (!valid_arc) continue;

                // Store arc data locally before committing to the global structure
                local_arcs.emplace_back(from_bucket, to_bucket);
                double              local_cost_inc = cost_inc;
                std::vector<double> local_res_inc  = res_inc;

                {
                    std::lock_guard<std::mutex> lock(buckets_mutex);
                    add_arc<D>(from_bucket, to_bucket, local_res_inc, local_cost_inc);
                    buckets[from_bucket].add_bucket_arc(from_bucket, to_bucket, local_res_inc, local_cost_inc,
                                                        D == Direction::Forward, false);
                }
            }
        }
    };

    // Parallelize the outer loop over jobs
    std::for_each(std::execution::par_unseq, jobs.begin(), jobs.end(), [&](const VRPJob &VRPJob) {
        std::vector<double>              res_inc = {static_cast<double>(VRPJob.duration)};
        std::vector<std::pair<int, int>> local_arcs;

        for (int i = 0; i < num_buckets[VRPJob.id]; ++i) {
            int from_bucket = i + num_buckets_index[VRPJob.id];
            add_arcs_for_job(VRPJob, from_bucket, res_inc, local_arcs);
        }
    });

    // fmt::print("Generated arcs for {} jobs.\n", jobs.size());
}

/**
 * Performs the labeling algorithm on the BucketGraph.
 *
 * @tparam D The direction of the algorithm (Forward or Backward).
 * @tparam S The stage of the algorithm (One or Two).
 * @param q_point The q-point used in the algorithm.
 * @return A vector of doubles representing the c_bar values for each bucket.
 */
template <Direction D, Stage S, Full F>
std::vector<double> BucketGraph::labeling_algorithm(std::vector<double> q_point, bool full) noexcept {

    auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);
    // auto &label_pool                 = assign_buckets<D>(label_pool_fw, label_pool_bw);
    auto &ordered_sccs      = assign_buckets<D>(fw_ordered_sccs, bw_ordered_sccs);
    auto &topological_order = assign_buckets<D>(fw_topological_order, bw_topological_order);
    auto &sccs              = assign_buckets<D>(fw_sccs, bw_sccs);
    auto &Phi               = assign_buckets<D>(Phi_fw, Phi_bw);
    auto &c_bar             = assign_buckets<D>(fw_c_bar, bw_c_bar);
    auto &fixed_buckets     = assign_buckets<D>(fw_fixed_buckets, bw_fixed_buckets);
    auto &n_labels          = assign_buckets<D>(n_fw_labels, n_bw_labels);
    auto &sorted_sccs       = assign_buckets<D>(fw_sccs_sorted, bw_sccs_sorted);
    auto &n_buckets         = assign_buckets<D>(fw_buckets_size, bw_buckets_size);
    auto &stat_n_labels     = assign_buckets<D>(stat_n_labels_fw, stat_n_labels_bw);
    auto &stat_n_dom        = assign_buckets<D>(stat_n_dom_fw, stat_n_dom_bw);

    n_labels                         = 0;
    const size_t          n_segments = n_buckets / 64 + 1;
    std::vector<uint64_t> Bvisited(n_segments, 0);

    bool all_ext;
    bool dominated;
    bool domin_smaller;
    for (const auto &scc_index : topological_order) {
        do {
            all_ext = true;
            for (const auto bucket : sorted_sccs[scc_index]) {
                auto bucket_labels = buckets[bucket].get_unextended_labels();
                for (Label *label : bucket_labels) {

                    domin_smaller = false;

                    if constexpr ((S != Stage::Enumerate)) {
                        std::memset(Bvisited.data(), 0, Bvisited.size() * sizeof(uint64_t));
                        domin_smaller =
                            DominatedInCompWiseSmallerBuckets<D, S>(label, bucket, c_bar, Bvisited, ordered_sccs);
                    }

                    if (!domin_smaller) {

                        const auto &arcs      = jobs[label->job_id].get_arcs<D>(scc_index);
                        const auto &jump_arcs = buckets[bucket].template get_jump_arcs<D>();
                        // if (jump_arcs.size() > 0) { fmt::print("Jump arcs: {}\n", jump_arcs.size()); }
                        for (const auto &arc : arcs) {
                            Label *new_label = Extend<D, S, ArcType::Job, Mutability::Mut>(label, arc);
                            if (!new_label) {
#ifdef UNREACHABLE_DOMINANCE
                                set_job_unreachable(label->unreachable_bitmap, arc.to);
#endif
                                continue;
                            }
                            if constexpr (F == Full::Partial) {
                                if constexpr (D == Direction::Forward) {
                                    if (label->resources[0] > q_point[0]) continue;
                                } else {
                                    if (label->resources[0] <= q_point[0]) continue;
                                }
                            }
                            stat_n_labels++;

                            int &to_bucket = new_label->vertex;

                            dominated                    = false;
                            const auto &to_bucket_labels = buckets[to_bucket].get_labels();
                            if constexpr (S == Stage::One) {
                                for (auto *existing_label : to_bucket_labels) {
                                    stat_n_dom++;
                                    if (label->cost < existing_label->cost) {
                                        buckets[to_bucket].remove_label(existing_label);
                                    } else {
                                        dominated = true;
                                        break;
                                    }
                                }
                            } else {
                                for (auto *existing_label : to_bucket_labels) {
                                    if (is_dominated<D, S>(new_label, existing_label)) {
                                        dominated = true;
                                        break;
                                    }
                                }
                                // dominated = new_label->check_dominance_against_vector<D, S>(to_bucket_labels);
                            }

                            if (!dominated) {
                                if constexpr (S != Stage::Enumerate) {
                                    for (auto *existing_label : to_bucket_labels) {
                                        if (is_dominated<D, S>(existing_label, new_label)) {
                                            buckets[to_bucket].remove_label(existing_label);
                                        }
                                    }
                                }

                                n_labels++;
#ifdef SORTED_LABELS
                                buckets[to_bucket].add_sorted_label(new_label);
#elif LIMITED_BUCKETS
                                buckets[to_bucket].add_label_lim(new_label, BUCKET_CAPACITY);
#else
                                buckets[to_bucket].add_label(new_label);
#endif
                                all_ext = false;
                            }
                        }
                    }
                    label->set_extended(true);
                }
            }
        } while (!all_ext);

        for (int bucket : sorted_sccs[scc_index]) {
            const auto &labels = buckets[bucket].get_labels();

            if (!labels.empty()) {
                // Use std::min_element to find the label with the minimum cost
                auto   min_label = std::min_element(labels.begin(), labels.end(),
                                                    [](const Label *a, const Label *b) { return a->cost < b->cost; });
                double min_cost  = (*min_label)->cost;
                c_bar[bucket]    = std::min(c_bar[bucket], min_cost);
            }

            for (auto phi_bucket : Phi[bucket]) { c_bar[bucket] = std::min(c_bar[bucket], c_bar[phi_bucket]); }
        }
    }

    Label *best_label = get_best_label<D>(topological_order, c_bar, sccs);

    if constexpr (D == Direction::Forward) {
        fw_best_label = best_label;
    } else {
        bw_best_label = best_label;
    }

    return c_bar;
}

/**
 * Extends the label L_prime with the given BucketArc gamma.
 *
 * @tparam D The direction of the extension (Forward or Backward).
 * @tparam S The stage of the extension (Stage::One, Stage::Two, or Stage::Three).
 * @param L_prime The label to be extended.
 * @param gamma The BucketArc to extend the label with.
 * @return A tuple containing a boolean indicating if the extension was successful and a pointer to the new
 * label.
 */

template <Direction D, Stage S, ArcType A, Mutability M>
inline Label *BucketGraph::Extend(const std::conditional_t<M == Mutability::Mut, Label *, const Label *> L_prime,
                                  const std::conditional_t<A == ArcType::Bucket, BucketArc, Arc> &gamma) noexcept {
    auto &buckets       = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &label_pool    = assign_buckets<D>(label_pool_fw, label_pool_bw);
    auto &fixed_buckets = assign_buckets<D>(fw_fixed_buckets, bw_fixed_buckets);
    // Precompute some values to avoid recalculating inside the loop
    const int    initial_job_id    = L_prime->job_id;
    const auto  &initial_resources = L_prime->resources;
    const double initial_cost      = L_prime->cost;

    int job_id = -1;
    if constexpr (A == ArcType::Bucket) {
        job_id = buckets[gamma.to_bucket].job_id;
    } else {
        job_id = gamma.to;
    }

    // Early exit if the arc is fixed
    if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
        if constexpr (Direction::Forward == D) {
            if (fixed_arcs[initial_job_id][job_id] == 1) { return nullptr; }
        } else {
            if (fixed_arcs[job_id][initial_job_id] == 1) { return nullptr; }
        }
    }

    // Early exit for enumeration
    if constexpr (S == Stage::Enumerate) {
        fmt::print("Job id: {}\n", job_id);
        if (is_job_visited(L_prime->visited_bitmap, job_id)) { return nullptr; }
    }

    // Perform 2-cycle elimination
    if (job_id == L_prime->job_id) { return nullptr; }

    // Check if job_id is in the neighborhood of initial_job_id and is visited
    size_t segment      = job_id / 64;
    size_t bit_position = job_id % 64;

    if constexpr (S != Stage::Enumerate) {
        if ((neighborhoods_bitmap[initial_job_id][segment] & (1ULL << bit_position)) &&
            is_job_visited(L_prime->visited_bitmap, job_id)) {
            return nullptr;
        }
    }

    const VRPJob &VRPJob = jobs[job_id];

    std::vector<double> new_resources(initial_resources.size());
    // print initial resources size
    for (size_t i = 0; i < initial_resources.size(); ++i) {
        if constexpr (D == Direction::Forward) {
            new_resources[i] =
                std::max(initial_resources[i] + gamma.resource_increment[i], static_cast<double>(VRPJob.lb[i]));
        } else {
            new_resources[i] =
                std::min(initial_resources[i] - gamma.resource_increment[i], static_cast<double>(VRPJob.ub[i]));
        }
    }

    for (size_t i = 0; i < new_resources.size(); ++i) {
        if constexpr (D == Direction::Forward) {
            if (new_resources[i] > VRPJob.ub[i]) { return nullptr; }
        } else {
            if (new_resources[i] < VRPJob.lb[i]) { return nullptr; }
        }
    }

    int to_bucket = get_bucket_number<D>(job_id, new_resources);

#ifdef FIX_BUCKETS
    if constexpr (S == Stage::Three || S == Stage::Four) {
        if (fixed_buckets[L_prime->vertex][to_bucket] == 1) { return nullptr; }
    }
#endif

#ifdef RCC
    ////////////////////////////////////////////
    /* CVRPSEP */
    double cvrpsep_dual = 0.0;
    if constexpr (D == Direction::Forward) {
        cvrpsep_dual = rcc_manager->getCachedDualSumForArc(initial_job_id, job_id);
    } else {
        cvrpsep_dual = rcc_manager->getCachedDualSumForArc(job_id, initial_job_id);
    }
    ////////////////////////////////////////////
#endif
    double travel_cost = getcij(initial_job_id, job_id);
    double new_cost    = initial_cost + travel_cost - VRPJob.cost;

#ifdef RCC
    new_cost -= cvrpsep_dual;
#endif

#ifdef KP_BOUND
    if constexpr (D == Direction::Forward) {
        auto kpBound = knapsackBound(L_prime);
        // fmt::print("Knapsack bound: {}\n", kpBound);

        if (kpBound > 0.0) {
            fmt::print("new_cost/kpBound: {}/{}\n", new_cost, kpBound);
            // fmt::print("Knapsack bound exceeded\n");
            return nullptr;
        }
    }
#endif

    auto new_label = label_pool.acquire();
    new_label->initialize(to_bucket, new_cost, new_resources, job_id);
    new_label->visited_bitmap = L_prime->visited_bitmap;
    set_job_visited(new_label->visited_bitmap, job_id);

#ifdef UNREACHABLE_DOMINANCE
    new_label->unreachable_bitmap = L_prime->unreachable_bitmap;
#endif
    new_label->real_cost = L_prime->real_cost + travel_cost;
    if constexpr (M == Mutability::Mut) {
        new_label->parent = static_cast<const Label *>(L_prime);
    } else {
        new_label->parent = L_prime;
    }

#ifdef SRC
    new_label->SRCmap  = L_prime->SRCmap;
    new_label->SRCcost = L_prime->SRCcost;
#endif
#ifdef SRC3
    new_label->SRCmap = L_prime->SRCmap;
#endif

    if constexpr (S != Stage::Enumerate) {
        for (size_t i = 0; i < new_label->visited_bitmap.size(); ++i) {
            uint64_t current_visited = new_label->visited_bitmap[i];

            if (current_visited == 0) continue;

            uint64_t neighborhood_mask = neighborhoods_bitmap[job_id][i];
            uint64_t bits_to_clear     = current_visited & ~neighborhood_mask;

            if (i == job_id / 64) { bits_to_clear &= ~(1ULL << (job_id % 64)); }

            new_label->visited_bitmap[i] &= ~bits_to_clear;
        }
    }

#if defined(SRC3) || defined(SRC)

    if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
        auto &cutter   = cut_storage;
        auto &SRCDuals = cutter->SRCDuals;

        for (std::size_t idx = 0; idx < cutter->size(); ++idx) {
            auto it = cutter->begin();
            std::advance(it, idx);
            const auto &cut = *it;

            const auto &baseSet      = cut.baseSet;
            const auto &baseSetorder = cut.baseSetOrder;
            const auto &neighbors    = cut.neighbors;
            const auto &multipliers  = cut.multipliers;
#endif
#ifdef SRC3
            bool bitIsSet = baseSet[segment] & (1 << bit_position);
            if (bitIsSet) {
                new_label->SRCmap[idx]++;
                if (new_label->SRCmap[idx] % 2 == 0) {
                    // fmt::print("SRC duals: {}\n", SRCDuals[id]);
                    new_label->cost -= SRCDuals[idx];
                }
            }
        }
    }
#endif

#ifdef SRC
    bool bitIsSet  = neighbors[segment] & (1ULL << bit_position);
    bool bitIsSet2 = baseSet[segment] & (1ULL << bit_position);
    if (bitIsSet) {
        new_label->SRCmap[idx] = L_prime->SRCmap[idx];
    } else {
        new_label->SRCmap[idx] = 0.0;
    }
    if (bitIsSet2) {
        double &value = new_label->SRCmap[idx];
        value += multipliers[baseSetorder[job_id]];
        if (value >= 1) {
            new_label->SRCmap[idx] -= 1;
            new_label->cost -= SRCDuals[idx];
            new_label->SRCcost += SRCDuals[idx];
        }
    }
}
}
#endif
return new_label;
}

/**
 * @brief Retrieves the best label from the bucket graph based on the given topological order, c_bar values, and
 * strongly connected components.
 *
 * This function iterates through the given topological order and for each component, it retrieves the labels
 * from the corresponding buckets in the bucket graph. It then compares the cost of each label and keeps track
 * of the label with the lowest cost. The best label, along with its associated bucket, is returned.
 *
 * @tparam D The direction of the bucket graph.
 * @param topological_order The topological order of the components.
 * @param c_bar The c_bar values.
 * @param sccs The strongly connected components.
 * @return The best label from the bucket graph.
 */
template <Direction D>
Label *BucketGraph::get_best_label(const std::vector<int> &topological_order, const std::vector<double> &c_bar,
                                   std::vector<std::vector<int>> &sccs) {
    double best_cost  = std::numeric_limits<double>::infinity();
    Label *best_label = nullptr; // Ensure this is initialized
    auto  &buckets    = assign_buckets<D>(fw_buckets, bw_buckets);

    for (int component_index : topological_order) {
        const auto &component_buckets = sccs[component_index];

        for (int bucket : component_buckets) {
            const auto &label = buckets[bucket].get_best_label();
            if (!label || label->status == -1) continue;

            if (label->cost < best_cost) {
                best_cost  = label->cost;
                best_label = label;
            }
        }
    }

    return best_label;
}

/**
 * @brief Checks if a label is dominated by a new label based on cost and resource conditions.
 *
 * @tparam D The direction of the graph traversal (Forward or Backward).
 * @tparam S The stage of the algorithm (One, Two, or Three).
 * @param new_label A pointer to the new label.
 * @param label A pointer to the label to be checked.
 * @return True if the label is dominated by the new label, false otherwise.
 */
template <Direction D, Stage S>
inline bool BucketGraph::is_dominated(Label *&new_label, Label *&label) noexcept {
    // Early return if the cost condition is not met
    if (label->cost > new_label->cost) { return false; }

    // Check resource conditions based on direction
    const auto &new_resources   = new_label->resources;
    const auto &label_resources = label->resources;

    if constexpr (D == Direction::Forward) {
        for (size_t i = 0; i < new_resources.size(); ++i) {
            if (label_resources[i] > new_resources[i]) { return false; }
        }
    } else if constexpr (D == Direction::Backward) {
        for (size_t i = 0; i < new_resources.size(); ++i) {
            if (label_resources[i] < new_resources[i]) { return false; }
        }
    }

#ifndef UNREACHABLE_DOMINANCE
    if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
        for (size_t i = 0; i < label->visited_bitmap.size(); ++i) {
            // Check if there is any client visited in label but not in new_label
            if ((label->visited_bitmap[i] & ~new_label->visited_bitmap[i]) != 0) { return false; }
        }
    }
#else
    if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
        for (size_t i = 0; i < label->visited_bitmap.size(); ++i) {
            // Combine unreachable and visited in the same check
            auto combined_label_bitmap = label->visited_bitmap[i] | label->unreachable_bitmap[i];
            // Check if there is any client visited or unreachable in label but not in new_label
            if ((combined_label_bitmap & ~new_label->unreachable_bitmap[i]) != 0) { return false; }
        }
    }
#endif
#if SRC3
    // Check SRCDuals condition for specific stages
    if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
        const auto &SRCDuals = cut_storage->SRCDuals;
        if (!SRCDuals.empty()) {
            double sumSRC = 0;
            for (size_t i = 0; i < SRCDuals.size(); ++i) {
                if ((label->SRCmap[i]) % 2 > (new_label->SRCmap[i] % 2)) { sumSRC += SRCDuals[i]; }
            }
            if (label->cost - sumSRC > new_label->cost) { return false; }
        }
    }
#endif
#ifdef SRC
    // Check SRCDuals condition for specific stages
    if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
        const auto &SRCDuals = cut_storage->SRCDuals;
        if (!SRCDuals.empty()) {
            double sumSRC = 0;
            for (size_t i = 0; i < SRCDuals.size(); ++i) {
                if (label->SRCmap[i] > new_label->SRCmap[i]) { sumSRC += SRCDuals[i]; }
            }
            if (label->cost - sumSRC > new_label->cost) { return false; }
        }
    }
#endif
    return true;
}

/**
 * @brief Checks if element 'a' precedes element 'b' in the given strongly connected components (SCCs).
 *
 * This function takes a vector of SCCs and two elements 'a' and 'b' as input. It searches for 'a' and 'b' in
 * the SCCs and determines if 'a' precedes 'b' in the SCC list.
 *
 * @tparam T The type of elements in the SCCs.
 * @param sccs The vector of SCCs.
 * @param a The element 'a' to check.
 * @param b The element 'b' to check.
 * @return True if 'a' precedes 'b' in the SCC list, false otherwise.
 */
template <typename T>
inline bool precedes(const std::vector<std::vector<T>> &sccs, const T &a, const T &b) {
    auto it_scc_a = sccs.end();
    auto it_scc_b = sccs.end();

    for (auto it = sccs.begin(); it != sccs.end(); ++it) {
        const auto &scc = *it;

        // Use std::find once for each SCC, checking both a and b in the same iteration
        auto it_a = std::find(scc.begin(), scc.end(), a);
        auto it_b = std::find(scc.begin(), scc.end(), b);

        if (it_a != scc.end()) { it_scc_a = it; }
        if (it_b != scc.end()) { it_scc_b = it; }

        // If both are found in the same SCC, return false
        if (it_scc_a == it_scc_b && it_scc_a != sccs.end()) { return false; }

        // Early exit if both are found in different SCCs
        if (it_scc_a != sccs.end() && it_scc_b != sccs.end()) { return it_scc_a < it_scc_b; }
    }

    // If a and/or b are not found, return false
    return false;
}

/**
 * @brief Determines if a label is dominated in component-wise smaller buckets.
 *
 * This function checks if a given label is dominated by any other label in component-wise smaller buckets.
 * The dominance is determined based on the cost and order of the buckets.
 *
 * @tparam D The direction of the buckets.
 * @tparam S The stage of the buckets.
 * @param L A pointer to the label to be checked.
 * @param bucket The index of the current bucket.
 * @param c_bar The vector of cost values for each bucket.
 * @param Bvisited The set of visited buckets.
 * @param bucket_order The order of the buckets.
 * @return True if the label is dominated, false otherwise.
 */
template <Direction D, Stage S>
inline bool BucketGraph::DominatedInCompWiseSmallerBuckets(Label *L, int bucket, std::vector<double> &c_bar,
                                                           std::vector<uint64_t>               &Bvisited,
                                                           const std::vector<std::vector<int>> &bucket_order) noexcept {
    auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &Phi     = assign_buckets<D>(Phi_fw, Phi_bw);

    const int       b_L = L->vertex;
    std::stack<int> bucketStack;
    bucketStack.push(bucket);

    while (!bucketStack.empty()) {
        int currentBucket = bucketStack.top();
        bucketStack.pop();

        // Mark the bucket as visited
        const size_t segment      = currentBucket / 64;
        const size_t bit_position = currentBucket % 64;
        Bvisited[segment] |= (1ULL << bit_position);

        // Check cost and precedence
        if (L->cost < c_bar[currentBucket] && ::precedes<int>(bucket_order, currentBucket, b_L)) { return false; }

        if (b_L != currentBucket) {
            const auto &bucket_labels = buckets[currentBucket].get_labels();
            for (auto *label : bucket_labels) {
                if (is_dominated<D, S>(L, label)) { return true; }
            }
        }

        // Add unvisited neighboring buckets to the stack
        for (const int b_prime : Phi[currentBucket]) {
            const size_t segment_prime      = b_prime / 64;
            const size_t bit_position_prime = b_prime % 64;

            if ((Bvisited[segment_prime] & (1ULL << bit_position_prime)) == 0) { bucketStack.push(b_prime); }
        }
    }

    return false;
}

/**
 * Performs the bi-labeling algorithm on the BucketGraph.
 *
 * @param q_star The vector of doubles representing the resource constraints.
 * @param S The stage of the algorithm to run (Stage::One, Stage::Two, or Stage::Three).
 * @return A vector of Label pointers representing the best labels obtained from the algorithm.
 */

template <Stage S>
std::vector<Label *> BucketGraph::bi_labeling_algorithm(std::vector<double> q_star) {

    if constexpr (S == Stage::Three) {
        heuristic_fixing<S>(q_star);
    } else if constexpr (S == Stage::Four) {
        if (first_reset) {
            reset_fixed();
            first_reset = false;
        }
    }

#ifdef FIX_BUCKETS
    if constexpr (S == Stage::Four) { bucket_fixing<S>(q_star); }
#endif

    reset_pool();
    common_initialization();

    std::vector<double> forward_cbar(fw_buckets.size(), std::numeric_limits<double>::infinity());
    std::vector<double> backward_cbar(bw_buckets.size(), std::numeric_limits<double>::infinity());

    run_labeling_algorithms<S, Full::Partial>(forward_cbar, backward_cbar, q_star);

    // Best complete path obtained in the two algorithms above
    auto best_label = label_pool_fw.acquire();

    if (check_feasibility(fw_best_label, bw_best_label)) {
        best_label = compute_label(fw_best_label, bw_best_label);
    } else {
        best_label->cost         = 0.0;
        best_label->real_cost    = std::numeric_limits<double>::infinity();
        best_label->jobs_covered = {};
    }

    merged_labels.push_back(best_label);

    if constexpr (S == Stage::Enumerate) { fmt::print("Labels generated, concatenating...\n"); }

    const size_t          n_segments = fw_buckets_size / 64 + 1;
    std::vector<uint64_t> Bvisited(n_segments, 0);

    for (auto bucket = 0; bucket < fw_buckets_size; ++bucket) {
        auto       &current_bucket = fw_buckets[bucket];
        const auto &labels         = current_bucket.get_labels();
        for (const Label *L : labels) {

#ifndef ORIGINAL_ARCS
            const auto &to_arcs = jobs[L->job_id].get_arcs<Direction::Forward>();
#else
            const auto &to_arcs = fw_buckets[bucket].get_bucket_arcs(true);
#endif
            for (const auto &arc : to_arcs) {
                auto &to_job = arc.to;
                if constexpr (S == Stage::Three) {
                    if (fixed_arcs[L->job_id][to_job] == 1) { continue; }
                }

                // Extend the current label based on the current stage
                auto L_prime = Extend<Direction::Forward, S, ArcType::Job, Mutability::Const>(L, arc);
                if (!L_prime || L_prime->resources[0] < q_star[0]) continue;
                auto b_prime = L_prime->vertex;

#ifndef BUCKET_FIXING
                // if (fw_fixed_buckets[bucket][b_prime] == 1) { continue; }
#endif
                std::memset(Bvisited.data(), 0, Bvisited.size() * sizeof(uint64_t));
                ConcatenateLabel<S>(L, b_prime, best_label, Bvisited, q_star);
            }
        }
    }

    std::sort(merged_labels.begin(), merged_labels.end(),
              [](const Label *a, const Label *b) { return a->cost < b->cost; });

#ifdef RIH
    if constexpr (S == Stage::Two || S == Stage::Three) {
        // Call the labeling heuristic improvement function
        std::priority_queue<Label *, std::vector<Label *>, LabelComparator> best_labels_in;
        std::priority_queue<Label *, std::vector<Label *>, LabelComparator> best_labels_out;

        auto LABELS_MAX = 2;
        // Populate the best_labels_in queue with the final merged_labels
        auto label_count = 0;
        for (auto &label : merged_labels) {
            best_labels_in.push(label);
            if (++label_count >= LABELS_MAX) break;
        }

        RIH2(best_labels_in, best_labels_out, LABELS_MAX);

        while (!best_labels_out.empty()) {
            best_labels_in.push(best_labels_out.top());
            best_labels_out.pop();
        }

        /*
        RIH1(best_labels_in, best_labels_out, LABELS_MAX);
        while (!best_labels_out.empty()) {
             best_labels_in.push(best_labels_out.top());
             best_labels_out.pop();
        }
*/

        RIH3(best_labels_in, best_labels_out, LABELS_MAX);
        while (!best_labels_out.empty()) {
            best_labels_in.push(best_labels_out.top());
            best_labels_out.pop();
        }

        RIH4(best_labels_in, best_labels_out, LABELS_MAX);

        while (!best_labels_in.empty()) {
            merged_labels.push_back(best_labels_in.top());
            best_labels_in.pop();
        }
    }
#endif

    return merged_labels;
}

/**
 * Concatenates the label L with the bucket b and updates the best label pbest.
 *
 * @param L The label to be concatenated.
 * @param b The bucket index.
 * @param pbest The best label found so far.
 * @param Bvisited The set of visited buckets.
 * @param q_star The vector of costs.
 */
template <Stage S>
void BucketGraph::ConcatenateLabel(const Label *&L, int &b, Label *&pbest, std::vector<uint64_t> &Bvisited,
                                   const std::vector<double> &q_star) {
    // Create a stack for iterative processing
    std::stack<int> bucket_stack;
    bucket_stack.push(b);

    while (!bucket_stack.empty()) {
        // Pop the next bucket from the stack
        int current_bucket = bucket_stack.top();
        bucket_stack.pop();

        // Mark the bucket as visited
        const size_t segment      = current_bucket / 64;
        const size_t bit_position = current_bucket % 64;
        Bvisited[segment] |= (1ULL << bit_position);

        int    bucketLjob      = L->job_id;
        int    bucketLprimejob = bw_buckets[current_bucket].job_id;
        double cost            = getcij(bucketLjob, bucketLprimejob);

#ifdef RCC
        cost -= rcc_manager->getCachedDualSumForArc(bucketLjob, bucketLprimejob);
#endif
        double L_cost_plus_cost = L->cost + cost;

#ifdef SRC
        auto cutter   = cut_storage;
        auto SRCDuals = cutter->SRCDuals;
#endif

        // Early exit based on cost comparison
        if constexpr (S != Stage::Enumerate) {
            if (L_cost_plus_cost + bw_c_bar[current_bucket] >= pbest->cost) { continue; }
        } else {
            if (L_cost_plus_cost + bw_c_bar[current_bucket] >= gap) { continue; }
        }

        const VRPJob &L_last_job = jobs[L->job_id];
        auto         &bucket     = bw_buckets[current_bucket];
        const auto   &labels     = bucket.get_labels(); // Assuming get_labels() returns a vector of Label*

        for (auto &L_bw : labels) {
            if (L_bw->job_id == L->job_id) { continue; }

            // Loop through the resource dimensions
            if (L->resources[0] + cost + L_last_job.duration > L_bw->resources[0]) { continue; }

            if constexpr (R_SIZE > 1) {
                bool valid = true;

                for (int r = 1; r < intervals.size(); ++r) {
                    if (L->resources[r] + L_last_job.consumption[r] + (R_max[r] - L_bw->resources[r]) > R_max[r]) {
                        valid = false;
                        break;
                    }
                }
                if (!valid) { continue; }
            }

            double candidate_cost = L_cost_plus_cost + L_bw->cost;

            /*
            #ifdef SRC
                        auto counter = 0;
                        for (auto it = cutter->begin(); it < cutter->end(); ++it) {

                            if (SRCDuals[it->id] == 0) continue;
                            if (L->SRCmap[it->id] + L_bw->SRCmap[it->id] >= 1) { candidate_cost += SRCDuals[it->id];
            }
                        }
            #endif
            */
            // Use bitwise operations for the visited bitmap comparison
            if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
                bool visited_overlap = false;
                for (size_t i = 0; i < L->visited_bitmap.size(); ++i) {
                    if ((L->visited_bitmap[i] & L_bw->visited_bitmap[i]) != 0) {
                        visited_overlap = true;
                        break; // No need to check further if there's an overlap
                    }
                }
                if (visited_overlap) continue;
            }
            // Early exit based on candidate cost
            if constexpr (S != Stage::Enumerate) {
                if (candidate_cost >= pbest->cost) continue;
            } else {
                if (candidate_cost >= gap) continue;
            }

            // Compute the new label and store it
            pbest = compute_label(L, L_bw);
            merged_labels.push_back(pbest); // Consider reserving space in merged_labels if frequently reallocating
        }

        // Push adjacent buckets onto the stack for further processing
        for (int b_prime : Phi_bw[current_bucket]) {
            const size_t segment_prime      = b_prime / 64;
            const size_t bit_position_prime = b_prime % 64;
            if ((Bvisited[segment_prime] & (1ULL << bit_position_prime)) == 0) {
                bucket_stack.push(b_prime); // Push to the stack instead of recursive call
            }
        }
    }
}

/**
 * @brief Handles the computation of Strongly Connected Components (SCCs) for the BucketGraph.
 *
 * This function processes the bucket graph to identify SCCs using Tarjan's algorithm. It extends the bucket
 * graph with arcs defined by the Phi sets, computes the SCCs, and orders them topologically. It also sorts the
 * buckets within each SCC based on their lower or upper bounds, depending on the direction. Finally, it splits
 * the arcs for each SCC and removes duplicates.
 *
 * @tparam D The direction of the graph traversal, either Forward or Backward.
 */
template <Direction D>
void BucketGraph::SCC_handler() {
    auto &Phi          = assign_buckets<D>(Phi_fw, Phi_bw);
    auto &buckets      = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &bucket_graph = assign_buckets<D>(fw_bucket_graph, bw_bucket_graph);

    std::unordered_map<int, std::vector<int>> extended_bucket_graph = bucket_graph;

    // Extend the bucket graph with arcs defined by the Phi sets
    if constexpr (D == Direction::Forward) {
        for (auto i = 0; i < extended_bucket_graph.size(); ++i) {
            auto phi_set = Phi[i];
            if (phi_set.empty()) continue;
            for (auto &phi_bucket : phi_set) { extended_bucket_graph[phi_bucket].push_back(i); }
        }
    } else {
        for (auto i = 0; i < extended_bucket_graph.size(); ++i) {
            auto phi_set = Phi[i];
            if (phi_set.empty()) continue;
            for (auto &phi_bucket : phi_set) { extended_bucket_graph[phi_bucket].push_back(i); }
        }
    }

    SCC scc_finder;
    scc_finder.convertFromUnorderedMap(extended_bucket_graph); // print extended bucket graph

    auto sccs              = scc_finder.tarjanSCC();
    auto topological_order = scc_finder.topologicalOrderOfSCCs(sccs);

    // print SCCs and buckets in it
    constexpr auto blue  = "\033[34m";
    constexpr auto reset = "\033[0m";
    if constexpr (D == Direction::Forward) {
        fmt::print("FW SCCs: \n");
    } else {
        fmt::print("BW SCCs: \n");
    }
    for (auto scc : topological_order) {
        if constexpr (D == Direction::Forward) {
            fmt::print("{}({}) -> {}", blue, scc, reset);
        } else {
            fmt::print("{}({}) -> {}", blue, scc, reset);
        }
        for (auto &bucket : sccs[scc]) { fmt::print("{} ", bucket); }
    }
    fmt::print("\n");

    std::vector<std::vector<int>> ordered_sccs;
    ordered_sccs.reserve(sccs.size()); // Reserve space for all SCCs
    for (int i : topological_order) {
        ordered_sccs.push_back(sccs[i]); // Use std::move to avoid copying
    }

    auto sorted_sccs = sccs;
    for (auto &scc : sorted_sccs) {
        if constexpr (D == Direction::Forward) {
            std::sort(scc.begin(), scc.end(), [&buckets](int a, int b) { return buckets[a].lb[0] < buckets[b].lb[0]; });
        } else {
            std::sort(scc.begin(), scc.end(), [&buckets](int a, int b) { return buckets[a].ub[0] > buckets[b].ub[0]; });
        }
    }

    // iterate over jobs
    for (auto &job : jobs) {
        if constexpr (D == Direction::Forward) {
            job.fw_arcs_scc.resize(sccs.size());
        } else {
            job.bw_arcs_scc.resize(sccs.size());
        }
    }

    // Split arcs for each SCC
    auto scc_ctr = 0;
    for (const auto &scc : sccs) {
        // Iterate over each bucket in the SCC
        for (int bucket : scc) {
            int     from_job_id = buckets[bucket].job_id; // Get the source job ID
            VRPJob &job         = jobs[from_job_id];      // Access the corresponding job
            //  Define filtered arcs depending on the direction
            if constexpr (D == Direction::Forward) {
                std::vector<Arc> &filtered_fw_arcs = jobs[from_job_id].fw_arcs_scc[scc_ctr]; // For forward direction

                // Iterate over the arcs from the current bucket
                const auto &bucket_arcs = buckets[bucket].template get_bucket_arcs<Direction::Forward>();
                for (const auto &arc : bucket_arcs) {
                    int to_job_id = buckets[arc.to_bucket].job_id; // Get the destination job ID

                    // Search for the arc from `from_job_id` to `to_job_id` in the job's arcs
                    auto it = std::find_if(job.fw_arcs.begin(), job.fw_arcs.end(),
                                           [&to_job_id](const Arc &a) { return a.to == to_job_id; });

                    // If both jobs are within the current SCC, retain the arc
                    if (it != job.fw_arcs.end()) {
                        // Add the arc to the filtered arcs
                        filtered_fw_arcs.push_back(*it); // Forward arcs
                    }
                }
            } else {
                std::vector<Arc> &filtered_bw_arcs = jobs[from_job_id].bw_arcs_scc[scc_ctr]; // For forward direction

                // Iterate over the arcs from the current bucket
                const auto &bucket_arcs = buckets[bucket].template get_bucket_arcs<Direction::Backward>();
                for (const auto &arc : bucket_arcs) {
                    int to_job_id = buckets[arc.to_bucket].job_id; // Get the destination job ID

                    // Search for the arc from `from_job_id` to `to_job_id` in the job's arcs
                    auto it = std::find_if(job.bw_arcs.begin(), job.bw_arcs.end(),
                                           [&to_job_id](const Arc &a) { return a.to == to_job_id; });

                    // If both jobs are within the current SCC, retain the arc
                    if (it != job.bw_arcs.end()) {
                        // Add the arc to the filtered arcs
                        filtered_bw_arcs.push_back(*it); // Forward arcs
                        // fmt::print("Adding arc from {} to {}\n", from_job_id, to_job_id);
                    }
                }
            }
        }

        // Increment SCC counter
        ++scc_ctr;
    }

    for (auto &job : jobs) {
        if constexpr (D == Direction::Forward) {
            // Iterate over all SCCs for each job
            for (auto &fw_arcs_scc : job.fw_arcs_scc) {
                // Sort arcs by from_bucket and to_bucket
                std::sort(fw_arcs_scc.begin(), fw_arcs_scc.end(),
                          [](const Arc &a, const Arc &b) { return std::tie(a.from, a.to) < std::tie(b.from, b.to); });

                // Remove consecutive duplicates
                auto last = std::unique(fw_arcs_scc.begin(), fw_arcs_scc.end(),
                                        [](const Arc &a, const Arc &b) { return a.from == b.from && a.to == b.to; });

                // Erase the duplicates from the vector
                fw_arcs_scc.erase(last, fw_arcs_scc.end());
            }
        } else {
            // Iterate over all SCCs for each job
            for (auto &bw_arcs_scc : job.bw_arcs_scc) {
                // Sort arcs by from_bucket and to_bucket
                std::sort(bw_arcs_scc.begin(), bw_arcs_scc.end(),
                          [](const Arc &a, const Arc &b) { return std::tie(a.from, a.to) < std::tie(b.from, b.to); });

                // Remove consecutive duplicates
                auto last = std::unique(bw_arcs_scc.begin(), bw_arcs_scc.end(),
                                        [](const Arc &a, const Arc &b) { return a.from == b.from && a.to == b.to; });

                // Erase the duplicates from the vector
                bw_arcs_scc.erase(last, bw_arcs_scc.end());
            }
        }
    }

    if constexpr (D == Direction::Forward) {
        fw_ordered_sccs      = ordered_sccs;
        fw_topological_order = topological_order;
        fw_sccs              = sccs;
        fw_sccs_sorted       = sorted_sccs;
    } else {
        bw_ordered_sccs      = ordered_sccs;
        bw_topological_order = topological_order;
        bw_sccs              = sccs;
        bw_sccs_sorted       = sorted_sccs;
    }
}

/**
 * @brief Get the opposite bucket number for a given bucket index.
 *
 * This function retrieves the opposite bucket number based on the current bucket index
 * and the specified direction. It determines the job and bounds of the current bucket,
 * then calculates the opposite bucket index using the appropriate direction.
 *
 * @tparam D The direction (Forward or Backward) to determine the opposite bucket.
 * @param current_bucket_index The index of the current bucket.
 * @return The index of the opposite bucket.
 */
template <Direction D>
int BucketGraph::get_opposite_bucket_number(int current_bucket_index) {
    // Get the current bucket's job and bounds
    auto &current_bucket =
        (D == Direction::Forward) ? fw_buckets[current_bucket_index] : bw_buckets[current_bucket_index];
    int job = current_bucket.job_id;
    int lb  = current_bucket.lb[0];
    int ub  = current_bucket.ub[0];

    auto &theJob = jobs[job];

    // Find the opposite bucket using the appropriate direction
    int opposite_bucket_index = -1;
    if constexpr (D == Direction::Forward) {
        std::vector<double> value = {static_cast<double>(std::max(ub, theJob.start_time))};
        opposite_bucket_index     = get_bucket_number<Direction::Backward>(job, value);
    } else {
        std::vector<double> value = {static_cast<double>(std::min(lb, theJob.end_time))};
        opposite_bucket_index     = get_bucket_number<Direction::Forward>(job, value);
    }

    return opposite_bucket_index;
}
