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
        std::vector<int> base_intervals(MAIN_RESOURCES);
        std::vector<int> first_lb(MAIN_RESOURCES);
        std::vector<int> first_ub(MAIN_RESOURCES);
        std::vector<int> remainders(MAIN_RESOURCES);

        // Compute base intervals, remainders, and bounds for each dimension
        for (int r = 0; r < MAIN_RESOURCES; ++r) {
            base_intervals[r] = (R_max[r] - R_min[r] + 1) / intervals[r].interval;
            remainders[r]     = static_cast<int>(R_max[r] - R_min[r] + 1) % intervals[r].interval;
            first_lb[r]       = vrpjob.lb[r]; // Lower bound for each dimension
            first_ub[r]       = vrpjob.ub[r]; // Upper bound for each dimension
        }

        int bucket_index = 0;
        int multiplier   = 1; // This will be used to compute the final bucket index across dimensions

        if constexpr (D == Direction::Forward) {
            // Iterate over each resource value and compute the bucket index for forward direction
            for (int r = MAIN_RESOURCES - 1; r >= 0; --r) {
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
            for (int r = MAIN_RESOURCES - 1; r >= 0; --r) {
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
        const int   base_interval     = (R_max[0] - R_min[0] + 1) / intervals[0].interval;
        int         value             = static_cast<int>(resource_values_vec[0]);

        const auto &vrpjob = jobs[job];

        if constexpr (D == Direction::Forward) {
            const int first_lb = vrpjob.lb[0];
            return (value - first_lb) / base_interval + start_index;
        } else if constexpr (D == Direction::Backward) {
            const int first_ub = vrpjob.ub[0];
            return (first_ub - value) / base_interval + start_index;
        }
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
    int              num_intervals = MAIN_RESOURCES;
    std::vector<int> total_ranges(num_intervals);
    std::vector<int> base_intervals(num_intervals);
    std::vector<int> remainders(num_intervals);

    // Determine the base interval and other relevant values for each resource
    for (int r = 0; r < num_intervals; ++r) {
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
    // Mutex to synchronize access to shared resources
    auto buckets_mutex = std::mutex();

    // Clear the appropriate bucket graph (forward or backward)
    if constexpr (D == Direction::Forward) {
        fw_bucket_graph.clear(); // Clear the forward bucket graph
    } else {
        bw_bucket_graph.clear(); // Clear the backward bucket graph
    }

    // Assign the forward or backward fixed buckets and other bucket-related structures
    auto &fixed_buckets     = assign_buckets<D>(fw_fixed_buckets, bw_fixed_buckets);
    auto &buckets           = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &num_buckets       = assign_buckets<D>(num_buckets_fw, num_buckets_bw);
    auto &num_buckets_index = assign_buckets<D>(num_buckets_index_fw, num_buckets_index_bw);

    // Compute base intervals for each resource dimension based on R_max and R_min
    std::vector<int> base_intervals(intervals.size());
    for (int r = 0; r < intervals.size(); ++r) {
        base_intervals[r] = std::floor((R_max[r] - R_min[r] + 1) / static_cast<int>(intervals[r].interval));
    }

    // Clear all buckets in parallel, removing any existing arcs
    std::for_each(std::execution::par_unseq, buckets.begin(), buckets.end(), [&](auto &bucket) {
        bucket.clear();                             // Clear bucket data
        bucket.clear_arcs(D == Direction::Forward); // Clear arcs in the bucket
    });

    auto add_arcs_for_job = [&](const VRPJob &job, int from_bucket, std::vector<double> &res_inc,
                                std::vector<std::pair<int, int>> &local_arcs) {
        // Retrieve the arcs for the job in the given direction (Forward/Backward)
        auto arcs = job.get_arcs<D>();

        // Iterate over all arcs of the job
        for (const auto &arc : arcs) {
            const auto &next_job = jobs[arc.to]; // Get the destination job of the arc

            // Skip self-loops (no arc from a job to itself)
            if (job.id == next_job.id) continue;

            // Calculate travel cost and cost increment based on job's properties
            const auto travel_cost = getcij(job.id, next_job.id);
            double     cost_inc    = travel_cost - next_job.cost;
            res_inc[0]             = travel_cost + job.duration; // Update resource increment based on job duration

            // Iterate over all possible destination buckets for the next job
            for (int j = 0; j < num_buckets[next_job.id]; ++j) {
                int to_bucket = j + num_buckets_index[next_job.id];
                if (from_bucket == to_bucket) continue; // Skip arcs that loop back to the same bucket

                if (fixed_buckets[from_bucket][to_bucket] == 1) continue; // Skip fixed arcs

                bool valid_arc = true;
                for (int r = 0; r < res_inc.size(); ++r) {
                    // Forward direction: Check that resource increment doesn't exceed upper bounds
                    if constexpr (D == Direction::Forward) {
                        if (buckets[from_bucket].lb[r] + res_inc[r] > buckets[to_bucket].ub[r]) {
                            valid_arc = false;
                            break;
                        }
                    }
                    // Backward direction: Check that resource decrement doesn't drop below lower bounds
                    else if constexpr (D == Direction::Backward) {
                        if (buckets[from_bucket].ub[r] - res_inc[r] < buckets[to_bucket].lb[r]) {
                            valid_arc = false;
                            break;
                        }
                    }
                }
                if (!valid_arc) continue; // Skip invalid arcs

                // Further refine arc validity based on the base intervals and job bounds
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
                if (!valid_arc) continue; // Skip invalid arcs

                // Store the arc data locally before committing it to the global structure
                local_arcs.emplace_back(from_bucket, to_bucket);
                double              local_cost_inc = cost_inc;
                std::vector<double> local_res_inc  = res_inc;

                // Add the arc to the global structure and the bucket
                {
                    std::lock_guard<std::mutex> lock(buckets_mutex); // Lock the mutex to ensure thread safety
                    add_arc<D>(from_bucket, to_bucket, local_res_inc, local_cost_inc); // Add the arc globally
                    buckets[from_bucket].add_bucket_arc(from_bucket, to_bucket, local_res_inc, local_cost_inc,
                                                        D == Direction::Forward, false); // Add the arc to the bucket
                }
            }
        }
    };

    // Iterate over all jobs in parallel, generating arcs for each
    std::for_each(std::execution::par_unseq, jobs.begin(), jobs.end(), [&](const VRPJob &VRPJob) {
        std::vector<double>              res_inc = {static_cast<double>(VRPJob.duration)}; // Resource increment vector
        std::vector<std::pair<int, int>> local_arcs;                                       // Local storage for arcs

        // Generate arcs for all buckets associated with the current job
        for (int i = 0; i < num_buckets[VRPJob.id]; ++i) {
            int from_bucket = i + num_buckets_index[VRPJob.id];         // Determine the source bucket
            add_arcs_for_job(VRPJob, from_bucket, res_inc, local_arcs); // Add arcs for this job and bucket
        }
    });
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

    for (const int component_index : topological_order) {
        const auto &component_buckets = sccs[component_index];

        for (const int bucket : component_buckets) {
            const auto &label = buckets[bucket].get_best_label();
            if (!label) continue;

            if (label->cost < best_cost) {
                best_cost  = label->cost;
                best_label = label;
            }
        }
    }

    return best_label;
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
void BucketGraph::ConcatenateLabel(const Label *L, int &b, Label *&pbest, std::vector<uint64_t> &Bvisited) {
    // Use a vector for iterative processing as a stack
    std::vector<int> bucket_stack;
    bucket_stack.reserve(10);
    bucket_stack.push_back(b);

    const auto &L_job_id    = L->job_id;
    const auto &L_resources = L->resources;
    const auto &L_last_job  = jobs[L_job_id];

    while (!bucket_stack.empty()) {
        // Pop the next bucket from the stack (vector back)
        int current_bucket = bucket_stack.back();
        bucket_stack.pop_back();

        // Mark the bucket as visited
        const size_t segment      = current_bucket >> 6; // Equivalent to current_bucket / 64
        const size_t bit_position = current_bucket & 63; // Equivalent to current_bucket % 64

        Bvisited[segment] |= (1ULL << bit_position);

        const auto  &bucketLprimejob = bw_buckets[current_bucket].job_id;
        const double cost            = getcij(L_job_id, bucketLprimejob);

#ifdef RCC
        cost -= rcc_manager->getCachedDualSumForArc(L_job_id, bucketLprimejob);
#endif

#ifdef SRC
        decltype(cut_storage)            cutter   = nullptr;
        decltype(cut_storage->SRCDuals) *SRCDuals = nullptr;

        if constexpr (S > Stage::Three) {
            cutter   = cut_storage;       // Initialize cutter
            SRCDuals = &cutter->SRCDuals; // Initialize SRCDuals
        }
#endif

        double L_cost_plus_cost = L->cost + cost;

        // Early exit based on cost comparison
        if ((S != Stage::Enumerate && L_cost_plus_cost + bw_c_bar[current_bucket] >= pbest->cost) ||
            (S == Stage::Enumerate && L_cost_plus_cost + bw_c_bar[current_bucket] >= gap)) {
            continue;
        }

        // Get the labels for the current bucket
        auto       &bucket = bw_buckets[current_bucket];
        const auto &labels = bucket.get_labels();

        for (const auto &L_bw : labels) {
            if (L_bw->job_id == L_job_id || !check_feasibility(L, L_bw)) continue;
            double candidate_cost = L_cost_plus_cost + L_bw->cost;

#ifdef SRC
            if constexpr (S > Stage::Three) {
                for (auto it = cutter->begin(); it < cutter->end(); ++it) {
                    if ((*SRCDuals)[it->id] == 0) continue;
                    if (L->SRCmap[it->id] + L_bw->SRCmap[it->id] >= 1) { candidate_cost -= (*SRCDuals)[it->id]; }
                }
            }
#endif

            // Check for visited overlap and skip if true
            if constexpr (S >= Stage::Three) {
                bool visited_overlap = false;
                for (size_t i = 0; i < L->visited_bitmap.size(); ++i) {
                    if (L->visited_bitmap[i] & L_bw->visited_bitmap[i]) {
                        visited_overlap = true;
                        break;
                    }
                }
                if (visited_overlap) continue;
            }

            // Early exit based on candidate cost
            if ((S != Stage::Enumerate && candidate_cost >= pbest->cost) ||
                (S == Stage::Enumerate && candidate_cost >= gap)) {
                continue;
            }

            // Compute and store the new label
            pbest = compute_label(L, L_bw);
            merged_labels.push_back(pbest);
        }

        // Add unvisited neighboring buckets to the stack (vector back)
        for (int b_prime : Phi_bw[current_bucket]) {
            const size_t segment_prime      = b_prime >> 6;
            const size_t bit_position_prime = b_prime & 63;
            if (!(Bvisited[segment_prime] & (1ULL << bit_position_prime))) {
                bucket_stack.push_back(b_prime); // Add to the vector stack
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
    for (auto i = 0; i < extended_bucket_graph.size(); ++i) {
        auto phi_set = Phi[i];
        if (phi_set.empty()) continue;
        for (auto &phi_bucket : phi_set) { extended_bucket_graph[phi_bucket].push_back(i); }
    }

    SCC scc_finder;
    scc_finder.convertFromUnorderedMap(extended_bucket_graph); // print extended bucket graph

    auto sccs              = scc_finder.tarjanSCC();
    auto topological_order = scc_finder.topologicalOrderOfSCCs(sccs);

#ifdef VERBOSE
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
#endif

    std::vector<std::vector<int>> ordered_sccs;
    ordered_sccs.reserve(sccs.size()); // Reserve space for all SCCs
    for (int i : topological_order) { ordered_sccs.push_back(sccs[i]); }

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
                const auto &bucket_arcs = buckets[bucket].template get_bucket_arcs<D>();
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
                const auto &bucket_arcs = buckets[bucket].template get_bucket_arcs<D>();
                for (const auto &arc : bucket_arcs) {
                    int to_job_id = buckets[arc.to_bucket].job_id; // Get the destination job ID

                    // Search for the arc from `from_job_id` to `to_job_id` in the job's arcs
                    auto it = std::find_if(job.bw_arcs.begin(), job.bw_arcs.end(),
                                           [&to_job_id](const Arc &a) { return a.to == to_job_id; });

                    // If both jobs are within the current SCC, retain the arc
                    if (it != job.bw_arcs.end()) {
                        // Add the arc to the filtered arcs
                        filtered_bw_arcs.push_back(*it); // Forward arcs
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

    // TODO: adjust to multi-resource case
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

template <Stage S>
void BucketGraph::bucket_fixing(const std::vector<double> &q_star) {
    // Stage 4 bucket arc fixing
    if (!fixed) {
        fixed = true;
        common_initialization();

        std::vector<double> forward_cbar(fw_buckets.size(), std::numeric_limits<double>::infinity());
        std::vector<double> backward_cbar(bw_buckets.size(), std::numeric_limits<double>::infinity());

        // if constexpr (S == Stage::Two) {
        run_labeling_algorithms<Stage::Four, Full::Full>(forward_cbar, backward_cbar, q_star);

        gap = incumbent - (relaxation + std::min(0.0, min_red_cost));

        // check if gap is -inf and early exit, due to IPM
        if (gap < 0) { return; }
        // print_info("Running arc elimination with gap: {}\n", gap);
        fw_c_bar = forward_cbar;
        bw_c_bar = backward_cbar;

        PARALLEL_SECTIONS(
            bi_sched,
            SECTION {
                // Section 1: Forward direction
                BucketArcElimination<Direction::Forward>(gap);
                ObtainJumpBucketArcs<Direction::Forward>();
            },
            SECTION {
                // Section 2: Backward direction
                BucketArcElimination<Direction::Backward>(gap);
                ObtainJumpBucketArcs<Direction::Backward>();
            });

        generate_arcs();
    }
}

/**
 * @brief Applies heuristic fixing to the current solution.
 *
 * This function modifies the current solution based on the heuristic
 * fixing strategy using the provided vector of values.
 *
 * @param q_star A vector of double values representing the heuristic
 *               fixing parameters.
 */
template <Stage S>
void BucketGraph::heuristic_fixing(const std::vector<double> &q_star) {
    // Stage 3 fixing heuristic
    reset_pool();
    reset_fixed();
    common_initialization();

    std::vector<double> forward_cbar(fw_buckets.size(), std::numeric_limits<double>::infinity());
    std::vector<double> backward_cbar(bw_buckets.size(), std::numeric_limits<double>::infinity());

    if constexpr (S == Stage::Three) {
        run_labeling_algorithms<Stage::Two, Full::Partial>(forward_cbar, backward_cbar, q_star);
    } else {
        run_labeling_algorithms<Stage::Three, Full::Partial>(forward_cbar, backward_cbar, q_star);
    }
    std::vector<std::vector<Label *>> fw_labels_map(jobs.size());
    std::vector<std::vector<Label *>> bw_labels_map(jobs.size());

    auto group_labels = [&](auto &buckets, auto &labels_map) {
        for (auto &bucket : buckets) {
            for (auto &label : bucket.get_labels()) {
                labels_map[label->job_id].push_back(label); // Directly index using job_id
            }
        }
    };

    // Create tasks for forward and backward labels grouping
    auto forward_task = stdexec::schedule(bi_sched) | stdexec::then([&]() { group_labels(fw_buckets, fw_labels_map); });
    auto backward_task =
        stdexec::schedule(bi_sched) | stdexec::then([&]() { group_labels(bw_buckets, bw_labels_map); });

    // Execute the tasks in parallel
    auto work = stdexec::when_all(std::move(forward_task), std::move(backward_task));

    stdexec::sync_wait(std::move(work));

    auto num_fixes = 0;
    //  Function to find the minimum cost label in a vector of labels
    auto find_min_cost_label = [](const std::vector<Label *> &labels) -> const Label * {
        return *std::min_element(labels.begin(), labels.end(),
                                 [](const Label *a, const Label *b) { return a->cost < b->cost; });
    };
    for (const auto &job_I : jobs) {
        const auto &fw_labels = fw_labels_map[job_I.id];
        if (fw_labels.empty()) continue; // Skip if no labels for this job_id

        for (const auto &job_J : jobs) {
            if (job_I.id == job_J.id) continue; // Compare based on id (or other key field)
            const auto &bw_labels = bw_labels_map[job_J.id];
            if (bw_labels.empty()) continue; // Skip if no labels for this job_id

            const Label *min_fw_label = find_min_cost_label(fw_labels);
            const Label *min_bw_label = find_min_cost_label(bw_labels);

            if (!min_fw_label || !min_bw_label) continue;

            const VRPJob &L_last_job = jobs[min_fw_label->job_id];
            auto          cost       = getcij(min_fw_label->job_id, min_bw_label->job_id);

            // Check for infeasibility
            if (min_fw_label->resources[TIME_INDEX] + cost + L_last_job.consumption[TIME_INDEX] >
                min_bw_label->resources[TIME_INDEX]) {
                continue;
            }

            if (min_fw_label->cost + cost + min_bw_label->cost > gap) {
                fixed_arcs[job_I.id][job_J.id] = 1; // Index with job ids
                num_fixes++;
            }
        }
    }
}