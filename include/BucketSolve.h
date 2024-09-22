/**
 * @file BucketSolve.h
 * @brief Defines the BucketGraph class and its solving methods for optimization problems.
 *
 * This file contains the implementation of the `BucketGraph` class, which solves a multi-stage bi-labeling
 * optimization problem using bucket graphs. The `BucketGraph` class handles:
 *
 * - Solving bucket graph optimization problems using multi-stage labeling algorithms.
 * - Adaptive handling of terminal time adjustments based on label distribution.
 * - Adaptive stage transitions to refine solutions based on inner objectives and iteration counts.
 * - Integration of different arc types (standard, jump, fixed) during label extensions.
 * - Dual management and RCC separation for handling advanced routing optimization constraints.
 *
 * The `BucketGraph` class is designed to solve complex routing problems, leveraging multiple stages, parallelism,
 * and adaptive heuristics to optimize paths while respecting resource constraints.
 */

#pragma once

#include "BucketJump.h"
#include "Definitions.h"

#include "../cuts/SRC.h"
#include <cstring>

/**
 * Solves the bucket graph optimization problem using a multi-stage bi-labeling algorithm.
 *
 * The function iteratively adjusts the terminal time and handles different stages of the
 * bi-labeling algorithm to find the optimal paths. The stages are adaptive and transition
 * based on the inner objective values and iteration counts.
 *
 * @return A vector of pointers to Label objects representing the optimal paths.
 */
inline std::vector<Label *> BucketGraph::solve() {
    status = Status::NotOptimal;
    //////////////////////////////////////////////////////////////////////
    // ADAPTIVE TERMINAL TIME
    //////////////////////////////////////////////////////////////////////
    for (auto split : q_star) {
        if (((static_cast<double>(n_bw_labels) - static_cast<double>(n_fw_labels)) / static_cast<double>(n_fw_labels)) >
            0.05) {
            split += 0.05 * R_max[TIME_INDEX];
        } else if (((static_cast<double>(n_fw_labels) - static_cast<double>(n_bw_labels)) /
                    static_cast<double>(n_bw_labels)) > 0.05) {
            split -= 0.05 * R_max[TIME_INDEX];
        }
    }
    // std::vector<double>  q_star = split;
    std::vector<Label *> paths;
    double               inner_obj;

    //////////////////////////////////////////////////////////////////////
    // ADAPTIVE STAGE HANDLING
    //////////////////////////////////////////////////////////////////////
    if (s1) {
        stage     = 1;
        paths     = bi_labeling_algorithm<Stage::One>(q_star);
        inner_obj = paths[0]->cost;
        if (inner_obj >= -1 || iter >= 10) {
            s1 = false;
            s2 = true;
        }
    } else if (s2) {
        s2        = true;
        stage     = 2;
        paths     = bi_labeling_algorithm<Stage::Two>(q_star);
        inner_obj = paths[0]->cost;
        if (inner_obj >= -100 || iter > 800) {
            s2 = false;
            s3 = true;
        }
    } else if (s3) {
        stage     = 3;
        paths     = bi_labeling_algorithm<Stage::Three>(q_star);
        inner_obj = paths[0]->cost;

        if (inner_obj >= -1e-2) {
            s4         = true;
            s3         = false;
            transition = true;
        }
    } else if (s4) {
        stage = 4;

#ifdef FIX_BUCKETS
        if (transition) {
            print_cut("Transitioning to stage 4\n");
            fixed        = true;
            paths        = bi_labeling_algorithm<Stage::Four>(q_star);
            transition   = false;
            fixed        = false;
            min_red_cost = paths[0]->cost;
            iter++;
            return paths;
        }
#endif
        paths     = bi_labeling_algorithm<Stage::Four>(q_star);
        inner_obj = paths[0]->cost;
        if (inner_obj >= -1e-1) {
            // print_cut("Going into separation mode..\n");

            ss = true;
#if !defined(SRC) && !defined(SRC3)
            status = Status::Optimal;
            return paths;
#endif
            status = Status::Separation;
        }
    }
    iter++;

    return paths;
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

                    std::memset(Bvisited.data(), 0, Bvisited.size() * sizeof(uint64_t));
                    domin_smaller =
                        DominatedInCompWiseSmallerBuckets<D, S>(label, bucket, c_bar, Bvisited, ordered_sccs);

                    if (!domin_smaller) {

                        // Lambda to process new labels
                        auto process_new_label = [&](Label *new_label) {
                            if constexpr (F == Full::Partial) {
                                if constexpr (D == Direction::Forward) {
                                    if (label->resources[TIME_INDEX] > q_point[TIME_INDEX]) return;
                                } else {
                                    if (label->resources[TIME_INDEX] <= q_point[TIME_INDEX]) return;
                                }
                            }
                            stat_n_labels++;

                            int &to_bucket               = new_label->vertex;
                            dominated                    = false;
                            const auto &to_bucket_labels = buckets[to_bucket].get_labels();

                            if constexpr (S == Stage::One) {
                                for (auto *existing_label : to_bucket_labels) {
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
                                        stat_n_dom++;
                                        dominated = true;
                                        break;
                                    }
                                }
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
                        };

                        // Process normal arcs
                        const auto &arcs = jobs[label->job_id].get_arcs<D>(scc_index);
                        for (const auto &arc : arcs) {
                            Label *new_label = Extend<D, S, ArcType::Job, Mutability::Mut>(label, arc);
                            if (!new_label) {
#ifdef UNREACHABLE_DOMINANCE
                                set_job_unreachable(label->unreachable_bitmap, arc.to);
#endif
                                continue;
                            }
                            process_new_label(new_label);
                        }

#ifdef FIX_BUCKETS
                        if constexpr (S == Stage::Four) {
                            // Process jump arcs
                            const auto &jump_arcs = buckets[bucket].template get_jump_arcs<D>();
                            for (const auto &jump_arc : jump_arcs) {
                                Label *new_label = Extend<D, S, ArcType::Jump, Mutability::Const>(label, jump_arc);
                                if (!new_label) { continue; }
                                process_new_label(new_label);
                            }
                        }
#endif
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

    std::vector<double> forward_cbar(fw_buckets.size());
    std::vector<double> backward_cbar(bw_buckets.size());

    // std::fill(forward_cbar.begin(), forward_cbar.end(), std::numeric_limits<double>::infinity());
    // std::fill(backward_cbar.begin(), backward_cbar.end(), std::numeric_limits<double>::infinity());

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
                const auto &to_job = arc.to;
                if constexpr (S == Stage::Three) {
                    if (fixed_arcs[L->job_id][to_job] == 1) { continue; }
                }
                // Extend the current label based on the current stage
                auto L_prime = Extend<Direction::Forward, S, ArcType::Job, Mutability::Const>(L, arc);
                if (!L_prime || L_prime->resources[TIME_INDEX] <= q_star[TIME_INDEX]) continue;
                auto b_prime = L_prime->vertex;

#ifdef FIX_BUCKETS
                if constexpr (S == Stage::Four) {
                    if (fw_fixed_buckets[bucket][b_prime] == 1) { continue; }
                }
#endif
                std::memset(Bvisited.data(), 0, Bvisited.size() * sizeof(uint64_t));
                ConcatenateLabel<S>(L, b_prime, best_label, Bvisited, q_star);
            }
        }
    }

    pdqsort(merged_labels.begin(), merged_labels.end(),
            [](const Label *a, const Label *b) { return a->cost < b->cost; });

#ifdef RIH
    const int LABELS_MAX = 2;
    if constexpr (S >= Stage::Two) {
        rih_thread = std::thread(&BucketGraph::async_rih_processing, this, merged_labels, LABELS_MAX);
        rih_thread.detach(); // Detach the thread to allow it to run in the background
    }

#endif

    return merged_labels;
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
inline Label *
BucketGraph::Extend(const std::conditional_t<M == Mutability::Mut, Label *, const Label *>          L_prime,
                    const std::conditional_t<A == ArcType::Bucket, BucketArc,
                                             std::conditional_t<A == ArcType::Jump, JumpArc, Arc>> &gamma) noexcept {
    auto &buckets       = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &label_pool    = assign_buckets<D>(label_pool_fw, label_pool_bw);
    auto &fixed_buckets = assign_buckets<D>(fw_fixed_buckets, bw_fixed_buckets);
    // Precompute some values to avoid recalculating inside the loop
    const int    initial_job_id    = L_prime->job_id;
    auto         initial_resources = L_prime->resources;
    const double initial_cost      = L_prime->cost;

    int job_id = -1;
    if constexpr (A == ArcType::Bucket) {
        job_id = buckets[gamma.to_bucket].job_id;
    } else if constexpr (A == ArcType::Job) {
        job_id = gamma.to;
    } else if constexpr (A == ArcType::Jump) {
        job_id = buckets[gamma.jump_bucket].job_id;
        // iterate over initial resources
        for (size_t i = 0; i < initial_resources.size(); ++i) {
            if constexpr (D == Direction::Forward) {
                initial_resources[i] =
                    std::max(initial_resources[i], static_cast<double>(buckets[gamma.jump_bucket].lb[i]));
            } else {
                initial_resources[i] =
                    std::min(initial_resources[i], static_cast<double>(buckets[gamma.jump_bucket].ub[i]));
            }
        }
    }

    // Early exit if the arc is fixed
    if constexpr (S == Stage::Three) {
        if constexpr (Direction::Forward == D) {
            if (fixed_arcs[initial_job_id][job_id] == 1) { return nullptr; }
        } else {
            if (fixed_arcs[job_id][initial_job_id] == 1) { return nullptr; }
        }
    }

    // Early exit for enumeration
    if constexpr (S == Stage::Enumerate) {
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
    if constexpr (S == Stage::Four && A != ArcType::Jump) {
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
        // new_label->parent = L_prime;
    }
    // if constexpr (M == Mutability::Mut) { L_prime->children.push_back(new_label); }

#if defined(SRC3) || defined(SRC)
    new_label->SRCmap = L_prime->SRCmap;
#endif

    if constexpr (S != Stage::Enumerate) {
        size_t limit = new_label->visited_bitmap.size();
        for (size_t i = 0; i < limit; ++i) {
            uint64_t current_visited = new_label->visited_bitmap[i];

            // Skip processing if no bits are set
            if (!current_visited) continue;

            uint64_t neighborhood_mask = neighborhoods_bitmap[job_id][i];
            uint64_t bits_to_clear     = current_visited & ~neighborhood_mask;

            // Use bit manipulation instead of condition
            if (i == job_id / 64) { bits_to_clear &= ~(1ULL << (job_id % 64)); }

            new_label->visited_bitmap[i] &= ~bits_to_clear;
        }
    }
#if defined(SRC3) || defined(SRC)

    if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
        auto          &cutter   = cut_storage;
        auto          &SRCDuals = cutter->SRCDuals;
        const uint64_t bit_mask = 1ULL << bit_position; // Precompute bit shift

        for (std::size_t idx = 0; idx < cutter->size(); ++idx) {
            auto it = cutter->begin();
            std::advance(it, idx);
            const auto &cut = *it;

            const auto &baseSet      = cut.baseSet;
            const auto &baseSetorder = cut.baseSetOrder;
            const auto &neighbors    = cut.neighbors;
            const auto &multipliers  = cut.multipliers;

#if defined(SRC3)
            bool bitIsSet3 = baseSet[segment] & bit_mask;
            if (bitIsSet3) {
                new_label->SRCmap[idx]++;
                if (new_label->SRCmap[idx] % 2 == 0) { new_label->cost -= SRCDuals[idx]; }
            }
#endif

#if defined(SRC)
            bool bitIsSet  = neighbors[segment] & bit_mask;
            bool bitIsSet2 = baseSet[segment] & bit_mask;

            double &src_map_value = new_label->SRCmap[idx]; // Use reference to avoid multiple accesses
            if (bitIsSet) {
                src_map_value = L_prime->SRCmap[idx];
            } else {
                src_map_value = 0.0;
            }

            if (bitIsSet2) {
                src_map_value += multipliers[baseSetorder[job_id]];
                if (src_map_value >= 1) {
                    src_map_value -= 1;
                    new_label->cost -= SRCDuals[idx];
                }
            }
#endif
        }
    }
#endif

    return new_label;
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

    const auto &new_resources   = new_label->resources;
    const auto &label_resources = label->resources;

    double sumSRC = 0;
#ifdef SRC
    if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
        const auto &SRCDuals = cut_storage->SRCDuals;
        if (!SRCDuals.empty()) {
            for (size_t i = 0; i < SRCDuals.size(); ++i) {
                if (label->SRCmap[i] > new_label->SRCmap[i]) { sumSRC += SRCDuals[i]; }
            }
        }
        if (label->cost - sumSRC > new_label->cost) { return false; }
    } else
#endif
    {
        if (label->cost > new_label->cost) { return false; }
    }

    // Check resource conditions (direction-dependent)
    for (size_t i = 0; i < new_resources.size(); ++i) {
        if constexpr (D == Direction::Forward) {
            if (label_resources[i] > new_resources[i]) { return false; }
        } else if constexpr (D == Direction::Backward) {
            if (label_resources[i] < new_resources[i]) { return false; }
        }
    }

#ifndef UNREACHABLE_DOMINANCE
    if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
        for (size_t i = 0; i < label->visited_bitmap.size(); ++i) {
            if ((label->visited_bitmap[i] & ~new_label->visited_bitmap[i]) != 0) { return false; }
        }
    }
#else
    if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
        for (size_t i = 0; i < label->visited_bitmap.size(); ++i) {
            auto combined_label_bitmap = label->visited_bitmap[i] | label->unreachable_bitmap[i];
            if ((combined_label_bitmap & ~new_label->visited_bitmap[i]) != 0) { return false; }
        }
    }
#endif

#ifdef SRC3
    if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
        const auto &SRCDuals = cut_storage->SRCDuals;
        if (!SRCDuals.empty()) {
            sumSRC = 0;
            for (size_t i = 0; i < SRCDuals.size(); ++i) {
                if ((label->SRCmap[i] % 2) > (new_label->SRCmap[i] % 2)) { sumSRC += SRCDuals[i]; }
            }
            if (label->cost + sumSRC > new_label->cost) { return false; }
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
