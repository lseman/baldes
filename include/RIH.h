

#pragma once
#include <algorithm>
#include <condition_variable>
#include <exec/task.hpp>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "Cut.h"
#include "Label.h"
#include "Path.h"
#include "RNG.h"
#include "Reader.h"
#include "TaskQueue.h"
#include "VRPNode.h"

class IteratedLocalSearch {
   public:
    InstanceData instance;
    std::vector<VRPNode> nodes;
    CutStorage cut_storage;

    // default default initialization passing InstanceData &instance
    IteratedLocalSearch(const InstanceData &instance)
        : instance(instance),
          pool(5),
          sched(pool.get_scheduler()),
          task_queue(5, sched, *this) {}

    ~IteratedLocalSearch() {}

    static std::vector<int> order_crossover(const std::vector<int> &parent1,
                                            const std::vector<int> &parent2,
                                            int i, int j) {
        int n = parent1.size();
        if (i < 0 || j >= n || i >= j) {
            throw std::invalid_argument("Invalid crossover points.");
        }

        // Initialize offspring with placeholders (-1 indicates empty)
        std::vector<int> offspring(n, -1);

        // Copy segment from parent1
        std::copy(parent1.begin() + i, parent1.begin() + j + 1,
                  offspring.begin() + i);

        // Fill remaining positions with elements from parent2
        auto it = parent2.begin();
        for (int k = 0; k < n; ++k) {
            if (offspring[k] == -1) {  // Empty slot in offspring
                // Skip elements already present in the offspring
                while (std::find(offspring.begin(), offspring.end(), *it) !=
                       offspring.end()) {
                    ++it;
                }
                offspring[k] = *it;
                ++it;
            }
        }

        return offspring;
    }

    std::pair<std::vector<uint16_t>, std::vector<uint16_t>> srex_crossover(
        const std::vector<uint16_t> &parent1,
        const std::vector<uint16_t> &parent2, int i, int j) {
        int n = parent1.size();
        if (i < 0 || j >= n || i >= j) {
            return {parent1, parent2};
        }

        // Copy subroutes
        std::vector<uint16_t> segment1(parent1.begin() + i,
                                       parent1.begin() + j + 1);
        std::vector<uint16_t> segment2(parent2.begin() + i,
                                       parent2.begin() + j + 1);

        // Create offspring by swapping the segments
        std::vector<uint16_t> offspring1, offspring2;

        // Insert segment2 into parent1 and reconstruct
        for (int node : parent1) {
            if (std::find(segment2.begin(), segment2.end(), node) ==
                segment2.end()) {
                offspring1.push_back(
                    node);  // Add nodes not in the swapped segment
            }
        }
        offspring1.insert(offspring1.begin() + i, segment2.begin(),
                          segment2.end());

        // Insert segment1 into parent2 and reconstruct
        for (int node : parent2) {
            if (std::find(segment1.begin(), segment1.end(), node) ==
                segment1.end()) {
                offspring2.push_back(
                    node);  // Add nodes not in the swapped segment
            }
        }
        offspring2.insert(offspring2.begin() + i, segment1.begin(),
                          segment1.end());

        return {std::move(offspring1), std::move(offspring2)};
    }

    /**
     * @brief Performs the 2-opt optimization on a given route.
     *
     * This function takes a route and two indices, i and j, and returns a new
     * route where the segment between i and j is reversed. The function ensures
     * that the indices are within bounds and that the depots (first and last
     * elements) are not involved in the reversal.
     *
     */
    static std::vector<uint16_t> two_opt(const std::vector<uint16_t> &route,
                                         int i, int j) {
        // Validate indices and ensure they do not include depots
        if (i <= 0 || j >= static_cast<int>(route.size()) - 1 || i >= j) {
            return route;
        }

        // Perform the 2-opt operation by reversing the segment between i and j
        std::vector<uint16_t> new_route(route);
        std::reverse(new_route.begin() + i, new_route.begin() + j + 1);
        return new_route;
    }

    /**
     * @brief Relocates a customer from one route to another.
     *
     * This function relocates the customer at position `i` in `route1` to
     * position `j` in `route2`. If the positions `i` or `j` involve depots
     * (i.e., the first or last positions in the routes), no changes are made
     * and the original routes are returned.
     *
     */
    std::pair<std::vector<uint16_t>, std::vector<uint16_t>> relocate_star(
        const std::vector<uint16_t> &route1,
        const std::vector<uint16_t> &route2, int i, int j) {
        // Check if indices are valid and do not involve depots
        if (i <= 0 || i >= static_cast<int>(route1.size()) - 1 || j < 0 ||
            j > static_cast<int>(route2.size())) {
            return {route1, route2};
        }

        // Perform relocation
        std::vector<uint16_t> new_route1(route1.begin(), route1.end());
        std::vector<uint16_t> new_route2(route2.begin(), route2.end());

        // Extract and move customer
        int customer = std::move(new_route1[i]);
        new_route1.erase(new_route1.begin() +
                         i);  // Remove customer from route1
        new_route2.insert(new_route2.begin() + j,
                          customer);  // Insert into route2

        return {std::move(new_route1), std::move(new_route2)};
    }

    /**
     * @brief Swaps segments of varying lengths between two routes.
     *
     * This function takes two routes and swaps segments of up to 3 elements
     * between them, starting at specified indices. If the start index is at the
     * beginning or end of a route, no changes are made to avoid swapping depot
     * nodes.
     *
     */
    std::pair<std::vector<uint16_t>, std::vector<uint16_t>> enhanced_swap(
        const std::vector<uint16_t> &route1,
        const std::vector<uint16_t> &route2, int i, int j) {
        // Check if indices are valid and do not involve depots
        if (i <= 0 || i >= static_cast<int>(route1.size()) - 1 || j <= 0 ||
            j >= static_cast<int>(route2.size()) - 1) {
            return {route1, route2};
        }

        // Determine maximum segment length to swap (up to 3 elements)
        int max_segment_length = 3;
        int segment_length =
            std::min({max_segment_length, static_cast<int>(route1.size() - i),
                      static_cast<int>(route2.size() - j)});

        std::vector<uint16_t> new_route1(route1);
        std::vector<uint16_t> new_route2(route2);

        // Swap the segments between the two routes
        std::swap_ranges(new_route1.begin() + i,
                         new_route1.begin() + i + segment_length,
                         new_route2.begin() + j);

        return {std::move(new_route1), std::move(new_route2)};
    }

    /**
     * @brief Performs a crossover operation between two routes at specified
     * positions.
     *
     * This function takes two routes and two crossover points, and swaps the
     * tails of the routes after the specified positions. The depots (first and
     * last elements) are not allowed to be crossover points.
     *
     */
    std::pair<std::vector<uint16_t>, std::vector<uint16_t>> cross(
        const std::vector<uint16_t> &route1,
        const std::vector<uint16_t> &route2, int k, int l) {
        // Ensure valid crossover points that do not involve depots
        if (k <= 0 || k >= static_cast<int>(route1.size()) - 1 || l <= 0 ||
            l >= static_cast<int>(route2.size()) - 1) {
            return {route1, route2};
        }

        // Create new routes by swapping tails
        std::vector<uint16_t> new_route1(route1.begin(), route1.begin() + k);
        new_route1.insert(new_route1.end(), route2.begin() + l, route2.end());

        std::vector<uint16_t> new_route2(route2.begin(), route2.begin() + l);
        new_route2.insert(new_route2.end(), route1.begin() + k, route1.end());

        return {std::move(new_route1), std::move(new_route2)};
    }

    /**
     * @brief Inserts a customer from one route into another route at specified
     * positions.
     *
     * This function takes two routes and inserts a customer from the first
     * route at position `k` into the second route at position `l`. It ensures
     * that depots (first and last positions) are not involved in the insertion
     * process.
     *
     */
    std::pair<std::vector<uint16_t>, std::vector<uint16_t>> insertion(
        const std::vector<uint16_t> &route1,
        const std::vector<uint16_t> &route2, int k, int l) {
        // Ensure valid positions that do not involve depots
        if (k <= 0 || k >= static_cast<int>(route1.size()) - 1 || l < 0 ||
            l > static_cast<int>(route2.size())) {
            return {route1, route2};
        }

        // Copy routes and perform insertion
        std::vector<uint16_t> new_route1(route1);
        std::vector<uint16_t> new_route2(route2);

        // Extract the customer from route1
        int customer =
            std::move(new_route1[k]);  // Move to avoid unnecessary copying
        new_route1.erase(new_route1.begin() + k);

        // Insert the customer into route2
        new_route2.insert(new_route2.begin() + l, std::move(customer));

        return {std::move(new_route1), std::move(new_route2)};
    }

    /**
     * @brief Swaps customers between two routes at specified positions.
     *
     * This function takes two routes and swaps the customers at the specified
     * positions k and l. It ensures that depot positions (first and last
     * elements) are not swapped.
     *
     */
    std::pair<std::vector<uint16_t>, std::vector<uint16_t>> swap(
        const std::vector<uint16_t> &route1,
        const std::vector<uint16_t> &route2, int k, int l) {
        // Ensure valid swap positions that do not involve depots
        if (k <= 0 || k >= static_cast<int>(route1.size()) - 1 || l <= 0 ||
            l >= static_cast<int>(route2.size()) - 1) {
            return {route1, route2};
        }

        // Swap customers between route1 and route2
        std::vector<uint16_t> new_route1(route1);
        std::vector<uint16_t> new_route2(route2);

        std::swap(new_route1[k], new_route2[l]);

        return {std::move(new_route1), std::move(new_route2)};
    }

    std::pair<std::vector<uint16_t>, std::vector<uint16_t>> nm_exchange(
        const std::vector<uint16_t> &route1,
        const std::vector<uint16_t> &route2, int i, int j) {
        return nm_exchange_fun(route1, route2, i, j, 2, 1);
    }

    // (N, M)-Exchange Operator
    std::pair<std::vector<uint16_t>, std::vector<uint16_t>> nm_exchange_fun(
        const std::vector<uint16_t> &route1,
        const std::vector<uint16_t> &route2, int i, int j, int n = 2,
        int m = 1) {
        if (i == 0 || j == 0 || i + n > route1.size() ||
            j + m > route2.size()) {
            return {route1, route2};
        }

        std::vector<uint16_t> new_route1 = route1;
        std::vector<uint16_t> new_route2 = route2;

        // Extract chains
        std::vector<uint16_t> chain1(new_route1.begin() + i,
                                     new_route1.begin() + i + n);
        std::vector<uint16_t> chain2(new_route2.begin() + j,
                                     new_route2.begin() + j + m);

        // Swap the chains
        new_route1.erase(new_route1.begin() + i, new_route1.begin() + i + n);
        new_route2.erase(new_route2.begin() + j, new_route2.begin() + j + m);

        new_route1.insert(new_route1.begin() + i, chain2.begin(), chain2.end());
        new_route2.insert(new_route2.begin() + j, chain1.begin(), chain1.end());

        return {new_route1, new_route2};
    }

    // MoveTwoClientsReversed Operator
    std::pair<std::vector<uint16_t>, std::vector<uint16_t>>
    move_two_clients_reversed(const std::vector<uint16_t> &route1,
                              const std::vector<uint16_t> &route2, int i,
                              int j) {
        if (i == 0 || i + 1 >= route1.size() || j == 0 || j >= route2.size()) {
            return {route1, route2};
        }

        std::vector<uint16_t> new_route1 = route1;
        std::vector<uint16_t> new_route2 = route2;

        // Extract the chain and reverse it
        std::vector<uint16_t> chain = {new_route1[i], new_route1[i + 1]};
        std::reverse(chain.begin(), chain.end());

        new_route1.erase(new_route1.begin() + i, new_route1.begin() + i + 2);
        new_route2.insert(new_route2.begin() + j, chain.begin(), chain.end());

        return {new_route1, new_route2};
    }

    /**
     * @brief Swaps sequences of customers between two routes and inserts them
     * at optimal positions.
     *
     * This operator swaps a chain of 2 or 3 consecutive customers between two
     * routes, and inserts each chain into the best possible position in the
     * opposite route, ensuring depots are not involved.
     */

    std::pair<std::vector<uint16_t>, std::vector<uint16_t>> extended_swap_star(
        const std::vector<uint16_t> &route1,
        const std::vector<uint16_t> &route2, int i, int j) {
        // Call the actual extended_swap_star with a default chain length of 2
        return extended_swap_star_fun(route1, route2, i, j, 2);
    }

    std::pair<std::vector<uint16_t>, std::vector<uint16_t>>
    extended_swap_star_fun(const std::vector<uint16_t> &route1,
                           const std::vector<uint16_t> &route2, int i, int j,
                           int chain_length = 2) {
        // Ensure indices are within bounds and do not involve depot nodes
        if (i <= 0 || i + chain_length >= static_cast<int>(route1.size()) ||
            j <= 0 || j + chain_length >= static_cast<int>(route2.size())) {
            return {route1, route2};
        }

        // Create mutable copies of the routes
        std::vector<uint16_t> new_route1(route1.begin(), route1.end());
        std::vector<uint16_t> new_route2(route2.begin(), route2.end());

        // Extract and erase chains in-place
        auto chain1_start = new_route1.begin() + i;
        auto chain1_end = chain1_start + chain_length;
        std::vector<uint16_t> chain1(std::make_move_iterator(chain1_start),
                                     std::make_move_iterator(chain1_end));
        new_route1.erase(chain1_start, chain1_end);

        auto chain2_start = new_route2.begin() + j;
        auto chain2_end = chain2_start + chain_length;
        std::vector<uint16_t> chain2(std::make_move_iterator(chain2_start),
                                     std::make_move_iterator(chain2_end));
        new_route2.erase(chain2_start, chain2_end);

        // Find the best insertion positions
        int best_pos_route1 = find_best_insertion_position(new_route1, chain2);
        int best_pos_route2 = find_best_insertion_position(new_route2, chain1);

        // Insert the chains at the best positions
        new_route1.insert(new_route1.begin() + best_pos_route1,
                          std::make_move_iterator(chain2.begin()),
                          std::make_move_iterator(chain2.end()));
        new_route2.insert(new_route2.begin() + best_pos_route2,
                          std::make_move_iterator(chain1.begin()),
                          std::make_move_iterator(chain1.end()));

        return {std::move(new_route1), std::move(new_route2)};
    }

    /**
     * @brief Finds the best position to insert a chain of customers into a
     * route.
     *
     * This helper function evaluates the cost of inserting a chain into each
     * possible position in the route and returns the index of the best
     * insertion point.
     */
    int find_best_insertion_position(const std::vector<uint16_t> &route,
                                     const std::vector<uint16_t> &chain) {
        int best_pos = 1;
        double best_cost = std::numeric_limits<double>::max();

        // Precompute the cost of the original route
        const double original_cost = this->compute_cost(route).second;

        // Iterate over valid insertion positions
        for (int pos = 1; pos < static_cast<int>(route.size()); ++pos) {
            // Incrementally compute the new cost for inserting the chain at
            // position `pos`
            const double incremental_cost =
                compute_insertion_cost(route, chain, pos, original_cost);

            // Track the best position based on the cost
            if (incremental_cost < best_cost) {
                best_cost = incremental_cost;
                best_pos = pos;
            }
        }
        return best_pos;
    }

    double compute_insertion_cost(const std::vector<uint16_t> &route,
                                  const std::vector<uint16_t> &chain, int pos,
                                  double original_cost) {
        // Simulate the insertion and calculate the cost incrementally
        std::vector<uint16_t> new_route(route.size() + chain.size());
        std::copy(route.begin(), route.begin() + pos, new_route.begin());
        std::copy(chain.begin(), chain.end(), new_route.begin() + pos);
        std::copy(route.begin() + pos, route.end(),
                  new_route.begin() + pos + chain.size());

        return this->compute_cost(new_route).second;
    }

    /**
     * @brief Relocates a sequence of customers from one route to another.
     *
     * This operator moves a chain of 2 or 3 consecutive customers from one
     * route to another. The depots (first and last elements) are not involved.
     */
    std::pair<std::vector<uint16_t>, std::vector<uint16_t>>
    extended_relocate_star(const std::vector<uint16_t> &route1,
                           const std::vector<uint16_t> &route2, int i, int j,
                           int chain_length = 2) {
        // Ensure we are not relocating depot nodes or exceeding bounds
        if (i == 0 || i + chain_length > route1.size() - 1 || j == 0 ||
            j == route2.size() - 1) {
            return {route1, route2};  // No changes if depots or out-of-bound
                                      // indices are involved
        }

        // Relocate a chain of customers from route1 to route2
        std::vector<uint16_t> new_route1 = route1;
        std::vector<uint16_t> new_route2 = route2;

        // Extract the chain of customers to relocate
        std::vector<uint16_t> chain(new_route1.begin() + i,
                                    new_route1.begin() + i + chain_length);

        // Remove the chain from route1
        new_route1.erase(new_route1.begin() + i,
                         new_route1.begin() + i + chain_length);

        // Insert the chain into route2 at position j
        new_route2.insert(new_route2.begin() + j, chain.begin(), chain.end());

        return {new_route1, new_route2};
    }

    ///////////////////////////////////////
    // End operators
    ///////////////////////////////////////
    // Adaptive operator selection
    using OperatorFunc =
        std::pair<std::vector<uint16_t>, std::vector<uint16_t>> (
            IteratedLocalSearch::*)(const std::vector<uint16_t> &,
                                    const std::vector<uint16_t> &, int, int);
    std::vector<OperatorFunc> operators = {
        &IteratedLocalSearch::cross,
        &IteratedLocalSearch::insertion,
        &IteratedLocalSearch::swap,
        &IteratedLocalSearch::relocate_star,
        &IteratedLocalSearch::enhanced_swap,
        &IteratedLocalSearch::move_two_clients_reversed,
        &IteratedLocalSearch::extended_swap_star};

    std::vector<double> operator_weights = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<double> operator_improvements = {0.0, 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0};
    std::vector<int> operator_success_count = {0, 0, 0, 0, 0, 0, 0};

    // Utility function to select an operator based on weights using
    // std::discrete_distribution
    int select_operator(Xoroshiro128Plus &rng) {
        std::discrete_distribution<int> dist(operator_weights.begin(),
                                             operator_weights.end());
        return dist(rng);
    }

    // Apply the selected operator
    std::pair<std::vector<uint16_t>, std::vector<uint16_t>> apply_operator(
        int op_index, const std::vector<uint16_t> &route1,
        const std::vector<uint16_t> &route2, int k, int l) {
        return (this->*operators[op_index])(route1, route2, k, l);
    }

    void update_operator_weights(const double decay_factor = 0.9,
                                 const double reward_factor = 0.1,
                                 const double min_weight = 0.01) noexcept {
        // Use SIMD-friendly std::reduce instead of std::accumulate
        const double total_improvement =
            std::reduce(std::execution::unseq, operator_improvements.begin(),
                        operator_improvements.end());

        const int total_successes =
            std::reduce(std::execution::unseq, operator_success_count.begin(),
                        operator_success_count.end());

        if (total_improvement > 0 || total_successes > 0) {
            // Precompute inverse totals to avoid divisions
            const double inv_total_improvement =
                total_improvement > 0 ? 1.0 / total_improvement : 0.0;
            const double inv_total_successes =
                total_successes > 0 ? 1.0 / total_successes : 0.0;

            // Single pass through weights with minimal branching
            for (size_t i = 0; i < operator_weights.size(); ++i) {
                const double normalized_improvement =
                    operator_improvements[i] * inv_total_improvement;
                const double success_rate =
                    operator_success_count[i] * inv_total_successes;

                operator_weights[i] = std::max(
                    min_weight, decay_factor * operator_weights[i] +
                                    reward_factor * (normalized_improvement +
                                                     success_rate));
            }

            // Two-pass normalization for better numerical stability
            const double total_weight =
                std::reduce(operator_weights.begin(), operator_weights.end());

            if (total_weight > 0) {
                const double inv_total_weight = 1.0 / total_weight;
                for (auto &weight : operator_weights) {
                    weight *= inv_total_weight;
                }
            }
        }

        // Fast memory clearing using memset
        std::memset(operator_improvements.data(), 0,
                    operator_improvements.size() * sizeof(double));
        std::memset(operator_success_count.data(), 0,
                    operator_success_count.size() * sizeof(int));
    }

    exec::static_thread_pool pool = exec::static_thread_pool(1);
    exec::static_thread_pool::scheduler sched = pool.get_scheduler();

    std::vector<Label *> perturbation(const std::vector<Label *> &paths) {
        // Preallocate vectors to avoid reallocations
        std::vector<Label *> best;
        best.reserve(paths.size());
        std::copy(paths.begin(), paths.end(), std::back_inserter(best));

        std::vector<Label *> best_new;
        Xoroshiro128Plus rng;  // Instantiate the custom RNG with default seed
        bool is_stuck = false;

        std::vector<double> best_costs(paths.size(),
                                       std::numeric_limits<double>::max());

        // Store tasks for parallel processing
        std::vector<std::tuple<const Label *, const Label *, size_t, size_t>>
            tasks;

        for (const Label *label_i : best) {
            for (const Label *label_j : best) {
                if (label_i == label_j) continue;
                const auto &route_i = label_i->nodes_covered;
                const auto &route_j = label_j->nodes_covered;

                if (route_i.size() < 3 || route_j.size() < 3) continue;

                for (size_t k = 1; k < route_i.size() - 1; ++k) {
                    for (size_t l = 1; l < route_j.size() - 1; ++l) {
                        tasks.emplace_back(label_i, label_j, k, l);
                    }
                }
            }
        }

        // Parallelize the task processing
        std::mutex best_new_mutex;  // Mutex to protect access to best_new and
                                    // operator_improvements
        std::mutex operator_mutex;  // Mutex to protect operator improvements

        // Define chunk size to balance load
        const int chunk_size =
            1;  // Adjust chunk size based on performance experiments
        auto bulk_sender = stdexec::bulk(
            stdexec::just(), (tasks.size() + chunk_size - 1) / chunk_size,
            [&best_new, &best_new_mutex, &operator_mutex, &tasks, chunk_size,
             &rng, this](std::size_t chunk_idx) {
                // Thread-local storage for intermediate results
                std::vector<Label *> local_best_new;
                local_best_new.reserve(chunk_size);

                // Process chunk
                size_t start_idx = chunk_idx * chunk_size;
                size_t end_idx = std::min(start_idx + chunk_size, tasks.size());

                for (size_t task_idx = start_idx; task_idx < end_idx;
                     ++task_idx) {
                    const auto &[label_i, label_j, k, l] = tasks[task_idx];

                    // Cache route references
                    const auto &route_i = label_i->getRoute();
                    const auto &route_j = label_j->getRoute();

                    if (route_i.size() < 3 || route_j.size() < 3) continue;

                    int op_idx = select_operator(rng);

                    // Move semantics for route creation
                    auto [new_route_i, new_route_j] =
                        apply_operator(op_idx, route_i, route_j, k, l);

                    if (new_route_i.empty() || new_route_j.empty()) continue;

                    // Compute costs once and reuse
                    auto cost_i = compute_cost(new_route_i);
                    auto cost_j = compute_cost(new_route_j);

                    // Create paths using move semantics
                    Path new_path_i{std::move(new_route_i), cost_i.first};
                    Path new_path_j{std::move(new_route_j), cost_j.first};

                    auto process_improvement = [&](const Path &path,
                                                   const auto &cost,
                                                   const Label *label,
                                                   int op_idx) {
                        double new_cost = cost.second;
                        double current_cost = label->cost;

                        if (new_cost < current_cost - 1e-3) {
                            auto *best_new_label = new Label();
                            best_new_label->addRoute(path.route);
                            best_new_label->real_cost = cost.first;
                            best_new_label->cost = new_cost;
                            local_best_new.push_back(best_new_label);

                            // Batch operator statistics updates
                            return std::make_pair(current_cost - new_cost, 1);
                        }
                        return std::make_pair(0.0, 0);
                    };

                    // Process improvements
                    if (is_feasible(new_path_i)) {
                        auto [improvement, success] = process_improvement(
                            new_path_i, cost_i, label_i, op_idx);
                        if (success) {
                            std::lock_guard<std::mutex> lock(operator_mutex);
                            operator_improvements[op_idx] += improvement;
                            operator_success_count[op_idx] += 1;
                        }
                    }
                    if (is_feasible(new_path_j)) {
                        auto [improvement, success] = process_improvement(
                            new_path_j, cost_j, label_j, op_idx);
                        if (success) {
                            std::lock_guard<std::mutex> lock(operator_mutex);
                            operator_improvements[op_idx] += improvement;
                            operator_success_count[op_idx] += 1;
                        }
                    }
                }

                // Batch update best_new
                if (!local_best_new.empty()) {
                    std::lock_guard<std::mutex> lock(best_new_mutex);
                    best_new.insert(best_new.end(), local_best_new.begin(),
                                    local_best_new.end());
                }
            });

        // Submit work to the thread pool
        auto work = stdexec::starts_on(sched, bulk_sender);
        stdexec::sync_wait(std::move(work));

        // Update operator weights based on performance
        update_operator_weights();

        // Sort and limit to top solutions
        pdqsort(
            best_new.begin(), best_new.end(),
            [](const Label *a, const Label *b) { return a->cost < b->cost; });
        if (best_new.size() > N_ADD) {
            best_new.erase(best_new.begin() + N_ADD, best_new.end());
        }

        return best_new;
    }

    // Function to submit new tasks to the queue
    void submit_task(const std::vector<Label *> &paths,
                     const std::vector<VRPNode> &nodes) {
        this->nodes = nodes;
        for (const auto &path : paths) {
            task_queue.submit_task(path);
        }
    }

    // Function to retrieve processed labels
    std::vector<Label *> get_labels() {
        auto tasks = task_queue.get_processed_tasks();
        std::vector<Label *> result;
        result.insert(result.end(), tasks.begin(), tasks.end());
        return result;
    }

   private:
    // Feasibility check and cost computation functions
    bool is_feasible(const Path &route) const noexcept {
        // Early exit checks for route validity
        const auto &path = route.route;
        if (path.size() < 2 || path.front() != 0 || path.back() != N_SIZE - 1) {
            return false;
        }

        // Thread-local storage for visited nodes tracking
        constexpr size_t n_segments = (N_SIZE + 63) / 64;  // Ceiling division
        static thread_local std::array<uint64_t, n_segments> Bvisited;

        // Use memset for faster clearing of visited array
        std::memset(Bvisited.data(), 0, n_segments * sizeof(uint64_t));

        double time = 0.0;
        double capacity = 0.0;
        const double total_capacity = instance.q;
        const size_t route_size = path.size();

        // Prefetch next nodes data
        _mm_prefetch(&nodes[path[1]], _MM_HINT_T0);

        for (size_t i = 0; i < route_size - 1; ++i) {
            // Prefetch data for next iteration
            if (i + 2 < route_size) {
                _mm_prefetch(&nodes[path[i + 2]], _MM_HINT_T0);
            }

            const uint32_t source_id = path[i];
            const uint32_t target_id = path[i + 1];

            // Check visited status using bit operations
            const uint32_t segment = source_id >> 6;
            const uint64_t bit_mask = 1ULL << (source_id & 63);

            if (Bvisited[segment] & bit_mask) {
                return false;  // Node already visited
            }
            Bvisited[segment] |= bit_mask;

            // Cache node references
            const auto &source = nodes[source_id];
            const auto &target = nodes[target_id];

            // Calculate travel time
            const double travel_time =
                source.duration + instance.getcij(source_id, target_id);
            const double start_time =
                std::max(static_cast<double>(target.lb[0]), time + travel_time);

            // Check time window constraint
            if (start_time > target.ub[0]) {
                return false;
            }
            time = start_time;

            // Update and check capacity constraint
            capacity += source.demand;
            if (capacity > total_capacity) {
                return false;
            }
        }

        return true;
    }

    std::pair<double, double> compute_cost(const std::vector<uint16_t> &route) {
        double cost = 0;
        double red_cost = 0;

        auto &cutter = cut_storage;  // Access the cut storage manager
        // auto cut_size = cutter->size();
        // print theSize
        const auto active_cuts = cutter.getActiveCuts();
        // const auto activeSize = active_cuts.size();
#ifdef SRC
        ankerl::unordered_dense::map<size_t, double> SRCmap;
#endif

        for (size_t i = 0; i < route.size() - 1; ++i) {
            auto travel_cost = instance.getcij(route[i], route[i + 1]);
            auto node = route[i];
            cost += travel_cost;
            // If the reduced cost should be based on travel cost minus
            // node-specific costs, we subtract that here
            red_cost += travel_cost - nodes[route[i]].cost;

            //     const size_t segment =
            //         node >> 6;  // Determine the segment in the bitmap
            //     const size_t bit_position =
            //         node & 63;  // Determine the bit position in the segment

            //     const uint64_t bit_mask =
            //         1ULL << bit_position;  // Precompute bit shift for the
            //                                // node's position

            //     for (const auto &active_cut : active_cuts) {
            //         if (!active_cut.cut_ptr) {
            //             continue;
            //         }

            //         const Cut &cut = *active_cut.cut_ptr;
            //         const size_t idx = active_cut.index;
            //         const double dual_value = active_cut.dual_value;

            //         if (SRCmap.find(idx) == SRCmap.end()) {
            //             SRCmap[idx] = 0;
            //         }
            //         auto &src_map_value = SRCmap[idx];

            //         if (cut.neighbors[segment] & bit_mask) {
            //             if (cut.baseSet[segment] & bit_mask) {
            //                 const auto &multipliers = cut.p;
            //                 // print size of cut.baseSetOrde
            //                 src_map_value =
            //                 multipliers.num[cut.baseSetOrder[node]]; if
            //                 (src_map_value >= multipliers.den) {
            //                     src_map_value -= multipliers.den;
            //                     cost -= dual_value;
            //                 }
            //             }
            //         }
            //     }
        }
        return {cost, red_cost};
    }
    // Task queue and synchronization primitives
    TaskQueue<Label *, IteratedLocalSearch> task_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_condition;
    bool shutdown = false;
    const int task_threshold = N_ADD;  // The threshold for processing tasks

    std::vector<Label *> processed_labels;  // Store results from perturbation

    // Function to gracefully shut down the worker
    void stop_worker() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            shutdown = true;  // Signal the worker to shut down
        }
        queue_condition.notify_all();  // Wake up the worker to shut it down
    }
};
