

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
    CutStorage *cut_storage = nullptr;

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

    void update_operator_weights(double decay_factor = 0.9,
                                 double reward_factor = 0.1,
                                 double min_weight = 0.01) {
        // First, compute total improvement and total successes.
        double total_improvement = 0.0;
        int total_successes = 0;
        const size_t n = operator_weights.size();
        for (size_t i = 0; i < n; ++i) {
            total_improvement += operator_improvements[i];
            total_successes += operator_success_count[i];
        }

        // Only update weights if we have any recorded improvement or success.
        if (total_improvement > 0 || total_successes > 0) {
            const double inv_total_improvement =
                (total_improvement > 0) ? (1.0 / total_improvement) : 0.0;
            const double inv_total_successes =
                (total_successes > 0) ? (1.0 / total_successes) : 0.0;

            // Update each operator weight and accumulate the total weight.
            double total_weight = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double normalized_improvement =
                    operator_improvements[i] * inv_total_improvement;
                double success_rate =
                    operator_success_count[i] * inv_total_successes;
                double new_weight =
                    decay_factor * operator_weights[i] +
                    reward_factor * (normalized_improvement + success_rate);
                // Ensure a minimum weight.
                operator_weights[i] =
                    (new_weight < min_weight) ? min_weight : new_weight;
                total_weight += operator_weights[i];
            }

            // Normalize the weights so that they sum to 1.
            if (total_weight > 0.0) {
                for (size_t i = 0; i < n; ++i) {
                    operator_weights[i] /= total_weight;
                }
            }
        }

        // Reset improvements and success counts for the next iteration.
        for (size_t i = 0; i < n; ++i) {
            operator_improvements[i] = 0.0;
            operator_success_count[i] = 0;
        }
    }

    exec::static_thread_pool pool =
        exec::static_thread_pool(std::thread::hardware_concurrency() / 4);
    exec::static_thread_pool::scheduler sched = pool.get_scheduler();

    Xoroshiro128Plus rng;  // Custom RNG with default seed
    std::mutex operator_mutex;

    std::vector<Label *> perturbation(const std::vector<Label *> &paths) {
        // Copy the input paths as the baseline best solution.
        std::vector<Label *> best = paths;
        std::vector<Label *> best_new;

        bool is_stuck = false;
        // Initialize best costs for each path to a large value.
        std::vector<double> best_costs(paths.size(),
                                       std::numeric_limits<double>::max());

        // Reserve an approximate number of tasks if possible.
        std::vector<std::tuple<const Label *, const Label *, size_t, size_t>>
            tasks;
        tasks.reserve(paths.size() * paths.size() *
                      4);  // adjust estimate as needed

        for (const Label *label_i : best) {
            for (const Label *label_j : best) {
                if (label_i == label_j) continue;
                const auto &route_i = label_i->nodes_covered;
                const auto &route_j = label_j->nodes_covered;
                // Skip if routes are too short.
                if (route_i.size() < 3 || route_j.size() < 3) continue;
                // Generate tasks for valid internal positions.
                for (size_t k = 1; k < route_i.size() - 1; ++k) {
                    for (size_t l = 1; l < route_j.size() - 1; ++l) {
                        tasks.emplace_back(label_i, label_j, k, l);
                    }
                }
            }
        }

        // We'll use per-chunk local storage to avoid locking per improvement.
        const size_t num_tasks = tasks.size();
        const size_t chunk_size = 1;  // Adjust chunk size based on experiments.
        const size_t num_chunks = (num_tasks + chunk_size - 1) / chunk_size;

        // Prepare a vector of per-chunk improved labels.
        std::vector<std::vector<Label *>> local_best_new(num_chunks);

        // Optionally, you can also create per-chunk accumulators for operator
        // metrics. For brevity, we'll leave operator metric updates using
        // locks.

        auto bulk_sender = stdexec::bulk(
            stdexec::just(), num_chunks,
            [&tasks, chunk_size, &local_best_new, this](std::size_t chunk_idx) {
                const size_t start_idx = chunk_idx * chunk_size;
                const size_t end_idx =
                    std::min(start_idx + chunk_size, tasks.size());

                // Local vector to accumulate improved labels.
                std::vector<Label *> local_improvements;
                local_improvements.reserve(10);

                for (size_t task_idx = start_idx; task_idx < end_idx;
                     ++task_idx) {
                    const auto &[label_i, label_j, k, l] = tasks[task_idx];
                    const auto &route_i = label_i->getRoute();
                    const auto &route_j = label_j->getRoute();

                    if (route_i.size() < 3 || route_j.size() < 3) continue;

                    // Select operator based on its weight.
                    int op_idx = select_operator(rng);

                    // Apply the chosen operator.
                    auto [new_route_i, new_route_j] =
                        apply_operator(op_idx, route_i, route_j, k, l);
                    if (new_route_i.empty() || new_route_j.empty()) continue;

                    // Compute new route costs.
                    auto cost_i = compute_cost(new_route_i);
                    auto cost_j = compute_cost(new_route_j);
                    Path new_path_i{new_route_i, cost_i.first};
                    Path new_path_j{new_route_j, cost_j.first};

                    // Lambda to update a label if an improvement is found.
                    auto add_improved_label =
                        [&](const Path &path,
                            const std::pair<double, double> &cost,
                            const Label *orig_label, int op_idx) {
                            double new_cost = cost.second;
                            double current_cost = orig_label->cost;
                            if (new_cost < current_cost - 1e-3) {
                                // Update operator performance metrics.
                                {
                                    std::lock_guard<std::mutex> lock(
                                        operator_mutex);
                                    operator_improvements[op_idx] +=
                                        current_cost - new_cost;
                                    operator_success_count[op_idx] += 1;
                                }
                                // Create a new improved label.
                                auto best_new_label = new Label();
                                best_new_label->addRoute(path.route);
                                best_new_label->real_cost = cost.first;
                                best_new_label->cost = new_cost;
                                local_improvements.push_back(best_new_label);
                            }
                        };

                    if (is_feasible(new_path_i))
                        add_improved_label(new_path_i, cost_i, label_i, op_idx);
                    if (is_feasible(new_path_j))
                        add_improved_label(new_path_j, cost_j, label_j, op_idx);
                }
                // Save the local improvements for this chunk.
                local_best_new[chunk_idx] = std::move(local_improvements);
            });

        // Submit work and wait for completion.
        auto work = stdexec::starts_on(sched, bulk_sender);
        stdexec::sync_wait(std::move(work));

        // Merge local improvements into the global best_new vector.
        for (auto &vec : local_best_new) {
            best_new.insert(best_new.end(), vec.begin(), vec.end());
        }

        // Update the operator weights based on the collected performance.
        update_operator_weights();

        // Sort the improved labels and keep only the top N_ADD solutions.
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
    bool is_feasible(const Path &route) const {
        double current_time = 0.0;
        double current_capacity = 0.0;
        const double total_capacity = instance.q;
        const size_t route_size = route.route.size();

        // Ensure the route starts at depot 0 and ends at depot N_SIZE - 1.
        if (route_size < 2 || route.route.front() != 0 ||
            route.route.back() != N_SIZE - 1) {
            return false;
        }

        // Prepare a thread-local bitset for tracking visited nodes.
        const size_t n_segments = (N_SIZE >> 6) + 1;  // Same as N_SIZE/64 + 1.
        static thread_local std::vector<uint64_t> visited(n_segments, 0);
        std::fill(visited.begin(), visited.end(), 0);

        // Cache pointer to the route data to avoid repeated bounds checking.
        const uint16_t *const rdata = route.route.data();

        // Iterate through the route.
        for (size_t i = 0; i < route_size - 1; ++i) {
            const int source_id = rdata[i];
            const int target_id = rdata[i + 1];

            // Determine the bitset segment and position for the source node.
            const size_t segment = source_id >> 6;
            const size_t bit_position = source_id & 63;

            // Check if the source node has already been visited.
            if (visited[segment] & (1ULL << bit_position)) {
                return false;
            }
            // Mark the source node as visited.
            visited[segment] |= (1ULL << bit_position);

            // Cache node references.
            const auto &source = nodes[source_id];
            const auto &target = nodes[target_id];

            // Compute travel time and arrival time.
            const double travel_time =
                source.duration + instance.getcij(source_id, target_id);
            const double arrival_time = std::max(
                static_cast<double>(target.lb[0]), current_time + travel_time);

            // Check the time window for the target node.
            if (arrival_time > target.ub[0]) {
                return false;
            }

            // Update current time and capacity.
            current_time = arrival_time;
            current_capacity += source.demand;
            if (current_capacity > total_capacity) {
                return false;
            }
        }
        return true;
    }

    std::pair<double, double> compute_cost(const std::vector<uint16_t> &route) {
        const size_t n = route.size();
        if (n < 2) return {0.0, 0.0};

        double cost = 0.0;
        double red_cost = 0.0;
        const uint16_t *const rdata = route.data();

        for (size_t i = 0; i < n - 1; ++i) {
            // Cache current and next route entries.
            const uint16_t cur = rdata[i];
            const uint16_t nxt = rdata[i + 1];

            // Compute travel cost between the current and next node.
            const double travel = instance.getcij(cur, nxt);

            // Add travel cost.
            cost += travel;

            // Subtract node-specific cost from travel cost to compute reduced
            // cost.
            red_cost += travel - nodes[cur].cost;
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
