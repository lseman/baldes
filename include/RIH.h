

#pragma once
#include "Path.h"
#include <algorithm>
#include <functional>

#include "Reader.h"

#include "VRPNode.h"

#include "Label.h"
#include "RNG.h"

#include "Cut.h"

#include <exec/task.hpp>
#include <queue>
class IteratedLocalSearch {
public:
    InstanceData         instance;
    std::vector<VRPNode> nodes;
    CutStorage          *cut_storage = new CutStorage();

    // default default initialization passing InstanceData &instance
    IteratedLocalSearch(const InstanceData &instance) : instance(instance) {}

    /**
     * @brief Performs the 2-opt optimization on a given route.
     *
     * This function takes a route and two indices, i and j, and returns a new route
     * where the segment between i and j is reversed. The function ensures that the
     * indices are within bounds and that the depots (first and last elements) are
     * not involved in the reversal.
     *
     */
    static std::vector<int> two_opt(const std::vector<int> &route, int i, int j) {
        // Ensure indices are within bounds
        if (i >= j || i == 0 || j == route.size() - 1) {
            return route; // No changes if depots are involved
        }

        // Create a new route with the segment between i and j reversed
        std::vector<int> new_route(route);
        std::reverse(new_route.begin() + i, new_route.begin() + j + 1);
        return new_route;
    }

    /**
     * @brief Relocates a customer from one route to another.
     *
     * This function relocates the customer at position `i` in `route1` to position `j` in `route2`.
     * If the positions `i` or `j` involve depots (i.e., the first or last positions in the routes),
     * no changes are made and the original routes are returned.
     *
     */
    std::pair<std::vector<int>, std::vector<int>> relocate_star(const std::vector<int> &route1,
                                                                const std::vector<int> &route2, int i, int j) {
        // Relocate customer at position i in route1 to position j in route2
        if (i == 0 || i == route1.size() - 1 || j == 0 || j == route2.size() - 1) {
            return {route1, route2}; // No changes if depots are involved
        }

        std::vector<int> new_route1 = route1;
        std::vector<int> new_route2 = route2;

        int customer = route1[i];
        new_route1.erase(new_route1.begin() + i);            // Remove from route1
        new_route2.insert(new_route2.begin() + j, customer); // Insert into route2

        return {new_route1, new_route2};
    }

    /**
     * @brief Swaps segments of varying lengths between two routes.
     *
     * This function takes two routes and swaps segments of up to 3 elements between them,
     * starting at specified indices. If the start index is at the beginning or end of a route,
     * no changes are made to avoid swapping depot nodes.
     *
     */
    std::pair<std::vector<int>, std::vector<int>> enhanced_swap(const std::vector<int> &route1,
                                                                const std::vector<int> &route2, int i, int j) {
        // Swap segments of varying lengths between two routes
        if (i == 0 || i == route1.size() - 1 || j == 0 || j == route2.size() - 1) {
            return {route1, route2}; // No changes if depots are selected
        }

        // Determine segment length to swap
        int              segment_length = std::min(3, static_cast<int>(std::min(route1.size() - i, route2.size() - j)));
        std::vector<int> new_route1(route1);
        std::vector<int> new_route2(route2);

        std::swap_ranges(new_route1.begin() + i, new_route1.begin() + i + segment_length, new_route2.begin() + j);

        return {new_route1, new_route2};
    }

    /**
     * @brief Performs a crossover operation between two routes at specified positions.
     *
     * This function takes two routes and two crossover points, and swaps the tails of the routes
     * after the specified positions. The depots (first and last elements) are not allowed to be
     * crossover points.
     *
     */
    std::pair<std::vector<int>, std::vector<int>> cross(const std::vector<int> &route1, const std::vector<int> &route2,
                                                        int k, int l) {
        // Ensure we don't cross depots (first and last elements)
        if (k == 0 || k == route1.size() - 1 || l == 0 || l == route2.size() - 1) {
            return {route1, route2}; // No changes if depots are selected
        }

        // Swap the tails of the routes after position k and l
        std::vector<int> new_route1(route1.begin(), route1.begin() + k);
        new_route1.insert(new_route1.end(), route2.begin() + l, route2.end());

        std::vector<int> new_route2(route2.begin(), route2.begin() + l);
        new_route2.insert(new_route2.end(), route1.begin() + k, route1.end());

        return {new_route1, new_route2};
    }

    /**
     * @brief Inserts a customer from one route into another route at specified positions.
     *
     * This function takes two routes and inserts a customer from the first route
     * at position `k` into the second route at position `l`. It ensures that
     * depots (first and last positions) are not involved in the insertion process.
     *
     */
    std::pair<std::vector<int>, std::vector<int>> insertion(const std::vector<int> &route1,
                                                            const std::vector<int> &route2, int k, int l) {
        // Ensure we are not moving or inserting into depots
        if (k == 0 || k == route1.size() - 1 || l == 0 || l == route2.size() - 1) {
            return {route1, route2}; // No changes if depots are involved
        }

        // Insert customer from route1 at position k into route2 at position l
        std::vector<int> new_route1 = route1;
        std::vector<int> new_route2 = route2;

        if (k < route1.size()) {
            int customer = route1[k];
            new_route1.erase(new_route1.begin() + k);
            new_route2.insert(new_route2.begin() + l, customer);
        }

        return {new_route1, new_route2};
    }

    /**
     * @brief Swaps customers between two routes at specified positions.
     *
     * This function takes two routes and swaps the customers at the specified
     * positions k and l. It ensures that depot positions (first and last elements)
     * are not swapped.
     *
     */
    std::pair<std::vector<int>, std::vector<int>> swap(const std::vector<int> &route1, const std::vector<int> &route2,
                                                       int k, int l) {
        // Ensure we are not swapping depots
        if (k == 0 || k == route1.size() - 1 || l == 0 || l == route2.size() - 1) {
            return {route1, route2}; // No changes if depots are selected
        }

        // Swap customers between route1 at position k and route2 at position l
        std::vector<int> new_route1 = route1;
        std::vector<int> new_route2 = route2;

        if (k < route1.size() && l < route2.size()) { std::swap(new_route1[k], new_route2[l]); }

        return {new_route1, new_route2};
    }

    /**
     * @brief Swaps sequences of customers between two routes and inserts them at optimal positions.
     *
     * This operator swaps a chain of 2 or 3 consecutive customers between two routes, and inserts
     * each chain into the best possible position in the opposite route, ensuring depots are not involved.
     */
    std::pair<std::vector<int>, std::vector<int>> extended_swap_star(const std::vector<int> &route1,
                                                                     const std::vector<int> &route2, int i, int j,
                                                                     int chain_length = 2) {
        // Ensure we are not swapping depot nodes or exceeding bounds
        if (i == 0 || i + chain_length > route1.size() - 1 || j == 0 || j + chain_length > route2.size() - 1) {
            return {route1, route2}; // No changes if depots or out-of-bound indices are involved
        }

        // Swap a chain of customers between route1 and route2
        std::vector<int> new_route1 = route1;
        std::vector<int> new_route2 = route2;

        // Extract the chains of customers
        std::vector<int> chain1(new_route1.begin() + i, new_route1.begin() + i + chain_length);
        std::vector<int> chain2(new_route2.begin() + j, new_route2.begin() + j + chain_length);

        // Remove the chains from their respective routes
        new_route1.erase(new_route1.begin() + i, new_route1.begin() + i + chain_length);
        new_route2.erase(new_route2.begin() + j, new_route2.begin() + j + chain_length);

        // Insert the chains at the best positions in the opposite routes
        auto best_pos_route1 = find_best_insertion_position(new_route1, chain2);
        auto best_pos_route2 = find_best_insertion_position(new_route2, chain1);

        new_route1.insert(new_route1.begin() + best_pos_route1, chain2.begin(), chain2.end());
        new_route2.insert(new_route2.begin() + best_pos_route2, chain1.begin(), chain1.end());

        return {new_route1, new_route2};
    }

    /**
     * @brief Finds the best position to insert a chain of customers into a route.
     *
     * This helper function evaluates the cost of inserting a chain into each possible position
     * in the route and returns the index of the best insertion point.
     */
    int find_best_insertion_position(const std::vector<int> &route, const std::vector<int> &chain) {
        int    best_pos  = 1;
        double best_cost = std::numeric_limits<double>::max();

        // Iterate over each possible insertion point (ignoring depot nodes)
        for (int pos = 1; pos < route.size(); ++pos) {
            std::vector<int> new_route = route;
            new_route.insert(new_route.begin() + pos, chain.begin(), chain.end());

            // Compute the cost of this insertion
            double cost = this->compute_cost(new_route).second;

            if (cost < best_cost) {
                best_cost = cost;
                best_pos  = pos;
            }
        }
        return best_pos;
    }

    /**
     * @brief Relocates a sequence of customers from one route to another.
     *
     * This operator moves a chain of 2 or 3 consecutive customers from one route
     * to another. The depots (first and last elements) are not involved.
     */
    std::pair<std::vector<int>, std::vector<int>> extended_relocate_star(const std::vector<int> &route1,
                                                                         const std::vector<int> &route2, int i, int j,
                                                                         int chain_length = 2) {
        // Ensure we are not relocating depot nodes or exceeding bounds
        if (i == 0 || i + chain_length > route1.size() - 1 || j == 0 || j == route2.size() - 1) {
            return {route1, route2}; // No changes if depots or out-of-bound indices are involved
        }

        // Relocate a chain of customers from route1 to route2
        std::vector<int> new_route1 = route1;
        std::vector<int> new_route2 = route2;

        // Extract the chain of customers to relocate
        std::vector<int> chain(new_route1.begin() + i, new_route1.begin() + i + chain_length);

        // Remove the chain from route1
        new_route1.erase(new_route1.begin() + i, new_route1.begin() + i + chain_length);

        // Insert the chain into route2 at position j
        new_route2.insert(new_route2.begin() + j, chain.begin(), chain.end());

        return {new_route1, new_route2};
    }

    ///////////////////////////////////////
    // End operators
    ///////////////////////////////////////
    // Adaptive operator selection
    std::vector<std::function<std::pair<std::vector<int>, std::vector<int>>(const std::vector<int> &,
                                                                            const std::vector<int> &, int, int)>>
        operators = {[this](const std::vector<int> &route1, const std::vector<int> &route2, int k, int l) {
                         return this->cross(route1, route2, k, l);
                     },
                     [this](const std::vector<int> &route1, const std::vector<int> &route2, int k, int l) {
                         return this->insertion(route1, route2, k, l);
                     },
                     [this](const std::vector<int> &route1, const std::vector<int> &route2, int k, int l) {
                         return this->swap(route1, route2, k, l);
                     },
                     [this](const std::vector<int> &route1, const std::vector<int> &route2, int k, int l) {
                         return this->relocate_star(route1, route2, k, l);
                     },
                     [this](const std::vector<int> &route1, const std::vector<int> &route2, int k, int l) {
                         return this->enhanced_swap(route1, route2, k, l);
                     }};

    std::vector<double> operator_weights      = {1.0, 1.0, 1.0}; // Start with equal weights
    std::vector<double> operator_improvements = {0.0, 0.0, 0.0};

    // Utility function to select operator based on weights
    int select_operator(Xoroshiro128Plus &rng) {
        std::vector<double> cumulative_weights(operator_weights.size());
        std::partial_sum(operator_weights.begin(), operator_weights.end(), cumulative_weights.begin());

        // Generate random number and select operator based on cumulative weights
        double random_value = (static_cast<long double>(rng()) / static_cast<long double>(Xoroshiro128Plus::max())) *
                              cumulative_weights.back();
        for (size_t i = 0; i < cumulative_weights.size(); ++i) {
            if (random_value <= cumulative_weights[i]) { return i; }
        }
        return cumulative_weights.size() - 1; // Fallback to the last operator
    }

    void update_operator_weights() {
        // Normalize improvements and update weights (use some smoothing factor)
        double total_improvement = std::accumulate(operator_improvements.begin(), operator_improvements.end(), 0.0);
        if (total_improvement > 0) {
            for (size_t i = 0; i < operator_weights.size(); ++i) {
                operator_weights[i] = 0.9 * operator_weights[i] + 0.1 * (operator_improvements[i] / total_improvement);
            }
        }
    }

    exec::static_thread_pool            pool  = exec::static_thread_pool(5);
    exec::static_thread_pool::scheduler sched = pool.get_scheduler();

    std::vector<Label *> perturbation(const std::vector<Label *> &paths, const std::vector<VRPNode> &nodes) {
        this->nodes               = nodes;
        std::vector<Label *> best = paths;
        std::vector<Label *> best_new;
        Xoroshiro128Plus     rng; // Instantiate the custom RNG with default seed
        bool                 is_stuck = false;

        std::vector<double> best_costs(paths.size(), std::numeric_limits<double>::max());

        // Store tasks for parallel processing
        std::vector<std::tuple<const Label *, const Label *, size_t, size_t>> tasks;

        for (const Label *label_i : best) {
            for (const Label *label_j : best) {
                if (label_i == label_j) continue;
                const auto &route_i = label_i->nodes_covered;
                const auto &route_j = label_j->nodes_covered;

                if (route_i.size() < 3 || route_j.size() < 3) continue;

                for (size_t k = 1; k < route_i.size() - 1; ++k) {
                    for (size_t l = 1; l < route_j.size() - 1; ++l) { tasks.emplace_back(label_i, label_j, k, l); }
                }
            }
        }

        // Parallelize the task processing
        std::mutex best_new_mutex; // Mutex to protect access to best_new and operator_improvements
        std::mutex operator_mutex; // Mutex to protect operator improvements

        // Define chunk size to balance load
        const int chunk_size = 1; // Adjust chunk size based on performance experiments

        // Parallel execution in chunks
        auto bulk_sender = stdexec::bulk(
            stdexec::just(), (tasks.size() + chunk_size - 1) / chunk_size,
            [&best_new, &best_new_mutex, &operator_mutex, &tasks, chunk_size, &rng, this](std::size_t chunk_idx) {
                size_t start_idx = chunk_idx * chunk_size;
                size_t end_idx   = std::min(start_idx + chunk_size, tasks.size());

                for (size_t task_idx = start_idx; task_idx < end_idx; ++task_idx) {
                    const auto &[label_i, label_j, k, l] = tasks[task_idx];
                    const auto &route_i                  = label_i->nodes_covered;
                    const auto &route_j                  = label_j->nodes_covered;

                    int op_idx                      = select_operator(rng); // Select operator based on weights
                    auto [new_route_i, new_route_j] = operators[op_idx](route_i, route_j, k, l);
                    if (new_route_i.empty() || new_route_j.empty()) continue;

                    auto cost_i     = compute_cost(new_route_i);
                    auto cost_j     = compute_cost(new_route_j);
                    Path new_path_i = Path{new_route_i, cost_i.first};
                    Path new_path_j = Path{new_route_j, cost_j.first};

                    // Lambda function to add an improved label to best_new
                    auto add_improved_label = [&](const auto &path, const auto &cost, const Label *label, int op_idx) {
                        double new_cost     = cost.second;
                        double current_cost = label->cost;

                        // Check if the new cost is better
                        if (new_cost < current_cost - 1e-3) {
                            // fmt::print("Improvement: {} -> {}\n", current_cost, new_cost);

                            // Lock to update operator performance
                            {
                                std::lock_guard<std::mutex> lock(operator_mutex);
                                operator_improvements[op_idx] += current_cost - new_cost;
                            }

                            auto best_new_label           = new Label();
                            best_new_label->nodes_covered = path.route;
                            best_new_label->real_cost     = cost.first;
                            best_new_label->cost          = new_cost;

                            // Lock to update best_new
                            {
                                std::lock_guard<std::mutex> lock(best_new_mutex);
                                best_new.push_back(best_new_label);
                            }
                        }
                    };

                    // Outside the lambda: check feasibility and improvements separately
                    if (is_feasible(new_path_i)) { add_improved_label(new_path_i, cost_i, label_i, op_idx); }

                    if (is_feasible(new_path_j)) { add_improved_label(new_path_j, cost_j, label_j, op_idx); }
                }
            });

        // Submit work to the thread pool
        auto work = stdexec::starts_on(sched, bulk_sender);
        stdexec::sync_wait(std::move(work));

        // Update operator weights based on performance
        update_operator_weights();

        // Sort and limit to top solutions
        pdqsort(best_new.begin(), best_new.end(), [](const Label *a, const Label *b) { return a->cost < b->cost; });
        if (best_new.size() > 5) { best_new.erase(best_new.begin() + 5, best_new.end()); }

        return best_new;
    }

private:
    // Feasibility check and cost computation functions
    bool is_feasible(Path &route) const {
        double       time           = 0.0;
        double       capacity       = 0.0;
        const double total_capacity = instance.q;

        // Ensure the route starts at depot 0 and ends at the last node (depot N_SIZE - 1)
        if (route.route.front() != 0 || route.route.back() != N_SIZE - 1) {
            return false; // Infeasible if the route doesn't start and end at the depots
        }

        // Preallocate and reuse the Bvisited bitset for tracking visited nodes
        const size_t                              n_segments = N_SIZE / 64 + 1;
        static thread_local std::vector<uint64_t> Bvisited(n_segments, 0); // Use thread-local to avoid reallocation

        // Reset visited nodes tracker
        std::fill(Bvisited.begin(), Bvisited.end(), 0);

        // Precompute cij values (store them in memory if possible for faster access)
        for (size_t i = 0; i < route.size() - 1; ++i) {
            const int source_id = route[i];
            const int target_id = route[i + 1];

            // Directly access the segment and bit position for the visited node
            const size_t segment      = source_id >> 6;
            const size_t bit_position = source_id & 63;

            // Check if node has been visited
            if (Bvisited[segment] & (1ULL << bit_position)) {
                return false; // Node was already visited, route is infeasible
            }

            // Mark node as visited
            Bvisited[segment] |= (1ULL << bit_position);

            const auto &source = nodes[source_id];
            const auto &target = nodes[target_id];

            // Use cached or precomputed values for cij
            double travel_time = source.duration + instance.getcij(source_id, target_id);
            double start_time  = std::max(static_cast<double>(target.lb[0]), time + travel_time);

            // If the start time exceeds the due date of the target customer, it's infeasible
            if (start_time > target.ub[0]) {
                return false; // Time window violation
            }

            time = start_time; // Update time for next node

            // Update the capacity and check if it exceeds the total capacity
            capacity += source.demand;
            if (capacity > total_capacity) {
                return false; // Capacity exceeded
            }
        }

        return true;
    }

    std::pair<double, double> compute_cost(const std::vector<int> &route) {
        double cost     = 0;
        double red_cost = 0;

        auto &cutter   = cut_storage;      // Access the cut storage manager
        auto &SRCDuals = cutter->SRCDuals; // Access the dual values for the SRC cuts
        SRC_MODE_BLOCK(std::vector<int> SRCmap; SRCmap.resize(cutter->size(), 0.0);)

        for (size_t i = 0; i < route.size() - 1; ++i) {
            auto travel_cost = instance.getcij(route[i], route[i + 1]);
            auto node        = route[i];
            cost += travel_cost;
            // If the reduced cost should be based on travel cost minus node-specific costs, we subtract that here
            red_cost += travel_cost - nodes[route[i]].cost;

            SRC_MODE_BLOCK(size_t segment      = node >> 6; // Determine the segment in the bitmap
                           size_t bit_position = node & 63; // Determine the bit position in the segment

                           const uint64_t bit_mask = 1ULL
                                                     << bit_position; // Precompute bit shift for the node's position
                           for (std::size_t idx = 0; idx < cutter->size(); ++idx) {
                               auto it = cutter->begin();
                               std::advance(it, idx);
                               const auto &cut          = *it;
                               const auto &baseSet      = cut.baseSet;
                               const auto &baseSetorder = cut.baseSetOrder;
                               const auto &neighbors    = cut.neighbors;
                               const auto &multipliers  = cut.p;

                               // Apply SRC logic: Update the SRC map based on neighbors and base set
                               bool bitIsSet  = neighbors[segment] & bit_mask;
                               bool bitIsSet2 = baseSet[segment] & bit_mask;

                               auto &src_map_value = SRCmap[idx]; // Use reference to avoid multiple accesses
                               if (!bitIsSet) {
                                   src_map_value = 0.0; // Reset the SRC map value
                               }

                               if (bitIsSet2) {
                                   auto &den = multipliers.den;
                                   src_map_value += multipliers.num[baseSetorder[node]];
                                   if (src_map_value >= den) {
                                       red_cost -= SRCDuals[idx]; // Apply the SRC dual value if threshold is exceeded
                                       src_map_value -= den;      // Reset the SRC map value
                                   }
                               }
                           })
        }
        return {cost, red_cost};
    }
    // Task queue and synchronization primitives
    std::queue<std::pair<std::vector<Label *>, std::vector<VRPNode>>> task_queue;
    std::mutex                                                        queue_mutex;
    std::condition_variable                                           queue_condition;
    bool                                                              shutdown = false;
    const int task_threshold = 5; // The threshold for processing tasks

    // Wait for enough tasks (5 tasks in the queue)
    exec::task<void> wait_for_tasks() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        queue_condition.wait(lock, [this] { return task_queue.size() >= task_threshold || shutdown; });
        co_return;
    }

    // Process a batch of 5 tasks
    exec::task<void> process_tasks(IteratedLocalSearch &ils) {
        std::vector<std::pair<std::vector<Label *>, std::vector<VRPNode>>> tasks_to_process;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            for (int i = 0; i < task_threshold && !task_queue.empty(); ++i) {
                tasks_to_process.push_back(std::move(task_queue.front()));
                task_queue.pop();
            }
        }

        // Process the batch of tasks
        for (const auto &task : tasks_to_process) {
            ils.perturbation(task.first, task.second);
            std::cout << "Processed a task!" << std::endl;
        }

        co_return;
    }

    // Continuous task processing loop using stdexec coroutines
    exec::task<void> task_worker(IteratedLocalSearch &ils) {
        while (!shutdown) {
            co_await wait_for_tasks();   // Wait until there are enough tasks
            co_await process_tasks(ils); // Process the tasks
        }
        co_return;
    }

    // Function to submit new tasks to the queue
    void submit_task(const std::vector<Label *> &paths, const std::vector<VRPNode> &nodes) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            task_queue.emplace(paths, nodes); // Add new task to the queue
        }
        queue_condition.notify_one(); // Notify the worker that a new task is available
    }

    // Function to gracefully shut down the worker
    void stop_worker() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            shutdown = true; // Signal the worker to shut down
        }
        queue_condition.notify_all(); // Wake up the worker to shut it down
    }
};
