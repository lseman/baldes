

#pragma once
#include "Path.h"
#include <algorithm>
#include <functional>

#include "Reader.h"

#include "VRPNode.h"

#include "Label.h"
#include "RNG.h"

class IteratedLocalSearch {
public:
    InstanceData         instance;
    std::vector<VRPNode> nodes;
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
    static std::pair<std::vector<int>, std::vector<int>> relocate_star(const std::vector<int> &route1,
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
    static std::pair<std::vector<int>, std::vector<int>> enhanced_swap(const std::vector<int> &route1,
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
    static std::pair<std::vector<int>, std::vector<int>> cross(const std::vector<int> &route1,
                                                               const std::vector<int> &route2, int k, int l) {
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
    static std::pair<std::vector<int>, std::vector<int>> insertion(const std::vector<int> &route1,
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
    static std::pair<std::vector<int>, std::vector<int>> swap(const std::vector<int> &route1,
                                                              const std::vector<int> &route2, int k, int l) {
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

    // Adaptive operator selection
    std::vector<std::function<std::pair<std::vector<int>, std::vector<int>>(const std::vector<int> &,
                                                                            const std::vector<int> &, int, int)>>
        operators = {cross, insertion, swap, relocate_star, enhanced_swap};

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

    std::vector<Label *> perturbation(const std::vector<Label *> &paths, std::vector<VRPNode> &nodes) {
        this->nodes               = nodes;
        std::vector<Label *> best = paths;
        std::vector<Label *> best_new;
        Xoroshiro128Plus     rng; // Instantiate the custom RNG with default seed
        bool                 is_stuck = false;

        std::vector<double> best_costs(paths.size(), std::numeric_limits<double>::max());

        // Parallelization of outer loops using std::execution::par
        for (const Label *label_i : best) {
            for (const Label *label_j : best) {
                if (label_i == label_j) continue;
                const auto &route_i = label_i->nodes_covered;
                const auto &route_j = label_j->nodes_covered;

                if (route_i.size() < 3 || route_j.size() < 3) continue;

                for (size_t k = 1; k < route_i.size() - 1; ++k) {
                    for (size_t l = 1; l < route_j.size() - 1; ++l) {
                        int op_idx                      = select_operator(rng); // Select operator based on weights
                        auto [new_route_i, new_route_j] = operators[op_idx](route_i, route_j, k, l);
                        if (new_route_i.empty() || new_route_j.empty()) continue;

                        auto cost_i     = compute_cost(new_route_i);
                        auto cost_j     = compute_cost(new_route_j);
                        Path new_path_i = Path{new_route_i, cost_i.first};
                        Path new_path_j = Path{new_route_j, cost_j.first};

                        if (is_feasible(new_path_i) && is_feasible(new_path_j)) {
                            double new_cost     = cost_i.second + cost_j.second;
                            double current_cost = label_i->cost + label_j->cost;

                            if (new_cost < current_cost) {
                                // Update operator performance
                                operator_improvements[op_idx] += current_cost - new_cost;

                                // Update best solution
                                auto best_new_i_label           = new Label();
                                best_new_i_label->nodes_covered = new_path_i.route;
                                best_new_i_label->real_cost     = cost_i.first;
                                best_new_i_label->cost          = cost_i.second;

                                auto best_new_j_label           = new Label();
                                best_new_j_label->nodes_covered = new_path_j.route;
                                best_new_j_label->real_cost     = cost_j.first;
                                best_new_j_label->cost          = cost_j.second;

                                best_new.push_back(best_new_i_label);
                                best_new.push_back(best_new_j_label);

                                is_stuck = false;
                            }
                        }
                    }
                }
            }
        }

        // Update operator weights based on performance
        update_operator_weights();

        // Sort and limit to top solutions
        std::sort(best_new.begin(), best_new.end(), [](const Label *a, const Label *b) { return a->cost < b->cost; });
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

        for (size_t i = 0; i < route.size() - 1; ++i) {
            auto travel_cost = instance.getcij(route[i], route[i + 1]);
            cost += travel_cost;
            // If the reduced cost should be based on travel cost minus node-specific costs, we subtract that here
            red_cost += travel_cost - nodes[route[i]].cost;
        }

        return {cost, red_cost};
    }
};
