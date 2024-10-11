

#include <algorithm>
#include <functional>
class IteratedLocalSearch {
public:
    std::vector<Path> perturbation(const std::vector<Path> &paths) {
        std::vector<Path> best     = paths;
        bool              is_stuck = false;

        // Define your operators as a list of functions
        std::vector<std::function<std::pair<std::vector<int>, std::vector<int>>(const std::vector<int> &,
                                                                                const std::vector<int> &, int, int)>>
            operators = {cross, insertion, swap};

        // Main iterative loop
        while (!is_stuck) {
            is_stuck = true;

            // Iterate through pairs of paths
            for (size_t i = 0; i < best.size() - 1; ++i) {
                for (size_t j = i + 1; j < best.size(); ++j) {

                    const auto &route_i = best[i].route;
                    const auto &route_j = best[j].route;

                    // Iterate over all positions in both paths
                    for (size_t k = 0; k <= route_i.size(); ++k) {
                        for (size_t l = 0; l <= route_j.size(); ++l) {

                            // Apply each operator (cross, insertion, swap)
                            for (const auto &op : operators) {
                                auto [new_route_i, new_route_j] = op(route_i, route_j, k, l);
                                Path new_path_i                 = Path{new_route_i, compute_cost(new_route_i)};
                                Path new_path_j                 = Path{new_route_j, compute_cost(new_route_j)};

                                // Check feasibility and if the new paths are better
                                if (new_path_i.is_feasible() && new_path_j.is_feasible()) {
                                    double new_total_distance     = new_path_i.cost + new_path_j.cost;
                                    double current_total_distance = best[i].cost + best[j].cost;

                                    if (new_total_distance < current_total_distance) {
                                        // Update paths if improvement found
                                        best[i]  = new_path_i;
                                        best[j]  = new_path_j;
                                        is_stuck = false; // Continue searching
                                    }
                                }
                            }

                        } // End of l-loop
                    } // End of k-loop
                } // End of j-loop
            } // End of i-loop
        } // End of while-loop

        return best; // Return the best set of paths found
    }

private:
    // Feasibility check and cost computation functions
    bool is_feasible(const Path &path) {
        // Add feasibility logic here
        return true; // Placeholder
    }

    double compute_cost(const std::vector<int> &route) {
        // Add cost computation logic here
        return route.size(); // Placeholder (replace with your actual cost calculation)
    }
};
