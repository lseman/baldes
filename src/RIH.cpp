/**
 * @file RIH.cpp
 * @brief Implementation of Route Improvement Heuristics (RIH) for Vehicle Routing Problems (VRP).
 *
 * This file contains the implementation of various route improvement heuristics (RIH) used to optimize the
 * solutions of vehicle routing problems (VRP). The heuristics aim to improve route costs by inserting, deleting,
 * swapping, or performing 2-opt exchanges on customers within routes.
 *
 * The main components of this file include:
 * - RIH1: Refines the routes by inserting new customers into the existing routes and recalculating the costs.
 * - RIH2: Refines the routes by removing customers from the routes and recalculating the costs.
 * - RIH3: Improves routes by swapping neighboring customers within a route and recalculating the costs.
 * - RIH4: Applies the 2-opt local search heuristic, performing exchanges between non-neighboring customers to improve
 * route costs.
 *
 * The functions make use of priority queues to manage the best labels (routes) and process them iteratively based on
 * their costs. The objective is to generate feasible solutions that minimize route costs while respecting
 * problem-specific constraints.
 *
 * The main methods implemented in this file provide functionality for:
 * - Inserting customers into routes, recalculating the cost and feasibility of the new routes.
 * - Deleting customers from routes, efficiently adjusting the affected segments and maintaining feasibility.
 * - Swapping customers within routes, and updating the cost and feasibility accordingly.
 * - Performing 2-opt exchanges, reversing portions of routes to explore better solutions.
 *
 * These heuristics are commonly used in branch-and-bound or branch-and-price algorithms to enhance the quality of
 * the solution by making local modifications to the routes.
 */

#include "../bucket/BucketGraph.h"
#include "../include/Definitions.h"

/* Improvement heuristics based on the insertion/deletion of customers */
/* Operation 1: modify the routes by adding one customer to each route */
/**
 * @brief Refines the best labels by inserting new customers into the routes.
 *
 * This function processes the best labels from the input priority queue, attempts to insert new customers
 * into the routes, and pushes the refined labels into the output priority queue if they are feasible and
 * have a lower cost than the current label.
 *
 */
int BucketGraph::RIH1(std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_in,
                      std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_out,
                      int                                                                  max_n_labels) {

    int iter = 0;
    while (!best_labels_in.empty() && iter < max_n_labels) {
        iter++;

        Label *current_label = best_labels_in.top();
        best_labels_in.pop();

        // Remove similar labels from the queue
        while (!best_labels_in.empty() && best_labels_in.top()->cost < current_label->cost + RCESPP_TOL_ZERO) {
            best_labels_in.pop();
        }

        // If the label has only the depots covered, skip it
        if (current_label->jobs_covered.size() <= 3) { // Depot at start and end + at least one job
            continue;
        }

        // Iterate over positions to insert a new customer
        for (size_t i = 1; i < current_label->jobs_covered.size() - 1; ++i) { // Skip the first and last depot
            for (size_t new_customer = 1; new_customer <= jobs.size() - 2;
                 ++new_customer) { // Iterate over all possible new customers
                if (std::find(current_label->jobs_covered.begin(), current_label->jobs_covered.end(), new_customer) !=
                    current_label->jobs_covered.end()) {
                    continue; // Skip if the customer is already in the route
                }

                Label *new_label = label_pool_fw.acquire();
                if (!new_label) {
                    // Handle label acquisition failure
                    continue;
                }

                new_label->initialize(current_label->vertex, current_label->cost, {current_label->resources[0]},
                                      current_label->job_id);

                // Copy jobs covered up to the insertion point
                new_label->jobs_covered.assign(current_label->jobs_covered.begin(),
                                               current_label->jobs_covered.begin() + i);
                // Insert the new customer
                new_label->jobs_covered.push_back(new_customer);
                // Copy the remaining jobs covered
                new_label->jobs_covered.insert(new_label->jobs_covered.end(), current_label->jobs_covered.begin() + i,
                                               current_label->jobs_covered.end());

                // Reset costs and real cost
                new_label->cost      = 0.0;
                new_label->real_cost = 0.0;

                bool   feasible     = true;
                double elapsed_time = 0.0;

                // Calculate the new cost and check feasibility for each job in the new route
                for (size_t j = 0; j < new_label->jobs_covered.size() - 1; ++j) {
                    int  from           = new_label->jobs_covered[j];
                    int  to             = new_label->jobs_covered[j + 1];
                    auto cost_increment = getcij(to, from);

                    // Check for time window constraints
                    double new_time = elapsed_time + cost_increment;
                    if (new_time > jobs[to].end_time) {
                        feasible = false;
                        break;
                    }
                    if (new_time < jobs[to].start_time) { new_time = jobs[to].start_time; }
                    elapsed_time = new_time + jobs[to].duration;

                    // Update costs
                    new_label->cost += cost_increment + jobs[to].cost; // Ensure that job cost is correctly added
                    new_label->real_cost += cost_increment;
                }

                if (!feasible) { continue; }

                // Check feasibility of remaining route, ensure valid time windows for remaining jobs
                int last_customer = new_label->jobs_covered.back();
                while (last_customer != N_SIZE - 1) { // Until the last depot
                    int next_customer = last_customer + 1;
                    if (next_customer >= new_label->jobs_covered.size()) break;

                    double new_time = elapsed_time + getcij(last_customer, next_customer);
                    if (new_time > jobs[next_customer].end_time) {
                        feasible = false;
                        break;
                    }
                    if (new_time < jobs[next_customer].start_time) { new_time = jobs[next_customer].start_time; }
                    elapsed_time  = new_time + jobs[next_customer].duration;
                    last_customer = next_customer;
                }

                if (feasible) {
                    if (new_label->cost < current_label->cost) {
                        best_labels_out.push(new_label);
                    } else {
                    }
                } else {
                }
            }
        }
    }

    return 1;
}

/* Improvement heuristics based on the insertion/deletion of customers */
/* Operation 2: modify the routes by deleting one customer from each */

/**
 * @brief Refines a set of labels by iteratively removing customers from routes and recalculating costs.
 *
 * This function processes labels from an input priority queue, removes customers from the routes,
 * recalculates the costs, and pushes improved labels to an output priority queue. The process is
 * repeated until a maximum number of iterations is reached or the input queue is empty.
 *
 */
int BucketGraph::RIH2(std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_in,
                      std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_out,
                      int                                                                  max_n_labels) {

    int iter = 0;

    while (!best_labels_in.empty() && iter < max_n_labels && !merged_labels.empty()) {
        iter++;

        Label *current_label = best_labels_in.top();
        best_labels_in.pop();

        // Remove similar labels from the queue
        while (!best_labels_in.empty() && best_labels_in.top()->cost < current_label->cost + RCESPP_TOL_ZERO) {
            best_labels_in.pop();
        }

        // If the label has no jobs covered or only one job covered, skip it
        if (current_label->jobs_covered.size() <= 3) { // Depot at start and end + at least one job
            continue;
        }

        // Iterate over each customer in the route, skipping the first and last (which are depots)
        for (size_t i = 1; i < current_label->jobs_covered.size() - 1; ++i) {

            // Reuse the current label to avoid full reinitialization
            Label *new_label = label_pool_fw.acquire();
            if (!new_label) {
                // Handle label acquisition failure
                continue;
            }

            new_label->initialize(current_label->vertex, current_label->cost, {current_label->resources[0]},
                                  current_label->job_id);

            // Copy jobs covered except the i-th job (the customer to delete)
            new_label->jobs_covered.clear();
            new_label->jobs_covered.reserve(current_label->jobs_covered.size() - 1); // Reserve space
            for (size_t j = 0; j < current_label->jobs_covered.size(); ++j) {
                if (j != i) { new_label->jobs_covered.push_back(current_label->jobs_covered[j]); }
            }

            // Efficient recalculation of the route cost by only recalculating the affected segments
            new_label->cost      = current_label->cost;
            new_label->real_cost = current_label->real_cost;

            // Handle the two affected segments (before and after the removed customer)
            if (i > 0 && i < current_label->jobs_covered.size() - 1) {
                int prev = current_label->jobs_covered[i - 1]; // Job before the removed one
                int next = current_label->jobs_covered[i + 1]; // Job after the removed one

                // Recalculate the cost difference from removing the i-th job
                auto original_cost =
                    getcij(prev, current_label->jobs_covered[i]) + getcij(current_label->jobs_covered[i], next);
                auto new_cost = getcij(prev, next);

                // Adjust the cost and real cost
                new_label->cost += new_cost - original_cost;
                new_label->real_cost += new_cost - original_cost;

                // Check for feasibility: ensure the new route is within time constraints
                double elapsed_time = 0.0;
                bool   feasible     = true;
                for (size_t j = 0; j < new_label->jobs_covered.size() - 1; ++j) {
                    int    from           = new_label->jobs_covered[j];
                    int    to             = new_label->jobs_covered[j + 1];
                    double cost_increment = getcij(from, to);

                    double new_time = elapsed_time + cost_increment;
                    if (new_time > jobs[to].end_time) {
                        feasible = false; // Infeasible if exceeding time windows
                        break;
                    }
                    if (new_time < jobs[to].start_time) { new_time = jobs[to].start_time; }
                    elapsed_time = new_time + jobs[to].duration;
                }

                if (!feasible) { continue; }
            }

            // If the new label's cost is better, push it to the output queue
            if (new_label->cost < current_label->cost) {
                best_labels_out.push(new_label);
            } else {
            }
        }
    }

    return 1;
}

/* Improvement heuristics based on changing the position of customers */
/* Operation 3: swap operator */
/**
 * @brief Refines the input labels by swapping neighboring jobs and checking feasibility.
 *
 * This function processes the input priority queue of labels, attempting to improve each label by swapping
 * neighboring jobs and recalculating the cost. If the new label is feasible and has a lower cost, it is added
 * to the output priority queue.
 *
 */
int BucketGraph::RIH3(std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_in,
                      std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_out,
                      int                                                                  max_n_labels) {

    int iter = 0;
    while (!best_labels_in.empty() && iter < max_n_labels) {
        iter++;

        Label *current_label = best_labels_in.top();
        best_labels_in.pop();

        // Remove similar labels from the queue based on cost tolerance
        while (!best_labels_in.empty() && best_labels_in.top()->cost < current_label->cost + RCESPP_TOL_ZERO) {
            best_labels_in.pop();
        }

        // Skip labels that only cover depots (at least one job + two depots required)
        if (current_label->jobs_covered.size() <= 3) { continue; }

        // Iterate over jobs, skipping depots (first and last)
        for (size_t i = 1; i < current_label->jobs_covered.size() - 2; ++i) {
            int job1 = current_label->jobs_covered[i];
            int job2 = current_label->jobs_covered[i + 1];

            // Skip if jobs are the same (no change in route)
            if (job1 == job2) { continue; }

            // Acquire a new label for the swap
            Label *new_label = label_pool_fw.acquire();
            if (!new_label) {
                // Handle label acquisition failure
                continue;
            }

            new_label->initialize(current_label->vertex, current_label->cost, {current_label->resources[TIME_INDEX]},
                                  current_label->job_id);

            // Copy and swap neighboring jobs i and i+1
            new_label->jobs_covered = current_label->jobs_covered;
            std::swap(new_label->jobs_covered[i], new_label->jobs_covered[i + 1]);

            // Recalculate the cost for the affected part of the route (before, swapped jobs, after)
            new_label->cost      = current_label->cost;
            new_label->real_cost = current_label->real_cost;

            // Previous and next jobs (outside of swapped range)
            int prev = current_label->jobs_covered[i - 1];
            int next = current_label->jobs_covered[i + 2];

            // Compute the original cost for the segment before the swap
            double original_cost = getcij(prev, job1) + getcij(job1, job2) + getcij(job2, next);

            // Compute the new cost after swapping job1 and job2
            double new_cost = getcij(prev, job2) + getcij(job2, job1) + getcij(job1, next);

            // Update the label's cost and real cost
            new_label->cost += new_cost - original_cost;
            new_label->real_cost += new_cost - original_cost;

            // Feasibility check: ensure the new route satisfies time constraints
            double elapsed_time = 0.0;
            bool   feasible     = true;

            // Iterate through the jobs to check time windows and durations
            for (size_t k = 0; k < new_label->jobs_covered.size() - 1; ++k) {
                int from = new_label->jobs_covered[k];
                int to   = new_label->jobs_covered[k + 1];

                // Calculate the travel cost (time) between jobs
                double cost_increment = getcij(from, to);
                double new_time       = elapsed_time + cost_increment;

                // Check if the new time violates the end time constraint
                if (new_time > jobs[to].end_time) {
                    feasible = false;
                    break; // No need to continue if the time window is violated
                }

                // Adjust for the job's start time
                if (new_time < jobs[to].start_time) {
                    new_time = jobs[to].start_time; // Move time forward to respect the start time
                }

                // Update elapsed time (including job duration)
                elapsed_time = new_time + jobs[to].duration;
            }

            // If the label is not feasible, release it and continue to the next iteration
            if (!feasible) { continue; }

            // Check if the new label is better (lower cost) and add it to the output queue
            if (new_label->cost < current_label->cost) {
                best_labels_out.push(new_label);
            } else {
                // Release the label if it's not an improvement
            }
        }
    }

    return iter;
}

/* Improvement heuristics based on changing the position of customers */
/* Operation 4: 2-opt exchange operator */
/**
 * @brief Performs a 2-opt local search heuristic on the given priority queues of labels.
 *
 * This function iterates through the input priority queue of labels, performing a 2-opt exchange
 * to generate new labels with potentially lower costs. The new labels are then checked for feasibility
 * and added to the output priority queue if they are better than the current label.
 *
 */
int BucketGraph::RIH4(std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_in,
                      std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_out,
                      int                                                                  max_n_labels) {

    int iter = 0;
    while (!best_labels_in.empty() && iter < max_n_labels) {
        iter++;

        Label *current_label = best_labels_in.top();
        best_labels_in.pop();

        // Remove similar labels based on cost tolerance
        while (!best_labels_in.empty() && best_labels_in.top()->cost < current_label->cost + RCESPP_TOL_ZERO) {
            best_labels_in.pop();
        }

        // Skip labels with only depots covered (at least one job + two depots required)
        if (current_label->jobs_covered.size() <= 3) { continue; }

        // Iterate over pairs of positions for 2-opt exchange
        for (size_t i = 1; i < current_label->jobs_covered.size() - 2; ++i) {
            for (size_t j = i + 1; j < current_label->jobs_covered.size() - 1; ++j) {

                // Acquire a new label for the 2-opt exchange
                Label *new_label = label_pool_fw.acquire();
                if (!new_label) {
                    // Handle label acquisition failure
                    continue;
                }

                new_label->initialize(current_label->vertex, current_label->cost, {current_label->resources[0]},
                                      current_label->job_id);

                new_label->jobs_covered.clear();
                new_label->jobs_covered.reserve(current_label->jobs_covered.size());

                // Copy jobs before i, reverse between i and j, and copy jobs after j
                for (size_t k = 0; k < i; ++k) { new_label->jobs_covered.push_back(current_label->jobs_covered[k]); }
                for (size_t k = j; k >= i; --k) { new_label->jobs_covered.push_back(current_label->jobs_covered[k]); }
                for (size_t k = j + 1; k < current_label->jobs_covered.size(); ++k) {
                    new_label->jobs_covered.push_back(current_label->jobs_covered[k]);
                }

                // Recalculate the entire cost
                double total_cost = 0.0;
                for (size_t k = 0; k < new_label->jobs_covered.size() - 1; ++k) {
                    total_cost += getcij(new_label->jobs_covered[k], new_label->jobs_covered[k + 1]);
                }
                new_label->cost      = total_cost;
                new_label->real_cost = total_cost;

                // Feasibility check: ensure time constraints are satisfied
                double elapsed_time = 0.0;
                bool   feasible     = true;
                for (size_t k = 0; k < new_label->jobs_covered.size() - 1; ++k) {
                    int    from           = new_label->jobs_covered[k];
                    int    to             = new_label->jobs_covered[k + 1];
                    double cost_increment = getcij(from, to);

                    double new_time = elapsed_time + cost_increment;
                    // Check time window constraints
                    if (new_time > jobs[to].end_time || new_time < jobs[to].start_time) {
                        feasible = false;
                        break;
                    }

                    elapsed_time = new_time + jobs[to].duration;
                }

                if (!feasible) { continue; }

                // Add to output queue if better
                if (new_label->cost < current_label->cost) {
                    best_labels_out.push(new_label);
                } else {
                }
            }
        }
    }
    return iter;
}

/** Find the best insertion positions for a customer in a route
 * @brief Finds the best insertion positions for a customer in a route.
 * This function calculates the cost of inserting a customer at each position in the route and returns the top three
 * positions with the lowest cost.
 * @param route     The current route to insert the customer into.
 * @param customer  The customer to insert into the route.
 * @return std::vector<size_t> Returns the top three positions with the lowest cost for inserting the customer.
 */
inline std::vector<size_t> BucketGraph::findBestInsertionPositions(const std::vector<int> &route, int &customer) {
    std::vector<std::pair<double, size_t>> insertion_costs;

    // Calculate the insertion cost for each position in the route
    for (size_t pos = 1; pos < route.size(); ++pos) { // Exclude depots
        double cost = calculateInsertionCost(route, customer, pos);
        insertion_costs.emplace_back(cost, pos);
    }

    // Sort the positions by cost
    std::sort(insertion_costs.begin(), insertion_costs.end());

    // Select the top three positions
    std::vector<size_t> best_positions;
    for (size_t k = 0; k < std::min(3ul, insertion_costs.size()); ++k) {
        best_positions.push_back(insertion_costs[k].second);
    }

    return best_positions;
}

/**
 * @brief Calculate the cost of inserting a customer at a given position in a route
 *
 */
double BucketGraph::calculateInsertionCost(const std::vector<int> &route, int &customer, size_t pos) {
    double cost = 0.0;

    // Calculate the cost of inserting the customer at the given position
    if (pos > 0 && pos < route.size()) {
        int prev = route[pos - 1];
        int next = route[pos];

        cost = getcij(prev, customer) + getcij(customer, next) - getcij(prev, next);
    }

    return cost;
}

/** Swap* operator
 * @brief Refines the input labels by swapping pairs of jobs and checking feasibility.
 * This function processes the input priority queue of labels, attempts to improve each label by swapping pairs of jobs
 * and recalculating the cost. If the new label is feasible and has a lower cost, it is added to the output priority
 * queue.
 */

int BucketGraph::RIH5(std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_in,
                      std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_out,
                      int                                                                  max_n_labels) {

    int iter = 0;
    while (!best_labels_in.empty() && iter < max_n_labels) {
        iter++;

        Label *current_label = best_labels_in.top();
        best_labels_in.pop();

        // Remove similar labels based on cost tolerance
        while (!best_labels_in.empty() && best_labels_in.top()->cost < current_label->cost + RCESPP_TOL_ZERO) {
            best_labels_in.pop();
        }

        // Skip labels with only depots covered (at least one job + two depots required)
        if (current_label->jobs_covered.size() <= 3) { continue; }

        // Iterate over all pairs of jobs for the Swap* operation
        for (size_t i = 1; i < current_label->jobs_covered.size() - 2; ++i) {
            for (size_t j = i + 1; j < current_label->jobs_covered.size() - 1; ++j) {

                int v       = current_label->jobs_covered[i]; // Job index i
                int v_prime = current_label->jobs_covered[j]; // Job index j

                // Step 1: Find best insertion positions for v and v_prime
                std::vector<size_t> best_insertion_pos_v_prime =
                    findBestInsertionPositions(current_label->jobs_covered, v_prime);
                std::vector<size_t> best_insertion_pos_v = findBestInsertionPositions(current_label->jobs_covered, v);

                // Step 2: Perform Swap* and evaluate the move
                for (size_t pos_v_prime : best_insertion_pos_v_prime) {
                    for (size_t pos_v : best_insertion_pos_v) {
                        // Acquire a new label for the Swap* operation
                        Label *new_label = label_pool_fw.acquire();
                        if (!new_label) {
                            // Handle label acquisition failure
                            continue;
                        }

                        new_label->initialize(current_label->vertex, current_label->cost, {current_label->resources[0]},
                                              current_label->job_id);

                        new_label->jobs_covered.clear();
                        new_label->jobs_covered.reserve(current_label->jobs_covered.size());

                        // Step 3: Swap and insert at the best positions
                        performSwap(new_label->jobs_covered, current_label->jobs_covered, i, j, pos_v, pos_v_prime);

                        // Step 4: Recalculate the entire cost
                        double total_cost = 0.0;
                        for (size_t k = 0; k < new_label->jobs_covered.size() - 1; ++k) {
                            total_cost += getcij(new_label->jobs_covered[k], new_label->jobs_covered[k + 1]);
                        }
                        new_label->cost      = total_cost;
                        new_label->real_cost = total_cost;

                        // Step 5: Feasibility check: ensure time constraints are satisfied
                        double elapsed_time = 0.0;
                        bool   feasible     = true;
                        for (size_t k = 0; k < new_label->jobs_covered.size() - 1; ++k) {
                            int    from           = new_label->jobs_covered[k];
                            int    to             = new_label->jobs_covered[k + 1];
                            double cost_increment = getcij(from, to);

                            double new_time = elapsed_time + cost_increment;
                            // Check time window constraints
                            if (new_time > jobs[to].end_time || new_time < jobs[to].start_time) {
                                feasible = false;
                                break;
                            }

                            elapsed_time = new_time + jobs[to].duration;
                        }

                        // Add to the output queue if feasible and better
                        if (feasible && new_label->cost < current_label->cost) {
                            best_labels_out.push(new_label);
                        } else {
                        }
                    }
                }
            }
        }
    }
    return iter;
}

void BucketGraph::performSwap(std::vector<int> &new_route, const std::vector<int> &current_route, size_t pos_i,
                              size_t pos_j, size_t best_pos_v, size_t best_pos_v_prime) {
    // Copy the jobs before the first swap position
    new_route.insert(new_route.end(), current_route.begin(), current_route.begin() + pos_i);

    // Insert job at position j (v_prime) at the best position in the new route
    new_route.insert(new_route.begin() + best_pos_v, current_route[pos_j]);

    // Insert job at position i (v) at the best position in the new route
    new_route.insert(new_route.begin() + best_pos_v_prime, current_route[pos_i]);

    // Copy the remaining jobs after the second swap position
    new_route.insert(new_route.end(), current_route.begin() + pos_j + 1, current_route.end());
}
