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
 * - RIH4: Applies the 2-opt local search heuristic, performing exchanges between non-neighboring customers to improve route costs.
 *
 * The functions make use of priority queues to manage the best labels (routes) and process them iteratively based on their costs.
 * The objective is to generate feasible solutions that minimize route costs while respecting problem-specific constraints.
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

#include "../include/Definitions.h"
#include "../include/BucketGraph.h"

/* Improvement heuristics based on the insertion/deletion of customers */
/* Operation 1: modify the routes by adding one customer to each route */
/**
 * @brief Refines the best labels by inserting new customers into the routes.
 *
 * This function processes the best labels from the input priority queue, attempts to insert new customers
 * into the routes, and pushes the refined labels into the output priority queue if they are feasible and
 * have a lower cost than the current label.
 *
 * @param best_labels_in  A priority queue containing the best labels to be refined.
 * @param best_labels_out A priority queue to store the refined labels.
 * @param max_n_labels    The maximum number of labels to process.
 * @return int            Returns 1 upon completion.
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
                new_label->initialize(current_label->vertex, current_label->cost, {current_label->resources[0]},
                                      current_label->job_id);

                // Copy jobs covered up to the insertion point
                new_label->jobs_covered.assign(current_label->jobs_covered.begin(),
                                               current_label->jobs_covered.begin() + i);
                // Insert the new customer
                new_label->jobs_covered.push_back(new_customer);
                // Copy the remaining jobs covered
                new_label->jobs_covered.insert(
                    new_label->jobs_covered.end(), current_label->jobs_covered.begin() + i,
                    current_label->jobs_covered.end()); // Recalculate the cost and resources for the new label
                new_label->cost      = 0.0;
                new_label->real_cost = 0.0;
                // new_label->resources.clear();
                // new_label->resources.push_back(0.0); // Initialize resources[0] as time
                bool   feasible     = true;
                double elapsed_time = 0.0;

                for (size_t j = 0; j < new_label->jobs_covered.size() - 1; ++j) {
                    int  from           = new_label->jobs_covered[j];
                    int  to             = new_label->jobs_covered[j + 1];
                    auto cost_increment = getcij(to, from);

                    // Check for feasibility based on problem-specific constraints
                    double new_time = elapsed_time + cost_increment;
                    if (new_time > jobs[to].end_time) {
                        feasible = false;
                        break;
                    }
                    if (new_time < jobs[to].start_time) { new_time = jobs[to].start_time; }
                    elapsed_time = new_time + jobs[to].duration;

                    new_label->cost += cost_increment - jobs[to].cost;
                    new_label->real_cost += cost_increment;
                }

                // Check the feasibility of the route beyond the newly inserted customer
                if (feasible) {
                    int last_customer = new_label->jobs_covered.back();
                    while (last_customer != N_SIZE - 1) { // Continue until the last depot
                        int    next_customer = new_label->jobs_covered[last_customer + 1];
                        double new_time      = elapsed_time + getcij(last_customer, next_customer);
                        if (new_time > jobs[next_customer].end_time) {
                            feasible = false;
                            break;
                        }
                        if (new_time < jobs[next_customer].start_time) { new_time = jobs[next_customer].start_time; }
                        elapsed_time  = new_time + jobs[next_customer].duration;
                        last_customer = next_customer;
                    }
                } else {
                    // label_pool_fw.release(new_label);
                    continue;
                }
                if (new_label->cost < current_label->cost) {
                    // std::print("RIH1: new_label->cost: {}, current_label->cost: {}\n", new_label->cost,
                    //            current_label->cost);
                    best_labels_out.push(new_label);
                } else {
                    // label_pool_fw.release(new_label);
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
 * @param best_labels_in  A priority queue of labels to be processed.
 * @param best_labels_out A priority queue to store the improved labels.
 * @param max_n_labels    The maximum number of labels to process.
 * @return int            Returns 1 upon completion.
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
        if (current_label->jobs_covered.size() <= 3) { continue; }

        // Iterate over each customer in the route, skipping the first and last (which are depots)
        for (size_t i = 1; i < current_label->jobs_covered.size() - 1; ++i) {

            // Reuse the current label to avoid full reinitialization
            Label *new_label = label_pool_fw.acquire();
            new_label->initialize(current_label->vertex, current_label->cost, {current_label->resources[0]},
                                  current_label->job_id);

            // Copy jobs covered except the i-th job (the customer to delete)
            new_label->jobs_covered.clear();
            new_label->jobs_covered.reserve(current_label->jobs_covered.size() -
                                            1); // Reserve space to avoid reallocations
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

                new_label->cost += new_cost - original_cost;      // Adjust the cost
                new_label->real_cost += new_cost - original_cost; // Adjust the real cost

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

                if (!feasible) {
                    // label_pool_fw.release(new_label);
                    continue;
                }
            }

            // If the new label's cost is better, push it to the output queue
            if (new_label->cost < current_label->cost) {
                best_labels_out.push(new_label);
            } else {
                // label_pool_fw.release(new_label); // Release unused labels to avoid memory leaks
            }
        }
    }

    return 1;
}

/* Improvement heuristics based on changing the position of customers */
/* Operation 3: swap operator */
/**2(std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_in,
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
        if (current_label->jobs_covered.size() <= 3) { continue; }

        // Iterate over each customer in the route, skipping the first and last (which are depots)
        for (size_t i = 1; i < current_label->jobs_covered.size() - 1; ++i) {

            // Reuse the current label to avoid full reinitialization
            Label *new_label = label_pool_fw.acquire();
            new_label->initialize(current_label->vertex, current_label->cost, {current_label->resources[0]},
                                  current_label->job_id);

            // Copy jobs covered except the i-th job (the customer to delete)
            new_label->jobs_covered.clear();
            new_label->jobs_covered.reserve(current_label->jobs_covered.size() -
                                            1); // Reserve space to avoid reallocations
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

                new_label->cost += new_cost - original_cost;      // Adjust the cost
                new_label->real_cost += new_cost - original_cost; // Adjust the real cost

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

                if (!feasible) {
                    // label_pool_fw.release(new_label);
                    continue;
                }
            }

            // If the new label's cost is better, push it to the output queue
            if (new_label->cost < current_label->cost) {
                best_labels_out.push(new_label);
            } else {
                // label_pool_fw.release(new_label); // Release unused labels to avoid memory leaks
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
 * @param best_labels_in  A priority queue of labels to be processed.
 * @param best_labels_out A priority queue where improved labels are stored.
 * @param max_n_labels    The maximum number of labels to process.
 * @return int            Returns 1 upon completion.
 */
int BucketGraph::RIH3(std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_in,
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

        // If the label has only depots covered, skip it (at least one job + two depots required)
        if (current_label->jobs_covered.size() <= 3) { continue; }

        // Iterate over jobs, skipping the first and last (which are depots)
        for (size_t i = 1; i < current_label->jobs_covered.size() - 2; ++i) {
            int job1 = current_label->jobs_covered[i];
            int job2 = current_label->jobs_covered[i + 1];

            // Skip swapping if the jobs are the same (unnecessary)
            if (job1 == job2) { continue; }

            Label *new_label = label_pool_fw.acquire();
            new_label->initialize(current_label->vertex, current_label->cost, {current_label->resources[0]},
                                  current_label->job_id);

            // Copy jobs covered and swap neighboring jobs i and i+1
            new_label->jobs_covered = current_label->jobs_covered;
            std::swap(new_label->jobs_covered[i], new_label->jobs_covered[i + 1]);

            // Only recalculate the affected cost for the swapped part of the route
            new_label->cost      = current_label->cost;
            new_label->real_cost = current_label->real_cost;

            // Recalculate the cost for just the swapped part (before and after the swap)
            int prev = current_label->jobs_covered[i - 1];
            int next = current_label->jobs_covered[i + 2];

            double original_cost = getcij(prev, job1) + getcij(job1, job2) + getcij(job2, next);
            double new_cost      = getcij(prev, job2) + getcij(job2, job1) + getcij(job1, next);

            new_label->cost += new_cost - original_cost; // Adjust the cost
            new_label->real_cost += new_cost - original_cost;

            // Check for feasibility: ensure the new route is within time constraints
            double elapsed_time = 0.0;
            bool   feasible     = true;
            for (size_t k = 0; k < new_label->jobs_covered.size() - 1; ++k) {
                int    from           = new_label->jobs_covered[k];
                int    to             = new_label->jobs_covered[k + 1];
                double cost_increment = getcij(from, to);

                double new_time = elapsed_time + cost_increment;
                if (new_time > jobs[to].end_time) {
                    feasible = false; // Infeasible if exceeding time windows
                    break;
                }
                if (new_time < jobs[to].start_time) { new_time = jobs[to].start_time; }
                elapsed_time = new_time + jobs[to].duration;
            }

            if (!feasible) {
                // label_pool_fw.release(new_label); // Release label if route is not feasible
                continue;
            }

            // Check if the new label is better and add it to the output queue
            if (new_label->cost < current_label->cost) {
                best_labels_out.push(new_label);
            } else {
                // label_pool_fw.release(new_label); // Release unused label
            }
        }
    }

    return 1;
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
 * @param best_labels_in  A priority queue of labels to be processed.
 * @param best_labels_out A priority queue to store the resulting labels after processing.
 * @param max_n_labels    The maximum number of labels to process.
 * @return int            Returns 1 upon completion.
 */
int BucketGraph::RIH4(std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_in,
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

        // If the label has only depots covered, skip it
        if (current_label->jobs_covered.size() <= 3) { continue; }

        // Iterate over pairs of positions to perform 2-opt exchange
        for (size_t i = 1; i < current_label->jobs_covered.size() - 2; ++i) {
            for (size_t j = i + 1; j < current_label->jobs_covered.size() - 1; ++j) {

                // Avoid creating a new label if i and j result in the same cost
                if (i == j) { continue; }

                // Reuse label instead of creating a new one every time
                Label *new_label = label_pool_fw.acquire();
                new_label->initialize(current_label->vertex, current_label->cost, {current_label->resources[0]},
                                      current_label->job_id);

                // Perform 2-opt exchange: reverse the segment between i and j
                new_label->jobs_covered.clear();
                new_label->jobs_covered.reserve(current_label->jobs_covered.size()); // Reserve to avoid reallocations

                // Copy jobs before i
                for (size_t k = 0; k < i; ++k) { new_label->jobs_covered.push_back(current_label->jobs_covered[k]); }

                // Reverse jobs between i and j
                for (size_t k = j; k >= i; --k) { new_label->jobs_covered.push_back(current_label->jobs_covered[k]); }

                // Copy jobs after j
                for (size_t k = j + 1; k < current_label->jobs_covered.size(); ++k) {
                    new_label->jobs_covered.push_back(current_label->jobs_covered[k]);
                }

                // Only recalculate the cost for the changed segments (between i-1 and j+1)
                int prev = current_label->jobs_covered[i - 1]; // Before the 2-opt segment
                int next = current_label->jobs_covered[j + 1]; // After the 2-opt segment

                double original_cost =
                    getcij(prev, current_label->jobs_covered[i]) + getcij(current_label->jobs_covered[j], next);
                double new_cost = getcij(prev, new_label->jobs_covered[i]) + getcij(new_label->jobs_covered[j], next);

                new_label->cost += new_cost - original_cost; // Adjust the cost
                new_label->real_cost += new_cost - original_cost;

                // Feasibility check
                double elapsed_time = 0.0;
                bool   feasible     = true;
                for (size_t k = 0; k < new_label->jobs_covered.size() - 1; ++k) {
                    int    from           = new_label->jobs_covered[k];
                    int    to             = new_label->jobs_covered[k + 1];
                    double cost_increment = getcij(from, to);

                    double new_time = elapsed_time + cost_increment;
                    if (new_time > jobs[to].end_time) {
                        feasible = false;
                        break;
                    }
                    if (new_time < jobs[to].start_time) { new_time = jobs[to].start_time; }
                    elapsed_time = new_time + jobs[to].duration;
                }

                if (!feasible) {
                    // label_pool_fw.release(new_label); // Release the label if not feasible
                    continue;
                }

                // Check if the new label is better and add it to the output queue
                if (new_label->cost < current_label->cost) {
                    best_labels_out.push(new_label);
                } else {
                    // label_pool_fw.release(new_label); // Release the label if it's not better
                }
            }
        }
    }

    return 1;
}
