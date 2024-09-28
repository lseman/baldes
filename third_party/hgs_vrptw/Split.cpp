#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "Individual.h"
#include "Params.h"
#include "Split.h"

void Split::generalSplit(Individual *indiv, int nbMaxVehicles) {
    // Do not apply Split with fewer vehicles than the trivial (LP) bin packing bound
    maxVehicles = std::max(nbMaxVehicles, static_cast<int>(std::ceil(params->totalDemand / params->vehicleCapacity)));

    // Initialization of the data structures for the Linear Split algorithm
    // Direct application of the code located at https://github.com/vidalt/Split-Library
    // Loop over all clients, excluding the depot
    for (int i = 1; i <= params->nbClients; i++) {
        // Store all information on clientSplits (use chromT[i-1] since the depot is not included in chromT)
        cliSplit[i].demand      = params->cli[indiv->chromT[i - 1]].demand;
        cliSplit[i].serviceTime = params->cli[indiv->chromT[i - 1]].serviceDuration;
        cliSplit[i].d0_x        = params->timeCost.get(0, indiv->chromT[i - 1]);
        cliSplit[i].dx_0        = params->timeCost.get(indiv->chromT[i - 1], 0);

        // The distance to the next client is INT_MIN for the last client
        if (i < params->nbClients) {
            cliSplit[i].dnext = params->timeCost.get(indiv->chromT[i - 1], indiv->chromT[i]);
        } else {
            cliSplit[i].dnext = INT_MIN;
        }

        // Store cumulative data on the demand, service time, and distance
        sumLoad[i]     = sumLoad[i - 1] + cliSplit[i].demand;
        sumService[i]  = sumService[i - 1] + cliSplit[i].serviceTime;
        sumDistance[i] = sumDistance[i - 1] + cliSplit[i - 1].dnext;
    }

    // We first try the Simple Split, and then the Split with Limited Fleet if this is not successful
    if (splitSimple(indiv) == 0) { splitLF(indiv); }

    // Build up the rest of the Individual structure
    indiv->evaluateCompleteCost();
}

int Split::splitSimple(Individual *indiv) {
    // Reinitialize the potential structure
    potential[0][0] = 0;
    for (int i = 1; i <= params->nbClients; i++) { potential[0][i] = 1.e30; }

    // MAIN SIMPLE SPLIT ALGORITHM -- Simple Split using Bellman's algorithm in topological order
    // This code has been maintained as it is very simple and can be easily adapted to a variety of constraints,
    // whereas the O(n) Split has a more restricted application scope
    if (params->isDurationConstraint) {
        // Loop over all clients (excluding the depot). This runs in O(nB).
        for (int i = 0; i < params->nbClients; i++) {
            int    load            = 0;
            int    distance        = 0;
            int    serviceDuration = 0;
            double cumulativeCost  = potential[0][i];

            // Loop over the next clients, as long as the total load is smaller than 1.5 * vehicleCapacity
            for (int j = i + 1; j <= params->nbClients; j++) {
                load += cliSplit[j].demand;
                if (load > 1.5 * params->vehicleCapacity) {
                    break; // Exit early if load exceeds the constraint
                }

                serviceDuration += cliSplit[j].serviceTime;

                // Update distance incrementally
                distance += (j == i + 1) ? cliSplit[j].d0_x : cliSplit[j - 1].dnext;

                // Calculate cost including penalties
                double cost =
                    distance + cliSplit[j].dx_0 + params->penaltyCapacity * std::max(load - params->vehicleCapacity, 0);

                // Update potential and predecessor if this new cost is better
                if (cumulativeCost + cost < potential[0][j]) {
                    potential[0][j] = cumulativeCost + cost;
                    pred[0][j]      = i;
                }
            }
        }
    } else {
        Trivial_Deque queue = Trivial_Deque(params->nbClients + 1, 0);

        for (int i = 1; i <= params->nbClients; i++) {
            int front       = queue.get_front();
            potential[0][i] = propagate(front, i, 0);
            pred[0][i]      = front;

            if (i < params->nbClients) {
                int back = queue.get_back();

                // Remove dominated clients
                if (!dominates(back, i, 0)) {
                    while (queue.size() > 0 && dominatesRight(queue.get_back(), i, 0)) { queue.pop_back(); }
                    queue.push_back(i);
                }

                // Remove front if dominated by next front
                while (queue.size() > 1 && propagate(queue.get_front(), i + 1, 0) >
                                               propagate(queue.get_next_front(), i + 1, 0) - MY_EPSILON) {
                    queue.pop_front();
                }
            }
        }
    }

    // Check if the cost of the last client is still very large. In that case, the Split algorithm did not reach the
    // last client
    if (potential[0][params->nbClients] > 1.e29) {
        throw std::string("ERROR : no Split solution has been propagated until the last node");
    }

    // Filling the chromR structure
    // First clear some chromR vectors. In practice, maxVehicles equals nbVehicles. Then, this loop is not needed and
    // the next loop starts at the last index of chromR
    for (int k = params->nbVehicles - 1; k >= maxVehicles; k--) { indiv->chromR[k].clear(); }

    // Loop over all vehicles, clear the route, get the predecessor and create a new chromR for that route
    int end = params->nbClients;
    for (int k = maxVehicles - 1; k >= 0; k--) {
        // Clear the corresponding chromR
        indiv->chromR[k].clear();

        // Loop from the begin to the end of the route corresponding to this vehicle
        int begin = pred[0][end];
        for (int ii = begin; ii < end; ii++) { indiv->chromR[k].push_back(indiv->chromT[ii]); }
        end = begin;
    }

    // Return OK in case the Split algorithm reached the beginning of the routes
    return (end == 0);
}

// Split for problems with limited fleet
int Split::splitLF(Individual *indiv) {
    // Initialize the potential structures
    potential[0][0] = 0;
    for (int k = 0; k <= maxVehicles; k++)
        for (int i = 1; i <= params->nbClients; i++) potential[k][i] = 1.e30;

    // MAIN ALGORITHM -- Simple Split using Bellman's algorithm in topological order
    // This code has been maintained as it is very simple and can be easily adapted to a variety of constraints, whereas
    // the O(n) Split has a more restricted application scope
    if (params->isDurationConstraint) {
        double maxLoad = 1.5 * params->vehicleCapacity; // Precompute to avoid repeated calculations

        for (int k = 0; k < maxVehicles; k++) {
            for (int i = k; i < params->nbClients && potential[k][i] < 1.e29; i++) {
                int    load            = 0;
                int    serviceDuration = 0;
                int    distance        = 0;
                double cumulativeCost  = potential[k][i]; // Cache the current potential for i

                for (int j = i + 1; j <= params->nbClients && load <= maxLoad; j++) {
                    // Incremental updates for load, service duration, and distance
                    load += cliSplit[j].demand;
                    if (load > maxLoad) {
                        break; // Early exit if load infeasibility is reached
                    }

                    serviceDuration += cliSplit[j].serviceTime;

                    // Increment distance based on whether it's the first or subsequent client
                    if (j == i + 1) {
                        distance += cliSplit[j].d0_x;
                    } else {
                        distance += cliSplit[j - 1].dnext;
                    }

                    // Calculate the cost including the penalty for exceeding vehicle capacity
                    double cost = distance + cliSplit[j].dx_0 +
                                  params->penaltyCapacity * std::max(load - params->vehicleCapacity, 0);

                    // Update potential and predecessor if a lower cost is found
                    if (cumulativeCost + cost < potential[k + 1][j]) {
                        potential[k + 1][j] = cumulativeCost + cost;
                        pred[k + 1][j]      = i;
                    }
                }
            }
        }
    } else // MAIN ALGORITHM -- Without duration constraints in O(n), from "Vidal, T. (2016). Split algorithm in O(n)
           // for the capacitated vehicle routing problem. C&OR"
    {
        Trivial_Deque queue = Trivial_Deque(params->nbClients + 1, 0);

        for (int k = 0; k < maxVehicles; k++) {
            // Reset the queue to start with k
            queue.reset(k);

            // Loop through clients, process until the queue is empty
            for (int i = k + 1; i <= params->nbClients && queue.size() > 0; i++) {
                // Cache the front of the queue
                int front = queue.get_front();

                // The front is the best predecessor for i
                potential[k + 1][i] = propagate(front, i, k);
                pred[k + 1][i]      = front;

                if (i < params->nbClients) {
                    // If i is not dominated by the back of the queue
                    int back = queue.get_back();
                    if (!dominates(back, i, k)) {
                        // Remove elements dominated by i from the back
                        while (queue.size() > 0 && dominatesRight(back, i, k)) {
                            queue.pop_back();
                            back = queue.get_back(); // Update back for the next check
                        }
                        queue.push_back(i);
                    }

                    // Cache the next front propagation value for i+1
                    double nextFrontPropagate = propagate(queue.get_front(), i + 1, k);

                    // Check if the current front is dominated by the next front
                    while (queue.size() > 1 &&
                           nextFrontPropagate > propagate(queue.get_next_front(), i + 1, k) - MY_EPSILON) {
                        queue.pop_front();
                        nextFrontPropagate = propagate(queue.get_front(), i + 1, k); // Update nextFrontPropagate
                    }
                }
            }
        }
    }

    if (potential[maxVehicles][params->nbClients] > 1.e29)
        throw std::string("ERROR : no Split solution has been propagated until the last node");

    // It could be cheaper to use a smaller number of vehicles
    double minCost  = potential[maxVehicles][params->nbClients];
    int    nbRoutes = maxVehicles;

    for (int k = 1; k < maxVehicles; k++) {
        double potentialCost = potential[k][params->nbClients]; // Cache potential[k][params->nbClients]
        if (potentialCost < minCost) {
            minCost  = potentialCost;
            nbRoutes = k;
        }
    }

    // Clear routes for excess vehicles
    for (int k = params->nbVehicles - 1; k >= nbRoutes; k--) {
        indiv->chromR[k].clear(); // Efficiently clear the route
    }

    int end = params->nbClients;
    for (int k = nbRoutes - 1; k >= 0; k--) {
        indiv->chromR[k].clear();
        int begin = pred[k + 1][end];

        // Fill chromR[k] from chromT[begin] to chromT[end - 1]
        indiv->chromR[k].reserve(end - begin); // Reserve space to avoid reallocations
        for (int ii = begin; ii < end; ii++) { indiv->chromR[k].push_back(indiv->chromT[ii]); }

        end = begin;
    }

    // Return OK in case the Split algorithm reached the beginning of the routes
    return (end == 0);
}

Split::Split(Params *params) : params(params) {
    // Initialize structures for the Linear Split
    cliSplit    = std::vector<ClientSplit>(params->nbClients + 1);
    sumDistance = std::vector<int>(params->nbClients + 1, 0);
    sumLoad     = std::vector<int>(params->nbClients + 1, 0);
    sumService  = std::vector<int>(params->nbClients + 1, 0);
    potential =
        std::vector<std::vector<double>>(params->nbVehicles + 1, std::vector<double>(params->nbClients + 1, 1.e30));
    pred = std::vector<std::vector<int>>(params->nbVehicles + 1, std::vector<int>(params->nbClients + 1, 0));
}
