#include "Individual.h"

#include <algorithm>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "Definitions.h"
#include "Params.h"
#include "fmt/core.h"

void Individual::evaluateCompleteCost() {
    myCostSol = CostSol();

    // Precompute timeCost.get results for all customer pairs in each route
    std::vector<std::vector<int>> timeCostCache(params->nbVehicles);
    for (int r = 0; r < params->nbVehicles; ++r) {
        if (chromR[r].empty()) continue;
        timeCostCache[r].resize(chromR[r].size());
        timeCostCache[r][0] = params->timeCost.get(0, chromR[r][0]);
        for (int i = 1; i < static_cast<int>(chromR[r].size()); ++i) {
            timeCostCache[r][i] =
                params->timeCost.get(chromR[r][i - 1], chromR[r][i]);
        }
    }

    // Evaluate each route
    for (int r = 0; r < params->nbVehicles; ++r) {
        if (chromR[r].empty()) continue;

        // Find the latest release time in the route
        int latestReleaseTime = params->cli[chromR[r][0]].releaseTime;
        for (int i = 1; i < static_cast<int>(chromR[r].size()); ++i) {
            latestReleaseTime = std::max(latestReleaseTime,
                                         params->cli[chromR[r][i]].releaseTime);
        }

        // Initialize route metrics
        int distance = timeCostCache[r][0];
        int load = params->cli[chromR[r][0]].demand;
        int service = params->cli[chromR[r][0]].serviceDuration;
        int time = latestReleaseTime + distance;
        int waitTime = 0, timeWarp = 0;

        // Handle time windows for the first customer
        if (params->cli[chromR[r][0]].n_tw > 0) {
            auto &firstTimeWindows = params->cli[chromR[r][0]].timeWindows;
            time =
                adjustToTimeWindow(time, firstTimeWindows, waitTime, timeWarp);
        } else {
            // Single time window logic
            if (time < params->cli[chromR[r][0]].earliestArrival) {
                time = params->cli[chromR[r][0]].earliestArrival;
            } else if (time > params->cli[chromR[r][0]].latestArrival) {
                timeWarp += time - params->cli[chromR[r][0]].latestArrival;
                time = params->cli[chromR[r][0]].latestArrival;
            }
        }

        predecessors[chromR[r][0]] = 0;

        // Evaluate the rest of the route
        for (int i = 1; i < static_cast<int>(chromR[r].size()); ++i) {
            int prev = chromR[r][i - 1], curr = chromR[r][i];
            distance += timeCostCache[r][i];
            load += params->cli[curr].demand;
            service += params->cli[curr].serviceDuration;
            time += params->cli[prev].serviceDuration + timeCostCache[r][i];

            // Handle time windows for the current customer
            if (params->cli[curr].n_tw > 0) {
                auto &currTimeWindows = params->cli[curr].timeWindows;
                time = adjustToTimeWindow(time, currTimeWindows, waitTime,
                                          timeWarp);
            } else {
                // Single time window logic
                if (time < params->cli[curr].earliestArrival) {
                    waitTime += params->cli[curr].earliestArrival - time;
                    time = params->cli[curr].earliestArrival;
                } else if (time > params->cli[curr].latestArrival) {
                    timeWarp += time - params->cli[curr].latestArrival;
                    time = params->cli[curr].latestArrival;
                }
            }

            predecessors[curr] = prev;
            successors[prev] = curr;
        }

        // Finalize the route
        successors[chromR[r].back()] = 0;
        distance += params->timeCost.get(chromR[r].back(), 0);
        time += params->cli[chromR[r].back()].serviceDuration +
                params->timeCost.get(chromR[r].back(), 0);

        // Handle depot time window
        if (time > params->cli[0].latestArrival) {
            timeWarp += time - params->cli[0].latestArrival;
            time = params->cli[0].latestArrival;
        }

        // Update solution metrics
        myCostSol.distance += distance;
        if (params->problemType == ProblemType::vrptw ||
            params->problemType == ProblemType::evrp) {
            myCostSol.timeWarp += timeWarp;
            myCostSol.waitTime += waitTime;
        } else {
            myCostSol.timeWarp = 0;
            myCostSol.waitTime = 0;
        }
        myCostSol.nbRoutes++;

        if (load > params->vehicleCapacity) {
            myCostSol.capacityExcess += load - params->vehicleCapacity;
        }
    }

    // Calculate penalized cost
    myCostSol.penalizedCost =
        myCostSol.distance +
        myCostSol.capacityExcess * params->penaltyCapacity +
        myCostSol.timeWarp * params->penaltyTimeWarp +
        myCostSol.waitTime * params->penaltyWaitTime;

    // Check feasibility
    isFeasible = (myCostSol.capacityExcess < MY_EPSILON &&
                  myCostSol.timeWarp < MY_EPSILON);
}

// Helper function to adjust time based on multiple time windows
int Individual::adjustToTimeWindow(int time,
                                   const std::vector<TimeWindow> &timeWindows,
                                   int &waitTime, int &timeWarp) {
    for (const auto &window : timeWindows) {
        if (time < window.earliestArrival) {
            waitTime += window.earliestArrival - time;
            return window.earliestArrival;
        } else if (time <= window.latestArrival) {
            return time;
        }
    }
    // If no valid time window, apply time warp to the latest one
    timeWarp += time - timeWindows.back().latestArrival;
    return timeWindows.back().latestArrival;
}

void Individual::shuffleChromT() {
    // Initialize the chromT with values from 1 to nbClients
    for (int i = 0; i < params->nbClients; i++) {
        chromT[i] = i + 1;
    }
    // Do a random shuffle chromT from begin to end
    std::shuffle(chromT.begin(), chromT.end(), params->rng);
}

void Individual::removeProximity(Individual *indiv) {
    // Find indiv in indivsPerProximity
    auto it = std::find_if(
        indivsPerProximity.begin(), indivsPerProximity.end(),
        [indiv](const auto &pair) { return pair.second == indiv; });
    // Remove if found
    if (it != indivsPerProximity.end()) {
        indivsPerProximity.erase(it);
    }
}

double Individual::brokenPairsDistance(Individual *indiv2) {
    // Initialize the difference to zero. Then loop over all clients of this
    // individual
    int differences = 0;
    for (int j = 1; j <= params->nbClients; j++) {
        // Increase the difference if the successor of j in this individual is
        // not directly linked to j in indiv2
        if (successors[j] != indiv2->successors[j] &&
            successors[j] != indiv2->predecessors[j]) {
            differences++;
        }
        // Last loop covers all but the first arc. Increase the difference if
        // the predecessor of j in this individual is not directly linked to j
        // in indiv2
        if (predecessors[j] == 0 && indiv2->predecessors[j] != 0 &&
            indiv2->successors[j] != 0) {
            differences++;
        }
    }
    return static_cast<double>(differences) / params->nbClients;
}

double Individual::averageBrokenPairsDistanceClosest(int nbClosest) {
    double result = 0;
    int maxSize =
        std::min(nbClosest, static_cast<int>(indivsPerProximity.size()));
    auto it = indivsPerProximity.begin();
    for (int i = 0; i < maxSize; i++) {
        result += it->first;
        ++it;
    }
    return result / maxSize;
}

Individual::Individual(Params *params, bool initializeChromTAndShuffle)
    : params(params), isFeasible(false), biasedFitness(0) {
    successors = std::vector<int>(params->nbClients + 1);
    predecessors = std::vector<int>(params->nbClients + 1);
    chromR = std::vector<std::vector<int>>(params->nbVehicles);
    chromT = std::vector<int>(params->nbClients);
    if (initializeChromTAndShuffle) {
        shuffleChromT();
    }
}

Individual::Individual(Params *params, std::string solutionStr)
    : params(params), isFeasible(false), biasedFitness(0) {
    successors = std::vector<int>(params->nbClients + 1);
    predecessors = std::vector<int>(params->nbClients + 1);
    chromR = std::vector<std::vector<int>>(params->nbVehicles);
    chromT = std::vector<int>(params->nbClients);

    std::stringstream ss(solutionStr);
    int inputCustomer;
    // Loops as long as there is an integer to read
    int pos = 0;
    int route = 0;
    while (ss >> inputCustomer) {
        if (inputCustomer == 0) {
            // Depot
            route++;
        } else {
            chromR[route].push_back(inputCustomer);
            chromT[pos] = inputCustomer;
            pos++;
        }
    }
    assert(pos == params->nbClients);
    evaluateCompleteCost();
}

Individual::Individual(Params *params, bool rcws,
                       std::vector<std::vector<int>> *pattern) {
    this->params = params;
    int nbClients = params->nbClients;
    int nbVehicles = params->nbVehicles;
    // Preallocate and initialize chromosome vectors.
    successors = std::vector<int>(nbClients + 1);
    predecessors = std::vector<int>(nbClients + 1);
    chromR.resize(nbVehicles);
    chromT = std::vector<int>(nbClients);

    if (rcws) {
        // Helper vectors and mapping: customerRoute maps customer -> route
        // index.
        std::vector<bool> inRoute(nbClients + 1, false);
        std::vector<bool> interior(nbClients + 1, false);
        std::vector<double> load(nbVehicles, 0.0);
        std::vector<int> customerRoute(nbClients + 1, -1);

        int nextEmptyRoute = 0;
        // If a pattern is provided, insert its routes.
        if (pattern) {
            for (size_t r = 0; r < pattern->size(); r++) {
                // Expand vehicle list if needed.
                if (nextEmptyRoute == nbVehicles) {
                    chromR.push_back(std::vector<int>());
                    load.push_back(0.0);
                    nbVehicles++;
                    params->nbVehicles = nbVehicles;
                }
                for (size_t c = 0; c < (*pattern)[r].size(); c++) {
                    int customer = (*pattern)[r][c];
                    chromR[nextEmptyRoute].push_back(customer);
                    load[nextEmptyRoute] += params->cli[customer].demand;
                    inRoute[customer] = true;
                    customerRoute[customer] = nextEmptyRoute;
                    // Mark as interior if not the first or last customer.
                    if (c > 0 && c < (*pattern)[r].size() - 1)
                        interior[customer] = true;
                }
                // Advance to the next empty route.
                while (nextEmptyRoute < nbVehicles &&
                       !chromR[nextEmptyRoute].empty())
                    nextEmptyRoute++;
            }
        }

        // Process the savings list to merge routes or create new ones.
        for (size_t sIdx = 0; sIdx < params->savingsList.size(); sIdx++) {
            Savings s = params->savingsList[sIdx];
            if (s.value <= 0) continue;  // Skip invalid savings

            // Only proceed if combined demand is feasible.
            if (params->cli[s.c1].demand + params->cli[s.c2].demand >
                params->vehicleCapacity)
                continue;

            // Case 1: Neither customer is yet in a route.
            if (!inRoute[s.c1] && !inRoute[s.c2]) {
                if (nextEmptyRoute == nbVehicles) {
                    chromR.push_back(std::vector<int>());
                    load.push_back(0.0);
                    nbVehicles++;
                    params->nbVehicles = nbVehicles;
                }
                chromR[nextEmptyRoute].push_back(s.c1);
                chromR[nextEmptyRoute].push_back(s.c2);
                load[nextEmptyRoute] +=
                    params->cli[s.c1].demand + params->cli[s.c2].demand;
                inRoute[s.c1] = inRoute[s.c2] = true;
                customerRoute[s.c1] = customerRoute[s.c2] = nextEmptyRoute;
                while (nextEmptyRoute < nbVehicles &&
                       !chromR[nextEmptyRoute].empty())
                    nextEmptyRoute++;
            }
            // Case 2: One customer is in a route (at an end) and the other is
            // not.
            else if (inRoute[s.c1] && !interior[s.c1] && !inRoute[s.c2]) {
                int routeIdx = customerRoute[s.c1];
                if (!chromR[routeIdx].empty()) {
                    if (chromR[routeIdx].front() == s.c1 &&
                        load[routeIdx] + params->cli[s.c2].demand <=
                            params->vehicleCapacity) {
                        chromR[routeIdx].insert(chromR[routeIdx].begin(), s.c2);
                        load[routeIdx] += params->cli[s.c2].demand;
                        inRoute[s.c2] = true;
                        customerRoute[s.c2] = routeIdx;
                        if (chromR[routeIdx].size() > 2) interior[s.c1] = true;
                    } else if (chromR[routeIdx].back() == s.c1 &&
                               load[routeIdx] + params->cli[s.c2].demand <=
                                   params->vehicleCapacity) {
                        chromR[routeIdx].push_back(s.c2);
                        load[routeIdx] += params->cli[s.c2].demand;
                        inRoute[s.c2] = true;
                        customerRoute[s.c2] = routeIdx;
                        if (chromR[routeIdx].size() > 2) interior[s.c1] = true;
                    }
                }
            } else if (inRoute[s.c2] && !interior[s.c2] && !inRoute[s.c1]) {
                int routeIdx = customerRoute[s.c2];
                if (!chromR[routeIdx].empty()) {
                    if (chromR[routeIdx].front() == s.c2 &&
                        load[routeIdx] + params->cli[s.c1].demand <=
                            params->vehicleCapacity) {
                        chromR[routeIdx].insert(chromR[routeIdx].begin(), s.c1);
                        load[routeIdx] += params->cli[s.c1].demand;
                        inRoute[s.c1] = true;
                        customerRoute[s.c1] = routeIdx;
                        if (chromR[routeIdx].size() > 2) interior[s.c2] = true;
                    } else if (chromR[routeIdx].back() == s.c2 &&
                               load[routeIdx] + params->cli[s.c1].demand <=
                                   params->vehicleCapacity) {
                        chromR[routeIdx].push_back(s.c1);
                        load[routeIdx] += params->cli[s.c1].demand;
                        inRoute[s.c1] = true;
                        customerRoute[s.c1] = routeIdx;
                        if (chromR[routeIdx].size() > 2) interior[s.c2] = true;
                    }
                }
            }
            // Case 3: Both customers are in different routes and can be merged.
            else if (inRoute[s.c1] && !interior[s.c1] && inRoute[s.c2] &&
                     !interior[s.c2]) {
                int r1 = customerRoute[s.c1];
                int r2 = customerRoute[s.c2];
                if (r1 != r2 &&
                    load[r1] + load[r2] <= params->vehicleCapacity) {
                    // Determine proper merge order based on route endpoints.
                    if (chromR[r1].front() == s.c1 &&
                        chromR[r2].front() == s.c2) {
                        chromR[r1].insert(chromR[r1].begin(),
                                          chromR[r2].rbegin(),
                                          chromR[r2].rend());
                    } else if (chromR[r1].back() == s.c1 &&
                               chromR[r2].front() == s.c2) {
                        chromR[r1].insert(chromR[r1].end(), chromR[r2].begin(),
                                          chromR[r2].end());
                    } else if (chromR[r1].front() == s.c1 &&
                               chromR[r2].back() == s.c2) {
                        chromR[r1].insert(chromR[r1].begin(),
                                          chromR[r2].begin(), chromR[r2].end());
                    } else if (chromR[r1].back() == s.c1 &&
                               chromR[r2].back() == s.c2) {
                        chromR[r1].insert(chromR[r1].end(), chromR[r2].rbegin(),
                                          chromR[r2].rend());
                    }
                    // Update combined load and update mapping for all customers
                    // in r2.
                    load[r1] += load[r2];
                    for (int cust : chromR[r2]) customerRoute[cust] = r1;
                    chromR[r2].clear();
                    load[r2] = 0.0;
                    interior[s.c1] = interior[s.c2] = true;
                }
            }
        }

        // Assign any remaining customers that are not yet assigned.
        for (int i = 1; i <= nbClients; i++) {
            if (!inRoute[i]) {
                int bestRoute = -1;
                double bestCost = DBL_MAX;
                // Evaluate each route that is not empty.
                for (int r = 0; r < nbVehicles; r++) {
                    if (!chromR[r].empty() && load[r] + params->cli[i].demand <=
                                                  params->vehicleCapacity) {
                        double cost = params->timeCost[chromR[r].back()][i];
                        if (cost < bestCost) {
                            bestCost = cost;
                            bestRoute = r;
                        }
                    }
                }
                if (bestRoute != -1) {
                    chromR[bestRoute].push_back(i);
                    load[bestRoute] += params->cli[i].demand;
                    inRoute[i] = true;
                    customerRoute[i] = bestRoute;
                }
            }
        }

        // Flatten the route representation into a single chromosome.
        int idx = 0;
        for (const auto &route : chromR) {
            for (int customer : route) {
                if (idx < nbClients) chromT[idx++] = customer;
            }
        }

        evaluateCompleteCost();
    } else {
        // If not using rcws, initialize with a random permutation.
        for (int i = 0; i < nbClients; i++) chromT[i] = i + 1;
        std::shuffle(chromT.begin(), chromT.end(), params->ran);
        myCostSol.penalizedCost = 1.e30;
    }
}

Individual::Individual()
    : params(nullptr), isFeasible(false), biasedFitness(0) {
    myCostSol.penalizedCost = 1.e30;
}
