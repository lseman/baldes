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
    successors = std::vector<int>(params->nbClients + 1);
    predecessors = std::vector<int>(params->nbClients + 1);
    chromR = std::vector<std::vector<int>>(params->nbVehicles);
    chromT = std::vector<int>(params->nbClients);
    this->params = params;

    if (rcws) {
        // Initialize helper vectors
        std::vector<bool> inRoute(params->nbClients + 1, false);
        std::vector<bool> interior(params->nbClients + 1, false);
        std::vector<double> load(params->nbVehicles, 0);
        int nextEmptyRoute = 0;

        // Insert pattern if provided
        if (pattern) {
            for (unsigned r = 0; r < pattern->size(); r++) {
                if (nextEmptyRoute == params->nbVehicles) {
                    chromR.push_back(std::vector<int>());
                    load.push_back(0);
                    params->nbVehicles++;
                }
                for (unsigned c = 0; c < (*pattern)[r].size(); c++) {
                    chromR[nextEmptyRoute].push_back((*pattern)[r][c]);
                    load[nextEmptyRoute] +=
                        params->cli[(*pattern)[r][c]].demand;
                    inRoute[(*pattern)[r][c]] = true;
                    if (c && c != (*pattern)[r].size() - 1)
                        interior[(*pattern)[r][c]] = true;
                }
                while (nextEmptyRoute < params->nbVehicles &&
                       !chromR[nextEmptyRoute].empty())
                    nextEmptyRoute++;
            }
        }

        // Process savings list
        unsigned savingsCount = 0;
        while (savingsCount < params->savingsList.size()) {
            Savings s = params->savingsList[savingsCount++];
            if (s.value <= 0) continue;  // Skip invalid savings

            // Check feasibility and merge routes
            if (params->cli[s.c1].demand + params->cli[s.c2].demand <=
                params->vehicleCapacity) {
                if (!inRoute[s.c1] && !inRoute[s.c2]) {
                    // Create a new route
                    if (nextEmptyRoute == params->nbVehicles) {
                        chromR.push_back(std::vector<int>());
                        load.push_back(0);
                        params->nbVehicles++;
                    }
                    chromR[nextEmptyRoute].push_back(s.c1);
                    chromR[nextEmptyRoute].push_back(s.c2);
                    load[nextEmptyRoute] +=
                        params->cli[s.c1].demand + params->cli[s.c2].demand;
                    inRoute[s.c1] = inRoute[s.c2] = true;
                    while (nextEmptyRoute < params->nbVehicles &&
                           !chromR[nextEmptyRoute].empty())
                        nextEmptyRoute++;
                } else if (inRoute[s.c1] && !interior[s.c1] && !inRoute[s.c2]) {
                    // Merge s.c2 into the route containing s.c1
                    for (int r = 0; r < params->nbVehicles; r++) {
                        if (!chromR[r].empty()) {
                            if (chromR[r].front() == s.c1) {
                                if (load[r] + params->cli[s.c2].demand <=
                                    params->vehicleCapacity) {
                                    chromR[r].insert(chromR[r].begin(), s.c2);
                                    load[r] += params->cli[s.c2].demand;
                                    inRoute[s.c2] = true;
                                    if (chromR[r].size() > 2)
                                        interior[s.c1] = true;
                                }
                                break;
                            } else if (chromR[r].back() == s.c1) {
                                if (load[r] + params->cli[s.c2].demand <=
                                    params->vehicleCapacity) {
                                    chromR[r].push_back(s.c2);
                                    load[r] += params->cli[s.c2].demand;
                                    inRoute[s.c2] = true;
                                    if (chromR[r].size() > 2)
                                        interior[s.c1] = true;
                                }
                                break;
                            }
                        }
                    }
                } else if (inRoute[s.c2] && !interior[s.c2] && !inRoute[s.c1]) {
                    // Merge s.c1 into the route containing s.c2
                    for (int r = 0; r < params->nbVehicles; r++) {
                        if (!chromR[r].empty()) {
                            if (chromR[r].front() == s.c2) {
                                if (load[r] + params->cli[s.c1].demand <=
                                    params->vehicleCapacity) {
                                    chromR[r].insert(chromR[r].begin(), s.c1);
                                    load[r] += params->cli[s.c1].demand;
                                    inRoute[s.c1] = true;
                                    if (chromR[r].size() > 2)
                                        interior[s.c2] = true;
                                }
                                break;
                            } else if (chromR[r].back() == s.c2) {
                                if (load[r] + params->cli[s.c1].demand <=
                                    params->vehicleCapacity) {
                                    chromR[r].push_back(s.c1);
                                    load[r] += params->cli[s.c1].demand;
                                    inRoute[s.c1] = true;
                                    if (chromR[r].size() > 2)
                                        interior[s.c2] = true;
                                }
                                break;
                            }
                        }
                    }
                } else if (inRoute[s.c1] && !interior[s.c1] && inRoute[s.c2] &&
                           !interior[s.c2]) {
                    // Merge two routes
                    int r1 = -1, r2 = -1;
                    for (int r = 0; r < params->nbVehicles; r++) {
                        if (!chromR[r].empty()) {
                            if (chromR[r].front() == s.c1 ||
                                chromR[r].back() == s.c1)
                                r1 = r;
                            if (chromR[r].front() == s.c2 ||
                                chromR[r].back() == s.c2)
                                r2 = r;
                            if (r1 != -1 && r2 != -1) break;
                        }
                    }
                    if (r1 != r2 &&
                        load[r1] + load[r2] <= params->vehicleCapacity) {
                        if (chromR[r1].front() == s.c1 &&
                            chromR[r2].front() == s.c2) {
                            chromR[r1].insert(chromR[r1].begin(),
                                              chromR[r2].rbegin(),
                                              chromR[r2].rend());
                        } else if (chromR[r1].back() == s.c1 &&
                                   chromR[r2].front() == s.c2) {
                            chromR[r1].insert(chromR[r1].end(),
                                              chromR[r2].begin(),
                                              chromR[r2].end());
                        } else if (chromR[r1].front() == s.c1 &&
                                   chromR[r2].back() == s.c2) {
                            chromR[r1].insert(chromR[r1].begin(),
                                              chromR[r2].begin(),
                                              chromR[r2].end());
                        } else if (chromR[r1].back() == s.c1 &&
                                   chromR[r2].back() == s.c2) {
                            chromR[r1].insert(chromR[r1].end(),
                                              chromR[r2].rbegin(),
                                              chromR[r2].rend());
                        }
                        load[r1] += load[r2];
                        chromR[r2].clear();
                        load[r2] = 0;
                        interior[s.c1] = interior[s.c2] = true;
                    }
                }
            }
        }

        // Assign unassigned customers to routes
        for (int i = 1; i <= params->nbClients; i++) {
            if (!inRoute[i]) {
                int bestRoute = -1;
                double bestCost = DBL_MAX;
                for (int r = 0; r < params->nbVehicles; r++) {
                    if (load[r] + params->cli[i].demand <=
                        params->vehicleCapacity) {
                        double cost = params->timeCost[chromR[r].back()][i];
                        if (cost < bestCost) {
                            bestRoute = r;
                            bestCost = cost;
                        }
                    }
                }
                if (bestRoute != -1) {
                    chromR[bestRoute].push_back(i);
                    load[bestRoute] += params->cli[i].demand;
                    inRoute[i] = true;
                }
            }
        }

        // Flatten chromR into chromT
        int c = 0;
        for (const auto &route : chromR) {
            for (int customer : route) {
                chromT[c++] = customer;
            }
        }

        evaluateCompleteCost();
    } else {
        // Initialize with a random permutation
        for (int i = 0; i < params->nbClients; i++) chromT[i] = i + 1;
        std::shuffle(chromT.begin(), chromT.end(), params->ran);
        myCostSol.penalizedCost = 1.e30;
    }
}

Individual::Individual()
    : params(nullptr), isFeasible(false), biasedFitness(0) {
    myCostSol.penalizedCost = 1.e30;
}
