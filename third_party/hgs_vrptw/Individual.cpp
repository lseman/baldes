#include <algorithm>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "Individual.h"
#include "Params.h"
#include "fmt/core.h"

void Individual::evaluateCompleteCost() {
    myCostSol = CostSol();

    for (int r = 0; r < params->nbVehicles; ++r) {
        if (chromR[r].empty()) continue;

        int latestReleaseTime = params->cli[chromR[r][0]].releaseTime;
        for (int i = 1; i < static_cast<int>(chromR[r].size()); ++i) {
            latestReleaseTime = std::max(latestReleaseTime, params->cli[chromR[r][i]].releaseTime);
        }

        int distance = params->timeCost.get(0, chromR[r][0]);
        int load     = params->cli[chromR[r][0]].demand;
        int service  = params->cli[chromR[r][0]].serviceDuration;
        int time     = latestReleaseTime + distance;
        int waitTime = 0, timeWarp = 0;

        // Handle time windows if n_tw > 0
        if (params->cli[chromR[r][0]].n_tw > 0) {
            // Use multiple time windows
            auto &firstTimeWindows = params->cli[chromR[r][0]].timeWindows;
            time                   = adjustToTimeWindow(time, firstTimeWindows, waitTime, timeWarp);
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

        for (int i = 1; i < static_cast<int>(chromR[r].size()); ++i) {
            int prev = chromR[r][i - 1], curr = chromR[r][i];
            distance += params->timeCost.get(prev, curr);
            load += params->cli[curr].demand;
            service += params->cli[curr].serviceDuration;
            time += params->cli[prev].serviceDuration + params->timeCost.get(prev, curr);

            // Handle time windows if n_tw > 0
            if (params->cli[curr].n_tw > 0) {
                auto &currTimeWindows = params->cli[curr].timeWindows;
                time                  = adjustToTimeWindow(time, currTimeWindows, waitTime, timeWarp);
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
            successors[prev]   = curr;
        }

        successors[chromR[r].back()] = 0;
        distance += params->timeCost.get(chromR[r].back(), 0);
        time += params->cli[chromR[r].back()].serviceDuration + params->timeCost.get(chromR[r].back(), 0);

        // Final depot logic (assuming depot time window)
        if (time > params->cli[0].latestArrival) {
            timeWarp += time - params->cli[0].latestArrival;
            time = params->cli[0].latestArrival;
        }

        myCostSol.distance += distance;
        myCostSol.waitTime += waitTime;
        myCostSol.timeWarp += timeWarp;
        myCostSol.nbRoutes++;

        if (load > params->vehicleCapacity) { myCostSol.capacityExcess += load - params->vehicleCapacity; }
    }

    myCostSol.penalizedCost = myCostSol.distance + myCostSol.capacityExcess * params->penaltyCapacity +
                              myCostSol.timeWarp * params->penaltyTimeWarp +
                              myCostSol.waitTime * params->penaltyWaitTime;
    isFeasible = (myCostSol.capacityExcess < MY_EPSILON && myCostSol.timeWarp < MY_EPSILON);
}

// Helper function to adjust time based on multiple time windows
int Individual::adjustToTimeWindow(int time, const std::vector<TimeWindow> &timeWindows, int &waitTime, int &timeWarp) {
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
    for (int i = 0; i < params->nbClients; i++) { chromT[i] = i + 1; }
    // Do a random shuffle chromT from begin to end
    std::shuffle(chromT.begin(), chromT.end(), params->rng);
}

void Individual::removeProximity(Individual *indiv) {
    // Find indiv in indivsPerProximity
    auto it = std::find_if(indivsPerProximity.begin(), indivsPerProximity.end(),
                           [indiv](const auto &pair) { return pair.second == indiv; });
    // Remove if found
    if (it != indivsPerProximity.end()) { indivsPerProximity.erase(it); }
}

double Individual::brokenPairsDistance(Individual *indiv2) {
    // Initialize the difference to zero. Then loop over all clients of this individual
    int differences = 0;
    for (int j = 1; j <= params->nbClients; j++) {
        // Increase the difference if the successor of j in this individual is not directly linked to j in indiv2
        if (successors[j] != indiv2->successors[j] && successors[j] != indiv2->predecessors[j]) { differences++; }
        // Last loop covers all but the first arc. Increase the difference if the predecessor of j in this individual is
        // not directly linked to j in indiv2
        if (predecessors[j] == 0 && indiv2->predecessors[j] != 0 && indiv2->successors[j] != 0) { differences++; }
    }
    return static_cast<double>(differences) / params->nbClients;
}

double Individual::averageBrokenPairsDistanceClosest(int nbClosest) {
    double result  = 0;
    int    maxSize = std::min(nbClosest, static_cast<int>(indivsPerProximity.size()));
    auto   it      = indivsPerProximity.begin();
    for (int i = 0; i < maxSize; i++) {
        result += it->first;
        ++it;
    }
    return result / maxSize;
}

Individual::Individual(Params *params, bool initializeChromTAndShuffle)
    : params(params), isFeasible(false), biasedFitness(0) {
    successors   = std::vector<int>(params->nbClients + 1);
    predecessors = std::vector<int>(params->nbClients + 1);
    chromR       = std::vector<std::vector<int>>(params->nbVehicles);
    chromT       = std::vector<int>(params->nbClients);
    if (initializeChromTAndShuffle) { shuffleChromT(); }
}

Individual::Individual(Params *params, std::string solutionStr) : params(params), isFeasible(false), biasedFitness(0) {
    successors   = std::vector<int>(params->nbClients + 1);
    predecessors = std::vector<int>(params->nbClients + 1);
    chromR       = std::vector<std::vector<int>>(params->nbVehicles);
    chromT       = std::vector<int>(params->nbClients);

    std::stringstream ss(solutionStr);
    int               inputCustomer;
    // Loops as long as there is an integer to read
    int pos   = 0;
    int route = 0;
    while (ss >> inputCustomer) {
        if (inputCustomer == 0) {
            // Depot
            route++;
            assert(route < params->nbVehicles);
        } else {
            chromR[route].push_back(inputCustomer);
            chromT[pos] = inputCustomer;
            pos++;
        }
    }
    assert(pos == params->nbClients);
    evaluateCompleteCost();
}

Individual::Individual(Params *params, bool rcws, std::vector<std::vector<int>> *pattern) {
    successors   = std::vector<int>(params->nbClients + 1);
    predecessors = std::vector<int>(params->nbClients + 1);
    chromR       = std::vector<std::vector<int>>(params->nbVehicles);
    chromT       = std::vector<int>(params->nbClients);
    this->params = params;

    if (rcws) // initialize the individual with a randomized version of the Clarke & Wright savings heuristic
    {
        std::vector<bool>    inRoute                    = std::vector<bool>(params->nbClients + 1, false);
        std::vector<bool>    interior                   = std::vector<bool>(params->nbClients + 1, false);
        std::vector<double>  load                       = std::vector<double>(params->nbVehicles, 0);
        std::vector<Savings> tournamentSavings          = std::vector<Savings>(6);
        std::vector<double>  selectionProbabilities     = std::vector<double>(6);
        int                  tournamentSavingsOccupancy = 0;
        unsigned             savingsCount               = 0;
        int                  nextEmptyRoute             = 0;
        int                  nbVehicles                 = params->nbVehicles;

        if (pattern) // insert pattern
            for (unsigned r = 0; r < pattern->size(); r++) {
                if (nextEmptyRoute == nbVehicles) {
                    nbVehicles++;
                    chromR.push_back(std::vector<int>());
                    load.push_back(0);
                }
                for (unsigned c = 0; c < (*pattern)[r].size(); c++) {
                    chromR[nextEmptyRoute].push_back((*pattern)[r][c]);
                    load[nextEmptyRoute] += params->cli[(*pattern)[r][c]].demand;
                    inRoute[(*pattern)[r][c]] = true;
                    if (c && c != (*pattern)[r].size() - 1) interior[(*pattern)[r][c]] = true;
                }
                while (nextEmptyRoute < nbVehicles && !chromR[nextEmptyRoute].empty()) nextEmptyRoute++;
            }

        while (savingsCount < params->savingsList.size() || tournamentSavingsOccupancy > 0) {
            int tournamentSize = std::min(2 + std::rand() % 5, (int)params->savingsList.size() - (int)savingsCount +
                                                                   tournamentSavingsOccupancy);

            while (tournamentSavingsOccupancy < tournamentSize) {
                Savings s = params->savingsList[savingsCount++];
                if (s.value > 0)
                    tournamentSavings[tournamentSavingsOccupancy++] = s;
                else {
                    tournamentSize = tournamentSavingsOccupancy;
                    savingsCount   = params->savingsList.size();
                }
            }

            double tournamentSavingsSum = 0;
            for (int i = 0; i < tournamentSize; i++) tournamentSavingsSum += tournamentSavings[i].value;

            for (int i = 0; i < tournamentSize; i++)
                selectionProbabilities[i] = tournamentSavings[i].value / tournamentSavingsSum;

            double cumulativeProbability = 0;
            double rand                  = (double)std::rand() / RAND_MAX;

            for (int i = 0; i < tournamentSize; i++) {
                if (rand <= selectionProbabilities[i] + cumulativeProbability) {
                    // Process tournamentSavings[i]
                    if (params->cli[tournamentSavings[i].c1].demand + params->cli[tournamentSavings[i].c2].demand <=
                        params->vehicleCapacity) {
                        if (!inRoute[tournamentSavings[i].c1] && !inRoute[tournamentSavings[i].c2]) {
                            if (nextEmptyRoute == nbVehicles) {
                                nbVehicles++;
                                chromR.push_back(std::vector<int>());
                                load.push_back(0);
                            }
                            chromR[nextEmptyRoute].push_back(tournamentSavings[i].c1);
                            chromR[nextEmptyRoute].push_back(tournamentSavings[i].c2);
                            load[nextEmptyRoute] += params->cli[tournamentSavings[i].c1].demand +
                                                    params->cli[tournamentSavings[i].c2].demand;
                            inRoute[tournamentSavings[i].c1] = true;
                            inRoute[tournamentSavings[i].c2] = true;
                            while (nextEmptyRoute < nbVehicles && !chromR[nextEmptyRoute].empty()) nextEmptyRoute++;
                        } else if (inRoute[tournamentSavings[i].c1] && !interior[tournamentSavings[i].c1] &&
                                   !inRoute[tournamentSavings[i].c2]) {
                            for (int r = 0; r < nbVehicles; r++)
                                if (!chromR[r].empty()) {
                                    if (chromR[r].front() == tournamentSavings[i].c1) {
                                        if (load[r] + params->cli[tournamentSavings[i].c2].demand <=
                                            params->vehicleCapacity) {
                                            chromR[r].insert(chromR[r].begin(), tournamentSavings[i].c2);
                                            load[r] += params->cli[tournamentSavings[i].c2].demand;
                                            inRoute[tournamentSavings[i].c2] = true;
                                            if (chromR[r].size() > 2) interior[tournamentSavings[i].c1] = true;
                                        }
                                        break;
                                    } else if (chromR[r].back() == tournamentSavings[i].c1) {
                                        if (load[r] + params->cli[tournamentSavings[i].c2].demand <=
                                            params->vehicleCapacity) {
                                            chromR[r].push_back(tournamentSavings[i].c2);
                                            load[r] += params->cli[tournamentSavings[i].c2].demand;
                                            inRoute[tournamentSavings[i].c2] = true;
                                            if (chromR[r].size() > 2) interior[tournamentSavings[i].c1] = true;
                                        }
                                        break;
                                    }
                                }
                        } else if (inRoute[tournamentSavings[i].c2] && !interior[tournamentSavings[i].c2] &&
                                   !inRoute[tournamentSavings[i].c1]) {
                            for (int r = 0; r < nbVehicles; r++)
                                if (!chromR[r].empty()) {
                                    if (chromR[r].front() == tournamentSavings[i].c2) {
                                        if (load[r] + params->cli[tournamentSavings[i].c1].demand <=
                                            params->vehicleCapacity) {
                                            chromR[r].insert(chromR[r].begin(), tournamentSavings[i].c1);
                                            load[r] += params->cli[tournamentSavings[i].c1].demand;
                                            inRoute[tournamentSavings[i].c1] = true;
                                            if (chromR[r].size() > 2) interior[tournamentSavings[i].c2] = true;
                                        }
                                        break;
                                    } else if (chromR[r].back() == tournamentSavings[i].c2) {
                                        if (load[r] + params->cli[tournamentSavings[i].c1].demand <=
                                            params->vehicleCapacity) {
                                            chromR[r].push_back(tournamentSavings[i].c1);
                                            load[r] += params->cli[tournamentSavings[i].c1].demand;
                                            inRoute[tournamentSavings[i].c1] = true;
                                            if (chromR[r].size() > 2) interior[tournamentSavings[i].c2] = true;
                                        }
                                        break;
                                    }
                                }
                        } else if (inRoute[tournamentSavings[i].c1] && !interior[tournamentSavings[i].c1] &&
                                   inRoute[tournamentSavings[i].c2] && !interior[tournamentSavings[i].c2]) {
                            int r1   = -1;
                            int r2   = -1;
                            int pos1 = 0;
                            int pos2 = 0;

                            for (int r = 0; r < nbVehicles; r++)
                                if (!chromR[r].empty()) {
                                    if (chromR[r].front() == tournamentSavings[i].c1) {
                                        r1   = r;
                                        pos1 = 0;
                                    } else if (chromR[r].back() == tournamentSavings[i].c1) {
                                        r1   = r;
                                        pos1 = chromR[r].size() - 1;
                                    }

                                    if (chromR[r].front() == tournamentSavings[i].c2) {
                                        r2   = r;
                                        pos2 = 0;
                                    } else if (chromR[r].back() == tournamentSavings[i].c2) {
                                        r2   = r;
                                        pos2 = chromR[r].size() - 1;
                                    }

                                    if (r1 > -1 && r2 > -1) {
                                        if (r1 != r2 && load[r1] + load[r2] <= params->vehicleCapacity) {
                                            if (pos1 == 0) {
                                                if (pos2 == 0) {
                                                    chromR[r1].insert(chromR[r1].begin(), chromR[r2].rbegin(),
                                                                      chromR[r2].rend());
                                                    load[r1] += load[r2];
                                                    chromR[r2].clear();
                                                    load[r2] = 0;
                                                    if (r2 < nextEmptyRoute) nextEmptyRoute = r2;
                                                } else {
                                                    chromR[r2].insert(chromR[r2].end(), chromR[r1].begin(),
                                                                      chromR[r1].end());
                                                    load[r2] += load[r1];
                                                    chromR[r1].clear();
                                                    load[r1] = 0;
                                                    if (r1 < nextEmptyRoute) nextEmptyRoute = r1;
                                                }

                                                interior[tournamentSavings[i].c1] = true;
                                                interior[tournamentSavings[i].c2] = true;
                                            } else if (pos2 == 0) {
                                                chromR[r1].insert(chromR[r1].end(), chromR[r2].begin(),
                                                                  chromR[r2].end());
                                                load[r1] += load[r2];
                                                chromR[r2].clear();
                                                load[r2] = 0;
                                                if (r2 < nextEmptyRoute) nextEmptyRoute = r2;

                                                interior[tournamentSavings[i].c1] = true;
                                                interior[tournamentSavings[i].c2] = true;
                                            } else {
                                                chromR[r1].insert(chromR[r1].end(), chromR[r2].rbegin(),
                                                                  chromR[r2].rend());
                                                load[r1] += load[r2];
                                                chromR[r2].clear();
                                                load[r2] = 0;
                                                if (r2 < nextEmptyRoute) nextEmptyRoute = r2;

                                                interior[tournamentSavings[i].c1] = true;
                                                interior[tournamentSavings[i].c2] = true;
                                            }
                                        }

                                        break;
                                    }
                                }
                        }
                    }

                    // Update tournament structures
                    for (int j = i; j < tournamentSavingsOccupancy - 1; j++)
                        tournamentSavings[j] = tournamentSavings[j + 1];

                    tournamentSavingsOccupancy--;
                    break;
                }
                cumulativeProbability += selectionProbabilities[i];
            }
        }

        int i = nextEmptyRoute + 1;
        while (nextEmptyRoute < nbVehicles && i < nbVehicles) {
            if (!chromR[i].empty()) {
                std::vector<int> temp  = chromR[nextEmptyRoute];
                chromR[nextEmptyRoute] = chromR[i];
                chromR[i]              = temp;
                while (nextEmptyRoute < nbVehicles && !chromR[nextEmptyRoute].empty()) nextEmptyRoute++;
                i = nextEmptyRoute;
            }
            i++;
        }

        for (int i = 0; i < nbVehicles - params->nbVehicles; i++) chromR.pop_back();

        for (int i = 1; i <= params->nbClients; i++)
            if (!inRoute[i]) {
                int    bestRoute         = -1;
                double bestInsertionCost = DBL_MAX;
                for (int r = 0; r < params->nbVehicles; r++)
                    if (load[r] + params->cli[i].demand <= params->vehicleCapacity) {
                        int c = 0;
                        if (!chromR[r].empty()) c = chromR[r].back();

                        if (params->timeCost[c][i] < bestInsertionCost) {
                            bestRoute         = r;
                            bestInsertionCost = params->timeCost[c][i];
                        }
                    }

                if (bestRoute < 0)
                    for (int r = 0; r < params->nbVehicles; r++) {
                        int    c = chromR[r].back();
                        double penalizedCost =
                            params->timeCost[c][i] +
                            (load[r] + params->cli[i].demand - params->vehicleCapacity) * params->penaltyCapacity;
                        if (penalizedCost < bestInsertionCost) {
                            bestRoute         = r;
                            bestInsertionCost = penalizedCost;
                        }
                    }

                chromR[bestRoute].push_back(i);
                load[bestRoute] += params->cli[i].demand;
            }

        int c = 0;
        for (unsigned i = 0; i < chromR.size(); i++)
            for (unsigned j = 0; j < chromR[i].size(); j++) chromT[c++] = chromR[i][j];

        evaluateCompleteCost();
    } else // initialize the individual with a random permutation
    {
        for (int i = 0; i < params->nbClients; i++) chromT[i] = i + 1;
        std::shuffle(chromT.begin(), chromT.end(), params->ran);
        myCostSol.penalizedCost = 1.e30;
    }
}

Individual::Individual() : params(nullptr), isFeasible(false), biasedFitness(0) { myCostSol.penalizedCost = 1.e30; }
