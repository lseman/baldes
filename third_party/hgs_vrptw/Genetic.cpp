#include "ankerl/unordered_dense.h"
#include <algorithm>
#include <bitset>
#include <iterator>
#include <time.h>

#include "Genetic.h"
#include "Individual.h"
#include "LocalSearch.h"
#include "Params.h"
#include "Population.h"
#include "Split.h"

void Genetic::run(int maxIterNonProd, int timeLimit) {
    if (params->nbClients == 1) return; // Edge case: 1 client, GA makes no sense

    auto mdmNURestarts = 0.05;
    int  nbRestarts = 0, nbIterNonProd = 1;

    for (int nbIter = 0; nbIterNonProd <= maxIterNonProd && !params->isTimeLimitExceeded(); ++nbIter) {
        /* SELECTION AND CROSSOVER */
        Individual *offspring = bestOfSREXAndOXCrossovers(population->getNonIdenticalParentsBinaryTournament());

        /* LOCAL SEARCH */
        localSearch->run(offspring, params->penaltyCapacity, params->penaltyTimeWarp);
        bool isNewBest = population->addIndividual(offspring, true);

        // Repair infeasible individual with some probability
        if (!offspring->isFeasible &&
            params->rng() % 100 < static_cast<unsigned int>(params->config.repairProbability)) {
            localSearch->run(offspring, params->penaltyCapacity * 10.0, params->penaltyTimeWarp * 10.0);
            if (offspring->isFeasible) { isNewBest = population->addIndividual(offspring, false) || isNewBest; }
        }

        /* TRACKING IMPROVEMENT */
        nbIterNonProd = isNewBest ? 1 : nbIterNonProd + 1;

        /* DIVERSIFICATION, PENALTY MANAGEMENT, AND LOGGING */
        if (nbIter % 100 == 0) population->managePenalties();
        if (nbIter % 500 == 0) population->printState(nbIter, nbIterNonProd);

        if (params->config.logPoolInterval > 0 && nbIter % params->config.logPoolInterval == 0) {
            population->exportPopulation(nbIter, params->config.pathSolution + ".log.csv");
        }

        /* RESTART LOGIC */
        if (timeLimit != INT_MAX && nbIterNonProd == maxIterNonProd && params->config.doRepeatUntilTimeLimit) {
            ++nbRestarts;
            double elapsedTime = (double)(clock() - params->startCPUTime) / CLOCKS_PER_SEC;
            int    estimatedRestarts =
                std::min(static_cast<int>(params->config.timeLimit / (elapsedTime / nbRestarts)), 1000);
            population->mdmEliteMaxNonUpdatingRestarts = static_cast<int>(mdmNURestarts * estimatedRestarts);
            population->mineElite();
            population->restart();
            nbIterNonProd = 1;
        }

        /* PARAMETER ADJUSTMENTS */
        if (params->config.growNbGranularSize != 0) {
            bool shouldGrowGranular = (params->config.growNbGranularAfterIterations > 0 &&
                                       nbIter % params->config.growNbGranularAfterIterations == 0) ||
                                      (params->config.growNbGranularAfterNonImprovementIterations > 0 &&
                                       nbIterNonProd % params->config.growNbGranularAfterNonImprovementIterations == 0);
            if (shouldGrowGranular) {
                params->config.nbGranular += params->config.growNbGranularSize;
                params->SetCorrelatedVertices();
            }
        }

        if (params->config.growPopulationSize != 0) {
            bool shouldGrowPopulation =
                (params->config.growPopulationAfterIterations > 0 &&
                 nbIter % params->config.growPopulationAfterIterations == 0) ||
                (params->config.growPopulationAfterNonImprovementIterations > 0 &&
                 nbIterNonProd % params->config.growPopulationAfterNonImprovementIterations == 0);
            if (shouldGrowPopulation) { params->config.minimumPopulationSize += params->config.growPopulationSize; }
        }
    }
}

Individual *Genetic::crossoverOX(std::pair<const Individual *, const Individual *> parents) {
    // Picking the start and end of the crossover zone
    int start = params->rng() % params->nbClients;
    int end   = params->rng() % params->nbClients;

    // If the start and end overlap, change the end of the crossover zone
    while (end == start) { end = params->rng() % params->nbClients; }

    // Create two individuals using OX
    doOXcrossover(candidateOffsprings[2], parents, start, end);
    doOXcrossover(candidateOffsprings[3], parents, start, end);

    // Return the best individual of the two, based on penalizedCost
    return candidateOffsprings[2]->myCostSol.penalizedCost < candidateOffsprings[3]->myCostSol.penalizedCost
               ? candidateOffsprings[2]
               : candidateOffsprings[3];
}

void Genetic::doOXcrossover(Individual *result, std::pair<const Individual *, const Individual *> parents, int start,
                            int end) {
    std::bitset<N_SIZE> freqClient; // Use bitset for fast lookup
    freqClient.reset();

    const int nbClients = params->nbClients;
    const int idxEnd    = (end + 1 == nbClients) ? 0 : end + 1;

    // First loop: copy the segment from parent A
    for (int j = start; j != idxEnd; j = (j + 1) % nbClients) {
        int elem          = parents.first->chromT[j];
        result->chromT[j] = elem;
        freqClient.set(elem);
    }

    // Second loop: fill remaining elements from parent B
    for (int i = idxEnd, j = idxEnd; j != start; i = (i + 1) % nbClients) {
        int temp = parents.second->chromT[i];
        if (!freqClient.test(temp)) {
            result->chromT[j] = temp;
            j                 = (j + 1) % nbClients;
        }
    }

    // Call Split algorithm to complete the individual
    split->generalSplit(result, params->nbVehicles);
}

Individual *Genetic::crossoverSREX(std::pair<const Individual *, const Individual *> parents) {
    // Get the number of routes of both parents
    int nOfRoutesA = parents.first->myCostSol.nbRoutes;
    int nOfRoutesB = parents.second->myCostSol.nbRoutes;

    // Picking the start index of routes to replace of parent A
    // We like to replace routes with a large overlap of tasks, so we choose adjacent routes (they are sorted on polar
    // angle)
    int startA = params->rng() % nOfRoutesA;
    int nOfMovedRoutes =
        std::min(nOfRoutesA, nOfRoutesB) == 1
            ? 1
            : params->rng() % (std::min(nOfRoutesA - 1, nOfRoutesB - 1)) + 1; // Prevent not moving any routes
    int startB = startA < nOfRoutesB ? startA : 0;

    ankerl::unordered_dense::set<int> clientsInSelectedA;
    for (int r = 0; r < nOfMovedRoutes; r++) {
        // Insert the first
        clientsInSelectedA.insert(parents.first->chromR[(startA + r) % nOfRoutesA].begin(),
                                  parents.first->chromR[(startA + r) % nOfRoutesA].end());
    }

    ankerl::unordered_dense::set<int> clientsInSelectedB;
    for (int r = 0; r < nOfMovedRoutes; r++) {
        clientsInSelectedB.insert(parents.second->chromR[(startB + r) % nOfRoutesB].begin(),
                                  parents.second->chromR[(startB + r) % nOfRoutesB].end());
    }

    bool improved = true;
    while (improved) {
        const int idxALeft      = (startA - 1 + nOfRoutesA) % nOfRoutesA;
        const int idxARight     = (startA + nOfMovedRoutes) % nOfRoutesA;
        const int idxALastMoved = (startA + nOfMovedRoutes - 1) % nOfRoutesA;
        const int idxBLeft      = (startB - 1 + nOfRoutesB) % nOfRoutesB;
        const int idxBRight     = (startB + nOfMovedRoutes) % nOfRoutesB;
        const int idxBLastMoved = (startB - 1 + nOfMovedRoutes) % nOfRoutesB;

        // Cache counts for A and B
        const auto countClientsOut = [](const std::vector<int>                  &route,
                                        const ankerl::unordered_dense::set<int> &clientsSet) {
            return std::count_if(route.begin(), route.end(),
                                 [&clientsSet](int c) { return clientsSet.find(c) == clientsSet.end(); });
        };

        const auto countClientsIn = [](const std::vector<int>                  &route,
                                       const ankerl::unordered_dense::set<int> &clientsSet) {
            return std::count_if(route.begin(), route.end(),
                                 [&clientsSet](int c) { return clientsSet.find(c) != clientsSet.end(); });
        };

        // Difference for moving 'left' in parent A
        const int differenceALeft = countClientsOut(parents.first->chromR[idxALeft], clientsInSelectedB) -
                                    countClientsOut(parents.first->chromR[idxALastMoved], clientsInSelectedB);

        // Difference for moving 'right' in parent A
        const int differenceARight = countClientsOut(parents.first->chromR[idxARight], clientsInSelectedB) -
                                     countClientsOut(parents.first->chromR[startA], clientsInSelectedB);

        // Difference for moving 'left' in parent B
        const int differenceBLeft = countClientsIn(parents.second->chromR[idxBLastMoved], clientsInSelectedA) -
                                    countClientsIn(parents.second->chromR[idxBLeft], clientsInSelectedA);

        // Difference for moving 'right' in parent B
        const int differenceBRight = countClientsIn(parents.second->chromR[startB], clientsInSelectedA) -
                                     countClientsIn(parents.second->chromR[idxBRight], clientsInSelectedA);

        const int bestDifference = std::min({differenceALeft, differenceARight, differenceBLeft, differenceBRight});
        // Avoiding infinite loop by adding a guard in case nothing changes
        if (bestDifference < 0) {
            if (bestDifference == differenceALeft) {
                for (int c : parents.first->chromR[(startA + nOfMovedRoutes - 1) % nOfRoutesA]) {
                    auto it = clientsInSelectedA.find(c);
                    if (it != clientsInSelectedA.end()) clientsInSelectedA.erase(it);
                }
                startA = (startA - 1 + nOfRoutesA) % nOfRoutesA;
                for (int c : parents.first->chromR[startA]) { clientsInSelectedA.insert(c); }
            } else if (bestDifference == differenceARight) {
                for (int c : parents.first->chromR[startA]) {
                    auto it = clientsInSelectedA.find(c);
                    if (it != clientsInSelectedA.end()) clientsInSelectedA.erase(it);
                }
                startA = (startA + 1) % nOfRoutesA;
                for (int c : parents.first->chromR[(startA + nOfMovedRoutes - 1) % nOfRoutesA]) {
                    clientsInSelectedA.insert(c);
                }
            } else if (bestDifference == differenceBLeft) {
                for (int c : parents.second->chromR[(startB + nOfMovedRoutes - 1) % nOfRoutesB]) {
                    auto it = clientsInSelectedB.find(c);
                    if (it != clientsInSelectedB.end()) clientsInSelectedB.erase(it);
                }
                startB = (startB - 1 + nOfRoutesB) % nOfRoutesB;
                for (int c : parents.second->chromR[startB]) { clientsInSelectedB.insert(c); }
            } else if (bestDifference == differenceBRight) {
                for (int c : parents.second->chromR[startB]) {
                    auto it = clientsInSelectedB.find(c);
                    if (it != clientsInSelectedB.end()) clientsInSelectedB.erase(it);
                }
                startB = (startB + 1) % nOfRoutesB;
                for (int c : parents.second->chromR[(startB + nOfMovedRoutes - 1) % nOfRoutesB]) {
                    clientsInSelectedB.insert(c);
                }
            }
        } else {
            improved = false;
        }
    }

    // Identify differences between route sets
    std::vector<int> clientsInSelectedANotBVec;
    clientsInSelectedANotBVec.reserve(clientsInSelectedA.size()); // Reserve space to avoid reallocations
    for (int client : clientsInSelectedA) {
        if (clientsInSelectedB.find(client) == clientsInSelectedB.end()) {
            clientsInSelectedANotBVec.push_back(client);
        }
    }

    std::vector<int> clientsInSelectedBNotAVec;
    clientsInSelectedBNotAVec.reserve(clientsInSelectedB.size()); // Reserve space to avoid reallocations
    for (int client : clientsInSelectedB) {
        if (clientsInSelectedA.find(client) == clientsInSelectedA.end()) {
            clientsInSelectedBNotAVec.push_back(client);
        }
    }

    // Convert vector to unordered_set
    ankerl::unordered_dense::set<int> clientsInSelectedANotB(clientsInSelectedANotBVec.begin(),
                                                             clientsInSelectedANotBVec.end());
    ankerl::unordered_dense::set<int> clientsInSelectedBNotA(clientsInSelectedBNotAVec.begin(),
                                                             clientsInSelectedBNotAVec.end());

    // Replace selected routes from parent B into parent A
    for (int r = 0; r < nOfMovedRoutes; r++) {
        int indexA = (startA + r) % nOfRoutesA;
        int indexB = (startB + r) % nOfRoutesB;

        auto &offspring0Route = candidateOffsprings[0]->chromR[indexA];
        auto &offspring1Route = candidateOffsprings[1]->chromR[indexA];

        offspring0Route.clear(); // Clears but retains capacity
        offspring1Route.clear();

        // Batch lookup: Make the lookup set cache-friendly by processing the route once
        for (int c : parents.second->chromR[indexB]) {
            offspring0Route.push_back(c); // Always copy into offspring 0

            // Efficiently check presence in clientsInSelectedBNotA and copy into offspring 1 if absent
            if (!clientsInSelectedBNotA.contains(c)) { // `contains` is cleaner and faster in modern C++20
                offspring1Route.push_back(c);
            }
        }
    }

    // Move routes from parent A that are kept
    for (int r = nOfMovedRoutes; r < nOfRoutesA; r++) {
        int indexA = (startA + r) % nOfRoutesA;

        auto &offspring0Route = candidateOffsprings[0]->chromR[indexA];
        auto &offspring1Route = candidateOffsprings[1]->chromR[indexA];

        offspring0Route.clear(); // Keeps capacity
        offspring1Route.clear();

        // Iterate over the route in parent A
        for (int c : parents.first->chromR[indexA]) {
            if (!clientsInSelectedBNotA.contains(c)) { // Use contains() in C++20
                offspring0Route.push_back(c);          // Only add to offspring 0 if not in set
            }
            offspring1Route.push_back(c); // Always add to offspring 1
        }
    }

    // Delete any remaining routes that still lived in offspring
    for (int r = nOfRoutesA; r < params->nbVehicles; r++) {
        candidateOffsprings[0]->chromR[r].clear();
        candidateOffsprings[1]->chromR[r].clear();
    }

    // Step 3: Insert unplanned clients (those that were in the removed routes of A but not the inserted routes of B)
    insertUnplannedTasks(candidateOffsprings[0], clientsInSelectedANotB);
    insertUnplannedTasks(candidateOffsprings[1], clientsInSelectedANotB);

    candidateOffsprings[0]->evaluateCompleteCost();
    candidateOffsprings[1]->evaluateCompleteCost();

    return candidateOffsprings[0]->myCostSol.penalizedCost < candidateOffsprings[1]->myCostSol.penalizedCost
               ? candidateOffsprings[0]
               : candidateOffsprings[1];
}

void Genetic::insertUnplannedTasks(Individual *offspring, const ankerl::unordered_dense::set<int> &unplannedTasks) {
    // Loop over all unplannedTasks
    for (int c : unplannedTasks) {
        // Get the earliest and latest possible arrival at the client
        const auto &client          = params->cli[c];
        int         earliestArrival = client.earliestArrival;
        int         latestArrival   = client.latestArrival;

        int                 bestDistance = INT_MAX;
        std::pair<int, int> bestLocation{-1, -1};

        // Loop over all routes
        for (int r = 0; r < params->nbVehicles; r++) {
            const auto &route = offspring->chromR[r];
            if (route.empty()) { continue; }

            // Check insertion at the start of the route
            int firstClient           = route[0];
            int newDistanceFromInsert = params->timeCost.get(c, firstClient);
            if (earliestArrival + newDistanceFromInsert < params->cli[firstClient].latestArrival) {
                int distanceDelta =
                    params->timeCost.get(0, c) + newDistanceFromInsert - params->timeCost.get(0, firstClient);
                if (distanceDelta < bestDistance) {
                    bestDistance = distanceDelta;
                    bestLocation = {r, 0};
                }
            }

            // Check insertion between existing clients in the route
            for (int i = 1; i < static_cast<int>(route.size()); i++) {
                int prevClient = route[i - 1];
                int nextClient = route[i];

                int newDistanceToInsert = params->timeCost.get(prevClient, c);
                newDistanceFromInsert   = params->timeCost.get(c, nextClient);

                if (params->cli[prevClient].earliestArrival + newDistanceToInsert < latestArrival &&
                    earliestArrival + newDistanceFromInsert < params->cli[nextClient].latestArrival) {
                    int distanceDelta =
                        newDistanceToInsert + newDistanceFromInsert - params->timeCost.get(prevClient, nextClient);
                    if (distanceDelta < bestDistance) {
                        bestDistance = distanceDelta;
                        bestLocation = {r, i};
                    }
                }
            }

            // Check insertion at the end of the route
            int lastClient          = route.back();
            int newDistanceToInsert = params->timeCost.get(lastClient, c);
            if (params->cli[lastClient].earliestArrival + newDistanceToInsert < latestArrival) {
                int distanceDelta =
                    newDistanceToInsert + params->timeCost.get(c, 0) - params->timeCost.get(lastClient, 0);
                if (distanceDelta < bestDistance) {
                    bestDistance = distanceDelta;
                    bestLocation = {r, static_cast<int>(route.size())};
                }
            }
        }

        // Insert the client at the best location
        if (bestLocation.first != -1) {
            offspring->chromR[bestLocation.first].insert(
                offspring->chromR[bestLocation.first].begin() + bestLocation.second, c);
        }
    }
}

Individual *Genetic::bestOfSREXAndOXCrossovers(std::pair<const Individual *, const Individual *> parents) {
    // Create two individuals, one with OX and one with SREX
    Individual *offspringOX   = crossoverOX(parents);
    Individual *offspringSREX = crossoverSREX(parents);

    // Return the best individual, based on penalizedCost
    return offspringOX->myCostSol.penalizedCost < offspringSREX->myCostSol.penalizedCost ? offspringOX : offspringSREX;
}

Genetic::Genetic(Params *params, Split *split, Population *population, HGSLocalSearch *localSearch)
    : params(params), split(split), population(population), localSearch(localSearch) {
    // After initializing the parameters of the Genetic object, also generate new individuals in the array
    // candidateOffsprings
    std::generate(candidateOffsprings.begin(), candidateOffsprings.end(), [&] { return new Individual(params); });
}

Genetic::~Genetic(void) {
    // Destruct the Genetic object by deleting all the individuals of the candidateOffsprings
    for (Individual *candidateOffspring : candidateOffsprings) { delete candidateOffspring; }
}
