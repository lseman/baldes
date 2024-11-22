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

// #include "SmartHGS.h"

#define FITNESS_THRESHOLD 0.5

void Genetic::run(int maxIterNonProd, int timeLimit) {
    if (params->nbClients == 1) return;

    const double       mdmNURestarts   = 0.05;
    const unsigned int repairThreshold = static_cast<unsigned int>(params->config.repairProbability);
    int                nbRestarts      = 0;
    int                nbIterNonProd   = 1;

    // Pre-calculate commonly used values
    const bool hasGranularGrowth             = params->config.growNbGranularSize != 0;
    const bool hasPopulationGrowth           = params->config.growPopulationSize != 0;
    const int  granularIterCheck             = params->config.growNbGranularAfterIterations;
    const int  granularNonImprovementCheck   = params->config.growNbGranularAfterNonImprovementIterations;
    const int  populationIterCheck           = params->config.growPopulationAfterIterations;
    const int  populationNonImprovementCheck = params->config.growPopulationAfterNonImprovementIterations;

    // Main loop
    for (int nbIter = 0; nbIterNonProd <= maxIterNonProd && !params->isTimeLimitExceeded(); ++nbIter) {
        // Selection and crossover combined in one step
        Individual *offspring = bestOfSREXAndOXCrossovers(population->getNonIdenticalParentsBinaryTournament());

        // Initial local search with standard penalties
        localSearch->run(offspring, params->penaltyCapacity, params->penaltyTimeWarp);
        bool isNewBest = population->addIndividual(offspring, true);

        // Repair phase - only if not feasible and passes probability check
        if (!offspring->isFeasible && params->rng() % 100 < repairThreshold) {
            localSearch->run(offspring, params->penaltyCapacity * 10.0, params->penaltyTimeWarp * 10.0);
            if (offspring->isFeasible) { isNewBest |= population->addIndividual(offspring, false); }
        }

        // Update non-productive iteration counter
        nbIterNonProd = isNewBest ? 1 : nbIterNonProd + 1;

        // Periodic operations - using bitwise operations for modulo
        if (!(nbIter & 0x63)) { // equivalent to nbIter % 100 == 0
            population->managePenalties();
        }
        if (!(nbIter & 0x1FF)) { // equivalent to nbIter % 500 == 0
            population->printState(nbIter, nbIterNonProd);
        }

        // Restart logic
        if (timeLimit != INT_MAX && nbIterNonProd == maxIterNonProd && params->config.doRepeatUntilTimeLimit) {
            ++nbRestarts;
            double elapsedTime = static_cast<double>(clock() - params->startCPUTime) / CLOCKS_PER_SEC;
            int    estimatedRestarts =
                std::min(static_cast<int>(params->config.timeLimit / (elapsedTime / nbRestarts)), 1000);
            population->mdmEliteMaxNonUpdatingRestarts = static_cast<int>(mdmNURestarts * estimatedRestarts);
            population->mineElite();
            population->restart();
            nbIterNonProd = 1;
        }

        // Parameter adjustments - combined checks
        if (hasGranularGrowth &&
            ((granularIterCheck > 0 && nbIter % granularIterCheck == 0) ||
             (granularNonImprovementCheck > 0 && nbIterNonProd % granularNonImprovementCheck == 0))) {
            params->config.nbGranular += params->config.growNbGranularSize;
            params->SetCorrelatedVertices();
        }

        if (hasPopulationGrowth &&
            ((populationIterCheck > 0 && nbIter % populationIterCheck == 0) ||
             (populationNonImprovementCheck > 0 && nbIterNonProd % populationNonImprovementCheck == 0))) {
            params->config.minimumPopulationSize += params->config.growPopulationSize;
        }
    }
}

Individual *Genetic::crossoverOX(std::pair<const Individual *, const Individual *> parents) {
    // Generate two different random positions efficiently
    const int nbClients = params->nbClients;
    int       start     = params->rng() % nbClients;
    int       end;
    do { end = params->rng() % nbClients; } while (end == start);

    // Perform crossovers in-place
    doOXcrossover(candidateOffsprings[2], parents, start, end);
    doOXcrossover(candidateOffsprings[3], parents, start, end);

    // Return best offspring using ternary operator
    return (candidateOffsprings[2]->myCostSol.penalizedCost < candidateOffsprings[3]->myCostSol.penalizedCost)
               ? candidateOffsprings[2]
               : candidateOffsprings[3];
}

void Genetic::doOXcrossover(Individual *result, std::pair<const Individual *, const Individual *> parents, int start,
                            int end) {
    // Cache frequently accessed values
    const int   nbClients = params->nbClients;
    const auto *parent1   = parents.first;
    const auto *parent2   = parents.second;
    auto       &chromT    = result->chromT; // Avoid repeated array access

    // Pre-calculate modulo values
    const int idxEnd = (end + 1) % nbClients;

    // Use bitset for O(1) lookup
    std::bitset<N_SIZE> freqClient;
    freqClient.reset(); // Faster than initializing to 0s

    // First phase: Copy segment from parent1 (previously parent A)
    if (start <= end) {
        // No wrap-around case - can use straight memcpy
        const int segmentLength = end - start + 1;
        std::memcpy(&chromT[start], &parent1->chromT[start], segmentLength * sizeof(int));
        for (int j = start; j <= end; ++j) { freqClient.set(parent1->chromT[j]); }
    } else {
        // Handle wrap-around case
        int j = start;
        do {
            const int elem = parent1->chromT[j];
            chromT[j]      = elem;
            freqClient.set(elem);
            j = (j + 1) % nbClients;
        } while (j != idxEnd);
    }

    // Second phase: Fill remaining elements from parent2 (previously parent B)
    // Use separate counter to avoid modulo operations
    int j = idxEnd;
    for (int i = idxEnd; j != start; i = (i + 1) % nbClients) {
        const int candidate = parent2->chromT[i];
        if (!freqClient.test(candidate)) {
            chromT[j] = candidate;
            j         = (j + 1) % nbClients;
        }
    }

    // Complete the individual using Split algorithm
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
        // Cache route indices with efficient modulo
        const int idxALeft      = (startA - 1 + nOfRoutesA) % nOfRoutesA;
        const int idxARight     = (startA + nOfMovedRoutes) % nOfRoutesA;
        const int idxALastMoved = (startA + nOfMovedRoutes - 1) % nOfRoutesA;
        const int idxBLeft      = (startB - 1 + nOfRoutesB) % nOfRoutesB;
        const int idxBRight     = (startB + nOfMovedRoutes) % nOfRoutesB;
        const int idxBLastMoved = (startB - 1 + nOfMovedRoutes) % nOfRoutesB;

        // Fast client counting using contains() instead of find()
        auto countClients = [](const std::vector<int> &route, const ankerl::unordered_dense::set<int> &clientSet,
                               bool countOutside) {
            int count = 0;
            for (int c : route) { count += (clientSet.contains(c) != countOutside); }
            return count;
        };

        // Cache parent pointers and commonly accessed routes
        const auto &parentA = parents.first;
        const auto &parentB = parents.second;
        const auto &routesA = parentA->chromR;
        const auto &routesB = parentB->chromR;

        // Calculate all counts in one batch for better cache utilization
        const int leftAFirst   = countClients(routesA[idxALeft], clientsInSelectedB, true);
        const int leftASecond  = countClients(routesA[idxALastMoved], clientsInSelectedB, true);
        const int rightAFirst  = countClients(routesA[idxARight], clientsInSelectedB, true);
        const int rightASecond = countClients(routesA[startA], clientsInSelectedB, true);

        const int leftBFirst   = countClients(routesB[idxBLastMoved], clientsInSelectedA, false);
        const int leftBSecond  = countClients(routesB[idxBLeft], clientsInSelectedA, false);
        const int rightBFirst  = countClients(routesB[startB], clientsInSelectedA, false);
        const int rightBSecond = countClients(routesB[idxBRight], clientsInSelectedA, false);

        // Calculate the differences
        const int differenceALeft  = leftAFirst - leftASecond;
        const int differenceARight = rightAFirst - rightASecond;
        const int differenceBLeft  = leftBFirst - leftBSecond;
        const int differenceBRight = rightBFirst - rightBSecond;

        const int bestDifference = std::min({differenceALeft, differenceARight, differenceBLeft, differenceBRight});
        // Avoiding infinite loop by adding a guard in case nothing changes
        if (bestDifference < 0) {
            // Cache frequently used values and references
            const auto &routesA = parents.first->chromR;
            const auto &routesB = parents.second->chromR;

            // Determine which difference matches and perform corresponding update
            if (bestDifference == differenceALeft) {
                const int removeIdx = (startA + nOfMovedRoutes - 1) % nOfRoutesA;
                const int addIdx    = (startA - 1 + nOfRoutesA) % nOfRoutesA;

                // Batch remove and insert for better cache usage
                for (int c : routesA[removeIdx]) {
                    clientsInSelectedA.erase(c); // erase() is safe even if element doesn't exist
                }
                startA = addIdx;
                clientsInSelectedA.insert(routesA[startA].begin(), routesA[startA].end());

            } else if (bestDifference == differenceARight) {
                const int removeIdx = startA;

                // Batch remove
                for (int c : routesA[removeIdx]) { clientsInSelectedA.erase(c); }
                startA = (startA + 1) % nOfRoutesA;

                // Batch insert
                const int addIdx = (startA + nOfMovedRoutes - 1) % nOfRoutesA;
                clientsInSelectedA.insert(routesA[addIdx].begin(), routesA[addIdx].end());

            } else if (bestDifference == differenceBLeft) {
                const int removeIdx = (startB + nOfMovedRoutes - 1) % nOfRoutesB;
                const int addIdx    = (startB - 1 + nOfRoutesB) % nOfRoutesB;

                // Batch remove and insert
                for (int c : routesB[removeIdx]) { clientsInSelectedB.erase(c); }
                startB = addIdx;
                clientsInSelectedB.insert(routesB[startB].begin(), routesB[startB].end());

            } else { // bestDifference == differenceBRight
                const int removeIdx = startB;

                // Batch remove
                for (int c : routesB[removeIdx]) { clientsInSelectedB.erase(c); }
                startB = (startB + 1) % nOfRoutesB;

                // Batch insert
                const int addIdx = (startB + nOfMovedRoutes - 1) % nOfRoutesB;
                clientsInSelectedB.insert(routesB[addIdx].begin(), routesB[addIdx].end());
            }

        } else {
            improved = false;
        }
    }

    // Identify differences between route sets
    std::vector<int> clientsInSelectedANotBVec;
    std::vector<int> clientsInSelectedBNotAVec;

    // Convert sets to sorted vectors for efficient set operations
    auto getSortedVec = [](const ankerl::unordered_dense::set<int> &set) {
        std::vector<int> vec(set.begin(), set.end());
        pdqsort(vec.begin(), vec.end());
        return vec;
    };

    // Create sorted vectors once and reuse
    const auto sortedVecA = getSortedVec(clientsInSelectedA);
    const auto sortedVecB = getSortedVec(clientsInSelectedB);

    // Calculate A - B
    clientsInSelectedANotBVec.clear();
    clientsInSelectedANotBVec.reserve(clientsInSelectedA.size());
    std::set_difference(sortedVecA.begin(), sortedVecA.end(), sortedVecB.begin(), sortedVecB.end(),
                        std::back_inserter(clientsInSelectedANotBVec));

    // Calculate B - A
    clientsInSelectedBNotAVec.clear();
    clientsInSelectedBNotAVec.reserve(clientsInSelectedB.size());
    std::set_difference(sortedVecB.begin(), sortedVecB.end(), sortedVecA.begin(), sortedVecA.end(),
                        std::back_inserter(clientsInSelectedBNotAVec));

    // Convert vector to unordered_set
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
    insertUnplannedTasks(candidateOffsprings[0], clientsInSelectedANotBVec);
    insertUnplannedTasks(candidateOffsprings[1], clientsInSelectedANotBVec);

    candidateOffsprings[0]->evaluateCompleteCost();
    candidateOffsprings[1]->evaluateCompleteCost();

    return candidateOffsprings[0]->myCostSol.penalizedCost < candidateOffsprings[1]->myCostSol.penalizedCost
               ? candidateOffsprings[0]
               : candidateOffsprings[1];
}

void Genetic::insertUnplannedTasks(Individual *offspring, const std::vector<int> &unplannedTasks) {
    // Pre-calculate routes that aren't empty to avoid checking empty routes repeatedly
    std::vector<int> nonEmptyRoutes;
    nonEmptyRoutes.reserve(params->nbVehicles);
    for (int r = 0; r < params->nbVehicles; r++) {
        if (!offspring->chromR[r].empty()) { nonEmptyRoutes.push_back(r); }
    }

    // Cache commonly used parameters
    const auto &timeCost = params->timeCost;
    const auto &clients  = params->cli;

    // Structure to hold insertion costs
    struct InsertionPoint {
        int routeIdx;
        int position;
        int cost;

        bool operator<(const InsertionPoint &other) const { return cost < other.cost; }
    };

    // Pre-allocate vector for potential insertion points
    std::vector<InsertionPoint> insertionPoints;
    insertionPoints.reserve(100); // Reasonable initial capacity

    // Process each unplanned task
    for (int client : unplannedTasks) {
        const auto &clientData      = clients[client];
        const int   earliestArrival = clientData.earliestArrival;
        const int   latestArrival   = clientData.latestArrival;

        insertionPoints.clear();

        // Check all non-empty routes
        for (int routeIdx : nonEmptyRoutes) {
            const auto &route     = offspring->chromR[routeIdx];
            const int   routeSize = static_cast<int>(route.size());

            // Check insertion at start
            const int firstClient     = route[0];
            const int distToFirst     = timeCost.get(client, firstClient);
            const int origDistToFirst = timeCost.get(0, firstClient);

            if (earliestArrival + distToFirst < clients[firstClient].latestArrival) {
                const int deltaCost = timeCost.get(0, client) + distToFirst - origDistToFirst;
                insertionPoints.push_back({routeIdx, 0, deltaCost});
            }

            // Check insertions between clients
            // Use previous client data to avoid repeated lookups
            int prevClient          = route[0];
            int prevEarliestArrival = clients[prevClient].earliestArrival;

            for (int pos = 1; pos < routeSize; pos++) {
                const int nextClient   = route[pos];
                const int distFromPrev = timeCost.get(prevClient, client);
                const int distToNext   = timeCost.get(client, nextClient);
                const int origDist     = timeCost.get(prevClient, nextClient);

                if (prevEarliestArrival + distFromPrev < latestArrival &&
                    earliestArrival + distToNext < clients[nextClient].latestArrival) {
                    const int deltaCost = distFromPrev + distToNext - origDist;
                    insertionPoints.push_back({routeIdx, pos, deltaCost});
                }

                prevClient          = nextClient;
                prevEarliestArrival = clients[prevClient].earliestArrival;
            }

            // Check insertion at end
            const int lastClient   = route.back();
            const int distFromLast = timeCost.get(lastClient, client);

            if (clients[lastClient].earliestArrival + distFromLast < latestArrival) {
                const int deltaCost = distFromLast + timeCost.get(client, 0) - timeCost.get(lastClient, 0);
                insertionPoints.push_back({routeIdx, routeSize, deltaCost});
            }
        }

        // Find best insertion point
        if (!insertionPoints.empty()) {
            const auto bestPoint   = std::min_element(insertionPoints.begin(), insertionPoints.end());
            auto      &targetRoute = offspring->chromR[bestPoint->routeIdx];
            targetRoute.insert(targetRoute.begin() + bestPoint->position, client);
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
