#include "Genetic.h"

#include <time.h>

#include <algorithm>
#include <bitset>
#include <iterator>

#include "Individual.h"
#include "LocalSearch.h"
#include "Params.h"
#include "Population.h"
#include "Split.h"
#include "ankerl/unordered_dense.h"

// #include "SmartHGS.h"

#define FITNESS_THRESHOLD 0.5

void Genetic::run(int maxIterNonProd, int timeLimit) {
    if (params->nbClients == 1) return;

    const double mdmNURestarts = 0.05;
    const unsigned int repairThreshold =
        static_cast<unsigned int>(params->config.repairProbability);
    int nbRestarts = 0;
    int nbIterNonProd = 1;

    // Pre-calculate commonly used values
    const bool hasGranularGrowth = params->config.growNbGranularSize != 0;
    const bool hasPopulationGrowth = params->config.growPopulationSize != 0;
    const int granularIterCheck = params->config.growNbGranularAfterIterations;
    const int granularNonImprovementCheck =
        params->config.growNbGranularAfterNonImprovementIterations;
    const int populationIterCheck =
        params->config.growPopulationAfterIterations;
    const int populationNonImprovementCheck =
        params->config.growPopulationAfterNonImprovementIterations;

    // Main loop
    for (int nbIter = 0;
         nbIterNonProd <= maxIterNonProd && !params->isTimeLimitExceeded();
         ++nbIter) {
        // Selection and crossover combined in one step
        Individual *offspring = bestOfSREXAndOXCrossovers(
            population->getNonIdenticalParentsBinaryTournament());

        // Initial local search with standard penalties
        localSearch->run(offspring, params->penaltyCapacity,
                         params->penaltyTimeWarp);
        bool isNewBest = population->addIndividual(offspring, true);

        // Repair phase - only if not feasible and passes probability check
        if (!offspring->isFeasible && params->rng() % 100 < repairThreshold) {
            localSearch->run(offspring, params->penaltyCapacity * 10.0,
                             params->penaltyTimeWarp * 10.0);
            if (offspring->isFeasible) {
                isNewBest |= population->addIndividual(offspring, false);
            }
        }

        // Update non-productive iteration counter
        nbIterNonProd = isNewBest ? 1 : nbIterNonProd + 1;

        // Periodic operations - using bitwise operations for modulo
        if (!(nbIter & 0x63)) {  // equivalent to nbIter % 100 == 0
            population->managePenalties();
        }
        if (!(nbIter & 0x1FF)) {  // equivalent to nbIter % 500 == 0
            population->printState(nbIter, nbIterNonProd);
        }

        // Restart logic
        if (timeLimit != INT_MAX && nbIterNonProd == maxIterNonProd &&
            params->config.doRepeatUntilTimeLimit) {
            ++nbRestarts;
            double elapsedTime =
                static_cast<double>(clock() - params->startCPUTime) /
                CLOCKS_PER_SEC;
            int estimatedRestarts =
                std::min(static_cast<int>(params->config.timeLimit /
                                          (elapsedTime / nbRestarts)),
                         1000);
            population->mdmEliteMaxNonUpdatingRestarts =
                static_cast<int>(mdmNURestarts * estimatedRestarts);
            population->mineElite();
            population->restart();
            nbIterNonProd = 1;
        }

        // Parameter adjustments - combined checks
        if (hasGranularGrowth &&
            ((granularIterCheck > 0 && nbIter % granularIterCheck == 0) ||
             (granularNonImprovementCheck > 0 &&
              nbIterNonProd % granularNonImprovementCheck == 0))) {
            params->config.nbGranular += params->config.growNbGranularSize;
            params->SetCorrelatedVertices();
        }

        if (hasPopulationGrowth &&
            ((populationIterCheck > 0 && nbIter % populationIterCheck == 0) ||
             (populationNonImprovementCheck > 0 &&
              nbIterNonProd % populationNonImprovementCheck == 0))) {
            params->config.minimumPopulationSize +=
                params->config.growPopulationSize;
        }
    }
}

Individual *Genetic::crossoverOX(
    std::pair<const Individual *, const Individual *> parents) {
    // Generate two different random positions efficiently
    const int nbClients = params->nbClients;
    int start = params->rng() % nbClients;
    int end;
    do {
        end = params->rng() % nbClients;
    } while (end == start);

    // Perform crossovers in-place
    doOXcrossover(candidateOffsprings[2], parents, start, end);
    doOXcrossover(candidateOffsprings[3], parents, start, end);

    // Return best offspring using ternary operator
    return (candidateOffsprings[2]->myCostSol.penalizedCost <
            candidateOffsprings[3]->myCostSol.penalizedCost)
               ? candidateOffsprings[2]
               : candidateOffsprings[3];
}

void Genetic::doOXcrossover(
    Individual *result,
    std::pair<const Individual *, const Individual *> parents, int start,
    int end) {
    // Cache frequently accessed values.
    const int nbClients = params->nbClients;
    const Individual *parent1 = parents.first;
    const Individual *parent2 = parents.second;
    auto &child = result->chromT;
    child.resize(nbClients);  // Ensure proper size

    // Create a boolean array to mark which clients have been copied.
    std::vector<bool> used(nbClients, false);

    // --- Phase 1: Copy the segment from parent1 ---
    if (start <= end) {
        // Non-wrap-around: simply copy from index 'start' to 'end'.
        for (int i = start; i <= end; ++i) {
            child[i] = parent1->chromT[i];
            used[parent1->chromT[i]] = true;
        }
    } else {
        // Wrap-around: copy from 'start' to end of chromosome, then from index
        // 0 to 'end'.
        int i = start;
        do {
            child[i] = parent1->chromT[i];
            used[parent1->chromT[i]] = true;
            i = (i + 1) % nbClients;
        } while (i != (end + 1) % nbClients);
    }

    // --- Phase 2: Fill in remaining genes from parent2 ---
    // Start filling the child from position (end+1) mod nbClients.
    int posChild = (end + 1) % nbClients;
    // In parent2, start scanning from (end+1) mod nbClients.
    for (int posP2 = (end + 1) % nbClients;; posP2 = (posP2 + 1) % nbClients) {
        int candidate = parent2->chromT[posP2];
        if (!used[candidate]) {
            child[posChild] = candidate;
            posChild = (posChild + 1) % nbClients;
        }
        // Once we have scanned all positions in parent2, break out.
        if (posP2 == end) break;
    }

    // Finalize the individual using the Split algorithm.
    split->generalSplit(result, params->nbVehicles);
}

Individual *Genetic::crossoverSREX(
    std::pair<const Individual *, const Individual *> parents) {
    // --- Phase 1: Select Routes to Replace ---
    // Get the number of routes in each parent.
    int nOfRoutesA = parents.first->myCostSol.nbRoutes;
    int nOfRoutesB = parents.second->myCostSol.nbRoutes;

    // Choose a starting route index in Parent A.
    int startA = params->rng() % nOfRoutesA;
    // Determine the number of adjacent routes to move.
    int nOfMovedRoutes =
        (std::min(nOfRoutesA, nOfRoutesB) == 1)
            ? 1
            : params->rng() % (std::min(nOfRoutesA - 1, nOfRoutesB - 1)) + 1;
    // For Parent B, align the starting route index if possible.
    int startB = (startA < nOfRoutesB) ? startA : 0;

    // Collect clients from the selected routes.
    ankerl::unordered_dense::set<int> clientsInSelectedA;
    for (int r = 0; r < nOfMovedRoutes; r++) {
        int routeIdx = (startA + r) % nOfRoutesA;
        clientsInSelectedA.insert(parents.first->chromR[routeIdx].begin(),
                                  parents.first->chromR[routeIdx].end());
    }

    ankerl::unordered_dense::set<int> clientsInSelectedB;
    for (int r = 0; r < nOfMovedRoutes; r++) {
        int routeIdx = (startB + r) % nOfRoutesB;
        clientsInSelectedB.insert(parents.second->chromR[routeIdx].begin(),
                                  parents.second->chromR[routeIdx].end());
    }

    // --- Phase 2: Local Improvement by Shifting the Selected Routes ---
    bool improved = true;
    while (improved) {
        // Compute indices for adjacent routes (with modulo arithmetic).
        const int idxALeft = (startA - 1 + nOfRoutesA) % nOfRoutesA;
        const int idxARight = (startA + nOfMovedRoutes) % nOfRoutesA;
        const int idxALastMoved = (startA + nOfMovedRoutes - 1) % nOfRoutesA;
        const int idxBLeft = (startB - 1 + nOfRoutesB) % nOfRoutesB;
        const int idxBRight = (startB + nOfMovedRoutes) % nOfRoutesB;
        const int idxBLastMoved = (startB - 1 + nOfMovedRoutes) % nOfRoutesB;

        // Helper lambda for fast client counting.
        auto countClients =
            [](const std::vector<int> &route,
               const ankerl::unordered_dense::set<int> &clientSet,
               bool countOutside) -> int {
            int count = 0;
            for (int c : route) {
                // If countOutside is true, count clients NOT in the set.
                count += (clientSet.contains(c) != countOutside);
            }
            return count;
        };

        // Cache parent pointers and their routes.
        const auto &parentA = parents.first;
        const auto &parentB = parents.second;
        const auto &routesA = parentA->chromR;
        const auto &routesB = parentB->chromR;

        // Batch count differences at boundaries.
        const int leftAFirst =
            countClients(routesA[idxALeft], clientsInSelectedB, true);
        const int leftASecond =
            countClients(routesA[idxALastMoved], clientsInSelectedB, true);
        const int rightAFirst =
            countClients(routesA[idxARight], clientsInSelectedB, true);
        const int rightASecond =
            countClients(routesA[startA], clientsInSelectedB, true);

        const int leftBFirst =
            countClients(routesB[idxBLastMoved], clientsInSelectedA, false);
        const int leftBSecond =
            countClients(routesB[idxBLeft], clientsInSelectedA, false);
        const int rightBFirst =
            countClients(routesB[startB], clientsInSelectedA, false);
        const int rightBSecond =
            countClients(routesB[idxBRight], clientsInSelectedA, false);

        const int differenceALeft = leftAFirst - leftASecond;
        const int differenceARight = rightAFirst - rightASecond;
        const int differenceBLeft = leftBFirst - leftBSecond;
        const int differenceBRight = rightBFirst - rightBSecond;

        const int bestDifference =
            std::min({differenceALeft, differenceARight, differenceBLeft,
                      differenceBRight});

        if (bestDifference < 0) {
            // Update the selection boundaries based on which difference is
            // best.
            if (bestDifference == differenceALeft) {
                // Shift selection to the left in Parent A.
                int removeIdx = (startA + nOfMovedRoutes - 1) % nOfRoutesA;
                int addIdx = (startA - 1 + nOfRoutesA) % nOfRoutesA;
                for (int c : routesA[removeIdx]) clientsInSelectedA.erase(c);
                startA = addIdx;
                clientsInSelectedA.insert(routesA[startA].begin(),
                                          routesA[startA].end());
            } else if (bestDifference == differenceARight) {
                // Shift selection to the right in Parent A.
                int removeIdx = startA;
                for (int c : routesA[removeIdx]) clientsInSelectedA.erase(c);
                startA = (startA + 1) % nOfRoutesA;
                int addIdx = (startA + nOfMovedRoutes - 1) % nOfRoutesA;
                clientsInSelectedA.insert(routesA[addIdx].begin(),
                                          routesA[addIdx].end());
            } else if (bestDifference == differenceBLeft) {
                // Shift selection to the left in Parent B.
                int removeIdx = (startB + nOfMovedRoutes - 1) % nOfRoutesB;
                int addIdx = (startB - 1 + nOfRoutesB) % nOfRoutesB;
                for (int c : routesB[removeIdx]) clientsInSelectedB.erase(c);
                startB = addIdx;
                clientsInSelectedB.insert(routesB[startB].begin(),
                                          routesB[startB].end());
            } else {  // bestDifference == differenceBRight
                // Shift selection to the right in Parent B.
                int removeIdx = startB;
                for (int c : routesB[removeIdx]) clientsInSelectedB.erase(c);
                startB = (startB + 1) % nOfRoutesB;
                int addIdx = (startB + nOfMovedRoutes - 1) % nOfRoutesB;
                clientsInSelectedB.insert(routesB[addIdx].begin(),
                                          routesB[addIdx].end());
            }
        } else {
            improved = false;
        }
    }

    // --- Phase 3: Identify Differences Between Selected Routes ---
    std::vector<int> clientsInSelectedANotBVec;
    std::vector<int> clientsInSelectedBNotAVec;
    auto getSortedVec = [](const ankerl::unordered_dense::set<int> &s) {
        std::vector<int> vec(s.begin(), s.end());
        pdqsort(vec.begin(), vec.end());
        return vec;
    };

    const auto sortedVecA = getSortedVec(clientsInSelectedA);
    const auto sortedVecB = getSortedVec(clientsInSelectedB);

    // Compute set differences: A - B.
    std::set_difference(sortedVecA.begin(), sortedVecA.end(),
                        sortedVecB.begin(), sortedVecB.end(),
                        std::back_inserter(clientsInSelectedANotBVec));
    // And B - A.
    std::set_difference(sortedVecB.begin(), sortedVecB.end(),
                        sortedVecA.begin(), sortedVecA.end(),
                        std::back_inserter(clientsInSelectedBNotAVec));

    // Convert one of the differences into an unordered set for fast lookup.
    ankerl::unordered_dense::set<int> clientsInSelectedBNotA(
        clientsInSelectedBNotAVec.begin(), clientsInSelectedBNotAVec.end());

    // --- Phase 4: Build Offspring by Replacing Selected Routes ---
    // Replace the selected routes in Parent A with those from Parent B.
    for (int r = 0; r < nOfMovedRoutes; r++) {
        int indexA = (startA + r) % nOfRoutesA;
        int indexB = (startB + r) % nOfRoutesB;
        auto &offspring0Route = candidateOffsprings[0]->chromR[indexA];
        auto &offspring1Route = candidateOffsprings[1]->chromR[indexA];

        offspring0Route.clear();
        offspring1Route.clear();

        // For offspring 0: Copy the entire route from Parent B.
        for (int c : parents.second->chromR[indexB]) {
            offspring0Route.push_back(c);
        }
        // For offspring 1: Copy only clients not in clientsInSelectedBNotA.
        for (int c : parents.second->chromR[indexB]) {
            if (!clientsInSelectedBNotA.contains(c))
                offspring1Route.push_back(c);
        }
    }

    // For routes not replaced, inherit from Parent A.
    for (int r = nOfMovedRoutes; r < nOfRoutesA; r++) {
        int indexA = (startA + r) % nOfRoutesA;
        auto &offspring0Route = candidateOffsprings[0]->chromR[indexA];
        auto &offspring1Route = candidateOffsprings[1]->chromR[indexA];
        offspring0Route.clear();
        offspring1Route.clear();
        for (int c : parents.first->chromR[indexA]) {
            if (!clientsInSelectedBNotA.contains(c))
                offspring0Route.push_back(c);
            offspring1Route.push_back(c);
        }
    }

    // Clear any routes beyond Parent A's route count.
    for (int r = nOfRoutesA; r < params->nbVehicles; r++) {
        candidateOffsprings[0]->chromR[r].clear();
        candidateOffsprings[1]->chromR[r].clear();
    }

    // --- Phase 5: Insert Unplanned Clients ---
    // Insert clients that were in Parent A's removed routes but not in the
    // inserted routes.
    insertUnplannedTasks(candidateOffsprings[0], clientsInSelectedANotBVec);
    insertUnplannedTasks(candidateOffsprings[1], clientsInSelectedANotBVec);

    candidateOffsprings[0]->evaluateCompleteCost();
    candidateOffsprings[1]->evaluateCompleteCost();

    // Return the offspring with the lower penalized cost.
    return (candidateOffsprings[0]->myCostSol.penalizedCost <
            candidateOffsprings[1]->myCostSol.penalizedCost)
               ? candidateOffsprings[0]
               : candidateOffsprings[1];
}

void Genetic::insertUnplannedTasks(Individual *offspring,
                                   const std::vector<int> &unplannedTasks) {
    // Collect indices of non-empty routes.
    std::vector<int> nonEmptyRoutes;
    nonEmptyRoutes.reserve(params->nbVehicles);
    for (int r = 0; r < params->nbVehicles; r++) {
        if (!offspring->chromR[r].empty()) {
            nonEmptyRoutes.push_back(r);
        }
    }

    const auto &timeCost = params->timeCost;
    const auto &clients = params->cli;

    // Structure to record an insertion candidate.
    struct InsertionPoint {
        int routeIdx;
        int position;  // Position in the route where the client will be
                       // inserted.
        int cost;      // Additional cost for insertion.
        bool operator<(const InsertionPoint &other) const {
            return cost < other.cost;
        }
    };

    // Reserve space for potential insertion points.
    std::vector<InsertionPoint> insertionPoints;
    insertionPoints.reserve(100);

    // Process each unplanned task.
    for (int client : unplannedTasks) {
        const auto &clientData = clients[client];
        const int clientEarliest = clientData.earliestArrival;
        const int clientLatest = clientData.latestArrival;

        insertionPoints.clear();  // Reset candidates for current client.

        // Evaluate each non-empty route.
        for (int routeIdx : nonEmptyRoutes) {
            const auto &route = offspring->chromR[routeIdx];
            const int routeSize = static_cast<int>(route.size());

            // --- Check insertion at the beginning of the route ---
            // Consider inserting before the first client.
            int firstClient = route[0];
            int costToClient = timeCost.get(0, client);
            int costClientToFirst = timeCost.get(client, firstClient);
            int origCost = timeCost.get(0, firstClient);
            // Check feasibility: arrival at first client should not exceed its
            // latest arrival.
            if (clientEarliest + costClientToFirst <
                clients[firstClient].latestArrival) {
                insertionPoints.push_back(
                    {routeIdx, 0, costToClient + costClientToFirst - origCost});
            }

            // --- Check insertion between clients ---
            int prevClient = route[0];
            int prevEarliest = clients[prevClient].earliestArrival;
            for (int pos = 1; pos < routeSize; pos++) {
                int nextClient = route[pos];
                int costFromPrev = timeCost.get(prevClient, client);
                int costToNext = timeCost.get(client, nextClient);
                int origCostBetween = timeCost.get(prevClient, nextClient);
                // Feasibility: arrival after previous client and before next
                // client's latest.
                if (prevEarliest + costFromPrev < clientLatest &&
                    clientEarliest + costToNext <
                        clients[nextClient].latestArrival) {
                    insertionPoints.push_back(
                        {routeIdx, pos,
                         costFromPrev + costToNext - origCostBetween});
                }
                prevClient = nextClient;
                prevEarliest = clients[prevClient].earliestArrival;
            }

            // --- Check insertion at the end of the route ---
            int lastClient = route.back();
            int costFromLast = timeCost.get(lastClient, client);
            // Feasibility: arrival at client from last client within client's
            // time window.
            if (clients[lastClient].earliestArrival + costFromLast <
                clientLatest) {
                int costFromClientToDepot = timeCost.get(client, 0);
                int origCostFromLastToDepot = timeCost.get(lastClient, 0);
                insertionPoints.push_back({routeIdx, routeSize,
                                           costFromLast +
                                               costFromClientToDepot -
                                               origCostFromLastToDepot});
            }
        }

        // If any valid insertion points exist, choose the one with minimum
        // cost.
        if (!insertionPoints.empty()) {
            auto bestPoint = std::min_element(insertionPoints.begin(),
                                              insertionPoints.end());
            auto &targetRoute = offspring->chromR[bestPoint->routeIdx];
            targetRoute.insert(targetRoute.begin() + bestPoint->position,
                               client);
        }
    }
}

Individual *Genetic::bestOfSREXAndOXCrossovers(
    std::pair<const Individual *, const Individual *> parents) {
    Individual *offspringOX = crossoverOX(parents);
    Individual *offspringSREX = crossoverSREX(parents);

    return offspringOX->myCostSol.penalizedCost <
                   offspringSREX->myCostSol.penalizedCost
               ? offspringOX
               : offspringSREX;
}

Genetic::Genetic(Params *params, Split *split, Population *population,
                 HGSLocalSearch *localSearch)
    : params(params),
      split(split),
      population(population),
      localSearch(localSearch) {
    // After initializing the parameters of the Genetic object, also generate
    // new individuals in the array candidateOffsprings
    std::generate(candidateOffsprings.begin(), candidateOffsprings.end(),
                  [&] { return new Individual(params); });
}

Genetic::~Genetic(void) {
    // Destruct the Genetic object by deleting all the individuals of the
    // candidateOffsprings
    for (Individual *candidateOffspring : candidateOffsprings) {
        delete candidateOffspring;
    }
}
