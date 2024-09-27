#include <algorithm>
#include <iterator>
#include <time.h>
#include <unordered_set>

#include "Genetic.h"
#include "Individual.h"
#include "LocalSearch.h"
#include "Params.h"
#include "Population.h"
#include "Split.h"

void Genetic::run(int maxIterNonProd, int timeLimit) {
    if (params->nbClients == 1) {
        // Edge case: with 1 client, crossover will fail, genetic algorithm makes no sense
        return;
    }
    // Do iterations of the Genetic Algorithm, until more then maxIterNonProd consecutive iterations without improvement
    // or a time limit (in seconds) is reached
    int nbIterNonProd = 1;
    for (int nbIter = 0; nbIterNonProd <= maxIterNonProd && !params->isTimeLimitExceeded(); nbIter++) {
        /* SELECTION AND CROSSOVER */
        // First select parents using getNonIdenticalParentsBinaryTournament
        // Then use the selected parents to create new individuals using OX and SREX
        // Finally select the best new individual based on bestOfSREXAndOXCrossovers
        Individual *offspring = bestOfSREXAndOXCrossovers(population->getNonIdenticalParentsBinaryTournament());

        /* LOCAL SEARCH */
        // Run the Local Search on the new individual
        localSearch->run(offspring, params->penaltyCapacity, params->penaltyTimeWarp);
        // Check if the new individual is the best feasible individual of the population, based on penalizedCost
        bool isNewBest = population->addIndividual(offspring, true);
        // In case of infeasibility, repair the individual with a certain probability
        if (!offspring->isFeasible && params->rng() % 100 < (unsigned int)params->config.repairProbability) {
            // Run the Local Search again, but with penalties for infeasibilities multiplied by 10
            localSearch->run(offspring, params->penaltyCapacity * 10., params->penaltyTimeWarp * 10.);
            // If the individual is feasible now, check if it is the best feasible individual of the population, based
            // on penalizedCost and add it to the population If the individual is not feasible now, it is not added to
            // the population
            if (offspring->isFeasible) { isNewBest = (population->addIndividual(offspring, false) || isNewBest); }
        }

        /* TRACKING THE NUMBER OF ITERATIONS SINCE LAST SOLUTION IMPROVEMENT */
        if (isNewBest) {
            nbIterNonProd = 1;
        } else
            nbIterNonProd++;

        /* DIVERSIFICATION, PENALTY MANAGEMENT AND TRACES */
        // Update the penaltyTimeWarp and penaltyCapacity every 100 iterations
        if (nbIter % 100 == 0) { population->managePenalties(); }
        // Print the state of the population every 500 iterations
        if (nbIter % 500 == 0) { population->printState(nbIter, nbIterNonProd); }
        // Log the current population to a .csv file every logPoolInterval iterations (if logPoolInterval is not 0)
        if (params->config.logPoolInterval > 0 && nbIter % params->config.logPoolInterval == 0) {
            population->exportPopulation(nbIter, params->config.pathSolution + ".log.csv");
        }

        /* FOR TESTS INVOLVING SUCCESSIVE RUNS UNTIL A TIME LIMIT: WE RESET THE ALGORITHM/POPULATION EACH TIME
         * maxIterNonProd IS ATTAINED*/
        if (timeLimit != INT_MAX && nbIterNonProd == maxIterNonProd && params->config.doRepeatUntilTimeLimit) {
            population->restart();
            nbIterNonProd = 1;
        }

        /* OTHER PARAMETER CHANGES*/
        // Increase the nbGranular by growNbGranularSize (and set the correlated vertices again) every certain number of
        // iterations, if growNbGranularSize is greater than 0
        if (nbIter > 0 && params->config.growNbGranularSize != 0 &&
            ((params->config.growNbGranularAfterIterations > 0 &&
              nbIter % params->config.growNbGranularAfterIterations == 0) ||
             (params->config.growNbGranularAfterNonImprovementIterations > 0 &&
              nbIterNonProd % params->config.growNbGranularAfterNonImprovementIterations == 0))) {
            // Note: changing nbGranular also changes how often the order is reshuffled
            params->config.nbGranular += params->config.growNbGranularSize;
            params->SetCorrelatedVertices();
        }

        // Increase the minimumPopulationSize by growPopulationSize every certain number of iterations, if
        // growPopulationSize is greater than 0
        if (nbIter > 0 && params->config.growPopulationSize != 0 &&
            ((params->config.growPopulationAfterIterations > 0 &&
              nbIter % params->config.growPopulationAfterIterations == 0) ||
             (params->config.growPopulationAfterNonImprovementIterations > 0 &&
              nbIterNonProd % params->config.growPopulationAfterNonImprovementIterations == 0))) {
            // This will automatically adjust after some iterations
            params->config.minimumPopulationSize += params->config.growPopulationSize;
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
    // Frequency vector to track the clients which have been inserted already
    std::vector<bool> freqClient(params->nbClients + 1, false);

    // Copy in place the elements from start to end
    int j = start;
    int nbClients = params->nbClients;
    int idxEnd = (end + 1) % nbClients;

    // First loop: directly copy the segment from parent A
    while (j != idxEnd) {
        result->chromT[j] = parents.first->chromT[j];
        freqClient[result->chromT[j]] = true;
        j = (j + 1 == nbClients) ? 0 : j + 1; // Reduce modulo operations
    }

    // Fill the remaining elements from parent B, skipping already copied ones
    int i = (end + 1) % nbClients;
    while (j != start) {
        int temp = parents.second->chromT[i];
        if (!freqClient[temp]) {
            result->chromT[j] = temp;
            j = (j + 1 == nbClients) ? 0 : j + 1; // Reduce modulo operations
        }
        i = (i + 1 == nbClients) ? 0 : i + 1; // Reduce modulo operations
    }

    // Completing the individual with the Split algorithm
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

    std::unordered_set<int> clientsInSelectedA;
    for (int r = 0; r < nOfMovedRoutes; r++) {
        // Insert the first
        clientsInSelectedA.insert(parents.first->chromR[(startA + r) % nOfRoutesA].begin(),
                                  parents.first->chromR[(startA + r) % nOfRoutesA].end());
    }

    std::unordered_set<int> clientsInSelectedB;
    for (int r = 0; r < nOfMovedRoutes; r++) {
        clientsInSelectedB.insert(parents.second->chromR[(startB + r) % nOfRoutesB].begin(),
                                  parents.second->chromR[(startB + r) % nOfRoutesB].end());
    }

    bool improved = true;
    while (improved) {
        // Difference for moving 'left' in parent A
        const int differenceALeft =
            static_cast<int>(std::count_if(
                parents.first->chromR[(startA - 1 + nOfRoutesA) % nOfRoutesA].begin(),
                parents.first->chromR[(startA - 1 + nOfRoutesA) % nOfRoutesA].end(),
                [&clientsInSelectedB](int c) { return clientsInSelectedB.find(c) == clientsInSelectedB.end(); })) -
            static_cast<int>(std::count_if(
                parents.first->chromR[(startA + nOfMovedRoutes - 1) % nOfRoutesA].begin(),
                parents.first->chromR[(startA + nOfMovedRoutes - 1) % nOfRoutesA].end(),
                [&clientsInSelectedB](int c) { return clientsInSelectedB.find(c) == clientsInSelectedB.end(); }));

        // Difference for moving 'right' in parent A
        const int differenceARight =
            static_cast<int>(std::count_if(
                parents.first->chromR[(startA + nOfMovedRoutes) % nOfRoutesA].begin(),
                parents.first->chromR[(startA + nOfMovedRoutes) % nOfRoutesA].end(),
                [&clientsInSelectedB](int c) { return clientsInSelectedB.find(c) == clientsInSelectedB.end(); })) -
            static_cast<int>(std::count_if(
                parents.first->chromR[startA].begin(), parents.first->chromR[startA].end(),
                [&clientsInSelectedB](int c) { return clientsInSelectedB.find(c) == clientsInSelectedB.end(); }));

        // Difference for moving 'left' in parent B
        const int differenceBLeft =
            static_cast<int>(std::count_if(
                parents.second->chromR[(startB - 1 + nOfMovedRoutes) % nOfRoutesB].begin(),
                parents.second->chromR[(startB - 1 + nOfMovedRoutes) % nOfRoutesB].end(),
                [&clientsInSelectedA](int c) { return clientsInSelectedA.find(c) != clientsInSelectedA.end(); })) -
            static_cast<int>(std::count_if(
                parents.second->chromR[(startB - 1 + nOfRoutesB) % nOfRoutesB].begin(),
                parents.second->chromR[(startB - 1 + nOfRoutesB) % nOfRoutesB].end(),
                [&clientsInSelectedA](int c) { return clientsInSelectedA.find(c) != clientsInSelectedA.end(); }));

        // Difference for moving 'right' in parent B
        const int differenceBRight =
            static_cast<int>(std::count_if(
                parents.second->chromR[startB].begin(), parents.second->chromR[startB].end(),
                [&clientsInSelectedA](int c) { return clientsInSelectedA.find(c) != clientsInSelectedA.end(); })) -
            static_cast<int>(std::count_if(
                parents.second->chromR[(startB + nOfMovedRoutes) % nOfRoutesB].begin(),
                parents.second->chromR[(startB + nOfMovedRoutes) % nOfRoutesB].end(),
                [&clientsInSelectedA](int c) { return clientsInSelectedA.find(c) != clientsInSelectedA.end(); }));

        const int bestDifference = std::min({differenceALeft, differenceARight, differenceBLeft, differenceBRight});

        if (bestDifference < 0) {
            if (bestDifference == differenceALeft) {
                for (int c : parents.first->chromR[(startA + nOfMovedRoutes - 1) % nOfRoutesA]) {
                    clientsInSelectedA.erase(clientsInSelectedA.find(c));
                }
                startA = (startA - 1 + nOfRoutesA) % nOfRoutesA;
                for (int c : parents.first->chromR[startA]) { clientsInSelectedA.insert(c); }
            } else if (bestDifference == differenceARight) {
                for (int c : parents.first->chromR[startA]) { clientsInSelectedA.erase(clientsInSelectedA.find(c)); }
                startA = (startA + 1) % nOfRoutesA;
                for (int c : parents.first->chromR[(startA + nOfMovedRoutes - 1) % nOfRoutesA]) {
                    clientsInSelectedA.insert(c);
                }
            } else if (bestDifference == differenceBLeft) {
                for (int c : parents.second->chromR[(startB + nOfMovedRoutes - 1) % nOfRoutesB]) {
                    clientsInSelectedB.erase(clientsInSelectedB.find(c));
                }
                startB = (startB - 1 + nOfRoutesB) % nOfRoutesB;
                for (int c : parents.second->chromR[startB]) { clientsInSelectedB.insert(c); }
            } else if (bestDifference == differenceBRight) {
                for (int c : parents.second->chromR[startB]) { clientsInSelectedB.erase(clientsInSelectedB.find(c)); }
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
    std::unordered_set<int> clientsInSelectedANotB;
    std::copy_if(clientsInSelectedA.begin(), clientsInSelectedA.end(),
                 std::inserter(clientsInSelectedANotB, clientsInSelectedANotB.end()),
                 [&clientsInSelectedB](int c) { return clientsInSelectedB.find(c) == clientsInSelectedB.end(); });

    std::unordered_set<int> clientsInSelectedBNotA;
    std::copy_if(clientsInSelectedB.begin(), clientsInSelectedB.end(),
                 std::inserter(clientsInSelectedBNotA, clientsInSelectedBNotA.end()),
                 [&clientsInSelectedA](int c) { return clientsInSelectedA.find(c) == clientsInSelectedA.end(); });

    // Replace selected routes from parent B into parent A
    for (int r = 0; r < nOfMovedRoutes; r++) {
        int indexA = (startA + r) % nOfRoutesA;
        int indexB = (startB + r) % nOfRoutesB;

        auto &offspring0Route = candidateOffsprings[0]->chromR[indexA];
        auto &offspring1Route = candidateOffsprings[1]->chromR[indexA];

        offspring0Route.clear();
        offspring1Route.clear();

        for (int c : parents.second->chromR[indexB]) {
            offspring0Route.push_back(c); // Always copy into offspring 0

            if (!clientsInSelectedBNotA.count(c)) // More cache-friendly lookup
            {
                offspring1Route.push_back(c);
            }
        }
    }

    // Move routes from parent A that are kept
    for (int r = nOfMovedRoutes; r < nOfRoutesA; r++) {
        int indexA = (startA + r) % nOfRoutesA;

        auto &offspring0Route = candidateOffsprings[0]->chromR[indexA];
        auto &offspring1Route = candidateOffsprings[1]->chromR[indexA];

        offspring0Route.clear();
        offspring1Route.clear();

        for (int c : parents.first->chromR[indexA]) {
            if (!clientsInSelectedBNotA.count(c)) // More cache-friendly lookup
            {
                offspring0Route.push_back(c);
            }
            offspring1Route.push_back(c); // Always copy into offspring 1
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

void Genetic::insertUnplannedTasks(Individual *offspring, const std::unordered_set<int> &unplannedTasks) {
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
