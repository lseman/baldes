#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <list>
#include <vector>

#include "Individual.h"
#include "LocalSearch.h"
#include "Params.h"
#include "Population.h"
#include "Split.h"

#include "../../third_party/fpmax/fitemset.cpp"
#include "../../third_party/fpmax/fpmax.h"

#include "Definitions.h"

std::vector<std::vector<int>> *Population::nextMDMPattern() {
    if (mdmPatterns.empty()) return NULL;

    std::vector<std::vector<int>> *pattern = &(mdmPatterns[mdmNextPattern]);
    mdmNextPattern                         = (mdmNextPattern + 1) % mdmPatterns.size();

    return pattern;
}

void Population::doHGSLocalSearchAndAddIndividual(Individual *indiv) {
    // Do a Local Search
    localSearch->run(indiv, params->penaltyCapacity, params->penaltyTimeWarp);

    // Add an individual
    addIndividual(indiv, true);

    // With a certain probability, repair half of the solutions by increasing the penalties for infeasibilities (w.r.t.
    // capacities and time warps) in a new Local Search in case of infeasibility
    if (!indiv->isFeasible && params->rng() % 100 < (unsigned int)params->config.repairProbability) {
        localSearch->run(indiv, params->penaltyCapacity * 10., params->penaltyTimeWarp * 10.);

        // Add the individual only when feasible
        if (indiv->isFeasible) { addIndividual(indiv, false); }
    }
}

void Population::generatePopulation() {
    if (params->nbClients == 1) {
        // Quickly generate the one solution
        Individual randomIndiv(params, true, nextMDMPattern());
        // Individual randomIndiv(params);

        split->generalSplit(&randomIndiv, params->nbVehicles);
        addIndividual(&randomIndiv, true);
        return;
    }

    if (params->config.initialSolution != "") {
        Individual initialIndiv(params, params->config.initialSolution);
        addIndividual(&initialIndiv, true);
    }
    // ------- The below parameters are configurable through command line arguments ---------
    double fractionGeneratedNearest      = params->config.fractionGeneratedNearest;
    double fractionGeneratedFurthest     = params->config.fractionGeneratedFurthest;
    double fractionGeneratedSweep        = params->config.fractionGeneratedSweep;
    double fractionGeneratedRandomly     = params->config.fractionGeneratedRandomly;
    double fractionGeneratedMDM          = params->config.fractionGeneratedMDM;
    int    minSweepFillPercentage        = params->config.minSweepFillPercentage;
    int    maxToleratedCapacityViolation = params->config.maxToleratedCapacityViolation;
    int    maxToleratedTimeWarp          = params->config.maxToleratedTimeWarp;
    double initialTimeWarpPenalty        = params->config.initialTimeWarpPenalty;
    // ------- End of configurable parameters -----------------------------------------------------

    // Generate same number of individuals as in original solution.
    int nofIndividuals = 4 * params->config.minimumPopulationSize;

    // TODO: Change next comment?
    // Note we actually set initial penalty in Params.cpp but by setting it here we also reset it when resetting the
    // population (probably not ideal but test before changing)
    params->penaltyTimeWarp = initialTimeWarpPenalty;
    // Too low fill percentage may cause that not all clients are planned
    minSweepFillPercentage               = std::max(minSweepFillPercentage, 30);
    int nofNearestIndividualsToGenerate  = round(fractionGeneratedNearest * nofIndividuals);
    int nofFurthestIndividualsToGenerate = round(fractionGeneratedFurthest * nofIndividuals);
    int nofSweepIndividualsToGenerate    = round(fractionGeneratedSweep * nofIndividuals);
    int nofRandomIndividualsToGenerate   = round(fractionGeneratedRandomly * nofIndividuals);
    int noOfMDMIndividualsToGenerate     = round(fractionGeneratedMDM * nofIndividuals);
    // Generate some individuals using the MDMPatterns
    for (int i = 0; i < noOfMDMIndividualsToGenerate; i++) {
        Individual indiv(params, true, nextMDMPattern());
        doHGSLocalSearchAndAddIndividual(&indiv);
        // addIndividual(&indiv, true);
    }

    print_heur("Generated {} individuals using MDMPatterns\n", noOfMDMIndividualsToGenerate);
    printState(-1, -1);
    // Generate some individuals using the NEAREST construction heuristic
    for (int i = 0; i < nofNearestIndividualsToGenerate; i++) {
        if (params->isTimeLimitExceeded()) {
            std::cout << "Time limit during generation of initial population" << std::endl;
            printState(-1, -1);
            return;
        }
        // Create the first individual without violations
        int        toleratedCapacityViolation = i == 0 ? 0 : params->rng() % (maxToleratedCapacityViolation + 1);
        int        toleratedTimeWarp          = i == 0 ? 0 : params->rng() % (maxToleratedTimeWarp + 1);
        Individual indiv(params, false);
        localSearch->constructIndividualWithSeedOrder(toleratedCapacityViolation, toleratedTimeWarp, false, &indiv);
        doHGSLocalSearchAndAddIndividual(&indiv);
    }

    // Output that some individuals have been created
    // std::cout << "Generated " << nofNearestIndividualsToGenerate << " individuals using Nearest" << std::endl;
    print_heur("Generated {} individuals using Nearest\n", nofNearestIndividualsToGenerate);
    printState(-1, -1);

    // Generate some individuals using the FURHEST construction heuristic
    for (int i = 0; i < nofFurthestIndividualsToGenerate; i++) {
        if (params->isTimeLimitExceeded()) {
            std::cout << "Time limit during generation of initial population" << std::endl;
            printState(-1, -1);
            return;
        }
        // Create the first individual without violations
        int        toleratedCapacityViolation = i == 0 ? 0 : params->rng() % (maxToleratedCapacityViolation + 1);
        int        toleratedTimeWarp          = i == 0 ? 0 : params->rng() % (maxToleratedTimeWarp + 1);
        Individual indiv(params, false);
        localSearch->constructIndividualWithSeedOrder(toleratedCapacityViolation, toleratedTimeWarp, true, &indiv);
        doHGSLocalSearchAndAddIndividual(&indiv);
    }

    // Output that some individuals have been created
    // std::cout << "Generated " << nofFurthestIndividualsToGenerate << " individuals using Furthest" << std::endl;
    print_heur("Generated {} individuals using Furthest\n", nofFurthestIndividualsToGenerate);
    printState(-1, -1);

    // Generate some individuals using the SWEEP construction heuristic
    for (int i = 0; i < nofSweepIndividualsToGenerate; i++) {
        if (params->isTimeLimitExceeded()) {
            std::cout << "Time limit during generation of initial population" << std::endl;
            printState(-1, -1);
            return;
        }
        // Create the first individual without load restrictions
        int fillPercentage = i == 0 ? 100 : minSweepFillPercentage + params->rng() % (100 - minSweepFillPercentage + 1);
        Individual indiv(params, false);
        localSearch->constructIndividualBySweep(fillPercentage, &indiv);
        doHGSLocalSearchAndAddIndividual(&indiv);
    }

    // Output that some individuals have been created
    // std::cout << "Generated " << nofSweepIndividualsToGenerate << " individuals using Sweep" << std::endl;
    print_heur("Generated {} individuals using Sweep\n", nofSweepIndividualsToGenerate);
    printState(-1, -1);

    // Generate some individuals using a RANDOM strategy
    for (int i = 0; i < nofRandomIndividualsToGenerate; i++) {
        if (params->isTimeLimitExceeded()) {
            std::cout << "Time limit during generation of initial population" << std::endl;
            printState(-1, -1);
            return;
        }
        Individual randomIndiv(params);
        split->generalSplit(&randomIndiv, params->nbVehicles);
        doHGSLocalSearchAndAddIndividual(&randomIndiv);
    }

    // Output that some individuals have been created
    // std::cout << "Generated " << nofRandomIndividualsToGenerate << " individuals Randomly" << std::endl;
    print_heur("Generated {} individuals Randomly\n", nofRandomIndividualsToGenerate);
    printState(-1, -1);
}

bool Population::addIndividual(const Individual *indiv, bool updateFeasible) {
    // Update the feasibility if needed
    if (updateFeasible) {
        listFeasibilityLoad.push_back(indiv->myCostSol.capacityExcess < MY_EPSILON);
        listFeasibilityTimeWarp.push_back(indiv->myCostSol.timeWarp < MY_EPSILON);
        listFeasibilityLoad.pop_front();
        listFeasibilityTimeWarp.pop_front();
    }

    // Find the adequate subpopulation in relation to the individual feasibility
    SubPopulation &subpop = (indiv->isFeasible) ? feasibleSubpopulation : infeasibleSubpopulation;

    // Create a copy of the individual and update the proximity structures calculating inter-individual distances
    Individual *myIndividual = new Individual(*indiv);
    for (Individual *myIndividual2 : subpop) {
        double myDistance = myIndividual->brokenPairsDistance(myIndividual2);
        myIndividual2->indivsPerProximity.insert({myDistance, myIndividual});
        myIndividual->indivsPerProximity.insert({myDistance, myIndividual2});
    }

    // Identify the correct location in the population and insert the individual
    int place = static_cast<int>(subpop.size());
    while (place > 0 && subpop[place - 1]->myCostSol.penalizedCost > indiv->myCostSol.penalizedCost - MY_EPSILON) {
        place--;
    }
    subpop.emplace(subpop.begin() + place, myIndividual);

    // Trigger a survivor selection if the maximimum population size is exceeded
    if (static_cast<int>(subpop.size()) > params->config.minimumPopulationSize + params->config.generationSize) {
        while (static_cast<int>(subpop.size()) > params->config.minimumPopulationSize) {
            removeWorstBiasedFitness(subpop);
        }
    }

    // Track best solution
    if (indiv->isFeasible &&
        indiv->myCostSol.penalizedCost < bestSolutionRestart.myCostSol.penalizedCost - MY_EPSILON) {
        updateMDMElite(indiv);
        bestSolutionRestart = *indiv;
        searchProgress.push_back({params->getTimeElapsedSeconds(), bestSolutionOverall.myCostSol.penalizedCost});
        return true;
    } else
        return false;
}

void Population::updateBiasedFitnesses(SubPopulation &pop) {
    // Ranking the individuals based on their diversity contribution (decreasing order of
    // averageBrokenPairsDistanceClosest)
    std::vector<std::pair<double, int>> ranking;
    for (int i = 0; i < static_cast<int>(pop.size()); i++) {
        ranking.push_back({-pop[i]->averageBrokenPairsDistanceClosest(params->config.nbClose), i});
    }
    pdqsort(ranking.begin(), ranking.end());

    // Updating the biased fitness values. If there is only one individual, its biasedFitness is 0
    if (pop.size() == 1) {
        pop[0]->biasedFitness = 0;
    } else {
        // Loop over all individuals
        for (int i = 0; i < static_cast<int>(pop.size()); i++) {
            // Ranking the individuals based on the diversity rank and diversity measure from 0 to 1
            double divRank = static_cast<double>(i) / static_cast<double>(pop.size() - 1);
            double fitRank = ranking[i].second / static_cast<double>(pop.size() - 1);

            // Elite individuals cannot be smaller than population size
            if (static_cast<int>(pop.size()) <= params->config.nbElite) {
                pop[ranking[i].second]->biasedFitness = fitRank;
            } else if (params->config.diversityWeight > 0) {
                pop[ranking[i].second]->biasedFitness = fitRank + params->config.diversityWeight * divRank;
            } else {
                pop[ranking[i].second]->biasedFitness =
                    fitRank +
                    (1.0 - static_cast<double>(params->config.nbElite) / static_cast<double>(pop.size())) * divRank;
            }
        }
    }
}

void Population::removeWorstBiasedFitness(SubPopulation &pop) {
    // Update the fitness values
    updateBiasedFitnesses(pop);

    // Throw an error of the population has at most one individual
    if (pop.size() <= 1) { throw std::string("Eliminating the best individual: this should not occur in HGS"); }

    Individual *worstIndividual              = nullptr;
    int         worstIndividualPosition      = -1;
    bool        isWorstIndividualClone       = false;
    double      worstIndividualBiasedFitness = -1.e30;
    // Loop over all individuals and save the wordt individual
    for (int i = 1; i < static_cast<int>(pop.size()); i++) {
        // An averageBrokenPairsDistanceClosest equal to 0 indicates that a clone exists
        bool isClone = (pop[i]->averageBrokenPairsDistanceClosest(1) < MY_EPSILON);
        if ((isClone && !isWorstIndividualClone) ||
            (isClone == isWorstIndividualClone && pop[i]->biasedFitness > worstIndividualBiasedFitness)) {
            worstIndividualBiasedFitness = pop[i]->biasedFitness;
            isWorstIndividualClone       = isClone;
            worstIndividualPosition      = i;
            worstIndividual              = pop[i];
        }
    }

    // Remove the worst individual from the population
    pop.erase(pop.begin() + worstIndividualPosition);
    // Cleaning its distances from the other individuals in the population
    for (Individual *myIndividual2 : pop) myIndividual2->removeProximity(worstIndividual);
    // Freeing memory
    delete worstIndividual;
}

void Population::restart() {
    std::cout << "----- RESET: CREATING A NEW POPULATION -----" << std::endl;

    // Delete all the individuals (feasible and infeasible)
    for (Individual *indiv : feasibleSubpopulation) { delete indiv; }
    for (Individual *indiv : infeasibleSubpopulation) { delete indiv; }

    // Clear the pools of solutions and make a new empty individual as the best solution after the restart
    feasibleSubpopulation.clear();
    infeasibleSubpopulation.clear();
    bestSolutionRestart = Individual();

    // Generate a new initial population
    generatePopulation();
}

void Population::managePenalties() {
    // Setting some bounds [0.1,100000] to the penalty values for safety
    double fractionFeasibleLoad =
        static_cast<double>(std::count(listFeasibilityLoad.begin(), listFeasibilityLoad.end(), true)) /
        static_cast<double>(listFeasibilityLoad.size());
    if (fractionFeasibleLoad <= 0.01 && params->config.penaltyBooster > 0. && params->penaltyCapacity < 100000.) {
        params->penaltyCapacity = std::min(params->penaltyCapacity * params->config.penaltyBooster, 100000.);
    } else if (fractionFeasibleLoad < params->config.targetFeasible - 0.05 && params->penaltyCapacity < 100000.) {
        params->penaltyCapacity = std::min(params->penaltyCapacity * 1.2, 100000.);
    } else if (fractionFeasibleLoad > params->config.targetFeasible + 0.05 && params->penaltyCapacity > 0.1) {
        params->penaltyCapacity = std::max(params->penaltyCapacity * 0.85, 0.1);
    }

    // Setting some bounds [0.1,100000] to the penalty values for safety
    double fractionFeasibleTimeWarp =
        static_cast<double>(std::count(listFeasibilityTimeWarp.begin(), listFeasibilityTimeWarp.end(), true)) /
        static_cast<double>(listFeasibilityTimeWarp.size());
    if (fractionFeasibleTimeWarp <= 0.01 && params->config.penaltyBooster > 0. && params->penaltyTimeWarp < 100000.) {
        params->penaltyTimeWarp = std::min(params->penaltyTimeWarp * params->config.penaltyBooster, 100000.);
    } else if (fractionFeasibleTimeWarp < params->config.targetFeasible - 0.05 && params->penaltyTimeWarp < 100000.) {
        params->penaltyTimeWarp = std::min(params->penaltyTimeWarp * 1.2, 100000.);
    } else if (fractionFeasibleTimeWarp > params->config.targetFeasible + 0.05 && params->penaltyTimeWarp > 0.1) {
        params->penaltyTimeWarp = std::max(params->penaltyTimeWarp * 0.85, 0.1);
    }

    // Update the evaluations
    for (int i = 0; i < static_cast<int>(infeasibleSubpopulation.size()); i++) {
        infeasibleSubpopulation[i]->myCostSol.penalizedCost =
            infeasibleSubpopulation[i]->myCostSol.distance +
            params->penaltyCapacity * infeasibleSubpopulation[i]->myCostSol.capacityExcess +
            params->penaltyTimeWarp * infeasibleSubpopulation[i]->myCostSol.timeWarp;
    }

    // If needed, reorder the individuals in the infeasible subpopulation since the penalty values have changed (simple
    // bubble sort for the sake of simplicity)
    for (int i = 0; i < static_cast<int>(infeasibleSubpopulation.size()); i++) {
        for (size_t j = 0; j < infeasibleSubpopulation.size() - i - 1; j++) {
            if (infeasibleSubpopulation[j]->myCostSol.penalizedCost >
                infeasibleSubpopulation[j + 1]->myCostSol.penalizedCost + MY_EPSILON) {
                Individual *indiv              = infeasibleSubpopulation[j];
                infeasibleSubpopulation[j]     = infeasibleSubpopulation[j + 1];
                infeasibleSubpopulation[j + 1] = indiv;
            }
        }
    }
}

Individual *Population::getBinaryTournament() {
    Individual *individual1;
    Individual *individual2;

    // Update the fitness values of all the individuals (feasible and infeasible)
    updateBiasedFitnesses(feasibleSubpopulation);
    updateBiasedFitnesses(infeasibleSubpopulation);

    // Pick a first random number individual from the total population (of both feasible and infeasible individuals)
    int place1 = params->rng() % (feasibleSubpopulation.size() + infeasibleSubpopulation.size());
    if (place1 >= static_cast<int>(feasibleSubpopulation.size())) {
        individual1 = infeasibleSubpopulation[place1 - feasibleSubpopulation.size()];
    } else {
        individual1 = feasibleSubpopulation[place1];
    }

    // Pick a second random number individual from the total population (of both feasible and infeasible individuals)
    int place2 = params->rng() % (feasibleSubpopulation.size() + infeasibleSubpopulation.size());
    if (place2 >= static_cast<int>(feasibleSubpopulation.size())) {
        individual2 = infeasibleSubpopulation[place2 - feasibleSubpopulation.size()];
    } else {
        individual2 = feasibleSubpopulation[place2];
    }

    // Return the individual with the lowest biasedFitness value
    if (individual1->biasedFitness < individual2->biasedFitness) {
        return individual1;
    } else {
        return individual2;
    }
}

std::pair<Individual *, Individual *> Population::getNonIdenticalParentsBinaryTournament() {
    // Pick two individual using a binary tournament
    Individual *parentA   = getBinaryTournament();
    Individual *parentB   = getBinaryTournament();
    int         num_tries = 1;
    // Pick two other individuals as long as they are identical (try at most 9 times)
    while (parentA->brokenPairsDistance(parentB) < MY_EPSILON && num_tries < 10) {
        parentB = getBinaryTournament();
        num_tries++;
    }

    // Return the two individuals as a pair
    return std::make_pair(parentA, parentB);
}

Individual *Population::getBestFeasible() {
    // Return the best feasible solution if a feasible solution exists
    if (!feasibleSubpopulation.empty()) {
        return feasibleSubpopulation[0];
    } else
        return nullptr;
}

Individual *Population::getBestInfeasible() {
    // Return the best infeasible solution if an infeasible solution exists
    if (!infeasibleSubpopulation.empty()) {
        return infeasibleSubpopulation[0];
    } else
        return nullptr;
}

Individual *Population::getBestFound() {
    // Return the best overall solution if a solution exists
    if (bestSolutionOverall.myCostSol.penalizedCost < 1.e29) {
        return &bestSolutionOverall;
    } else
        return nullptr;
}

void Population::printState(int nbIter, int nbIterNoImprovement) {
    // Print the number of iterations, the number of iterations since the last improvement, and the running time
    std::printf("| It %6d %6d | T(s) %.2f", nbIter, nbIterNoImprovement, params->getTimeElapsedSeconds());

    // If there is at least one feasible solution, print the number of feasible solutions, the cost of the best feasible
    // solution, and the average cost of the feasible solutions
    if (getBestFeasible() != nullptr) {
        std::printf(" | Feas %zu %.2f %.2f", feasibleSubpopulation.size(), getBestFeasible()->myCostSol.penalizedCost,
                    getAverageCost(feasibleSubpopulation));
    } else {
        std::printf(" | NO-FEASIBLE");
    }

    // If there is at least one infeasible solution, print the number of infeasible solutions, the cost of the best
    // infeasible solution, and the average cost of the infeasible solutions
    if (getBestInfeasible() != nullptr) {
        std::printf(" | Inf %zu %.2f %.2f", infeasibleSubpopulation.size(),
                    getBestInfeasible()->myCostSol.penalizedCost, getAverageCost(infeasibleSubpopulation));
    } else {
        std::printf(" | NO-INFEASIBLE");
    }

    // Print the diversity of both pools of solutions, the average load- and time warp feasibilities of the last 100
    // solutions generated by LS, and the penalties for the capacit and the time warp
    std::printf(" | Div %.2f %.2f", getDiversity(feasibleSubpopulation), getDiversity(infeasibleSubpopulation));
    std::printf(" | Feas %.2f %.2f",
                static_cast<double>(std::count(listFeasibilityLoad.begin(), listFeasibilityLoad.end(), true)) /
                    static_cast<double>(listFeasibilityLoad.size()),
                static_cast<double>(std::count(listFeasibilityTimeWarp.begin(), listFeasibilityTimeWarp.end(), true)) /
                    static_cast<double>(listFeasibilityTimeWarp.size()));
    std::printf(" | Pen %.2f %.2f |", params->penaltyCapacity, params->penaltyTimeWarp);
    std::cout << std::endl;
}

double Population::getDiversity(const SubPopulation &pop) {
    // The diversity of the population: The average of the averageBrokenPairsDistanceClosest over the best "mu"
    // individuals of the population
    double average = 0.;

    // Sum all the averageBrokenPairsDistanceClosest of the individuals
    // Only monitoring the "mu" best solutions to avoid too much noise in the measurements
    int size = std::min(params->config.minimumPopulationSize, static_cast<int>(pop.size()));
    for (int i = 0; i < size; i++) { average += pop[i]->averageBrokenPairsDistanceClosest(size); }

    // Calculate the average and return if possible
    if (size > 0) {
        return average / static_cast<double>(size);
    } else {
        return -1.0;
    }
}

double Population::getAverageCost(const SubPopulation &pop) {
    // The average cost of the population: The average of the penalizedCost over the best "mu" individuals of the
    // population
    double average = 0.;

    // Sum all the penalizedCost of the individuals
    // Only monitoring the "mu" best solutions to avoid too much noise in the measurements
    int size = std::min(params->config.minimumPopulationSize, static_cast<int>(pop.size()));
    for (int i = 0; i < size; i++) { average += pop[i]->myCostSol.penalizedCost; }

    // Calculate the average and return if possible
    if (size > 0) {
        return average / static_cast<double>(size);
    } else {
        return -1.0;
    }
}

Population::Population(Params *params, Split *split, HGSLocalSearch *localSearch)
    : params(params), split(split), localSearch(localSearch) {
    // Create lists for the load feasibility of the last 100 individuals generated by LS, where all feasibilities are
    // set to true

    listFeasibilityLoad     = std::list<bool>(100, true);
    listFeasibilityTimeWarp = std::list<bool>(100, true);

    // Generate a new population
    generatePopulation();
}

Population::~Population() {
    // Delete all information from the feasibleSubpopulation
    for (int i = 0; i < static_cast<int>(feasibleSubpopulation.size()); i++) { delete feasibleSubpopulation[i]; }

    // Delete all information from the infeasibleSubpopulation
    for (int i = 0; i < static_cast<int>(infeasibleSubpopulation.size()); i++) { delete infeasibleSubpopulation[i]; }
}

// Inserts the individual in MDM elite set if:
// (1) it is different from those already in the set; and
// (2) the MDM elite set is not full OR this individual has a better penalized cost than at least one of those already
// in the set
void Population::updateMDMElite(const Individual *indiv) {
    constexpr int mdmNbElite     = 5; // Maximum number of elite individuals
    unsigned      old_elite_size = mdmElite.size();

    // Insert the new individual into the elite queue
    // create copy of indiv
    Individual *indivCopy = new Individual(*indiv);
    mdmElite.push(indivCopy);

    // If the number of elites exceeds the limit, remove the worst individual (which is on top of the priority queue)
    if (mdmElite.size() > mdmNbElite) {
        mdmElite.pop(); // Remove the highest penalizedCost individual
    }

    // Since we updated the elite set, reset the counter and mark it as updated
    mdmEliteNonUpdatingRestarts = 0;
    mdmEliteUpdated             = true;
}

/*
 * Returns the next pattern to be used in the MDM construction heuristic
 */
void Population::mineElite() {
    constexpr int    mdmNbPatterns = 5;
    constexpr double mdmMinSup     = 0.8;

    // Ensure mining only happens under certain conditions
    if (mdmEliteUpdated && mdmEliteNonUpdatingRestarts >= mdmEliteMaxNonUpdatingRestarts && mdmElite.size() > 1) {
        if (params->verbose) { std::cout << "----- MINING PATTERNS FROM MDM ELITE SET -----" << std::endl; }

        int minSup      = std::max(2, static_cast<int>(mdmMinSup * mdmElite.size()));
        int numPatterns = mdmNbPatterns;
        int nbNodes     = params->nbClients + 1; // all clients + depot

        // Dataset to avoid manual deletion
        Dataset dataset;

        std::vector<const Individual *> eliteVector;

        // Copy elements from priority queue to a vector for iteration
        std::priority_queue<const Individual *, std::vector<const Individual *>, CompareIndividual> tempQueue =
            mdmElite;

        while (!tempQueue.empty()) {
            eliteVector.push_back(tempQueue.top());
            tempQueue.pop();
        }

        // Now you can iterate over eliteVector
        for (const auto &indiv : eliteVector) {
            std::set<int> transaction;

            for (int r = 0; r < params->nbVehicles; ++r) {
                const auto &chromR = indiv->chromR[r];
                for (int c = 0; c < static_cast<int>(chromR.size()) - 1; ++c) {
                    int index = chromR[c] * nbNodes + chromR[c + 1]; // Map 2D matrix to vector index
                    transaction.insert(index);
                }
            }
            dataset.push_back(transaction);
        }

        // Execute fpmax to get frequent itemsets
        FISet *frequentItemsets = fpmax(&dataset, minSup, numPatterns);

        // Clear previous patterns and mine new ones
        mdmPatterns.clear();
        for (const auto &itemset : *frequentItemsets) {
            std::vector<std::vector<int>> tempPattern;
            std::list<std::vector<int> *> routes;

            // Process each itemset to form routes
            for (int index : itemset) {
                int n1 = index / nbNodes;
                int n2 = index % nbNodes;

                std::vector<int> *lhs = nullptr;
                std::vector<int> *rhs = nullptr;

                // Find routes that match the nodes n1 or n2
                for (auto &route : routes) {
                    if (n1 && n1 == route->back()) lhs = route;
                    if (n2 && n2 == route->front()) rhs = route;
                }

                // Merge or extend routes
                if (lhs) {
                    if (rhs) {
                        // Merge rhs into lhs and remove rhs from routes
                        lhs->insert(lhs->end(), rhs->begin(), rhs->end());
                        routes.remove(rhs);
                        delete rhs;
                    } else {
                        lhs->push_back(n2);
                    }
                } else if (rhs) {
                    rhs->insert(rhs->begin(), n1);
                } else {
                    // Create a new route with n1 and n2
                    auto *newRoute = new std::vector<int>{n1, n2};
                    routes.push_back(newRoute);
                }
            }

            // Move routes into the pattern and clean up
            for (auto &route : routes) {
                tempPattern.push_back(std::move(*route));
                delete route; // Clean up after moving
            }

            mdmPatterns.push_back(std::move(tempPattern));
        }

        // Clean up memory
        delete frequentItemsets;

        // Reset flags
        mdmEliteUpdated = false;
        mdmNextPattern  = 0;
    }
}
