#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "../Definitions.h"
#include "../Reader.h"

#include "CircleSector.h"
#include "Matrix.h"

#include "Params.h"

#include "xorshift128.h"

Params::Params(const InstanceData &instance) {
    // Read and create some parameter values from the commandline
    // config             = cl.config;
    nbVehicles         = config.nbVeh;
    rng                = XorShift128(config.seed);
    startWallClockTime = std::chrono::system_clock::now();
    startCPUTime       = std::clock();

    // Convert the circle sector parameters from degrees ([0,359]) to [0,65535] to allow for faster calculations
    circleSectorOverlapTolerance = static_cast<int>(config.circleSectorOverlapToleranceDegrees / 360. * 65536);
    minCircleSectorSize          = static_cast<int>(config.minCircleSectorSizeDegrees / 360. * 65536);

    // Initialize some parameter values
    std::string content, content2, content3;
    int         serviceTimeData = 0;
    int         node;
    bool        hasServiceTimeSection = false;
    // nbClients                         = N_SIZE - 1;
    totalDemand     = 0;
    maxDemand       = 0;
    nbVehicles      = instance.nV;
    vehicleCapacity = instance.q;
    durationLimit   = INT_MAX;
    // vehicleCapacity          = INT_MAX;
    isDurationConstraint     = false;
    isExplicitDistanceMatrix = false;

    // Read INPUT dataset from the new format
    cli       = std::vector<Client>(1001); // Assuming max of 1000 clients + depot
    nbClients = 0;

    // Loop over all customer lines
    for (int i = 0; i < N_SIZE - 1; i++) {
        cli[nbClients].custNum = i;

        cli[nbClients].coordX          = instance.x_coord[i];
        cli[nbClients].coordY          = instance.y_coord[i];
        cli[nbClients].demand          = instance.demand[i];
        cli[nbClients].earliestArrival = instance.window_open[i];
        cli[nbClients].latestArrival   = instance.window_close[i];
        cli[nbClients].serviceDuration = instance.service_time[i];

        // Scale coordinates and times by a factor of 10
        cli[nbClients].coordX *= 10;
        cli[nbClients].coordY *= 10;
        cli[nbClients].polarAngle = CircleSector::positive_mod(static_cast<int>(
            32768. * atan2(cli[nbClients].coordY - cli[0].coordY, cli[nbClients].coordX - cli[0].coordX) / PI));

        // Keep track of max demand and total demand
        if (cli[nbClients].demand > maxDemand) { maxDemand = cli[nbClients].demand; }
        totalDemand += cli[nbClients].demand;

        nbClients++;
    }

    // Reduce the size of the vector of clients
    cli.resize(nbClients);
    nbClients--; // Don't count the depot as a client

    // Check depot constraints
    if (cli[0].earliestArrival != 0) { throw std::string("Time window for depot should start at 0"); }
    if (cli[0].serviceDuration != 0) { throw std::string("Service duration for depot should be 0"); }

    // Set default fleet size if not provided by the user
    if (nbVehicles == INT_MAX) {
        nbVehicles = static_cast<int>(std::ceil(1.3 * totalDemand / vehicleCapacity) + 3.);
        // std::cout << "----- FLEET SIZE WAS NOT SPECIFIED: DEFAULT INITIALIZATION TO " << nbVehicles << " VEHICLES"
        //           << std::endl;
        // print_info("Fleet size was not specified: default initialization to {} vehicles \n", nbVehicles);
    } else {
        // print_info("Fleet size specified in the commandline: set to {} vehicles \n", nbVehicles);
    }

    // For DIMACS runs, or when dynamic parameters have to be used, set more parameter values
    if (config.isDimacsRun || config.useDynamicParameters) {
        // Determine categories of instances based on number of stops/route and whether it has large time windows
        // Calculate an upper bound for the number of stops per route based on capacities
        double stopsPerRoute = vehicleCapacity / (totalDemand / nbClients);
        // Routes are large when more than 25 stops per route
        bool hasLargeRoutes = stopsPerRoute > 25;
        // Get the time horizon (by using the time window of the depot)
        int horizon   = cli[0].latestArrival - cli[0].earliestArrival;
        int nbLargeTW = 0;

        // Loop over all clients (excluding the depot) and count the amount of large time windows (greater than
        // 0.7*horizon)
        for (int i = 1; i <= nbClients; i++) {
            if (cli[i].latestArrival - cli[i].earliestArrival > 0.7 * horizon) { nbLargeTW++; }
        }
        // Output if an instance has large routes and a large time window
        bool hasLargeTW = nbLargeTW > 0;
        std::cout << "----- HasLargeRoutes: " << hasLargeRoutes << ", HasLargeTW: " << hasLargeTW << std::endl;

        // Set the parameter values based on the characteristics of the instance
        if (hasLargeRoutes) {
            config.nbGranular = 40;
            // Grow neighborhood and population size
            config.growNbGranularAfterIterations = 10000;
            config.growNbGranularSize            = 5;
            config.growPopulationAfterIterations = 10000;
            config.growPopulationSize            = 5;
            // Intensify occasionally
            config.intensificationProbabilityLS = 15;
        } else {
            // Grow population size only
            // config.growNbGranularAfterIterations = 10000;
            // config.growNbGranularSize = 5;
            if (hasLargeTW) {
                // Smaller neighbourhood so iterations are faster
                // So take more iterations before growing population
                config.nbGranular                    = 20;
                config.growPopulationAfterIterations = 20000;
            } else {
                config.nbGranular                    = 40;
                config.growPopulationAfterIterations = 10000;
            }
            config.growPopulationSize = 5;
            // Intensify always
            config.intensificationProbabilityLS = 100;
        }
    }

    if (!isExplicitDistanceMatrix) {
        // Calculation of the distance matrix
        maxDist  = 0;
        timeCost = Matrix(nbClients + 1);
        // Loop over all clients (including the depot)
        for (int i = 0; i <= nbClients; i++) {
            // Set the diagonal element to zero (travel to itself)
            timeCost.set(i, i, 0);
            // Loop over all other clients
            for (int j = i + 1; j <= nbClients; j++) {
                // Calculate Euclidian distance d
                double d = std::sqrt((cli[i].coordX - cli[j].coordX) * (cli[i].coordX - cli[j].coordX) +
                                     (cli[i].coordY - cli[j].coordY) * (cli[i].coordY - cli[j].coordY));
                // Integer truncation
                int cost = static_cast<int>(d);
                // Keep track of the max distance
                if (cost > maxDist) { maxDist = cost; }
                // Save the distances in the matrix
                timeCost.set(i, j, cost);
                timeCost.set(j, i, cost);
            }
        }
    }

    // Compute order proximities once
    orderProximities = std::vector<std::vector<std::pair<double, int>>>(nbClients + 1);
    // Loop over all clients (excluding the depot)
    for (int i = 1; i <= nbClients; i++) {
        // Remove all elements from the vector
        auto &orderProximity = orderProximities[i];
        orderProximity.clear();

        // Loop over all clients (excluding the depot and the specific client itself)
        for (int j = 1; j <= nbClients; j++) {
            if (i != j) {
                // Compute proximity using Eq. 4 in Vidal 2012, and append at the end of orderProximity
                const int timeIJ = timeCost.get(i, j);
                orderProximity.emplace_back(
                    timeIJ +
                        std::min(
                            proximityWeightWaitTime * std::max(cli[j].earliestArrival - timeIJ -
                                                                   cli[i].serviceDuration - cli[i].latestArrival,
                                                               0) +
                                proximityWeightTimeWarp * std::max(cli[i].earliestArrival + cli[i].serviceDuration +
                                                                       timeIJ - cli[j].latestArrival,
                                                                   0),
                            proximityWeightWaitTime * std::max(cli[i].earliestArrival - timeIJ -
                                                                   cli[j].serviceDuration - cli[j].latestArrival,
                                                               0) +
                                proximityWeightTimeWarp * std::max(cli[j].earliestArrival + cli[j].serviceDuration +
                                                                       timeIJ - cli[i].latestArrival,
                                                                   0)),
                    j);
            }
        }

        // Sort orderProximity (for the specific client)
        std::sort(orderProximity.begin(), orderProximity.end());
    }

    // Calculate, for all vertices, the correlation for the nbGranular closest vertices
    SetCorrelatedVertices();

    // Safeguards to avoid possible numerical instability in case of instances containing arbitrarily small or large
    // numerical values
    if (maxDist < 0.1 || maxDist > 100000) {
        throw std::string("The distances are of very small or large scale. This could impact numerical stability. "
                          "Please rescale the dataset and run again.");
    }
    if (maxDemand < 0.1 || maxDemand > 100000) {
        throw std::string("The demand quantities are of very small or large scale. This could impact numerical "
                          "stability. Please rescale the dataset and run again.");
    }
    if (nbVehicles < std::ceil(totalDemand / vehicleCapacity)) {
        throw std::string("Fleet size is insufficient to service the considered clients.");
    }

    // A reasonable scale for the initial values of the penalties
    penaltyCapacity = std::max(0.1, std::min(1000., static_cast<double>(maxDist) / maxDemand));

    // Initial parameter values of these two parameters are not argued
    penaltyWaitTime = 0.;
    penaltyTimeWarp = config.initialTimeWarpPenalty;

    // See Vidal 2012, HGS for VRPTW
    proximityWeightWaitTime = 0.2;
    proximityWeightTimeWarp = 1;

    savingsList = std::vector<Savings>(nbClients * (nbClients - 1) / 2); // Assuming the distance matrix is symmetric

    int savingsCount = 0;
    for (int i = 1; i <= nbClients; i++)
        for (int j = 1; j < i; j++) {
            savingsList[savingsCount].c1    = i;
            savingsList[savingsCount].c2    = j;
            savingsList[savingsCount].value = timeCost[0][i] + timeCost[0][j] - timeCost[i][j];
            savingsCount++;
        }

    std::sort(savingsList.begin(), savingsList.end(), compSavings);
}

Params::Params(const std::string &path_location) {
    // Read and create some parameter values from the commandline
    // config             = cl.config;
    nbVehicles         = config.nbVeh;
    rng                = XorShift128(config.seed);
    startWallClockTime = std::chrono::system_clock::now();
    startCPUTime       = std::clock();

    // Convert the circle sector parameters from degrees ([0,359]) to [0,65535] to allow for faster calculations
    circleSectorOverlapTolerance = static_cast<int>(config.circleSectorOverlapToleranceDegrees / 360. * 65536);
    minCircleSectorSize          = static_cast<int>(config.minCircleSectorSizeDegrees / 360. * 65536);

    // Initialize some parameter values
    std::string content, content2, content3;
    int         serviceTimeData = 0;
    int         node;
    bool        hasServiceTimeSection = false;
    nbClients                         = 0;
    totalDemand                       = 0;
    maxDemand                         = 0;
    durationLimit                     = INT_MAX;
    vehicleCapacity                   = INT_MAX;
    isDurationConstraint              = false;
    isExplicitDistanceMatrix          = false;

    // Read INPUT dataset from the new format
    std::ifstream inputFile(path_location);
    if (inputFile.is_open()) {
        // Read and skip any metadata lines until we find the VEHICLE section
        while (getline(inputFile, content)) {
            content.erase(std::remove(content.begin(), content.end(), '\r'), content.end()); // Remove any \r characters

            // If the content includes VEHICLE, stop skipping lines
            if (content == "VEHICLE") { break; }
        }

        // If we didn't find the VEHICLE section, throw an error
        if (content != "VEHICLE") { throw std::invalid_argument("Expected VEHICLE section"); }

        // Read the VEHICLE section
        getline(inputFile, content); // Read "NUMBER CAPACITY" line
        inputFile >> nbVehicles >> vehicleCapacity;

        // Read the CUSTOMER section
        getline(inputFile, content); // Read empty line or "CUSTOMER"
        getline(inputFile, content); // Read empty line or "CUSTOMER"
        getline(inputFile, content); // Read the column headers

        if (content.substr(0, 8) == "CUSTOMER") {
            getline(inputFile, content); // Skip the header line

            // Create a vector to store client information
            cli       = std::vector<Client>(1001); // Assuming max of 1000 clients + depot
            nbClients = 0;
            int node;

            // Loop over all customer lines
            while (inputFile >> node) {
                cli[nbClients].custNum = node;
                inputFile >> cli[nbClients].coordX >> cli[nbClients].coordY >> cli[nbClients].demand >>
                    cli[nbClients].earliestArrival >> cli[nbClients].latestArrival >> cli[nbClients].serviceDuration;

                // Scale coordinates and times by a factor of 10
                cli[nbClients].coordX *= 10;
                cli[nbClients].coordY *= 10;
                cli[nbClients].earliestArrival *= 10;
                cli[nbClients].latestArrival *= 10;
                cli[nbClients].serviceDuration *= 10;
                cli[nbClients].polarAngle = CircleSector::positive_mod(static_cast<int>(
                    32768. * atan2(cli[nbClients].coordY - cli[0].coordY, cli[nbClients].coordX - cli[0].coordX) / PI));

                // Keep track of max demand and total demand
                if (cli[nbClients].demand > maxDemand) { maxDemand = cli[nbClients].demand; }
                totalDemand += cli[nbClients].demand;

                nbClients++;
            }

            // Reduce the size of the vector of clients
            cli.resize(nbClients);
            nbClients--; // Don't count the depot as a client

            // Check depot constraints
            if (cli[0].earliestArrival != 0) { throw std::string("Time window for depot should start at 0"); }
            if (cli[0].serviceDuration != 0) { throw std::string("Service duration for depot should be 0"); }
        } else {
            throw std::invalid_argument("Expected CUSTOMER section");
        }
    } else {
        throw std::invalid_argument("Unable to open instance file: " + config.pathInstance);
    }

    // Set default fleet size if not provided by the user
    if (nbVehicles == INT_MAX) {
        nbVehicles = static_cast<int>(std::ceil(1.3 * totalDemand / vehicleCapacity) + 3.);
        std::cout << "----- FLEET SIZE WAS NOT SPECIFIED: DEFAULT INITIALIZATION TO " << nbVehicles << " VEHICLES"
                  << std::endl;
    } else {
        std::cout << "----- FLEET SIZE SPECIFIED IN THE COMMANDLINE: SET TO " << nbVehicles << " VEHICLES" << std::endl;
    }

    // For DIMACS runs, or when dynamic parameters have to be used, set more parameter values
    if (config.isDimacsRun || config.useDynamicParameters) {
        // Determine categories of instances based on number of stops/route and whether it has large time windows
        // Calculate an upper bound for the number of stops per route based on capacities
        double stopsPerRoute = vehicleCapacity / (totalDemand / nbClients);
        // Routes are large when more than 25 stops per route
        bool hasLargeRoutes = stopsPerRoute > 25;
        // Get the time horizon (by using the time window of the depot)
        int horizon   = cli[0].latestArrival - cli[0].earliestArrival;
        int nbLargeTW = 0;

        // Loop over all clients (excluding the depot) and count the amount of large time windows (greater than
        // 0.7*horizon)
        for (int i = 1; i <= nbClients; i++) {
            if (cli[i].latestArrival - cli[i].earliestArrival > 0.7 * horizon) { nbLargeTW++; }
        }
        // Output if an instance has large routes and a large time window
        bool hasLargeTW = nbLargeTW > 0;
        std::cout << "----- HasLargeRoutes: " << hasLargeRoutes << ", HasLargeTW: " << hasLargeTW << std::endl;

        // Set the parameter values based on the characteristics of the instance
        if (hasLargeRoutes) {
            config.nbGranular = 40;
            // Grow neighborhood and population size
            config.growNbGranularAfterIterations = 10000;
            config.growNbGranularSize            = 5;
            config.growPopulationAfterIterations = 10000;
            config.growPopulationSize            = 5;
            // Intensify occasionally
            config.intensificationProbabilityLS = 15;
        } else {
            // Grow population size only
            // config.growNbGranularAfterIterations = 10000;
            // config.growNbGranularSize = 5;
            if (hasLargeTW) {
                // Smaller neighbourhood so iterations are faster
                // So take more iterations before growing population
                config.nbGranular                    = 20;
                config.growPopulationAfterIterations = 20000;
            } else {
                config.nbGranular                    = 40;
                config.growPopulationAfterIterations = 10000;
            }
            config.growPopulationSize = 5;
            // Intensify always
            config.intensificationProbabilityLS = 100;
        }
    }

    if (!isExplicitDistanceMatrix) {
        // Calculation of the distance matrix
        maxDist  = 0;
        timeCost = Matrix(nbClients + 1);
        // Loop over all clients (including the depot)
        for (int i = 0; i <= nbClients; i++) {
            // Set the diagonal element to zero (travel to itself)
            timeCost.set(i, i, 0);
            // Loop over all other clients
            for (int j = i + 1; j <= nbClients; j++) {
                // Calculate Euclidian distance d
                double d = std::sqrt((cli[i].coordX - cli[j].coordX) * (cli[i].coordX - cli[j].coordX) +
                                     (cli[i].coordY - cli[j].coordY) * (cli[i].coordY - cli[j].coordY));
                // Integer truncation
                int cost = static_cast<int>(d);
                // Keep track of the max distance
                if (cost > maxDist) { maxDist = cost; }
                // Save the distances in the matrix
                timeCost.set(i, j, cost);
                timeCost.set(j, i, cost);
            }
        }
    }

    // Compute order proximities once
    orderProximities = std::vector<std::vector<std::pair<double, int>>>(nbClients + 1);
    // Loop over all clients (excluding the depot)
    for (int i = 1; i <= nbClients; i++) {
        // Remove all elements from the vector
        auto &orderProximity = orderProximities[i];
        orderProximity.clear();

        // Loop over all clients (excluding the depot and the specific client itself)
        for (int j = 1; j <= nbClients; j++) {
            if (i != j) {
                // Compute proximity using Eq. 4 in Vidal 2012, and append at the end of orderProximity
                const int timeIJ = timeCost.get(i, j);
                orderProximity.emplace_back(
                    timeIJ +
                        std::min(
                            proximityWeightWaitTime * std::max(cli[j].earliestArrival - timeIJ -
                                                                   cli[i].serviceDuration - cli[i].latestArrival,
                                                               0) +
                                proximityWeightTimeWarp * std::max(cli[i].earliestArrival + cli[i].serviceDuration +
                                                                       timeIJ - cli[j].latestArrival,
                                                                   0),
                            proximityWeightWaitTime * std::max(cli[i].earliestArrival - timeIJ -
                                                                   cli[j].serviceDuration - cli[j].latestArrival,
                                                               0) +
                                proximityWeightTimeWarp * std::max(cli[j].earliestArrival + cli[j].serviceDuration +
                                                                       timeIJ - cli[i].latestArrival,
                                                                   0)),
                    j);
            }
        }

        // Sort orderProximity (for the specific client)
        std::sort(orderProximity.begin(), orderProximity.end());
    }

    // Calculate, for all vertices, the correlation for the nbGranular closest vertices
    SetCorrelatedVertices();

    // Safeguards to avoid possible numerical instability in case of instances containing arbitrarily small or large
    // numerical values
    if (maxDist < 0.1 || maxDist > 100000) {
        throw std::string("The distances are of very small or large scale. This could impact numerical stability. "
                          "Please rescale the dataset and run again.");
    }
    if (maxDemand < 0.1 || maxDemand > 100000) {
        throw std::string("The demand quantities are of very small or large scale. This could impact numerical "
                          "stability. Please rescale the dataset and run again.");
    }
    if (nbVehicles < std::ceil(totalDemand / vehicleCapacity)) {
        throw std::string("Fleet size is insufficient to service the considered clients.");
    }

    // A reasonable scale for the initial values of the penalties
    penaltyCapacity = std::max(0.1, std::min(1000., static_cast<double>(maxDist) / maxDemand));

    // Initial parameter values of these two parameters are not argued
    penaltyWaitTime = 0.;
    penaltyTimeWarp = config.initialTimeWarpPenalty;

    // See Vidal 2012, HGS for VRPTW
    proximityWeightWaitTime = 0.2;
    proximityWeightTimeWarp = 1;
}

double Params::getTimeElapsedSeconds() {
    if (config.useWallClockTime) {
        std::chrono::duration<double> wctduration = (std::chrono::system_clock::now() - startWallClockTime);
        return wctduration.count();
    }
    return (std::clock() - startCPUTime) / (double)CLOCKS_PER_SEC;
}

bool Params::isTimeLimitExceeded() { return getTimeElapsedSeconds() >= config.timeLimit; }

void Params::SetCorrelatedVertices() {
    // Calculation of the correlated vertices for each client (for the granular restriction)
    correlatedVertices = std::vector<std::vector<int>>(nbClients + 1);

    // First create a set of correlated vertices for each vertex (where the depot is not taken into account)
    std::vector<std::set<int>> setCorrelatedVertices = std::vector<std::set<int>>(nbClients + 1);

    // Loop over all clients (excluding the depot)
    for (int i = 1; i <= nbClients; i++) {
        auto &orderProximity = orderProximities[i];

        // Loop over all clients (taking into account the max number of clients and the granular restriction)
        for (int j = 0; j < std::min(config.nbGranular, nbClients - 1); j++) {
            // If i is correlated with j, then j should be correlated with i (unless we have asymmetric problem with
            // time windows) Insert vertices in setCorrelatedVertices, in the order of orderProximity, where .second is
            // used since the first index correponds to the depot
            setCorrelatedVertices[i].insert(orderProximity[j].second);

            // For symmetric problems, set the other entry to the same value
            if (config.useSymmetricCorrelatedVertices) { setCorrelatedVertices[orderProximity[j].second].insert(i); }
        }
    }

    // Now, fill the vector of correlated vertices, using setCorrelatedVertices
    for (int i = 1; i <= nbClients; i++) {
        for (int x : setCorrelatedVertices[i]) {
            // Add x at the end of the vector
            correlatedVertices[i].push_back(x);
        }
    }
}
