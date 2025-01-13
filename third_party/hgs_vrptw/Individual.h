/*MIT License

Original HGS-CVRP code: Copyright(c) 2020 Thibaut Vidal
Additional contributions: Copyright(c) 2022 ORTEC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include <set>
#include <string>
#include <vector>

#include "Params.h"

// Object to store all relevant information that may be needed to calculate some
// cost corresponding to a solution
struct CostSol {
    double penalizedCost;  // Penalized cost of the solution
    int nbRoutes;          // Number of routes
    int distance;          // Total Distance
    int capacityExcess;    // Total excess load over all routes
    // int    durationExcess; // Total excess duration over all routes
    int waitTime;     // Total wait time (time to wait to meet earliest possible
                      // arrival) over all routes
    int timeWarp;     // Total time warp (going back in time to meet latest
                      // possible arrival) over all routes
    bool isFeasible;  // Feasibility status of the individual

    // Constructor, initialize everything with 0
    CostSol()
        : penalizedCost(0.),
          nbRoutes(0),
          distance(0),
          capacityExcess(0),
          waitTime(0),
          timeWarp(0) {}
};

// Object to represent one individual/solution of a population.
class Individual {
   public:
    bool compareIndividuals(const Individual *a, const Individual *b);
    int adjustToTimeWindow(int time, const std::vector<TimeWindow> &timeWindows,
                           int &waitTime, int &timeWarp);

    Params *params;  // Problem parameters
    int nbVehicles;
    CostSol myCostSol;        // Information on the cost of the solution
    std::vector<int> chromT;  // Giant tour representing the individual: list of
                              // integers representing clients (can not be the
                              // depot 0). Size is nbClients
    std::vector<std::vector<int>>
        chromR;  // For each vehicle, the associated sequence of deliveries
                 // (complete solution). Size is nbVehicles. Routes are stored
                 // starting index maxVehicles-1, so the first indices will
                 // likely be empty
    std::vector<int>
        successors;  // For each node, the successor in the solution (can be the
                     // depot 0). Size is nbClients+1
    std::vector<int>
        predecessors;  // For each node, the predecessor in the solution (can be
                       // the depot 0). Size is nbClients+1
    std::multiset<std::pair<double, Individual *>>
        indivsPerProximity;  // The other individuals in the population (can not
                             // be the depot 0), ordered by increasing proximity
                             // (the set container follows a natural ordering
                             // based on the value of the first pair)
    bool isFeasible;         // Feasibility status of the individual
    double biasedFitness;    // Biased fitness of the solution

    // Measuring cost of a solution from the information of chromR
    void evaluateCompleteCost();

    // Initializes and shuffles chromT, needs to call evaluateCompleteCost
    // afterwards;
    // TODO: From line above: "needs to call evaluateCompleteCost afterwards;"
    void shuffleChromT();

    // Removing the individual indiv from the structure of proximity
    void removeProximity(Individual *indiv);

    // Distance measure with another individual, based on the number of arcs
    // that differ between two solutions
    double brokenPairsDistance(Individual *indiv2);

    // Returns the average distance of this individual with the nbClosest
    // individuals
    double averageBrokenPairsDistanceClosest(int nbClosest);

    // Exports a solution in CVRPLib format (adds a final line with the
    // computational time)
    void exportCVRPLibFormat(std::string fileName);

    // Prints a solution in CVRPLib format (adds a final line with the
    // computational time)
    void printCVRPLibFormat();

    // Reads a solution in CVRPLib format, returns TRUE if the process worked,
    // or FALSE if the file does not exist or is not readable
    static bool readCVRPLibFormat(std::string fileName,
                                  std::vector<std::vector<int>> &readSolution,
                                  double &readCost);

    // Constructor: create a random individual
    Individual(Params *params, bool initializeChromTAndShuffle = true);

    // Constructor: create an individual from a giant tour representation
    // including 0 for depot
    Individual(Params *params, std::string solutionStr);

    Individual(Params *params, bool rcws,
               std::vector<std::vector<int>> *pattern);

    // Constructor: create an empty individual
    Individual();

    bool is_time_window_feasible(int customer, const std::vector<int> &route,
                                 const Params &params) {
        double arrival_time = 0.0;

        // Initial vehicle departure time
        arrival_time =
            params.cli[0]
                .earliestArrival;  // assuming depot has readyTime as start

        // Compute the arrival times at each customer in the route
        for (int i = 0; i < route.size(); ++i) {
            int current_customer = route[i];
            double travel_time = params.timeCost[customer][current_customer];

            // Update arrival time
            arrival_time += travel_time;

            // Check if arrival time exceeds the latest allowable time window
            // for this customer
            if (arrival_time > params.cli[current_customer].latestArrival) {
                return false;
            }

            // If arrival is too early, wait until the customer's readyTime
            if (arrival_time < params.cli[current_customer].earliestArrival) {
                arrival_time = params.cli[current_customer].earliestArrival;
            }

            // Add service time at the current customer
            arrival_time += params.cli[current_customer].serviceDuration;
        }

        return true;
    }

    void update_arrival_times(const std::vector<int> &route,
                              std::vector<double> &arrival_times,
                              const Params &params) {
        if (arrival_times.size() < route.size()) {
            arrival_times.resize(route.size());
        }

        double arrival_time =
            params.cli[0].earliestArrival;  // Start from the depot

        for (int i = 0; i < route.size(); ++i) {
            int current_customer = route[i];

            // Compute travel time from the previous customer (or depot)
            double travel_time =
                (i == 0) ? params.timeCost[0][current_customer]
                         : params.timeCost[route[i - 1]][current_customer];
            arrival_time += travel_time;

            // Check if we arrive too early, and adjust arrival time accordingly
            if (arrival_time < params.cli[current_customer].earliestArrival) {
                arrival_time = params.cli[current_customer].earliestArrival;
            }

            // Update the arrival time for this customer
            arrival_times[i] = arrival_time;

            // Add service time at the current customer
            arrival_time += params.cli[current_customer].serviceDuration;
        }
    }
};

#endif
