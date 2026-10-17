#include "LocalSearch.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

#include "../third_party/pdqsort.h"
#include "CircleSector.h"
#include "Individual.h"
#include "Params.h"

bool operator==(const TimeWindowData &twData1, const TimeWindowData &twData2) {
    return twData1.firstNodeIndex == twData2.firstNodeIndex &&
           twData1.lastNodeIndex == twData2.lastNodeIndex &&
           twData1.duration == twData2.duration &&
           twData1.timeWarp == twData2.timeWarp &&
           twData1.earliestArrival == twData2.earliestArrival &&
           twData1.latestArrival == twData2.latestArrival;
}

bool cmpd(double a, double b, double eps = 1e-5) {
    return std::fabs(a - b) < eps;
}
void HGSLocalSearch::initializeConstruction(
    Individual *indiv, std::vector<NodeToInsert> *nodesToInsert) {
    // Initialize datastructures relevant for constructions.
    // Local search-related data structures are not initialized.
    emptyRoutes.clear();
    TimeWindowData depotTwData;
    depotTwData.firstNodeIndex = 0;
    depotTwData.lastNodeIndex = 0;
    depotTwData.duration = 0;
    depotTwData.timeWarp = 0;
    depotTwData.earliestArrival = params->cli[0].earliestArrival;
    depotTwData.latestArrival = params->cli[0].latestArrival;

    // Initializing time window data for clients
    for (int i = 1; i <= params->nbClients; i++) {
        // clients[i].twDataList.clear(); // Clear previous time window data (if
        // any)

        /*
                if (params->cli[i].n_tw > 0) {
                    // If the client has multiple time windows, initialize all
           of them for (int twIdx = 0; twIdx < params->cli[i].n_tw; twIdx++) {
                        TimeWindowData myTwData;
                        myTwData.firstNodeIndex  = i;
                        myTwData.lastNodeIndex   = i;
                        myTwData.duration        =
           params->cli[i].serviceDuration; myTwData.earliestArrival =
           params->cli[i].timeWindows[twIdx].earliestArrival;
                        myTwData.latestArrival   =
           params->cli[i].timeWindows[twIdx].latestArrival;

                        clients[i].twDataList.push_back(myTwData); // Store the
           multiple time windows
                    }
                } else {
                */
        // Initializing time window data for clients
        for (int i = 1; i <= params->nbClients; i++) {
            TimeWindowData *myTwData = &clients[i].twData;
            myTwData->firstNodeIndex = i;
            myTwData->lastNodeIndex = i;
            myTwData->duration = params->cli[i].serviceDuration;
            myTwData->earliestArrival = params->cli[i].earliestArrival;
            myTwData->latestArrival = params->cli[i].latestArrival;
        }
    }

    // Initialize routes
    for (int r = 0; r < params->nbVehicles; r++) {
        Node *myDepot = &depots[r];
        Node *myDepotFin = &depotsEnd[r];
        myDepot->prev = myDepotFin;
        myDepotFin->next = myDepot;
        myDepot->next = myDepotFin;
        myDepotFin->prev = myDepot;

        myDepot->twData = depotTwData;
        myDepot->prefixTwData = depotTwData;
        myDepot->postfixTwData = depotTwData;

        myDepotFin->twData = depotTwData;
        myDepotFin->prefixTwData = depotTwData;
        myDepotFin->postfixTwData = depotTwData;

        updateRouteData(&routes[r]);
    }

    // Initialize clients with their time window data
    for (int i = 1; i <= params->nbClients; i++) {
        NodeToInsert nodeToInsert;
        nodeToInsert.clientIdx = i;
        nodeToInsert.load = params->cli[i].demand;
        nodeToInsert.angleFromDepot =
            atan2(params->cli[i].coordY - params->cli[0].coordY,
                  params->cli[i].coordX - params->cli[0].coordX);
        nodeToInsert.serviceDuration = params->cli[i].serviceDuration;

        // Store the time windows in NodeToInsert
        // if (params->cli[i].n_tw > 0)
        //    nodeToInsert.twDataList = clients[i].twDataList;
        // else
        nodeToInsert.twData = clients[i].twData;

        nodesToInsert->push_back(nodeToInsert);
    }
}

void HGSLocalSearch::constructIndividualBySweep(int fillPercentage,
                                                Individual *indiv) {
    // Prepare the list of nodes to insert.
    std::vector<NodeToInsert> nodesToInsert;
    initializeConstruction(indiv, &nodesToInsert);

    // Sort nodes by the angle they form with the depot.
    pdqsort(nodesToInsert.begin(), nodesToInsert.end(),
            [](const NodeToInsert &a, const NodeToInsert &b) {
                return a.angleFromDepot < b.angleFromDepot;
            });

    // Distribute nodes across routes using a cumulative load approach.
    std::vector<std::vector<int>> nodeIndicesPerRoute;
    std::vector<int> currentRouteIndices;
    int currentLoad = 0;
    int nNodes = static_cast<int>(nodesToInsert.size());
    for (int i = 0; i < nNodes; i++) {
        // Check if adding the next node exceeds the fill threshold
        if (currentLoad > 0 &&
            currentLoad + nodesToInsert[i].load >
                fillPercentage * params->vehicleCapacity / 100 &&
            static_cast<int>(nodeIndicesPerRoute.size()) + 1 <
                static_cast<int>(routes.size())) {
            nodeIndicesPerRoute.push_back(currentRouteIndices);
            currentRouteIndices.clear();
            currentLoad = 0;
        }
        currentLoad += nodesToInsert[i].load;
        currentRouteIndices.push_back(i);
    }
    nodeIndicesPerRoute.push_back(currentRouteIndices);

    // Construct each route based on the assigned node indices.
    for (size_t r = 0; r < nodeIndicesPerRoute.size(); r++) {
        // Compute the depot's time window duration.
        int depotOpeningDuration =
            depots[r].twData.latestArrival - depots[r].twData.earliestArrival;
        std::vector<int> shortTWIndices;
        std::vector<int> longTWIndices;

        // Partition nodes into those with short and long time windows.
        for (int idx : nodeIndicesPerRoute[r]) {
            int twDuration = nodesToInsert[idx].twData.latestArrival -
                             nodesToInsert[idx].twData.earliestArrival;
            // Arbitrary division: if the time window duration is more than half
            // the depot's duration, consider it "long."
            if (static_cast<long>(twDuration) * 2 > depotOpeningDuration)
                longTWIndices.push_back(idx);
            else
                shortTWIndices.push_back(idx);
        }

        // Sort short time-window nodes by increasing latest arrival time.
        pdqsort(shortTWIndices.begin(), shortTWIndices.end(),
                [&nodesToInsert](int a, int b) {
                    return nodesToInsert[a].twData.latestArrival <
                           nodesToInsert[b].twData.latestArrival;
                });

        // Insert short time-window nodes sequentially after the depot.
        Node *prev = routes[r].depot;
        for (int idx : shortTWIndices) {
            Node *toInsert = &clients[nodesToInsert[idx].clientIdx];
            // Insert toInsert after prev.
            toInsert->prev = prev;
            toInsert->next = prev->next;
            prev->next->prev = toInsert;
            prev->next = toInsert;
            prev = toInsert;
        }
        updateRouteData(&routes[r]);

        // Insert long time-window nodes at the best possible position based on
        // insertion cost.
        for (int idx : longTWIndices) {
            double bestCost = std::numeric_limits<double>::max();
            Node *bestPredecessor = nullptr;
            Node *curr = routes[r].depot;
            // Try all possible insertion positions along the route.
            for (int j = 0; j <= routes[r].nbCustomers; j++) {
                double insertionCost =
                    params->timeCost.get(curr->cour,
                                         nodesToInsert[idx].clientIdx) +
                    params->timeCost.get(nodesToInsert[idx].clientIdx,
                                         curr->next->cour) -
                    params->timeCost.get(curr->cour, curr->next->cour);
                if (insertionCost < bestCost) {
                    bestCost = insertionCost;
                    bestPredecessor = curr;
                }
                curr = curr->next;
            }
            Node *toInsert = &clients[nodesToInsert[idx].clientIdx];
            // Insert toInsert after the bestPredecessor.
            toInsert->prev = bestPredecessor;
            toInsert->next = bestPredecessor->next;
            bestPredecessor->next->prev = toInsert;
            bestPredecessor->next = toInsert;
            updateRouteData(&routes[r]);
        }
    }

    // Export the constructed solution into the individual.
    exportIndividual(indiv);
}

void HGSLocalSearch::constructIndividualWithSeedOrder(
    int toleratedCapacityViolation, int toleratedTimeWarp,
    bool useSeedClientFurthestFromDepot, Individual *indiv) {
    // Prepare the list of nodes to insert.
    std::vector<NodeToInsert> nodesToInsert;
    initializeConstruction(indiv, &nodesToInsert);

    // Create a set of indices for all unassigned clients (0-indexed).
    std::set<int> unassignedIndices;
    for (int i = 0; i < params->nbClients; i++) {
        unassignedIndices.insert(i);
    }

    // Construct routes one by one.
    for (int r = 0;
         r < static_cast<int>(routes.size()) && !unassignedIndices.empty();
         r++) {
        // If using a seed based on the furthest client from the depot,
        // find the unassigned node with the highest insertion cost.
        if (useSeedClientFurthestFromDepot) {
            int furthestIdx = -1;
            double maxInsertionCost = -1.0;
            // Evaluate each unassigned node.
            for (int idx : unassignedIndices) {
                double costToInsert =
                    params->timeCost.get(routes[r].depot->cour,
                                         nodesToInsert[idx].clientIdx) +
                    params->timeCost.get(nodesToInsert[idx].clientIdx,
                                         routes[r].depot->next->cour) -
                    params->timeCost.get(routes[r].depot->cour,
                                         routes[r].depot->next->cour);
                if (costToInsert > maxInsertionCost) {
                    maxInsertionCost = costToInsert;
                    furthestIdx = idx;
                }
            }

            // Insert the selected seed client right after the depot.
            Node *seedNode = &clients[nodesToInsert[furthestIdx].clientIdx];
            seedNode->prev = routes[r].depot;
            seedNode->next = routes[r].depot->next;
            routes[r].depot->next->prev = seedNode;
            routes[r].depot->next = seedNode;
            updateRouteData(&routes[r]);
            unassignedIndices.erase(furthestIdx);
        }

        // Try to insert as many unassigned nodes as possible into the current
        // route.
        bool nodeInserted = true;
        while (nodeInserted) {
            nodeInserted = false;
            double bestInsertionCost = std::numeric_limits<double>::max();
            Node *bestPredecessor = nullptr;
            int bestNodeIdx = -1;

            // Loop over all unassigned nodes to find the best insertion
            // position.
            for (int idx : unassignedIndices) {
                // Check capacity: skip if adding this node exceeds capacity
                // tolerance.
                if (routes[r].load + nodesToInsert[idx].load >
                    params->vehicleCapacity + toleratedCapacityViolation)
                    continue;

                // Try every possible insertion position in the current route.
                Node *prev = routes[r].depot;
                for (int j = 0; j <= routes[r].nbCustomers; j++) {
                    // Merge time window data from the route prefix, candidate
                    // node, and suffix.
                    TimeWindowData mergedTW = MergeTWDataRecursive(
                        prev->prefixTwData, nodesToInsert[idx].twData,
                        prev->next->postfixTwData);
                    // Skip this insertion if the time warp violation is too
                    // high.
                    if (mergedTW.timeWarp > toleratedTimeWarp) {
                        prev = prev->next;
                        continue;
                    }

                    // Calculate the cost increase from inserting here.
                    double insertionCost =
                        params->timeCost.get(prev->cour,
                                             nodesToInsert[idx].clientIdx) +
                        params->timeCost.get(nodesToInsert[idx].clientIdx,
                                             prev->next->cour) -
                        params->timeCost.get(prev->cour, prev->next->cour);

                    if (insertionCost < bestInsertionCost) {
                        bestInsertionCost = insertionCost;
                        bestPredecessor = prev;
                        bestNodeIdx = idx;
                    }
                    prev = prev->next;
                }
            }

            // If a valid insertion is found, perform the insertion.
            if (bestInsertionCost < std::numeric_limits<double>::max()) {
                Node *nodeToInsert =
                    &clients[nodesToInsert[bestNodeIdx].clientIdx];
                nodeToInsert->prev = bestPredecessor;
                nodeToInsert->next = bestPredecessor->next;
                bestPredecessor->next->prev = nodeToInsert;
                bestPredecessor->next = nodeToInsert;
                updateRouteData(&routes[r]);
                unassignedIndices.erase(bestNodeIdx);
                nodeInserted = true;
            }
        }
    }

    // If there are any remaining unassigned nodes, append them at the end of
    // the last route.
    if (!unassignedIndices.empty()) {
        int lastRouteIdx = routes.size() - 1;
        // Get the last node before the finish depot.
        Node *prevNode = depotsEnd[lastRouteIdx].prev;
        while (!unassignedIndices.empty()) {
            int idx = *unassignedIndices.begin();
            Node *nodeToInsert = &clients[nodesToInsert[idx].clientIdx];
            nodeToInsert->prev = prevNode;
            nodeToInsert->next = prevNode->next;
            prevNode->next->prev = nodeToInsert;
            prevNode->next = nodeToInsert;
            unassignedIndices.erase(idx);
        }
        updateRouteData(&routes[lastRouteIdx]);
    }

    // Finalize by exporting the constructed solution.
    exportIndividual(indiv);
}

void HGSLocalSearch::run(Individual *indiv, double penaltyCapacityLS,
                         double penaltyTimeWarpLS) {
    // Pre-calculate intensification flags.
    static const bool neverIntensify =
        (params->config.intensificationProbabilityLS == 0);
    static const bool alwaysIntensify =
        (params->config.intensificationProbabilityLS == 100);
    const bool runLS_INT =
        (params->rng() % 100) <
        static_cast<unsigned int>(params->config.intensificationProbabilityLS);

    // Set penalty values based on problem type.
    this->penaltyCapacityLS = penaltyCapacityLS;
    this->penaltyTimeWarpLS = (params->problemType == ProblemType::vrptw ||
                               params->problemType == ProblemType::evrp)
                                  ? penaltyTimeWarpLS
                                  : 0;
    loadIndividual(indiv);

    // Shuffle nodes and routes to diversify the search.
    std::shuffle(orderNodes.begin(), orderNodes.end(), params->rng);
    std::shuffle(orderRoutes.begin(), orderRoutes.end(), params->rng);
    for (int i = 1; i <= params->nbClients; i++) {
        // Shuffle correlated vertices sporadically to avoid O(n^2) behavior.
        if (params->rng() % params->config.nbGranular == 0) {
            std::shuffle(params->correlatedVertices[i].begin(),
                         params->correlatedVertices[i].end(), params->rng);
        }
    }

    // Main local search loop.
    searchCompleted = false;
    for (loopID = 0; !searchCompleted; loopID++) {
        // After at least two iterations, allow termination.
        if (loopID > 1) searchCompleted = true;

        // --- Phase 1: Classical Route Improvement (RI) Moves ---
        for (int posU = 0; posU < params->nbClients; posU++) {
            nodeU = &clients[orderNodes[posU]];
            int lastTestRINodeU = nodeU->whenLastTestedRI;
            nodeU->whenLastTestedRI = nbMoves;

            // Process each correlated neighbor for nodeU.
            const auto &correlated = params->correlatedVertices[nodeU->cour];
            for (const auto &v : correlated) {
                nodeV = &clients[v];
                // Evaluate only if at least one route was modified after
                // nodeU's last test.
                if (loopID == 0 || std::max(nodeU->route->whenLastModified,
                                            nodeV->route->whenLastModified) >
                                       lastTestRINodeU) {
                    // Reset local route variables for nodeU and nodeV.
                    setLocalbaldesVarsRouteU();
                    setLocalbaldesVarsRouteV();

                    // Attempt various moves sequentially.
                    if (MoveSingleClient()) continue;
                    if (MoveTwoClients()) continue;
                    if (MoveTwoClientsReversed()) continue;
                    if (nodeUIndex < nodeVIndex && SwapTwoSingleClients())
                        continue;
                    if (SwapTwoClientsForOne()) continue;
                    if (nodeUIndex < nodeVIndex && SwapTwoClientPairs())
                        continue;
                    if (routeU->cour < routeV->cour && TwoOptBetweenTrips())
                        continue;
                    if (routeU == routeV && TwoOptWithinTrip()) continue;

                    // Additional attempt: try inserting nodeU after the depot
                    // if nodeV's predecessor is the depot.
                    if (nodeV->prev->isDepot) {
                        nodeV = nodeV->prev;
                        setLocalbaldesVarsRouteV();
                        if (MoveSingleClient()) continue;
                        if (MoveTwoClients()) continue;
                        if (MoveTwoClientsReversed()) continue;
                        if (routeU->cour < nodeV->route->cour &&
                            TwoOptBetweenTrips())
                            continue;
                    }
                }
            }

            // --- Phase 2: Moves Involving an Empty Route ---
            if (loopID > 0 && !emptyRoutes.empty()) {
                nodeV = routes[*emptyRoutes.begin()].depot;
                setLocalbaldesVarsRouteU();
                setLocalbaldesVarsRouteV();
                if (MoveSingleClient()) continue;
                if (MoveTwoClients()) continue;
                if (MoveTwoClientsReversed()) continue;
                if (TwoOptBetweenTrips()) continue;
            }
        }

        // --- Phase 3: Intensification Moves (SWAP* on Overlapping Route Pairs)
        // ---
        if (!neverIntensify && searchCompleted &&
            (alwaysIntensify || runLS_INT)) {
            for (int rU = 0; rU < params->nbVehicles; rU++) {
                routeU = &routes[orderRoutes[rU]];
                if (routeU->nbCustomers == 0) continue;

                int lastTestLargeNbRouteU = routeU->whenLastTestedLargeNb;
                routeU->whenLastTestedLargeNb = nbMoves;

                for (int rV = 0; rV < params->nbVehicles; rV++) {
                    routeV = &routes[orderRoutes[rV]];
                    // Skip if routeV is empty or if routeU's courier is not
                    // less than routeV's.
                    if (routeV->nbCustomers == 0 ||
                        routeU->cour >= routeV->cour)
                        continue;

                    // Only proceed if at least one route has been recently
                    // modified.
                    if (loopID > 0 && std::max(routeU->whenLastModified,
                                               routeV->whenLastModified) <=
                                          lastTestLargeNbRouteU)
                        continue;

                    // Process only route pairs with overlapping circle sectors.
                    if (!CircleSector::overlap(
                            routeU->sector, routeV->sector,
                            params->circleSectorOverlapTolerance))
                        continue;

                    // Attempt RelocateStar; if unsuccessful, try variations of
                    // swapStar.
                    if (!RelocateStar()) {
                        if (params->config.skipSwapStarDist ||
                            !swapStar(false)) {
                            if (params->config.useSwapStarTW) swapStar(true);
                        }
                    }
                }
            }
        }
    }

    // --- Final Step: Export the locally improved solution back to the
    // individual ---
    exportIndividual(indiv);
}

void HGSLocalSearch::setLocalbaldesVarsRouteU() {
    routeU = nodeU->route;
    nodeX = nodeU->next;
    nodeXNextIndex = nodeX->next->cour;
    nodeUIndex = nodeU->cour;
    nodeUPrevIndex = nodeU->prev->cour;
    nodeXIndex = nodeX->cour;
    loadU = params->cli[nodeUIndex].demand;
    serviceU = params->cli[nodeUIndex].serviceDuration;
    loadX = params->cli[nodeXIndex].demand;
    serviceX = params->cli[nodeXIndex].serviceDuration;
    if (params->problemType == ProblemType::vrptw ||
        params->problemType == ProblemType::evrp) {
        routeUTimeWarp = routeU->twData.timeWarp > 0;
    } else {
        routeUTimeWarp = false;
    }
    routeULoadPenalty = routeU->load > params->vehicleCapacity;
}

void HGSLocalSearch::setLocalbaldesVarsRouteV() {
    routeV = nodeV->route;
    nodeY = nodeV->next;
    nodeYNextIndex = nodeY->next->cour;
    nodeVIndex = nodeV->cour;
    nodeVPrevIndex = nodeV->prev->cour;
    nodeYIndex = nodeY->cour;
    loadV = params->cli[nodeVIndex].demand;
    serviceV = params->cli[nodeVIndex].serviceDuration;
    loadY = params->cli[nodeYIndex].demand;
    serviceY = params->cli[nodeYIndex].serviceDuration;
    if (params->problemType == ProblemType::vrptw ||
        params->problemType == ProblemType::evrp) {
        routeVTimeWarp = routeV->twData.timeWarp > 0;
    } else {
        routeVTimeWarp = false;
    }
    routeVLoadPenalty = routeV->load > params->vehicleCapacity;
}

bool HGSLocalSearch::MoveSingleClient() {
    // If U already comes directly after V, this move has no effect
    if (nodeUIndex == nodeYIndex) return false;

    double costSuppU = params->timeCost.get(nodeUPrevIndex, nodeXIndex) -
                       params->timeCost.get(nodeUPrevIndex, nodeUIndex) -
                       params->timeCost.get(nodeUIndex, nodeXIndex);
    double costSuppV = params->timeCost.get(nodeVIndex, nodeUIndex) +
                       params->timeCost.get(nodeUIndex, nodeYIndex) -
                       params->timeCost.get(nodeVIndex, nodeYIndex);
    TimeWindowData routeUTwData, routeVTwData;

    if (routeU != routeV) {
        if (!routeULoadPenalty && !routeUTimeWarp &&
            costSuppU + costSuppV > -MY_EPSILON) {
            return false;
        }

        routeUTwData = MergeTWDataRecursive(nodeU->prev->prefixTwData,
                                            nodeX->postfixTwData);
        routeVTwData = MergeTWDataRecursive(nodeV->prefixTwData, nodeU->twData,
                                            nodeY->postfixTwData);

        costSuppU += penaltyExcessLoad(routeU->load - loadU) +
                     penaltyTimeWindows(routeUTwData) - routeU->penalty;

        costSuppV += penaltyExcessLoad(routeV->load + loadU) +
                     penaltyTimeWindows(routeVTwData) - routeV->penalty;
    } else {
        if (!routeUTimeWarp && costSuppU + costSuppV > -MY_EPSILON) {
            return false;
        }

        // Move within the same route
        if (nodeU->position < nodeV->position) {
            // Edge case V directly after U, so X == V, this works
            // start - ... - UPrev - X - ... - V - U - Y - ... - end
            routeUTwData = MergeTWDataRecursive(
                nodeU->prev->prefixTwData, getRouteSegmentTwData(nodeX, nodeV),
                nodeU->twData, nodeY->postfixTwData);
        } else {
            // Edge case U directly after V is excluded from beginning of
            // function start - ... - V - U - Y - ... - UPrev - X - ... - end
            routeUTwData =
                MergeTWDataRecursive(nodeV->prefixTwData, nodeU->twData,
                                     getRouteSegmentTwData(nodeY, nodeU->prev),
                                     nodeX->postfixTwData);
        }

        // Compute new total penalty
        costSuppU += penaltyExcessLoad(routeU->load) +
                     penaltyTimeWindows(routeUTwData) - routeU->penalty;
    }

    if (costSuppU + costSuppV > -MY_EPSILON) return false;

    insertNode(nodeU, nodeV);
    nbMoves++;  // Increment move counter before updating route data
    searchCompleted = false;
    updateRouteData(routeU);
    if (routeU != routeV) updateRouteData(routeV);

    return true;
}
bool HGSLocalSearch::MoveTwoClients() {
    if (nodeU == nodeY || nodeV == nodeX || nodeX->isDepot) return false;

    // Cache timeCost results
    double costBeforeU = params->timeCost.get(nodeUPrevIndex, nodeXNextIndex);
    double costBeforeV = params->timeCost.get(nodeVIndex, nodeYIndex);

    double costSuppU = costBeforeU -
                       params->timeCost.get(nodeUPrevIndex, nodeUIndex) -
                       params->timeCost.get(nodeXIndex, nodeXNextIndex);
    double costSuppV = params->timeCost.get(nodeVIndex, nodeUIndex) +
                       params->timeCost.get(nodeXIndex, nodeYIndex) -
                       costBeforeV;

    TimeWindowData routeUTwData, routeVTwData;

    if (routeU != routeV) {
        if (!routeULoadPenalty && !routeUTimeWarp &&
            (costSuppU + costSuppV > -MY_EPSILON))
            return false;

        // Cache merged TW data
        routeUTwData = MergeTWDataRecursive(nodeU->prev->prefixTwData,
                                            nodeX->next->postfixTwData);
        routeVTwData = MergeTWDataRecursive(nodeV->prefixTwData,
                                            getEdgeTwData(nodeU, nodeX),
                                            nodeY->postfixTwData);

        // Adjust costSuppU and costSuppV
        costSuppU += penaltyExcessLoad(routeU->load - loadU - loadX) +
                     penaltyTimeWindows(routeUTwData) - routeU->penalty;
        costSuppV += penaltyExcessLoad(routeV->load + loadU + loadX) +
                     penaltyTimeWindows(routeVTwData) - routeV->penalty;
    } else {
        if (!routeUTimeWarp && (costSuppU + costSuppV > -MY_EPSILON))
            return false;

        // Move within the same route
        if (nodeU->position < nodeV->position) {
            routeUTwData = MergeTWDataRecursive(
                nodeU->prev->prefixTwData,
                getRouteSegmentTwData(nodeX->next, nodeV),
                getEdgeTwData(nodeU, nodeX), nodeY->postfixTwData);
        } else {
            routeUTwData = MergeTWDataRecursive(
                nodeV->prefixTwData, getEdgeTwData(nodeU, nodeX),
                getRouteSegmentTwData(nodeY, nodeU->prev),
                nodeX->next->postfixTwData);
        }

        costSuppU += penaltyExcessLoad(routeU->load) +
                     penaltyTimeWindows(routeUTwData) - routeU->penalty;
    }

    // Final cost comparison
    if (costSuppU + costSuppV > -MY_EPSILON) return false;

    // Perform node insertion and update route data
    insertNode(nodeU, nodeV);
    insertNode(nodeX, nodeU);
    nbMoves++;
    searchCompleted = false;
    updateRouteData(routeU);
    if (routeU != routeV) updateRouteData(routeV);

    return true;
}

bool HGSLocalSearch::MoveTwoClientsReversed() {
    if (nodeU == nodeY || nodeX == nodeV || nodeX->isDepot) return false;

    double costSuppU = params->timeCost.get(nodeUPrevIndex, nodeXNextIndex) -
                       params->timeCost.get(nodeUPrevIndex, nodeUIndex) -
                       params->timeCost.get(nodeUIndex, nodeXIndex) -
                       params->timeCost.get(nodeXIndex, nodeXNextIndex);
    double costSuppV = params->timeCost.get(nodeVIndex, nodeXIndex) +
                       params->timeCost.get(nodeXIndex, nodeUIndex) +
                       params->timeCost.get(nodeUIndex, nodeYIndex) -
                       params->timeCost.get(nodeVIndex, nodeYIndex);
    TimeWindowData routeUTwData, routeVTwData;

    if (routeU != routeV) {
        if (!routeULoadPenalty && !routeUTimeWarp &&
            costSuppU + costSuppV > -MY_EPSILON) {
            return false;
        }

        routeUTwData = MergeTWDataRecursive(nodeU->prev->prefixTwData,
                                            nodeX->next->postfixTwData);
        routeVTwData = MergeTWDataRecursive(nodeV->prefixTwData,
                                            getEdgeTwData(nodeX, nodeU),
                                            nodeY->postfixTwData);

        costSuppU += penaltyExcessLoad(routeU->load - loadU - loadX) +
                     penaltyTimeWindows(routeUTwData) - routeU->penalty;

        costSuppV += penaltyExcessLoad(routeV->load + loadU + loadX) +
                     penaltyTimeWindows(routeVTwData) - routeV->penalty;
    } else {
        if (!routeUTimeWarp && costSuppU + costSuppV > -MY_EPSILON) {
            return false;
        }

        // Move within the same route
        if (nodeU->position < nodeV->position) {
            // Edge case V directly after U, so X == V is excluded, V directly
            // after X so XNext == V works start - ... - UPrev - XNext - ... - V
            // - X - U - Y - ... - end
            routeUTwData = MergeTWDataRecursive(
                nodeU->prev->prefixTwData,
                getRouteSegmentTwData(nodeX->next, nodeV),
                getEdgeTwData(nodeX, nodeU), nodeY->postfixTwData);
        } else {
            // Edge case U directly after V is excluded from beginning of
            // function start - ... - V - X - U - Y - ... - UPrev - XNext - ...
            // - end
            routeUTwData = MergeTWDataRecursive(
                nodeV->prefixTwData, getEdgeTwData(nodeX, nodeU),
                getRouteSegmentTwData(nodeY, nodeU->prev),
                nodeX->next->postfixTwData);
        }

        // Compute new total penalty
        costSuppU += penaltyExcessLoad(routeU->load) +
                     penaltyTimeWindows(routeUTwData) - routeU->penalty;
    }

    if (costSuppU + costSuppV > -MY_EPSILON) return false;

    insertNode(nodeX, nodeV);
    insertNode(nodeU, nodeX);
    nbMoves++;  // Increment move counter before updating route data
    searchCompleted = false;
    updateRouteData(routeU);
    if (routeU != routeV) updateRouteData(routeV);

    return true;
}

bool HGSLocalSearch::SwapTwoSingleClients() {
    if (nodeUIndex == nodeVPrevIndex || nodeUIndex == nodeYIndex) return false;

    double costSuppU = params->timeCost.get(nodeUPrevIndex, nodeVIndex) +
                       params->timeCost.get(nodeVIndex, nodeXIndex) -
                       params->timeCost.get(nodeUPrevIndex, nodeUIndex) -
                       params->timeCost.get(nodeUIndex, nodeXIndex);
    double costSuppV = params->timeCost.get(nodeVPrevIndex, nodeUIndex) +
                       params->timeCost.get(nodeUIndex, nodeYIndex) -
                       params->timeCost.get(nodeVPrevIndex, nodeVIndex) -
                       params->timeCost.get(nodeVIndex, nodeYIndex);
    TimeWindowData routeUTwData, routeVTwData;

    if (routeU != routeV) {
        if (!routeULoadPenalty && !routeUTimeWarp && !routeVLoadPenalty &&
            !routeVTimeWarp && costSuppU + costSuppV > -MY_EPSILON) {
            return false;
        }

        routeUTwData = MergeTWDataRecursive(
            nodeU->prev->prefixTwData, nodeV->twData, nodeX->postfixTwData);
        routeVTwData = MergeTWDataRecursive(
            nodeV->prev->prefixTwData, nodeU->twData, nodeY->postfixTwData);

        costSuppU += penaltyExcessLoad(routeU->load + loadV - loadU) +
                     penaltyTimeWindows(routeUTwData) - routeU->penalty;

        costSuppV += penaltyExcessLoad(routeV->load + loadU - loadV) +
                     penaltyTimeWindows(routeVTwData) - routeV->penalty;
    } else {
        if (!routeUTimeWarp && costSuppU + costSuppV > -MY_EPSILON) {
            return false;
        }

        // Swap within the same route
        if (nodeU->position < nodeV->position) {
            // Edge case V directly after U, so X == V is excluded, V directly
            // after X so XNext == V works start - ... - UPrev - V - X - ... -
            // VPrev - U - Y - ... - end
            routeUTwData =
                MergeTWDataRecursive(nodeU->prev->prefixTwData, nodeV->twData,
                                     getRouteSegmentTwData(nodeX, nodeV->prev),
                                     nodeU->twData, nodeY->postfixTwData);
        } else {
            // Edge case U directly after V is excluded from beginning of
            // function start - ... - VPrev - U - Y - ... - UPrev - V - X - ...
            // - end
            routeUTwData =
                MergeTWDataRecursive(nodeV->prev->prefixTwData, nodeU->twData,
                                     getRouteSegmentTwData(nodeY, nodeU->prev),
                                     nodeV->twData, nodeX->postfixTwData);
        }

        // Compute new total penalty
        costSuppU += penaltyExcessLoad(routeU->load) +
                     penaltyTimeWindows(routeUTwData) - routeU->penalty;
    }

    if (costSuppU + costSuppV > -MY_EPSILON) return false;

    swapNode(nodeU, nodeV);
    nbMoves++;  // Increment move counter before updating route data
    searchCompleted = false;
    updateRouteData(routeU);
    if (routeU != routeV) updateRouteData(routeV);

    return true;
}

bool HGSLocalSearch::SwapTwoClientsForOne() {
    if (nodeU == nodeV->prev || nodeX == nodeV->prev || nodeU == nodeY ||
        nodeX->isDepot)
        return false;

    double costSuppU = params->timeCost.get(nodeUPrevIndex, nodeVIndex) +
                       params->timeCost.get(nodeVIndex, nodeXNextIndex) -
                       params->timeCost.get(nodeUPrevIndex, nodeUIndex) -
                       params->timeCost.get(nodeXIndex, nodeXNextIndex);
    double costSuppV = params->timeCost.get(nodeVPrevIndex, nodeUIndex) +
                       params->timeCost.get(nodeXIndex, nodeYIndex) -
                       params->timeCost.get(nodeVPrevIndex, nodeVIndex) -
                       params->timeCost.get(nodeVIndex, nodeYIndex);
    TimeWindowData routeUTwData, routeVTwData;

    if (routeU != routeV) {
        if (!routeULoadPenalty && !routeUTimeWarp && !routeVLoadPenalty &&
            !routeVTimeWarp && costSuppU + costSuppV > -MY_EPSILON) {
            return false;
        }

        routeUTwData =
            MergeTWDataRecursive(nodeU->prev->prefixTwData, nodeV->twData,
                                 nodeX->next->postfixTwData);
        routeVTwData = MergeTWDataRecursive(nodeV->prev->prefixTwData,
                                            getEdgeTwData(nodeU, nodeX),
                                            nodeY->postfixTwData);

        costSuppU += penaltyExcessLoad(routeU->load + loadV - loadU - loadX) +
                     penaltyTimeWindows(routeUTwData) - routeU->penalty;

        costSuppV += penaltyExcessLoad(routeV->load + loadU + loadX - loadV) +
                     penaltyTimeWindows(routeVTwData) - routeV->penalty;
    } else {
        if (!routeUTimeWarp && costSuppU + costSuppV > -MY_EPSILON) {
            return false;
        }

        // Swap within the same route
        if (nodeU->position < nodeV->position) {
            // start - ... - UPrev - V - XNext - ... - VPrev - U - X - Y - ... -
            // end
            routeUTwData = MergeTWDataRecursive(
                nodeU->prev->prefixTwData, nodeV->twData,
                getRouteSegmentTwData(nodeX->next, nodeV->prev),
                getEdgeTwData(nodeU, nodeX), nodeY->postfixTwData);
        } else {
            // start - ... - VPrev - U - X - Y - ... - UPrev - V - XNext - ... -
            // end
            routeUTwData = MergeTWDataRecursive(
                nodeV->prev->prefixTwData, getEdgeTwData(nodeU, nodeX),
                getRouteSegmentTwData(nodeY, nodeU->prev), nodeV->twData,
                nodeX->next->postfixTwData);
        }

        // Compute new total penalty
        costSuppU += penaltyExcessLoad(routeU->load) +
                     penaltyTimeWindows(routeUTwData) - routeU->penalty;
    }

    if (costSuppU + costSuppV > -MY_EPSILON) return false;

    // Note: next two lines are a bit inefficient but we only update
    // occasionally and updateRouteData is much more costly anyway, efficient
    // checks are more important
    swapNode(nodeU, nodeV);
    insertNode(nodeX, nodeU);
    nbMoves++;  // Increment move counter before updating route data
    searchCompleted = false;
    updateRouteData(routeU);
    if (routeU != routeV) updateRouteData(routeV);

    return true;
}

bool HGSLocalSearch::SwapTwoClientPairs() {
    if (nodeX->isDepot || nodeY->isDepot || nodeY == nodeU->prev ||
        nodeU == nodeY || nodeX == nodeV || nodeV == nodeX->next)
        return false;

    double costSuppU = params->timeCost.get(nodeUPrevIndex, nodeVIndex) +
                       params->timeCost.get(nodeYIndex, nodeXNextIndex) -
                       params->timeCost.get(nodeUPrevIndex, nodeUIndex) -
                       params->timeCost.get(nodeXIndex, nodeXNextIndex);
    double costSuppV = params->timeCost.get(nodeVPrevIndex, nodeUIndex) +
                       params->timeCost.get(nodeXIndex, nodeYNextIndex) -
                       params->timeCost.get(nodeVPrevIndex, nodeVIndex) -
                       params->timeCost.get(nodeYIndex, nodeYNextIndex);
    TimeWindowData routeUTwData, routeVTwData;

    if (routeU != routeV) {
        if (!routeULoadPenalty && !routeUTimeWarp && !routeVLoadPenalty &&
            !routeVTimeWarp && costSuppU + costSuppV > -MY_EPSILON) {
            return false;
        }

        routeUTwData = MergeTWDataRecursive(nodeU->prev->prefixTwData,
                                            getEdgeTwData(nodeV, nodeY),
                                            nodeX->next->postfixTwData);
        routeVTwData = MergeTWDataRecursive(nodeV->prev->prefixTwData,
                                            getEdgeTwData(nodeU, nodeX),
                                            nodeY->next->postfixTwData);

        costSuppU +=
            penaltyExcessLoad(routeU->load + loadV + loadY - loadU - loadX) +
            penaltyTimeWindows(routeUTwData) - routeU->penalty;

        costSuppV +=
            penaltyExcessLoad(routeV->load + loadU + loadX - loadV - loadY) +
            penaltyTimeWindows(routeVTwData) - routeV->penalty;
    } else {
        if (!routeUTimeWarp && costSuppU + costSuppV > -MY_EPSILON) {
            return false;
        }

        // Swap within the same route
        if (nodeU->position < nodeV->position) {
            // start - ... - UPrev - V - Y - XNext - ... - VPrev - U - X - YNext
            // - ... - end
            routeUTwData = MergeTWDataRecursive(
                nodeU->prev->prefixTwData, getEdgeTwData(nodeV, nodeY),
                getRouteSegmentTwData(nodeX->next, nodeV->prev),
                getEdgeTwData(nodeU, nodeX), nodeY->next->postfixTwData);
        } else {
            // start - ... - VPrev - U - X - YNext - ... - UPrev - V - Y - XNext
            // - ... - end
            routeUTwData = MergeTWDataRecursive(
                nodeV->prev->prefixTwData, getEdgeTwData(nodeU, nodeX),
                getRouteSegmentTwData(nodeY->next, nodeU->prev),
                getEdgeTwData(nodeV, nodeY), nodeX->next->postfixTwData);
        }

        // Compute new total penalty
        costSuppU += penaltyExcessLoad(routeU->load) +
                     penaltyTimeWindows(routeUTwData) - routeU->penalty;
    }

    if (costSuppU + costSuppV > -MY_EPSILON) return false;

    swapNode(nodeU, nodeV);
    swapNode(nodeX, nodeY);
    nbMoves++;  // Increment move counter before updating route data
    searchCompleted = false;
    updateRouteData(routeU);
    if (routeU != routeV) updateRouteData(routeV);

    return true;
}

bool HGSLocalSearch::TwoOptWithinTrip() {
    if (nodeU->position >= nodeV->position - 1) return false;

    double cost = params->timeCost.get(nodeUIndex, nodeVIndex) +
                  params->timeCost.get(nodeXIndex, nodeYIndex) -
                  params->timeCost.get(nodeUIndex, nodeXIndex) -
                  params->timeCost.get(nodeVIndex, nodeYIndex) +
                  nodeV->cumulatedReversalDistance -
                  nodeX->cumulatedReversalDistance;

    if (!routeUTimeWarp && cost > -MY_EPSILON) {
        return false;
    }

    TimeWindowData routeTwData = nodeU->prefixTwData;
    Node *itRoute = nodeV;
    while (itRoute != nodeU) {
        routeTwData = MergeTWDataRecursive(routeTwData, itRoute->twData);
        itRoute = itRoute->prev;
    }
    routeTwData = MergeTWDataRecursive(routeTwData, nodeY->postfixTwData);

    // Compute new total penalty
    cost += penaltyExcessLoad(routeU->load) + penaltyTimeWindows(routeTwData) -
            routeU->penalty;

    if (cost > -MY_EPSILON) {
        return false;
    }

    itRoute = nodeV;
    Node *insertionPoint = nodeU;
    while (itRoute != nodeX)  // No need to move x, we pivot around it
    {
        Node *current = itRoute;
        itRoute = itRoute->prev;
        insertNode(current, insertionPoint);
        insertionPoint = current;
    }

    nbMoves++;  // Increment move counter before updating route data
    searchCompleted = false;
    updateRouteData(routeU);

    return true;
}

bool HGSLocalSearch::TwoOptBetweenTrips() {
    double costSuppU = params->timeCost.get(nodeUIndex, nodeYIndex) -
                       params->timeCost.get(nodeUIndex, nodeXIndex);
    double costSuppV = params->timeCost.get(nodeVIndex, nodeXIndex) -
                       params->timeCost.get(nodeVIndex, nodeYIndex);

    if (!routeULoadPenalty && !routeUTimeWarp && !routeVLoadPenalty &&
        !routeVTimeWarp && costSuppU + costSuppV > -MY_EPSILON) {
        return false;
    }

    TimeWindowData routeUTwData, routeVTwData;

    routeUTwData =
        MergeTWDataRecursive(nodeU->prefixTwData, nodeY->postfixTwData);
    routeVTwData =
        MergeTWDataRecursive(nodeV->prefixTwData, nodeX->postfixTwData);

    costSuppU += penaltyExcessLoad(nodeU->cumulatedLoad + routeV->load -
                                   nodeV->cumulatedLoad) +
                 penaltyTimeWindows(routeUTwData) - routeU->penalty;

    costSuppV += penaltyExcessLoad(nodeV->cumulatedLoad + routeU->load -
                                   nodeU->cumulatedLoad) +
                 penaltyTimeWindows(routeVTwData) - routeV->penalty;

    if (costSuppU + costSuppV > -MY_EPSILON) return false;

    Node *itRouteV = nodeY;
    Node *insertLocation = nodeU;
    while (!itRouteV->isDepot) {
        Node *current = itRouteV;
        itRouteV = itRouteV->next;
        insertNode(current, insertLocation);
        insertLocation = current;
    }

    Node *itRouteU = nodeX;
    insertLocation = nodeV;
    while (!itRouteU->isDepot) {
        Node *current = itRouteU;
        itRouteU = itRouteU->next;
        insertNode(current, insertLocation);
        insertLocation = current;
    }

    nbMoves++;  // Increment move counter before updating route data
    searchCompleted = false;
    updateRouteData(routeU);
    updateRouteData(routeV);

    return true;
}

bool HGSLocalSearch::swapStar(const bool withTW) {
    SwapStarElement myBestSwapStar;

    if (!bestInsertInitializedForRoute[routeU->cour]) {
        bestInsertInitializedForRoute[routeU->cour] = true;
        for (int i = 1; i <= params->nbClients; i++) {
            bestInsertClient[routeU->cour][i].whenLastCalculated = -1;
            bestInsertClientTW[routeU->cour][i].whenLastCalculated = -1;
        }
    }
    if (!bestInsertInitializedForRoute[routeV->cour]) {
        bestInsertInitializedForRoute[routeV->cour] = true;
        for (int i = 1; i <= params->nbClients; i++) {
            bestInsertClient[routeV->cour][i].whenLastCalculated = -1;
            bestInsertClientTW[routeV->cour][i].whenLastCalculated = -1;
        }
    }

    // Preprocessing insertion costs
    if (withTW) {
        preprocessInsertionsWithTW(routeU, routeV);
        preprocessInsertionsWithTW(routeV, routeU);
    } else {
        preprocessInsertions(routeU, routeV);
        preprocessInsertions(routeV, routeU);
    }

    // Evaluating the moves
    for (nodeU = routeU->depot->next; !nodeU->isDepot; nodeU = nodeU->next) {
        for (nodeV = routeV->depot->next; !nodeV->isDepot;
             nodeV = nodeV->next) {
            // We cannot determine impact on timewarp without adding too much
            // complexity (O(n^3) instead of O(n^2))
            const double loadPenU = penaltyExcessLoad(
                routeU->load + params->cli[nodeV->cour].demand -
                params->cli[nodeU->cour].demand);
            const double loadPenV = penaltyExcessLoad(
                routeV->load + params->cli[nodeU->cour].demand -
                params->cli[nodeV->cour].demand);
            const double deltaLoadPen = loadPenU + loadPenV -
                                        penaltyExcessLoad(routeU->load) -
                                        penaltyExcessLoad(routeV->load);
            const int deltaRemoval =
                withTW ? nodeU->deltaRemovalTW + nodeV->deltaRemovalTW
                       : nodeU->deltaRemoval + nodeV->deltaRemoval;

            // Quick filter: possibly early elimination of many SWAP* due to the
            // capacity constraints/penalties and bounds on insertion costs
            if (deltaLoadPen + deltaRemoval <= 0) {
                SwapStarElement mySwapStar;
                mySwapStar.U = nodeU;
                mySwapStar.V = nodeV;

                int extraV, extraU;
                if (withTW) {
                    // Evaluate best reinsertion cost of U in the route of V
                    // where V has been removed
                    extraV = getCheapestInsertSimultRemovalWithTW(
                        nodeU, nodeV, mySwapStar.bestPositionU);

                    // Evaluate best reinsertion cost of V in the route of U
                    // where U has been removed
                    extraU = getCheapestInsertSimultRemovalWithTW(
                        nodeV, nodeU, mySwapStar.bestPositionV);
                } else {
                    // Evaluate best reinsertion cost of U in the route of V
                    // where V has been removed
                    extraV = getCheapestInsertSimultRemoval(
                        nodeU, nodeV, mySwapStar.bestPositionU);

                    // Evaluate best reinsertion cost of V in the route of U
                    // where U has been removed
                    extraU = getCheapestInsertSimultRemoval(
                        nodeV, nodeU, mySwapStar.bestPositionV);
                }

                // Evaluating final cost
                mySwapStar.moveCost =
                    deltaLoadPen + deltaRemoval + extraU + extraV;

                if (mySwapStar.moveCost < myBestSwapStar.moveCost) {
                    myBestSwapStar = mySwapStar;
                    myBestSwapStar.loadPenU = loadPenU;
                    myBestSwapStar.loadPenV = loadPenV;
                }
            }
        }
    }

    if (!myBestSwapStar.bestPositionU || !myBestSwapStar.bestPositionV) {
        return false;
    }

    // Compute actual cost including TimeWarp penalty
    double costSuppU = params->timeCost.get(myBestSwapStar.bestPositionV->cour,
                                            myBestSwapStar.V->cour) -
                       params->timeCost.get(myBestSwapStar.U->prev->cour,
                                            myBestSwapStar.U->cour) -
                       params->timeCost.get(myBestSwapStar.U->cour,
                                            myBestSwapStar.U->next->cour);
    double costSuppV = params->timeCost.get(myBestSwapStar.bestPositionU->cour,
                                            myBestSwapStar.U->cour) -
                       params->timeCost.get(myBestSwapStar.V->prev->cour,
                                            myBestSwapStar.V->cour) -
                       params->timeCost.get(myBestSwapStar.V->cour,
                                            myBestSwapStar.V->next->cour);

    if (myBestSwapStar.bestPositionV == myBestSwapStar.U->prev) {
        // Insert in place of U
        costSuppU += params->timeCost.get(myBestSwapStar.V->cour,
                                          myBestSwapStar.U->next->cour);
    } else {
        costSuppU +=
            params->timeCost.get(myBestSwapStar.V->cour,
                                 myBestSwapStar.bestPositionV->next->cour) +
            params->timeCost.get(myBestSwapStar.U->prev->cour,
                                 myBestSwapStar.U->next->cour) -
            params->timeCost.get(myBestSwapStar.bestPositionV->cour,
                                 myBestSwapStar.bestPositionV->next->cour);
    }

    if (myBestSwapStar.bestPositionU == myBestSwapStar.V->prev) {
        // Insert in place of V
        costSuppV += params->timeCost.get(myBestSwapStar.U->cour,
                                          myBestSwapStar.V->next->cour);
    } else {
        costSuppV +=
            params->timeCost.get(myBestSwapStar.U->cour,
                                 myBestSwapStar.bestPositionU->next->cour) +
            params->timeCost.get(myBestSwapStar.V->prev->cour,
                                 myBestSwapStar.V->next->cour) -
            params->timeCost.get(myBestSwapStar.bestPositionU->cour,
                                 myBestSwapStar.bestPositionU->next->cour);
        ;
    }

    TimeWindowData routeUTwData, routeVTwData;
    // It is not possible to have bestPositionU == V or bestPositionV == U, so
    // the positions are always strictly different
    if (myBestSwapStar.bestPositionV->position ==
        myBestSwapStar.U->position - 1) {
        // Special case
        routeUTwData = MergeTWDataRecursive(
            myBestSwapStar.bestPositionV->prefixTwData,
            myBestSwapStar.V->twData, myBestSwapStar.U->next->postfixTwData);
    } else if (myBestSwapStar.bestPositionV->position <
               myBestSwapStar.U->position) {
        routeUTwData = MergeTWDataRecursive(
            myBestSwapStar.bestPositionV->prefixTwData,
            myBestSwapStar.V->twData,
            getRouteSegmentTwData(myBestSwapStar.bestPositionV->next,
                                  myBestSwapStar.U->prev),
            myBestSwapStar.U->next->postfixTwData);
    } else {
        routeUTwData = MergeTWDataRecursive(
            myBestSwapStar.U->prev->prefixTwData,
            getRouteSegmentTwData(myBestSwapStar.U->next,
                                  myBestSwapStar.bestPositionV),
            myBestSwapStar.V->twData,
            myBestSwapStar.bestPositionV->next->postfixTwData);
    }

    if (myBestSwapStar.bestPositionU->position ==
        myBestSwapStar.V->position - 1) {
        // Special case
        routeVTwData = MergeTWDataRecursive(
            myBestSwapStar.bestPositionU->prefixTwData,
            myBestSwapStar.U->twData, myBestSwapStar.V->next->postfixTwData);
    } else if (myBestSwapStar.bestPositionU->position <
               myBestSwapStar.V->position) {
        routeVTwData = MergeTWDataRecursive(
            myBestSwapStar.bestPositionU->prefixTwData,
            myBestSwapStar.U->twData,
            getRouteSegmentTwData(myBestSwapStar.bestPositionU->next,
                                  myBestSwapStar.V->prev),
            myBestSwapStar.V->next->postfixTwData);
    } else {
        routeVTwData = MergeTWDataRecursive(
            myBestSwapStar.V->prev->prefixTwData,
            getRouteSegmentTwData(myBestSwapStar.V->next,
                                  myBestSwapStar.bestPositionU),
            myBestSwapStar.U->twData,
            myBestSwapStar.bestPositionU->next->postfixTwData);
    }

    costSuppU += myBestSwapStar.loadPenU + penaltyTimeWindows(routeUTwData) -
                 routeU->penalty;

    costSuppV += myBestSwapStar.loadPenV + penaltyTimeWindows(routeVTwData) -
                 routeV->penalty;

    if (costSuppU + costSuppV > -MY_EPSILON) {
        return false;
    }

    // Applying the best move in case of improvement
    insertNode(myBestSwapStar.U, myBestSwapStar.bestPositionU);
    insertNode(myBestSwapStar.V, myBestSwapStar.bestPositionV);
    nbMoves++;  // Increment move counter before updating route data
    searchCompleted = false;
    updateRouteData(routeU);
    updateRouteData(routeV);

    return true;
}

bool HGSLocalSearch::RelocateStar() {
    double bestCost = 0;
    Node *insertionPoint = nullptr;
    Node *nodeToInsert = nullptr;
    for (nodeU = routeU->depot->next; !nodeU->isDepot; nodeU = nodeU->next) {
        setLocalbaldesVarsRouteU();

        // Precompute nodeU related values that don't change within inner loop
        const TimeWindowData routeUTwData = MergeTWDataRecursive(
            nodeU->prev->prefixTwData, nodeX->postfixTwData);
        const double penaltyU = penaltyExcessLoad(routeU->load - loadU) +
                                penaltyTimeWindows(routeUTwData);

        const double baseCostU =
            params->timeCost.get(nodeUPrevIndex, nodeXIndex) -
            params->timeCost.get(nodeUPrevIndex, nodeUIndex) -
            params->timeCost.get(nodeUIndex, nodeXIndex);
        const double costSuppU = baseCostU + penaltyU - routeU->penalty;

        for (Node *V = routeV->depot->next; !V->isDepot; V = V->next) {
            // Precompute time window and penalty only when necessary
            const TimeWindowData routeVTwData = MergeTWDataRecursive(
                V->prefixTwData, nodeU->twData, V->next->postfixTwData);
            const double penaltyV = penaltyExcessLoad(routeV->load + loadU) +
                                    penaltyTimeWindows(routeVTwData);

            const double costSuppV =
                params->timeCost.get(V->cour, nodeUIndex) +
                params->timeCost.get(nodeUIndex, V->next->cour) -
                params->timeCost.get(V->cour, V->next->cour) + penaltyV -
                routeV->penalty;

            const double totalCost = costSuppU + costSuppV;
            if (totalCost < bestCost - MY_EPSILON) {
                bestCost = totalCost;
                insertionPoint = V;
                nodeToInsert = nodeU;
            }
        }
    }

    if (!insertionPoint) {
        return false;
    }

    routeU = nodeToInsert->route;
    insertNode(nodeToInsert, insertionPoint);
    nbMoves++;  // Increment move counter before updating route data
    searchCompleted = false;
    updateRouteData(routeU);
    updateRouteData(insertionPoint->route);

    return true;
}

int HGSLocalSearch::getCheapestInsertSimultRemoval(Node *U, Node *V,
                                                   Node *&bestPosition) {
    ThreeBestInsert *myBestInsert = &bestInsertClient[V->route->cour][U->cour];
    bool found = false;

    // Find best insertion in the route such that V is not next or pred (can
    // only belong to the top three locations)
    bestPosition = myBestInsert->bestLocation[0];
    int bestCost = myBestInsert->bestCost[0];
    found = (bestPosition != V && bestPosition->next != V);
    if (!found && myBestInsert->bestLocation[1] != nullptr) {
        bestPosition = myBestInsert->bestLocation[1];
        bestCost = myBestInsert->bestCost[1];
        found = (bestPosition != V && bestPosition->next != V);
        if (!found && myBestInsert->bestLocation[2] != nullptr) {
            bestPosition = myBestInsert->bestLocation[2];
            bestCost = myBestInsert->bestCost[2];
            found = true;
        }
    }

    // Compute insertion in the place of V
    int deltaCost = params->timeCost.get(V->prev->cour, U->cour) +
                    params->timeCost.get(U->cour, V->next->cour) -
                    params->timeCost.get(V->prev->cour, V->next->cour);
    if (!found || deltaCost < bestCost) {
        bestPosition = V->prev;
        bestCost = deltaCost;
    }

    return bestCost;
}

// TODO make this double as cost??
int HGSLocalSearch::getCheapestInsertSimultRemovalWithTW(Node *U, Node *V,
                                                         Node *&bestPosition) {
    // TODO ThreeBestInsert must also use double as cost?
    ThreeBestInsert *myBestInsert =
        &bestInsertClientTW[V->route->cour][U->cour];
    bool found = false;

    // Find best insertion in the route such that V is not next or pred (can
    // only belong to the top three locations)
    bestPosition = myBestInsert->bestLocation[0];
    int bestCost = myBestInsert->bestCost[0];
    found = (bestPosition != V && bestPosition->next != V);
    if (!found && myBestInsert->bestLocation[1] != nullptr) {
        bestPosition = myBestInsert->bestLocation[1];
        bestCost = myBestInsert->bestCost[1];
        found = (bestPosition != V && bestPosition->next != V);
        if (!found && myBestInsert->bestLocation[2] != nullptr) {
            bestPosition = myBestInsert->bestLocation[2];
            bestCost = myBestInsert->bestCost[2];
            found = true;
        }
    }

    // Compute insertion in the place of V
    TimeWindowData twData = MergeTWDataRecursive(
        V->prev->prefixTwData, U->twData, V->next->postfixTwData);
    int deltaCost = params->timeCost.get(V->prev->cour, U->cour) +
                    params->timeCost.get(U->cour, V->next->cour) -
                    params->timeCost.get(V->prev->cour, V->next->cour) +
                    deltaPenaltyTimeWindows(twData, V->route->twData);
    if (!found || deltaCost < bestCost) {
        bestPosition = V->prev;
        bestCost = deltaCost;
    }

    return bestCost;
}

void HGSLocalSearch::preprocessInsertions(HGSRoute *R1, HGSRoute *R2) {
    for (Node *U = R1->depot->next; !U->isDepot; U = U->next) {
        // Performs the preprocessing
        U->deltaRemoval = params->timeCost.get(U->prev->cour, U->next->cour) -
                          params->timeCost.get(U->prev->cour, U->cour) -
                          params->timeCost.get(U->cour, U->next->cour);
        auto &currentOption = bestInsertClient[R2->cour][U->cour];
        if (R2->whenLastModified > currentOption.whenLastCalculated) {
            currentOption.reset();
            currentOption.whenLastCalculated = nbMoves;
            currentOption.bestCost[0] =
                params->timeCost.get(0, U->cour) +
                params->timeCost.get(U->cour, R2->depot->next->cour) -
                params->timeCost.get(0, R2->depot->next->cour);
            currentOption.bestLocation[0] = R2->depot;
            for (Node *V = R2->depot->next; !V->isDepot; V = V->next) {
                int deltaCost = params->timeCost.get(V->cour, U->cour) +
                                params->timeCost.get(U->cour, V->next->cour) -
                                params->timeCost.get(V->cour, V->next->cour);
                currentOption.compareAndAdd(deltaCost, V);
            }
        }
    }
}

void HGSLocalSearch::preprocessInsertionsWithTW(HGSRoute *R1, HGSRoute *R2) {
    TimeWindowData twData;
    for (Node *U = R1->depot->next; !U->isDepot; U = U->next) {
        // Performs the preprocessing
        // Note: when removing U and adding V to a route, the timewarp penalties
        // may interact, however in most cases it will hold that the reduced
        // timewarp from removing U + added timewarp from adding V will be
        // bigger than the actual delta timewarp such that assuming independence
        // gives a conservative estimate

        if (R1->isDeltaRemovalTWOutdated) {
            twData = MergeTWDataRecursive(U->prev->prefixTwData,
                                          U->next->postfixTwData);
            U->deltaRemovalTW =
                params->timeCost.get(U->prev->cour, U->next->cour) -
                params->timeCost.get(U->prev->cour, U->cour) -
                params->timeCost.get(U->cour, U->next->cour) +
                deltaPenaltyTimeWindows(twData, R1->twData);
        }
        auto &currentOption = bestInsertClientTW[R2->cour][U->cour];
        if (R2->whenLastModified > currentOption.whenLastCalculated) {
            currentOption.reset();
            currentOption.whenLastCalculated = nbMoves;

            // Compute additional timewarp we get when inserting U in R2, this
            // may be actually less if we remove U but we ignore this to have a
            // conservative estimate
            twData = MergeTWDataRecursive(R2->depot->prefixTwData, U->twData,
                                          R2->depot->next->postfixTwData);

            currentOption.bestCost[0] =
                params->timeCost.get(0, U->cour) +
                params->timeCost.get(U->cour, R2->depot->next->cour) -
                params->timeCost.get(0, R2->depot->next->cour) +
                deltaPenaltyTimeWindows(twData, R2->twData);

            currentOption.bestLocation[0] = R2->depot;
            for (Node *V = R2->depot->next; !V->isDepot; V = V->next) {
                twData = MergeTWDataRecursive(V->prefixTwData, U->twData,
                                              V->next->postfixTwData);
                int deltaCost = params->timeCost.get(V->cour, U->cour) +
                                params->timeCost.get(U->cour, V->next->cour) -
                                params->timeCost.get(V->cour, V->next->cour) +
                                deltaPenaltyTimeWindows(twData, R2->twData);
                currentOption.compareAndAdd(deltaCost, V);
            }
        }
    }
    R1->isDeltaRemovalTWOutdated = false;
}

TimeWindowData HGSLocalSearch::getEdgeTwData(Node *U, Node *V) {
    // TODO this could be cached?
    return MergeTWDataRecursive(U->twData, V->twData);
}

TimeWindowData HGSLocalSearch::getRouteSegmentTwData(Node *U, Node *V) {
    if (U->isDepot) return V->prefixTwData;
    if (V->isDepot) return U->postfixTwData;

    TimeWindowData twData = U->twData;  // Retain initial twData from U
    Node *mynode = U;
    const int targetPos = V->position;

    while (mynode != V) {
        if (mynode->isSeed && mynode->position + 4 <= targetPos) {
            twData = MergeTWDataRecursive(twData, mynode->toNextSeedTwD);
            mynode = mynode->nextSeed;
        } else {
            mynode = mynode->next;  // Move to the next node
            twData = MergeTWDataRecursive(twData, mynode->twData);
        }
    }

    return twData;  // Return the merged result
}

TimeWindowData HGSLocalSearch::MergeTWDataRecursive(
    const TimeWindowData &twData1, const TimeWindowData &twData2) {
    TimeWindowData mergedTwData;
    // Note, assume time equals cost
    int deltaCost =
        params->timeCost.get(twData1.lastNodeIndex, twData2.firstNodeIndex);
    int deltaDuration = deltaCost;
    int delta = twData1.duration - twData1.timeWarp + deltaDuration;
    int deltaWaitTime =
        std::max(twData2.earliestArrival - delta - twData1.latestArrival, 0);
    int deltaTimeWarp =
        std::max(twData1.earliestArrival + delta - twData2.latestArrival, 0);
    mergedTwData.firstNodeIndex = twData1.firstNodeIndex;
    mergedTwData.lastNodeIndex = twData2.lastNodeIndex;
    mergedTwData.duration =
        twData1.duration + twData2.duration + deltaDuration + deltaWaitTime;
    mergedTwData.timeWarp = twData1.timeWarp + twData2.timeWarp + deltaTimeWarp;
    mergedTwData.earliestArrival =
        std::max(twData2.earliestArrival - delta, twData1.earliestArrival) -
        deltaWaitTime;
    mergedTwData.latestArrival =
        std::min(twData2.latestArrival - delta, twData1.latestArrival) +
        deltaTimeWarp;
    mergedTwData.latestReleaseTime =
        std::max(twData1.latestReleaseTime, twData2.latestReleaseTime);

    return mergedTwData;
}

void HGSLocalSearch::insertNode(Node *toInsert, Node *insertionPoint) {
    toInsert->prev->next = toInsert->next;
    toInsert->next->prev = toInsert->prev;
    insertionPoint->next->prev = toInsert;
    toInsert->prev = insertionPoint;
    toInsert->next = insertionPoint->next;
    insertionPoint->next = toInsert;
    toInsert->route = insertionPoint->route;
}

void HGSLocalSearch::swapNode(Node *U, Node *V) {
    Node *myVPred = V->prev;
    Node *myVSuiv = V->next;
    Node *myUPred = U->prev;
    Node *myUSuiv = U->next;
    HGSRoute *myRouteU = U->route;
    HGSRoute *myRouteV = V->route;

    myUPred->next = V;
    myUSuiv->prev = V;
    myVPred->next = U;
    myVSuiv->prev = U;

    U->prev = myVPred;
    U->next = myVSuiv;
    V->prev = myUPred;
    V->next = myUSuiv;

    U->route = myRouteV;
    V->route = myRouteU;
}

void HGSLocalSearch::updateRouteData(HGSRoute *myRoute) {
    int myplace = 0;
    int myload = 0;
    int mytime = 0;
    int myReversalDistance = 0;
    int cumulatedX = 0;
    int cumulatedY = 0;

    Node *mynode = myRoute->depot;
    mynode->position = 0;
    mynode->cumulatedLoad = 0;
    mynode->cumulatedTime = 0;
    mynode->cumulatedReversalDistance = 0;

    bool firstIt = true;
    TimeWindowData seedTwD;
    Node *seedNode = nullptr;
    while (!mynode->isDepot || firstIt) {
        mynode = mynode->next;
        myplace++;
        mynode->position = myplace;
        myload += params->cli[mynode->cour].demand;
        mytime += params->timeCost.get(mynode->prev->cour, mynode->cour) +
                  params->cli[mynode->cour].serviceDuration;
        myReversalDistance +=
            params->timeCost.get(mynode->cour, mynode->prev->cour) -
            params->timeCost.get(mynode->prev->cour, mynode->cour);
        mynode->cumulatedLoad = myload;
        mynode->cumulatedTime = mytime;
        mynode->cumulatedReversalDistance = myReversalDistance;
        mynode->prefixTwData =
            MergeTWDataRecursive(mynode->prev->prefixTwData, mynode->twData);
        mynode->isSeed = false;
        mynode->nextSeed = nullptr;
        if (!mynode->isDepot) {
            cumulatedX += params->cli[mynode->cour].coordX;
            cumulatedY += params->cli[mynode->cour].coordY;
            if (firstIt)
                myRoute->sector.initialize(
                    params->cli[mynode->cour].polarAngle);
            else
                myRoute->sector.extend(params->cli[mynode->cour].polarAngle);
            if (myplace % 4 == 0) {
                if (seedNode != nullptr) {
                    seedNode->isSeed = true;
                    seedNode->toNextSeedTwD =
                        MergeTWDataRecursive(seedTwD, mynode->twData);
                    seedNode->nextSeed = mynode;
                }
                seedNode = mynode;
            } else if (myplace % 4 == 1) {
                seedTwD = mynode->twData;
            } else {
                seedTwD = MergeTWDataRecursive(seedTwD, mynode->twData);
            }
        }
        firstIt = false;
    }

    myRoute->duration = mytime;  // Driving duration + service duration,
                                 // excluding waiting time / time warp
    myRoute->load = myload;
    myRoute->twData = mynode->prefixTwData;
    if (params->problemType == ProblemType::vrptw ||
        params->problemType == ProblemType::evrp) {
        myRoute->penalty =
            penaltyExcessLoad(myload) + penaltyTimeWindows(myRoute->twData);
    } else if (params->problemType == ProblemType::cvrp) {
        myRoute->penalty = penaltyExcessLoad(myload);
    }
    myRoute->nbCustomers = myplace - 1;
    myRoute->reversalDistance = myReversalDistance;
    // Remember "when" this route has been last modified (will be used to filter
    // unnecessary move evaluations)
    myRoute->whenLastModified = nbMoves;
    myRoute->isDeltaRemovalTWOutdated = true;

    // Time window data in reverse direction, mynode should be end depot now
    firstIt = true;
    while (!mynode->isDepot || firstIt) {
        mynode = mynode->prev;
        mynode->postfixTwData =
            MergeTWDataRecursive(mynode->twData, mynode->next->postfixTwData);
        firstIt = false;
    }

    if (myRoute->nbCustomers == 0) {
        myRoute->polarAngleBarycenter = 1.e30;
        emptyRoutes.insert(myRoute->cour);
    } else {
        // Precompute nbCustomers inverse for reuse
        const double invNbCustomers =
            1.0 / static_cast<double>(myRoute->nbCustomers);

        // Calculate differences once
        const double diffX =
            (cumulatedX * invNbCustomers) - params->cli[0].coordX;
        const double diffY =
            (cumulatedY * invNbCustomers) - params->cli[0].coordY;

        myRoute->polarAngleBarycenter = atan2(diffY, diffX);

        // Enforce minimum size of circle sector
        if (params->minCircleSectorSize > 0) {
            int sectorSize = myRoute->sector.positive_mod(
                myRoute->sector.end - myRoute->sector.start);
            int growSectorBy =
                (params->minCircleSectorSize - sectorSize + 1) / 2;

            if (growSectorBy > 0) {
                myRoute->sector.extend(myRoute->sector.start - growSectorBy);
                myRoute->sector.extend(myRoute->sector.end + growSectorBy);
            }
        }
        emptyRoutes.erase(myRoute->cour);
    }
}

CostSol HGSLocalSearch::getCostSol(bool usePenaltiesLS) {
    CostSol myCostSol;

    myCostSol.nbRoutes = 0;  // TODO
    for (int r = 0; r < params->nbVehicles; r++) {
        HGSRoute *myRoute = &routes[r];
        myCostSol.distance += myRoute->duration;
        myCostSol.capacityExcess +=
            std::max(0, myRoute->load - params->vehicleCapacity);
        myCostSol.waitTime += 0;  // TODO
        if (params->problemType == ProblemType::vrptw ||
            params->problemType == ProblemType::evrp) {
            myCostSol.timeWarp += myRoute->twData.timeWarp;
        } else {
            myCostSol.timeWarp = 0;
        }
    }

    // Subtract service durations which are not part of distance
    for (int i = 1; i <= params->nbClients; i++) {
        myCostSol.distance -= params->cli[i].serviceDuration;
    }

    if (usePenaltiesLS) {
        myCostSol.penalizedCost =
            myCostSol.distance + myCostSol.capacityExcess * penaltyCapacityLS;
        if (params->problemType == ProblemType::vrptw ||
            params->problemType == ProblemType::evrp) {
            myCostSol.penalizedCost +=
                myCostSol.timeWarp * penaltyTimeWarpLS +
                myCostSol.waitTime * params->penaltyWaitTime;
        }
    } else {
        myCostSol.penalizedCost =
            myCostSol.distance +
            myCostSol.capacityExcess * params->penaltyCapacity;
        if (params->problemType == ProblemType::vrptw ||
            params->problemType == ProblemType::evrp) {
            myCostSol.penalizedCost +=
                myCostSol.timeWarp * params->penaltyTimeWarp +
                myCostSol.waitTime * params->penaltyWaitTime;
        }
    }
    return myCostSol;
}

void HGSLocalSearch::loadIndividual(Individual *indiv) {
    emptyRoutes.clear();
    nbMoves = 0;
    TimeWindowData depotTwData;
    depotTwData.firstNodeIndex = 0;
    depotTwData.lastNodeIndex = 0;
    depotTwData.duration = 0;
    depotTwData.timeWarp = 0;
    depotTwData.earliestArrival = params->cli[0].earliestArrival;
    depotTwData.latestArrival = params->cli[0].latestArrival;
    depotTwData.latestReleaseTime = params->cli[0].releaseTime;

    // Initializing time window data (before loop since it is needed in update
    // route)
    for (int i = 1; i <= params->nbClients; i++) {
        TimeWindowData *myTwData = &clients[i].twData;
        myTwData->firstNodeIndex = i;
        myTwData->lastNodeIndex = i;
        myTwData->duration = params->cli[i].serviceDuration;
        myTwData->earliestArrival = params->cli[i].earliestArrival;
        myTwData->latestArrival = params->cli[i].latestArrival;
        myTwData->latestReleaseTime = params->cli[i].releaseTime;
        // myTwData->load = params->cli[i].demand;
    }

    for (int r = 0; r < params->nbVehicles; r++) {
        Node *myDepot = &depots[r];
        Node *myDepotFin = &depotsEnd[r];
        HGSRoute *myRoute = &routes[r];
        myDepot->prev = myDepotFin;
        myDepotFin->next = myDepot;
        if (!indiv->chromR[r].empty()) {
            Node *myClient = &clients[indiv->chromR[r][0]];
            myClient->route = myRoute;
            myClient->prev = myDepot;
            myDepot->next = myClient;
            for (int i = 1; i < static_cast<int>(indiv->chromR[r].size());
                 i++) {
                Node *myClientPred = myClient;
                myClient = &clients[indiv->chromR[r][i]];
                myClient->prev = myClientPred;
                myClientPred->next = myClient;
                myClient->route = myRoute;
            }
            myClient->next = myDepotFin;
            myDepotFin->prev = myClient;
        } else {
            myDepot->next = myDepotFin;
            myDepotFin->prev = myDepot;
        }

        myDepot->twData = depotTwData;
        myDepot->prefixTwData = depotTwData;
        myDepot->postfixTwData = depotTwData;
        myDepot->isSeed = false;

        myDepotFin->twData = depotTwData;
        myDepotFin->prefixTwData = depotTwData;
        myDepotFin->postfixTwData = depotTwData;
        myDepotFin->isSeed = false;

        updateRouteData(&routes[r]);
        routes[r].whenLastTestedLargeNb = -1;
        bestInsertInitializedForRoute[r] = false;
    }

    for (int i = 1; i <= params->nbClients;
         i++)  // Initializing memory structures
        clients[i].whenLastTestedRI = -1;
}

void HGSLocalSearch::exportIndividual(Individual *indiv) {
    std::vector<std::pair<double, int>> routePolarAngles;
    for (int r = 0; r < params->nbVehicles; r++)
        routePolarAngles.push_back(
            std::pair<double, int>(routes[r].polarAngleBarycenter, r));
    pdqsort(
        routePolarAngles.begin(),
        routePolarAngles.end());  // empty routes have a polar angle of 1.e30,
                                  // and therefore will always appear at the end

    int pos = 0;
    for (int r = 0; r < params->nbVehicles; r++) {
        indiv->chromR[r].clear();
        Node *node = depots[routePolarAngles[r].second].next;
        while (!node->isDepot) {
            indiv->chromT[pos] = node->cour;
            indiv->chromR[r].push_back(node->cour);
            node = node->next;
            pos++;
        }
    }

    indiv->evaluateCompleteCost();
}

HGSLocalSearch::HGSLocalSearch(Params *params) : params(params) {
    clients = std::vector<Node>(params->nbClients + 1);
    routes = std::vector<HGSRoute>(params->nbVehicles);
    depots = std::vector<Node>(params->nbVehicles);
    depotsEnd = std::vector<Node>(params->nbVehicles);
    bestInsertInitializedForRoute =
        std::vector<bool>(params->nbVehicles, false);
    bestInsertClient = std::vector<std::vector<ThreeBestInsert>>(
        params->nbVehicles,
        std::vector<ThreeBestInsert>(params->nbClients + 1));
    bestInsertClientTW = std::vector<std::vector<ThreeBestInsert>>(
        params->nbVehicles,
        std::vector<ThreeBestInsert>(params->nbClients + 1));

    for (int i = 0; i <= params->nbClients; i++) {
        clients[i].cour = i;
        clients[i].isDepot = false;
    }
    for (int i = 0; i < params->nbVehicles; i++) {
        routes[i].cour = i;
        routes[i].depot = &depots[i];
        depots[i].cour = 0;
        depots[i].isDepot = true;
        depots[i].route = &routes[i];
        depotsEnd[i].cour = 0;
        depotsEnd[i].isDepot = true;
        depotsEnd[i].route = &routes[i];
    }
    for (int i = 1; i <= params->nbClients; i++) orderNodes.push_back(i);
    for (int r = 0; r < params->nbVehicles; r++) orderRoutes.push_back(r);
}
