/*
 * @file Branching.h
 * @brief Branching strategy for the BNB algorithm.
 *
 * This file contains the definition of the Branching class, which implements the branching strategy
 * for the Branch-and-Bound (BNB) algorithm. The class provides methods to calculate aggregated variables
 * for branching constraints, generate branching candidates, and apply branching constraints to the BNB nodes.
 *
 */
#pragma once

#include "../third_party/pdqsort.h"

#include "Definitions.h"

#include "Node.h"
#include "Reader.h"
#include "VRPCandidate.h"

#include "ankerl/unordered_dense.h"

#include <optional>
#include <variant>

#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

constexpr double TOLERANCE_ZERO    = 1E-6;
constexpr double TOLERANCE_INTEGER = 1E-2;
constexpr double SCORE_INCREMENT   = 0.0001;

// Parameters for branching
constexpr size_t maxCandidatesPhase0 = 10; // Max candidates in Phase 0
constexpr size_t maxCandidatesPhase1 = 5;  // Max candidates in Phase 1

constexpr double DEPOT_DISTANCE_THRESHOLD = 100.0; // Threshold for depot distance

enum class BranchingStrategy {
    Hierarchical, // Hierarchical branching
};

class Branching {
public:
    BranchingStrategy strategy = BranchingStrategy::Hierarchical;

    explicit Branching(BranchingStrategy strat) : strategy(strat) {}
    Branching() = default;

    // Helper function to check if a value is fractional
    inline static bool isFractional(double value) { return std::fabs(value - std::round(value)) > TOLERANCE_INTEGER; }

    /**
     * Calculate aggregated variables for branching constraints
     * @param node: Current BNB node
     * @param instance: Instance data
     * @return List of branching candidates
     */
    static std::vector<BranchingQueueItem> calculateAggregatedVariables(BNBNode *node, const InstanceData *instance) {
        std::vector<BranchingQueueItem> queueItems;
        auto                            LPSolution = node->extractSolution();
        auto                            routes     = node->getPaths();

        // Aggregated variables for all vehicle types and routes
        ankerl::unordered_dense::map<int, double> g_m; // Aggregated g_m (number of vehicles of type m used)
        ankerl::unordered_dense::map<int, double>
            g_m_v; // Aggregated g_m_v (whether customer v is served by a vehicle of type m)
        ankerl::unordered_dense::map<std::pair<int, int>, double>
            g_v_vp; // Aggregated g_{v,v'} (whether edge (v, v') is used)

        // Iterate over all routes in the LPSolution
        for (int i = 0; i < LPSolution.size(); ++i) {
            if (LPSolution[i] == 0) continue; // Skip if no solution for this route

            if (LPSolution[i] > TOLERANCE_ZERO && isFractional(LPSolution[i])) {

                double totalWindowDifference = 0.0;

                // Assuming vehicleType is stored in the instance for each route
                int vehicleType = 0;

                // Calculate g_m: Sum the solution value for each vehicle type m
                g_m[vehicleType] += LPSolution[i]; // Accumulate the usage of vehicle type m

                // Iterate over the route to calculate g_m_v and g_{v,v'}
                for (size_t c = 0; c < routes[i].route.size() - 1; ++c) {
                    auto source = routes[i].route[c];
                    auto target = routes[i].route[c + 1];

                    // Calculate window differences (for scoring, not directly affecting g_m_v or g_{v,v'})
                    double windowDifference =
                        std::fabs(instance->window_open[source] - instance->window_open[target]) +
                        std::fabs(instance->window_close[source] - instance->window_close[target]);
                    totalWindowDifference += windowDifference;

                    // Accumulate g_{v,v'}: Edge usage for both directions (v, v') and (v', v)
                    g_v_vp[{source, target}] += LPSolution[i]; // Accumulate for (v, v')
                }

                // Calculate g_m_v: Accumulate if customer v is served by a vehicle of type m
                for (size_t c = 1; c < routes[i].route.size() - 1; ++c) { // Skip depot (assumed to be first and last)
                    auto customer = routes[i].route[c];
                    g_m_v[customer] +=
                        LPSolution[i]; // Accumulate the value for this customer being served by vehicle type m
                }
            }
        }

        // Create separate queue items for g_m, g_m_v, and g_{v,v'} variables

        // 1. Queue items for g_m (vehicle types)
        for (auto &[vehicleType, value] : g_m) {
            queueItems.push_back(
                BranchingQueueItem{0, 0, value, 0.0, {{vehicleType, value}}, {}, {}, CandidateType::Vehicle});
        }

        // 2. Queue items for g_m_v (customers served by vehicles)
        for (auto &[customer, value] : g_m_v) {
            queueItems.push_back(
                BranchingQueueItem{0, customer, value, 0.0, {}, {{customer, value}}, {}, CandidateType::Node});
        }

        // 3. Queue items for g_{v,v'} (edge usage)
        for (auto &[edge, value] : g_v_vp) {
            queueItems.push_back(
                BranchingQueueItem{edge.first, edge.second, value, 0.0, {}, {}, {{edge, value}}, CandidateType::Edge});
        }

        return queueItems;
    }
    static void addConstraintForCandidate(BNBNode *childNode, const BranchingQueueItem &item, double bound,
                                          BranchingDirection direction) {
        if (item.candidateType == CandidateType::Vehicle) {
            childNode->addBranchingConstraint(bound, direction, CandidateType::Vehicle);
        } else if (item.candidateType == CandidateType::Node) {
            childNode->addBranchingConstraint(bound, direction, CandidateType::Node, item.targetNode);
        } else if (item.candidateType == CandidateType::Edge) {
            std::pair<int, int> edge = {item.sourceNode, item.targetNode};
            childNode->addBranchingConstraint(bound, direction, CandidateType::Edge, edge);
        }
    }

    // Apply branching constraints based on the fractional value of the candidate
    static std::pair<BNBNode *, BNBNode *>
    applyBranchingConstraints(BNBNode *parentNode, const BranchingQueueItem &item, double fractionalValue) {
        BNBNode *childNode1 = parentNode->newChild();
        BNBNode *childNode2 = parentNode->newChild();

        double lowerBound = std::floor(fractionalValue);
        double upperBound = std::ceil(fractionalValue);

        addConstraintForCandidate(childNode1, item, upperBound, BranchingDirection::Greater);
        addConstraintForCandidate(childNode2, item, lowerBound, BranchingDirection::Less);

        return {childNode1, childNode2};
    }

    /**
     * Evaluate candidates with branching and return the results
     * @brief This function evaluates the candidates with branching and returns the results
     */
    static std::vector<BranchingQueueItem>
    evaluateWithBranching(BNBNode *node, const std::vector<BranchingQueueItem> &phase0Candidates) {
        std::vector<BranchingQueueItem> results;
        std::mutex                      results_mutex; // Mutex to protect shared results

        const int                JOBS = std::thread::hardware_concurrency();
        exec::static_thread_pool pool(JOBS); // Pool with concurrent threads
        auto                     sched = pool.get_scheduler();

        // Define chunk size based on performance tuning (adjust as needed)
        const int chunk_size   = 10;
        auto      total_chunks = (phase0Candidates.size() + chunk_size - 1) / chunk_size;

        // Parallel bulk execution
        auto bulk_sender = stdexec::bulk(
            stdexec::just(), total_chunks,
            [&results, &results_mutex, node, &phase0Candidates, chunk_size](std::size_t chunk_idx) {
                size_t start_idx = chunk_idx * chunk_size;
                size_t end_idx   = std::min(start_idx + chunk_size, phase0Candidates.size());

                std::vector<BranchingQueueItem> local_results; // Local storage to avoid locking too often

                // Process the chunk of candidates
                for (size_t idx = start_idx; idx < end_idx; ++idx) {
                    const auto &candidate = phase0Candidates[idx];

                    // Add branching constraints and create two child nodes
                    auto [childNode1, childNode2] =
                        applyBranchingConstraints(node, candidate, candidate.fractionalValue);

                    // Solve the Restricted Master LP for each child node
                    double deltaLB1, deltaLB2;
                    double feasLB1, feasLB2;
                    std::tie(feasLB1, deltaLB1) = childNode1->solveRestrictedMasterLP();
                    std::tie(feasLB2, deltaLB2) = childNode2->solveRestrictedMasterLP();

                    // Calculate product value for branching
                    double productValue = deltaLB1 * deltaLB2;

                    // Store the result locally
                    local_results.push_back({candidate.sourceNode, candidate.targetNode, candidate.fractionalValue,
                                             productValue, candidate.g_m, candidate.g_m_v, candidate.g_v_vp,
                                             candidate.candidateType, std::make_pair(feasLB1, feasLB2)});
                }

                // Add local results to shared results with mutex protection
                {
                    std::lock_guard<std::mutex> lock(results_mutex);
                    results.insert(results.end(), local_results.begin(), local_results.end());
                }
            });

        // Execute bulk task using stdexec
        stdexec::sync_wait(stdexec::when_all(bulk_sender));

        // Sort results by product value in descending order (highest product value first)
        pdqsort(results.begin(), results.end(), [](const BranchingQueueItem &a, const BranchingQueueItem &b) {
            return a.productValue > b.productValue;
        });

        return results;
    }

    /**
     * Generate VRP candidates based on the selected branching candidates
     * @brief This function generates VRP candidates based on the selected branching candidates
     */
    static std::vector<VRPCandidate *> generateVRPCandidates(BNBNode                               *node,
                                                             const std::vector<BranchingQueueItem> &phase1Candidates) {
        std::vector<VRPCandidate *> candidates;

        // Lambda to generate candidates
        auto generateCandidates = [&](CandidateType type, int sourceNode, int targetNode, double fractionalValue,
                                      std::pair<bool, bool>                                 flags,
                                      std::optional<std::variant<int, std::pair<int, int>>> payload = std::nullopt) {
            // check if flag first is true
            if (flags.first) {
                candidates.emplace_back(new VRPCandidate(sourceNode, targetNode, BranchingDirection::Greater,
                                                         std::floor(fractionalValue), type, payload));
            }
            if (flags.second) {
                candidates.emplace_back(new VRPCandidate(sourceNode, targetNode, BranchingDirection::Less,
                                                         std::ceil(fractionalValue), type, payload));
            }
        };

        for (const auto &item : phase1Candidates) {
            if (!item.g_m.empty()) {
                generateCandidates(CandidateType::Vehicle, item.sourceNode, item.targetNode, item.fractionalValue,
                                   item.flags,
                                   std::nullopt); // No payload for vehicles
            }
            if (!item.g_m_v.empty()) {
                generateCandidates(CandidateType::Node, item.sourceNode, item.targetNode, item.fractionalValue,
                                   item.flags,
                                   item.targetNode); // Node as payload
            }
            if (!item.g_v_vp.empty()) {
                std::pair<int, int> edge = {item.sourceNode, item.targetNode};
                generateCandidates(CandidateType::Edge, item.sourceNode, item.targetNode, item.fractionalValue,
                                   item.flags,
                                   edge); // Edge as payload
            }
        }

        return candidates;
    }

    /**
     * Create two child nodes based on the selected branching candidate
     * @brief This function creates two child nodes based on the selected branching candidate
     */
    static std::pair<BNBNode *, BNBNode *> createChildNodes(BNBNode *parentNode, const BranchingQueueItem &candidate) {
        BNBNode *childNode1 = parentNode->newChild();
        BNBNode *childNode2 = parentNode->newChild();

        // Helper to add branching constraint
        auto addConstraints = [&](double bound, BranchingDirection direction, BNBNode *child) {
            if (candidate.candidateType == CandidateType::Vehicle) {
                child->addBranchingConstraint(bound, direction, candidate.candidateType);
            } else if (candidate.candidateType == CandidateType::Node) {
                child->addBranchingConstraint(bound, direction, candidate.candidateType, candidate.targetNode);
            } else { // CandidateType::Edge
                std::pair<int, int> edge = {candidate.sourceNode, candidate.targetNode};
                child->addBranchingConstraint(bound, direction, candidate.candidateType, edge);
            }
        };

        double lowerBound = std::floor(candidate.fractionalValue);
        double upperBound = std::ceil(candidate.fractionalValue);

        // Add constraints to both child nodes
        addConstraints(lowerBound, BranchingDirection::Less, childNode1);
        addConstraints(upperBound, BranchingDirection::Greater, childNode2);

        return {childNode1, childNode2};
    }

    /**
     * Select candidates for Phase 0 using the new branching logic
     * @brief This function implements the new branching logic for selecting candidates in Phase 0
     */
    static std::vector<BranchingQueueItem>
    selectCandidatesPhase0(const std::vector<BranchingQueueItem> &allCandidates,
                           const std::vector<BranchingQueueItem> &historyCandidates, size_t maxCandidatesPhase0,
                           InstanceData *instance) {

        std::vector<BranchingQueueItem> selectedCandidates;

        // Step 1: Select half of the candidates from history using pseudo-costs if available
        size_t historySize    = historyCandidates.size();
        size_t numFromHistory = std::min(maxCandidatesPhase0 / 2, historySize);

        // Sort history candidates by pseudo-costs (assumed to be stored in score field)
        std::vector<BranchingQueueItem> sortedHistoryCandidates = historyCandidates;
        pdqsort(sortedHistoryCandidates.begin(), sortedHistoryCandidates.end(),
                [](const BranchingQueueItem &a, const BranchingQueueItem &b) {
                    return a.productValue > b.productValue; // Higher pseudo-cost first
                });

        // Select top candidates from history
        for (size_t i = 0; i < numFromHistory; ++i) { selectedCandidates.push_back(sortedHistoryCandidates[i]); }

        // Step 2: Select remaining candidates using the three branching strategies
        size_t                          remainingCandidates = maxCandidatesPhase0 - selectedCandidates.size();
        std::vector<BranchingQueueItem> strategyCandidates;

        // Strategy 1: Fractional value distance to the closest integer (for branching on variables)
        for (const auto &candidate : allCandidates) {
            if (remainingCandidates == 0) break;

            // Check if candidate is suitable based on its fractional value
            if (candidate.fractionalValue > TOLERANCE_INTEGER) {
                strategyCandidates.push_back(candidate);
                remainingCandidates--;
            }
        }

        // Strategy 2: Branching on edges with proximity to the depot
        for (const auto &candidate : allCandidates) {
            if (remainingCandidates == 0) break;

            if (candidate.candidateType == CandidateType::Vehicle) continue; // Skip vehicle candidates

            // Assuming `getDepotDistance()` computes distance of the edge to the closest depot
            double depotDistance = instance->getcij(0, candidate.targetNode);
            if (depotDistance < DEPOT_DISTANCE_THRESHOLD) { // Threshold for proximity to depot
                strategyCandidates.push_back(candidate);
                remainingCandidates--;
            }
        }

        // Step 3: Sort remaining candidates based on distance to the closest integer or depot distance
        pdqsort(strategyCandidates.begin(), strategyCandidates.end(),
                [](const BranchingQueueItem &a, const BranchingQueueItem &b) {
                    return a.fractionalValue < b.fractionalValue; // Closest to integer first for fractional strategy
                });

        // Add remaining candidates to the selected candidates list
        for (size_t i = 0; i < strategyCandidates.size() && selectedCandidates.size() < maxCandidatesPhase0; ++i) {
            selectedCandidates.push_back(strategyCandidates[i]);
        }

        return selectedCandidates;
    }

    static std::vector<BranchingQueueItem>
    evaluateWithCG(BNBNode *node, const std::vector<BranchingQueueItem> &phase1Candidates, Problem *problem) {
        std::vector<BranchingQueueItem> results;
        std::mutex                      results_mutex; // Mutex to protect shared results

        const int                JOBS = std::thread::hardware_concurrency();
        exec::static_thread_pool pool(JOBS); // Pool with concurrent threads
        auto                     sched = pool.get_scheduler();

        // Define chunk size based on performance tuning (adjust as needed)
        const int chunk_size   = 1;
        auto      total_chunks = (phase1Candidates.size() + chunk_size - 1) / chunk_size;

        // Parallel bulk execution
        auto bulk_sender = stdexec::bulk(
            stdexec::just(), total_chunks,
            [&results, &results_mutex, node, &phase1Candidates, problem, chunk_size](std::size_t chunk_idx) {
                size_t start_idx = chunk_idx * chunk_size;
                size_t end_idx   = std::min(start_idx + chunk_size, phase1Candidates.size());

                // Clone the problem for this thread
                std::unique_ptr<Problem> problem_copy(problem->clone()); // Assuming clone method is available

                std::vector<BranchingQueueItem> local_results; // Local storage to avoid locking too often

                // Process the chunk of candidates
                for (size_t idx = start_idx; idx < end_idx; ++idx) {
                    const auto &candidate = phase1Candidates[idx];

                    // Add branching constraints and create two child nodes
                    auto [childNode1, childNode2] =
                        applyBranchingConstraints(node, candidate, candidate.fractionalValue);

                    // Solve CG and bound for each child node using the cloned problem
                    double deltaLB1, deltaLB2;
                    double feasLB1, feasLB2;
                    feasLB1  = problem_copy->heuristicCG(childNode1, 50); // Use the copy
                    deltaLB1 = problem_copy->bound(childNode1);           // Use the copy
                    feasLB2  = problem_copy->heuristicCG(childNode2, 50); // Use the copy
                    deltaLB2 = problem_copy->bound(childNode2);           // Use the copy

                    // Calculate product value for branching
                    double productValue = deltaLB1 * deltaLB2;

                    // Store the result locally
                    local_results.push_back({candidate.sourceNode, candidate.targetNode, candidate.fractionalValue,
                                             productValue, candidate.g_m, candidate.g_m_v, candidate.g_v_vp,
                                             candidate.candidateType, std::make_pair(feasLB1, feasLB2)});
                }

                // Add local results to shared results with mutex protection
                {
                    std::lock_guard<std::mutex> lock(results_mutex);
                    results.insert(results.end(), local_results.begin(), local_results.end());
                }
            });

        // Execute bulk task using stdexec
        stdexec::sync_wait(stdexec::when_all(bulk_sender));

        // Sort results by product value in descending order (highest product value first)
        pdqsort(results.begin(), results.end(), [](const BranchingQueueItem &a, const BranchingQueueItem &b) {
            return a.productValue > b.productValue;
        });

        return results;
    }

    /**
     * VRPTW Standard Branching Strategy
     * @brief This function implements the standard branching strategy for the VRPTW problem
     *
     */
    static std::vector<VRPCandidate *> VRPTWStandardBranching(BNBNode *node, InstanceData *instance, Problem *problem) {
        std::vector<VRPCandidate *> candidates;

        node->optimize();

        auto aggregatedCandidates = calculateAggregatedVariables(node, instance);

        // Select candidates based on the new Phase 0 logic
        print_info("Phase 0: selecting candidates for branching...\n");
        auto phase0Candidates =
            selectCandidatesPhase0(aggregatedCandidates, node->historyCandidates, maxCandidatesPhase0, instance);

        // Store selected candidates in the node for future iterations
        node->historyCandidates = phase0Candidates;

        // Evaluate candidates and apply branching constraints
        print_info("Phase 1: evaluating candidates for branching...\n");
        auto phase1Candidates = evaluateWithBranching(node, phase0Candidates);

        // TODO: add logic to select the best candidates from phase1Candidates
        print_info("Phase 2: selecting best candidates from Phase 1...\n");
        auto phase2Candidates = evaluateWithCG(node, phase1Candidates, problem);

        print_info("Phase 3: generating VRP candidates...\n");
        auto generatedCandidates = generateVRPCandidates(node, phase1Candidates);
        candidates.insert(candidates.end(), generatedCandidates.begin(), generatedCandidates.end());

        return candidates;
    }
};
