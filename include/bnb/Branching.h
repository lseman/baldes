#pragma once

#include "../third_party/pdqsort.h"
#include "Definitions.h"
#include "Node.h"
#include "Reader.h"

#include "VRPCandidate.h"

constexpr double BANDX_TOL_ZERO       = 1E-6;
constexpr double BANDX_TOL_INTEGER    = 1E-2;
constexpr double AUX_DOUBLE_INCREMENT = 0.0001;

// Parameters for the branching evaluation
constexpr size_t ζ0 = 10; // Maximum number of candidates in Phase 0
constexpr size_t ζ1 = 5;  // Maximum number of candidates in Phase 1

enum class BranchingStrategy {
    MostFractional,  // Most fractional variable
    StrongBranching, // Strong branching
    semanRule,
    VRPTWStandard, // VRPTW Standard Branching
    VRPTWStrong    // VRPTW Strong Branching
};

class InstanceData;

struct BranchingQueueItem {
    int    routeIndex;
    int    sourceNode;
    double fractionalValue; // Original fractionality
    double score;           // Combined score (fractionality + window difference)
};

class Branching {

public:
    BranchingStrategy strategy = BranchingStrategy::MostFractional;

    explicit Branching(BranchingStrategy strategy) : strategy(strategy) {}
    Branching() = default;

    // Phase 0: Candidate Selection
    static std::vector<BranchingQueueItem>
    selectCandidatesPhase0(const std::vector<BranchingQueueItem> &allCandidates,
                           const std::vector<BranchingQueueItem> &historyCandidates) {

        std::vector<BranchingQueueItem> selectedCandidates;

        // Select half from history using pseudo-costs if history exists
        size_t historySize    = historyCandidates.size();
        size_t numFromHistory = std::min(ζ0 / 2, historySize);

        for (size_t i = 0; i < numFromHistory; ++i) { selectedCandidates.push_back(historyCandidates[i]); }

        // Select the remaining half from the rest using fractional proximity and depot distance
        size_t remaining = ζ0 - selectedCandidates.size();
        for (size_t i = 0; i < remaining; ++i) { selectedCandidates.push_back(allCandidates[i]); }

        return selectedCandidates;
    }

    // Phase 1: Evaluate Candidates by solving restricted master LP without generating columns
    static std::vector<BranchingQueueItem>
    evaluateCandidatesPhase1(BNBNode *node, const std::vector<BranchingQueueItem> &phase0Candidates) {

        std::vector<BranchingQueueItem> phase1Results;

        // Evaluate each candidate by solving restricted master LP
        for (const auto &candidate : phase0Candidates) {
            // Create child nodes and solve restricted master LP without column generation
            auto [childNode1, childNode2] = createChildNodes(node, candidate);
            double deltaLB1               = childNode1->solveRestrictedMasterLP();
            double deltaLB2               = childNode2->solveRestrictedMasterLP();

            // Product Rule
            double productValue = deltaLB1 * deltaLB2;

            // Store the result
            phase1Results.push_back(
                {candidate.routeIndex, candidate.sourceNode, candidate.fractionalValue, productValue});
        }

        // Sort results by product value (Product Rule)
        pdqsort(phase1Results.begin(), phase1Results.end(),
                [](const BranchingQueueItem &a, const BranchingQueueItem &b) {
                    return a.fractionalValue > b.fractionalValue;
                });

        return phase1Results;
    }

    // Phase 2: Evaluate candidates with relaxation and heuristic column generation
    static BranchingQueueItem evaluateCandidatesPhase2(BNBNode                               *node,
                                                       const std::vector<BranchingQueueItem> &phase1Candidates) {

        BranchingQueueItem bestCandidate;
        double             bestProductValue = -std::numeric_limits<double>::infinity();

        // Evaluate each candidate using relaxation and heuristic column generation
        for (const auto &candidate : phase1Candidates) {
            auto [childNode1, childNode2] = createChildNodes(node, candidate);
            double deltaLB1               = 0; // node->solveRelaxationWithColumnGeneration(childNode1);
            double deltaLB2               = 0; // node->solveRelaxationWithColumnGeneration(childNode2);

            // Apply product rule
            double productValue = deltaLB1 * deltaLB2;

            if (productValue > bestProductValue) {
                bestProductValue = productValue;
                bestCandidate    = candidate;
            }
        }

        return bestCandidate;
    }

    // Main function to handle the branching strategy for VRPTW
    static std::vector<VRPCandidate> VRPTWStandardBranching(BNBNode *node, InstanceData *instance) {
        std::vector<VRPCandidate> candidates;

        node->optimize();
        auto LPSolution = node->extractSolution();
        auto routes     = node->getPaths();

        int nN           = N_SIZE - 2;
        int solutionSize = LPSolution.size();

        std::vector<BranchingQueueItem> queue;
        std::vector<BranchingQueueItem> historyCandidates; // Assume this contains history from previous nodes

        // Generate candidate list based on fractional variables and edge proximity to depot
        // Generate candidate list based on fractional variables and edge proximity to depot
        for (int i = 0; i < solutionSize; i++) {
            if (LPSolution[i] == 0) continue;

            // Fractionality calculation (how far from being an integer)
            double fractionality = std::fabs(LPSolution[i] - std::round(LPSolution[i]));

            // Only consider fractional variables
            if (LPSolution[i] > BANDX_TOL_ZERO && fractionality > BANDX_TOL_INTEGER) {
                double totalWindowDifference = 0.0; // Used to calculate edge proximity

                for (size_t c = 0; c < routes[i].route.size() - 1; c++) {
                    auto source = routes[i].route[c];
                    auto target = routes[i].route[c + 1];

                    // Calculate window difference between source and target nodes
                    double windowDifference =
                        (std::fabs(instance->window_open[source] - instance->window_open[target]) +
                         std::fabs(instance->window_close[source] - instance->window_close[target]));

                    totalWindowDifference += windowDifference;
                }

                // Optionally, normalize the window difference
                double normalizedWindowDifference = totalWindowDifference / (routes[i].route.size() - 1);

                // Combine fractionality and proximity into a score (you can tweak this formula)
                double combinedScore = fractionality + AUX_DOUBLE_INCREMENT * normalizedWindowDifference;

                // Push the candidate with both fractionality and the combined score into the queue
                queue.push_back({i, routes[i].route[0], fractionality, combinedScore});

                // Optional: Debugging output
                // fmt::print("Route: {} Fractionality: {} Combined Score: {}\n", i, fractionality, combinedScore);
            }
        }

        // Phase 0: Select initial candidates
        auto phase0Candidates = selectCandidatesPhase0(queue, historyCandidates);

        // Phase 1: Evaluate candidates based on restricted master LP without column generation
        auto phase1Candidates = evaluateCandidatesPhase1(node, phase0Candidates);

        // candidates.push_back(generateVRPCandidates(node, phase1Candidates));
        // Append the result of generateVRPCandidates to the candidates vector
        std::vector<VRPCandidate> generated = generateVRPCandidates(node, phase1Candidates);
        candidates.insert(candidates.end(), generated.begin(), generated.end());

        // Phase 2: Evaluate candidates with full relaxation and heuristic column generation
        // auto bestCandidate = evaluateCandidatesPhase2(node, phase1Candidates);

        // Add logic to generate final candidates based on the best branching candidate
        // candidates.push_back(generateVRPCandidate(bestCandidate));

        return candidates;
    }

    /**
     * Create two child nodes based on the selected branching candidate
     * @param parentNode Parent node
     * @param candidate Branching candidate
     * @return Pair of child nodes
     */
    static std::pair<BNBNode *, BNBNode *> createChildNodes(BNBNode *parentNode, const BranchingQueueItem &candidate) {
        // Step 1: Clone the parent node to create two child nodes using the newChild() method
        BNBNode *childNode1 = parentNode->newChild(); // Create first child node
        BNBNode *childNode2 = parentNode->newChild(); // Create second child node

        // Step 2: Prepare coefficients for branching constraints
        std::vector<double> coeffs(parentNode->getNumVariables(), 0.0); // Assuming getNumVariables() exists in BNBNode

        // Assuming the candidate provides routeIndex and sourceNode
        coeffs[candidate.routeIndex] = 1.0; // Set the coefficient for the route index

        // Child Node 1: Add constraint g <= floor(fractional value)
        double lowerBoundConstraint = std::floor(candidate.fractionalValue);
        childNode1->addBranchingConstraint(coeffs, lowerBoundConstraint); // Constraint type: Upper bound

        // Child Node 2: Add constraint g >= ceil(fractional value)
        double upperBoundConstraint = std::ceil(candidate.fractionalValue);
        childNode2->addBranchingConstraint(coeffs, upperBoundConstraint); // Constraint type: Lower bound

        // Step 3: Return the two child nodes
        return {childNode1, childNode2};
    }

    /**
     * Generate VRPCandidates from the selected branching candidates
     * @param node Parent node
     * @param phase1Candidates Selected candidates from Phase 1
     * @return List of VRPCandidates
     */
    static std::vector<VRPCandidate> generateVRPCandidates(BNBNode                               *node,
                                                           const std::vector<BranchingQueueItem> &phase1Candidates) {
        std::vector<VRPCandidate> candidates;

        // Iterate over the selected candidates from Phase 1
        for (const auto &candidate : phase1Candidates) {
            fmt::print("Route: {} Source: {} Fractional Value: {}\n", candidate.routeIndex, candidate.sourceNode,
                       candidate.fractionalValue);
            // Create two child nodes based on the selected branching candidate
            auto [childNode1, childNode2] = createChildNodes(node, candidate);

            // Generate VRPCandidates from the two child nodes
            // For each child node, you can generate a corresponding VRPCandidate
            VRPCandidate vrpCandidate1(candidate.routeIndex, candidate.sourceNode, VRPCandidate::BoundType::Upper,
                                       std::floor(candidate.fractionalValue));
            VRPCandidate vrpCandidate2(candidate.routeIndex, candidate.sourceNode, VRPCandidate::BoundType::Lower,
                                       std::ceil(candidate.fractionalValue));

            // Add both candidates to the list
            candidates.push_back(vrpCandidate1);
            candidates.push_back(vrpCandidate2);
        }

        return candidates;
    }
};
