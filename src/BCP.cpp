#include "bnb/BCP.h"
#include "bnb/Branching.h"

std::unique_ptr<VRProblem> VRProblem::clone() const {
    auto newProblem      = std::make_unique<VRProblem>();
    newProblem->instance = instance;
    newProblem->nodes    = nodes;
    return newProblem;
}

void VRProblem::branch(BNBNode *node) {
    fmt::print("\033[34m_STARTING BRANCH PROCEDURE \033[0m");
    fmt::print("\n");

    node->relaxNode();

    auto candidates       = Branching::VRPStandardBranching(node, &instance, this);
    auto candidateCounter = 0;

    for (auto candidate : candidates) {

        if (node->hasCandidate(candidate)) continue;
        if (node->hasRaisedChild(candidate)) continue;

        auto candidatosNode = node->getCandidatos();
        // print len candidatosNode
        auto childNode = node->newChild();
        childNode->addCandidate(candidate);
        node->addRaisedChildren(candidate);
        node->addChildren(childNode);

        candidateCounter++;
        if (candidateCounter >= NUMERO_CANDIDATOS) break;
    }

    fmt::print("\033[34m_FINISHED BRANCH PROCEDURE \033[0m");
    fmt::print("\n");
}

std::vector<BranchingQueueItem>
Branching::evaluateWithCG(BNBNode *node, const std::vector<BranchingQueueItem> &phase1Candidates, VRProblem *problem) {
    std::vector<BranchingQueueItem> results;
    std::mutex                      results_mutex; // Mutex to protect shared results

    const int                JOBS = std::thread::hardware_concurrency();
    exec::static_thread_pool pool(JOBS); // Pool with concurrent threads
    auto                     sched = pool.get_scheduler();

    // Define chunk size based on performance tuning (adjust as needed)
    const int chunk_size   = 10;
    auto      total_chunks = phase1Candidates.size() / JOBS;

    // Parallel bulk execution
    auto bulk_sender = stdexec::bulk(
        stdexec::just(), total_chunks,
        [&results, &results_mutex, node, &phase1Candidates, problem, chunk_size](std::size_t chunk_idx) {
            size_t start_idx = chunk_idx * chunk_size;
            size_t end_idx   = std::min(start_idx + chunk_size, phase1Candidates.size());

            // Clone the problem for this thread
            std::unique_ptr<VRProblem> problem_copy(problem->clone()); // Assuming clone method is available

            std::vector<BranchingQueueItem> local_results; // Local storage to avoid locking too often

            // Process the chunk of candidates
            for (size_t idx = start_idx; idx < end_idx; ++idx) {
                const auto &candidate = phase1Candidates[idx];

                // Add branching constraints and create two child nodes
                auto [childNode1, childNode2] = applyBranchingConstraints(node, candidate, candidate.fractionalValue);

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
    pdqsort(results.begin(), results.end(),
            [](const BranchingQueueItem &a, const BranchingQueueItem &b) { return a.productValue > b.productValue; });

    return results;
}