#pragma once

#include "Definitions.h"
#include "Node.h"
#include "Problem.h"

#include "../third_party/concurrentqueue.h"
#include <deque>
#include <queue>

enum class BNBNodeSelectionStrategy {
    DFS,      // Depth-First Search
    BFS,      // Breadth-First Search
    BestFirst // Best-First Search based on the boundValue
};

enum class BNBNodeHierarchy { Outer, Inner };

struct BNBNodeCompare {
    bool operator()(const BNBNode *lhs, const BNBNode *rhs) const {
        return lhs > rhs; // Min heap
    }
};

class BranchAndBound {
private:
    Problem                                                               *problem;
    std::priority_queue<BNBNode *, std::vector<BNBNode *>, BNBNodeCompare> bestFirstBNBNodes;
    // std::deque<BNBNode*> otherBNBNodes;
    moodycamel::ConcurrentQueue<BNBNode *> otherBNBNodes;

    BNBNodeSelectionStrategy strategy;
    mutable std::mutex       mutex; // Protects below shared states

    void addBNBNode(BNBNode *&node) {
        switch (strategy) {
        case BNBNodeSelectionStrategy::DFS: otherBNBNodes.enqueue(node); break;
        case BNBNodeSelectionStrategy::BFS: otherBNBNodes.enqueue(node); break;
        case BNBNodeSelectionStrategy::BestFirst: bestFirstBNBNodes.push(std::move(node)); break;
        }
    }

    BNBNode *getNextBNBNode() {
        BNBNode *node;
        if (strategy == BNBNodeSelectionStrategy::BestFirst && !bestFirstBNBNodes.empty()) {
            node = bestFirstBNBNodes.top();
            bestFirstBNBNodes.pop();
        } else if (otherBNBNodes.try_dequeue(node)) {
            // Successfully dequeued a node
        }
        return node;
    }

public:
    double     globalBestObjective = std::numeric_limits<double>::lowest();
    BNBNode   *globalBestBNBNode   = nullptr;
    bool       optimal             = false;
    BNBNode   *rootBNBNode         = nullptr;
    bool       solutionFound       = false;
    std::mutex solutionFoundMutex;

    explicit BranchAndBound(Problem* problem,
                            BNBNodeSelectionStrategy        strategy = BNBNodeSelectionStrategy::BestFirst)
        : problem(problem), strategy(strategy) {}

    explicit BranchAndBound(BNBNodeSelectionStrategy strategy = BNBNodeSelectionStrategy::BestFirst)
        : strategy(strategy) {}

    void setProblem(Problem *problem) { this->problem = problem; }

    void markSolutionFound() {
        std::lock_guard<std::mutex> lock(solutionFoundMutex);
        solutionFound = true;
    }

    void setRootNode(BNBNode *rootBNBNode) {
        this->rootBNBNode = rootBNBNode;
        std::lock_guard<std::mutex> lock(mutex);
        addBNBNode(rootBNBNode);
    }

    void updateGlobalBest(double objectiveValue, BNBNode *currentBNBNode, double boundValue) {
        std::lock_guard<std::mutex> lock(solutionFoundMutex); // Assuming same mutex can be used for simplicity
        if (objectiveValue > globalBestObjective) {
            globalBestObjective = objectiveValue;
            globalBestBNBNode   = currentBNBNode;
        }
        if ((std::floor(boundValue) == globalBestObjective) && boundValue > 0) { solutionFound = true; }
    }

    bool isSolutionFound() {
        std::lock_guard<std::mutex> lock(solutionFoundMutex);
        return solutionFound;
    }

    void processBNBNode(BNBNode *currentBNBNode) {
        if (isSolutionFound()) return; // Early exit if solution already found

        currentBNBNode->start();
        currentBNBNode->enforceBranching();
        double boundValue = problem->bound(currentBNBNode);

        if (isSolutionFound()) return; // Check again after a potentially long operation

        if (currentBNBNode->getPrune() || boundValue < globalBestObjective) {
            return; // BNBNode can be pruned
        }

        double objectiveValue = problem->objective(currentBNBNode);
        updateGlobalBest(objectiveValue, currentBNBNode, boundValue);

        if (isSolutionFound()) return; // Exit if the solution was found during update

        branch(currentBNBNode);
    }

    void solveParallel(size_t numThreads) {
        std::vector<std::thread> threads;
        for (size_t i = 0; i < numThreads; ++i) {
            threads.emplace_back([this]() {
                while (!isSolutionFound()) {
                    BNBNode *currentBNBNode;
                    if (!otherBNBNodes.try_dequeue(currentBNBNode))
                        break; // Exit if queue is empty or no more nodes to process

                    if (isSolutionFound()) break; // Check again before processing, for immediate exit if solution found

                    processBNBNode(currentBNBNode);
                }
            });
        }
        for (auto &thread : threads) {
            if (thread.joinable()) thread.join();
        }
    }

    void solve() {

        while (auto currentBNBNode = getNextBNBNode()) {

            currentBNBNode->start();
            currentBNBNode->enforceBranching();
            // currentBNBNode->enforceVRPBranching();
            print_info("Current node id: {}\n", currentBNBNode->getUUID());
            double boundValue = problem->bound(currentBNBNode);
            print_info("Bound value: {}\n", boundValue);

            if (currentBNBNode->getPrune()) {
                print_info("Pruned node: {}\n", currentBNBNode->getUUID());
                continue;
            }

            if (boundValue < globalBestObjective) continue;

            // auto objectiveValue = currentBNBNode->objectiveValue;
            auto objectiveValue = problem->objective(currentBNBNode);
            print_info("Objective value: {}\n", objectiveValue);

            if ((std::floor(boundValue) == objectiveValue)) {
                print_info("SOLUTION FOUND\n");
                print_info("Objective value: {}\n", objectiveValue);
                print_info("Solution is optimal: {}\n", objectiveValue);

                optimal             = true;
                globalBestBNBNode   = currentBNBNode;
                globalBestObjective = objectiveValue;
                break;
            }

            std::lock_guard<std::mutex> lock(mutex); // Protect shared state modifications
            if (objectiveValue > globalBestObjective) {
                globalBestObjective = objectiveValue;
                globalBestBNBNode   = currentBNBNode;
            }

            branch(currentBNBNode);
        }
    }

    void branch(BNBNode *&node) {
        problem->branch(node);
        auto children = node->getChildren();
        for (auto &child : children) { addBNBNode(child); }
    }

    [[nodiscard]] auto getBestBNBNode() const { return globalBestBNBNode; }

    [[nodiscard]] auto getBestObjective() const { return globalBestObjective; }

    [[nodiscard]] auto isOptimal() const { return optimal; }
};