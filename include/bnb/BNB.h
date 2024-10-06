#pragma once

#include "Definitions.h"
#include "Node.h"
#include "Problem.h"

#include "../third_party/concurrentqueue.h"
#include <cmath>
#include <deque>
#include <limits>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <thread>

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
    moodycamel::ConcurrentQueue<BNBNode *>                                 otherBNBNodes;

    BNBNodeSelectionStrategy  strategy;
    mutable std::shared_mutex mutex; // Protects below shared states

    void addBNBNode(BNBNode *&node) {
        if (strategy == BNBNodeSelectionStrategy::BestFirst) {
            bestFirstBNBNodes.push(std::move(node));
        } else {
            otherBNBNodes.enqueue(node);
        }
    }

    BNBNode *getNextBNBNode() {
        BNBNode *node = nullptr;
        if (strategy == BNBNodeSelectionStrategy::BestFirst && !bestFirstBNBNodes.empty()) {
            node = bestFirstBNBNodes.top();
            bestFirstBNBNodes.pop();
            return node;
        } else if (otherBNBNodes.try_dequeue(node)) {
            return node;
        }
        return nullptr;
    }

public:
    double            globalBestObjective = std::numeric_limits<double>::lowest();
    BNBNode          *globalBestBNBNode   = nullptr;
    bool              optimal             = false;
    BNBNode          *rootBNBNode         = nullptr;
    bool              solutionFound       = false;
    std::shared_mutex solutionFoundMutex;

    explicit BranchAndBound(Problem *problem, BNBNodeSelectionStrategy strategy = BNBNodeSelectionStrategy::BestFirst)
        : problem(problem), strategy(strategy) {}

    explicit BranchAndBound(BNBNodeSelectionStrategy strategy = BNBNodeSelectionStrategy::BestFirst)
        : strategy(strategy) {}

    void setProblem(Problem *problem) { this->problem = problem; }

    void markSolutionFound() {
        std::lock_guard lock(solutionFoundMutex);
        solutionFound = true;
    }

    void setRootNode(BNBNode *rootBNBNode) {
        this->rootBNBNode = rootBNBNode;
        std::lock_guard lock(mutex);
        addBNBNode(rootBNBNode);
    }

    void updateGlobalBest(double objectiveValue, BNBNode *currentBNBNode, double boundValue) {
        std::scoped_lock lock(mutex, solutionFoundMutex);
        if (objectiveValue > globalBestObjective) {
            globalBestObjective = objectiveValue;
            globalBestBNBNode   = currentBNBNode;
        }
        if ((std::floor(boundValue) == globalBestObjective) && boundValue > 0) { solutionFound = true; }
    }

    bool isSolutionFound() {
        std::shared_lock lock(solutionFoundMutex);
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
        std::vector<std::jthread> threads;
        for (size_t i = 0; i < numThreads; ++i) {
            threads.emplace_back([this]() {
                while (!isSolutionFound()) {
                    BNBNode *currentBNBNode = getNextBNBNode();
                    if (currentBNBNode) {
                        processBNBNode(currentBNBNode);
                    } else {
                        break; // Exit if no more nodes to process
                    }
                }
            });
        }
    }

    void solve() {
        while (BNBNode *currentBNBNode = getNextBNBNode()) {
            currentBNBNode->start();
            currentBNBNode->enforceBranching();
            print_info("Current node id: {}\n", currentBNBNode->getUUID());
            double boundValue = problem->bound(currentBNBNode);
            print_info("Bound value: {}\n", boundValue);

            if (currentBNBNode->getPrune()) {
                print_info("Pruned node: {}\n", currentBNBNode->getUUID());
                continue;
            }

            if (boundValue < globalBestObjective) continue;

            auto objectiveValue = problem->objective(currentBNBNode);
            print_info("Objective value: {}\n", objectiveValue);

            if (std::abs(boundValue - objectiveValue) < 1e-2) {
                print_info("SOLUTION FOUND\n");
                print_info("Objective value: {}\n", objectiveValue);
                print_info("Solution is optimal: {}\n", objectiveValue);

                optimal             = true;
                globalBestBNBNode   = currentBNBNode;
                globalBestObjective = objectiveValue;
                break;
            }

            std::lock_guard lock(mutex); // Protect shared state modifications
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
