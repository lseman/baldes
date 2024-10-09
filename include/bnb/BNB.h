/*
 * @file BNB.h
 * @brief Branch-and-Bound algorithm implementation.
 *
 * This file contains the implementation of the Branch-and-Bound (BNB) algorithm for solving optimization problems.
 * The algorithm uses a tree search strategy to explore the solution space and find the optimal solution.
 *
 */
#pragma once

#include "Definitions.h"

#include "bnb/Node.h"
#include "bnb/Problem.h"

#include "../third_party/concurrentqueue.h"

#include <atomic>
#include <cmath>
#include <deque>
#include <limits>
#include <queue>
#include <thread>
#include <vector>

enum class BNBNodeSelectionStrategy {
    DFS,      // Depth-First Search
    BFS,      // Breadth-First Search
    BestFirst // Best-First Search based on the boundValue
};

struct BNBNodeCompare {
    bool operator()(const BNBNode *lhs, const BNBNode *rhs) const {
        return lhs > rhs; // Min heap for Best-First Search
    }
};

class BranchAndBound {
private:
    Problem                                                               *problem;
    std::priority_queue<BNBNode *, std::vector<BNBNode *>, BNBNodeCompare> bestFirstBNBNodes;
    moodycamel::ConcurrentQueue<BNBNode *>                                 otherBNBNodes;
    BNBNode                                                               *rootBNBNode;

    BNBNodeSelectionStrategy strategy;
    std::mutex               bestFirstMutex; // Mutex for bestFirstBNBNodes

    std::atomic<double> globalBestObjective{std::numeric_limits<double>::lowest()};
    std::atomic<bool>   solutionFound{false};

    void addBNBNode(BNBNode *&node) {
        if (strategy == BNBNodeSelectionStrategy::BestFirst) {
            std::lock_guard<std::mutex> lock(bestFirstMutex); // Protect bestFirst queue
            bestFirstBNBNodes.push(node);
        } else {
            otherBNBNodes.enqueue(node);
        }
    }

    BNBNode *getNextBNBNode() {
        BNBNode *node = nullptr;

        if (strategy == BNBNodeSelectionStrategy::BestFirst) {
            std::lock_guard<std::mutex> lock(bestFirstMutex); // Protect bestFirst queue
            if (!bestFirstBNBNodes.empty()) {
                node = bestFirstBNBNodes.top();
                bestFirstBNBNodes.pop();
                return node;
            }
        }

        if (otherBNBNodes.try_dequeue(node)) { return node; }

        return nullptr; // No node available
    }

public:
    explicit BranchAndBound(Problem *problem, BNBNodeSelectionStrategy strategy = BNBNodeSelectionStrategy::BestFirst)
        : problem(problem), strategy(strategy) {}

    explicit BranchAndBound(BNBNodeSelectionStrategy strategy = BNBNodeSelectionStrategy::BestFirst)
        : strategy(strategy) {}

    void setProblem(Problem *problem) { this->problem = problem; }

    void markSolutionFound() {
        solutionFound.store(true, std::memory_order_release); // Use atomic store
    }

    bool isSolutionFound() const {
        return solutionFound.load(std::memory_order_acquire); // Use atomic load
    }

    void setRootNode(BNBNode *rootBNBNode) {
        this->rootBNBNode = rootBNBNode;
        addBNBNode(rootBNBNode);
    }

    void updateGlobalBest(double objectiveValue, BNBNode *currentBNBNode, double boundValue) {
        // Use atomic compare and swap for updating global best objective
        double prevGlobalBest = globalBestObjective.load(std::memory_order_acquire);
        while (objectiveValue > prevGlobalBest &&
               !globalBestObjective.compare_exchange_weak(prevGlobalBest, objectiveValue, std::memory_order_release)) {
            // Repeat until update is successful
        }

        if ((std::floor(boundValue) == globalBestObjective.load(std::memory_order_acquire)) && boundValue > 0) {
            markSolutionFound(); // If bound equals the best objective, mark solution found
        }
    }

    void processBNBNode(BNBNode *currentBNBNode) {
        if (isSolutionFound()) return; // Early exit if solution already found

        currentBNBNode->start();
        currentBNBNode->enforceBranching();
        problem->evaluate(currentBNBNode);
        double boundValue = problem->bound(currentBNBNode);

        if (isSolutionFound()) return; // Check again after a potentially long operation

        if (currentBNBNode->getPrune() || boundValue < globalBestObjective.load()) {
            return; // BNBNode can be pruned
        }

        double objectiveValue = problem->objective(currentBNBNode);
        updateGlobalBest(objectiveValue, currentBNBNode, boundValue);

        if (isSolutionFound()) return; // Exit if the solution was found during update

        branch(currentBNBNode);
    }

    // Solve using multiple threads
    void solveParallel(size_t numThreads) {
        std::vector<std::jthread> threads;
        for (size_t i = 0; i < numThreads; ++i) {
            threads.emplace_back([this]() {
                while (!isSolutionFound()) {
                    BNBNode *currentBNBNode = getNextBNBNode();
                    if (currentBNBNode) {
                        processBNBNode(currentBNBNode);
                    } else {
                        std::this_thread::yield(); // Avoid busy-waiting, yield the thread
                    }
                }
            });
        }
    }

    // Non-parallel version of solve
    void solve() {
        // init timer
        fmt::print("\n");
        fmt::print("\033[34m_STARTING BnB \033[0m");
        fmt::print("\n");
        auto start = std::chrono::high_resolution_clock::now();
        while (BNBNode *currentBNBNode = getNextBNBNode()) {
            currentBNBNode->start();
            currentBNBNode->enforceBranching();
            print_info("Current node id: {}\n", currentBNBNode->getUUID());

            problem->evaluate(currentBNBNode);
            double boundValue = problem->bound(currentBNBNode);
            print_info("Bound value: {}\n", boundValue);

            if (currentBNBNode->getPrune()) {
                print_info("Pruned node: {}\n", currentBNBNode->getUUID());
                continue;
            }

            if (boundValue < globalBestObjective.load()) continue;

            auto objectiveValue = problem->objective(currentBNBNode);
            print_info("Objective value: {}\n", objectiveValue);

            if (std::abs(boundValue - objectiveValue) < 1e-2) {
                fmt::print("\n");
                fmt::print("\033[34m_SOLUTION FOUND \033[0m: {}\n", objectiveValue);

                globalBestObjective.store(objectiveValue, std::memory_order_release);
                markSolutionFound();
                break;
            }

            if (objectiveValue > globalBestObjective.load()) {
                globalBestObjective.store(objectiveValue, std::memory_order_release);
            }

            branch(currentBNBNode);
        }
        auto                          end     = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        print_info("Elapsed time: {:.2f}\n", elapsed.count());
    }

    void branch(BNBNode *&node) {
        problem->branch(node);
        auto children = node->getChildren();
        for (auto &child : children) { addBNBNode(child); }
    }

    [[nodiscard]] auto getBestObjective() const { return globalBestObjective.load(std::memory_order_acquire); }
};
