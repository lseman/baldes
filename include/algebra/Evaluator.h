#pragma once
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <queue>

template <typename Lhs, typename Rhs, int Mode, bool IsLower, bool IsRowMajor>
struct SparseSolveTriangular {
    typedef typename Rhs::Scalar                                    Scalar;
    typedef Eigen::internal::evaluator<Lhs>                         LhsEval;
    typedef typename Eigen::internal::evaluator<Lhs>::InnerIterator LhsIterator;
    typedef typename Lhs::Index                                     Index;

    template <typename Scalar>
    static typename std::enable_if<std::is_floating_point<Scalar>::value, bool>::type is_exactly_zero(Scalar value) {
        return std::abs(value) < std::numeric_limits<Scalar>::epsilon();
    }

    // For integral types, check for exact zero.
    template <typename Scalar>
    static typename std::enable_if<std::is_integral<Scalar>::value, bool>::type is_exactly_zero(Scalar value) {
        return value == Scalar(0);
    }
    static void run(const Lhs &lhs, Rhs &other, Scalar regularizationFactor) {
        LhsEval lhsEval(lhs);

        for (Index col = 0; col < other.cols(); ++col) {
            if constexpr (IsRowMajor) {
                if constexpr (IsLower) {
                    for (Index i = 0; i < lhs.rows(); ++i) {
                        Scalar tmp       = other.coeff(i, col);
                        Scalar lastVal   = 0;
                        Index  lastIndex = 0;

                        for (LhsIterator it(lhsEval, i); it; ++it) {
                            lastVal   = it.value();
                            lastIndex = it.index();
                            if (lastIndex == i) break;
                            tmp -= lastVal * other.coeff(lastIndex, col);
                        }

                        if (Mode & Eigen::UnitDiag)
                            other.coeffRef(i, col) = tmp;
                        else {
                            if (std::abs(lastVal) < regularizationFactor) {
                                lastVal = (lastVal >= 0 ? regularizationFactor : -regularizationFactor);
                            }
                            other.coeffRef(i, col) = tmp / lastVal;
                        }
                    }
                } else {
                    for (Index i = lhs.rows() - 1; i >= 0; --i) {
                        Scalar tmp  = other.coeff(i, col);
                        Scalar l_ii = 0;

                        LhsIterator it(lhsEval, i);
                        while (it && it.index() < i) ++it;
                        if (!(Mode & Eigen::UnitDiag)) {
                            eigen_assert(it && it.index() == i);
                            l_ii = it.value();
                            ++it;
                        } else if (it && it.index() == i)
                            ++it;

                        for (; it; ++it) { tmp -= it.value() * other.coeff(it.index(), col); }

                        if (Mode & Eigen::UnitDiag)
                            other.coeffRef(i, col) = tmp;
                        else {
                            if (std::abs(l_ii) < regularizationFactor) {
                                l_ii = (l_ii >= 0 ? regularizationFactor : -regularizationFactor);
                            }
                            other.coeffRef(i, col) = tmp / l_ii;
                        }
                    }
                }
            } else { // Column-major case
                if constexpr (IsLower) {
                    for (Index i = 0; i < lhs.cols(); ++i) {
                        Scalar &tmp = other.coeffRef(i, col);
                        if (!is_exactly_zero(tmp)) {
                            LhsIterator it(lhsEval, i);
                            while (it && it.index() < i) ++it;
                            if (!(Mode & Eigen::UnitDiag)) {
                                constexpr Scalar epsilon = Scalar(1e-12); // Regularization threshold
                                tmp /= (std::abs(it.value()) < epsilon) ? it.value() + epsilon : it.value();
                            }
                            if (it && it.index() == i) ++it;
                            for (; it; ++it) { other.coeffRef(it.index(), col) -= tmp * it.value(); }
                        }
                    }
                } else {
                    for (Index i = lhs.cols() - 1; i >= 0; --i) {
                        Scalar &tmp = other.coeffRef(i, col);
                        if (!is_exactly_zero(tmp)) {
                            LhsIterator it(lhsEval, i);
                            if (!(Mode & Eigen::UnitDiag)) {
                                constexpr Scalar epsilon = Scalar(1e-12); // Regularization threshold
                                while (it && it.index() != i) ++it;
                                tmp /= (std::abs(it.value()) < epsilon) ? it.value() + epsilon : it.value();
                            }
                            for (; it && it.index() < i; ++it) { other.coeffRef(it.index(), col) -= tmp * it.value(); }
                        }
                    }
                }
            }
        }
    }
};

// Dispatch to the correct specialization
template <typename Lhs, typename Rhs, int Mode>
void solveTriangular(const Lhs &lhs, Rhs &other, typename Lhs::Scalar regularizationFactor) {
    constexpr bool isLower    = (Mode & Eigen::Lower) != 0;
    constexpr bool isRowMajor = (int(Lhs::Flags) & Eigen::RowMajorBit) != 0;

    SparseSolveTriangular<Lhs, Rhs, Mode, isLower, isRowMajor>::run(lhs, other, regularizationFactor);
}

#include <queue>
#include <bitset>   // Or use std::vector<uint8_t> if n is too large
#include <functional> // For std::greater
#include <queue>
#include <bitset>   // Or use std::vector<uint8_t> if n is too large
#include <functional> // For std::greater
#include <limits>     // For max value

class AdvancedFastRCMOrdering {
public:
    template <typename MatrixType>
    void operator()(const MatrixType &matrix, std::vector<int> &permutation) const {
        const int n = matrix.rows();
        permutation.resize(n);
        std::vector<uint8_t> visited(n, 0);  // Optimized memory for large `n`
        std::vector<int> degrees(n);
        std::vector<int> nodeBuffer;
        nodeBuffer.reserve(n);        // Preallocate memory to avoid reallocations
        std::vector<int> componentBuffer;  // To store connected components
        componentBuffer.reserve(n);

        // Compute degrees of each vertex (number of non-zeros per row)
        for (int i = 0; i < n; ++i) {
            degrees[i] = matrix.outerIndexPtr()[i + 1] - matrix.outerIndexPtr()[i];
        }

        int permIndex = 0;

        // Process each connected component separately
        for (int start = 0; start < n; ++start) {
            if (!visited[start]) {
                // Find a pseudo-peripheral node to start
                int startNode = findPseudoPeripheralNode(matrix, visited, degrees, start);

                // BFS traversal starting from the pseudo-peripheral node
                bfsRCM(matrix, startNode, visited, degrees, permutation, permIndex);
            }
        }

        // Reverse the final permutation to obtain RCM ordering
        std::reverse(permutation.begin(), permutation.end());
    }

private:
    template <typename MatrixType>
    int findPseudoPeripheralNode(const MatrixType &matrix, std::vector<uint8_t> &visited,
                                 const std::vector<int> &degrees, int start) const {
        // Multi-level BFS to estimate a pseudo-peripheral node
        int currentNode = start;
        int lastNode = start;

        std::queue<int> bfsQueue;
        bfsQueue.push(currentNode);
        visited[currentNode] = true;

        while (!bfsQueue.empty()) {
            int nodeCount = bfsQueue.size();
            lastNode = currentNode;

            for (int i = 0; i < nodeCount; ++i) {
                currentNode = bfsQueue.front();
                bfsQueue.pop();

                // Traverse neighbors
                for (typename MatrixType::InnerIterator it(matrix, currentNode); it; ++it) {
                    int neighbor = it.index();
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        bfsQueue.push(neighbor);
                    }
                }
            }
        }

        // Reset visited array for actual BFS traversal later
        visited.assign(visited.size(), 0);
        return lastNode;
    }

    template <typename MatrixType>
    void bfsRCM(const MatrixType &matrix, int startNode, std::vector<uint8_t> &visited,
                const std::vector<int> &degrees, std::vector<int> &permutation, int &permIndex) const {
        std::vector<int> nodeBuffer;
        std::queue<int> bfsQueue;
        bfsQueue.push(startNode);
        visited[startNode] = true;

        // BFS traversal
        while (!bfsQueue.empty()) {
            int node = bfsQueue.front();
            bfsQueue.pop();
            nodeBuffer.push_back(node);

            // Explore neighbors of the current node
            for (typename MatrixType::InnerIterator it(matrix, node); it; ++it) {
                int neighbor = it.index();
                if (!visited[neighbor]) {
                    bfsQueue.push(neighbor);
                    visited[neighbor] = true;
                }
            }
        }

        // Sort nodes by degree within the level
        std::sort(nodeBuffer.begin(), nodeBuffer.end(), [&](int a, int b) {
            return degrees[a] < degrees[b];
        });

        // Add the level's nodes in sorted order to the permutation
        for (int i = 0; i < nodeBuffer.size(); ++i) {
            permutation[permIndex++] = nodeBuffer[i];
        }
    }
};
