#ifndef EIGEN_CUSTOM_SIMPLICIAL_LDLT_H
#define EIGEN_CUSTOM_SIMPLICIAL_LDLT_H

#include <vector>
#define EIGEN_USE_MKL_ALL

#include "Evaluator.h"
// #include "EvaluatorGPU.h"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <experimental/simd>
#include <limits>
#include <set>

#include <execution>
#include <stdexec/execution.hpp>

#include "AMD.h"
template <typename T>
using BVector = std::vector<T>;

class Permutation {
public:
    std::vector<int> perm;

    Permutation(int size) : perm(size) {
        for (int i = 0; i < size; ++i) perm[i] = i;
    }

    // Apply the permutation to a vector
    template <typename T>
    std::vector<T> apply(const std::vector<T> &vec) const {
        std::vector<T> result(vec.size());
        for (int i = 0; i < vec.size(); ++i) { result[i] = vec[perm[i]]; }
        return result;
    }

    // Invert the permutation
    Permutation inverse() const {
        Permutation inv_perm(perm.size());
        for (int i = 0; i < perm.size(); ++i) { inv_perm.perm[perm[i]] = i; }
        return inv_perm;
    }
};

namespace Eigen {

template <typename MatrixType_, int UpLo_ = Lower, typename Ordering_ = AMDOrdering<typename MatrixType_::StorageIndex>>
class CustomSimplicialLDLT {

public:
    typedef MatrixType_ MatrixType;
    enum { UpLo = UpLo_ };
    typedef typename MatrixType::Scalar                  Scalar;
    typedef typename MatrixType::RealScalar              RealScalar;
    typedef typename MatrixType::StorageIndex            StorageIndex;
    typedef SparseMatrix<Scalar, ColMajor, StorageIndex> CholMatrixType;
    typedef Matrix<Scalar, Dynamic, 1>                   VectorType;

    typedef TriangularView<const CholMatrixType, Eigen::UnitLower>                             MatrixL;
    typedef TriangularView<const typename CholMatrixType::AdjointReturnType, Eigen::UnitUpper> MatrixU;
    typedef CholMatrixType const                                                              *ConstCholMatrixPtr;

    // First, add these member variables to the CustomSimplicialLDLT class:
private:
    std::vector<std::vector<StorageIndex>> m_supernodes;       // Lists of columns in each supernode
    std::vector<StorageIndex>              m_col_to_supernode; // Maps each column to its supernode
    std::vector<StorageIndex>              m_supernode_sizes;  // Size of each supernode
    StorageIndex                           m_num_supernodes;   // Total number of supernodes

    // Add these methods to the class:
public:
public:
    void computeSupernodes() {
        const StorageIndex size = m_matrix.rows();
        m_col_to_supernode.resize(size, -1);
        m_supernodes.clear();

        // Build similarity graph
        std::vector<std::vector<std::pair<double, StorageIndex>>> similarity_graph(size);
        std::vector<bool>                                         processed(size, false);

        // Phase 1: Build similarity graph with multiple metrics
        for (StorageIndex col = 0; col < size; ++col) {
            auto         base_pattern = getRelaxedColumnPattern(col);
            StorageIndex window_end   = std::min(col + 20, size);

            for (StorageIndex next_col = col + 1; next_col < window_end; ++next_col) {
                auto next_pattern = getRelaxedColumnPattern(next_col);

                // Compute multiple similarity metrics
                double jaccard         = computeJaccardSimilarity(base_pattern, next_pattern, col, next_col);
                double pattern_overlap = computePatternOverlap(base_pattern, next_pattern, col, next_col);
                double structural      = computeStructuralSimilarity(base_pattern, next_pattern, col, next_col);

                // Weighted combination of metrics
                double combined_score = (0.4 * jaccard) + (0.4 * pattern_overlap) + (0.2 * structural);

                // Add edge if score is high enough
                if (combined_score >= 0.4) { // Lowered threshold based on analysis
                    similarity_graph[col].push_back({combined_score, next_col});
                    similarity_graph[next_col].push_back({combined_score, col});
                }
            }
        }

        StorageIndex current_supernode = 0;

        // Phase 2: Form supernodes using connected components
        for (StorageIndex start_col = 0; start_col < size; ++start_col) {
            if (processed[start_col]) continue;

            std::vector<StorageIndex> supernode = {start_col};
            std::vector<StorageIndex> candidates;
            processed[start_col] = true;

            // Find all strongly connected columns
            for (const auto &edge : similarity_graph[start_col]) {
                if (!processed[edge.second] && edge.first >= 0.4) { candidates.push_back(edge.second); }
            }

            // Sort candidates by similarity score
            std::sort(candidates.begin(), candidates.end(),
                      [&similarity_graph, start_col](StorageIndex a, StorageIndex b) {
                          double score_a = 0, score_b = 0;
                          for (const auto &edge : similarity_graph[start_col]) {
                              if (edge.second == a) score_a = edge.first;
                              if (edge.second == b) score_b = edge.first;
                          }
                          return score_a > score_b;
                      });

            // Try to add candidates
            for (StorageIndex candidate : candidates) {
                if (processed[candidate]) continue;

                bool compatible = true;
                // Check compatibility with all current supernode members
                for (StorageIndex member : supernode) {
                    bool   found     = false;
                    double max_score = 0;
                    for (const auto &edge : similarity_graph[member]) {
                        if (edge.second == candidate) {
                            max_score = edge.first;
                            found     = true;
                            break;
                        }
                    }
                    if (!found || max_score < 0.3) { // Relaxed threshold for additional members
                        compatible = false;
                        break;
                    }
                }

                if (compatible && supernode.size() < 8) { // Increased max size
                    supernode.push_back(candidate);
                    processed[candidate] = true;
                }
            }

            // Store the supernode
            if (!supernode.empty()) {
                for (StorageIndex col : supernode) { m_col_to_supernode[col] = current_supernode; }
                m_supernodes.push_back(supernode);
                current_supernode++;
            }
        }

        m_num_supernodes = current_supernode;
        m_supernode_sizes.resize(m_num_supernodes);
        for (StorageIndex i = 0; i < m_num_supernodes; ++i) { m_supernode_sizes[i] = m_supernodes[i].size(); }
    }

private:
    double computeLargePatternSimilarity(const std::vector<StorageIndex> &pattern1,
                                         const std::vector<StorageIndex> &pattern2, StorageIndex col1,
                                         StorageIndex col2) {
        auto it1 = std::lower_bound(pattern1.begin(), pattern1.end(), std::min(col1, col2));
        auto it2 = std::lower_bound(pattern2.begin(), pattern2.end(), std::min(col1, col2));

        size_t remaining1 = std::distance(it1, pattern1.end());
        size_t remaining2 = std::distance(it2, pattern2.end());

        // Strict size ratio check for large patterns
        double size_ratio = std::min(remaining1, remaining2) / double(std::max(remaining1, remaining2));
        if (size_ratio < 0.9) return 0.0;

        // Count matching elements with position-sensitive weighting
        size_t matches        = 0;
        size_t total          = 0;
        double weighted_score = 0.0;

        while (it1 != pattern1.end() && it2 != pattern2.end()) {
            if (*it1 == *it2) {
                // Give more weight to matches near the diagonal
                double position_weight = 1.0 / (1.0 + std::abs(*it1 - col1) * 0.1);
                weighted_score += position_weight;
                matches++;
                ++it1;
                ++it2;
            } else if (*it1 < *it2) {
                ++it1;
            } else {
                ++it2;
            }
            total++;
        }

        // Combine metrics with emphasis on exact matches
        return (0.7 * matches / total) + (0.3 * weighted_score / total);
    }

    void analyzeEarlyColumns() {
        const StorageIndex size          = m_matrix.rows();
        const StorageIndex analyze_count = 10; // Analyze first 10 columns

        std::cout << "\nDetailed Early Column Analysis:\n";

        // Store patterns for comparison
        std::vector<std::vector<StorageIndex>> patterns(analyze_count);

        // First pass: collect patterns
        for (StorageIndex col = 0; col < analyze_count; ++col) {
            patterns[col] = getRelaxedColumnPattern(col);

            std::cout << "\nColumn " << col << ":\n";
            std::cout << "Pattern size: " << patterns[col].size() << "\n";
            std::cout << "Pattern: ";
            for (StorageIndex idx : patterns[col]) { std::cout << idx << " "; }
            std::cout << "\n";

            // Get nonzero structure
            size_t nnz = 0;
            for (typename CholMatrixType::InnerIterator it(m_matrix, col); it; ++it) {
                if (it.row() >= col) nnz++;
            }
            std::cout << "Nonzeros below diagonal: " << nnz << "\n";
        }

        // Compute pairwise similarity metrics
        std::cout << "\nPairwise Similarity Matrix:\n";
        for (StorageIndex i = 0; i < analyze_count; ++i) {
            for (StorageIndex j = i + 1; j < analyze_count; ++j) {
                // Compute various similarity metrics
                double jaccard         = computeJaccardSimilarity(patterns[i], patterns[j], i, j);
                double pattern_overlap = computePatternOverlap(patterns[i], patterns[j], i, j);
                double structural_sim  = computeStructuralSimilarity(patterns[i], patterns[j], i, j);

                std::cout << "Columns " << i << " and " << j << ":\n";
                std::cout << "  Jaccard similarity: " << jaccard << "\n";
                std::cout << "  Pattern overlap: " << pattern_overlap << "\n";
                std::cout << "  Structural similarity: " << structural_sim << "\n";
            }
        }
    }

private:
    double computeJaccardSimilarity(const std::vector<StorageIndex> &pattern1,
                                    const std::vector<StorageIndex> &pattern2, StorageIndex col1, StorageIndex col2) {
        auto it1 = std::lower_bound(pattern1.begin(), pattern1.end(), std::min(col1, col2));
        auto it2 = std::lower_bound(pattern2.begin(), pattern2.end(), std::min(col1, col2));

        size_t intersection = 0;
        size_t union_size   = 0;

        while (it1 != pattern1.end() && it2 != pattern2.end()) {
            if (*it1 == *it2) {
                intersection++;
                union_size++;
                ++it1;
                ++it2;
            } else if (*it1 < *it2) {
                union_size++;
                ++it1;
            } else {
                union_size++;
                ++it2;
            }
        }

        union_size += std::distance(it1, pattern1.end());
        union_size += std::distance(it2, pattern2.end());

        return union_size > 0 ? double(intersection) / union_size : 0.0;
    }

    double computePatternOverlap(const std::vector<StorageIndex> &pattern1, const std::vector<StorageIndex> &pattern2,
                                 StorageIndex col1, StorageIndex col2) {
        auto it1 = std::lower_bound(pattern1.begin(), pattern1.end(), std::min(col1, col2));
        auto it2 = std::lower_bound(pattern2.begin(), pattern2.end(), std::min(col1, col2));

        size_t common_elements = 0;
        size_t total_elements  = std::min(std::distance(it1, pattern1.end()), std::distance(it2, pattern2.end()));

        while (it1 != pattern1.end() && it2 != pattern2.end()) {
            if (*it1 == *it2) {
                common_elements++;
                ++it1;
                ++it2;
            } else if (*it1 < *it2) {
                ++it1;
            } else {
                ++it2;
            }
        }

        return total_elements > 0 ? double(common_elements) / total_elements : 0.0;
    }

    double computeStructuralSimilarity(const std::vector<StorageIndex> &pattern1,
                                       const std::vector<StorageIndex> &pattern2, StorageIndex col1,
                                       StorageIndex col2) {
        // Compare the structure by looking at gaps and distances
        if (pattern1.empty() || pattern2.empty()) return 0.0;

        std::vector<StorageIndex> gaps1, gaps2;
        for (size_t i = 1; i < pattern1.size(); ++i) { gaps1.push_back(pattern1[i] - pattern1[i - 1]); }
        for (size_t i = 1; i < pattern2.size(); ++i) { gaps2.push_back(pattern2[i] - pattern2[i - 1]); }

        // Compare gap patterns
        size_t min_size      = std::min(gaps1.size(), gaps2.size());
        size_t matching_gaps = 0;

        for (size_t i = 0; i < min_size; ++i) {
            if (gaps1[i] == gaps2[i]) matching_gaps++;
        }

        return min_size > 0 ? double(matching_gaps) / min_size : 0.0;
    }

    double computeRegularSimilarity(const std::vector<StorageIndex> &pattern1,
                                    const std::vector<StorageIndex> &pattern2, StorageIndex col1, StorageIndex col2) {
        auto it1 = std::lower_bound(pattern1.begin(), pattern1.end(), std::min(col1, col2));
        auto it2 = std::lower_bound(pattern2.begin(), pattern2.end(), std::min(col1, col2));

        size_t remaining1 = std::distance(it1, pattern1.end());
        size_t remaining2 = std::distance(it2, pattern2.end());

        double size_ratio = std::min(remaining1, remaining2) / double(std::max(remaining1, remaining2));
        if (size_ratio < 0.7) return 0.0;

        size_t common = 0;
        size_t total  = 0;

        while (it1 != pattern1.end() || it2 != pattern2.end()) {
            if (it1 == pattern1.end()) {
                ++it2;
                ++total;
            } else if (it2 == pattern2.end()) {
                ++it1;
                ++total;
            } else if (*it1 == *it2) {
                ++common;
                ++total;
                ++it1;
                ++it2;
            } else if (*it1 < *it2) {
                ++it1;
                ++total;
            } else {
                ++it2;
                ++total;
            }
        }

        return double(common) / total;
    }

private:
    double computeAdaptiveSimilarity(const std::vector<StorageIndex> &pattern1,
                                     const std::vector<StorageIndex> &pattern2, StorageIndex col1, StorageIndex col2) {
        if (pattern1.empty() || pattern2.empty()) return 0.0;

        // Find starting points after the diagonal entries
        auto it1 = std::lower_bound(pattern1.begin(), pattern1.end(), std::min(col1, col2));
        auto it2 = std::lower_bound(pattern2.begin(), pattern2.end(), std::min(col1, col2));

        size_t remaining1 = std::distance(it1, pattern1.end());
        size_t remaining2 = std::distance(it2, pattern2.end());

        // Early size compatibility check
        double size_ratio = std::min(remaining1, remaining2) / double(std::max(remaining1, remaining2));
        if (size_ratio < 0.6) return 0.0;

        // Count common and total elements
        size_t common = 0;
        size_t total  = 0;
        auto   end1   = pattern1.end();
        auto   end2   = pattern2.end();

        while (it1 != end1 || it2 != end2) {
            if (it1 == end1) {
                ++it2;
                ++total;
            } else if (it2 == end2) {
                ++it1;
                ++total;
            } else if (*it1 == *it2) {
                ++common;
                ++total;
                ++it1;
                ++it2;
            } else if (*it1 < *it2) {
                ++it1;
                ++total;
            } else {
                ++it2;
                ++total;
            }
        }

        // Weighted score combining various metrics
        double jaccard = double(common) / total;
        double pattern_density =
            std::min(remaining1, remaining2) / double(std::max(pattern1.back() - col1, pattern2.back() - col2) + 1);

        return 0.7 * jaccard + 0.3 * pattern_density;
    }

private:
    void computeEliminationTreeLevels(std::vector<StorageIndex> &levels) {
        const StorageIndex size = m_matrix.rows();
        std::fill(levels.begin(), levels.end(), 0);

        for (StorageIndex i = 0; i < size; ++i) {
            StorageIndex level = 0;
            StorageIndex k     = m_parent[i];
            while (k != -1) {
                level++;
                k = m_parent[k];
            }
            levels[i] = level;
        }
    }

    double computeEnhancedSimilarity(const std::vector<StorageIndex> &pattern1,
                                     const std::vector<StorageIndex> &pattern2, StorageIndex col1, StorageIndex col2) {
        auto it1 = std::lower_bound(pattern1.begin(), pattern1.end(), std::min(col1, col2));
        auto it2 = std::lower_bound(pattern2.begin(), pattern2.end(), std::min(col1, col2));

        size_t remaining1 = std::distance(it1, pattern1.end());
        size_t remaining2 = std::distance(it2, pattern2.end());

        // Base similarity calculation
        size_t common       = 0;
        size_t total_unique = 0;
        auto   end1         = pattern1.end();
        auto   end2         = pattern2.end();

        std::vector<StorageIndex> unique_indices;
        unique_indices.reserve(remaining1 + remaining2);

        while (it1 != end1 || it2 != end2) {
            if (it1 == end1) {
                unique_indices.push_back(*it2++);
            } else if (it2 == end2) {
                unique_indices.push_back(*it1++);
            } else if (*it1 == *it2) {
                common++;
                unique_indices.push_back(*it1);
                ++it1;
                ++it2;
            } else if (*it1 < *it2) {
                unique_indices.push_back(*it1++);
            } else {
                unique_indices.push_back(*it2++);
            }
        }

        total_unique = unique_indices.size();

        // Calculate Jaccard similarity
        double jaccard = total_unique > 0 ? double(common) / total_unique : 0.0;

        // Calculate density similarity
        double density1    = remaining1 > 0 ? double(remaining1) / (pattern1.back() - col1 + 1) : 0.0;
        double density2    = remaining2 > 0 ? double(remaining2) / (pattern2.back() - col2 + 1) : 0.0;
        double density_sim = 1.0 - std::abs(density1 - density2);

        // Combine metrics with weights
        return 0.7 * jaccard + 0.3 * density_sim;
    }

private:
    double computePatternSimilarity(const std::vector<StorageIndex> &pattern1,
                                    const std::vector<StorageIndex> &pattern2, StorageIndex col1, StorageIndex col2) {
        // Find starting points after the diagonal entries
        auto it1 = std::lower_bound(pattern1.begin(), pattern1.end(), std::min(col1, col2));
        auto it2 = std::lower_bound(pattern2.begin(), pattern2.end(), std::min(col1, col2));

        size_t remaining1 = std::distance(it1, pattern1.end());
        size_t remaining2 = std::distance(it2, pattern2.end());

        // First check size ratio
        double size_ratio = std::min(remaining1, remaining2) / double(std::max(remaining1, remaining2));
        if (size_ratio < 0.5) return 0.0;

        // Count common elements
        size_t common = 0;
        auto   end1   = pattern1.end();
        auto   end2   = pattern2.end();

        while (it1 != end1 && it2 != end2) {
            if (*it1 == *it2) {
                common++;
                ++it1;
                ++it2;
            } else if (*it1 < *it2) {
                ++it1;
            } else {
                ++it2;
            }
        }

        // Consider both pattern similarity and size ratio
        double similarity = double(common) / double(std::max(remaining1, remaining2));
        return similarity * size_ratio;
    }

    std::vector<StorageIndex> getRelaxedColumnPattern(StorageIndex col) {
        const StorageIndex        size = m_matrix.rows();
        std::vector<StorageIndex> pattern;
        std::vector<bool>         visited(size, false);
        pattern.reserve(32); // Reserve reasonable size for efficiency

        // Get nonzeros below or on diagonal
        for (typename CholMatrixType::InnerIterator it(m_matrix, col); it; ++it) {
            StorageIndex row = it.row();
            if (row >= col && !visited[row]) {
                pattern.push_back(row);
                visited[row] = true;
            }
        }

        // Follow elimination tree path
        StorageIndex k = col;
        while (k != -1 && !visited[k]) {
            visited[k] = true;
            pattern.push_back(k);
            k = m_parent[k];
        }

        std::sort(pattern.begin(), pattern.end());
        return pattern;
    }

    std::vector<StorageIndex> getColumnPattern(StorageIndex col) {
        const StorageIndex        size = m_matrix.rows();
        std::vector<StorageIndex> pattern;
        std::vector<bool>         visited(size, false);
        pattern.reserve(size);

        // Get direct nonzeros
        for (typename CholMatrixType::InnerIterator it(m_matrix, col); it; ++it) {
            StorageIndex row = it.row();
            if (row >= col && !visited[row]) {
                pattern.push_back(row);
                visited[row] = true;
            }
        }

        // Follow elimination tree
        StorageIndex k = col;
        while (k != -1 && !visited[k]) {
            visited[k] = true;
            pattern.push_back(k);
            k = m_parent[k];
        }

        std::sort(pattern.begin(), pattern.end());
        return pattern;
    }

    void debugSupernodes() {
        std::cout << "\nSupernode Analysis:" << std::endl;
        std::cout << "Total matrix size: " << m_matrix.rows() << "x" << m_matrix.cols() << std::endl;
        std::cout << "Number of nonzeros: " << m_matrix.nonZeros() << std::endl;
        std::cout << "Number of supernodes: " << m_num_supernodes << std::endl;

        size_t total_columns      = 0;
        size_t largest_supernode  = 0;
        size_t smallest_supernode = std::numeric_limits<size_t>::max();

        for (StorageIndex i = 0; i < m_num_supernodes; ++i) {
            const auto &supernode  = m_supernodes[i];
            size_t      snode_size = supernode.size();
            total_columns += snode_size;
            largest_supernode  = std::max(largest_supernode, snode_size);
            smallest_supernode = std::min(smallest_supernode, snode_size);

            // Print detailed info for first few and last few supernodes
            if (i < 5 || i >= m_num_supernodes - 5) {
                std::cout << "Supernode " << i << ":" << std::endl;
                std::cout << "  Size: " << snode_size << " columns" << std::endl;
                std::cout << "  Columns: ";
                for (StorageIndex col : supernode) { std::cout << col << " "; }
                std::cout << std::endl;

                // Print pattern size for first column of supernode
                if (!supernode.empty()) {
                    auto pattern = getColumnPattern(supernode[0]);
                    std::cout << "  Pattern size: " << pattern.size() << std::endl;
                }
            }
        }

        std::cout << "\nSummary:" << std::endl;
        std::cout << "Total columns in supernodes: " << total_columns << std::endl;
        std::cout << "Largest supernode size: " << largest_supernode << std::endl;
        std::cout << "Smallest supernode size: " << smallest_supernode << std::endl;
        std::cout << "Average supernode size: " << (double)total_columns / m_num_supernodes << std::endl;
    }

private:
    bool haveSameNonzeroPattern(StorageIndex col1, StorageIndex col2) {
        // Get column pointers
        const StorageIndex *col1_ptr    = m_matrix.outerIndexPtr();
        const StorageIndex *col2_ptr    = m_matrix.outerIndexPtr();
        const StorageIndex *row_indices = m_matrix.innerIndexPtr();

        // Get ranges for both columns
        StorageIndex start1 = col1_ptr[col1];
        StorageIndex end1   = col1_ptr[col1 + 1];
        StorageIndex start2 = col2_ptr[col2];
        StorageIndex end2   = col2_ptr[col2 + 1];

        // Check if columns have same length
        if ((end1 - start1) != (end2 - start2)) return false;

        // Compare row indices
        for (StorageIndex i = 0; i < (end1 - start1); ++i) {
            if (row_indices[start1 + i] != row_indices[start2 + i]) { return false; }
        }

        return true;
    }

private:
    void factorize_supernodal() {
        const StorageIndex size = m_matrix.rows();
        std::cout << "Starting supernodal factorization with matrix size: " << size << std::endl;

        // Workspace vectors with debug checks
        std::vector<Scalar>       dense_block;
        std::vector<StorageIndex> pattern;
        std::vector<StorageIndex> flag(size, -1);
        std::vector<StorageIndex> row_indices(size, -1);
        pattern.reserve(size);

        for (StorageIndex snode = 0; snode < m_num_supernodes; ++snode) {
            // std::cout << "Processing supernode " << snode << "/" << m_num_supernodes << std::endl;

            const auto &supernode_cols = m_supernodes[snode];
            if (supernode_cols.empty()) {
                std::cout << "Empty supernode, skipping" << std::endl;
                continue;
            }

            const StorageIndex snode_size = supernode_cols.size();
            const StorageIndex first_col  = supernode_cols[0];

            // Build pattern and map row indices
            pattern.clear();
            StorageIndex pattern_size = 0;

            // First, collect all row indices for this supernode
            for (StorageIndex j = 0; j < snode_size; ++j) {
                StorageIndex col = supernode_cols[j];
                if (col >= size) {
                    std::cout << "Invalid column index: " << col << std::endl;
                    continue;
                }

                for (typename CholMatrixType::InnerIterator it(m_matrix, col); it; ++it) {
                    StorageIndex row = it.row();
                    if (row >= first_col && flag[row] == -1) {
                        flag[row] = pattern_size;
                        pattern.push_back(row);
                        row_indices[pattern_size] = row;
                        pattern_size++;
                    }
                }
            }

            // std::cout << "Supernode " << snode << " pattern size: " << pattern_size << std::endl;

            // Allocate and initialize dense block
            dense_block.resize(pattern_size * snode_size);
            std::fill(dense_block.begin(), dense_block.end(), Scalar(0));

            // Copy data to dense block with bounds checking
            for (StorageIndex j = 0; j < snode_size; ++j) {
                StorageIndex col = supernode_cols[j];
                // std::cout << "Copying column " << j << "/" << snode_size << std::endl;

                for (typename CholMatrixType::InnerIterator it(m_matrix, col); it; ++it) {
                    StorageIndex row     = it.row();
                    StorageIndex row_pos = flag[row];

                    // std::cout << "Copying row " << row << "/" << row_pos << std::endl;

                    if (row_pos >= 0 && row_pos < pattern_size) {
                        dense_block[j * pattern_size + row_pos] = it.value();
                    }
                }
            }

            // Perform LDLT factorization on dense block
            for (StorageIndex k = 0; k < snode_size; ++k) {
                StorageIndex global_k = supernode_cols[k];
                if (k >= pattern_size) continue;

                // Get and validate diagonal element
                Scalar d = dense_block[k * pattern_size + k];
                if (std::abs(d) < std::numeric_limits<RealScalar>::epsilon()) {
                    d = std::numeric_limits<RealScalar>::epsilon();
                }

                // Update diagonal vector
                if (global_k < m_diag.size()) {
                    m_diag[global_k] = d;
                } else {
                    // std::cout << "Invalid diagonal index: " << global_k << std::endl;
                    continue;
                }

                // Update L factors
                for (StorageIndex i = k + 1; i < pattern_size; ++i) {
                    StorageIndex idx = k * pattern_size + i;
                    if (idx < dense_block.size()) { dense_block[idx] /= d; }
                }

                // Update remaining block
                for (StorageIndex j = k + 1; j < snode_size && j < pattern_size; ++j) {
                    Scalar multiplier = dense_block[j * pattern_size + k];
                    for (StorageIndex i = j; i < pattern_size; ++i) {
                        StorageIndex update_idx = k * pattern_size + i;
                        StorageIndex target_idx = j * pattern_size + i;
                        if (update_idx < dense_block.size() && target_idx < dense_block.size()) {
                            dense_block[target_idx] -= multiplier * numext::conj(dense_block[update_idx]);
                        }
                    }
                }
            }

            // Copy results back to sparse matrix
            for (StorageIndex j = 0; j < snode_size; ++j) {
                StorageIndex col = supernode_cols[j];
                if (col >= size) continue;

                StorageIndex col_start = m_matrix.outerIndexPtr()[col];
                StorageIndex nnz       = 0;

                // Only copy elements below or on the diagonal
                for (StorageIndex i = 0; i < pattern_size; ++i) {
                    StorageIndex row = row_indices[i];
                    if (row >= col) {
                        StorageIndex pos = j * pattern_size + i;
                        if (pos < dense_block.size() && col_start + nnz < m_matrix.nonZeros() &&
                            std::abs(dense_block[pos]) > std::numeric_limits<RealScalar>::epsilon()) {

                            m_matrix.innerIndexPtr()[col_start + nnz] = row;
                            m_matrix.valuePtr()[col_start + nnz]      = dense_block[pos];
                            ++nnz;
                        }
                    }
                }

                // Update nonzeros count
                if (col < m_nonZerosPerCol.size()) { m_nonZerosPerCol[col] = nnz; }
            }

            // Reset flags
            for (StorageIndex i = 0; i < pattern_size; ++i) { flag[row_indices[i]] = -1; }
        }
    }

public:
    CustomSimplicialLDLT()
        : m_isInitialized(false), m_info(Success), m_P(0), m_Pinv(0), m_matrix(1, 1), m_L(m_matrix),
          m_U(m_matrix.adjoint()), m_epsilon(1e-9), m_factorizationIsOk(false) {}

    template <typename Lhs, typename Rhs, int Mode, bool IsLower, bool IsRowMajor>
    struct SparseSolveTriangular;

    explicit CustomSimplicialLDLT(const MatrixType &matrix) : m_isInitialized(false), m_info(Success), m_epsilon(1e-9) {
        // compute(matrix);
    }

    exec::static_thread_pool            pool  = exec::static_thread_pool(std::thread::hardware_concurrency());
    exec::static_thread_pool::scheduler sched = pool.get_scheduler();

    CustomSimplicialLDLT &compute(const MatrixType &matrix) {
        analyzePattern(matrix);
        factorize_preordered<true, false>(m_matrix);
        return *this;
    }

    void analyzePattern_preordered(const CholMatrixType &ap, bool doLDLT) {
        const StorageIndex size = StorageIndex(ap.rows());

        // Pre-allocate all vectors at once with appropriate sizes
        m_matrix.resize(size, size);
        m_parent.resize(size);
        m_nonZerosPerCol.resize(size);
        std::vector<StorageIndex> tags(size);

        // Initialize arrays - use memset for better performance on POD types
        std::memset(m_parent.data(), -1, size * sizeof(StorageIndex));
        std::memset(m_nonZerosPerCol.data(), 0, size * sizeof(StorageIndex));

        // Compute elimination tree and count nonzeros per column
        for (StorageIndex k = 0; k < size; ++k) {
            tags[k] = k; // Mark node k as visited

            // Traverse column k using iterator
            for (typename CholMatrixType::InnerIterator it(ap, k); it; ++it) {
                StorageIndex i = it.index();
                if (i < k) {
                    // Use local variables to reduce memory access
                    StorageIndex current = i;
                    StorageIndex parent;

                    // Follow path from i to root of etree
                    while (tags[current] != k) {
                        parent = m_parent[current];
                        if (parent == -1) {
                            m_parent[current] = k;
                            parent            = k;
                        }
                        m_nonZerosPerCol[current]++;
                        tags[current] = k;
                        current       = parent;
                    }
                }
            }
        }

        // Build column pointers array
        StorageIndex *Lp            = m_matrix.outerIndexPtr();
        StorageIndex  running_total = 0;

        // Use prefix sum for better cache utilization
        Lp[0] = 0;
        for (StorageIndex k = 0; k < size; ++k) {
            running_total += m_nonZerosPerCol[k];
            Lp[k + 1] = running_total;
        }

        // Allocate space for non-zeros
        m_matrix.resizeNonZeros(running_total);

        // Set status flags
        m_isInitialized     = true;
        m_info              = Success;
        m_analysisIsOk      = true;
        m_factorizationIsOk = false;
    }
    typedef typename MatrixType::RealScalar DiagonalScalar;
    static inline DiagonalScalar            getDiag(Scalar x) { return numext::real(x); }
    static inline Scalar                    getSymm(Scalar x) { return numext::conj(x); }

    void reset() {
        m_isInitialized     = false;
        m_analysisIsOk      = false;
        m_factorizationIsOk = false;
        m_info              = Success;
        patternAnalyzed     = false;
    }

    bool patternAnalyzed = false;
    void factorizeMatrix(const MatrixType &matrix) {
        auto matrixToFactorize = matrix;
        if (!patternAnalyzed) {
            analyzePattern(matrixToFactorize); // Analyze the sparsity pattern
            patternAnalyzed = true;
        }
        factorize(matrixToFactorize);
    }

    template <bool DoLDLT, bool NonHermitian>
    void factorize_preordered(const CholMatrixType &ap) {
        const StorageIndex  size = StorageIndex(ap.rows());
        const StorageIndex *Lp   = m_matrix.outerIndexPtr();
        StorageIndex       *Li   = m_matrix.innerIndexPtr();
        Scalar             *Lx   = m_matrix.valuePtr();

        alignas(64) std::vector<Scalar>       y(size, Scalar(0));
        alignas(64) std::vector<StorageIndex> pattern(size, 0);
        alignas(64) std::vector<StorageIndex> tags(size, 0);

        if (m_num_supernodes > 0) {
            fmt::print("Factorizing supernodal\n");
            factorize_supernodal();
            m_info = Success;
            fmt::print("Factorized supernodal\n");
        } else {
            m_diag.resize(size);
            std::atomic<bool> ok{true};

            // Optimize thread count and chunk size
            const int hardware_threads     = std::thread::hardware_concurrency();
            const int min_tasks_per_thread = 32;
            const int chunk_size = std::max(size / (hardware_threads * min_tasks_per_thread), StorageIndex(1));

            using simd_vec           = std::experimental::native_simd<double>;
            constexpr int simd_width = simd_vec::size();

            auto bulk_sender = stdexec::bulk(
                stdexec::just(), (size + chunk_size - 1) / chunk_size,
                [this, &y, &pattern, &tags, Lp, Li, Lx, size, &ap, &ok, chunk_size, simd_width](std::size_t chunk_idx) {
                    const size_t start_k = chunk_idx * chunk_size;
                    const size_t end_k   = std::min(start_k + chunk_size, size_t(size));

                    std::vector<Scalar> y_local(size, Scalar(0));

                    for (size_t k = start_k; k < end_k && ok; ++k) {
                        StorageIndex top    = size;
                        tags[k]             = k;
                        m_nonZerosPerCol[k] = 0;

                        // Process column k using local buffer
                        for (typename CholMatrixType::InnerIterator it(ap, k); it; ++it) {
                            StorageIndex i = it.index();
                            if (i <= k) {
                                y_local[i] += getSymm(it.value());

                                StorageIndex len = 0;
                                for (; tags[i] != k; i = m_parent[i]) {
                                    pattern[len++] = i;
                                    tags[i]        = k;
                                }
                                while (len > 0) pattern[--top] = pattern[--len];
                            }
                        }

                        DiagonalScalar d = getDiag(y_local[k]) * m_shiftScale + m_shiftOffset;
                        y_local[k]       = Scalar(0);

                        for (; top < size; ++top) {
                            const StorageIndex i  = pattern[top];
                            const Scalar       yi = y_local[i];
                            y_local[i]            = Scalar(0);

                            const Scalar       l_ki = yi / getDiag(m_diag[i]);
                            const StorageIndex p2   = Lp[i] + m_nonZerosPerCol[i];

                            // Vectorized sparse update with improved cache efficiency
                            for (StorageIndex p = Lp[i]; p < p2; p += simd_width) {
                                const int remaining = std::min(simd_width, static_cast<int>(p2 - p));
                                if (remaining < simd_width) {
                                    // Handle remaining elements sequentially
                                    for (int k = 0; k < remaining; ++k) {
                                        y_local[Li[p + k]] -= getSymm(Lx[p + k]) * yi;
                                    }
                                } else {
                                    // Full SIMD processing
                                    simd_vec     lx_vec;
                                    StorageIndex vec_indices[simd_width];

                                    // Load data
                                    for (int k = 0; k < simd_width; ++k) {
                                        lx_vec[k]      = getSymm(Lx[p + k]);
                                        vec_indices[k] = Li[p + k];
                                    }

                                    // Compute and store
                                    auto result = lx_vec * simd_vec(yi);
                                    for (int k = 0; k < simd_width; ++k) { y_local[vec_indices[k]] -= result[k]; }
                                }
                            }

                            d -= getDiag(l_ki * getSymm(yi));
                            Li[p2] = k;
                            Lx[p2] = l_ki;
                            ++m_nonZerosPerCol[i];
                        }

                        m_diag[k] = d;
                        if (d == RealScalar(0)) {
                            ok.store(false, std::memory_order_relaxed);
                            break;
                        }
                    }
                });

            stdexec::sync_wait(stdexec::when_all(bulk_sender));
            m_info = ok ? Success : NumericalIssue;
        }
        m_factorizationIsOk = true;
    }

    template <int SrcMode_, int DstMode_, bool NonHermitian, typename MatrixType, int DstOrder>
    void
    permute_symm_to_symm(const MatrixType                                                                       &mat,
                         SparseMatrix<typename MatrixType::Scalar, DstOrder, typename MatrixType::StorageIndex> &_dest,
                         const typename MatrixType::StorageIndex                                                *perm) {
        using StorageIndex = typename MatrixType::StorageIndex;
        using Scalar       = typename MatrixType::Scalar;
        SparseMatrix<Scalar, DstOrder, StorageIndex> &dest(_dest.derived());
        using VectorI     = Matrix<StorageIndex, Dynamic, 1>;
        using MatEval     = internal::evaluator<MatrixType>;
        using MatIterator = CustomMatIterator<MatrixType>;

        enum {
            SrcOrder          = MatrixType::IsRowMajor ? RowMajor : ColMajor,
            StorageOrderMatch = int(SrcOrder) == int(DstOrder),
            DstMode           = DstOrder == RowMajor ? (DstMode_ == Upper ? Lower : Upper) : DstMode_,
            SrcMode           = SrcOrder == RowMajor ? (SrcMode_ == Upper ? Lower : Upper) : SrcMode_
        };

        MatEval matEval(mat);
        Index   size = mat.rows();
        VectorI count(size);
        count.setZero();
        dest.resize(size, size);

        const bool isLower    = int(SrcMode) == int(Lower);
        const bool isUpper    = int(SrcMode) == int(Upper);
        const bool isDstLower = int(DstMode) == int(Lower);

        // Precompute permutation
        std::vector<StorageIndex> perm_cache(size);
        for (StorageIndex j = 0; j < size; ++j) { perm_cache[j] = perm ? perm[j] : j; }

        // First pass: Count the non-zero elements for each column/row
        for (StorageIndex j = 0; j < size; ++j) {
            StorageIndex jp = perm_cache[j];

            for (MatIterator it(matEval, j); it; ++it) {
                StorageIndex i = it.index();
                if ((isLower && i < j) || (isUpper && i > j)) continue;

                StorageIndex ip = perm_cache[i];

                // Minimize conditional checks
                StorageIndex min_ip_jp = std::min(ip, jp);
                StorageIndex max_ip_jp = std::max(ip, jp);

                count[isDstLower ? min_ip_jp : max_ip_jp]++;
            }
        }

        // Allocate space based on the counted non-zero entries
        dest.outerIndexPtr()[0] = 0;
        for (Index j = 0; j < size; ++j) { dest.outerIndexPtr()[j + 1] = dest.outerIndexPtr()[j] + count[j]; }
        dest.resizeNonZeros(dest.outerIndexPtr()[size]);

        // Reset counts for actual filling
        for (Index j = 0; j < size; ++j) { count[j] = dest.outerIndexPtr()[j]; }

        // Main loop: Populate the destination sparse matrix
        for (StorageIndex j = 0; j < size; ++j) {
            StorageIndex jp = perm_cache[j];
            //__builtin_prefetch(&dest.innerIndexPtr()[count[j]], 1, 1); // Prefetch for insertion

            for (MatIterator it(matEval, j); it; ++it) {
                StorageIndex i = it.index();
                if ((isLower && i < j) || (isUpper && i > j)) continue;

                StorageIndex ip = perm_cache[i];

                // Minimize conditional checks
                StorageIndex min_ip_jp = std::min(ip, jp);
                StorageIndex max_ip_jp = std::max(ip, jp);

                Index k                 = count[isDstLower ? min_ip_jp : max_ip_jp]++;
                dest.innerIndexPtr()[k] = isDstLower ? max_ip_jp : min_ip_jp;

                // Prefetch values for efficient memory access
                //__builtin_prefetch(&dest.valuePtr()[k], 1, 1);

                if (!StorageOrderMatch) std::swap(ip, jp);
                if ((isDstLower && ip < jp) || (!isDstLower && ip > jp)) {
                    dest.valuePtr()[k] = NonHermitian ? it.value() : numext::conj(it.value());
                } else {
                    dest.valuePtr()[k] = it.value();
                }
            }
        }
    }

    template <int Mode, bool NonHermitian, typename MatrixType, int DestOrder>
    void permute_symm_to_fullsymm(
        const MatrixType                                                                        &mat,
        SparseMatrix<typename MatrixType::Scalar, DestOrder, typename MatrixType::StorageIndex> &_dest,
        const typename MatrixType::StorageIndex                                                 *perm) {

        using StorageIndex = typename MatrixType::StorageIndex;
        using Scalar       = typename MatrixType::Scalar;
        using Dest         = SparseMatrix<Scalar, DestOrder, StorageIndex>;
        using VectorI      = Matrix<StorageIndex, Dynamic, 1>;
        using MatEval      = internal::evaluator<MatrixType>;
        using MatIterator  = typename internal::evaluator<MatrixType>::InnerIterator;

        MatEval matEval(mat);
        Dest   &dest(_dest.derived());

        enum { StorageOrderMatch = int(Dest::IsRowMajor) == int(MatrixType::IsRowMajor) };

        Index   size = mat.rows();
        VectorI count(size);
        count.setZero();
        dest.resize(size, size);

        // First pass: Count non-zeros for each column
        for (Index j = 0; j < size; ++j) {
            Index jp = perm ? perm[j] : j;

            for (MatIterator it(matEval, j); it; ++it) {
                Index i  = it.index();
                Index r  = it.row();
                Index c  = it.col();
                Index ip = perm ? perm[i] : i;

                if constexpr (Mode == int(Upper | Lower)) {
                    count[StorageOrderMatch ? jp : ip]++;
                } else if (r == c) {
                    count[ip]++;
                } else if ((Mode == Lower && r > c) || (Mode == Upper && r < c)) {
                    count[ip]++;
                    count[jp]++;
                }
            }
        }

        Index nnz = count.sum();

        // Resize for non-zeros and fill outer index
        dest.resizeNonZeros(nnz);
        dest.outerIndexPtr()[0] = 0;

        // Unrolling this loop for small loop performance gain
        for (Index j = 0; j < size; ++j) { dest.outerIndexPtr()[j + 1] = dest.outerIndexPtr()[j] + count[j]; }

        // Reset count for actual insertion
        for (Index j = 0; j < size; ++j) { count[j] = dest.outerIndexPtr()[j]; }

        // Second pass: Copy data into destination matrix
        for (StorageIndex j = 0; j < size; ++j) {
            for (MatIterator it(matEval, j); it; ++it) {
                StorageIndex i = internal::convert_index<StorageIndex>(it.index());
                Index        r = it.row();
                Index        c = it.col();

                StorageIndex jp = perm ? perm[j] : j;
                StorageIndex ip = perm ? perm[i] : i;

                if constexpr (Mode == int(Upper | Lower)) {
                    Index k                 = count[StorageOrderMatch ? jp : ip]++;
                    dest.innerIndexPtr()[k] = StorageOrderMatch ? ip : jp;
                    dest.valuePtr()[k]      = it.value();
                } else if (r == c) {
                    Index k                 = count[ip]++;
                    dest.innerIndexPtr()[k] = ip;
                    dest.valuePtr()[k]      = it.value();
                } else if (((Mode & Lower) == Lower && r > c) || ((Mode & Upper) == Upper && r < c)) {
                    if (!StorageOrderMatch) std::swap(ip, jp);
                    Index k                 = count[jp]++;
                    dest.innerIndexPtr()[k] = ip;
                    dest.valuePtr()[k]      = it.value();
                    k                       = count[ip]++;
                    dest.innerIndexPtr()[k] = jp;
                    dest.valuePtr()[k]      = (NonHermitian ? it.value() : numext::conj(it.value()));
                }
            }
        }
    }

    template <bool NonHermitian>
    void ordering_local(const MatrixType &a, ConstCholMatrixPtr &pmat, CholMatrixType &ap) {
        const Index size = a.rows();
        pmat             = &ap;
        // Note that ordering methods compute the inverse permutation
        CholMatrixType C;
        permute_symm_to_fullsymm<UpLo, NonHermitian>(a, C, NULL);
        // Ordering_ ordering;
        CholMatrixType symm;
        internal::ordering_helper_at_plus_a(C, symm);
        //(symm, m_Pinv);
        internal::minimum_degree_ordering(symm, m_Pinv);

        // if (m_Pinv.size() > 0)
        m_P = m_Pinv.inverse();
        // else
        //     m_P.resize(0);

        ap.resize(size, size);
        permute_symm_to_symm<UpLo, Upper, false>(a, ap, m_P.indices().data());
    }

    void analyzePattern(const MatrixType &a) {
        Index size = a.cols();

        CholMatrixType     tmp(size, size);
        ConstCholMatrixPtr pmat;
        ordering_local<false>(a, pmat, tmp);
        analyzePattern_preordered(*pmat, true);

        computeSupernodes(); // Add this line
        // debugSupernodes();
        // analyzeEarlyColumns();
        // print m_num_supernodes

        // computeSupernodes();
        // printSupernodes();
    }

    StorageIndex m_size;

    ComputationInfo info() const { return m_info; }

    const MatrixL matrixL() const { return m_L; }

    const MatrixU matrixU() const { return m_U; }

    template <typename T>
    void elementwise_divide(BVector<T> &vec, const BVector<T> &diag) {
        for (int i = 0; i < vec.size(); ++i) { vec[i] /= diag[i]; }
    }

    template <typename Rhs>
    Eigen::VectorXd solve(const MatrixBase<Rhs> &b) const {
        eigen_assert(m_isInitialized && "Decomposition not initialized.");

        Eigen::VectorXd dest;

        // Apply forward permutation
        if (m_P.size() > 0) {
            dest = m_P * b;
        } else {
            dest = b;
        }

        // Choose between supernodal and regular solve
        if (m_num_supernodes > 0) {
            supernodal_solve(dest);
        } else {
            // Regular solve path
            const Scalar residualNorm         = (matrixL() * dest - b).norm();
            const Scalar regularizationFactor = std::min(Scalar(1e-10), residualNorm * Scalar(1e-12));

            solveTriangular<decltype(matrixL()), decltype(dest), Lower>(matrixL(), dest, regularizationFactor);
            dest.array() /= m_diag.array();
            solveTriangular<decltype(matrixU()), decltype(dest), Upper>(matrixU(), dest, regularizationFactor);
        }

        // Apply backward permutation
        if (m_Pinv.size() > 0) { dest = m_Pinv * dest; }

        return dest;
    }

private:
    void supernodal_solve(Eigen::VectorXd &x) const {
        const StorageIndex size = x.size();

        // Forward substitution with supernodes
        for (StorageIndex snode = 0; snode < m_num_supernodes; ++snode) {
            const auto &supernode = m_supernodes[snode];
            if (supernode.empty()) continue;

            const StorageIndex snode_size = supernode.size();
            const StorageIndex first_col  = supernode[0];
            const StorageIndex last_col   = supernode[snode_size - 1];

            if (snode_size == 1) {
                // Handle singleton supernode
                if (first_col >= size) continue;

                StorageIndex col_start = m_matrix.outerIndexPtr()[first_col];
                StorageIndex next_col_start =
                    (first_col + 1 < m_matrix.cols()) ? m_matrix.outerIndexPtr()[first_col + 1] : m_matrix.nonZeros();
                StorageIndex col_size = next_col_start - col_start;

                // Update x[col] with diagonal element
                if (m_diag[first_col] != Scalar(0)) { x[first_col] /= m_diag[first_col]; }

                // Update remaining elements
                for (StorageIndex i = 0; i < col_size; ++i) {
                    StorageIndex row = m_matrix.innerIndexPtr()[col_start + i];
                    if (row >= size) continue;
                    if (row > first_col) { x[row] -= m_matrix.valuePtr()[col_start + i] * x[first_col]; }
                }
            } else {
                // Verify all indices are within bounds
                bool valid_supernode = true;
                for (StorageIndex j = 0; j < snode_size; ++j) {
                    if (supernode[j] >= size) {
                        valid_supernode = false;
                        break;
                    }
                }
                if (!valid_supernode) continue;

                // Handle supernode block
                Eigen::MatrixXd dense_block = Eigen::MatrixXd::Zero(snode_size, snode_size);
                Eigen::VectorXd snode_x(snode_size);

                // Extract the diagonal block and corresponding x values
                for (StorageIndex j = 0; j < snode_size; ++j) {
                    StorageIndex col = supernode[j];
                    snode_x(j)       = x[col];

                    StorageIndex col_start = m_matrix.outerIndexPtr()[col];
                    StorageIndex next_col_start =
                        (col + 1 < m_matrix.cols()) ? m_matrix.outerIndexPtr()[col + 1] : m_matrix.nonZeros();
                    StorageIndex col_nnz = next_col_start - col_start;

                    for (StorageIndex i = 0; i < col_nnz; ++i) {
                        StorageIndex row = m_matrix.innerIndexPtr()[col_start + i];
                        if (row >= size) continue;

                        // Only include elements within the supernode
                        for (StorageIndex k = 0; k < snode_size; ++k) {
                            if (row == supernode[k]) {
                                dense_block(k, j) = m_matrix.valuePtr()[col_start + i];
                                break;
                            }
                        }
                    }
                }

                // Solve the dense block system
                Eigen::LDLT<Eigen::MatrixXd> ldlt(dense_block);
                snode_x = ldlt.solve(snode_x);

                // Copy back the solution
                for (StorageIndex j = 0; j < snode_size; ++j) { x[supernode[j]] = snode_x(j); }

                // Update remaining rows
                for (StorageIndex j = 0; j < snode_size; ++j) {
                    StorageIndex col = supernode[j];
                    if (col >= size) continue;

                    StorageIndex col_start = m_matrix.outerIndexPtr()[col];
                    StorageIndex next_col_start =
                        (col + 1 < m_matrix.cols()) ? m_matrix.outerIndexPtr()[col + 1] : m_matrix.nonZeros();
                    StorageIndex col_nnz = next_col_start - col_start;

                    for (StorageIndex i = 0; i < col_nnz; ++i) {
                        StorageIndex row = m_matrix.innerIndexPtr()[col_start + i];
                        if (row >= size) continue;
                        if (row > last_col) { x[row] -= m_matrix.valuePtr()[col_start + i] * x[col]; }
                    }
                }
            }
        }

        // Backward substitution
        for (StorageIndex snode = m_num_supernodes - 1; snode >= 0; --snode) {
            const auto &supernode = m_supernodes[snode];
            if (supernode.empty()) continue;

            const StorageIndex snode_size = supernode.size();
            const StorageIndex first_col  = supernode[0];
            const StorageIndex last_col   = supernode[snode_size - 1];

            if (snode_size == 1) {
                // Handle singleton supernode
                if (first_col >= size) continue;

                StorageIndex col_start = m_matrix.outerIndexPtr()[first_col];
                StorageIndex next_col_start =
                    (first_col + 1 < m_matrix.cols()) ? m_matrix.outerIndexPtr()[first_col + 1] : m_matrix.nonZeros();
                StorageIndex col_size = next_col_start - col_start;

                if (m_diag[first_col] != Scalar(0)) { x[first_col] /= m_diag[first_col]; }

                for (StorageIndex i = 0; i < col_size; ++i) {
                    StorageIndex row = m_matrix.innerIndexPtr()[col_start + i];
                    if (row >= size) continue;
                    if (row < first_col) { x[row] -= m_matrix.valuePtr()[col_start + i] * x[first_col]; }
                }
            } else {
                // Verify all indices are within bounds
                bool valid_supernode = true;
                for (StorageIndex j = 0; j < snode_size; ++j) {
                    if (supernode[j] >= size) {
                        valid_supernode = false;
                        break;
                    }
                }
                if (!valid_supernode) continue;

                // Handle supernode block similar to forward substitution
                Eigen::MatrixXd dense_block = Eigen::MatrixXd::Zero(snode_size, snode_size);
                Eigen::VectorXd snode_x(snode_size);

                for (StorageIndex j = 0; j < snode_size; ++j) {
                    StorageIndex col = supernode[j];
                    snode_x(j)       = x[col];

                    StorageIndex col_start = m_matrix.outerIndexPtr()[col];
                    StorageIndex next_col_start =
                        (col + 1 < m_matrix.cols()) ? m_matrix.outerIndexPtr()[col + 1] : m_matrix.nonZeros();
                    StorageIndex col_nnz = next_col_start - col_start;

                    for (StorageIndex i = 0; i < col_nnz; ++i) {
                        StorageIndex row = m_matrix.innerIndexPtr()[col_start + i];
                        if (row >= size) continue;

                        for (StorageIndex k = 0; k < snode_size; ++k) {
                            if (row == supernode[k]) {
                                dense_block(k, j) = m_matrix.valuePtr()[col_start + i];
                                break;
                            }
                        }
                    }
                }

                Eigen::LDLT<Eigen::MatrixXd> ldlt(dense_block);
                snode_x = ldlt.solve(snode_x);

                for (StorageIndex j = 0; j < snode_size; ++j) { x[supernode[j]] = snode_x(j); }

                for (StorageIndex j = 0; j < snode_size; ++j) {
                    StorageIndex col = supernode[j];
                    if (col >= size) continue;

                    StorageIndex col_start = m_matrix.outerIndexPtr()[col];
                    StorageIndex next_col_start =
                        (col + 1 < m_matrix.cols()) ? m_matrix.outerIndexPtr()[col + 1] : m_matrix.nonZeros();
                    StorageIndex col_nnz = next_col_start - col_start;

                    for (StorageIndex i = 0; i < col_nnz; ++i) {
                        StorageIndex row = m_matrix.innerIndexPtr()[col_start + i];
                        if (row >= size) continue;
                        if (row < first_col) { x[row] -= m_matrix.valuePtr()[col_start + i] * x[col]; }
                    }
                }
            }
        }
    }

    void setRegularization(Scalar epsilon) { m_epsilon = epsilon; }

    bool isFactorized = false;
    void factorize(const MatrixType &a) {
        bool DoLDLT       = true;
        bool NonHermitian = false;
        eigen_assert(a.rows() == a.cols());
        Index              size = a.cols();
        CholMatrixType     tmp(size, size);
        ConstCholMatrixPtr pmat;

        permute_symm_to_symm<UpLo, Upper, false>(a, tmp, m_P.indices().data());
        pmat = &tmp;

        // if (!isFactorized) {
        factorize_preordered<true, false>(*pmat);
        isFactorized = true;
    }

private:
    CholMatrixType                                    m_matrix;
    VectorType                                        m_diag;
    MatrixL                                           m_L;
    MatrixU                                           m_U;
    PermutationMatrix<Dynamic, Dynamic, StorageIndex> m_P, m_Pinv;
    std::vector<StorageIndex>                         m_nonZerosPerCol;
    std::vector<StorageIndex>                         m_parent;
    bool                                              m_isInitialized;
    bool                                              m_analysisIsOk;
    bool                                              m_factorizationIsOk;
    ComputationInfo                                   m_info;
    Scalar                                            m_epsilon; // Regularization parameter for numerical stability
    Scalar                                            m_shiftScale  = Scalar(1);
    Scalar                                            m_shiftOffset = Scalar(0);
};

} // namespace Eigen

#endif // EIGEN_CUSTOM_SIMPLICIAL_LDLT_H
