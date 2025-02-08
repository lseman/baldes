#ifndef SUPERNODAL_H
#define SUPERNODAL_H

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <set>
#include <vector>

template <typename Scalar, typename StorageIndex>
class Supernodal {
   public:
    using MatrixType =
        Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    Supernodal(const MatrixType& matrix) : m_matrix(matrix) {
        if (m_matrix.rows() != m_matrix.cols()) {
            throw std::runtime_error("Matrix must be square");
        }
        computeEliminationTree();
        computeSupernodes();
    }

    void printEliminationTree() const {
        const StorageIndex n = m_matrix.rows();
        std::cout << "\nElimination Tree Visualization:" << std::endl;
        std::cout << "================================" << std::endl;

        std::cout << "Raw parent array:" << std::endl;
        for (StorageIndex i = 0; i < n; i++) {
            std::cout << "Node " << i << " -> Parent: " << m_parent[i]
                      << std::endl;
        }

        std::cout << "Detailed structure:" << std::endl;
        std::vector<std::vector<StorageIndex>> children(n);
        std::vector<bool> is_root(n, true);

        // Build children lists and identify roots
        for (StorageIndex i = 0; i < n; i++) {
            if (m_parent[i] != -1) {
                children[m_parent[i]].push_back(i);
                is_root[i] = false;
            }
        }

        // Print from each root
        for (StorageIndex i = 0; i < n; i++) {
            if (is_root[i]) {
                printNode(i, children, "", "");
            }
        }
    }

    void debugSupernodes() const {
        std::cout << "\nSupernode Analysis:" << std::endl;
        std::cout << "Number of supernodes: " << m_supernodes.size()
                  << std::endl;

        for (size_t i = 0; i < m_supernodes.size(); ++i) {
            const auto& supernode = m_supernodes[i];
            std::cout << "Supernode " << i << ": size = " << supernode.size()
                      << ", columns = ";
            for (StorageIndex col : supernode) {
                std::cout << col << " ";
            }
            std::cout << std::endl;

            // Print the pattern of the first column in the supernode
            StorageIndex first_col = supernode[0];
            std::cout << "Pattern for first column " << first_col << ": ";
            for (typename MatrixType::InnerIterator it(m_matrix, first_col); it;
                 ++it) {
                if (it.row() >= first_col) {
                    std::cout << it.row() << " ";
                }
            }
            std::cout << std::endl;
        }
    }
    VectorType solve(const VectorType& b) {
        if (b.size() != m_matrix.rows()) {
            throw std::runtime_error("Right-hand side size mismatch");
        }

        // Compute factorization if needed
        if (!m_is_factorized) {
            computeLDLTFactorization();
            m_is_factorized = true;
        }

        VectorType x = b;
        solveInternal(x);
        return x;
    }

   private:
    void printNode(StorageIndex node,
                   const std::vector<std::vector<StorageIndex>>& children,
                   const std::string& prefix,
                   const std::string& childPrefix) const {
        std::cout << prefix << node << std::endl;
        for (size_t i = 0; i < children[node].size(); ++i) {
            bool isLast = (i == children[node].size() - 1);
            std::string newPrefix = childPrefix + (isLast ? "└── " : "├── ");
            std::string newChildPrefix =
                childPrefix + (isLast ? "    " : "│   ");
            printNode(children[node][i], children, newPrefix, newChildPrefix);
        }
    }

    void computeEliminationTree() {
        const StorageIndex n = m_matrix.rows();
        m_parent.assign(n, -1);

        std::vector<StorageIndex> ancestor(n, -1);

        for (StorageIndex k = 0; k < n; k++) {
            ancestor[k] = k;

            for (typename MatrixType::InnerIterator it(m_matrix, k); it; ++it) {
                StorageIndex i = it.row();
                if (i < k) continue;  // Skip upper triangular part

                StorageIndex j = k;
                while (j != -1 && j < i) {
                    StorageIndex next = ancestor[j];
                    ancestor[j] = i;
                    if (next == j) {
                        m_parent[j] = i;
                        break;
                    }
                    j = next;
                }
            }
        }
    }

    void computeSupernodes() {
        const StorageIndex n = m_matrix.rows();
        m_supernodes.clear();
        m_col_to_supernode.resize(n);

        // Get the bandwidth of the matrix
        StorageIndex bandwidth = 0;
        for (StorageIndex col = 0; col < n; col++) {
            for (typename MatrixType::InnerIterator it(m_matrix, col); it;
                 ++it) {
                StorageIndex row_idx = static_cast<StorageIndex>(it.row());
                bandwidth = std::max(bandwidth, row_idx - col);
            }
        }

        // Group columns into supernodes based on the bandwidth
        StorageIndex current_start = 0;
        while (current_start < n) {
            StorageIndex supernode_size =
                std::min(bandwidth + 1, n - current_start);
            std::vector<StorageIndex> supernode;
            for (StorageIndex i = 0; i < supernode_size; i++) {
                supernode.push_back(current_start + i);
                m_col_to_supernode[current_start + i] = m_supernodes.size();
            }
            m_supernodes.push_back(supernode);
            current_start += supernode_size;
        }

        // Debug print supernodes
        debugSupernodes();
    }

    void factorSupernode(StorageIndex first_col, StorageIndex size,
                         const std::vector<StorageIndex>& supernode,
                         const std::vector<StorageIndex>& map_to_packed,
                         const std::vector<StorageIndex>& idx_to_pattern,
                         StorageIndex pattern_size, Eigen::MatrixXd& dense) {
        // Factor diagonal block
        std::cout << "\nFactoring diagonal block:\n";
        Eigen::MatrixXd diag_block = dense.topRows(size);
        std::cout << "Diagonal block:\n" << diag_block << "\n";

        Eigen::LDLT<Eigen::MatrixXd> ldlt;
        ldlt.compute(diag_block.template triangularView<Eigen::Lower>());
        Eigen::MatrixXd L = ldlt.matrixL();
        Eigen::VectorXd D = ldlt.vectorD();

        std::cout << "L factor:\n" << L << "\nD factor:\n" << D << "\n";

        // Store diagonal block factors
        for (StorageIndex j = 0; j < size; j++) {
            m_D[supernode[j]] = D(j);
            for (StorageIndex i = j; i < size; i++) {
                if (std::abs(L(i, j)) > 1e-14) {
                    m_L.insert(supernode[i], supernode[j]) = L(i, j);
                    std::cout << "Store L(" << supernode[i] << ","
                              << supernode[j] << ") = " << L(i, j) << "\n";
                }
            }
        }

        // Handle off-diagonal block if it exists
        if (pattern_size > size) {
            std::cout << "\nComputing off-diagonal block:\n";
            Eigen::MatrixXd L21 = dense.bottomRows(pattern_size - size);
            std::cout << "Initial L21:\n" << L21 << "\n";

            // Forward solve L21 * D = A21
            for (StorageIndex j = 0; j < size; j++) {
                // Scale by D inverse
                if (std::abs(D(j)) > 1e-14) {
                    L21.col(j) /= D(j);
                }
                std::cout << "After D scaling for col " << j << ":\n"
                          << L21 << "\n";

                // Update remaining columns
                for (StorageIndex k = j + 1; k < size; k++) {
                    L21.col(k) -= L21.col(j) * L(k, j);
                }
                std::cout << "After elimination for col " << j << ":\n"
                          << L21 << "\n";
            }

            // Store off-diagonal entries
            for (StorageIndex i = 0; i < pattern_size - size; i++) {
                StorageIndex row = idx_to_pattern[i + size];
                for (StorageIndex j = 0; j < size; j++) {
                    if (std::abs(L21(i, j)) > 1e-14) {
                        m_L.insert(row, supernode[j]) = L21(i, j);
                        std::cout << "Store L(" << row << "," << supernode[j]
                                  << ") = " << L21(i, j) << "\n";
                    }
                }
            }
        }
    }

    void computeLDLTFactorization() {
        const StorageIndex n = m_matrix.rows();
        m_L.resize(n, n);
        m_L.setZero();
        m_D.resize(n);

        std::cout << "\n=== Starting Factorization with Enhanced Debug ===\n";

        // Pre-compute column structures
        std::vector<std::vector<StorageIndex>> col_struct(n);
        for (StorageIndex j = 0; j < n; j++) {
            for (typename MatrixType::InnerIterator it(m_matrix, j); it; ++it) {
                if (it.row() >= j) col_struct[j].push_back(it.row());
            }
            std::cout << "Col " << j << " structure: ";
            for (auto idx : col_struct[j]) std::cout << idx << " ";
            std::cout << "\n";
        }

        // Process each supernode
        for (size_t snode_idx = 0; snode_idx < m_supernodes.size();
             snode_idx++) {
            const auto& supernode = m_supernodes[snode_idx];
            StorageIndex size = supernode.size();
            StorageIndex first_col = supernode[0];

            std::cout << "\n=== Processing Supernode " << snode_idx << " (cols "
                      << first_col << " to " << first_col + size - 1
                      << ") ===\n";

            // Build pattern including updates
            std::set<StorageIndex> pattern;
            for (StorageIndex j = 0; j < size; j++) {
                StorageIndex col = supernode[j];
                // Add original pattern
                for (auto row : col_struct[col]) pattern.insert(row);
                // Add update pattern
                for (StorageIndex k = 0; k < first_col; k++) {
                    for (typename MatrixType::InnerIterator it(m_L, k); it;
                         ++it) {
                        if (it.row() >= first_col) pattern.insert(it.row());
                    }
                }
            }

            // Create mappings
            std::vector<StorageIndex> map_to_packed(n, -1);
            std::vector<StorageIndex> idx_to_pattern;
            StorageIndex curr_idx = 0;
            for (StorageIndex row : pattern) {
                if (row >= first_col) {
                    map_to_packed[row] = curr_idx++;
                    idx_to_pattern.push_back(row);
                }
            }
            StorageIndex pattern_size = curr_idx;

            // Initialize dense matrix
            Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(pattern_size, size);

            // Fill from original matrix
            std::cout << "\nFilling dense matrix from original entries:\n";
            for (StorageIndex j = 0; j < size; j++) {
                StorageIndex col = supernode[j];
                for (typename MatrixType::InnerIterator it(m_matrix, col); it;
                     ++it) {
                    StorageIndex row = it.row();
                    if (row >= first_col) {
                        dense(map_to_packed[row], j) = it.value();
                        std::cout << "A(" << row << "," << col
                                  << ") = " << it.value() << "\n";
                    }
                }
            }
            std::cout << "\nInitial dense matrix:\n" << dense << "\n";

            // Process updates column by column
            for (StorageIndex k = 0; k < first_col; k++) {
                std::cout << "\nChecking updates from column " << k << ":\n";

                // Gather L column k
                Eigen::VectorXd L_col = Eigen::VectorXd::Zero(pattern_size);
                bool has_update = false;

                for (typename MatrixType::InnerIterator it(m_L, k); it; ++it) {
                    StorageIndex row = it.row();
                    if (row >= first_col && map_to_packed[row] != -1) {
                        L_col(map_to_packed[row]) = it.value();
                        has_update = true;
                    }
                }

                if (has_update) {
                    // Get current supernode's L entries for column k
                    Eigen::VectorXd L_curr = Eigen::VectorXd::Zero(size);
                    for (StorageIndex i = 0; i < size; i++) {
                        L_curr(i) = m_L.coeff(supernode[i], k);
                    }

                    std::cout << "L column " << k << ":\n"
                              << L_col.transpose() << "\n";
                    std::cout << "Current L entries:\n"
                              << L_curr.transpose() << "\n";

                    // Compute and apply update
                    if (std::abs(m_D[k]) > 1e-14) {
                        Eigen::MatrixXd update =
                            (L_col * L_curr.transpose()) * m_D[k];
                        std::cout << "Update matrix:\n" << update << "\n";
                        dense -= update;
                        std::cout << "Dense matrix after update:\n"
                                  << dense << "\n";
                    }
                }
            }

            // Factor supernode
            factorSupernode(first_col, size, supernode, map_to_packed,
                            idx_to_pattern, pattern_size, dense);

            // Debug output
            std::cout << "\nFactors after processing supernode " << snode_idx
                      << ":\n";
            for (StorageIndex i = 0; i <= first_col + size - 1; i++) {
                std::cout << "D[" << i << "] = " << m_D[i] << "\n";
                for (typename MatrixType::InnerIterator it(m_L, i); it; ++it) {
                    std::cout << "L(" << it.row() << "," << i
                              << ") = " << it.value() << "\n";
                }
            }
        }

        m_L.makeCompressed();
    }

    void solveInternal(VectorType& x) const {
        const StorageIndex n = x.size();
        std::cout << "\n=== Starting Solve Phase ===\n";

        // Make a copy of the right-hand side
        VectorType b = x;

        // 1. Forward substitution (Ly = b)
        std::cout << "\nForward substitution (Ly = b):\n";
        std::cout << "Initial x: " << x.transpose() << "\n";

        // Process one column at a time, not by supernodes
        for (StorageIndex j = 0; j < n; j++) {
            // Apply the current column's updates
            Scalar xj = x(j);
            for (typename MatrixType::InnerIterator it(m_L, j); it; ++it) {
                if (it.row() > j) {  // Only use strictly lower triangular part
                    x(it.row()) -= it.value() * xj;
                }
            }
            std::cout << "After column " << j << ": " << x.transpose() << "\n";
        }

        // 2. Diagonal solve (Dz = y)
        std::cout << "\nDiagonal solve (Dz = y):\n";
        std::cout << "Before: " << x.transpose() << "\n";

        // Process each diagonal entry
        for (StorageIndex i = 0; i < n; i++) {
            if (std::abs(m_D[i]) < 1e-14) {
                throw std::runtime_error("Near-zero diagonal encountered");
            }
            x(i) /= m_D[i];
        }
        std::cout << "After: " << x.transpose() << "\n";

        // 3. Backward substitution (L^T x = z)
        std::cout << "\nBackward substitution (L^T x = z):\n";

        // Process columns in reverse order
        for (StorageIndex j = n - 1; j >= 0; j--) {
            // Compute the dot product with already-computed values
            Scalar sum = 0;
            for (typename MatrixType::InnerIterator it(m_L, j); it; ++it) {
                if (it.row() > j) {
                    sum += it.value() * x(it.row());
                }
            }
            x(j) -= sum;

            std::cout << "After column " << j << ": " << x.transpose() << "\n";
            if (j == 0) break;  // Handle unsigned underflow
        }

        // Verify solution quality
        std::cout << "\n=== Solution Verification ===\n";

        // Compute residual vector r = b - Ax
        VectorType r = b;
        for (StorageIndex j = 0; j < n; j++) {
            for (typename MatrixType::InnerIterator it(m_matrix, j); it; ++it) {
                r(it.row()) -= it.value() * x(j);
                if (it.row() != j) {  // Add symmetric contribution
                    r(j) -= it.value() * x(it.row());
                }
            }
        }

        // Compute various norms
        Scalar r_norm = r.norm();
        Scalar x_norm = x.norm();
        Scalar b_norm = b.norm();
        Scalar A_norm = 0;  // Estimate matrix norm
        for (StorageIndex j = 0; j < n; j++) {
            Scalar col_sum = 0;
            for (typename MatrixType::InnerIterator it(m_matrix, j); it; ++it) {
                col_sum += std::abs(it.value());
                if (it.row() != j) {
                    col_sum += std::abs(it.value());  // Add symmetric part
                }
            }
            A_norm = std::max(A_norm, col_sum);
        }

        std::cout << "Matrix norm (1-norm estimate): " << A_norm << "\n";
        std::cout << "Residual norm |b - Ax|: " << r_norm << "\n";
        std::cout << "Solution norm |x|: " << x_norm << "\n";
        std::cout << "Right-hand side norm |b|: " << b_norm << "\n";
        std::cout << "Relative residual |b - Ax|/|b|: " << r_norm / b_norm
                  << "\n";
        std::cout << "Scaled residual |b - Ax|/(|A|*|x|): "
                  << r_norm / (A_norm * x_norm) << "\n";

        // Check component-wise errors
        Scalar max_error = 0;
        for (StorageIndex i = 0; i < n; i++) {
            Scalar error = std::abs(x(i) - 1.0);
            max_error = std::max(max_error, error);
            std::cout << "x[" << i << "] = " << x(i) << " (error: " << error
                      << ")\n";
        }
        std::cout << "Maximum component-wise error: " << max_error << "\n";

        // Check residual components
        std::cout << "\nResidual components:\n";
        for (StorageIndex i = 0; i < n; i++) {
            std::cout << "r[" << i << "] = " << r(i) << "\n";
        }
    }

    // Helper method to verify factorization quality
    void checkFactorizationAccuracy() const {
        const StorageIndex n = m_matrix.rows();

        std::cout << "\n=== Checking Factorization Quality ===\n";

        // Reconstruct A from L*D*L^T
        MatrixType reconstructed(n, n);
        reconstructed.setZero();

        // First compute L*D
        MatrixType LD = m_L;
        for (StorageIndex j = 0; j < n; j++) {
            LD.coeffRef(j, j) = 1.0;  // Add diagonal of L
            for (typename MatrixType::InnerIterator it(LD, j); it; ++it) {
                it.valueRef() *= m_D[j];
            }
        }

        // Then multiply by L^T
        for (StorageIndex j = 0; j < n; j++) {
            for (typename MatrixType::InnerIterator it1(LD, j); it1; ++it1) {
                StorageIndex row1 = it1.row();
                Scalar val1 = it1.value();

                for (typename MatrixType::InnerIterator it2(m_L, j); it2;
                     ++it2) {
                    StorageIndex row2 = it2.row();
                    reconstructed.coeffRef(row1, row2) += val1 * it2.value();
                }
            }
        }

        // Compare with original matrix
        Scalar max_diff = 0;
        for (StorageIndex j = 0; j < n; j++) {
            for (typename MatrixType::InnerIterator it(m_matrix, j); it; ++it) {
                Scalar orig = it.value();
                Scalar recon = reconstructed.coeff(it.row(), j);
                Scalar diff = std::abs(orig - recon);
                max_diff = std::max(max_diff, diff);
                if (diff > 1e-10) {
                    std::cout << "Large difference at (" << it.row() << "," << j
                              << "): original=" << orig
                              << ", reconstructed=" << recon
                              << ", diff=" << diff << "\n";
                }
            }
        }
        std::cout << "Maximum difference between original and reconstructed: "
                  << max_diff << "\n";
    }

    // Add verification of solve accuracy

   private:
    const MatrixType& m_matrix;
    std::vector<StorageIndex> m_parent;
    std::vector<std::vector<StorageIndex>> m_supernodes;
    std::vector<StorageIndex> m_col_to_supernode;
    MatrixType m_L;
    VectorType m_D;
    bool m_is_factorized = false;
};

#endif  // SUPERNODAL_H
