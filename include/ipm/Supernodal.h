#pragma once
#include "Evaluator.h"
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <fstream>
#include <iostream>
#include <set>
#include <vector>

struct Supernode {
    int                                      start;            // Start index of this supernode
    int                                      size;             // Size of this supernode
    std::vector<int>                         structure;        // Sparsity structure of this supernode
    Supernode                               *parent = nullptr; // Parent supernode
    std::vector<Supernode *>                 children;         // Children supernodes
    Eigen::SparseMatrix<double>              L;                // Sparse lower triangular matrix L
    Eigen::VectorXd                          D;                // Diagonal matrix D (stored as a vector)
    Eigen::PermutationMatrix<Eigen::Dynamic> P;                // Permutation matrix

    Supernode(int start, int size, const std::vector<int> &struct_) : start(start), size(size), structure(struct_) {}

    void add_child(Supernode *child) {
        children.push_back(child);
        child->parent = this;
    }

    // Destructor to free child nodes
    ~Supernode() {
        for (auto *child : children) { delete child; }
    }

    // Bisect the matrix into supernodes
    Supernode *bisect(const Eigen::SparseMatrix<double> &matrix, int row_start = 0, int threshold = 64) {
        int n = matrix.rows();

        // Step 1: Reorder the matrix using custom AMD
        std::vector<int> amd_ordering = custom_amd(matrix);
        // convert amd ordering to Eigen permutation
        Eigen::PermutationMatrix<Eigen::Dynamic> perm(n);
        for (int i = 0; i < n; ++i) { perm.indices()[i] = amd_ordering[i]; }

        // Step 2: Permute the matrix based on AMD ordering
        // Eigen::PermutationMatrix<Eigen::Dynamic> perm(amd_ordering);
        Eigen::SparseMatrix<double> permuted_matrix = perm.transpose() * matrix * perm;

        // Step 3: Recursively bisect the permuted matrix
        if (n <= threshold) {
            // Base case: create a leaf supernode
            std::vector<int> structure;
            for (int k = 0; k < permuted_matrix.outerSize(); ++k) {
                if (permuted_matrix.col(k).nonZeros() > 0) { structure.push_back(k); }
            }
            return new Supernode(row_start, n, structure);
        }

        int mid = n / 2;

        Eigen::SparseMatrix<double> top_left     = extract_sparse_block(permuted_matrix, 0, 0, mid, mid);
        Eigen::SparseMatrix<double> bottom_right = extract_sparse_block(permuted_matrix, mid, mid, n - mid, n - mid);

        Supernode *left_supernode  = bisect(top_left, row_start, threshold);
        Supernode *right_supernode = bisect(bottom_right, row_start + mid, threshold);

        // Create the parent node, accounting for the entire structure
        std::vector<int> parent_structure;
        for (int k = 0; k < permuted_matrix.outerSize(); ++k) {
            if (permuted_matrix.col(k).nonZeros() > 0) { parent_structure.push_back(k); }
        }

        Supernode *parent_supernode = new Supernode(row_start, n, parent_structure);
        parent_supernode->add_child(left_supernode);
        parent_supernode->add_child(right_supernode);

        return parent_supernode;
    }

    // Helper function to extract a sparse sub-matrix block
    Eigen::SparseMatrix<double> extract_sparse_block(const Eigen::SparseMatrix<double> &matrix, int row_start,
                                                     int col_start, int rows, int cols) {
        Eigen::SparseMatrix<double> block(rows, cols);
        for (int k = 0; k < matrix.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it; ++it) {
                if (it.row() >= row_start && it.row() < row_start + rows && it.col() >= col_start &&
                    it.col() < col_start + cols) {
                    block.insert(it.row() - row_start, it.col() - col_start) = it.value();
                }
            }
        }
        block.makeCompressed();
        return block;
    }

    std::vector<int> custom_amd(const Eigen::SparseMatrix<double> &matrix) {
        int n = matrix.rows();

        // Create a graph representation where each row/column corresponds to a node
        std::vector<std::set<int>> adjacency_list(n);

        // Populate the adjacency list based on non-zero entries in the matrix
        for (int k = 0; k < matrix.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it; ++it) {
                if (it.row() != it.col()) {
                    adjacency_list[it.row()].insert(it.col());
                    adjacency_list[it.col()].insert(it.row());
                }
            }
        }

        // Vector to store the final ordering
        std::vector<int>              ordering;
        std::set<std::pair<int, int>> degree_queue; // (degree, node)

        // Initialize the degrees (number of non-zero connections) and priority queue
        std::vector<int> degrees(n);
        for (int i = 0; i < n; ++i) {
            degrees[i] = adjacency_list[i].size();
            degree_queue.insert({degrees[i], i});
        }

        while (!degree_queue.empty()) {
            // Choose the node with the lowest degree
            int node = degree_queue.begin()->second;
            degree_queue.erase(degree_queue.begin());

            // Add this node to the ordering
            ordering.push_back(node);

            // Update the graph by connecting the neighbors of the chosen node
            std::set<int> neighbors = adjacency_list[node];
            for (int neighbor : neighbors) {
                adjacency_list[neighbor].erase(node);
                for (int other_neighbor : neighbors) {
                    if (neighbor != other_neighbor) { adjacency_list[neighbor].insert(other_neighbor); }
                }
                // Update the degrees of the remaining neighbors
                degree_queue.erase({degrees[neighbor], neighbor});
                degrees[neighbor] = adjacency_list[neighbor].size();
                degree_queue.insert({degrees[neighbor], neighbor});
            }
        }

        return ordering;
    }

    std::vector<int> symbolic_factorization(Supernode *supernode) {
        if (supernode->children.empty()) {
            // Leaf node: return its own structure
            return supernode->structure;
        }

        // Collect structure from children
        std::vector<int> struct_union = supernode->structure;

        for (auto *child : supernode->children) {
            std::vector<int> child_structure = symbolic_factorization(child);
            struct_union.insert(struct_union.end(), child_structure.begin(), child_structure.end());
        }

        // Ensure unique structure (can use set for uniqueness)
        std::sort(struct_union.begin(), struct_union.end());
        struct_union.erase(std::unique(struct_union.begin(), struct_union.end()), struct_union.end());

        supernode->structure = struct_union;
        return struct_union;
    }

    void apply_regularization(Eigen::SparseMatrix<double> &matrix, double regularization, double threshold = 1e-6) {
        for (int k = 0; k < matrix.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it; ++it) {
                if (it.row() == it.col() && std::abs(it.value()) < threshold) {
                    matrix.coeffRef(it.row(), it.col()) += regularization;
                }
            }
        }
    }
    void numeric_factorization(Supernode *supernode, const Eigen::SparseMatrix<double> &matrix,
                               double regularizationFactor = 1e-5, double eps = 1e-8, double delta = 1e-3) {
        if (supernode->children.empty()) {
            Eigen::SparseMatrix<double> block =
                extract_sparse_block(matrix, supernode->start, supernode->start, supernode->size, supernode->size);

            if (block.nonZeros() > supernode->size * supernode->size * 0.5) {
                // Dense case
                Eigen::MatrixXd              dense_block = Eigen::MatrixXd(block); // Convert to dense
                Eigen::LDLT<Eigen::MatrixXd> ldlt;
                ldlt.compute(dense_block);

                if (ldlt.info() != Eigen::Success) {
                    std::cerr << "Dense LDLT decomposition failed. Applying regularization..." << std::endl;
                    apply_aggressive_regularization(dense_block, regularizationFactor);
                    ldlt.compute(dense_block);

                    if (ldlt.info() != Eigen::Success) {
                        std::cerr << "Dense LDLT decomposition still failed after regularization." << std::endl;
                        return;
                    }
                }

                // Dynamic regularization based on diagonal values
                Eigen::VectorXd D = ldlt.vectorD(); // Diagonal
                for (int i = 0; i < D.size(); ++i) {
                    if (D(i) < eps) {
                        D(i) = delta; // Apply regularization if too small
                    }
                }

                // Convert dense triangular matrix to sparse
                Eigen::MatrixXd dense_L = ldlt.matrixL();       // Extract dense L
                supernode->L            = dense_L.sparseView(); // Convert to sparse matrix
                supernode->D            = D;                    // Set regularized diagonal
                supernode->P.setIdentity(supernode->size);

            } else {
                // Sparse case
                Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
                ldlt.compute(block);

                if (ldlt.info() != Eigen::Success) {
                    std::cerr << "Sparse LDLT decomposition failed. Applying regularization..." << std::endl;
                    apply_aggressive_regularization(block, regularizationFactor);
                    ldlt.compute(block);

                    if (ldlt.info() != Eigen::Success) {
                        std::cerr << "Sparse LDLT decomposition still failed after regularization." << std::endl;
                        return;
                    }
                }

                // Apply dynamic regularization to diagonal values
                Eigen::VectorXd D = ldlt.vectorD();
                for (int i = 0; i < D.size(); ++i) {
                    if (std::abs(D(i)) < eps) {
                        D(i) = (D(i) < 0) ? -delta : delta; // Apply delta with correct sign
                    }
                }

                supernode->L = ldlt.matrixL();
                supernode->D = D; // Set regularized diagonal
                supernode->P = Eigen::PermutationMatrix<Eigen::Dynamic>(ldlt.permutationP());
            }

            return;
        }

        // Recursively process children supernodes
        for (auto *child : supernode->children) {
            numeric_factorization(child, matrix, regularizationFactor, eps, delta);
        }

        // Process non-leaf supernodes
        Eigen::SparseMatrix<double> block =
            extract_sparse_block(matrix, supernode->start, supernode->start, supernode->size, supernode->size);
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
        ldlt.compute(block);

        if (ldlt.info() != Eigen::Success) {
            std::cerr << "LDLT decomposition failed. Applying regularization..." << std::endl;
            apply_regularization(block, regularizationFactor);
            ldlt.compute(block);

            if (ldlt.info() != Eigen::Success) {
                std::cerr << "LDLT decomposition still failed after regularization." << std::endl;
                return;
            }
        }

        // Apply dynamic regularization
        Eigen::VectorXd D = ldlt.vectorD();
        for (int i = 0; i < D.size(); ++i) {
            if (std::abs(D(i)) < eps) { D(i) = (D(i) < 0) ? -delta : delta; }
        }

        supernode->L = ldlt.matrixL();
        supernode->D = D;
        supernode->P = Eigen::PermutationMatrix<Eigen::Dynamic>(ldlt.permutationP());
    }
    void apply_aggressive_regularization(Eigen::MatrixXd &matrix, double &regularization, double threshold = 1e-6) {
        const double max_regularization       = 1e-3;
        const double regularization_increment = 10;

        while (regularization < max_regularization) {
            bool regularized = false;
            for (int i = 0; i < matrix.rows(); ++i) {
                if (std::abs(matrix(i, i)) < threshold) {
                    matrix(i, i) += regularization;
                    regularized = true;
                }
            }
            if (regularized) break;
            regularization *= regularization_increment;
        }
    }

    void apply_aggressive_regularization(Eigen::SparseMatrix<double> &matrix, double &regularization,
                                         double threshold = 1e-6) {
        const double max_regularization = 1e-3; // Set a cap on the regularization factor
        while (regularization < max_regularization) {
            bool regularized = false;
            for (int k = 0; k < matrix.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it; ++it) {
                    if (it.row() == it.col() && std::abs(it.value()) < threshold) {
                        matrix.coeffRef(it.row(), it.col()) += regularization;
                        regularized = true;
                    }
                }
            }
            if (regularized) break;
            regularization *= 10; // Increase regularization by an order of magnitude
        }
    }

    void check_nan_and_regularize(Eigen::SparseMatrix<double>                        &block,
                                  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> &ldlt,
                                  double &regularizationFactor, int max_retries = 5) {
        int retries = 0;
        while (retries < max_retries) {
            bool has_nan = false;
            for (int k = 0; k < block.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(block, k); it; ++it) {
                    if (std::isnan(it.value())) {
                        has_nan = true;
                        break;
                    }
                }
                if (has_nan) break;
            }

            if (!has_nan) {
                // No NaN found, return normally
                return;
            }

            // Apply aggressive regularization and retry
            std::cerr << "NaN detected. Applying aggressive regularization. Retry #" << retries + 1 << std::endl;
            apply_aggressive_regularization(block, regularizationFactor);
            ldlt.compute(block);

            retries++;
        }

        if (retries >= max_retries) {
            std::cerr << "Max retries reached. NaN values still present, stopping factorization." << std::endl;
        }
    }

    Eigen::VectorXd forward_solve_ldlt(Supernode *supernode, const Eigen::VectorXd &b) {
        Eigen::VectorXd b_slice              = b.segment(supernode->start, supernode->size); // Slice the vector
        Eigen::VectorXd P_T_b                = supernode->P * b_slice;                       // Apply permutation
        const double    regularizationFactor = double(1e-5); // Small regularization term

        // Eigen::VectorXd y = supernode->L.triangularView<Eigen::Lower>().solve(P_T_b); // Forward solve L y = P^T b
        solveTriangular<decltype(supernode->L), decltype(P_T_b), Eigen::Lower>(supernode->L, P_T_b,
                                                                               regularizationFactor);

        return P_T_b;
    }

    Eigen::VectorXd backward_solve_ldlt(Supernode *supernode, const Eigen::VectorXd &y) {
        Eigen::VectorXd z = y.array() / supernode->D.array(); // Solve D z = y (element-wise division)
        const double    regularizationFactor = double(1e-5);  // Small regularization term

        // Eigen::VectorXd x =
        //     supernode->L.transpose().triangularView<Eigen::Upper>().solve(z); // Backward solve L^T x = z
        solveTriangular<decltype(supernode->L.transpose()), decltype(z), Eigen::Upper>(supernode->L.transpose(), z,
                                                                                       regularizationFactor);

        return z;
    }

    Eigen::VectorXd solve_supernode_ldlt(Supernode *supernode, const Eigen::VectorXd &b) {
        Eigen::VectorXd y = forward_solve_ldlt(supernode, b);
        Eigen::VectorXd x = backward_solve_ldlt(supernode, y);
        return x;
    }

    Eigen::VectorXd supernodal_factorization_ldlt(const Eigen::SparseMatrix<double> &matrix, const Eigen::VectorXd &b,
                                                  int threshold = 64, double regularizationFactor = 1e-5) {
        // Step 1: Construct supernodes
        Supernode *supernode = bisect(matrix, 0, threshold);

        // Step 2: Perform symbolic factorization
        symbolic_factorization(supernode);

        // Step 3: Perform numeric factorization (LDLT) with regularization
        numeric_factorization(supernode, matrix, regularizationFactor);

        // Step 4: Solve using forward and backward solves for LDLT
        Eigen::VectorXd x = solve_supernode_ldlt(supernode, b);

        // Clean up memory
        delete supernode;

        return x;
    }

    // define compute, analyzePattern, factorize and solve methods
    void compute(const Eigen::SparseMatrix<double> &matrix) {
        // Perform supernodal factorization
        supernodal_factorization_ldlt(matrix, Eigen::VectorXd::Zero(matrix.rows()));
    }

    void factorize(const Eigen::SparseMatrix<double> &matrix, double regularizationFactor = 1e-5) {
        // Perform symbolic factorization
        numeric_factorization(supernode, matrix);
    }

    // default constructor
    Supernode() = default;

    Supernode *supernode = nullptr;
    void       analyzePattern(const Eigen::SparseMatrix<double> &matrix, int leafSize = 64) {
        // Perform symbolic factorization
        supernode = bisect(matrix, 0, leafSize);
        symbolic_factorization(supernode);
    }

    Eigen::VectorXd solve(const Eigen::VectorXd &b) {
        // Solve the system using forward and backward substitution
        Eigen::VectorXd x = solve_supernode_ldlt(supernode, b);
        return x;
    }

    bool saved = false;
    // Function to save a sparse matrix and RHS vector to a CSV file
    void save_matrix_and_rhs_to_csv(const Eigen::SparseMatrix<double> &matrix, const Eigen::VectorXd &rhs,
                                    const std::string &filename) {

        if (saved) return;
        std::ofstream file(filename);

        // Write the sparse matrix in COO format
        file << "row,col,value\n"; // Header
        for (int k = 0; k < matrix.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it; ++it) {
                file << it.row() << "," << it.col() << "," << it.value() << "\n";
            }
        }

        // Write the RHS vector
        file << "\nRHS\n"; // Header for RHS
        for (int i = 0; i < rhs.size(); ++i) { file << rhs[i] << "\n"; }

        file.close();
        saved = true;
    }
    // throw fatal error
    
};