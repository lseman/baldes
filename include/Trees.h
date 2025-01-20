/**
 * @file Trees.h
 * @brief Header file containing the implementation of various tree structures
 * used for interval and multidimensional range queries.
 *
 * The classes provide functionalities for inserting intervals, searching for
 * overlapping intervals, and printing the tree structures for debugging
 * purposes.
 *
 * @note The implementation uses various helper functions for tree operations
 * such as insertion, searching, and splay operations.
 */
#pragma once
#include "Definitions.h"
#include "utils/NumericUtils.h"

/**
 * @class TreeNode
 * @brief Represents a node in a multi-dimensional tree structure.
 *
 */
class TreeNode {
   public:
    // Use fixed-size arrays if dimension is known at compile time
    // Otherwise, consider using std::array for small dimensions
    std::vector<double> low;
    std::vector<double> high;
    int bucket_index;
    TreeNode *left;
    TreeNode *right;
    TreeNode *parent;
    size_t dimension;  // Cache the dimension to avoid repeated size() calls

    TreeNode(const std::vector<double> &low_bounds,
             const std::vector<double> &high_bounds, int bucket_idx)
        : low(low_bounds),
          high(high_bounds),
          bucket_index(bucket_idx),
          left(nullptr),
          right(nullptr),
          parent(nullptr),
          dimension(low_bounds.size()) {
        // Reserve memory if vectors will be modified
        low.reserve(dimension);
        high.reserve(dimension);
    }

    // Pass point by const reference to avoid copying
    bool contains(const std::vector<double> &point) const {
        // Early dimension check
        if (point.size() != dimension) return false;

        // Manual loop unrolling for common dimensions (e.g., 2D/3D cases)
        if (dimension == 2) {
            return !(point[0] < low[0] || point[0] > high[0] ||
                     point[1] < low[1] || point[1] > high[1]);
        }
        if (dimension == 3) {
            return !(point[0] < low[0] || point[0] > high[0] ||
                     point[1] < low[1] || point[1] > high[1] ||
                     point[2] < low[2] || point[2] > high[2]);
        }

        // General case with SIMD-friendly pattern
        const size_t vec_size = dimension;
        for (size_t i = 0; i < vec_size; i += 4) {
            // Process 4 dimensions at once when possible
            size_t remaining = std::min(size_t(4), vec_size - i);
            for (size_t j = 0; j < remaining; ++j) {
                if (point[i + j] < low[i + j] || point[i + j] > high[i + j]) {
                    return false;
                }
            }
        }
        return true;
    }

    bool is_less_than(const std::vector<double> &point) const {
        // Early dimension check
        if (point.size() != dimension) return false;

        // Optimized 2D/3D cases
        if (dimension == 2) {
            if (high[0] < point[0]) return true;
            if (low[0] > point[0]) return false;
            if (high[1] < point[1]) return true;
            if (low[1] > point[1]) return false;
            return false;
        }
        if (dimension == 3) {
            if (high[0] < point[0]) return true;
            if (low[0] > point[0]) return false;
            if (high[1] < point[1]) return true;
            if (low[1] > point[1]) return false;
            if (high[2] < point[2]) return true;
            if (low[2] > point[2]) return false;
            return false;
        }

        // General case with early returns
        for (size_t i = 0; i < dimension; ++i) {
            if (high[i] < point[i]) return true;
            if (low[i] > point[i]) return false;
        }
        return false;
    }

    // Optional: Add move constructor and assignment operators for better
    // performance
    TreeNode(TreeNode &&other) noexcept
        : low(std::move(other.low)),
          high(std::move(other.high)),
          bucket_index(other.bucket_index),
          left(other.left),
          right(other.right),
          parent(other.parent),
          dimension(other.dimension) {
        other.left = other.right = other.parent = nullptr;
    }

    TreeNode &operator=(TreeNode &&other) noexcept {
        if (this != &other) {
            low = std::move(other.low);
            high = std::move(other.high);
            bucket_index = other.bucket_index;
            left = other.left;
            right = other.right;
            parent = other.parent;
            dimension = other.dimension;
            other.left = other.right = other.parent = nullptr;
        }
        return *this;
    }
};

/**
 * @class SplayTree
 * @brief A class representing a Splay Tree, which is a self-adjusting binary
 * search tree.
 *
 * The Splay Tree supports efficient insertion, deletion, and search operations
 * by performing splay operations that move accessed nodes closer to the root,
 * thereby improving access times for frequently accessed nodes.
 *
 * The tree nodes store intervals [low, high] and a bucket index associated with
 * each interval.
 *
 */
class SplayTree {
    TreeNode *root;

    // Single unified rotation function
    void rotate(TreeNode *x) {
        TreeNode *p = x->parent;
        if (!p) return;

        TreeNode *g = p->parent;
        bool isLeft = (p->left == x);
        TreeNode *child = isLeft ? x->right : x->left;

        // Update parent pointers
        x->parent = g;
        if (g) {
            if (g->left == p)
                g->left = x;
            else
                g->right = x;
        }

        // Perform rotation
        if (isLeft) {
            p->left = child;
            x->right = p;
        } else {
            p->right = child;
            x->left = p;
        }
        p->parent = x;
        if (child) child->parent = p;
    }

    // Optimized splay operation
    void splay(TreeNode *x) {
        while (x->parent) {
            TreeNode *p = x->parent;
            TreeNode *g = p->parent;

            if (!g) {
                rotate(x);  // Zig case
            } else {
                // Determine if we have zig-zig or zig-zag
                bool zigzig = (g->left == p) == (p->left == x);
                if (zigzig) {
                    rotate(p);  // Zig-zig: rotate parent first
                    rotate(x);
                } else {
                    rotate(x);  // Zig-zag: rotate x twice
                    rotate(x);
                }
            }
        }
        root = x;
    }

   public:
    SplayTree() : root(nullptr) {}

    TreeNode *find(const std::vector<double> &point) {
        TreeNode *curr = root;
        TreeNode *last = nullptr;  // Keep track of last accessed node

        while (curr) {
            last = curr;
            if (curr->contains(point)) {
                splay(curr);
                return curr;
            }
            curr = curr->is_less_than(point) ? curr->right : curr->left;
        }

        // Semi-splaying: bring the last accessed node to root
        if (last) splay(last);
        return nullptr;
    }

    int query(const std::vector<double> &point) {
        TreeNode *node = find(point);
        if (node != nullptr) return node->bucket_index;
        return -1;
    }

    // Optimized insert operation
    void insert(const std::vector<double> &low, const std::vector<double> &high,
                int bucket_index) {
        if (!root) {
            root = new TreeNode(low, high, bucket_index);
            return;
        }

        TreeNode *curr = root;
        TreeNode *parent = nullptr;
        bool isLeft = false;

        // Find insertion point without recursion
        while (curr) {
            parent = curr;
            if (low < curr->low) {
                isLeft = true;
                curr = curr->left;
            } else if (low > curr->low) {
                isLeft = false;
                curr = curr->right;
            } else {
                splay(curr);
                return;  // Duplicate interval
            }
        }

        // Create and insert new node
        TreeNode *newNode = new TreeNode(low, high, bucket_index);
        newNode->parent = parent;
        if (isLeft)
            parent->left = newNode;
        else
            parent->right = newNode;

        splay(newNode);
    }

    void inOrderPrint(TreeNode *node) {
        if (node == nullptr) return;
        inOrderPrint(node->left);
        std::cout << "Bucket " << node->bucket_index << ": [";
        for (size_t i = 0; i < node->low.size(); ++i) {
            std::cout << "[" << node->low[i] << ", " << node->high[i] << "]";
            if (i != node->low.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        inOrderPrint(node->right);
    }

    void print() { inOrderPrint(root); }
};
