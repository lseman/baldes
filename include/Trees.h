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

// Define a small tolerance for double comparisons.
constexpr double EPSILON = 1e-3;

class TreeNode {
   public:
    // Lower and upper bounds for each dimension.
    // If the dimension is fixed, consider using std::array<double, N> for
    // better performance.
    std::vector<double> low;
    std::vector<double> high;
    int bucket_index;  // Bucket index for this node.
    TreeNode* left;
    TreeNode* right;
    TreeNode* parent;

    TreeNode(const std::vector<double>& low, const std::vector<double>& high,
             int bucket_index)
        : low(low),
          high(high),
          bucket_index(bucket_index),
          left(nullptr),
          right(nullptr),
          parent(nullptr) {}

    // Utility function to check if two doubles are nearly equal.
    inline bool nearlyEqual(double a, double b,
                            double epsilon = EPSILON) const {
        return std::fabs(a - b) < epsilon;
    }

    // Returns true if the point is inside the hyper-rectangle defined by [low,
    // high].
    inline bool contains(const std::vector<double>& point) const {
        for (size_t i = 0; i < low.size(); ++i) {
            // Allow a little wiggle room with EPSILON.
            if (point[i] < low[i] - EPSILON || point[i] > high[i] + EPSILON) {
                return false;
            }
        }
        return true;
    }

    // Compares the current node's interval with a point.
    // Here we use a lexicographical comparison on the high bounds.
    // Adjust this function if you need a different comparison semantics.
    inline bool is_less_than(const std::vector<double>& point) const {
        // Use lexicographical comparison but with an epsilon tolerance.
        for (size_t i = 0; i < high.size(); ++i) {
            if (!nearlyEqual(high[i], point[i])) {
                return high[i] < point[i];
            }
        }
        return false;  // They are nearly equal.
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
   private:
    TreeNode* root;

    // Rotate x up in the tree.
    inline void rotate(TreeNode* x) {
        TreeNode* p = x->parent;
        TreeNode* g = p->parent;
        bool is_left = (p->left == x);

        // Adjust parent's child pointer.
        TreeNode* child = is_left ? x->right : x->left;
        if (is_left) {
            p->left = child;
            x->right = p;
        } else {
            p->right = child;
            x->left = p;
        }
        if (child) {
            child->parent = p;
        }

        // Update parent pointers.
        x->parent = g;
        p->parent = x;

        // Update grandparent's child pointer.
        if (g) {
            if (g->left == p) {
                g->left = x;
            } else {
                g->right = x;
            }
        }
    }

    // Splay x to the root.
    inline void splay(TreeNode* x) {
        while (x->parent) {
            TreeNode* p = x->parent;
            TreeNode* g = p->parent;
            if (!g) {
                // Zig step.
                rotate(x);
            } else if ((g->left == p) == (p->left == x)) {
                // Zig-zig step.
                rotate(p);
                rotate(x);
            } else {
                // Zig-zag step.
                rotate(x);
                rotate(x);
            }
        }
        root = x;
    }

    // Recursively destroy the tree. For very deep trees, an iterative version
    // might be preferred.
    void destroyTree(TreeNode* node) {
        if (node) {
            destroyTree(node->left);
            destroyTree(node->right);
            delete node;
        }
    }

    // Recursively copy the tree.
    TreeNode* copyTree(TreeNode* node, TreeNode* parent = nullptr) {
        if (!node) return nullptr;
        TreeNode* newNode =
            new TreeNode(node->low, node->high, node->bucket_index);
        newNode->parent = parent;
        newNode->left = copyTree(node->left, newNode);
        newNode->right = copyTree(node->right, newNode);
        return newNode;
    }

   public:
    SplayTree() : root(nullptr) {}

    // Copy constructor.
    SplayTree(const SplayTree& other) : root(nullptr) {
        if (other.root) {
            root = copyTree(other.root);
        }
    }

    // Copy assignment operator.
    SplayTree& operator=(const SplayTree& other) {
        if (this != &other) {
            destroyTree(root);
            root = other.root ? copyTree(other.root) : nullptr;
        }
        return *this;
    }

    // Destructor.
    ~SplayTree() { destroyTree(root); }

    // Find a node whose interval contains the given point.
    TreeNode* find(const std::vector<double>& point) {
        TreeNode* curr = root;
        TreeNode* last = nullptr;
        while (curr) {
            last = curr;
            if (curr->contains(point)) {
                splay(curr);
                return curr;
            }
            // Move right if point is greater, else left.
            curr = curr->is_less_than(point) ? curr->right : curr->left;
        }
        if (last) splay(last);
        return nullptr;
    }

    TreeNode* find_without_splay(const std::vector<double>& point) {
        TreeNode* curr = root;
        TreeNode* last = nullptr;
        while (curr) {
            last = curr;
            if (curr->contains(point)) {
                return curr;
            }
            // Move right if point is greater, else left.
            curr = curr->is_less_than(point) ? curr->right : curr->left;
        }
        return nullptr;
    }

    // Query the bucket index for the point.
    int query(const std::vector<double>& point) {
        TreeNode* node = find(point);
        return node ? node->bucket_index : -1;
    }

    int queryStatic(const std::vector<double>& point) {
        TreeNode* node = find_without_splay(point);
        return node ? node->bucket_index : -1;
    }

    // Insert a new interval into the splay tree.
    void insert(const std::vector<double>& low, const std::vector<double>& high,
                int bucket_index) {
        if (!root) {
            root = new TreeNode(low, high, bucket_index);
            return;
        }

        TreeNode* curr = root;
        while (true) {
            if (low < curr->low) {
                if (!curr->left) {
                    curr->left = new TreeNode(low, high, bucket_index);
                    curr->left->parent = curr;
                    splay(curr->left);
                    break;
                }
                curr = curr->left;
            } else if (low > curr->low) {
                if (!curr->right) {
                    curr->right = new TreeNode(low, high, bucket_index);
                    curr->right->parent = curr;
                    splay(curr->right);
                    break;
                }
                curr = curr->right;
            } else {
                splay(curr);
                break;  // Duplicate interval, do nothing.
            }
        }
    }

    // Utility function to print the tree.
    void print() const { printTree(root, 0); }

   private:
    // Recursively print the tree in-order with indentation based on depth.
    void printTree(const TreeNode* node, int depth) const {
        if (!node) return;
        printTree(node->right, depth + 1);
        std::cout << std::string(depth * 2, ' ') << "Bucket "
                  << node->bucket_index << '\n';
        printTree(node->left, depth + 1);
    }
};
