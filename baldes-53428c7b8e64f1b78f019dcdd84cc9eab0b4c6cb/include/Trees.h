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
    std::vector<double> low;   // Lower bounds for each dimension
    std::vector<double> high;  // Upper bounds for each dimension
    int bucket_index;          // Bucket index for this node
    TreeNode *left;
    TreeNode *right;
    TreeNode *parent;

    TreeNode(const std::vector<double> &low, const std::vector<double> &high,
             int bucket_index)
        : low(low),
          high(high),
          bucket_index(bucket_index),
          left(nullptr),
          right(nullptr),
          parent(nullptr) {}

    bool contains(const std::vector<double> &point) const {
        for (size_t i = 0; i < low.size(); ++i) {
            if (point[i] < low[i] || point[i] > high[i]) {
                return false;
            }
        }
        return true;
    }

    bool is_less_than(const std::vector<double> &point) const {
        for (size_t i = 0; i < low.size(); ++i) {
            if (high[i] < point[i]) {
                // if (numericutils::less_than(high[i], point[i])) {
                return true;
            } else if (low[i] > point[i]) {
                //} else if (numericutils::greater_than(low[i], point[i])) {
                return false;
            }
        }
        return false;  // This case shouldn't be reached if comparing proper
                       // intervals
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

    void rotate(TreeNode *x) {
        TreeNode *p = x->parent;
        TreeNode *g = p->parent;

        if (p->left == x) {  // Left child
            p->left = x->right;
            if (x->right) x->right->parent = p;
            x->right = p;
        } else {  // Right child
            p->right = x->left;
            if (x->left) x->left->parent = p;
            x->left = p;
        }

        x->parent = g;
        p->parent = x;

        if (g) {
            if (g->left == p)
                g->left = x;
            else
                g->right = x;
        }
    }

    void splay(TreeNode *x) {
        while (x->parent != nullptr) {
            TreeNode *p = x->parent;
            TreeNode *g = p->parent;

            if (g == nullptr) {
                // Zig step (single rotation)
                rotate(x);
            } else if ((g->left == p && p->left == x) ||
                       (g->right == p && p->right == x)) {
                // Zig-zig step (double rotation)
                rotate(p);  // First rotate parent
                rotate(x);  // Then rotate x
            } else {
                // Zig-zag step (rotating x twice in different directions)
                rotate(x);  // Rotate x first
                rotate(x);  // Then rotate x again
            }
        }
        root = x;
    }

   public:
    SplayTree() : root(nullptr) {}

    TreeNode *find(const std::vector<double> &point) {
        TreeNode *curr = root;

        while (curr != nullptr) {
            if (curr->contains(point)) {
                splay(curr);  // Splay only if we find the node
                return curr;
            } else if (curr->is_less_than(point)) {
                curr = curr->right;
            } else {
                curr = curr->left;
            }
        }

        // If not found, no splaying needed for the closest node
        return nullptr;
    }

    int query(const std::vector<double> &point) {
        TreeNode *node = find(point);
        if (node != nullptr) return node->bucket_index;
        return -1;
    }

    // Insert a new multidimensional interval
    void insert(const std::vector<double> &low, const std::vector<double> &high,
                int bucket_index) {
        if (root == nullptr) {
            root = new TreeNode(low, high, bucket_index);
            return;
        }

        TreeNode *curr = root;
        while (curr != nullptr) {
            if (low < curr->low) {
                if (curr->left == nullptr) {
                    TreeNode *newNode = new TreeNode(low, high, bucket_index);
                    curr->left = newNode;
                    newNode->parent = curr;
                    splay(newNode);
                    return;
                } else {
                    curr = curr->left;
                }
            } else if (low > curr->low) {
                if (curr->right == nullptr) {
                    TreeNode *newNode = new TreeNode(low, high, bucket_index);
                    curr->right = newNode;
                    newNode->parent = curr;
                    splay(newNode);
                    return;
                } else {
                    curr = curr->right;
                }
            } else {
                splay(curr);
                return;  // Duplicate interval
            }
        }
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
