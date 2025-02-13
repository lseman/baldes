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

    bool contains(const std::vector<double>& point) const {
        for (size_t i = 0; i < low.size(); ++i) {
            if (point[i] < low[i] || point[i] > high[i]) {
                return false;
            }
        }
        return true;
    }

    bool is_less_than(const std::vector<double>& point) const {
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
   private:
    TreeNode* root;

    void rotate(TreeNode* x) {
        TreeNode* p = x->parent;
        TreeNode* g = p->parent;
        bool is_left = (p->left == x);

        // Update the parent's child pointer
        TreeNode* child = is_left ? x->right : x->left;
        if (is_left) {
            p->left = child;
            x->right = p;
        } else {
            p->right = child;
            x->left = p;
        }
        if (child) child->parent = p;

        // Update parent pointers
        x->parent = g;
        p->parent = x;

        // Update grandparent's child pointer
        if (g) {
            (g->left == p ? g->left : g->right) = x;
        }
    }

    void splay(TreeNode* x) {
        while (x->parent) {
            TreeNode* p = x->parent;
            TreeNode* g = p->parent;

            if (!g) {
                rotate(x);  // Zig
            } else if ((g->left == p) == (p->left == x)) {
                rotate(p);  // Zig-zig
                rotate(x);
            } else {
                rotate(x);  // Zig-zag
                rotate(x);
            }
        }
        root = x;
    }

    void destroyTree(TreeNode* node) {
        if (node) {
            destroyTree(node->left);
            destroyTree(node->right);
            delete node;
        }
    }

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

    // Copy constructor
    SplayTree(const SplayTree& other) : root(nullptr) {
        if (other.root) {
            root = copyTree(other.root);
        }
    }

    // Copy assignment
    SplayTree& operator=(const SplayTree& other) {
        if (this != &other) {
            destroyTree(root);
            root = other.root ? copyTree(other.root) : nullptr;
        }
        return *this;
    }

    // Destructor
    ~SplayTree() { destroyTree(root); }

    TreeNode* find(const std::vector<double>& point) {
        TreeNode* curr = root;
        TreeNode* last = nullptr;

        while (curr) {
            last = curr;
            if (curr->contains(point)) {
                splay(curr);
                return curr;
            }
            curr = curr->is_less_than(point) ? curr->right : curr->left;
        }

        if (last) splay(last);
        return nullptr;
    }

    int query(const std::vector<double>& point) {
        TreeNode* node = find(point);
        return node ? node->bucket_index : -1;
    }

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
                break;  // Duplicate interval
            }
        }
    }

    void print() const { printTree(root, 0); }

   private:
    void printTree(const TreeNode* node, int depth) const {
        if (!node) return;
        printTree(node->right, depth + 1);
        std::cout << std::string(depth * 2, ' ') << "Bucket "
                  << node->bucket_index << '\n';
        printTree(node->left, depth + 1);
    }
};
