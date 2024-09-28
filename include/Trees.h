/**
 * @file Trees.h
 * @brief Header file containing the implementation of various tree structures used for interval and multidimensional range queries.
 *
 * This file includes the following classes:
 * - BucketRange: Represents a range with lower and upper bounds.
 * - IntervalNode: Represents a node in an interval tree.
 * - IntervalTree: A tree structure for managing and querying intervals.
 * - BucketIntervalTree: A tree structure for managing and querying bucket intervals.
 * - TreeNode: Represents a node in a multi-dimensional tree structure.
 * - SplayTree: A self-adjusting binary search tree for efficient interval queries.
 *
 * The classes provide functionalities for inserting intervals, searching for overlapping intervals,
 * and printing the tree structures for debugging purposes.
 *
 * @note The implementation uses various helper functions for tree operations such as insertion, searching,
 *       and splay operations.
 */
#pragma once
#include "Definitions.h"

struct BucketRange {
    int lower_bound;
    int upper_bound;

    // Comparator for ordering ranges by their lower bound
    bool operator<(const BucketRange &other) const { return lower_bound < other.lower_bound; }
};

struct IntervalNode {
    BucketRange   from_range; // Interval representing from_bucket range
    BucketRange   to_range;   // Interval representing to_bucket range
    int           to_job;     // The to_job associated with this range
    int           max;        // Maximum upper bound of this subtree
    IntervalNode *left;
    IntervalNode *right;

    // Constructor
    IntervalNode(const BucketRange &f_range, const BucketRange &t_range, int to_job)
        : from_range(f_range), to_range(t_range), to_job(to_job), max(f_range.upper_bound), left(nullptr),
          right(nullptr) {}
};

class IntervalTree {
private:
    IntervalNode *root;

    // Helper function to check if two intervals overlap
    bool doOverlap(const BucketRange &i1, const BucketRange &i2) const {
        return i1.lower_bound <= i2.upper_bound && i2.lower_bound <= i1.upper_bound;
    }

    // Helper function to find if both from_range and to_range overlap with the node
    bool searchCombination(IntervalNode *node, const BucketRange &from_range, const BucketRange &to_range,
                           int to_job) const {
        if (node == nullptr) { return false; }

        if (doOverlap(node->from_range, from_range)) {
            if (doOverlap(node->to_range, to_range) && node->to_job == to_job) { return true; }
        }

        if (node->left != nullptr && node->left->max >= from_range.lower_bound) {
            if (searchCombination(node->left, from_range, to_range, to_job)) { return true; }
        }

        return searchCombination(node->right, from_range, to_range, to_job);
    }

    // Helper function to insert a new range into the tree
    IntervalNode *insert(IntervalNode *node, const BucketRange &from_range, const BucketRange &to_range, int to_job) {
        if (node == nullptr) {
            return new IntervalNode(from_range, to_range, to_job);
        }

        // Compare based on the lower bound of the from_range
        if (from_range < node->from_range) {
            node->left = insert(node->left, from_range, to_range, to_job);
        } else {
            node->right = insert(node->right, from_range, to_range, to_job);
        }

        // Update the max value at this node
        node->max = std::max(node->max, to_range.upper_bound);

        return node;
    }

    // Helper function to print intervals (for debugging)
    void printTree(IntervalNode *node) const {
        if (node == nullptr) return;

        printTree(node->left);
        std::cout << "From range: [" << node->from_range.lower_bound << ", " << node->from_range.upper_bound << "] "
                  << "To range: [" << node->to_range.lower_bound << ", " << node->to_range.upper_bound
                  << "] with to_job: " << node->to_job << " max = " << node->max << "\n";
        printTree(node->right);
    }

public:
    // Constructor
    IntervalTree() : root(nullptr) {}

    // Insert a new range
    void insert(const BucketRange &from_range, const BucketRange &to_range, int to_job) {
        root = insert(root, from_range, to_range, to_job);
    }

    // Search for any overlap with the given from_range, to_range, and to_job
    bool search(const BucketRange &from_range, const BucketRange &to_range, int to_job) const {
        return searchCombination(root, from_range, to_range, to_job);
    }

    // Print the intervals (for debugging)
    void print() const { printTree(root); }
};

class BucketIntervalTree {
private:
    struct FromBucketNode {
        BucketRange     from_range; // Interval for from_bucket
        IntervalTree   *to_tree;    // Interval tree for to_bucket ranges
        int             max;        // Max upper bound of from_bucket interval
        FromBucketNode *left;
        FromBucketNode *right;

        // Constructor
        FromBucketNode(const BucketRange &r) : from_range(r), max(r.upper_bound), left(nullptr), right(nullptr) {
            to_tree = new IntervalTree(); // Create a new interval tree for to_bucket ranges
        }

        ~FromBucketNode() { delete to_tree; }
    };

    FromBucketNode *root;

    // Insert a new from_bucket range with a to_bucket range
    FromBucketNode *insert(FromBucketNode *node, const BucketRange &from_range, const BucketRange &to_range,
                           int to_job) {
        if (node == nullptr) { node = new FromBucketNode(from_range); }

        // Insert the to_bucket range into the associated to_tree
        node->to_tree->insert(from_range, to_range, to_job);

        // Update the max value at this node
        node->max = std::max(node->max, from_range.upper_bound);

        return node;
    }

    // Search for any combination of from_range and to_range
    bool searchCombination(FromBucketNode *node, const BucketRange &from_range, const BucketRange &to_range,
                           int to_job) const {
        if (node == nullptr) return false;

        if (doOverlap(node->from_range, from_range)) {
            // Now check if the to_bucket range and to_job match
            if (node->to_tree->search(from_range, to_range, to_job)) { return true; }
        }

        if (node->left != nullptr && node->left->max >= from_range.lower_bound)
            return searchCombination(node->left, from_range, to_range, to_job);

        return searchCombination(node->right, from_range, to_range, to_job);
    }

    bool doOverlap(const BucketRange &i1, const BucketRange &i2) const {
        return i1.lower_bound <= i2.upper_bound && i2.lower_bound <= i1.upper_bound;
    }

    void printTree(FromBucketNode *node) const {
        if (node == nullptr) return;

        printTree(node->left);
        std::cout << "From range: [" << node->from_range.lower_bound << ", " << node->from_range.upper_bound << "]\n";
        node->to_tree->print(); // Print to_bucket ranges
        printTree(node->right);
    }

public:
    // Constructor
    BucketIntervalTree() : root(nullptr) {}

    // Insert a from_bucket and to_bucket combination
    void insert(const BucketRange &from_range, const BucketRange &to_range, int to_job) {
        root = insert(root, from_range, to_range, to_job);
    }

    // Search for a from_bucket and to_bucket combination
    bool search(const BucketRange &from_range, const BucketRange &to_range, int to_job) const {
        return searchCombination(root, from_range, to_range, to_job);
    }

    // Print the intervals (for debugging)
    void print() const { printTree(root); }
};

/**
 * @class TreeNode
 * @brief Represents a node in a multi-dimensional tree structure.
 *
 */
class TreeNode {
public:
    std::vector<int> low;          // Lower bounds for each dimension
    std::vector<int> high;         // Upper bounds for each dimension
    int              bucket_index; // Bucket index for this node
    TreeNode        *left;
    TreeNode        *right;
    TreeNode        *parent;

    TreeNode(const std::vector<int> &low, const std::vector<int> &high, int bucket_index)
        : low(low), high(high), bucket_index(bucket_index), left(nullptr), right(nullptr), parent(nullptr) {}

    bool contains(const std::vector<int> &point) const {
        for (size_t i = 0; i < low.size(); ++i) {
            if (point[i] < low[i] || point[i] > high[i]) { return false; }
        }
        return true;
    }

    bool is_less_than(const std::vector<int> &point) const {
        for (size_t i = 0; i < low.size(); ++i) {
            if (high[i] < point[i]) {
                return true;
            } else if (low[i] > point[i]) {
                return false;
            }
        }
        return false; // This case shouldn't be reached if comparing proper intervals
    }
};

/**
 * @class SplayTree
 * @brief A class representing a Splay Tree, which is a self-adjusting binary search tree.
 *
 * The Splay Tree supports efficient insertion, deletion, and search operations by performing
 * splay operations that move accessed nodes closer to the root, thereby improving access times
 * for frequently accessed nodes.
 *
 * The tree nodes store intervals [low, high] and a bucket index associated with each interval.
 *
 */
class SplayTree {
    TreeNode *root;

    void zig(TreeNode *x) {
        TreeNode *p = x->parent;
        TreeNode *B = (p->left == x) ? x->right : x->left;

        x->parent = p->parent;
        p->parent = x;

        if (p->left == x) {
            x->right = p;
            p->left  = B;
        } else {
            x->left  = p;
            p->right = B;
        }

        if (B != nullptr) B->parent = p;
    }

    void zig_zig(TreeNode *x) {
        TreeNode *p = x->parent;
        TreeNode *g = p->parent;
        if (p->left == x) {
            TreeNode *B = x->right;
            TreeNode *C = p->right;

            x->parent = g->parent;
            x->right  = p;

            p->parent = x;
            p->left   = B;
            p->right  = g;

            g->parent = p;
            g->left   = C;

            if (x->parent != nullptr) {
                if (x->parent->left == g)
                    x->parent->left = x;
                else
                    x->parent->right = x;
            }

            if (B != nullptr) B->parent = p;
            if (C != nullptr) C->parent = g;
        } else {
            TreeNode *B = p->left;
            TreeNode *C = x->left;

            x->parent = g->parent;
            x->left   = p;

            p->parent = x;
            p->left   = g;
            p->right  = C;

            g->parent = p;
            g->right  = B;

            if (x->parent != nullptr) {
                if (x->parent->left == g)
                    x->parent->left = x;
                else
                    x->parent->right = x;
            }

            if (B != nullptr) B->parent = g;
            if (C != nullptr) C->parent = p;
        }
    }

    void zig_zag(TreeNode *x) {
        TreeNode *p = x->parent;
        TreeNode *g = p->parent;

        if (p->right == x) {
            TreeNode *B = x->left;
            TreeNode *C = x->right;

            x->parent = g->parent;
            x->left   = p;
            x->right  = g;

            p->parent = x;
            p->right  = B;

            g->parent = x;
            g->left   = C;

            if (x->parent != nullptr) {
                if (x->parent->left == g)
                    x->parent->left = x;
                else
                    x->parent->right = x;
            }

            if (B != nullptr) B->parent = p;
            if (C != nullptr) C->parent = g;
        } else {
            TreeNode *B = x->left;
            TreeNode *C = x->right;

            x->parent = g->parent;
            x->left   = g;
            x->right  = p;

            p->parent = x;
            p->left   = C;

            g->parent = x;
            g->right  = B;

            if (x->parent != nullptr) {
                if (x->parent->left == g)
                    x->parent->left = x;
                else
                    x->parent->right = x;
            }

            if (B != nullptr) B->parent = g;
            if (C != nullptr) C->parent = p;
        }
    }

    void rotate(TreeNode *x) {
        TreeNode *p = x->parent;
        TreeNode *g = p->parent;

        if (p->left == x) { // Left child
            p->left = x->right;
            if (x->right) x->right->parent = p;
            x->right = p;
        } else { // Right child
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
            } else if ((g->left == p && p->left == x) || (g->right == p && p->right == x)) {
                // Zig-zig step (double rotation)
                rotate(p); // First rotate parent
                rotate(x); // Then rotate x
            } else {
                // Zig-zag step (rotating x twice in different directions)
                rotate(x); // Rotate x first
                rotate(x); // Then rotate x again
            }
        }
        root = x;
    }

public:
    SplayTree() : root(nullptr) {}

    TreeNode *find(const std::vector<int> &point) {
        TreeNode *curr = root;

        while (curr != nullptr) {
            if (curr->contains(point)) {
                splay(curr); // Splay only if we find the node
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

    int query(const std::vector<int> &point) {
        TreeNode *node = find(point);
        if (node != nullptr) return node->bucket_index;
        return -1;
    }

    // Insert a new multidimensional interval
    void insert(const std::vector<int> &low, const std::vector<int> &high, int bucket_index) {

        if (root == nullptr) {
            root = new TreeNode(low, high, bucket_index);
            return;
        }

        TreeNode *curr = root;
        while (curr != nullptr) {
            if (low < curr->low) {
                if (curr->left == nullptr) {
                    TreeNode *newNode = new TreeNode(low, high, bucket_index);
                    curr->left        = newNode;
                    newNode->parent   = curr;
                    splay(newNode);
                    return;
                } else {
                    curr = curr->left;
                }
            } else if (low > curr->low) {
                if (curr->right == nullptr) {
                    TreeNode *newNode = new TreeNode(low, high, bucket_index);
                    curr->right       = newNode;
                    newNode->parent   = curr;
                    splay(newNode);
                    return;
                } else {
                    curr = curr->right;
                }
            } else {
                splay(curr);
                return; // Duplicate interval
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

#define CONDITIONAL(D, FW_ACTION, BW_ACTION)         \
    if constexpr (D == Direction::Forward) {         \
        FW_ACTION;                                   \
    } else if constexpr (D == Direction::Backward) { \
        BW_ACTION;                                   \
    }
