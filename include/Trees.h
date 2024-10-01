/**
 * @file Trees.h
 * @brief Header file containing the implementation of various tree structures used for interval and multidimensional
 * range queries.
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
    std::vector<int> lower_bound;
    std::vector<int> upper_bound;

    // Lexicographical comparison for ordering ranges by their lower bound
    bool operator<(const BucketRange &other) const {
        // Compare lower bounds first
        for (size_t i = 0; i < lower_bound.size(); ++i) {
            if (lower_bound[i] < other.lower_bound[i]) {
                return true; // This `BucketRange` is smaller based on lower_bound
            }
            if (lower_bound[i] > other.lower_bound[i]) {
                return false; // This `BucketRange` is larger based on lower_bound
            }
        }

        // If lower bounds are the same, compare upper bounds
        for (size_t i = 0; i < upper_bound.size(); ++i) {
            if (upper_bound[i] < other.upper_bound[i]) {
                return true; // This `BucketRange` is smaller based on upper_bound
            }
            if (upper_bound[i] > other.upper_bound[i]) {
                return false; // This `BucketRange` is larger based on upper_bound
            }
        }

        // If both lower and upper bounds are the same, they are equal, so return false
        return false;
    }

    bool operator>(const BucketRange &other) const {
        // Compare lower bounds first
        for (size_t i = 0; i < lower_bound.size(); ++i) {
            if (lower_bound[i] > other.lower_bound[i]) {
                return true; // This `BucketRange` is larger based on lower_bound
            }
            if (lower_bound[i] < other.lower_bound[i]) {
                return false; // This `BucketRange` is smaller based on lower_bound
            }
        }

        // If lower bounds are the same, compare upper bounds
        for (size_t i = 0; i < upper_bound.size(); ++i) {
            if (upper_bound[i] > other.upper_bound[i]) {
                return true; // This `BucketRange` is larger based on upper_bound
            }
            if (upper_bound[i] < other.upper_bound[i]) {
                return false; // This `BucketRange` is smaller based on upper_bound
            }
        }

        // If both lower and upper bounds are the same, they are equal, so return false
        return false;
    }

    // Optional: Define operator== and operator!= for completeness
    bool operator==(const BucketRange &other) const {
        return lower_bound == other.lower_bound && upper_bound == other.upper_bound;
    }

    bool operator!=(const BucketRange &other) const { return !(*this == other); }

    bool contained_in(const BucketRange &other) const {
        for (size_t i = 0; i < lower_bound.size(); ++i) {
            if (!(lower_bound[i] >= other.lower_bound[i] && upper_bound[i] <= other.upper_bound[i])) { return false; }
        }
        return true;
    }
};

// TODO: better check the balancing and the comparisons
struct IntervalNode {
    BucketRange      from_range;
    BucketRange      to_range;
    int              to_job;
    std::vector<int> max;
    IntervalNode    *left;
    IntervalNode    *right;
    mutable bool     merge_pending;
    int              height; // New: Height of the node

    IntervalNode(const BucketRange &f_range, const BucketRange &t_range, int to_job)
        : from_range(f_range), to_range(t_range), to_job(to_job), max(f_range.upper_bound), left(nullptr),
          right(nullptr), merge_pending(false), height(1) {} // Height is initialized to 1
};

class IntervalTree {
private:
    IntervalNode *root;

    int height(IntervalNode *node) { return node ? node->height : 0; }

    int getBalance(IntervalNode *node) { return node ? height(node->left) - height(node->right) : 0; }

    IntervalNode *rightRotate(IntervalNode *y) {
        if (y == nullptr || y->left == nullptr) {
            return y; // Return the node itself if rotation can't be performed
        }

        IntervalNode *x  = y->left;  // x becomes the new root of the subtree
        IntervalNode *T2 = x->right; // T2 will be the right child of x after rotation

        // Perform rotation
        x->right = y;
        y->left  = T2;

        // Update heights
        y->height = std::max(height(y->left), height(y->right)) + 1; // y's height depends on its new children
        x->height = std::max(height(x->left), height(x->right)) + 1; // x's height depends on its new children

        // Return new root
        return x;
    }

    IntervalNode *leftRotate(IntervalNode *x) {
        if (x == nullptr || x->right == nullptr) {
            return x; // Return the node itself if rotation can't be performed
        }

        IntervalNode *y  = x->right; // y becomes the new root of the subtree
        IntervalNode *T2 = y->left;  // T2 will be the left child of x after rotation

        // Perform rotation
        y->left  = x;
        x->right = T2;

        // Update heights
        x->height = std::max(height(x->left), height(x->right)) + 1; // x's height depends on its new children
        y->height = std::max(height(y->left), height(y->right)) + 1; // y's height depends on its new children

        // Return new root
        return y;
    }

    // Helper function to check if two intervals overlap
    bool doOverlap(const BucketRange &i1, const BucketRange &i2) const {
        for (size_t i = 0; i < i1.lower_bound.size(); ++i) {
            if (!(i1.lower_bound[i] <= i2.upper_bound[i] && i2.lower_bound[i] <= i1.upper_bound[i])) { return false; }
        }
        return true;
    }

    // Helper function to merge two overlapping ranges
    BucketRange mergeRanges(const BucketRange &i1, const BucketRange &i2) const {
        BucketRange merged;
        for (size_t i = 0; i < i1.lower_bound.size(); ++i) {
            merged.lower_bound.push_back(std::min(i1.lower_bound[i], i2.lower_bound[i]));
            merged.upper_bound.push_back(std::max(i1.upper_bound[i], i2.upper_bound[i]));
        }
        return merged;
    }

    // Helper function to insert a new range into the tree, with merging of overlapping to_ranges
    IntervalNode *insert(IntervalNode *node, const BucketRange &from_range, const BucketRange &to_range, int to_job) {
        // Insert as in the original, unbalanced version
        if (node == nullptr) { return new IntervalNode(from_range, to_range, to_job); }

        // Compare based on `from_range.lower_bound` first
        if (from_range < node->from_range) {
            node->left = insert(node->left, from_range, to_range, to_job);
        } else if (from_range > node->from_range) {
            node->right = insert(node->right, from_range, to_range, to_job);
        } else {
            // If `from_range` is the same, check `to_range` and `to_job`
            bool toRangeEqual = true;
            for (size_t i = 0; i < to_range.lower_bound.size(); ++i) {
                if (to_range.lower_bound[i] != node->to_range.lower_bound[i] ||
                    to_range.upper_bound[i] != node->to_range.upper_bound[i]) {
                    toRangeEqual = false;
                    break;
                }
            }

            // If `to_range` and `to_job` match, mark for merging
            if (toRangeEqual && node->to_job == to_job) {
                node->merge_pending = true; // Mark for merging
            } else {
                // Insert as a new node if `to_range` or `to_job` differs
                node->right = insert(node->right, from_range, to_range, to_job);
            }
        }

        // Update max values (assuming the upper_bound size matches)
        for (size_t i = 0; i < node->max.size(); ++i) {
            node->max[i] = std::max({node->max[i], from_range.upper_bound[i], node->to_range.upper_bound[i]});
        }

        // Update height of this node
        node->height = std::max(height(node->left), height(node->right)) + 1;
        /*
        // Balance the node if necessary (AVL rotations)
        int balance = getBalance(node);

        // Left Left Case
        if (balance > 1 && from_range < node->left->from_range) { return rightRotate(node); }

        // Right Right Case
        if (balance < -1 && from_range > node->right->from_range) { return leftRotate(node); }

        // Left Right Case
        if (balance > 1 && from_range > node->left->from_range) {
            node->left = leftRotate(node->left);
            return rightRotate(node);
        }

        // Right Left Case
        if (balance < -1 && from_range < node->right->from_range) {
            node->right = rightRotate(node->right);
            return leftRotate(node);
        }
*/
        return node;
    }

    void applyPendingMerges(IntervalNode *node) const {
        if (node != nullptr && node->merge_pending) {
            if (node->left && doOverlap(node->to_range, node->left->to_range)) {
                node->to_range = mergeRanges(node->to_range, node->left->to_range);
            }
            if (node->right && doOverlap(node->to_range, node->right->to_range)) {
                node->to_range = mergeRanges(node->to_range, node->right->to_range);
            }
            node->merge_pending = false;
        }
    }

    bool searchCombination(IntervalNode *node, const BucketRange &from_range, const BucketRange &to_range,
                           int to_job) const {
        if (node == nullptr) return false;

        // Apply any pending merges to ensure node's `to_range` is up-to-date
        applyPendingMerges(node);

        // Check if `from_range` is contained within `node->from_range`
        bool fromRangeContained = from_range.contained_in(node->from_range);

        // Check if `to_range` is contained within `node->to_range` and the job matches
        bool toRangeContained = to_range.contained_in(node->to_range);

        // If both ranges are contained and the job matches, return true
        if (fromRangeContained && toRangeContained && node->to_job == to_job) { return true; }

        // Use updated comparison for traversing the tree
        if (from_range < node->from_range) {
            return searchCombination(node->left, from_range, to_range, to_job);
        } else {
            return searchCombination(node->right, from_range, to_range, to_job);
        }
    }

public:
    IntervalTree() : root(nullptr) {}

    void insert(const BucketRange &from_range, const BucketRange &to_range, int to_job) {
        root = insert(root, from_range, to_range, to_job);
    }

    bool search(const BucketRange &from_range, const BucketRange &to_range, int to_job) const {
        return searchCombination(root, from_range, to_range, to_job);
    }

    void print() const { printTree(root); }

    void printTree(IntervalNode *node) const {
        if (node == nullptr) return;

        printTree(node->left);
        std::cout << "From range: [";
        for (size_t i = 0; i < node->from_range.lower_bound.size(); ++i) {
            std::cout << node->from_range.lower_bound[i] << ", " << node->from_range.upper_bound[i];
            if (i < node->from_range.lower_bound.size() - 1) std::cout << "; ";
        }
        std::cout << "] To range: [";
        for (size_t i = 0; i < node->to_range.lower_bound.size(); ++i) {
            std::cout << node->to_range.lower_bound[i] << ", " << node->to_range.upper_bound[i];
            if (i < node->to_range.lower_bound.size() - 1) std::cout << "; ";
        }
        std::cout << "] with to_job: " << node->to_job << "\n";
        printTree(node->right);
    }
};

class BucketIntervalTree {
private:
    struct FromBucketNode {
        BucketRange      from_range;
        IntervalTree    *to_tree;
        std::vector<int> max;
        FromBucketNode  *left;
        FromBucketNode  *right;

        FromBucketNode(const BucketRange &r) : from_range(r), max(r.upper_bound), left(nullptr), right(nullptr) {
            to_tree = new IntervalTree();
        }

        ~FromBucketNode() { delete to_tree; }
    };

    FromBucketNode *root;

    FromBucketNode *insert(FromBucketNode *node, const BucketRange &from_range, const BucketRange &to_range,
                           int to_job) {
        if (node == nullptr) {
            // Create a new FromBucketNode for this range
            return new FromBucketNode(from_range);
        }

        // Compare based on `from_range`
        if (from_range < node->from_range) {
            node->left = insert(node->left, from_range, to_range, to_job);
        } else if (from_range > node->from_range) {
            node->right = insert(node->right, from_range, to_range, to_job);
        } else {
            // If `from_range` matches exactly, insert into the `to_tree`
            node->to_tree->insert(from_range, to_range, to_job);
        }

        // Update max values (assuming the upper_bound size matches)
        for (size_t i = 0; i < node->max.size(); ++i) {
            node->max[i] = std::max(node->max[i], from_range.upper_bound[i]);
        }

        return node;
    }

    bool searchCombination(FromBucketNode *node, const BucketRange &from_range, const BucketRange &to_range,
                           int to_job) const {
        if (node == nullptr) return false;

        // Check if `from_range` is completely contained within `node->from_range`
        if (from_range.contained_in(node->from_range)) {
            // If `from_range` is contained, search in `to_tree` for `to_range` and `to_job`
            if (node->to_tree->search(from_range, to_range, to_job)) { return true; }
        }

        // Use updated comparison for traversing the tree
        if (from_range < node->from_range) {
            return searchCombination(node->left, from_range, to_range, to_job);
        } else {
            return searchCombination(node->right, from_range, to_range, to_job);
        }
    }

    void printTree(FromBucketNode *node) const {
        if (node == nullptr) return;

        printTree(node->left);
        std::cout << "From range: [";
        for (size_t i = 0; i < node->from_range.lower_bound.size(); ++i) {
            std::cout << node->from_range.lower_bound[i] << ", " << node->from_range.upper_bound[i];
            if (i < node->from_range.lower_bound.size() - 1) std::cout << "; ";
        }
        std::cout << "]\n";
        node->to_tree->print();
        printTree(node->right);
    }

public:
    BucketIntervalTree() : root(nullptr) {}

    void insert(const BucketRange &from_range, const BucketRange &to_range, int to_job) {
        root = insert(root, from_range, to_range, to_job);
    }

    bool search(const BucketRange &from_range, const BucketRange &to_range, int to_job) const {
        return searchCombination(root, from_range, to_range, to_job);
    }

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
