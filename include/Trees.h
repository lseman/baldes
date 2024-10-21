/**
 * @file Trees.h
 * @brief Header file containing the implementation of various tree structures used for interval and multidimensional
 * range queries.
 *
 * The classes provide functionalities for inserting intervals, searching for overlapping intervals,
 * and printing the tree structures for debugging purposes.
 *
 * @note The implementation uses various helper functions for tree operations such as insertion, searching,
 *       and splay operations.
 */
#pragma once
#include "Definitions.h"
#include "utils/NumericUtils.h"

/**
 * @struct BucketRange
 * @brief Represents a range with lower and upper bounds.
 *
 * The BucketRange structure stores the lower and upper bounds for a range of buckets.
 * The structure also provides comparison operators for ordering ranges based on the lower bounds.
 */
template <Direction D>
struct BucketRange {
    std::vector<double> lower_bound;
    std::vector<double> upper_bound;

    // Lexicographical comparison for ordering ranges
    bool operator<(const BucketRange &other) const {
        if constexpr (D == Direction::Forward) {
            return std::lexicographical_compare(lower_bound.begin(), lower_bound.end(), other.lower_bound.begin(),
                                                other.lower_bound.end());
        } else if constexpr (D == Direction::Backward) {
            return std::lexicographical_compare(upper_bound.begin(), upper_bound.end(), other.upper_bound.begin(),
                                                other.upper_bound.end());
        }
    }

    bool operator>(const BucketRange &other) const {
        if constexpr (D == Direction::Forward) {
            return std::lexicographical_compare(other.lower_bound.begin(), other.lower_bound.end(), lower_bound.begin(),
                                                lower_bound.end());
        } else if constexpr (D == Direction::Backward) {
            return std::lexicographical_compare(other.upper_bound.begin(), other.upper_bound.end(), upper_bound.begin(),
                                                upper_bound.end());
        }
    }

    // Optional: Define operator== and operator!= for completeness
    bool operator==(const BucketRange &other) const {
        if constexpr (D == Direction::Forward) {
            return lower_bound == other.lower_bound && upper_bound == other.upper_bound;
        } else if constexpr (D == Direction::Backward) {
            return upper_bound == other.upper_bound && lower_bound == other.lower_bound;
        }
    }

    bool operator!=(const BucketRange &other) const { return !(*this == other); }

    bool contained_in(const BucketRange &other) const {
        if constexpr (D == Direction::Forward) {
            for (size_t i = 0; i < lower_bound.size(); ++i) {
                if (!(lower_bound[i] >= other.lower_bound[i])) { return false; }
            }
        } else if constexpr (D == Direction::Backward) {
            for (size_t i = 0; i < upper_bound.size(); ++i) {
                if (!(upper_bound[i] <= other.upper_bound[i])) { return false; }
            }
        }
        return true;
    }
};

/**
 * @struct IntervalNode
 * @brief Represents a node in an interval tree structure.
 *
 * The IntervalNode structure stores the from_range, to_range, to_node, max values, and pointers to left and right
 * children.
 */
template <Direction D>
struct IntervalNode {
    BucketRange<D>      from_range;
    BucketRange<D>      to_range;
    int                 to_node;
    std::vector<double> max;
    IntervalNode       *left;
    IntervalNode       *right;
    mutable bool        merge_pending;
    int                 height; // New: Height of the node

    IntervalNode(const BucketRange<D> &f_range, const BucketRange<D> &t_range, int to_node)
        : from_range(f_range), to_range(t_range), to_node(to_node), max(f_range.upper_bound), left(nullptr),
          right(nullptr), merge_pending(false), height(1) {} // Height is initialized to 1
};

/**
 * @class IntervalTree
 * @brief A class representing an interval tree structure for managing and querying intervals.
 *
 * The IntervalTree class is used to store and query intervals, where each node in the tree represents
 * an interval with a corresponding to_node. The tree structure is used to efficiently search for overlapping
 * intervals based on the from_range and to_range of the intervals.
 * The tree supports insertion and search operations for intervals.
 * The tree is balanced using AVL rotations to maintain a balanced tree structure.
 * The tree nodes store the from_range, to_range, to_node, and max values for each node.
 *
 */
template <Direction D>
class IntervalTree {
private:
    IntervalNode<D> *root;

    int height(IntervalNode<D> *node) { return node ? node->height : 0; }

    int getBalance(IntervalNode<D> *node) { return node ? height(node->left) - height(node->right) : 0; }

    IntervalNode<D> *rightRotate(IntervalNode<D> *y) {
        if (y == nullptr || y->left == nullptr) {
            return y; // Return the node itself if rotation can't be performed
        }

        IntervalNode<D> *x  = y->left;  // x becomes the new root of the subtree
        IntervalNode<D> *T2 = x->right; // T2 will be the right child of x after rotation

        // Perform rotation
        x->right = y;
        y->left  = T2;

        // Update heights
        y->height = std::max(height(y->left), height(y->right)) + 1; // y's height depends on its new children
        x->height = std::max(height(x->left), height(x->right)) + 1; // x's height depends on its new children

        // Return new root
        return x;
    }

    IntervalNode<D> *leftRotate(IntervalNode<D> *x) {
        if (x == nullptr || x->right == nullptr) {
            return x; // Return the node itself if rotation can't be performed
        }

        IntervalNode<D> *y  = x->right; // y becomes the new root of the subtree
        IntervalNode<D> *T2 = y->left;  // T2 will be the left child of x after rotation

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
    bool doOverlap(const BucketRange<D> &i1, const BucketRange<D> &i2) const {
        if constexpr (D == Direction::Forward) {
            for (size_t i = 0; i < i1.lower_bound.size(); ++i) {
                if (!(i1.lower_bound[i] <= i2.lower_bound[i])) { return false; }
            }
            return true;
        } else if constexpr (D == Direction::Backward) {
            for (size_t i = 0; i < i1.upper_bound.size(); ++i) {
                if (!(i1.upper_bound[i] <= i2.upper_bound[i])) { return false; }
            }
            return true;
        }
        return true;
    }

    // Helper function to merge two overlapping ranges
    BucketRange<D> mergeRanges(const BucketRange<D> &i1, const BucketRange<D> &i2) const {
        BucketRange<D> merged;
        for (size_t i = 0; i < i1.lower_bound.size(); ++i) {
            merged.lower_bound.push_back(std::min(i1.lower_bound[i], i2.lower_bound[i]));
            merged.upper_bound.push_back(std::max(i1.upper_bound[i], i2.upper_bound[i]));
        }
        return merged;
    }

    // Helper function to insert a new range into the tree, with merging of overlapping to_ranges
    IntervalNode<D> *insert(IntervalNode<D> *node, const BucketRange<D> &from_range, const BucketRange<D> &to_range,
                            int to_node) {
        // Insert as in the original, unbalanced version
        if (node == nullptr) { return new IntervalNode(from_range, to_range, to_node); }

        // Compare based on from_range.lower_bound first
        if (from_range < node->from_range) {
            node->left = insert(node->left, from_range, to_range, to_node);
        } else if (from_range > node->from_range) {
            node->right = insert(node->right, from_range, to_range, to_node);
        } else {
            // If from_range is the same, check to_range and to_node
            bool toRangeEqual = to_range == node->to_range;

            // If to_range and to_node match, mark for merging
            if (toRangeEqual && node->to_node == to_node) {
                node->merge_pending = true; // Mark for merging
            } else {
                // Insert as a new node if to_range or to_node differs
                if constexpr (D == Direction::Forward) {
                    node->right = insert(node->right, from_range, to_range, to_node);
                } else if constexpr (D == Direction::Backward) {
                    node->left = insert(node->left, from_range, to_range, to_node);
                }
            }
        }

        // Update max values (assuming the upper_bound size matches)
        for (size_t i = 0; i < node->max.size(); ++i) {
            node->max[i] = std::max({node->max[i], from_range.upper_bound[i], node->to_range.upper_bound[i]});
        }

        // Update height of this node
        node->height = std::max(height(node->left), height(node->right)) + 1;

        // Get the balance factor of this node
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

        return node;
    }

    void applyPendingMerges(IntervalNode<D> *node) const {
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

    bool searchCombination(IntervalNode<D> *node, const BucketRange<D> &from_range, const BucketRange<D> &to_range,
                           int to_node) const {
        if (node == nullptr) return false;

        // Apply any pending merges to ensure node's `to_range` is up-to-date
        applyPendingMerges(node);

        // Check if `to_range` is contained within `node->to_range`
        bool toRangeContained = to_range.contained_in(node->to_range);

        // Debug output for `to_range` containment check
        // If both ranges are contained and the node matches, return true
        if (toRangeContained && node->to_node == to_node) { return true; }

        // Use updated comparison for traversing the tree
        if (from_range < node->from_range) {
            return searchCombination(node->left, from_range, to_range, to_node);
        } else if (from_range > node->from_range) {
            return searchCombination(node->right, from_range, to_range, to_node);
        }
        return true;
    }

public:
    IntervalTree() : root(nullptr) {}

    void insert(const BucketRange<D> &from_range, const BucketRange<D> &to_range, int to_node) {
        root = insert(root, from_range, to_range, to_node);
    }

    bool search(const BucketRange<D> &from_range, const BucketRange<D> &to_range, int to_node) const {
        return searchCombination(root, from_range, to_range, to_node);
    }

    void print() const { printTree(root); }

    void printTree(IntervalNode<D> *node) const {
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
        std::cout << "] with to_node: " << node->to_node << "\n";
        printTree(node->right);
    }
};

/**
 * @class BucketIntervalTree
 * @brief A class representing a tree structure for managing and querying bucket intervals.
 *
 * The BucketIntervalTree class is used to store and query bucket intervals, where each node in the tree
 * represents a range of buckets and the corresponding intervals associated with those buckets.
 * The tree structure is used to efficiently search for overlapping intervals based on the bucket ranges.
 * The tree supports insertion and search operations for bucket intervals.
 *
 */
template <Direction D>
class BucketIntervalTree {
private:
    struct FromBucketNode {
        BucketRange<D>      from_range;
        IntervalTree<D>    *to_tree;
        std::vector<double> max;
        FromBucketNode     *left;
        FromBucketNode     *right;

        FromBucketNode(const BucketRange<D> &r) : from_range(r), max(r.upper_bound), left(nullptr), right(nullptr) {
            to_tree = new IntervalTree<D>();
        }

        ~FromBucketNode() { delete to_tree; }
    };

    FromBucketNode *root;

    FromBucketNode *insert(FromBucketNode *node, const BucketRange<D> &from_range, const BucketRange<D> &to_range,
                           int to_node) {
        if (node == nullptr) {
            // Create a new FromBucketNode for this range
            return new FromBucketNode(from_range);
        }

        // Compare based on `from_range`
        if (from_range < node->from_range) {
            node->left = insert(node->left, from_range, to_range, to_node);
        } else if (from_range > node->from_range) {
            node->right = insert(node->right, from_range, to_range, to_node);
        } else if (from_range == node->from_range) {
            // If `from_range` matches exactly, insert into the `to_tree`
            node->to_tree->insert(from_range, to_range, to_node);
        }

        // Update max values (assuming the upper_bound size matches)
        for (size_t i = 0; i < node->max.size(); ++i) {
            node->max[i] = std::max(node->max[i], from_range.upper_bound[i]);
        }

        return node;
    }

    bool searchCombination(FromBucketNode *node, const BucketRange<D> &from_range, const BucketRange<D> &to_range,
                           int to_node) const {
        if (node == nullptr) return false;

        // Check if `from_range` is completely contained within `node->from_range`
        if (from_range.contained_in(node->from_range)) {
            // If `from_range` is contained, search in `to_tree` for `to_range` and `to_node`
            if (node->to_tree->search(from_range, to_range, to_node)) { return true; }
        }

        // Use updated comparison for traversing the tree
        if (from_range < node->from_range) {
            return searchCombination(node->left, from_range, to_range, to_node);
        } else if (from_range > node->from_range) {
            return searchCombination(node->right, from_range, to_range, to_node);
        } else {
            return true;
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

    void insert(const BucketRange<D> &from_range, const BucketRange<D> &to_range, int to_node) {
        root = insert(root, from_range, to_range, to_node);
    }

    bool search(const BucketRange<D> &from_range, const BucketRange<D> &to_range, int to_node) const {
        return searchCombination(root, from_range, to_range, to_node);
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
    std::vector<double> low;          // Lower bounds for each dimension
    std::vector<double> high;         // Upper bounds for each dimension
    int                 bucket_index; // Bucket index for this node
    TreeNode           *left;
    TreeNode           *right;
    TreeNode           *parent;

    TreeNode(const std::vector<double> &low, const std::vector<double> &high, int bucket_index)
        : low(low), high(high), bucket_index(bucket_index), left(nullptr), right(nullptr), parent(nullptr) {}

    bool contains(const std::vector<double> &point) const {
        for (size_t i = 0; i < low.size(); ++i) {
            if (point[i] < low[i] || point[i] > high[i]) { return false; }
        }
        return true;
    }

    bool is_less_than(const std::vector<double> &point) const {
        for (size_t i = 0; i < low.size(); ++i) {
            if (high[i] < point[i]) {
            //if (numericutils::less_than(high[i], point[i])) {
                return true;
            } else if (low[i] > point[i]) {
            //} else if (numericutils::greater_than(low[i], point[i])) {
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

    TreeNode *find(const std::vector<double> &point) {
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

    int query(const std::vector<double> &point) {
        TreeNode *node = find(point);
        if (node != nullptr) return node->bucket_index;
        return -1;
    }

    // Insert a new multidimensional interval
    void insert(const std::vector<double> &low, const std::vector<double> &high, int bucket_index) {

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
