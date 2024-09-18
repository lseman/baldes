/**
 * @file BucketJump.cpp
 * @brief Implementation of the methods for handling jump arcs in the BucketGraph.
 *
 * This file contains the implementation of functions related to adding and managing jump arcs between buckets
 * in the BucketGraph. Jump arcs allow for direct connections between non-adjacent buckets, enhancing the efficiency
 * of the graph-based approach for solving vehicle routing problems (VRP).
 *
 * The main components of this file include:
 * - ObtainJumpBucketArcs: A method for calculating and adding jump arcs between buckets based on the given set of
 *   BucketArcs and a set of criteria for determining component-wise minimality.
 * - BucketSetContains: A helper function to check if a given bucket is present in a specified set.
 *
 * The main methods implemented in this file provide functionality for:
 * - Iterating over buckets and checking for potential jump arcs.
 * - Ensuring that non-component-wise minimal buckets are excluded.
 * - Adding jump arcs between buckets based on cost and resource increments.
 * - Performing set operations for checking membership of buckets.
 *
 * This file is part of the BucketGraph implementation, focusing on managing jump arcs as part of a larger
 * vehicle routing optimization framework.
 */

#include "../include/BucketGraph.h"
#include "../include/Definitions.h"
#include <algorithm>
#include <cmath>

#include <fmt/color.h>
#include <functional>
#include <vector>

