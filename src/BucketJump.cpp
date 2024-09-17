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

#include "../include/Definitions.h"
#include "../include/BucketGraph.h"
#include <algorithm>
#include <cmath>

#include <functional>
#include <vector>

/**
 * @brief Obtains jump bucket arcs for the BucketGraph.
 *
 * This function iterates over each bucket in the BucketGraph and adds jump arcs to the set of buckets.
 * It checks if each arc exists in the given Gamma vector for the current bucket.
 * If an arc exists, it adds the corresponding bucket to the set of buckets to add jump arcs to.
 * It then removes the non-component-wise minimal buckets from the set.
 * Finally, it adds jump arcs from the current bucket to each bucket in the set.
 *
 * @param Gamma The vector of BucketArcs to check for each bucket.
 */
void BucketGraph::ObtainJumpBucketArcs(const std::vector<BucketArc> &Gamma) {
    auto               &buckets = fw_buckets; // Assuming fw_buckets is a global or accessible variable
    std::vector<double> res     = {0.0};
    auto                cost    = 0.0;

    for (auto b = 0; b < fw_buckets_size; ++b) {
        std::vector<int> B_bar;
        // Iterate over Gamma
        auto arcs = buckets[b].get_bucket_arcs(true);
        for (const auto &gamma : arcs) {
            if (fw_fixed_buckets[gamma.from_bucket][gamma.to_bucket] == 0) { continue; }

            auto b_from = gamma.from_bucket;
            auto b_to   = gamma.to_bucket;

            auto from_job = buckets[b_from].job_id;
            auto to_job   = buckets[b_to].job_id;

            auto job_start = to_job * num_buckets_per_job;
            for (auto b_prime = job_start; b_prime < job_start + num_buckets_per_job; ++b_prime) {
                // std::print("Adding bucket {} to B_bar\n", b_prime);
                B_bar.push_back(b_prime);
            }
            cost = gamma.cost_increment;
            res  = gamma.resource_increment;

            for (auto it = B_bar.begin(); it != B_bar.end();) {
                // check if element is in Phi[b]
                if (std::find(Phi_fw[b].begin(), Phi_fw[b].end(), *it) == Phi_fw[b].end()) {
                    it = B_bar.erase(it);
                } else {
                    ++it;
                }
            }

            for (const auto &b_prime : B_bar) {
                std::print("Adding jump arc from {} to {}\n", b, b_prime);
                buckets[b_from].add_jump_arc(b_from, b_prime, res, cost, true);
            }
        }
    }
}

/**
 * Checks if a given bucket is present in the bucket set.
 *
 * @param bucket_set The set of buckets to search in.
 * @param bucket The bucket to check for.
 * @return True if the bucket is found in the set, false otherwise.
 */
bool BucketGraph::BucketSetContains(const std::set<int> &bucket_set, const int &bucket) {
    return bucket_set.find(bucket) != bucket_set.end();
}
