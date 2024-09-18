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
void BucketGraph::ObtainJumpBucketArcs() {
    auto               &buckets         = fw_buckets; // Assuming fw_buckets is a global or accessible variable
    std::vector<double> res             = {0.0};
    auto                cost            = 0.0;
    auto                missing_counter = 0;

    int arc_counter = 0;
    for (auto b = 0; b < fw_buckets_size; ++b) {
        std::vector<int> B_bar;

        // Retrieve the arcs from the current bucket
        auto arcs          = buckets[b].template get_bucket_arcs<Direction::Forward>();
        auto original_arcs = jobs[buckets[b].job_id].template get_arcs<Direction::Forward>();
        if (arcs.empty()) { continue; }

        for (const auto &gamma : arcs) {
            if (fw_fixed_buckets[gamma.from_bucket][gamma.to_bucket] == 0) { continue; }
            auto from_job  = buckets[gamma.from_bucket].job_id;
            auto to_job    = buckets[gamma.to_bucket].job_id;
            bool have_path = false;
            for (const auto &gamma_prime : arcs) {
                if (fw_fixed_buckets[gamma.from_bucket][gamma.to_bucket] == 1) { continue; }

                auto prime_from_job = buckets[gamma_prime.from_bucket].job_id;
                auto prime_to_job   = buckets[gamma_prime.to_bucket].job_id;

                if (from_job == prime_from_job && to_job == prime_to_job) {
                    have_path = true;
                    break;
                }
            }
            if (!have_path) {
                // fmt::print("Path missing from {} to {}\n", from_job, to_job);
                missing_counter++;
            }
        }
        // if (!have_path) {
        //     // print path missing from b to b_prime
        //     fmt::print("Path missing from {} to {}\n", from_job, to_job);
        // }

        // remove from B_bar buckets that are not component-wise minimal

        // Add jump arcs for the remaining elements in B_bar
        for (const auto &b_prime : B_bar) { arc_counter++; }
    }
    fmt::print("Missing paths: {}\n", missing_counter);

    print_info("Added {} jump arcs\n", arc_counter);
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
