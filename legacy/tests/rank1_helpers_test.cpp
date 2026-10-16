#include "cuts/rank1/Helpers.h"

#include <cassert>
#include <vector>

namespace {

CandidateSet makeCandidate(const std::vector<int> &nodes, const std::vector<int> &neighbors, double violation) {
    SRCPermutation permutation;
    permutation.num = {1, 1, 1};
    permutation.den = 2;
    return CandidateSet(nodes, violation, permutation, neighbors);
}

void test_hash_is_independent_of_set_insertion_order() {
    const CandidateSet first  = makeCandidate({1, 2, 3}, {4, 5, 6}, 0.2);
    const CandidateSet second = makeCandidate({3, 1, 2}, {6, 4, 5}, 0.9);

    assert(first == second);
    assert(CandidateSetHasher{}(first) == CandidateSetHasher{}(second));
}

void test_candidate_collection_uses_identity_not_violation_as_equality() {
    CandidateSetCollection candidates;
    candidates.emplace(makeCandidate({1, 2, 3}, {4, 5}, 0.2));
    candidates.emplace(makeCandidate({3, 2, 1}, {5, 4}, 0.9));

    assert(candidates.size() == 1);

    candidates.emplace(makeCandidate({1, 2, 3}, {4, 6}, 0.9));
    assert(candidates.size() == 2);
}

void test_rank3_sparse_delta_matches_full_floor_coefficient() {
    for (int i_visits = 0; i_visits <= 4; ++i_visits) {
        for (int j_visits = 0; j_visits <= 4; ++j_visits) {
            for (int k_visits = 0; k_visits <= 4; ++k_visits) {
                const int base = i_visits + j_visits;
                const int expected = ((base + k_visits) / 2) - (base / 2);
                assert(rank1::rank3_floor_coefficient_delta(base, k_visits) == expected);
            }
        }
    }
}

} // namespace

int main() {
    test_hash_is_independent_of_set_insertion_order();
    test_candidate_collection_uses_identity_not_violation_as_equality();
    test_rank3_sparse_delta_matches_full_floor_coefficient();
}
