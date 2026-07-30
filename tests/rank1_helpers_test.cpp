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

} // namespace

int main() {
    test_hash_is_independent_of_set_insertion_order();
    test_candidate_collection_uses_identity_not_violation_as_equality();
}
