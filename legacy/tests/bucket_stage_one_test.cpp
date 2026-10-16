#include "pricing/bucket_graph/model/Bucket.h"

#include <cassert>
#include <cstdint>
#include <vector>

namespace {

void test_empty_bucket_accepts_without_scanning() {
    Bucket   bucket(1, {0.0}, {100.0});
    Label    candidate;
    uint64_t scanned = 0;

    candidate.cost = 10.0;

    assert(!bucket.apply_stage_one_dominance(&candidate, scanned));
    assert(scanned == 0);
}

void test_cached_minimum_rejects_worse_and_equal_candidates() {
    Bucket bucket(1, {0.0}, {100.0});
    Label  incumbent;
    Label  worse;
    Label  equal;

    incumbent.cost = 10.0;
    worse.cost     = 12.0;
    equal.cost     = 10.0;
    bucket.add_label(&incumbent);

    uint64_t scanned = 0;
    assert(bucket.apply_stage_one_dominance(&worse, scanned));
    assert(scanned == 1);
    assert(!incumbent.is_dominated);

    scanned = 0;
    assert(bucket.apply_stage_one_dominance(&equal, scanned));
    assert(scanned == 1);
    assert(!incumbent.is_dominated);
}

void test_new_record_invalidates_staged_and_committed_labels() {
    Bucket bucket(1, {0.0}, {100.0});
    Label  committed;
    Label  staged;
    Label  record;

    committed.cost = 10.0;
    bucket.add_label(&committed);
    bucket.flush_extra_labels();

    staged.cost = 8.0;
    bucket.add_label(&staged);

    record.cost     = 5.0;
    uint64_t scanned = 0;
    assert(!bucket.apply_stage_one_dominance(&record, scanned));
    assert(scanned == 2);
    assert(committed.is_dominated);
    assert(staged.is_dominated);
}

} // namespace

int main() {
    test_empty_bucket_accepts_without_scanning();
    test_cached_minimum_rejects_worse_and_equal_candidates();
    test_new_record_invalidates_staged_and_committed_labels();
}
