use super::CandidateBudget;

/// Keep separator invocations distinct until history has reserved its share.
/// Every family in a visited bucket receives the whole current quota.
pub(in crate::master) fn fresh_family_batches<T>(
    mut quota: usize,
    buckets: &[f64],
    mut budget: Option<&mut CandidateBudget>,
    mut generate: impl FnMut(f64, usize) -> Vec<Vec<T>>,
) -> (Vec<Vec<T>>, f64) {
    let mut batches = Vec::new();
    let mut count = 0;
    let mut cutoff = buckets.first().copied().unwrap_or(f64::NEG_INFINITY);
    let mut group = f64::INFINITY;
    if quota == 0 {
        return (batches, 0.);
    }
    for &priority in buckets {
        cutoff = priority;
        if let Some(budget) = budget.as_deref_mut() {
            budget.visit(priority);
            quota = budget.quota;
        }
        if count != 0 && priority < group {
            break;
        }
        group = priority.floor();
        for mut batch in generate(priority, quota) {
            batch.truncate(quota);
            if !batch.is_empty() {
                count += batch.len();
                batches.push(batch);
            }
        }
        if count >= quota {
            break;
        }
    }
    (batches, cutoff)
}

/// Divide the post-history quota between nonempty family sets. Smaller sets
/// go first; equal sizes retain separator encounter order. Historical names
/// are excluded without spending a share. Fresh names do not exclude each other.
pub(in crate::master) fn allocate_family_batches<T>(
    mut batches: Vec<Vec<T>>,
    mut quota: usize,
    mut historical: impl FnMut(&T) -> bool,
) -> Vec<T> {
    batches.retain(|batch| !batch.is_empty());
    batches.sort_by_key(Vec::len);
    let mut remaining = batches.len();
    let mut selected = Vec::new();
    for batch in batches {
        let mut share = quota / remaining;
        for candidate in batch {
            if share == 0 {
                break;
            }
            if historical(&candidate) {
                continue;
            }
            selected.push(candidate);
            share -= 1;
            quota -= 1;
        }
        remaining -= 1;
    }
    selected
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/tree/strong/family_allocation_tests.rs"]
mod tests;
