//! Fresh-priority stopping and stored-generator admission are ranking rules.
use super::*;

pub(super) fn admitted(priority: f64, cutoff: f64) -> bool {
    cutoff - (1e-6 + priority.abs().max(cutoff.abs()) * 1e-10) <= priority
}

pub(in crate::master::tree) fn registered(options: &BranchingOptions) -> Vec<f64> {
    let p = &options.priorities;
    let mut values = vec![p.arc_flow];
    for (enabled, value) in [
        (options.number_of_vehicles, p.number_of_vehicles),
        (options.accumulated_resource, p.accumulated_resource),
        (options.clusters.is_some(), p.clusters),
        (options.packing_set_assignment, p.packing_set_assignment),
        (options.ryan_foster, p.ryan_foster),
        (options.aggregate_flow(), p.aggregate_flow),
        (options.window_arcs.is_some(), p.time_window_selection),
    ] {
        if enabled {
            values.push(value);
        }
    }
    values.sort_by(|a, b| b.total_cmp(a));
    values.dedup_by(|a, b| *a == *b);
    values
}

/// Return fresh indices and the actual loop cutoff. Empty registered families
/// still affect the cutoff. Batch output can fill/overrun quota before advancing.
fn separate(priorities: &[f64], quota: usize, buckets: &[f64]) -> (Vec<usize>, f64) {
    fresh_batches(quota, buckets, |priority| {
        priorities
            .iter()
            .enumerate()
            .filter_map(|(i, &p)| (p == priority).then_some(i))
            .collect()
    })
}
pub(in crate::master::tree) fn fresh_batches<T>(
    quota: usize,
    buckets: &[f64],
    generate: impl FnMut(f64) -> Vec<T>,
) -> (Vec<T>, f64) {
    fresh_batches_with_budget(quota, buckets, None, generate)
}
pub(in crate::master::tree) fn fresh_batches_with_budget<T>(
    mut quota: usize,
    buckets: &[f64],
    mut budget: Option<&mut super::super::strong::CandidateBudget>,
    mut generate: impl FnMut(f64) -> Vec<T>,
) -> (Vec<T>, f64) {
    let mut selected = vec![];
    let mut cutoff = buckets.first().copied().unwrap_or(f64::NEG_INFINITY);
    let mut group = f64::INFINITY;
    if quota == 0 {
        return (selected, cutoff);
    }
    for &priority in buckets {
        cutoff = priority;
        if let Some(budget) = budget.as_deref_mut() {
            budget.visit(priority);
            quota = budget.quota;
        }
        if !selected.is_empty() && priority < group {
            break;
        }
        group = priority.floor();
        selected.extend(generate(priority));
        if selected.len() >= quota {
            break;
        }
    }
    (selected, cutoff)
}

impl BranchEvaluationHistory {
    pub(in crate::master::tree) fn cutoff(
        candidates: &[BranchCandidate],
        quota: usize,
        options: &BranchingOptions,
    ) -> f64 {
        separate(
            &candidates
                .iter()
                .map(|c| options.priorities.priority(&c.down))
                .collect::<Vec<_>>(),
            quota,
            &registered(options),
        )
        .1
    }
    pub(in crate::master::tree) fn select_admitted(
        &mut self,
        candidates: &mut Vec<BranchCandidate>,
        fresh_count: usize,
        quota: usize,
        branching: &BranchingOptions,
        history: Option<&BranchHistoryOptions>,
        gap: f64,
    ) {
        let priorities: Vec<_> = candidates
            .iter()
            .map(|c| branching.priorities.priority(&c.down))
            .collect();
        let (fresh, cutoff) = separate(&priorities[..fresh_count], quota, &registered(branching));
        let mut ranked = vec![];
        if let Some(options) = history.filter(|_| quota >= 2) {
            for (i, candidate) in candidates.iter().enumerate() {
                if !admitted(priorities[i], cutoff) {
                    continue;
                }
                if let Some((stage, score)) = self.prediction(candidate, options, gap) {
                    let name = branching.generator_names.label(&candidate.down);
                    ranked.push((i, stage, score, name));
                }
            }
            ranked.sort_by(|a, b| {
                b.1.cmp(&a.1)
                    .then_with(|| b.2.partial_cmp(&a.2).unwrap())
                    .then_with(|| a.3.cmp(&b.3))
                    .then(a.0.cmp(&b.0))
            });
            ranked.truncate(quota / 2);
        }
        self.ranked_candidates += ranked.len();
        let mut used = vec![false; candidates.len()];
        let mut selected = Vec::with_capacity(quota.min(candidates.len()));
        for (i, _, _, _) in ranked {
            used[i] = true;
            selected.push(candidates[i].clone());
        }
        selected.extend(
            fresh
                .into_iter()
                .filter(|&i| !used[i])
                .map(|i| candidates[i].clone()),
        );
        selected.truncate(quota);
        *candidates = selected;
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/tree/history/admission/tests.rs"]
mod tests;
