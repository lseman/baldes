//! Strong-branching controls, child scores and candidate budgets.
use std::time::Duration;
mod budget;
pub(in crate::master) use budget::{CandidateBudget, SubtreeEstimate};
mod allocation;
pub(in crate::master) use allocation::{allocate_family_batches, fresh_family_batches};
mod subtree_history;
pub(in crate::master) use subtree_history::SubtreeHistory;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StrongPhase {
    pub candidates: usize,
    /// Trial budget as a fraction of the estimated subtree size. Zero disables
    /// the adaptive limit; estimates affect scheduling, never pruning.
    pub subtree_ratio: f64,
    pub pricing_phase_floor: usize,
    pub max_iterations: usize,
    pub tree_depth_score: bool,
    pub restricted_master_only: bool,
    /// Whether this phase uses exact-node treatment. `None` infers this from
    /// a pricing floor of zero and `restricted_master_only == false`.
    /// This controls treatment and reuse, never certifies an unfinished trial.
    pub exact: Option<bool>,
    /// Minimum rounds for nonexact treatment. Exact treatment retains the
    /// ordinary node's minimum-round policy.
    pub min_cut_rounds: usize,
    pub max_cut_rounds: usize,
    /// Enable the node's reduced-cost fixing and incumbent-gap enumeration.
    pub reduced_cost_fixing: bool,
}
impl StrongPhase {
    pub fn is_exact(&self) -> bool {
        self.exact
            .unwrap_or(!self.restricted_master_only && self.pricing_phase_floor == 0)
    }
}
impl Default for StrongPhase {
    fn default() -> Self {
        Self {
            candidates: 1,
            subtree_ratio: 1.,
            pricing_phase_floor: 0,
            max_iterations: 10000,
            tree_depth_score: true,
            restricted_master_only: false,
            exact: None,
            min_cut_rounds: 0,
            max_cut_rounds: 10000,
            reduced_cost_fixing: true,
        }
    }
}
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StrongBranchingOptions {
    pub phases: Vec<StrongPhase>,
    /// Optional pass cap, in addition to the tree deadline. Unlimited by default.
    pub time_limit: Duration,
    /// Optional child cap, in addition to the pass/tree deadline.
    pub trial_time_limit: Duration,
    /// Prefer the mean size of completed subtrees at this depth when available.
    /// Disabled by default, matching the native parent-estimate policy.
    pub use_history_subtree_size: bool,
    /// Stop an exact phase after its best candidate improves the parent's
    /// finite subtree estimate. This affects work allocation only.
    pub stop_after_improved_subtree: bool,
    /// Multiply gains of children not yet evaluated in this exact phase by
    /// this factor before comparing a partial group against the best score.
    pub partial_score_conservativeness: f64,
}
impl Default for StrongBranchingOptions {
    fn default() -> Self {
        Self {
            phases: vec![
                StrongPhase {
                    candidates: 100,
                    subtree_ratio: f64::from(0.3f32),
                    pricing_phase_floor: 0,
                    max_iterations: 0,
                    tree_depth_score: false,
                    restricted_master_only: true,
                    exact: Some(false),
                    min_cut_rounds: 0,
                    max_cut_rounds: 0,
                    reduced_cost_fixing: false,
                },
                StrongPhase {
                    candidates: 3,
                    subtree_ratio: f64::from(0.1f32),
                    pricing_phase_floor: 1,
                    max_iterations: 10000,
                    tree_depth_score: false,
                    restricted_master_only: false,
                    exact: Some(false),
                    min_cut_rounds: 0,
                    max_cut_rounds: 0,
                    reduced_cost_fixing: false,
                },
                StrongPhase {
                    candidates: 1,
                    subtree_ratio: 1.,
                    pricing_phase_floor: 0,
                    max_iterations: 10000,
                    tree_depth_score: true,
                    restricted_master_only: false,
                    exact: Some(true),
                    min_cut_rounds: 0,
                    max_cut_rounds: 10000,
                    reduced_cost_fixing: true,
                },
            ],
            time_limit: Duration::MAX,
            trial_time_limit: Duration::MAX,
            use_history_subtree_size: false,
            stop_after_improved_subtree: false,
            partial_score_conservativeness: 1.,
        }
    }
}
/// Product ranking. Gains after the two smallest are normalized by the
/// incumbent gap; a singleton uses gain times gap, an empty group gap squared.
/// Scores never supply a proof bound.
pub fn product_score(parent: f64, incumbent: f64, children: impl AsRef<[f64]>) -> f64 {
    let children = children.as_ref();
    let gap = incumbent - parent;
    if let [a, b] = children {
        // Preserve the allocation-free binary specialization and arithmetic.
        let mut gains = [a - parent, b - parent];
        let all_closed = gains.iter().all(|&g| g >= gap);
        for gain in &mut gains {
            if *gain > gap && !all_closed {
                *gain = (*gain - gap) * 1e-6 + gap;
            }
            *gain = gain.max(1e-6);
        }
        return gains[0] * gains[1];
    }
    if children.is_empty() {
        return gap * gap;
    }
    if let [child] = children {
        return (child - parent).max(1e-6) * gap;
    }
    let mut values = children.to_vec();
    values.sort_by(f64::total_cmp);
    let all_closed = values.iter().all(|&value| value - parent >= gap);
    let mut score = 1.;
    for (index, value) in values.into_iter().enumerate() {
        let mut gain = value - parent;
        if gain > gap && (!all_closed || index >= 2) {
            gain = (gain - gap) * 1e-6 + gap;
        }
        gain = gain.max(1e-6);
        score *= if index >= 2 { gain / gap } else { gain };
    }
    score
}
fn number_of_leaves(gap: f64, gains: &[f64]) -> f64 {
    let mut low = 0.;
    let mut high = 1e20;
    let mut middle: f64 = 5e19;
    for iteration in 0..100 {
        let sum = gains
            .iter()
            .map(|&gain| middle.powf(-gain / gap))
            .sum::<f64>();
        if sum < 1. {
            high = middle;
        } else {
            low = middle;
        }
        if middle > 9.99e19 {
            return -1.;
        }
        if iteration == 99 {
            return middle;
        }
        middle = (low + high) * 0.5;
        if high / 1e6 > high - low {
            return middle;
        }
    }
    middle
}
/// Tree-depth ranking, including the singleton inverse-gain rule. A normalization
/// gap of one corresponds to completed native evaluation. Other normalizations
/// remain available to callers; this score is not a proof bound.
pub fn tree_depth_score(
    parent: f64,
    incumbent: f64,
    children: impl AsRef<[f64]>,
    normalization: f64,
) -> f64 {
    tree_depth_metrics(parent, incumbent, children, normalization).0
}

fn prepare_tree_gains(
    gap: f64,
    gains: &mut [f64],
    processed: usize,
    conservativeness: f64,
) -> bool {
    // Detection uses the unscaled gains, before pending treatment and clipping.
    let zero = gains.iter().filter(|&&gain| gain < 1e-6).count();
    if zero == gains.len() {
        return false;
    }
    for (side, gain) in gains.iter_mut().enumerate() {
        if side >= processed {
            *gain *= conservativeness;
        }
        *gain = gain.min(gap);
        if zero != 0 {
            *gain = gain.max(gap * 0.0001);
        }
    }
    true
}

fn normalized_tree_score(normalization: f64, gains: &[f64]) -> f64 {
    if let [gain] = gains {
        return -normalization / gain;
    }
    let leaves = number_of_leaves(normalization, gains);
    if leaves >= 0. {
        -leaves.ln() / (gains.len() as f64).ln()
    } else {
        -1e12
    }
}

/// Compute ranking and actual subtree size without a logarithm/exponential
/// round trip. Scratch remains on the stack for the ordinary binary case.
pub(in crate::master) fn tree_depth_metrics(
    parent: f64,
    incumbent: f64,
    children: impl AsRef<[f64]>,
    normalization: f64,
) -> (f64, f64) {
    let children = children.as_ref();
    if children.is_empty() {
        return (-1e12, 1e12);
    }
    if let [child] = children {
        let gap = incumbent - parent;
        let mut gains = [child - parent];
        if !prepare_tree_gains(gap, &mut gains, 1, 1.) {
            return (-1e12, 1e12);
        }
        return (normalized_tree_score(normalization, &gains), gap / gains[0]);
    }
    let mut pair = [children[0] - parent, children[1] - parent];
    let mut many = Vec::new();
    let gains = if children.len() == 2 {
        &mut pair[..]
    } else {
        many.extend(children.iter().map(|bound| bound - parent));
        &mut many[..]
    };
    let gap = incumbent - parent;
    if !prepare_tree_gains(gap, gains, gains.len(), 1.) {
        return (-1e12, 1e12);
    }
    let score = normalized_tree_score(normalization, gains);
    let actual_leaves = number_of_leaves(gap, gains);
    let size = if actual_leaves > 0. {
        // Preserve the native division/multiplication order at every arity.
        (actual_leaves - 1.) / (gains.len() - 1) as f64 * gains.len() as f64
    } else {
        1e12
    };
    (score, size)
}

/// Optimistic score in current child order. Only the pending suffix is scaled;
/// zero detection precedes scaling, which precedes incumbent-gap clipping.
pub(in crate::master) fn partial_tree_depth_score(
    parent: f64,
    incumbent: f64,
    children: impl AsRef<[f64]>,
    processed: usize,
    conservativeness: f64,
) -> f64 {
    let children = children.as_ref();
    if children.is_empty() {
        return -1e12;
    }
    if let [child] = children {
        let mut gains = [child - parent];
        if !prepare_tree_gains(incumbent - parent, &mut gains, processed, conservativeness) {
            return -1e12;
        }
        return normalized_tree_score(1., &gains);
    }
    let mut pair = [children[0] - parent, children[1] - parent];
    let mut many = Vec::new();
    let gains = if children.len() == 2 {
        &mut pair[..]
    } else {
        many.extend(children.iter().map(|bound| bound - parent));
        &mut many[..]
    };
    if !prepare_tree_gains(incumbent - parent, gains, processed, conservativeness) {
        return -1e12;
    }
    normalized_tree_score(1., gains)
}

/// Tree-depth ranking uses a wider equality band than ordinary score gains.
/// `first_child_order` preserves relative child allocation order across groups.
/// This predicate is intentionally separate from the exact product ordering.
pub(in crate::master) fn tree_depth_score_less(a: (f64, usize), b: (f64, usize)) -> bool {
    let magnitude = a.0.abs().max(b.0.abs());
    let equality = 1e-6 * (1. + magnitude);
    if a.0 >= b.0 - equality && b.0 >= a.0 - equality {
        a.1 < b.1
    } else {
        a.0 - (1e-6 + magnitude * 1e-10) > b.0
    }
}
// Compare child estimates in their original order, across candidate families.
pub(in crate::master) fn retain_distinct_trials<T, V: AsRef<[f64]>>(
    candidates: &mut Vec<T>,
    values: impl Fn(&T) -> (f64, V),
) {
    let mut retained: Vec<(f64, V)> = vec![];
    candidates.retain(|candidate| {
        let (candidate_score, estimates) = values(candidate);
        let duplicate = retained.iter().rev().any(|(score, previous)| {
            *score <= candidate_score + 1e-6
                && previous.as_ref().len() == estimates.as_ref().len()
                && previous
                    .as_ref()
                    .iter()
                    .copied()
                    .zip(estimates.as_ref().iter().copied())
                    .all(|(a, b)| {
                        if !a.is_finite() || !b.is_finite() {
                            return false;
                        }
                        // Recovered continuous Bound comparisons against a +/-
                        // precision interval, with relative Double slack.
                        let lower = a - 1e-6;
                        let upper = a + 1e-6;
                        lower - (1e-6 + lower.abs().max(b.abs()) * 1e-10) <= b
                            && b - (1e-6 + upper.abs().max(b.abs()) * 1e-10) <= upper
                    })
        });
        if !duplicate {
            retained.push((candidate_score, estimates));
        }
        !duplicate
    });
}

/// Admit fresh candidates by decreasing priority bucket until the quota is
/// filled; a bucket started below the first selected integer group stops it.
/// Returns the candidates and the actual loop cutoff.
pub(in crate::master) fn fresh_batches_with_budget<T>(
    mut quota: usize,
    buckets: &[f64],
    mut budget: Option<&mut CandidateBudget>,
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

#[cfg(test)]
#[path = "../../../../tests/unit/master/tree/strong/trial_equivalence_tests.rs"]
mod trial_equivalence_tests;

#[cfg(test)]
#[path = "../../../../tests/unit/master/tree/strong/numerical_metrics_tests.rs"]
mod numerical_metrics_tests;

#[cfg(test)]
#[path = "../../../../tests/unit/master/tree/strong/multiway_metrics_tests.rs"]
mod multiway_metrics_tests;
