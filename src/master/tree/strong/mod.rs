//! Strong-branching controls, child scores and candidate budgets.
use std::time::Duration;
mod budget;
pub(in crate::master) use budget::{CandidateBudget, SubtreeEstimate};
mod subtree_history;
pub(in crate::master) use subtree_history::SubtreeHistory;

#[derive(Clone, Debug)]
pub struct StrongPhase {
    pub candidates: usize,
    /// Trial budget as a fraction of the estimated subtree size. Zero disables
    /// the adaptive limit; estimates affect scheduling, never pruning.
    pub subtree_ratio: f64,
    pub pricing_phase_floor: usize,
    pub max_iterations: usize,
    pub tree_depth_score: bool,
    pub restricted_master_only: bool,
}
#[derive(Clone, Debug)]
pub struct StrongBranchingOptions {
    pub phases: Vec<StrongPhase>,
    pub time_limit: Duration,
    pub trial_time_limit: Duration,
    /// Prefer the mean size of completed subtrees at this depth when available.
    /// Disabled by default, matching the native parent-estimate policy.
    pub use_history_subtree_size: bool,
}
impl Default for StrongBranchingOptions {
    fn default() -> Self {
        Self {
            phases: vec![
                StrongPhase {
                    candidates: 100,
                    subtree_ratio: f64::from(0.3f32),
                    pricing_phase_floor: 0,
                    max_iterations: 1,
                    tree_depth_score: false,
                    restricted_master_only: true,
                },
                StrongPhase {
                    candidates: 3,
                    subtree_ratio: f64::from(0.1f32),
                    pricing_phase_floor: 1,
                    max_iterations: 10000,
                    tree_depth_score: false,
                    restricted_master_only: false,
                },
                StrongPhase {
                    candidates: 1,
                    subtree_ratio: 1.,
                    pricing_phase_floor: 0,
                    max_iterations: 10000,
                    tree_depth_score: true,
                    restricted_master_only: false,
                },
            ],
            time_limit: Duration::from_secs(10),
            trial_time_limit: Duration::from_secs(1),
            use_history_subtree_size: false,
        }
    }
}
/// Binary specialization of CandidateBranchGroup::computeBranchingScoreAsProduct.
/// If just one child exceeds the incumbent gap, compress its excess by 1e-6;
/// if both do, preserve both gains. Always floor a gain at 1e-6.
pub fn product_score(parent: f64, incumbent: f64, children: [f64; 2]) -> f64 {
    let gap = incumbent - parent;
    let mut gains = children.map(|b| b - parent);
    let all_closed = gains.iter().all(|&g| g >= gap);
    for gain in &mut gains {
        if *gain > gap && !all_closed {
            *gain = (*gain - gap) * 1e-6 + gap;
        }
        *gain = gain.max(1e-6);
    }
    gains[0] * gains[1]
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
/// Native binary tree-depth estimate. A fixed normalization gap of one is
/// sufficient for ranking; callers can supply the reference's normalization.
pub fn tree_depth_score(
    parent: f64,
    incumbent: f64,
    children: [f64; 2],
    normalization: f64,
) -> f64 {
    tree_metrics(parent, incumbent, children, normalization).0
}
fn tree_metrics(parent: f64, incumbent: f64, children: [f64; 2], normalization: f64) -> (f64, f64) {
    let gap = incumbent - parent;
    let raw = children.map(|b| b - parent);
    let zero = raw.iter().filter(|&&g| g < 1e-6).count();
    if zero == 2 {
        return (-1e12, 1e12);
    }
    let mut gains = raw.map(|g| g.min(gap));
    if zero != 0 {
        for gain in &mut gains {
            *gain = gain.max(gap * 0.0001);
        }
    }
    let leaves = number_of_leaves(normalization, &gains);
    let score = if leaves >= 0. {
        -leaves.ln() / 2f64.ln()
    } else {
        -1e12
    };
    // Actual tree size uses the incumbent gap, independently of the
    // normalization used to compare ranking scores.
    let actual_leaves = number_of_leaves(gap, &gains);
    let size = if actual_leaves > 0. {
        2. * (actual_leaves - 1.)
    } else {
        1e12
    };
    (score, size)
}
// The product-score phase keeps the first candidate for equivalent sorted
// child estimates before applying the next phase's quota. This is a ranking
// reduction only; no node is pruned by an equivalent trial estimate.
pub(in crate::master) fn retain_distinct_trials<T>(
    candidates: &mut Vec<T>,
    values: impl Fn(&T) -> (f64, [f64; 2]),
) {
    let mut retained: Vec<(f64, [f64; 2])> = vec![];
    candidates.retain(|candidate| {
        let (candidate_score, mut estimates) = values(candidate);
        estimates.sort_by(f64::total_cmp);
        let duplicate = retained.iter().rev().any(|&(score, previous)| {
            score <= candidate_score + 1e-6
                && previous.into_iter().zip(estimates).all(|(a, b)| {
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
