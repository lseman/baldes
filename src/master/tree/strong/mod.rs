//! Phased strong-branching decision rules for arc-flow candidates.
use super::*;
use crate::master::{clp::ClpMaster, LpMaster};
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
impl StrongPhase {
    fn requires_evaluation(&self, next: Option<&Self>, candidates: usize) -> bool {
        (!self.restricted_master_only && self.pricing_phase_floor == 0)
            || !next.is_some_and(|next| next.candidates > 0 && candidates <= next.candidates)
    }
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

struct Candidate {
    branch: super::candidates::BranchCandidate,
    order: usize,
    score: f64,
    estimates: [f64; 2],
    bounds: [f64; 2],
    stages: [usize; 2],
}
fn restricted_value<const D: usize>(
    parent: Option<&Rc<routing::NodeSnapshot<D>>>,
    problem: &RoutingContext<D>,
    branches: &[Branch],
    pool: &Pool,
    options: &TreeOptions,
) -> Result<f64> {
    let penalty = options.root.cg.artificial_cost;
    if let Some(parent) = parent {
        let depth = parent.branch_depth();
        if depth <= branches.len() {
            return Ok(parent.restricted_value(&branches[depth..], penalty)?.0);
        }
    }
    let child = branch_problem(problem, branches)?;
    let mut master = ClpMaster::new()?;
    master.set_solver(options.root.master_solver);
    master.set_backend_threads(options.root.master_threads)?;
    master.set_tolerances(1e-7, 1e-7)?;
    for row in &child.model.rows {
        master.add_row(row.sense, row.rhs, &[])?;
    }
    for route in &pool.routes {
        let entries = child.base_entries(route)?;
        master.add_column(route.original_cost, &entries)?;
    }
    for (r, row) in child.model.rows.iter().enumerate() {
        if row.sense != RowSense::LessEqual {
            master.add_column(penalty, &[(r, 1.)])?;
        }
        if row.sense != RowSense::GreaterEqual {
            master.add_column(penalty, &[(r, -1.)])?;
        }
    }
    Ok(if master.solve() {
        master.objective()
    } else {
        1e12
    })
}
/// Return a chosen candidate and its selected-mode child bounds. Restricted LP
/// objectives and heuristic pricing values affect scores only.
pub(super) fn choose<const D: usize>(
    parent: Option<&Rc<routing::NodeSnapshot<D>>>,
    problem: &RoutingContext<D>,
    node: &Node<D>,
    supplied: Vec<super::candidates::BranchCandidate>,
    fresh_count: usize,
    quota: usize,
    cutoff: f64,
    pool: &mut Pool,
    options: &TreeOptions,
    output: &mut TreeOutput,
    deadline: Instant,
) -> Result<
    Option<(
        super::candidates::BranchCandidate,
        [f64; 2],
        SubtreeEstimate,
    )>,
> {
    let mut supplied = supplied;
    if options
        .branch_history
        .as_ref()
        .is_some_and(|h| h.policy == BranchHistoryPolicy::Reliable)
    {
        if !node.branches.is_empty() {
            if let Some(history) = &options.branch_history {
                let limit = options
                    .strong_branching
                    .as_ref()
                    .map_or(supplied.len().saturating_mul(2), |strong| {
                        strong.phases.first().map_or(0, |p| p.candidates)
                    });
                output.branch_history.prioritize(
                    &mut supplied,
                    limit,
                    history,
                    output.incumbent - node.bound,
                );
            }
        }
    } else {
        let limit = if options.strong_branching.is_some() {
            quota
        } else {
            fresh_count.saturating_mul(2).max(2)
        };
        output.branch_history.select_admitted(
            &mut supplied,
            fresh_count,
            limit,
            &options.branching,
            options
                .branch_history
                .as_ref()
                .filter(|_| !node.branches.is_empty()),
            output.incumbent - node.bound,
        );
    }
    let mut candidates: Vec<_> = supplied
        .into_iter()
        .enumerate()
        .map(|(order, branch)| Candidate {
            branch,
            order,
            score: 0.,
            estimates: [node.bound; 2],
            bounds: [node.bound; 2],
            stages: [0; 2],
        })
        .collect();
    let Some(config) = &options.strong_branching else {
        return Ok(candidates
            .first()
            .map(|c| (c.branch.clone(), c.bounds, SubtreeEstimate::default())));
    };
    let adaptive = !options
        .branch_history
        .as_ref()
        .is_some_and(|history| history.policy == BranchHistoryPolicy::Reliable);
    let budget = CandidateBudget::new(quota, 0., node.subtree);
    let mut subtree = SubtreeEstimate {
        size: 1e12,
        priority: cutoff,
    };
    let mut best_score = -1e12;
    let end = deadline.min(Instant::now() + config.time_limit);
    for (phase_index, phase) in config.phases.iter().enumerate() {
        candidates.truncate(phase.candidates);
        if candidates.is_empty() || Instant::now() >= end {
            break;
        }
        // Intermediate ranking is unnecessary if every candidate fits in the
        // next phase. Native history stages count evaluations actually run on
        // each side, so a skipped phase must not consume a stage number.
        // Explicit heuristic-only terminal phases still run their trials.
        if !phase.requires_evaluation(config.phases.get(phase_index + 1), candidates.len()) {
            continue;
        }
        let mut evaluated = 0;
        for candidate in &mut candidates {
            if adaptive && !budget.permits(evaluated, phase.subtree_ratio, subtree.size) {
                break;
            }
            evaluated += 1;
            let mut estimates = [node.bound; 2];
            for (side, constraint) in [candidate.branch.down.clone(), candidate.branch.up.clone()]
                .into_iter()
                .enumerate()
            {
                if Instant::now() >= end {
                    break;
                }
                if closed(candidate.bounds[side], output.incumbent, options) {
                    estimates[side] = candidate.bounds[side];
                    continue;
                }
                candidate.stages[side] += 1;
                let mut branches = node.branches.clone();
                branches.push(constraint);
                let trial_started = Instant::now();
                let mut observation = None;
                output.strong_branching_trials += 1;
                if phase.restricted_master_only {
                    estimates[side] = restricted_value(parent, problem, &branches, pool, options)?;
                    observation = Some(history::EvaluationKind::RestrictedMaster);
                } else {
                    let result = evaluate(
                        parent,
                        None,
                        problem,
                        &branches,
                        pool,
                        options,
                        output.incumbent,
                        end.min(Instant::now() + config.trial_time_limit),
                        phase.max_iterations,
                        phase.pricing_phase_floor,
                        false,
                    );
                    match result {
                        Ok(result) => {
                            output.pricing_calls += result.master.pricing_calls;
                            output.fixing_calls += result.fixing_calls;
                            output.enumerated_routes += result.enumerated_routes;
                            let artificial_zero = result.master.artificials.iter().all(|&c| {
                                result.values.get(c).copied().unwrap_or(0.).abs() <= 1e-6
                            });
                            estimates[side] = if artificial_zero {
                                result.master.lp.min(1e12)
                            } else {
                                1e12
                            };
                            if artificial_zero {
                                if result.master.status == Status::Optimal
                                    && !result.master.pruned_by_bound
                                    && phase.pricing_phase_floor == 0
                                {
                                    observation = Some(history::EvaluationKind::ExactPricing);
                                } else if phase.pricing_phase_floor > 0 {
                                    observation = Some(history::EvaluationKind::HeuristicPricing);
                                }
                            }
                            if result.preprocessing.infeasible
                                || (result.master.status == Status::Infeasible
                                    && result.master.bound == f64::INFINITY)
                            {
                                candidate.bounds[side] = f64::INFINITY;
                            } else if result.master.bound.is_finite() {
                                candidate.bounds[side] =
                                    candidate.bounds[side].max(result.master.bound);
                            }
                            for route in &result.routes {
                                pool.add(problem, &route.arcs);
                            }
                            output.mip_nodes += integer_heuristic(
                                problem,
                                pool,
                                &mut output.incumbent,
                                &mut output.routes,
                                options,
                                end,
                            );
                        }
                        Err(e) if budget_error(&e) => {}
                        Err(e) => return Err(e),
                    }
                }
                if options.branch_history.is_some() {
                    if let Some(kind) = observation {
                        output.branch_history.record_at_stage(
                            &candidate.branch,
                            side,
                            node.branches.len(),
                            kind,
                            Some(candidate.stages[side]),
                            node.bound,
                            estimates[side],
                            trial_started.elapsed().as_secs_f64(),
                        );
                    }
                }
            }
            candidate.estimates = estimates;
            candidate.score = if phase.tree_depth_score {
                let (score, size) =
                    tree_metrics(node.bound, output.incumbent.min(1e12), estimates, 1.);
                if !phase.restricted_master_only
                    && phase.pricing_phase_floor == 0
                    && budget::exceeds(score, best_score)
                {
                    best_score = score;
                    subtree.size = size;
                }
                score
            } else {
                product_score(node.bound, output.incumbent.min(1e12), estimates)
            };
        }
        if evaluated > 0 {
            candidates.truncate(evaluated);
        }
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.order.cmp(&b.order)));
        if adaptive && evaluated > 0 && !phase.tree_depth_score {
            retain_distinct_trials(&mut candidates, |c| (c.score, c.estimates));
        }
    }
    Ok(candidates
        .first()
        .map(|c| (c.branch.clone(), c.bounds, subtree)))
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/tree/strong/trial_equivalence_tests.rs"]
mod trial_equivalence_tests;
