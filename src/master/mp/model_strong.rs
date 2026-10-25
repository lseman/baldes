//! Isolated phased evaluation of registered original-variable generators.
//! Values and signed history predict candidate quality; none are proof bounds.
use super::*;
use crate::master::{tree::strong, LpMaster};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MappedEvaluationKind {
    RestrictedMaster,
    HeuristicPricing,
    ExactPricing,
}

#[derive(Clone, Debug)]
pub struct MappedBranchEvaluation {
    pub name: String,
    pub depth: usize,
    pub side: usize,
    /// Number of evaluations actually run on this side at this node.
    pub stage: usize,
    pub kind: MappedEvaluationKind,
    pub parent: f64,
    pub estimate: f64,
    pub distance: f64,
}

#[derive(Clone, Copy, Default)]
struct RunningCost {
    count: usize,
    mean: f64,
}

#[derive(Default)]
pub(super) struct History {
    pub evaluations: Vec<MappedBranchEvaluation>,
    pub trials: usize,
    pub ranked: usize,
    costs: BTreeMap<String, BTreeMap<usize, [RunningCost; 2]>>,
}

impl History {
    fn record(&mut self, evaluation: MappedBranchEvaluation) {
        let cost = (evaluation.estimate - evaluation.parent) / evaluation.distance;
        if evaluation.parent.is_finite()
            && evaluation.estimate.is_finite()
            && evaluation.estimate < 1e12
            && evaluation.distance > 1e-6
            && cost.is_finite()
        {
            let stat = &mut self
                .costs
                .entry(evaluation.name.clone())
                .or_default()
                .entry(evaluation.stage)
                .or_default()[evaluation.side];
            stat.count += 1;
            stat.mean = ((stat.count - 1) as f64 * stat.mean + cost) / stat.count as f64;
            self.evaluations.push(evaluation);
        }
    }

    fn prediction(&self, name: &str, value: f64, gap: f64) -> Option<(usize, f64)> {
        let (&stage, stats) = self
            .costs
            .get(name)?
            .iter()
            .rev()
            .find(|(_, s)| s.iter().any(|v| v.count > 0))?;
        let fraction = value - value.floor();
        let score =
            (stats[0].mean * fraction).min(gap) * (stats[1].mean * (1. - fraction)).min(gap);
        score.is_finite().then_some((stage, score))
    }
}

struct Candidate<'a> {
    variable: &'a MappedBranchVariable,
    value: f64,
    score: f64,
    stages: [usize; 2],
    order: usize,
    subtree_size: f64,
}

pub(super) fn validate(options: &crate::master::tree::StrongBranchingOptions) -> Result<()> {
    if options.use_history_subtree_size {
        return Err(invalid("mapped strong branching uses the native parent subtree estimate; completed-subtree history is unavailable"));
    }
    if options.phases.is_empty()
        || options.phases.iter().any(|phase| {
            !phase.subtree_ratio.is_finite()
                || phase.subtree_ratio < 0.
                || phase.max_iterations == 0
        })
    {
        return Err(invalid(
            "mapped strong branching requires valid nonempty phases",
        ));
    }
    Ok(())
}

pub(super) fn choose<const D: usize>(
    output: &Output<D>,
    options: &AssignmentTreeOptions,
    history: &mut History,
    incumbent: f64,
    deadline: Instant,
) -> Result<Option<Vec<NodeSnapshot<D>>>> {
    let Some(config) = &options.mapped_strong_branching else {
        return model_mapping::children(output, &options.mapped_variables, options.tolerance);
    };
    let mut candidates: Vec<_> = options
        .mapped_variables
        .iter()
        .filter_map(|variable| {
            let value = variable.value(output)?;
            (value.is_finite()
                && value.abs() < 9_007_199_254_740_992.
                && (value - value.round()).abs() > options.tolerance)
                .then_some((variable, value))
        })
        .collect();
    candidates.sort_by(|(a, x), (b, y)| {
        b.priority
            .total_cmp(&a.priority)
            .then_with(|| {
                ((x - x.floor()) - 0.5)
                    .abs()
                    .total_cmp(&((y - y.floor()) - 0.5).abs())
            })
            .then_with(|| a.name.cmp(&b.name))
    });
    let parent = output.master.bound;
    let cutoff = incumbent.min(1e12);
    let gap = cutoff - parent;
    let limit = config.phases.first().map_or(0, |p| p.candidates);
    // Native history is attached to generator identities and refreshed values,
    // not the old fractional split. Reserve at most half the admission slots.
    if output.snapshot.state.node_depth > 0 && gap > 0. {
        let mut ranked: Vec<_> = candidates
            .iter()
            .enumerate()
            .filter_map(|(i, (v, x))| {
                history
                    .prediction(&v.name, *x, gap)
                    .map(|(stage, score)| (i, stage, score, &v.name))
            })
            .collect();
        ranked.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then_with(|| b.2.total_cmp(&a.2))
                .then_with(|| a.3.cmp(b.3))
        });
        ranked.truncate(limit / 2);
        let selected: HashSet<_> = ranked.iter().map(|r| r.0).collect();
        let mut next: Vec<_> = ranked.iter().map(|r| candidates[r.0]).collect();
        next.extend(
            candidates
                .into_iter()
                .enumerate()
                .filter(|(i, _)| !selected.contains(i))
                .map(|(_, c)| c),
        );
        history.ranked += ranked.len();
        candidates = next;
    }
    let mut candidates: Vec<_> = candidates
        .into_iter()
        .enumerate()
        .map(|(order, (variable, value))| Candidate {
            variable,
            value,
            score: 0.,
            stages: [0; 2],
            order,
            subtree_size: 1e12,
        })
        .collect();
    let end = deadline.min(
        Instant::now()
            .checked_add(config.time_limit)
            .unwrap_or(deadline),
    );
    let inherited_size = output.snapshot.state.mapped_subtree_size;
    let mut best_size: f64 = 1e12;
    let mut best_score = -1e12;
    for (phase_index, phase) in config.phases.iter().enumerate() {
        candidates.truncate(phase.candidates);
        if candidates.is_empty() || Instant::now() >= end {
            break;
        }
        let exact = !phase.restricted_master_only && phase.pricing_phase_floor == 0;
        if !exact
            && config
                .phases
                .get(phase_index + 1)
                .is_some_and(|next| next.candidates > 0 && candidates.len() <= next.candidates)
        {
            continue;
        }
        let mut evaluated = 0;
        for candidate in &mut candidates {
            if phase.subtree_ratio > 0.
                && exceeds(
                    evaluated as f64,
                    phase.subtree_ratio * inherited_size.min(best_size),
                )
            {
                break;
            }
            if Instant::now() >= end {
                break;
            }
            evaluated += 1;
            let mut estimates = [parent; 2];
            for (side, bounds) in [
                crate::master::preprocessing::Bounds {
                    lower: f64::NEG_INFINITY,
                    upper: candidate.value.floor(),
                },
                crate::master::preprocessing::Bounds {
                    lower: candidate.value.ceil(),
                    upper: f64::INFINITY,
                },
            ]
            .into_iter()
            .enumerate()
            {
                if Instant::now() >= end {
                    break;
                }
                let mut child = candidate.variable.apply(&output.snapshot, bounds)?;
                child.state.node_depth = output.snapshot.state.node_depth.saturating_add(1);
                candidate.stages[side] += 1;
                history.trials += 1;
                let remaining = end
                    .saturating_duration_since(Instant::now())
                    .min(config.trial_time_limit);
                let observation = if phase.restricted_master_only {
                    // This model includes original/artificial columns but no
                    // private proximal auxiliaries. Its objective ranks only.
                    let solved = child.master.solve_limited(i32::MAX as usize, remaining)?;
                    if solved == crate::master::clp::ClpSolveOutcome::Optimal {
                        estimates[side] = child.master.objective().min(1e12);
                        Some(MappedEvaluationKind::RestrictedMaster)
                    } else {
                        None
                    }
                } else {
                    let mut node = options.node.clone();
                    node.enumeration = None;
                    node.cg.max_iterations = phase.max_iterations;
                    node.cg.time_limit = remaining;
                    // Install all caller controls before trimming the private
                    // schedule; no solve-time selector may restore phase zero.
                    for oracle in &mut child.state.oracles {
                        for (name, value) in &node.pricing_options {
                            oracle.solver.set_option(name, *value)?;
                        }
                        if phase.pricing_phase_floor >= oracle.solver.phases.len() {
                            return Err(invalid("mapped strong branching pricing phase floor"));
                        }
                        oracle.solver.phases =
                            oracle.solver.phases[phase.pricing_phase_floor..].to_vec();
                    }
                    node.pricing_options.clear();
                    // The deadline inside the node bounds each labeling call.
                    child.state.deadline = Instant::now()
                        .checked_add(remaining)
                        .unwrap_or(end)
                        .min(end);
                    match child.solve(node) {
                        Ok(result) => {
                            estimates[side] = result.master.lp.min(1e12);
                            let artificial_zero = result.master.artificials.iter().all(|&c| {
                                result
                                    .snapshot
                                    .master
                                    .values
                                    .get(c)
                                    .copied()
                                    .unwrap_or(0.)
                                    .abs()
                                    <= 1e-6
                            });
                            if artificial_zero && estimates[side].is_finite() {
                                if exact && result.master.status == cg::Status::Optimal {
                                    Some(MappedEvaluationKind::ExactPricing)
                                } else if phase.pricing_phase_floor > 0 {
                                    Some(MappedEvaluationKind::HeuristicPricing)
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        }
                        Err(error)
                            if error.message.contains("time limit")
                                || error.message.contains("label limit") =>
                        {
                            None
                        }
                        Err(error) => return Err(error),
                    }
                };
                if let Some(kind) = observation {
                    let fraction = candidate.value - candidate.value.floor();
                    history.record(MappedBranchEvaluation {
                        name: candidate.variable.name.clone(),
                        depth: output.snapshot.state.node_depth,
                        side,
                        stage: candidate.stages[side],
                        kind,
                        parent,
                        estimate: estimates[side],
                        distance: if side == 0 { fraction } else { 1. - fraction },
                    });
                }
            }
            candidate.score = if phase.tree_depth_score {
                strong::tree_depth_score(parent, cutoff, estimates, 1.)
            } else {
                strong::product_score(parent, cutoff, estimates)
            };
            if phase.tree_depth_score && exact && exceeds(candidate.score, best_score) {
                best_score = candidate.score;
                let depth = strong::tree_depth_score(parent, cutoff, estimates, gap);
                let size = 2. * (2f64.powf(-depth) - 1.);
                candidate.subtree_size = if size.is_finite() && size >= 0. {
                    size.min(1e12)
                } else {
                    1e12
                };
                best_size = candidate.subtree_size;
            }
        }
        if evaluated > 0 {
            candidates.truncate(evaluated);
        }
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.order.cmp(&b.order)));
    }
    let Some(candidate) = candidates.first() else {
        return Ok(None);
    };
    // Recreate unpriced children of the original checkpoint. Trial estimates
    // and artificial objectives never enter their inherited lower bounds.
    let mut children =
        model_mapping::children(output, &[candidate.variable.clone()], options.tolerance)?;
    if let Some(children) = &mut children {
        for child in children {
            child.state.mapped_subtree_size = candidate.subtree_size;
        }
    }
    Ok(children)
}

fn exceeds(a: f64, b: f64) -> bool {
    a - (1e-6 + a.abs().max(b.abs()) * 1e-10) > b
}

#[cfg(test)]
#[path = "../../../tests/unit/master/mp/model_strong_tests.rs"]
mod tests;
