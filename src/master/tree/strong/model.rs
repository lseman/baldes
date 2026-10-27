//! Isolated phased evaluation of registered original-variable generators.
//! Values and signed history predict candidate quality; none are proof bounds.
use super::*;
use crate::master::{tree::strong, LpMaster};

/// Original-domain phase-one proof, also required by ordinary tree closure.
pub(super) fn proven_infeasible<const D: usize>(output: &Output<D>) -> bool {
    output.master.status == cg::Status::Infeasible
        && output
            .master
            .feasibility
            .iter()
            .any(|r| r.certified_infeasible)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MappedEvaluationKind {
    RestrictedMaster,
    HeuristicPricing,
    ExactPricing,
}

#[derive(Clone, Debug, serde::Serialize)]
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

type IntegerWitness = (f64, Vec<FixedRoute>, Vec<(i32, f64)>);

#[derive(Default)]
pub(super) struct History {
    pub column_pool: column_pool::Pool,
    pub evaluations: Vec<MappedBranchEvaluation>,
    pub trials: usize,
    /// Native processed-node work excludes restricted/heuristic screens.
    pub processed_trials: usize,
    pub ranked: usize,
    pub pricing_calls: usize,
    pub iterations: usize,
    pub oracle_calls: BTreeMap<i32, usize>,
    pub completed_estimate: Option<f64>,
    // Original-domain feasible solutions survive trial isolation; numerical
    // trial objectives and warm bounds remain ranking data only.
    pub incumbent: Option<IntegerWitness>,
    resources: BTreeMap<String, ResourceGroupCandidate>,
    costs: BTreeMap<String, BTreeMap<usize, [RunningCost; 2]>>,
}

impl History {
    fn retain_witness<const D: usize>(
        &mut self,
        node: &NodeSnapshot<D>,
        tolerance: f64,
        incumbent: f64,
    ) -> Result<()> {
        if let Some(witness) = node.integer_witness(tolerance)? {
            if witness.0 < incumbent
                && self
                    .incumbent
                    .as_ref()
                    .map_or(true, |best| witness.0 < best.0)
            {
                self.incumbent = Some(witness);
            }
        }
        Ok(())
    }

    fn record_with_key(&mut self, key: &str, evaluation: MappedBranchEvaluation) {
        let cost = (evaluation.estimate - evaluation.parent) / evaluation.distance;
        if evaluation.parent.is_finite()
            && evaluation.estimate.is_finite()
            && evaluation.estimate < 1e12
            && evaluation.distance > 1e-6
            && cost.is_finite()
        {
            let stat = &mut self
                .costs
                .entry(key.to_owned())
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

/// Lightweight original-domain restriction; full masters are forked only after
/// native priority/quota admission, never for the entire physical-arc registry.
pub(super) enum Restriction {
    Mapped(MappedBranchVariable),
    Resource(ResourceGroupCandidate),
    Assignment { shared_row: usize, network: i32 },
    Master { id: i32 },
    Multiplicity { network: i32 },
    Activity { network: i32, arcs: Vec<(i32, f64)> },
}

pub(super) struct Proposal {
    pub name: String,
    pub history_key: String,
    pub priority: f64,
    pub value: f64,
    pub restriction: Restriction,
    pub history_eligible: bool,
    pub fresh: bool,
}
impl Proposal {
    fn children<const D: usize>(&self, node: &NodeSnapshot<D>) -> Result<Vec<NodeSnapshot<D>>> {
        use crate::master::preprocessing::Bounds;
        let bounds = [
            Bounds {
                lower: f64::NEG_INFINITY,
                upper: self.value.floor(),
            },
            Bounds {
                lower: self.value.ceil(),
                upper: f64::INFINITY,
            },
        ];
        match &self.restriction {
            Restriction::Mapped(variable) => bounds
                .into_iter()
                .map(|b| variable.apply(node, b))
                .collect(),
            Restriction::Resource(candidate) => [false, true]
                .into_iter()
                .map(|up| candidate.apply(node, up))
                .collect(),
            Restriction::Assignment {
                shared_row,
                network,
            } => [false, true]
                .into_iter()
                .map(|up| node.branch_assignment(*shared_row, *network, up))
                .collect(),
            Restriction::Master { id } => bounds
                .into_iter()
                .map(|b| node.with_master_variable_bounds(*id, b))
                .collect(),
            Restriction::Multiplicity { network } => bounds
                .into_iter()
                .map(|mut b| {
                    b.lower = b.lower.max(0.);
                    node.with_multiplicity(*network, b)
                })
                .collect(),
            Restriction::Activity { network, arcs } => {
                let oracle = node
                    .state
                    .oracles
                    .iter()
                    .find(|o| o.network.id == *network)
                    .ok_or_else(|| invalid("unknown proposal activity network"))?;
                let mut coefficients = vec![0.; oracle.network.graph.arcs.len()];
                for &(id, coefficient) in arcs {
                    let &position = oracle
                        .index
                        .get(&id)
                        .ok_or_else(|| invalid("unknown proposal activity arc"))?;
                    coefficients[position] = coefficient;
                }
                let coefficients = [RowCoefficients {
                    network: *network,
                    arcs: coefficients,
                    per_route: 0.,
                }];
                bounds
                    .into_iter()
                    .map(|mut b| {
                        b.lower = b.lower.max(0.);
                        node.with_activity_bounds(b, &coefficients)
                    })
                    .collect()
            }
        }
    }
    fn unsupported_resource(&self) -> bool {
        matches!(self.restriction, Restriction::Resource(_))
    }
}

struct Candidate<const D: usize> {
    proposal: Proposal,
    children: Option<Vec<NodeSnapshot<D>>>,
    usable: bool,
    score: f64,
    stages: [usize; 2],
    order: usize,
    subtree_size: f64,
    evaluated: [Option<Output<D>>; 2],
    estimates: [f64; 2],
    // Bounds published by the exact-pricing driver, independently of whether
    // that trial completed. Restricted/heuristic LP estimates never enter.
    proof_bounds: [(f64, f64); 2],
}

pub(super) struct Selection<const D: usize> {
    pub children: Vec<NodeSnapshot<D>>,
    pub evaluated: [Option<Output<D>>; 2],
    pub direction: Option<NetworkBranchDirection>,
}

impl<const D: usize> Selection<D> {
    pub fn fresh(children: Vec<NodeSnapshot<D>>) -> Self {
        Self {
            children,
            evaluated: [None, None],
            direction: None,
        }
    }

    pub fn reverse(&mut self) {
        self.children.reverse();
        self.evaluated.reverse();
    }

    pub fn admit(&mut self, default: NetworkBranchDirection) {
        if self.direction.unwrap_or(default) == NetworkBranchDirection::UpperFirst {
            self.reverse();
        }
    }
}

pub(super) fn mapped_proposals<const D: usize>(
    output: &Output<D>,
    variables: &[MappedBranchVariable],
    tolerance: f64,
) -> Result<Vec<Proposal>> {
    let mut proposals = vec![];
    for variable in variables {
        let Some(value) = variable.value(output) else {
            continue;
        };
        if !value.is_finite()
            || value.abs() >= 9_007_199_254_740_992.
            || (value - value.round()).abs() <= tolerance
        {
            continue;
        }
        proposals.push(Proposal {
            name: variable.name.clone(),
            history_key: format!("mapped:{}", variable.name),
            priority: variable.priority,
            value,
            restriction: Restriction::Mapped(variable.clone()),
            history_eligible: true,
            fresh: true,
        });
    }
    Ok(proposals)
}

pub(super) fn validate(options: &crate::master::tree::StrongBranchingOptions) -> Result<()> {
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
    options: &ModelTreeOptions,
    history: &mut History,
    incumbent: f64,
    deadline: Instant,
) -> Result<Option<Selection<D>>> {
    let proposals = mapped_proposals(output, &options.mapped_variables, options.tolerance)?;
    choose_proposals(output, options, history, incumbent, deadline, proposals)
}

pub(super) fn choose_proposals<const D: usize>(
    output: &Output<D>,
    options: &ModelTreeOptions,
    history: &mut History,
    incumbent: f64,
    deadline: Instant,
    mut proposals: Vec<Proposal>,
) -> Result<Option<Selection<D>>> {
    if options.branching == NetworkBranchingPolicy::Phased && output.snapshot.state.node_depth > 0 {
        let present: HashSet<_> = proposals.iter().map(|p| p.history_key.clone()).collect();
        for (key, template) in &history.resources {
            if present.contains(key) {
                continue;
            }
            if let Some(proposal) = super::tree_adapter::proposals::refresh_resource(
                output,
                template,
                options.branch_priorities.resource,
                options.tolerance,
            ) {
                proposals.push(proposal);
            }
        }
    }
    if proposals
        .iter()
        .any(|p| !p.priority.is_finite() || !p.value.is_finite())
    {
        return Err(invalid("invalid network branch proposal"));
    }
    proposals.sort_by(|a, b| {
        b.priority
            .total_cmp(&a.priority)
            .then_with(|| {
                ((a.value - a.value.floor()) - 0.5)
                    .abs()
                    .total_cmp(&((b.value - b.value.floor()) - 0.5).abs())
            })
            .then_with(|| {
                options
                    .packing
                    .as_ref()
                    .map_or(std::cmp::Ordering::Equal, |packing| {
                        packing.tie_key(&a.name).cmp(&packing.tie_key(&b.name))
                    })
            })
            .then_with(|| a.name.cmp(&b.name))
    });
    // Multiplicity generators are visited in pricing registration order,
    // including unequal fractional counts. Keep their priority-family slots
    // and the ordering of every other generator unchanged.
    let mut order: Vec<_> = (0..proposals.len()).collect();
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (index, proposal) in proposals.iter().enumerate() {
        if matches!(proposal.restriction, Restriction::Multiplicity { .. }) {
            let first = proposals
                .iter()
                .position(|p| p.priority == proposal.priority)
                .unwrap();
            groups.entry(first).or_default().push(index);
        }
    }
    for slots in groups.values() {
        let mut registered = slots.clone();
        registered.sort_by_key(|&index| match proposals[index].restriction {
            Restriction::Multiplicity { network } => output
                .snapshot
                .state
                .oracles
                .iter()
                .position(|o| o.network.id == network)
                .unwrap_or(usize::MAX),
            _ => unreachable!(),
        });
        for (&slot, index) in slots.iter().zip(registered) {
            order[slot] = index;
        }
    }
    let mut slots: Vec<_> = proposals.into_iter().map(Some).collect();
    let mut proposals: Vec<_> = order
        .into_iter()
        .map(|i| slots[i].take().unwrap())
        .collect();
    let Some(config) = &options.mapped_strong_branching else {
        for proposal in proposals {
            match proposal.children(&output.snapshot) {
                Ok(children) => {
                    let mut selection = Selection::fresh(children);
                    selection.direction = options.branch_directions.get(&proposal.name).copied();
                    return Ok(Some(selection));
                }
                Err(_) if proposal.unsupported_resource() => continue,
                Err(error) => return Err(error),
            }
        }
        return Ok(None);
    };
    // Native scoring receives the ordinary LP dual channel, independently
    // of the selected certificate used for pruning and publication.
    let parent = super::tree_adapter::queue_bound(output.master.numerical_bound);
    if !options.scoring_gap_tolerance.is_finite() || options.scoring_gap_tolerance < 0. {
        return Err(invalid("scoring gap tolerance"));
    }
    // Generic MP objectives use native's continuous minimization tag. Its
    // scoring cutoff is half an absolute-gap tolerance below the incumbent.
    let cutoff = (incumbent - options.scoring_gap_tolerance * 0.5).min(1e12);
    let gap = cutoff - parent;
    let inherited_size = history
        .completed_estimate
        .unwrap_or(output.snapshot.state.mapped_subtree_size);
    let mut budget = strong::CandidateBudget::new(
        config.phases.first().map_or(0, |p| p.candidates),
        config.phases.first().map_or(0., |p| p.subtree_ratio),
        strong::SubtreeEstimate {
            size: inherited_size,
            priority: output.snapshot.state.mapped_subtree_priority,
        },
    );
    if history.completed_estimate.is_some() {
        budget.use_history();
    }
    let mut priorities: Vec<_> = options
        .mapped_variables
        .iter()
        .map(|v| v.priority)
        .collect();
    if let Some(packing) = &options.packing {
        priorities.push(packing.priority);
    }
    if options.branching == NetworkBranchingPolicy::Phased {
        let p = &options.branch_priorities;
        priorities.extend([
            p.multiplicity,
            p.resource,
            p.assignment,
            p.master_variable,
            p.arc,
            p.window,
        ]);
    }
    priorities.sort_by(|a, b| b.total_cmp(a));
    priorities.dedup_by(|a, b| *a == *b);
    let (fresh, priority_cutoff) = crate::master::tree::history::fresh_batches_with_budget(
        budget.quota,
        &priorities,
        Some(&mut budget),
        |priority| {
            proposals
                .iter()
                .enumerate()
                .filter_map(|(i, p)| (p.fresh && p.priority == priority).then_some(i))
                .collect()
        },
    );
    let limit = budget.quota;
    // Native history admission uses the actual fresh-loop cutoff, including
    // empty registered priority families, and reserves at most half the quota.
    let mut ranked = vec![];
    if output.snapshot.state.node_depth > 0 && gap > 0. {
        ranked = proposals
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                if !p.history_eligible
                    || priority_cutoff
                        - (1e-6 + p.priority.abs().max(priority_cutoff.abs()) * 1e-10)
                        > p.priority
                {
                    return None;
                }
                history
                    .prediction(&p.history_key, p.value, gap)
                    .map(|(stage, score)| (i, stage, score, p.name.clone()))
            })
            .collect();
        ranked.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then_with(|| b.2.total_cmp(&a.2))
                .then_with(|| a.3.cmp(&b.3))
        });
        ranked.truncate(limit / 2);
    }
    history.ranked += ranked.len();
    let mut slots: Vec<_> = proposals.into_iter().map(Some).collect();
    let mut next: Vec<_> = ranked.iter().filter_map(|r| slots[r.0].take()).collect();
    next.extend(fresh.into_iter().filter_map(|i| slots[i].take()));
    proposals = next;
    proposals.truncate(limit);
    let mut candidates: Vec<_> = proposals
        .into_iter()
        .enumerate()
        .map(|(order, proposal)| Candidate {
            proposal,
            children: None,
            usable: true,
            score: 0.,
            stages: [0; 2],
            order,
            subtree_size: 1e12,
            evaluated: [None, None],
            estimates: [1e12; 2],
            proof_bounds: [(
                output.snapshot.warm.numerical_bound,
                output.snapshot.warm.certified_bound,
            ); 2],
        })
        .collect();
    let end = deadline.min(
        Instant::now()
            .checked_add(config.time_limit)
            .unwrap_or(deadline),
    );
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
            if !budget.permits(evaluated, phase.subtree_ratio, best_size) {
                break;
            }
            if Instant::now() >= end {
                break;
            }
            if candidate.children.is_none() {
                match candidate.proposal.children(&output.snapshot) {
                    Ok(children) => candidate.children = Some(children),
                    Err(_) if candidate.proposal.unsupported_resource() => {
                        candidate.usable = false;
                        candidate.score = f64::NEG_INFINITY;
                        continue;
                    }
                    Err(error) => return Err(error),
                }
            }
            evaluated += 1;
            let mut estimates = candidate.estimates;
            // Native generator direction applies to trial construction too,
            // not just final frontier insertion. Canonical side indices stay
            // tied to lower/upper history distances and proof bounds.
            let direction = options
                .branch_directions
                .get(&candidate.proposal.name)
                .copied()
                .unwrap_or_else(|| {
                    if options.branching == NetworkBranchingPolicy::Phased
                        || matches!(
                            candidate.proposal.restriction,
                            Restriction::Multiplicity { .. }
                        )
                    {
                        NetworkBranchDirection::UpperFirst
                    } else {
                        NetworkBranchDirection::LowerFirst
                    }
                });
            let sides = direction.sides();
            // Native admits a candidate's whole child group. An elapsed
            // deadline leaves the sibling a zero-budget attempt, rather than
            // removing it from trial accounting or branch evaluation.
            for side in sides {
                let mut child = candidate.children.as_ref().unwrap()[side].fork()?;
                if options.share_inactive_columns {
                    history.column_pool.absorb(&output.snapshot);
                    history.column_pool.seed(&mut child, end)?;
                }
                child.state.node_depth = output.snapshot.state.node_depth.saturating_add(1);
                candidate.stages[side] += 1;
                history.trials += 1;
                history.processed_trials += usize::from(exact);
                let remaining = end
                    .saturating_duration_since(Instant::now())
                    .min(config.trial_time_limit);
                let observation = if phase.restricted_master_only {
                    // This model includes original/artificial columns but no
                    // private proximal auxiliaries. Its objective ranks only.
                    let solved = child.master.solve_limited(i32::MAX as usize, remaining)?;
                    let observation = if solved == crate::master::clp::ClpSolveOutcome::Optimal {
                        history.retain_witness(&child, options.tolerance, incumbent)?;
                        estimates[side] = child.master.objective().min(1e12);
                        Some(MappedEvaluationKind::RestrictedMaster)
                    } else {
                        None
                    };
                    candidate.children.as_mut().unwrap()[side] = child;
                    observation
                } else {
                    let mut node = options.node.clone();
                    node.search_inactive_columns |= options.share_inactive_columns;
                    node.enumeration = None;
                    node.cg.max_iterations = phase.max_iterations;
                    node.cg.time_limit = remaining;
                    // Install all caller controls before trimming the private
                    // schedule; no solve-time selector may restore phase zero.
                    for oracle in &mut child.state.oracles {
                        // A surviving workspace may retain the preceding
                        // heuristic slice. Restore its original full schedule
                        // before applying this phase's private floor.
                        oracle.solver.phases = output
                            .snapshot
                            .state
                            .oracles
                            .iter()
                            .find(|parent| parent.network.id == oracle.network.id)
                            .ok_or_else(|| invalid("missing parent trial pricing schedule"))?
                            .solver
                            .phases
                            .clone();
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
                            if options.share_inactive_columns {
                                history.column_pool.absorb(&result.snapshot);
                            }
                            history.retain_witness(
                                &result.snapshot,
                                options.tolerance,
                                incumbent,
                            )?;
                            history.pricing_calls += result.master.pricing_calls;
                            history.iterations += result.master.iterations;
                            for &(id, calls) in &result.oracle_calls {
                                *history.oracle_calls.entry(id).or_default() += calls;
                            }
                            // As in the recovered routing policy, an incomplete
                            // original LP value can rank a trial; it is never a
                            // child bound, and incomplete exact trials are not
                            // recorded as exact history.
                            // No LP was attempted in a zero-budget sibling;
                            // CG's default zero objective is not an estimate.
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
                            if result.master.iterations > 0 {
                                // An unresolved artificial solution uses the
                                // reference's ranking sentinel. Its arbitrary
                                // penalty magnitude must not choose a generator.
                                estimates[side] = if artificial_zero {
                                    result.master.lp.min(1e12)
                                } else {
                                    1e12
                                };
                            }
                            let observation = if artificial_zero && estimates[side].is_finite() {
                                if exact && result.master.status == cg::Status::Optimal {
                                    Some(MappedEvaluationKind::ExactPricing)
                                } else if phase.pricing_phase_floor > 0 {
                                    Some(MappedEvaluationKind::HeuristicPricing)
                                } else {
                                    None
                                }
                            } else {
                                None
                            };
                            if exact && !proven_infeasible(&result) {
                                // CG publishes each channel only from its own
                                // completed pricing/proof passes. An iteration
                                // limit does not invalidate those earlier bounds.
                                let bounds = &mut candidate.proof_bounds[side];
                                bounds.0 = bounds.0.max(result.master.numerical_bound);
                                bounds.1 = bounds.1.max(result.master.certified_bound);
                            }
                            if observation == Some(MappedEvaluationKind::ExactPricing)
                                || exact && proven_infeasible(&result)
                            {
                                if phase_index + 1 < config.phases.len() {
                                    candidate.children.as_mut().unwrap()[side] =
                                        result.snapshot.fork()?;
                                }
                                // Reuse the solved original model and its selected
                                // bound channel, never the ranking estimate above.
                                candidate.evaluated[side] = Some(result);
                            } else {
                                // Selected children retain their setup even if
                                // the last phase stops before exact completion.
                                candidate.children.as_mut().unwrap()[side] = result.snapshot;
                            }
                            observation
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
                if let Some(kind) = observation.filter(|_| candidate.proposal.history_eligible) {
                    let fraction = candidate.proposal.value - candidate.proposal.value.floor();
                    if let Restriction::Resource(template) = &candidate.proposal.restriction {
                        history
                            .resources
                            .entry(candidate.proposal.history_key.clone())
                            .or_insert_with(|| template.clone());
                    }
                    history.record_with_key(
                        &candidate.proposal.history_key,
                        MappedBranchEvaluation {
                            name: candidate.proposal.name.clone(),
                            depth: output.snapshot.state.node_depth,
                            side,
                            stage: candidate.stages[side],
                            kind,
                            parent,
                            estimate: estimates[side],
                            distance: if side == 0 { fraction } else { 1. - fraction },
                        },
                    );
                }
            }
            candidate.estimates = estimates;
            // Trial witnesses are published only after this pass returns,
            // as in native; a child's local primal value is not the parent's
            // shared cutoff for this pass.
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
        candidates.retain(|c| c.usable);
        if evaluated > 0 {
            candidates.truncate(evaluated);
        }
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.order.cmp(&b.order)));
        if evaluated > 0 && !phase.tree_depth_score {
            strong::retain_distinct_trials(&mut candidates, |c| (c.score, c.estimates));
        }
    }
    for mut candidate in candidates.into_iter().filter(|c| c.usable) {
        let mut children = if let Some(children) = candidate.children {
            children
        } else {
            match candidate.proposal.children(&output.snapshot) {
                Ok(children) => children,
                Err(_) if candidate.proposal.unsupported_resource() => continue,
                Err(error) => return Err(error),
            }
        };
        for (side, child) in children.iter_mut().enumerate() {
            if let Some(evaluated) = &mut candidate.evaluated[side] {
                evaluated.snapshot.state.mapped_subtree_size = candidate.subtree_size;
                evaluated.snapshot.state.mapped_subtree_priority = priority_cutoff;
                *child = evaluated.snapshot.fork()?;
            } else {
                // Carry the setup and independently published proof channels,
                // without treating an unfinished output as a completed solve.
                // A heuristic/RMP objective can rank but cannot replace these.
                let (numerical, certified) = candidate.proof_bounds[side];
                child.warm.bound_mode = output.snapshot.warm.bound_mode;
                child.warm.numerical_bound = numerical;
                child.warm.certified_bound = certified;
                child.warm.bound = match child.warm.bound_mode {
                    cg::BoundMode::Numerical => numerical,
                    cg::BoundMode::Certified => certified,
                };
                child.state.phase_one_infeasible = false;
                for oracle in &mut child.state.oracles {
                    oracle.solver.phases = output
                        .snapshot
                        .state
                        .oracles
                        .iter()
                        .find(|parent| parent.network.id == oracle.network.id)
                        .ok_or_else(|| invalid("missing parent trial pricing schedule"))?
                        .solver
                        .phases
                        .clone();
                }
            }
            child.state.mapped_subtree_size = candidate.subtree_size;
            child.state.mapped_subtree_priority = priority_cutoff;
        }
        return Ok(Some(Selection {
            children,
            evaluated: candidate.evaluated,
            direction: options
                .branch_directions
                .get(&candidate.proposal.name)
                .copied(),
        }));
    }
    Ok(None)
}

fn exceeds(a: f64, b: f64) -> bool {
    a - (1e-6 + a.abs().max(b.abs()) * 1e-10) > b
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/tree/strong/model_tests.rs"]
mod tests;
