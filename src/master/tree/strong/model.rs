//! Isolated phased evaluation of registered original-variable generators.
//! Values and signed history predict candidate quality; none are proof bounds.
use super::*;
use crate::master::tree::strong;

/// Original-domain phase-one proof, also required by ordinary tree closure.
pub(super) fn proven_infeasible<const D: usize>(output: &Output<D>) -> bool {
    output.master.status == cg::Status::Infeasible
        && (output.snapshot.state.phase_one_infeasible
            || output.master.feasibility.iter().any(|r| {
                r.certified_infeasible
                    || (output.master.bound_mode == cg::BoundMode::Numerical
                        && r.status == cg::FeasibilityStatus::Infeasible)
            }))
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
    /// Treatment stage, including ordinary treatment after screening.
    pub stage: usize,
    /// Treatment kind; this does not imply convergence or a reusable result.
    pub kind: MappedEvaluationKind,
    pub parent: f64,
    /// Raw trial LP ranking value. Pseudocost means separately cap this value
    /// at the child's incumbent with numerical slack; neither is a proof bound.
    pub estimate: f64,
    pub distance: f64,
}

#[derive(Clone, Copy, Default)]
struct RunningCost {
    count: usize,
    mean: f64,
}

type IntegerWitness = (f64, Vec<FixedRoute>, Vec<(i32, f64)>);

/// Original branch identity carried to an ordinary child treatment.
#[derive(Clone, Debug)]
pub(super) struct BranchHistoryOrigin {
    key: String,
    name: String,
    depth: usize,
    side: usize,
    stage: usize,
    parent: f64,
    distance: f64,
}

#[derive(Default)]
pub(super) struct History {
    pub column_pool: column_pool::Pool,
    pub incidence_history: Option<crate::cuts::IncidenceCutSeparator>,
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
    templates: BTreeMap<String, HistoryTemplate>,
    costs: BTreeMap<String, BTreeMap<usize, [RunningCost; 2]>>,
}

enum HistoryTemplate {
    Resource(ResourceGroupCandidate),
    PackingAssignment(String),
}

impl History {
    fn remember_proposal(&mut self, proposal: &Proposal, options: &ModelTreeOptions) {
        if !proposal.history_eligible || self.templates.contains_key(&proposal.history_key) {
            return;
        }
        let template = match &proposal.restriction {
            Restriction::Resource(template) => Some(HistoryTemplate::Resource(template.clone())),
            Restriction::Mapped(_)
                if proposal.origin == Some(ProducerFamily::PackingAssignment)
                    && options
                        .branching_registry()
                        .packing
                        .as_ref()
                        .is_some_and(|packing| packing.tie_key(&proposal.name).0 == 0) =>
            {
                Some(HistoryTemplate::PackingAssignment(proposal.name.clone()))
            }
            _ => None,
        };
        if let Some(template) = template {
            self.templates
                .insert(proposal.history_key.clone(), template);
        }
    }

    pub(super) fn record_committed<const D: usize>(
        &mut self,
        output: &mut Output<D>,
        incumbent: f64,
        reused: bool,
        tolerance: f64,
    ) -> Result<()> {
        let Some(origin) = output.snapshot.state.branch_history_origin.take() else {
            return Ok(());
        };
        let timed_out = output.master.status == cg::Status::TimeLimit;
        if reused || (output.master.iterations == 0 && !timed_out) || proven_infeasible(output) {
            return Ok(());
        }
        let artificial_zero = output.master.artificials.iter().all(|&column| {
            output
                .snapshot
                .master
                .values
                .get(column)
                .copied()
                .unwrap_or(0.)
                .abs()
                <= 1e-6
        });
        // Ordinary iteration-limited treatment records its finite penalized LP
        // in native history even while artificials remain. This is ranking
        // data only; completed infeasibility above still supplies no observation.
        let iteration_limited = output.master.status == cg::Status::IterationLimit;
        if !timed_out
            && !iteration_limited
            && (!artificial_zero || output.master.status != cg::Status::Optimal)
        {
            return Ok(());
        }
        let local_incumbent = if timed_out {
            incumbent
        } else {
            output
                .snapshot
                .integer_witness(tolerance)?
                .map_or(incumbent, |witness| incumbent.min(witness.0))
        };
        // Interrupted native treatment keeps its unavailable-LP sentinel even
        // after a partial LP iteration. CG's placeholder or partial penalty
        // objective remains unchanged and never becomes a history proof bound.
        let estimate = if timed_out {
            1e12
        } else {
            output.master.lp.min(1e12)
        };
        self.record_with_key(
            &origin.key,
            MappedBranchEvaluation {
                name: origin.name,
                depth: origin.depth,
                side: origin.side,
                stage: origin.stage,
                kind: MappedEvaluationKind::ExactPricing,
                parent: origin.parent,
                estimate,
                distance: origin.distance,
            },
            local_incumbent,
        );
        Ok(())
    }

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

    fn record_with_key(&mut self, key: &str, evaluation: MappedBranchEvaluation, incumbent: f64) {
        // A trial's original LP may already exceed its local incumbent. Its
        // pseudocost records only the improvement needed to close that gap,
        // with the usual numerical comparison band. Keep the raw LP estimate
        // in the diagnostic trace; neither value is a proof bound.
        let estimate = if exceeds(evaluation.estimate, incumbent) {
            incumbent
        } else {
            evaluation.estimate
        };
        let cost = (estimate - evaluation.parent) / evaluation.distance;
        if evaluation.parent.is_finite()
            && evaluation.estimate.is_finite()
            && !incumbent.is_nan()
            && incumbent != f64::NEG_INFINITY
            // Tolerant aggregate endpoints can put the stored fractional
            // scalar outside [0, 1]. Its signed finite cost still participates
            // in history; distance is not a feasibility or pruning test.
            && evaluation.distance.is_finite()
            && evaluation.distance != 0.
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
        self.prediction_from_fraction(name, value - value.floor(), gap)
    }

    fn prediction_from_fraction(
        &self,
        name: &str,
        fraction: f64,
        gap: f64,
    ) -> Option<(usize, f64)> {
        let (&stage, stats) = self
            .costs
            .get(name)?
            .iter()
            .rev()
            .find(|(_, s)| s.iter().any(|v| v.count > 0))?;
        let score =
            (stats[0].mean * fraction).min(gap) * (stats[1].mean * (1. - fraction)).min(gap);
        score.is_finite().then_some((stage, score))
    }
}

/// Lightweight original-domain restriction; full masters are forked only after
/// native priority/quota admission, never for the entire physical-arc registry.
pub(super) enum Restriction {
    Component(ComponentClassBranching),
    GeneratedComponent {
        class: ComponentClassBranching,
        ordinal: usize,
    },
    Mapped(MappedBranchVariable),
    Aggregate {
        variable: MappedBranchVariable,
        fixed_activity: f64,
    },
    Resource(ResourceGroupCandidate),
    Assignment {
        shared_row: usize,
        network: i32,
    },
    Master {
        id: i32,
    },
    Multiplicity {
        network: i32,
    },
    PathCount {
        network: i32,
    },
    Activity {
        network: i32,
        arcs: Vec<(i32, f64)>,
    },
    /// Apart (lower) and together (upper) packing-set children.
    RyanFoster {
        members: Vec<super::RyanFosterMember>,
    },
    ArcSetRyanFoster {
        members: Vec<super::ryan_foster::ArcSetMember>,
    },
}

pub(super) struct Proposal {
    pub name: String,
    pub history_key: String,
    pub priority: f64,
    pub value: f64,
    pub restriction: Restriction,
    pub history_eligible: bool,
    pub fresh: bool,
    pub origin: Option<ProducerFamily>,
    pub component_admission: Option<Box<ComponentAdmission>>,
}
impl Proposal {
    /// Only explicit mapped payloads own mapped selection metadata. Display
    /// names may intentionally overlap generators from other producers.
    fn mapped_registration_name(&self) -> Option<&str> {
        if self.origin.is_some() {
            return None;
        }
        match &self.restriction {
            Restriction::Mapped(variable) | Restriction::Aggregate { variable, .. } => {
                Some(variable.name.as_str())
            }
            _ => None,
        }
    }

    fn history_exclusion_key(&self) -> &str {
        match self.restriction {
            // A class family can publish many sequences with one display name.
            // Only the same logical generator is already supplied by history.
            Restriction::Component(_) | Restriction::GeneratedComponent { .. } => &self.history_key,
            _ => &self.name,
        }
    }

    fn lower_fraction(&self) -> f64 {
        if let Restriction::Component(class) | Restriction::GeneratedComponent { class, .. } =
            &self.restriction
        {
            component::split_bounds(self.value, !class.sequence.bounds.is_empty())
                .map_or(f64::NAN, |(lower, _)| self.value - lower)
        } else if matches!(self.restriction, Restriction::Aggregate { .. }) {
            super::aggregate_projection::fractional_distances(self.value)
                .map_or(f64::NAN, |distances| distances.0)
        } else {
            self.value - self.value.floor()
        }
    }

    fn distance(&self, side: usize) -> f64 {
        if let Restriction::Component(class) | Restriction::GeneratedComponent { class, .. } =
            &self.restriction
        {
            component::split_bounds(self.value, !class.sequence.bounds.is_empty()).map_or(
                f64::NAN,
                |(lower, upper)| {
                    if side == 0 {
                        self.value - lower
                    } else {
                        upper - self.value
                    }
                },
            )
        } else {
            let fraction = self.lower_fraction();
            if side == 0 {
                fraction
            } else {
                1. - fraction
            }
        }
    }

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
        let mut children: Vec<_> = match &self.restriction {
            Restriction::Component(class) | Restriction::GeneratedComponent { class, .. } => {
                class.branching_children(node, self.value)
            }
            Restriction::Mapped(variable) => bounds
                .into_iter()
                .map(|b| variable.apply(node, b))
                .collect(),
            Restriction::Aggregate {
                variable,
                fixed_activity,
            } => {
                let (lower, upper) = super::aggregate_projection::split_bounds(self.value)
                    .ok_or_else(|| invalid("aggregate split activity"))?;
                [
                    Bounds {
                        lower: f64::NEG_INFINITY,
                        upper: lower,
                    },
                    Bounds {
                        lower: upper,
                        upper: f64::INFINITY,
                    },
                ]
                .into_iter()
                .map(|b| variable.apply_residual(node, b, *fixed_activity))
                .collect()
            }
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
            Restriction::Multiplicity { network } | Restriction::PathCount { network } => bounds
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
            Restriction::RyanFoster { members } => [false, true]
                .into_iter()
                .map(|together| node.with_ryan_foster(members, together))
                .collect(),
            Restriction::ArcSetRyanFoster { members } => [false, true]
                .into_iter()
                .map(|together| node.with_arc_set_branch(members, together))
                .collect(),
        }?;
        component::inherit_branch_rows(node, &mut children);
        Ok(children)
    }
    /// Generators whose children may be rejected (resource schedules, or the
    /// pricing limit of 512 Ryan–Foster resource bits); try the next one.
    fn unsupported_resource(&self) -> bool {
        matches!(
            self.restriction,
            Restriction::Resource(_)
                | Restriction::RyanFoster { .. }
                | Restriction::ArcSetRyanFoster { .. }
        )
    }
}

struct Candidate<const D: usize> {
    proposal: Proposal,
    children: Option<Vec<NodeSnapshot<D>>>,
    usable: bool,
    score: f64,
    stages: Vec<usize>,
    last_exact: Option<bool>,
    order: usize,
    first_child_order: Option<usize>,
    trial_order: Option<Vec<usize>>,
    generator_order: bool,
    subtree_size: f64,
    evaluated: Vec<Option<Output<D>>>,
    estimates: Vec<f64>,
    // Bounds published by the exact-pricing driver, independently of whether
    // that trial completed. Restricted/heuristic LP estimates never enter.
    proof_bounds: Vec<(f64, f64)>,
}

impl<const D: usize> Candidate<D> {
    fn initialize_children(&mut self, children: Vec<NodeSnapshot<D>>, parent: &NodeSnapshot<D>) {
        let count = children.len();
        self.generator_order = children
            .first()
            .and_then(|child| child.state.component_local_rows.first())
            .is_some_and(|row| row.lower && !row.sequence.bounds.is_empty());
        self.stages = vec![0; count];
        self.evaluated = std::iter::repeat_with(|| None).take(count).collect();
        self.estimates = vec![1e12; count];
        self.proof_bounds = vec![(parent.warm.numerical_bound, parent.warm.certified_bound); count];
        self.children = Some(children);
    }
}

pub(super) struct Selection<const D: usize> {
    pub children: Vec<NodeSnapshot<D>>,
    pub evaluated: Vec<Option<Output<D>>>,
    pub direction: Option<NetworkBranchDirection>,
    order: Option<Vec<usize>>,
}

impl<const D: usize> Selection<D> {
    pub fn fresh(children: Vec<NodeSnapshot<D>>) -> Self {
        let order = children
            .first()
            .and_then(|child| child.state.component_local_rows.first())
            .is_some_and(|row| row.lower && !row.sequence.bounds.is_empty())
            .then(|| (0..children.len()).collect());
        let evaluated = std::iter::repeat_with(|| None)
            .take(children.len())
            .collect();
        Self {
            children,
            evaluated,
            direction: None,
            order,
        }
    }

    pub fn reverse(&mut self) {
        self.children.reverse();
        self.evaluated.reverse();
    }

    pub fn admit(&mut self, default: NetworkBranchDirection) {
        if let Some(order) = self.order.take() {
            let mut children: Vec<_> = std::mem::take(&mut self.children)
                .into_iter()
                .map(Some)
                .collect();
            let mut evaluated = std::mem::take(&mut self.evaluated);
            for side in order {
                self.children
                    .push(children[side].take().expect("unique child order"));
                self.evaluated.push(evaluated[side].take());
            }
        } else if self.direction.unwrap_or(default) == NetworkBranchDirection::UpperFirst {
            self.reverse();
        }
    }
}

/// Native `GenVar` admission: the distance to the nearest integer must
/// exceed the 1e-8 integrality parameter by the native comparison band
/// (`1e-6 + 1e-10 * max(|x|, |y|)`), and also the tree tolerance.
pub(super) fn mapped_fractional(value: f64, tolerance: f64) -> bool {
    const INTEGRALITY: f64 = 1e-8;
    let fraction = (value - value.round()).abs();
    fraction > tolerance && fraction - (1e-6 + 1e-10 * fraction.max(INTEGRALITY)) > INTEGRALITY
}

/// A generator family's own child order, independent of the tree policy.
/// Packing-set Ryan–Foster generators are created with native direction `U`.
fn family_direction(
    proposal: &Proposal,
    _options: &ModelTreeOptions,
) -> Option<NetworkBranchDirection> {
    matches!(
        proposal.restriction,
        Restriction::Aggregate { .. }
            | Restriction::RyanFoster { .. }
            | Restriction::ArcSetRyanFoster { .. }
            | Restriction::PathCount { .. }
    )
    .then_some(NetworkBranchDirection::UpperFirst)
}

pub(super) fn mapped_proposals<const D: usize>(
    output: &Output<D>,
    options: &ModelTreeOptions,
) -> Result<Vec<Proposal>> {
    let variables = &options.branching_registry().mapped_variables;
    let mut proposals = vec![];
    let ordinary_values = if options.branching_registry().aggregate_projection.is_empty() {
        super::model_mapping::values(variables, output)
    } else {
        super::model_mapping::values_except(variables, output, |variable| {
            options
                .branching_registry()
                .aggregate_projection
                .contains_key(&variable.name)
        })
    };
    for (variable, ordinary_value) in variables.iter().zip(ordinary_values) {
        let projection = options
            .branching_registry()
            .aggregate_projection
            .get(&variable.name);
        let (value, restriction) = if let Some(projection) = projection {
            if options.branching_registry().strong_branching.is_none() {
                continue;
            }
            let (Some(value), Some(fixed_activity)) = (
                projection.full_value(output),
                variable.fixed_activity(output),
            ) else {
                continue;
            };
            (
                value,
                Restriction::Aggregate {
                    variable: variable.clone(),
                    fixed_activity,
                },
            )
        } else {
            let Some(value) = ordinary_value else {
                continue;
            };
            (value, Restriction::Mapped(variable.clone()))
        };
        let fresh = match projection {
            Some(projection) => projection
                .value(
                    output,
                    options.branching_registry().strong_branching.is_some(),
                )
                .is_some_and(|projected| {
                    super::model_mapping::aggregate_fractional(
                        projected,
                        options
                            .branching_registry()
                            .aggregate_admission
                            .get(&variable.name)
                            .copied()
                            .unwrap_or(1e-8),
                    )
                }),
            None => match options
                .branching_registry()
                .aggregate_admission
                .get(&variable.name)
            {
                Some(&threshold) => super::model_mapping::aggregate_fractional(value, threshold),
                None => mapped_fractional(value, options.tolerance),
            },
        };
        let fraction = if projection.is_some() {
            super::aggregate_projection::fractional_distances(value)
                .map_or(f64::NAN, |distances| distances.0)
        } else {
            value - value.floor()
        };
        let refreshable = projection.is_some()
            && options.branching_registry().strong_branching.is_some()
            && options.branching_registry().use_branching_history
            && output.snapshot.state.node_depth > 0
            && fraction.abs() > 1e-6;
        if !value.is_finite() || value.abs() >= 9_007_199_254_740_992. || !(fresh || refreshable) {
            continue;
        }
        proposals.push(Proposal {
            name: variable.name.clone(),
            history_key: format!("mapped:{}", variable.name),
            priority: variable.priority,
            value,
            restriction,
            history_eligible: true,
            fresh,
            origin: None,
            component_admission: None,
        });
    }
    Ok(proposals)
}

/// Reorder registered mapped candidates within their family slots, keeping
/// unrelated generators in place. Input proposals are sorted by priority.
fn rank_mapped_slots(proposals: &[Proposal], options: &ModelTreeOptions, order: &mut [usize]) {
    let variables = &options.branching_registry().mapped_variables;
    let selection = &options.branching_registry().branch_selection;
    let registrations: HashMap<&str, usize> = variables
        .iter()
        .enumerate()
        .map(|(index, variable)| (variable.name.as_str(), index))
        .collect();
    let mut groups: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    let mut first_at_level = 0;
    for (index, proposal) in proposals.iter().enumerate() {
        if index > 0
            && (exceeds(proposals[first_at_level].priority, proposal.priority)
                || exceeds(proposal.priority, proposals[first_at_level].priority))
        {
            first_at_level = index;
        }
        if let Some(name) = proposal
            .mapped_registration_name()
            .filter(|name| !selection.contains_key(*name))
        {
            if let Some(&registration) = registrations.get(name) {
                groups
                    .entry(first_at_level)
                    .or_default()
                    .push((index, registration));
            }
        }
    }
    for slots in groups.values() {
        let mut registered = slots.clone();
        registered.sort_by_key(|&(_, registration)| registration);
        let indices: HashMap<&str, usize> = slots
            .iter()
            .map(|&(index, _)| (proposals[index].name.as_str(), index))
            .collect();
        let ranked = super::model_mapping::rank_candidates_with_selection(
            registered.iter().map(|&(index, registration)| {
                (
                    &variables[registration],
                    selection_value(&proposals[index], options),
                )
            }),
            selection,
        );
        for (&(slot, _), (variable, _)) in slots.iter().zip(ranked) {
            order[slot] = indices[variable.name.as_str()];
        }
    }
}

/// Exact family levels partition disjoint slots, so group iteration order has
/// no effect on the resulting proposal permutation. Normalize signed zero to
/// preserve floating-point equality without rescanning the proposal registry.
fn rank_count_slots<const D: usize>(
    proposals: &[Proposal],
    options: &ModelTreeOptions,
    output: &Output<D>,
    order: &mut [usize],
) {
    let mut path_groups: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    let mut groups: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for (index, proposal) in proposals.iter().enumerate() {
        let level = if proposal.priority == 0. {
            0
        } else {
            proposal.priority.to_bits()
        };
        match proposal.restriction {
            Restriction::PathCount { .. }
                if !options
                    .branching_registry()
                    .branch_selection
                    .contains_key(&proposal.name) =>
            {
                path_groups.entry(level).or_default().push(index)
            }
            Restriction::Multiplicity { .. } => groups.entry(level).or_default().push(index),
            _ => {}
        }
    }
    let mut path_order = None;
    for slots in path_groups.values() {
        let mut ranked = slots.clone();
        // Small registries are faster with their contiguous linear scan. Large
        // registries share one lookup table across every priority group.
        let registration =
            if slots.len() > 1 && options.branching_registry().paths_per_network.len() >= 128 {
                Some(path_order.get_or_insert_with(|| {
                    options
                        .branching_registry()
                        .paths_per_network
                        .iter()
                        .enumerate()
                        .map(|(index, path)| (path.network, index))
                        .collect::<HashMap<_, _>>()
                }))
            } else {
                None
            };
        let rank = |index: usize| match proposals[index].restriction {
            Restriction::PathCount { network } => registration.as_ref().map_or_else(
                || {
                    options
                        .branching_registry()
                        .paths_per_network
                        .iter()
                        .position(|path| path.network == network)
                        .unwrap_or(usize::MAX)
                },
                |ranks| ranks.get(&network).copied().unwrap_or(usize::MAX),
            ),
            _ => usize::MAX,
        };
        ranked.sort_by(|&a, &b| {
            let distance = |index: usize| (proposals[index].value.fract() - 0.5).abs();
            distance(a)
                .total_cmp(&distance(b))
                .then_with(|| rank(a).cmp(&rank(b)))
        });
        for (&slot, index) in slots.iter().zip(ranked) {
            order[slot] = index;
        }
    }
    let mut network_order = None;
    for slots in groups.values() {
        let mut registered = slots.clone();
        let registration = if slots.len() > 1 && output.snapshot.state.oracles.len() >= 128 {
            Some(network_order.get_or_insert_with(|| {
                output
                    .snapshot
                    .state
                    .oracles
                    .iter()
                    .enumerate()
                    .map(|(index, oracle)| (oracle.network.id, index))
                    .collect::<HashMap<_, _>>()
            }))
        } else {
            None
        };
        registered.sort_by_key(|&index| match proposals[index].restriction {
            Restriction::Multiplicity { network } => registration.as_ref().map_or_else(
                || {
                    output
                        .snapshot
                        .state
                        .oracles
                        .iter()
                        .position(|oracle| oracle.network.id == network)
                        .unwrap_or(usize::MAX)
                },
                |ranks| ranks.get(&network).copied().unwrap_or(usize::MAX),
            ),
            _ => unreachable!(),
        });
        for (&slot, index) in slots.iter().zip(registered) {
            order[slot] = index;
        }
    }
}

/// Fresh generated aggregates enter the candidate set before full activity
/// refresh. Keep that selection view separate from split/history activity.
fn selection_value(proposal: &Proposal, _options: &ModelTreeOptions) -> f64 {
    if proposal.fresh && matches!(proposal.restriction, Restriction::Aggregate { .. }) {
        1.
    } else {
        proposal.value
    }
}

fn order_proposals(proposals: &mut Vec<Proposal>, options: &ModelTreeOptions) {
    let registration = super::family_registration::selection_order(options);
    let less = |a: &Proposal, b: &Proposal| {
        if exceeds(a.priority, b.priority) {
            return true;
        }
        if exceeds(b.priority, a.priority) {
            return false;
        }
        let metadata = |proposal: &Proposal| {
            let name = proposal.mapped_registration_name().or_else(|| {
                matches!(
                    proposal.origin,
                    Some(
                        ProducerFamily::Cluster(_)
                            | ProducerFamily::ComponentClass(_)
                            | ProducerFamily::PathsPerNetwork(_)
                            | ProducerFamily::PackingAssignment
                    )
                )
                .then_some(proposal.name.as_str())
            })?;
            options
                .branching_registry()
                .branch_selection
                .get(name)
                .zip(registration.get(name))
        };
        if let (Some((a_selection, &a_registration)), Some((b_selection, &b_registration))) =
            (metadata(a), metadata(b))
        {
            return super::model_mapping::selection_before(
                a_selection,
                selection_value(a, options),
                a_registration,
                b_selection,
                selection_value(b, options),
                b_registration,
            );
        }
        let a_value = selection_value(a, options);
        let b_value = selection_value(b, options);
        let a_fraction = (a_value - a_value.round()).abs();
        let b_fraction = (b_value - b_value.round()).abs();
        if exceeds(a_fraction, b_fraction) {
            return true;
        }
        if exceeds(b_fraction, a_fraction) {
            return false;
        }
        options
            .branching_registry()
            .packing
            .as_ref()
            .map_or(std::cmp::Ordering::Equal, |packing| {
                let key = |proposal: &Proposal| {
                    if proposal.origin == Some(ProducerFamily::PackingAssignment) {
                        packing.tie_key(&proposal.name)
                    } else {
                        (1, 0, 0)
                    }
                };
                key(a).cmp(&key(b))
            })
            .then_with(|| a.name.cmp(&b.name))
            .is_lt()
    };
    // Tolerant comparisons are not transitive over chains of nearby levels.
    // Preserve ordered-set insertion/rebalancing rather than passing them to
    // Rust's comparison sort. Registration order distinguishes exact ties.
    let mut set = crate::cuts::shared::rbtree::CandidateSet::new();
    for index in 0..proposals.len() {
        set.insert_by(index, |a, b| {
            less(&proposals[a], &proposals[b]) || (!less(&proposals[b], &proposals[a]) && a < b)
        });
    }
    let mut slots: Vec<_> = std::mem::take(proposals).into_iter().map(Some).collect();
    *proposals = set
        .ordered()
        .into_iter()
        .map(|(_, index)| slots[index].take().unwrap())
        .collect();
}

pub(super) fn validate(options: &crate::master::tree::StrongBranchingOptions) -> Result<()> {
    if options.phases.is_empty()
        || !options.partial_score_conservativeness.is_finite()
        || options.partial_score_conservativeness <= 0.
        || options.phases.iter().any(|phase| {
            !phase.subtree_ratio.is_finite()
                || phase.subtree_ratio < 0.
                || (!phase.restricted_master_only && phase.max_iterations == 0)
                || phase.min_cut_rounds > phase.max_cut_rounds
                || (phase.is_exact()
                    && (phase.restricted_master_only || phase.pricing_phase_floor != 0))
        })
    {
        return Err(invalid(
            "mapped strong branching requires valid nonempty phases",
        ));
    }
    Ok(())
}

#[cfg(test)]
use super::family_registration::registered_proposals;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ProducerFamily {
    PackingAssignment,
    PackingResource,
    PackingRyanFoster,
    ComponentGenerator(usize),
    Cluster(usize),
    ComponentClass(usize),
    PathsPerNetwork(usize),
}

pub(super) struct ComponentAdmission {
    pub publication: std::sync::Arc<component::ComponentPublication>,
    pub raw_index: usize,
    pub native_order: Option<usize>,
    pub legacy_order: Option<usize>,
}

fn prepare_component_priority(
    proposals: &mut [Proposal],
    priority: f64,
    quota: usize,
    deadline: Instant,
) -> Option<()> {
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (index, proposal) in proposals.iter().enumerate() {
        if proposal.priority == priority && proposal.fresh {
            if let Some(ProducerFamily::ComponentGenerator(family)) = proposal.origin {
                groups.entry(family).or_default().push(index);
            }
        }
    }
    for members in groups.values() {
        let first = proposals[members[0]].component_admission.as_ref()?;
        let selected = first.publication.publish(quota, Some(deadline))?;
        let ranks: HashMap<_, _> = selected
            .into_iter()
            .enumerate()
            .map(|(rank, index)| (index, rank))
            .collect();
        for &index in members {
            let admission = proposals[index].component_admission.as_mut()?;
            admission.native_order = ranks.get(&admission.raw_index).copied();
        }
    }
    Some(())
}

fn component_view_member(proposal: &Proposal, native: bool) -> bool {
    proposal.component_admission.as_ref().is_none_or(|view| {
        if native {
            view.native_order.is_some()
        } else {
            view.legacy_order.is_some()
        }
    })
}

fn family_kind(family: &BranchFamilyRegistration) -> u8 {
    use BranchFamilyRegistration::*;
    match family {
        MappedFamily(_) | MappedRegistration(_) => 0,
        ComponentGenerator(_) => 1,
        PackingAssignment => 2,
        PackingResource => 3,
        PackingRyanFoster => 4,
        Cluster(_) => 5,
        ComponentClass(_) => 6,
        PathsPerNetwork(_) => 7,
    }
}

/// Resolve actual producer identities before candidate sorting. Different
/// kinds require explicitly declared order; no category ordering is inferred.
fn registered_family_batches(
    proposals: &[Proposal],
    options: &ModelTreeOptions,
    declaration: &HashMap<BranchFamilyRegistration, usize>,
    priority: f64,
) -> Option<Vec<Vec<usize>>> {
    use BranchFamilyRegistration::*;
    let registrations = super::family_registration::mapped_families(options);
    let mut families: Vec<(BranchFamilyRegistration, usize, Vec<usize>)> = Vec::new();
    let mut family_slots: HashMap<BranchFamilyRegistration, usize> = HashMap::new();
    for (index, proposal) in proposals.iter().enumerate() {
        if !proposal.fresh
            || proposal.priority != priority
            || !component_view_member(proposal, true)
        {
            continue;
        }
        let (family, creation_order) = match proposal.origin {
            Some(ProducerFamily::PackingAssignment)
                if options.branching_registry().packing.is_some() =>
            {
                (PackingAssignment, 0)
            }
            Some(ProducerFamily::PackingResource) => (PackingResource, 0),
            Some(ProducerFamily::PackingRyanFoster)
                if options.branching_registry().ryan_foster.is_some() =>
            {
                (PackingRyanFoster, 0)
            }
            Some(ProducerFamily::ComponentGenerator(registration))
                if registration < options.branching_registry().component_generators.len() =>
            {
                (ComponentGenerator(registration), registration)
            }
            Some(ProducerFamily::Cluster(registration))
                if registration < options.branching_registry().clusters.len()
                    && declaration.contains_key(&Cluster(registration)) =>
            {
                (Cluster(registration), registration)
            }
            Some(ProducerFamily::ComponentClass(registration))
                if registration < options.branching_registry().component_classes.len()
                    && declaration.contains_key(&ComponentClass(registration)) =>
            {
                (ComponentClass(registration), registration)
            }
            Some(ProducerFamily::PathsPerNetwork(registration))
                if registration < options.branching_registry().paths_per_network.len()
                    && declaration.contains_key(&PathsPerNetwork(registration)) =>
            {
                (PathsPerNetwork(registration), registration)
            }
            Some(_) => return None,
            None => {
                if !matches!(
                    proposal.restriction,
                    Restriction::Mapped(_) | Restriction::Aggregate { .. }
                ) {
                    return None;
                }
                registrations.get(&proposal.name)?.clone()
            }
        };
        if let Some(&slot) = family_slots.get(&family) {
            families[slot].2.push(index);
        } else {
            family_slots.insert(family.clone(), families.len());
            families.push((family, creation_order, vec![index]));
        }
    }
    let all_declared = families
        .iter()
        .all(|(family, _, _)| declaration.contains_key(family));
    let none_declared = families
        .iter()
        .all(|(family, _, _)| !declaration.contains_key(family));
    let same_kind = families.first().is_none_or(|first| {
        families
            .iter()
            .all(|(family, _, _)| family_kind(family) == family_kind(&first.0))
    });
    if families.len() > 1 && !all_declared && !(none_declared && same_kind) {
        return None;
    }
    if all_declared {
        families.sort_by_key(|(family, _, _)| declaration[family]);
    } else {
        families.sort_by_key(|(_, registration, _)| *registration);
    }
    Some(
        families
            .into_iter()
            .map(|(_, _, members)| members)
            .collect(),
    )
}

// Compatibility wrapper retained for the mapped-only regression fixtures.
#[cfg(test)]
fn mapped_family_batches(
    proposals: &[Proposal],
    options: &ModelTreeOptions,
    priority: f64,
) -> Option<Vec<Vec<usize>>> {
    let declaration = super::family_registration::registration_order(options).ok()?;
    registered_family_batches(proposals, options, &declaration, priority)
}

fn select_component_view(
    proposals: &mut Vec<Proposal>,
    native: bool,
    declaration: &HashMap<BranchFamilyRegistration, usize>,
    options: &ModelTreeOptions,
) {
    proposals.retain(|proposal| component_view_member(proposal, native));
    let slots: Vec<_> = proposals
        .iter()
        .enumerate()
        .filter_map(|(index, proposal)| proposal.component_admission.as_ref().map(|_| index))
        .collect();
    let mut ranked = slots.clone();
    ranked.sort_by(|&a, &b| {
        let view_a = proposals[a].component_admission.as_ref().unwrap();
        let view_b = proposals[b].component_admission.as_ref().unwrap();
        if !native {
            return view_a.legacy_order.cmp(&view_b.legacy_order);
        }
        let ProducerFamily::ComponentGenerator(family_a) = proposals[a].origin.unwrap() else {
            unreachable!()
        };
        let ProducerFamily::ComponentGenerator(family_b) = proposals[b].origin.unwrap() else {
            unreachable!()
        };
        let order_a = declaration
            .get(&BranchFamilyRegistration::ComponentGenerator(family_a))
            .copied()
            .unwrap_or(family_a);
        let order_b = declaration
            .get(&BranchFamilyRegistration::ComponentGenerator(family_b))
            .copied()
            .unwrap_or(family_b);
        options.branching_registry().component_generators[family_b]
            .priority
            .total_cmp(&options.branching_registry().component_generators[family_a].priority)
            .then(order_a.cmp(&order_b))
            .then(view_a.native_order.cmp(&view_b.native_order))
    });
    let mut permutation: Vec<_> = (0..proposals.len()).collect();
    for (&slot, &index) in slots.iter().zip(&ranked) {
        permutation[slot] = index;
    }
    let mut values: Vec<_> = std::mem::take(proposals).into_iter().map(Some).collect();
    *proposals = permutation
        .into_iter()
        .map(|index| values[index].take().unwrap())
        .collect();
    for (ordinal, &slot) in slots.iter().enumerate() {
        if let Restriction::GeneratedComponent { ordinal: value, .. } =
            &mut proposals[slot].restriction
        {
            *value = ordinal;
        }
    }
}

fn fresh_priorities(proposals: &[Proposal], options: &ModelTreeOptions) -> Vec<f64> {
    let mut priorities: Vec<_> =
        super::family_registration::registered_priorities(options).collect();
    if options.branching_registry().branching != NetworkBranchingPolicy::Phased {
        priorities.extend(proposals.iter().map(|proposal| proposal.priority));
    } else {
        let p = &options.branching_registry().branch_priorities;
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
    priorities
}

fn fresh_budget<const D: usize>(
    output: &Output<D>,
    config: &strong::StrongBranchingOptions,
    history: &History,
) -> strong::CandidateBudget {
    let mut budget = strong::CandidateBudget::new(
        config.phases.first().map_or(0, |phase| phase.candidates),
        config
            .phases
            .first()
            .map_or(0., |phase| phase.subtree_ratio),
        strong::SubtreeEstimate {
            size: history
                .completed_estimate
                .unwrap_or(output.snapshot.state.mapped_subtree_size),
            priority: output.snapshot.state.mapped_subtree_priority,
        },
    );
    if history.completed_estimate.is_some() {
        budget.use_history();
    }
    budget
}

pub(super) fn choose_proposals<const D: usize>(
    output: &Output<D>,
    options: &ModelTreeOptions,
    history: &mut History,
    incumbent: f64,
    deadline: Instant,
    mut proposals: Vec<Proposal>,
) -> Result<Option<Selection<D>>> {
    if options.threads == 1 && history.incidence_history.is_none() {
        history.incidence_history = output.snapshot.incidence_separator_history();
    }
    let declaration = super::family_registration::registration_order(options)?;
    let resolved_families = if let Some(config) = &options.branching_registry().strong_branching {
        let mut scope_budget = fresh_budget(output, config, history);
        let priorities = fresh_priorities(&proposals, options);
        let mut resolved = true;
        let mut interrupted = false;
        strong::fresh_family_batches(
            scope_budget.quota,
            &priorities,
            Some(&mut scope_budget),
            |priority, quota| {
                if prepare_component_priority(&mut proposals, priority, quota, deadline).is_none() {
                    interrupted = true;
                    return vec![];
                }
                match registered_family_batches(&proposals, options, &declaration, priority) {
                    Some(families) => families,
                    None => {
                        resolved = false;
                        vec![]
                    }
                }
            },
        );
        if interrupted {
            return Ok(None);
        }
        resolved
    } else {
        false
    };
    select_component_view(&mut proposals, resolved_families, &declaration, options);
    if options.branching_registry().branching == NetworkBranchingPolicy::Phased
        && options.branching_registry().use_branching_history
        && options.branching_registry().strong_branching.is_some()
        && output.snapshot.state.node_depth > 0
    {
        let present: HashSet<_> = proposals.iter().map(|p| p.history_key.clone()).collect();
        for (key, template) in &history.templates {
            if present.contains(key) {
                continue;
            }
            let refreshed = match template {
                HistoryTemplate::Resource(template) => {
                    super::tree_adapter::proposals::refresh_resource(
                        output,
                        template,
                        options.branching_registry().branch_priorities.resource,
                        options.tolerance,
                    )
                }
                HistoryTemplate::PackingAssignment(name) => options
                    .branching_registry()
                    .packing
                    .as_ref()
                    .and_then(|packing| packing.refresh(output, name, options.tolerance)),
            };
            if let Some(proposal) = refreshed {
                // Unobserved retained generators are refreshed, but do not
                // enter history admission until a treatment records a cost.
                if history.costs.get(key).is_some_and(|stages| {
                    stages
                        .values()
                        .any(|sides| sides.iter().any(|stat| stat.count > 0))
                }) {
                    proposals.push(proposal);
                }
            }
        }
    }
    if proposals
        .iter()
        .any(|p| !p.priority.is_finite() || !p.value.is_finite())
    {
        return Err(invalid("invalid network branch proposal"));
    }
    order_proposals(&mut proposals, options);
    // Multiplicity generators are visited in pricing registration order,
    // including unequal fractional counts. Keep their priority-family slots
    // and the ordering of every other generator unchanged.
    let mut order: Vec<_> = (0..proposals.len()).collect();
    // Explicit mapped variables use the same in-family comparator as the
    // non-strong path. Reorder only their own slots: packing and other
    // families retain their separately defined tie rules and priorities.
    rank_mapped_slots(&proposals, options, &mut order);
    let component_slots: Vec<_> = proposals
        .iter()
        .enumerate()
        .filter_map(|(index, p)| match p.restriction {
            Restriction::GeneratedComponent { ordinal, .. } => Some((index, ordinal)),
            _ => None,
        })
        .collect();
    let mut ranked_components = component_slots.clone();
    ranked_components.sort_by_key(|&(_, ordinal)| ordinal);
    for (&(slot, _), (index, _)) in component_slots.iter().zip(ranked_components) {
        order[slot] = index;
    }
    // Explicit path-count separation ranks by distance to one half, unlike
    // the inferred empty-component family, which keeps network order.
    rank_count_slots(&proposals, options, output, &mut order);
    let mut slots: Vec<_> = proposals.into_iter().map(Some).collect();
    let mut proposals: Vec<_> = order
        .into_iter()
        .map(|i| slots[i].take().unwrap())
        .collect();
    let Some(config) = &options.branching_registry().strong_branching else {
        for proposal in proposals {
            match proposal.children(&output.snapshot) {
                Ok(children) => {
                    let mut selection = Selection::fresh(children);
                    selection.direction = options
                        .branching_registry()
                        .branch_directions
                        .get(&proposal.name)
                        .copied()
                        .or_else(|| family_direction(&proposal, options));
                    return Ok(Some(selection));
                }
                Err(_) if proposal.unsupported_resource() => continue,
                Err(error) => return Err(error),
            }
        }
        return Ok(None);
    };
    if config
        .phases
        .iter()
        .any(|phase| phase.is_exact() && options.node.cg.min_cut_rounds > phase.max_cut_rounds)
    {
        return Err(invalid("exact strong phase cut-round limits"));
    }
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
    let priorities = fresh_priorities(&proposals, options);
    let (families, fresh, priority_cutoff) = if resolved_families {
        let (families, cutoff) = strong::fresh_family_batches(
            budget.quota,
            &priorities,
            Some(&mut budget),
            |priority, _| {
                registered_family_batches(&proposals, options, &declaration, priority)
                    .expect("family admission preflight and selected view agree")
            },
        );
        (families, Vec::new(), cutoff)
    } else {
        let (fresh, cutoff) = strong::fresh_batches_with_budget(
            budget.quota,
            &priorities,
            Some(&mut budget),
            |priority| {
                proposals
                    .iter()
                    .enumerate()
                    .filter_map(|(index, proposal)| {
                        (proposal.fresh && proposal.priority == priority).then_some(index)
                    })
                    .collect()
            },
        );
        (Vec::new(), fresh, cutoff)
    };
    let limit = budget.quota;
    // Native history admission uses the actual fresh-loop cutoff, including
    // empty registered priority families, and reserves at most half the quota.
    let mut ranked = vec![];
    if options.branching_registry().use_branching_history
        && output.snapshot.state.node_depth > 0
        && gap > 0.
    {
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
                let prediction = if matches!(
                    p.restriction,
                    Restriction::Aggregate { .. }
                        | Restriction::Component(_)
                        | Restriction::GeneratedComponent { .. }
                ) {
                    history.prediction_from_fraction(&p.history_key, p.lower_fraction(), gap)
                } else {
                    history.prediction(&p.history_key, p.value, gap)
                };
                prediction.map(|(stage, score)| (i, stage, score, p.name.clone()))
            })
            .collect();
        ranked.sort_by(|a, b| {
            b.1.cmp(&a.1)
                // Predictions are finite. IEEE equality keeps both zero signs
                // tied so the generator name decides, as for other equal scores.
                .then_with(|| b.2.partial_cmp(&a.2).unwrap())
                .then_with(|| a.3.cmp(&b.3))
        });
        ranked.truncate(limit / 2);
    }
    history.ranked += ranked.len();
    let fresh = if resolved_families {
        let historical_names: HashSet<_> = ranked
            .iter()
            .map(|r| proposals[r.0].history_exclusion_key())
            .collect();
        strong::allocate_family_batches(families, limit - ranked.len(), |&i| {
            historical_names.contains(proposals[i].history_exclusion_key())
        })
    } else {
        fresh
    };
    let mut slots: Vec<_> = proposals.into_iter().map(Some).collect();
    let mut next: Vec<_> = ranked.iter().filter_map(|r| slots[r.0].take()).collect();
    next.extend(fresh.into_iter().filter_map(|i| slots[i].take()));
    proposals = next;
    proposals.truncate(limit);
    // A retained generator exists before any child evaluation. A skipped or
    // interrupted screen can still yield committed-child history later.
    for proposal in &proposals {
        history.remember_proposal(proposal, options);
    }
    let mut candidates: Vec<_> = proposals
        .into_iter()
        .enumerate()
        .map(|(order, proposal)| Candidate {
            proposal,
            children: None,
            usable: true,
            score: 0.,
            stages: vec![],
            last_exact: None,
            order,
            first_child_order: None,
            trial_order: None,
            generator_order: false,
            subtree_size: 1e12,
            evaluated: vec![],
            estimates: vec![],
            proof_bounds: vec![],
        })
        .collect();
    let end = deadline.min(
        Instant::now()
            .checked_add(config.time_limit)
            .unwrap_or(deadline),
    );
    let mut best_size: f64 = 1e12;
    let mut best_score = -1e12;
    let mut parent_columns_absorbed = false;
    let mut next_child_order = 0;
    // A partial group's score is optimistic only under the no-cut-family
    // policy. Feasibility/recourse separators also keep this optimization off.
    let partial_scores_comparable = !options.node.separation.is_active();
    for (phase_index, phase) in config.phases.iter().enumerate() {
        candidates.truncate(phase.candidates);
        if candidates.is_empty() || Instant::now() >= end {
            break;
        }
        let exact = phase.is_exact();
        let next_quota = config
            .phases
            .get(phase_index + 1)
            .filter(|next| next.candidates > 0)
            .map_or(1, |next| next.candidates);
        if !exact && candidates.len() <= next_quota {
            continue;
        }
        let mut evaluated = 0;
        let mut processed_prefix = 0;
        let mut phase_best_score = None;
        let mut stop_phase = false;
        for (candidate_index, candidate) in candidates.iter_mut().enumerate() {
            if stop_phase {
                break;
            }
            if !budget.permits(evaluated, phase.subtree_ratio, best_size) {
                break;
            }
            if Instant::now() >= end {
                break;
            }
            if candidate.children.is_none() {
                match candidate.proposal.children(&output.snapshot) {
                    Ok(children) => {
                        candidate.first_child_order = Some(next_child_order);
                        next_child_order += children.len();
                        candidate.initialize_children(children, &output.snapshot);
                    }
                    Err(_) if candidate.proposal.unsupported_resource() => {
                        candidate.usable = false;
                        candidate.score = f64::NEG_INFINITY;
                        continue;
                    }
                    Err(error) => return Err(error),
                }
            }
            processed_prefix = candidate_index + 1;
            candidate.last_exact = Some(exact);
            evaluated += 1;
            let mut estimates = std::mem::take(&mut candidate.estimates);
            // Native generator direction applies to trial construction too,
            // not just final frontier insertion. Canonical side indices stay
            // tied to lower/upper history distances and proof bounds.
            let direction = options
                .branching_registry()
                .branch_directions
                .get(&candidate.proposal.name)
                .copied()
                .or_else(|| family_direction(&candidate.proposal, options))
                .unwrap_or_else(|| {
                    if options.branching_registry().branching == NetworkBranchingPolicy::Phased
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
            let sides = trial_sides(
                candidate.trial_order.take().unwrap_or_else(|| {
                    if estimates.len() == 2 && !candidate.generator_order {
                        direction.sides().into()
                    } else {
                        (0..estimates.len()).collect()
                    }
                }),
                &estimates,
                !phase.restricted_master_only,
            );
            candidate.trial_order = Some(sides.clone());
            // Pricing phases revisit the strongest preceding LP estimate
            // first; ties preserve the group's generator order. Canonical
            // indices still identify lower/upper proof and history channels.
            // Native admits a candidate's whole child group. An elapsed
            // deadline leaves the sibling a zero-budget attempt, rather than
            // removing it from trial accounting or branch evaluation.
            let mut abandoned = false;
            for (processed, side) in sides.iter().copied().enumerate() {
                if exact && phase.tree_depth_score && partial_scores_comparable {
                    if let Some(best) = phase_best_score {
                        let binary;
                        let multiway;
                        let ordered: &[f64] = if sides.len() == 2 {
                            binary = [estimates[sides[0]], estimates[sides[1]]];
                            &binary
                        } else {
                            multiway = sides
                                .iter()
                                .map(|&side| estimates[side])
                                .collect::<Vec<_>>();
                            &multiway
                        };
                        if partial_group_cannot_improve(
                            parent,
                            cutoff,
                            ordered,
                            processed,
                            config.partial_score_conservativeness,
                            best,
                        ) {
                            abandoned = true;
                            break;
                        }
                    }
                }
                let mut child = candidate.children.as_ref().unwrap()[side].fork()?;
                if options.share_inactive_columns {
                    // The parent is unchanged throughout candidate evaluation;
                    // the shared pool only grows as trial columns are added.
                    if !parent_columns_absorbed {
                        history.column_pool.absorb(&output.snapshot);
                        parent_columns_absorbed = true;
                    }
                    history.column_pool.seed(&mut child, end)?;
                }
                child.state.node_depth = output.snapshot.state.node_depth.saturating_add(1);
                child.prepare_evaluation()?;
                candidate.stages[side] += 1;
                history.trials += 1;
                history.processed_trials += usize::from(exact);
                let remaining = end
                    .saturating_duration_since(Instant::now())
                    .min(config.trial_time_limit);
                let observation = if phase.restricted_master_only {
                    // This model includes original/artificial columns but no
                    // private proximal auxiliaries. Its objective ranks only.
                    let (solved, objective) = child
                        .master
                        .solve_limited_with_objective(i32::MAX as usize, remaining)?;
                    let observation = if let Some(objective) = objective {
                        if solved == crate::master::clp::ClpSolveOutcome::Optimal {
                            history.retain_witness(&child, options.tolerance, incumbent)?;
                        }
                        estimates[side] = objective.min(1e12);
                        Some(MappedEvaluationKind::RestrictedMaster)
                    } else if solved == crate::master::clp::ClpSolveOutcome::Stopped {
                        // No fresh optimal LP value: preserve the unavailable-LP
                        // sentinel, including a zero-budget sibling attempt.
                        estimates[side] = 1e12;
                        Some(MappedEvaluationKind::RestrictedMaster)
                    } else {
                        None
                    };
                    candidate.children.as_mut().unwrap()[side] = child;
                    observation
                } else {
                    let mut node = options.node.clone();
                    let current_incumbent = history
                        .incumbent
                        .as_ref()
                        .map_or(incumbent, |witness| incumbent.min(witness.0));
                    configure_trial_options(&mut node, phase, current_incumbent);
                    node.search_inactive_columns |= options.share_inactive_columns;
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
                        let last = oracle.solver.phases.len().checked_sub(1).ok_or_else(|| {
                            invalid("mapped strong branching requires a pricing schedule")
                        })?;
                        // A restriction beyond the available stages keeps the
                        // last stage, matching native phase preparation.
                        let floor = phase.pricing_phase_floor.min(last);
                        oracle.solver.phases = oracle.solver.phases[floor..].to_vec();
                    }
                    node.pricing_options.clear();
                    // The deadline inside the node bounds each labeling call.
                    child.state.deadline = Instant::now()
                        .checked_add(remaining)
                        .unwrap_or(end)
                        .min(end);
                    if options.threads == 1 {
                        if let Some(incidence) = &history.incidence_history {
                            child.set_incidence_separator_history(incidence);
                        }
                    }
                    match child.solve_forked_until(node, end) {
                        Ok(result) => {
                            if options.threads == 1 {
                                if let Some(incidence) =
                                    result.snapshot.incidence_separator_history()
                                {
                                    history.incidence_history = Some(incidence);
                                }
                            }
                            if options.share_inactive_columns {
                                history.column_pool.absorb(&result.snapshot);
                            }
                            let timed_out = result.master.status == cg::Status::TimeLimit;
                            if !timed_out {
                                history.retain_witness(
                                    &result.snapshot,
                                    options.tolerance,
                                    incumbent,
                                )?;
                            }
                            history.pricing_calls += result.master.pricing_calls;
                            history.iterations += result.master.iterations;
                            for &(id, calls) in &result.oracle_calls {
                                *history.oracle_calls.entry(id).or_default() += calls;
                            }
                            // As in the recovered routing policy, an incomplete
                            // original LP value can rank a trial; it is never a
                            // child bound. Treatment history is independent of
                            // whether the completed output can be reused.
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
                            if timed_out {
                                // Native leaves its unavailable-LP sentinel on
                                // timeout, even after a partial LP or before
                                // a zero-budget sibling enters the backend.
                                estimates[side] = 1e12;
                            } else if result.master.iterations > 0 {
                                // An unresolved artificial solution uses the
                                // reference's ranking sentinel, except that an
                                // iteration-limited evaluation retains its
                                // finite penalized LP as native history does.
                                estimates[side] = if artificial_zero
                                    || result.master.status == cg::Status::IterationLimit
                                {
                                    result.master.lp.min(1e12)
                                } else {
                                    1e12
                                };
                            }
                            // Native history retains the finite ranking sentinel
                            // for completed infeasible trials and attempted
                            // heuristic screens with unresolved artificials.
                            // It remains ranking data, never a child proof bound.
                            let completed_exact = exact
                                && artificial_zero
                                && result.master.status == cg::Status::Optimal;
                            let observation = if exact && proven_infeasible(&result) {
                                estimates[side] = 1e12;
                                Some(MappedEvaluationKind::ExactPricing)
                            } else if (timed_out || result.master.iterations > 0)
                                && estimates[side].is_finite()
                                && (!exact
                                    || timed_out
                                    || result.master.status == cg::Status::IterationLimit)
                            {
                                Some(if exact {
                                    MappedEvaluationKind::ExactPricing
                                } else {
                                    MappedEvaluationKind::HeuristicPricing
                                })
                            } else if artificial_zero
                                && estimates[side].is_finite()
                                && exact
                                && result.master.status == cg::Status::Optimal
                            {
                                Some(MappedEvaluationKind::ExactPricing)
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
                            if (completed_exact
                                && observation == Some(MappedEvaluationKind::ExactPricing))
                                || exact && proven_infeasible(&result)
                                || exact
                                    && result.master.status == cg::Status::Optimal
                                    && result.master.pruned_by_bound
                                    && result.master.bound.is_finite()
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
                if let Some(kind) = observation {
                    let evaluation = MappedBranchEvaluation {
                        name: candidate.proposal.name.clone(),
                        depth: output.snapshot.state.node_depth,
                        side,
                        stage: candidate.stages[side],
                        kind,
                        parent,
                        estimate: estimates[side],
                        distance: candidate.proposal.distance(side),
                    };
                    if candidate.proposal.history_eligible {
                        // A witness discovered in this child tightens its
                        // history cap immediately. The pass's scoring cutoff
                        // remains the separate fixed entry value above.
                        let history_incumbent = history
                            .incumbent
                            .as_ref()
                            .map_or(incumbent, |witness| incumbent.min(witness.0));
                        history.record_with_key(
                            &candidate.proposal.history_key,
                            evaluation,
                            history_incumbent,
                        );
                    } else {
                        // History-ineligible trials are traced but, as in native,
                        // never enter the pseudocost statistics.
                        history.evaluations.push(evaluation);
                    }
                }
            }
            if abandoned {
                candidate.estimates = estimates;
                // This discards a ranking candidate, never one of the committed
                // search nodes or any proof bound from its disjunction.
                candidate.usable = false;
                candidate.score = f64::NEG_INFINITY;
                continue;
            }
            // Trial witnesses are published only after this pass returns,
            // as in native; a child's local primal value is not the parent's
            // shared cutoff for this pass.
            let (score, subtree_size) = if phase.tree_depth_score {
                strong::tree_depth_metrics(parent, cutoff, &estimates, 1.)
            } else {
                (strong::product_score(parent, cutoff, &estimates), 1e12)
            };
            candidate.estimates = estimates;
            candidate.score = score;
            if phase.tree_depth_score && exact {
                candidate.subtree_size = if subtree_size.is_finite() && subtree_size >= 0. {
                    subtree_size.min(1e12)
                } else {
                    1e12
                };
                if exceeds(candidate.score, best_score) {
                    best_score = candidate.score;
                    best_size = candidate.subtree_size;
                }
            }
            if exact && phase.tree_depth_score {
                if phase_best_score.is_none_or(|best| exceeds(candidate.score, best)) {
                    phase_best_score = Some(candidate.score);
                }
                if config.stop_after_improved_subtree
                    && exceeds(1e12, inherited_size)
                    && exceeds(inherited_size, candidate.subtree_size)
                {
                    stop_phase = true;
                }
            }
        }
        retain_trial_prefix(&mut candidates, processed_prefix, |c| c.usable);
        if phase.tree_depth_score {
            let mut indices: Vec<_> = (0..candidates.len()).collect();
            crate::cuts::shared::stdsort::stable_sort(&mut indices, |&a, &b| {
                strong::tree_depth_score_less(
                    (
                        candidates[a].score,
                        candidates[a].first_child_order.unwrap_or(usize::MAX),
                    ),
                    (
                        candidates[b].score,
                        candidates[b].first_child_order.unwrap_or(usize::MAX),
                    ),
                )
            });
            let mut slots: Vec<_> = candidates.into_iter().map(Some).collect();
            candidates = indices
                .into_iter()
                .map(|index| slots[index].take().unwrap())
                .collect();
        } else {
            candidates.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.order.cmp(&b.order)));
        }
        if evaluated > 0 && !phase.tree_depth_score {
            strong::retain_distinct_trials(&mut candidates, |c| (c.score, c.estimates.clone()));
        }
    }
    for mut candidate in candidates.into_iter().filter(|c| c.usable) {
        if candidate.children.is_none() {
            match candidate.proposal.children(&output.snapshot) {
                Ok(children) => candidate.initialize_children(children, &output.snapshot),
                Err(_) if candidate.proposal.unsupported_resource() => continue,
                Err(error) => return Err(error),
            }
        }
        let mut children = candidate.children.take().unwrap();
        for (side, child) in children.iter_mut().enumerate() {
            let mut proofs = candidate.proof_bounds[side];
            if let Some(evaluated) = &mut candidate.evaluated[side] {
                evaluated.snapshot.state.mapped_subtree_size = candidate.subtree_size;
                evaluated.snapshot.state.mapped_subtree_priority = priority_cutoff;
                proofs.0 = proofs.0.max(evaluated.master.numerical_bound);
                proofs.1 = proofs.1.max(evaluated.master.certified_bound);
                *child = evaluated.snapshot.fork()?;
            } else {
                // Carry the setup and independently published proof channels,
                // without treating an unfinished output as a completed solve.
                // A heuristic/RMP objective can rank but cannot replace these.
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
            // Forking an objective-fixed trial correctly restores the original
            // pricing domain and clears its domain-dependent warm state. The
            // independently published channels (including the inherited tree
            // floor) remain valid bounds for this original child disjunction.
            // Restore those channels after either completed or partial setup;
            // restricted/heuristic ranking estimates never enter this path.
            child.warm.bound_mode = output.snapshot.warm.bound_mode;
            child.warm.numerical_bound = proofs.0;
            child.warm.certified_bound = proofs.1;
            child.warm.bound = match child.warm.bound_mode {
                cg::BoundMode::Numerical => proofs.0,
                cg::BoundMode::Certified => proofs.1,
            };
            child.state.mapped_subtree_size = candidate.subtree_size;
            child.state.mapped_subtree_priority = priority_cutoff;
            // Completed exact trials are reused without another treatment.
            // Nonexact groups are committed at the stage after all configured
            // screens, even when earlier screens did not need to run.
            child.state.branch_history_origin =
                if candidate.proposal.history_eligible && candidate.evaluated[side].is_none() {
                    Some(BranchHistoryOrigin {
                        key: candidate.proposal.history_key.clone(),
                        name: candidate.proposal.name.clone(),
                        depth: output.snapshot.state.node_depth,
                        side,
                        stage: if candidate.last_exact == Some(false) {
                            config.phases.len() + 1
                        } else {
                            candidate.stages[side].max(1)
                        },
                        parent,
                        distance: candidate.proposal.distance(side),
                    })
                } else {
                    None
                };
        }
        let order = (children.len() > 2 || candidate.generator_order).then(|| {
            candidate
                .trial_order
                .take()
                .unwrap_or_else(|| (0..children.len()).collect())
        });
        return Ok(Some(Selection {
            children,
            evaluated: candidate.evaluated,
            order,
            // The committed group keeps the last phase's sorted vector.
            // Selection::admit applies this order to both canonical arrays.
            direction: candidate
                .trial_order
                .filter(|sides| sides.len() == 2)
                .map(|sides| {
                    if sides[0] == 1 {
                        NetworkBranchDirection::UpperFirst
                    } else {
                        NetworkBranchDirection::LowerFirst
                    }
                })
                .or_else(|| {
                    options
                        .branching_registry()
                        .branch_directions
                        .get(&candidate.proposal.name)
                        .copied()
                })
                .or_else(|| family_direction(&candidate.proposal, options)),
        }));
    }
    Ok(None)
}

fn exceeds(a: f64, b: f64) -> bool {
    a - (1e-6 + a.abs().max(b.abs()) * 1e-10) > b
}

/// Preserve group order at the numerical tie band. Only pricing phases
/// reorder children; the restricted-master screen follows generator order.
fn trial_sides<S: AsMut<[usize]>>(mut sides: S, estimates: impl AsRef<[f64]>, pricing: bool) -> S {
    if pricing {
        let estimates = estimates.as_ref();
        let order = sides.as_mut();
        if order.len() == 2 {
            if exceeds(estimates[order[1]], estimates[order[0]]) {
                order.swap(0, 1);
            }
        } else {
            crate::cuts::shared::stdsort::stable_sort(order, |a, b| {
                exceeds(estimates[*a], estimates[*b])
            });
        }
    }
    sides
}

fn configure_trial_options(node: &mut Options, phase: &strong::StrongPhase, incumbent: f64) {
    node.integer_primal_termination = phase.is_exact();
    node.objective_cutoff = incumbent.is_finite().then_some(incumbent);
    node.fixing_upper_bound = incumbent.is_finite().then_some(incumbent);
    node.reduced_cost_fixing &= phase.reduced_cost_fixing;
    if !node.reduced_cost_fixing {
        node.fixing_upper_bound = None;
        node.gap_enumeration = false;
        node.gap_mip = None;
    }
    if !phase.reduced_cost_fixing {
        node.enumeration = None;
    }
    // Exact treatment retains the ordinary node's minimum cut-round policy;
    // nonexact screens use the phase-specific minimum instead.
    if !phase.is_exact() {
        node.cg.min_cut_rounds = phase.min_cut_rounds;
        node.separation.rank_one.min_rounds_before_stop_by_sp = phase.min_cut_rounds;
    }
    node.cg.max_cut_rounds = phase.max_cut_rounds;
    node.cg.max_iterations = phase.max_iterations;
}

fn partial_group_cannot_improve(
    parent: f64,
    cutoff: f64,
    estimates: impl AsRef<[f64]>,
    processed: usize,
    conservativeness: f64,
    best: f64,
) -> bool {
    exceeds(best, -1e12)
        && !exceeds(
            strong::partial_tree_depth_score(
                parent,
                cutoff,
                estimates,
                processed,
                conservativeness,
            ),
            best,
        )
}

fn retain_trial_prefix<T>(
    candidates: &mut Vec<T>,
    processed_prefix: usize,
    usable: impl FnMut(&T) -> bool,
) {
    if processed_prefix > 0 {
        // Truncate before removing rejected groups: a discarded trial must
        // never move an unvisited successor inside the evaluated prefix.
        candidates.truncate(processed_prefix);
    }
    candidates.retain(usable);
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/tree/strong/model_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../../tests/unit/master/tree/strong/count_ordering_tests.rs"]
mod count_ordering_tests;

#[cfg(test)]
#[path = "../../../../tests/unit/master/tree/strong/family_allocation_model_tests.rs"]
mod family_allocation_model_tests;

#[cfg(test)]
#[path = "../../../../tests/unit/master/tree/strong/proposal_provenance_tests.rs"]
mod proposal_provenance_tests;

#[cfg(test)]
#[path = "../../../../tests/unit/master/tree/strong/component_history_tests.rs"]
mod component_history_tests;
