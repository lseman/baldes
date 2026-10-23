//! Direct resource operations shared by physical-graph labels and path replay.
use super::*;
mod accounting;
pub(super) mod bounds;
mod bucketing;
#[cfg(feature = "clp")]
mod certificate;
mod charging;
mod continuation;
mod dependency_order;
mod domains;
mod dominance;
mod front;
mod frozen;
mod paired;
pub(super) mod relation;
mod reversal;
mod slope_interval;
pub(super) mod slope_paired;
mod storage;
use storage::StateValues;
mod transitions;
use bounds::{CostEnvelope, PreparedBounds};
pub(crate) use bounds::{CoupledPairSummary, CoupledStateBounds};
use transitions::PreparedTransitions;

#[derive(Clone, Debug, PartialEq)]
struct AffineBound {
    coefficients: StateValues<f64, 4>,
    offset: f64,
    bounds: Window,
}
#[derive(Clone, Debug, PartialEq)]
struct Suffix {
    bounds: Vec<AffineBound>,
    coefficients: StateValues<f64, 4>,
    constant: f64,
    program: continuation::Continuation,
    state_costs: CostEnvelope,
}
/// Owned specialized state. Numeric suffixes retain affine entry conditions;
/// they are not evaluated against an invented arrival value during backward
/// extension. Only a forward/backward join supplies that value.
#[derive(Clone, Debug, PartialEq)]
pub struct CoupledLabelState {
    selection: StateValues<u64, 2>,
    loads: StateValues<[f64; 3], 1>,
    reversal_peaks: Option<Box<StateValues<f64, 1>>>,
    numeric: StateValues<f64, 4>,
    suffix: Option<Box<Suffix>>,
    fronts: Vec<front::CostFront>,
    slope_fronts: Vec<slope_interval::SlopeLabel>,
    slope_relations: Vec<slope_interval::SlopeLabel>,
    relations: Vec<paired::PairedLabel>,
    charging: Vec<charging::ChargingLabel>,
    numeric_cost: f64,
    resource_objective: f64,
    accounting: Option<accounting::ScalarCosts>,
}
/// Prepared resource definitions over the unchanged physical arc vector.
#[derive(Clone, Debug, PartialEq)]
pub struct CoupledKernel {
    // Shared by semantic clones; cost suppression receives a fresh token.
    identity: std::sync::Arc<()>,
    definitions: std::sync::Arc<[CoupledResource]>,
    slots: Vec<Option<usize>>,
    initial: Vec<f64>,
    cost_scale: f64,
    selection_initial: Vec<u64>,
    load_capacities: Vec<f64>,
    reversal_union: bool,
    cost_resource: bool,
    front_indices: Vec<usize>,
    slope_indices: Vec<usize>,
    slope_roots: Vec<slope_interval::SlopeLabel>,
    slope_charges: Vec<std::sync::Arc<[slope_interval::PreparedSlopeCharge]>>,
    slope_paired: Vec<slope_paired::PreparedSlopePaired>,
    relation_indices: Vec<usize>,
    paired: Vec<paired::PreparedPaired>,
    charging: Vec<charging::PreparedCharging>,
    front_bucket_policy: bucketing::FrontBucketPolicy,
    frozen: frozen::PreparedFrozen,
    arc_count: usize,
    optimization: PreparedBounds,
    numeric_cost_order: dominance::NumericCostOrder,
    transitions: PreparedTransitions,
    route_domains: domains::RouteDomains,
}
/// Reusable per-thread resource-extension scratch. Retained labels own all
/// their state and never borrow temporary workspace storage.
#[derive(Default)]
pub struct CoupledWorkspace {
    relation: relation::RelationWorkspace,
}
fn dot(a: &[f64], b: &[f64]) -> Result<f64> {
    a.iter()
        .zip(b)
        .try_fold(0., |sum, (&x, &y)| finite(sum + finite(x * y)?))
}
fn combine_load(a: [f64; 3], b: [f64; 3]) -> Result<[f64; 3]> {
    Ok([
        finite(a[0] + b[0])?,
        finite(a[1] + b[1])?,
        a[2].max(finite(a[1] + b[2])?),
    ])
}
impl CoupledKernel {
    pub(crate) fn prepare<const D: usize>(
        g: &Graph<D>,
        definitions: &[CoupledResource],
    ) -> Result<Self> {
        let mut initial = Vec::new();
        let mut slots = Vec::new();
        for (i, r) in definitions.iter().enumerate() {
            r.validate(g, i, definitions)?;
            match r {
                CoupledResource::CumulativeCost { initial: x, .. }
                | CoupledResource::Dependent { initial: x, .. }
                | CoupledResource::PiecewiseLinearDependent { initial: x, .. }
                | CoupledResource::PiecewiseLinearResource { initial: x, .. }
                | CoupledResource::EncodedPiecewiseLinearResource { initial: x, .. }
                | CoupledResource::EncodedPiecewiseLinearDependent { initial: x, .. } => {
                    slots.push(Some(initial.len()));
                    initial.push(*x);
                }
                _ => slots.push(None),
            }
        }
        let numeric_cost_order =
            dominance::NumericCostOrder::new(definitions, &slots, initial.len());
        let optimization = PreparedBounds::new(g, definitions, &slots, &initial);
        let transitions =
            PreparedTransitions::new(definitions, &slots, initial.len(), g.arcs.len(), 1.);
        let route_domains =
            domains::RouteDomains::prepare(g, &transitions, &optimization.numeric, &initial);
        let front_indices: Vec<_> = definitions
            .iter()
            .enumerate()
            .filter_map(|(i, r)| matches!(r, CoupledResource::IntervalCost { .. }).then_some(i))
            .collect();
        let slope_indices: Vec<_> = definitions
            .iter()
            .enumerate()
            .filter_map(|(i, r)| {
                matches!(r, CoupledResource::SlopeIntervalCost { .. }).then_some(i)
            })
            .collect();
        let slope_roots = slope_indices
            .iter()
            .map(|&i| {
                let CoupledResource::SlopeIntervalCost { initial, .. } = &definitions[i] else {
                    unreachable!()
                };
                descriptor::SlopeFront::zero(initial.lower, initial.upper)
                    .map(|front| slope_interval::SlopeLabel::root(&front, true))
                    .map_err(|error| invalid(error.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        let slope_charges = slope_indices
            .iter()
            .map(|&i| {
                let CoupledResource::SlopeIntervalCost { functions, .. } = &definitions[i] else {
                    unreachable!()
                };
                functions
                    .iter()
                    .map(slope_interval::PreparedSlopeCharge::new)
                    .collect::<Result<Vec<_>>>()
                    .map(std::sync::Arc::<[slope_interval::PreparedSlopeCharge]>::from)
            })
            .collect::<Result<Vec<_>>>()?;
        let frozen = frozen::PreparedFrozen::new(g, definitions, &front_indices);
        let slope_paired = definitions
            .iter()
            .filter_map(|resource| match resource {
                CoupledResource::SlopePaired {
                    initial,
                    initial_base,
                    arcs,
                    mode,
                    normalization,
                    policy,
                } => Some(slope_paired::PreparedSlopePaired::new(
                    initial,
                    *initial_base,
                    arcs,
                    *mode,
                    *normalization,
                    *policy,
                )),
                _ => None,
            })
            .collect::<Result<Vec<_>>>()?;
        let paired = definitions
            .iter()
            .filter_map(|r| match r {
                CoupledResource::PairedPiecewiseLinear {
                    initial,
                    initial_base,
                    arcs,
                    mode,
                } => Some(paired::PreparedPaired::new(
                    initial,
                    *initial_base,
                    arcs,
                    *mode,
                )),
                _ => None,
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            reversal_union: definitions.iter().any(|resource| match resource {
                CoupledResource::SimultaneousPickupDelivery {
                    deliveries,
                    pickups,
                    ..
                } => {
                    deliveries.iter().zip(pickups).any(|(d, p)| p > d)
                        && deliveries.iter().zip(pickups).any(|(d, p)| p < d)
                }
                _ => false,
            }),
            identity: std::sync::Arc::new(()),
            definitions: definitions.to_vec().into(),
            slots,
            initial,
            cost_scale: 1.,
            frozen,
            paired,
            charging: definitions
                .iter()
                .filter_map(|r| match r {
                    CoupledResource::Charging {
                        battery,
                        initial_energy,
                        initial_time,
                        arcs,
                        max_work,
                    } => Some(charging::PreparedCharging::new(
                        *battery,
                        *initial_energy,
                        *initial_time,
                        arcs,
                        *max_work,
                    )),
                    _ => None,
                })
                .collect(),
            front_bucket_policy: Default::default(),
            selection_initial: definitions
                .iter()
                .filter_map(|r| match r {
                    CoupledResource::Selection { initial, .. } => Some(*initial),
                    _ => None,
                })
                .collect(),
            load_capacities: definitions
                .iter()
                .filter_map(|r| match r {
                    CoupledResource::SimultaneousPickupDelivery { capacity, .. } => Some(*capacity),
                    _ => None,
                })
                .collect(),
            relation_indices: definitions
                .iter()
                .enumerate()
                .filter_map(|(i, r)| {
                    matches!(r, CoupledResource::PairedPiecewiseLinear { .. }).then_some(i)
                })
                .collect(),
            front_indices,
            slope_indices,
            slope_roots,
            slope_charges,
            slope_paired,
            cost_resource: definitions.iter().any(|r| {
                matches!(
                    r,
                    CoupledResource::CumulativeCost { .. }
                        | CoupledResource::IntervalCost { .. }
                        | CoupledResource::SlopeIntervalCost { .. }
                        | CoupledResource::PiecewiseLinearCost { .. }
                        | CoupledResource::EncodedPiecewiseLinearCost { .. }
                        | CoupledResource::SlopePaired {
                            mode: PairedResourceMode::CostUnbounded
                                | PairedResourceMode::CostBounded,
                            ..
                        }
                        | CoupledResource::PairedPiecewiseLinear {
                            mode: PairedResourceMode::CostUnbounded
                                | PairedResourceMode::CostBounded,
                            ..
                        }
                )
            }),
            arc_count: g.arcs.len(),
            optimization,
            numeric_cost_order,
            transitions,
            route_domains,
        })
    }
    /// Configure optional borrowed paired-front indexing views. Zero disables
    /// subdivision; no label, path, or objective is split or renormalized.
    pub fn configure_paired_front_views(
        &mut self,
        pieces_per_view: usize,
        max_views: usize,
    ) -> Result<()> {
        self.front_bucket_policy = match (
            std::num::NonZeroUsize::new(pieces_per_view),
            std::num::NonZeroUsize::new(max_views),
        ) {
            (Some(pieces_per_view), Some(max_views)) => bucketing::FrontBucketPolicy::PieceGroups {
                pieces_per_view,
                max_views,
            },
            (None, None) => bucketing::FrontBucketPolicy::Disabled,
            _ => {
                return Err(invalid(
                    "paired view width and budget must both be positive, or both zero",
                ))
            }
        };
        Ok(())
    }
    /// Conservative complete-route preprocessing diagnostics. A work-limit
    /// stop leaves safe, potentially loose domains and does not prove failure.
    pub fn preprocessing(&self) -> CoupledPreprocessing {
        CoupledPreprocessing {
            passes: self.route_domains.passes,
            converged: self.route_domains.converged,
            arcs: self.arc_count,
            usable_arcs: self
                .route_domains
                .possible
                .iter()
                .filter(|&&possible| possible)
                .count(),
        }
    }
    /// Whether preprocessing retained this physical arc for a complete route.
    /// True is a necessary condition, not a feasibility certificate.
    pub fn route_arc_possible(&self, arc: usize) -> Result<bool> {
        self.route_domains
            .possible
            .get(arc)
            .copied()
            .ok_or_else(|| invalid("unknown coupled resource arc"))
    }
    pub(crate) fn cache_identity(&self) -> &std::sync::Arc<()> {
        &self.identity
    }
    pub fn definitions(&self) -> &[CoupledResource] {
        &self.definitions
    }
    pub(crate) fn validate_for<const D: usize>(&self, g: &Graph<D>) -> Result<()> {
        for (index, definition) in self.definitions.iter().enumerate() {
            definition.validate(g, index, &self.definitions)?;
        }
        let optimization = PreparedBounds::new(g, &self.definitions, &self.slots, &self.initial);
        if self.arc_count != g.arcs.len()
            || self.optimization != optimization
            || !self.route_domains.matches(g)
        {
            return Err(invalid("reinstall coupled resources after changing elementarity or physical graph topology"));
        }
        Ok(())
    }
    pub fn has_costs(&self) -> bool {
        self.cost_scale != 0. && self.cost_resource
    }
    #[cfg(feature = "clp")]
    pub(crate) fn suppress_costs(&mut self) {
        self.identity = std::sync::Arc::new(());
        self.cost_scale = 0.;
        self.transitions.suppress_costs(self.initial.len());
    }
    pub fn root(&self, forward: bool) -> Box<CoupledLabelState> {
        Box::new(CoupledLabelState {
            selection: StateValues::from_slice(&self.selection_initial),
            loads: StateValues::filled(self.load_capacities.len(), [0.; 3]),
            reversal_peaks: None,
            numeric: StateValues::from_slice(&self.initial),
            slope_fronts: self
                .slope_roots
                .iter()
                .map(|root| {
                    if forward {
                        root.clone()
                    } else {
                        slope_interval::SlopeLabel {
                            front: None,
                            minimum: 0.,
                        }
                    }
                })
                .collect(),
            slope_relations: self
                .slope_paired
                .iter()
                .map(|paired| paired.root(forward))
                .collect(),
            numeric_cost: 0.,
            resource_objective: 0.,
            accounting: Some(accounting::ScalarCosts::default()),
            relations: self.paired.iter().map(|p| p.root(forward)).collect(),
            charging: self.charging.iter().map(|p| p.root(forward)).collect(),
            fronts: self
                .front_indices
                .iter()
                .map(|&i| match &self.definitions[i] {
                    CoupledResource::IntervalCost {
                        initial, bounds, ..
                    } => front::CostFront::zero(if forward { *initial } else { *bounds }),
                    _ => unreachable!(),
                })
                .collect(),
            suffix: (!forward).then(|| {
                Box::new(Suffix {
                    bounds: vec![],
                    coefficients: StateValues::filled(self.initial.len(), 0.),
                    constant: 0.,
                    program: Default::default(),
                    state_costs: Default::default(),
                })
            }),
        })
    }
    /// Zero-compensation dominance with directional load ordering and equal
    /// numeric entry conditions. Use `dominance_penalty` for cost tradeoffs.
    pub fn dominates(a: Option<&CoupledLabelState>, b: Option<&CoupledLabelState>) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                let account_safe = a.numeric_cost <= b.numeric_cost
                    && match (a.accounting, b.accounting) {
                        (Some(a_cost), Some(b_cost)) => a_cost.reduced <= b_cost.reduced,
                        (None, None) => true,
                        _ => false,
                    };
                account_safe
                    && dominance::loads_dominate(a, b)
                    && a.fronts == b.fronts
                    && a.slope_fronts == b.slope_fronts
                    && a.slope_relations == b.slope_relations
                    && a.relations == b.relations
                    && a.charging == b.charging
                    && a.numeric == b.numeric
                    && a.suffix == b.suffix
                    && a.selection.len() == b.selection.len()
                    && a.selection
                        .iter()
                        .zip(&b.selection)
                        .all(|(a, b)| a & b == *b)
            }
            _ => false,
        }
    }
    /// Extend a physical arc using the necessary complete-route filter. The
    /// returned increment belongs to both reduced and original objectives.
    /// Arithmetic failure is distinct from infeasibility. Standalone partial
    /// paths can be evaluated separately with `replay` or `replay_state`.
    pub fn extend(
        &self,
        state: &CoupledLabelState,
        arc: usize,
        forward: bool,
    ) -> Result<Option<(Box<CoupledLabelState>, f64)>> {
        self.extend_with_workspace(state, arc, forward, &mut CoupledWorkspace::default())
    }
    pub fn extend_with_workspace(
        &self,
        state: &CoupledLabelState,
        arc: usize,
        forward: bool,
        workspace: &mut CoupledWorkspace,
    ) -> Result<Option<(Box<CoupledLabelState>, f64)>> {
        self.extend_internal(state, arc, forward, true, false, workspace)
    }
    fn extend_internal(
        &self,
        state: &CoupledLabelState,
        arc: usize,
        forward: bool,
        route_filter: bool,
        reversal_union: bool,
        workspace: &mut CoupledWorkspace,
    ) -> Result<Option<(Box<CoupledLabelState>, f64)>> {
        if arc >= self.arc_count {
            return Err(invalid("unknown coupled resource arc"));
        }
        self.check_state(state)?;
        if forward == state.suffix.is_some() {
            return Err(invalid("coupled label direction mismatch"));
        }
        if self.transitions.arcs[arc].impossible
            || (route_filter && !self.route_domains.possible[arc])
        {
            return Ok(None);
        }
        // The old suffix is read below while its replacement is composed.
        // Cloning it here would copy every old entry condition only to discard it.
        let mut next = CoupledLabelState {
            selection: state.selection.clone(),
            loads: state.loads.clone(),
            reversal_peaks: state.reversal_peaks.clone(),
            numeric: state.numeric.clone(),
            suffix: None,
            fronts: Vec::with_capacity(state.fronts.len()),
            slope_fronts: Vec::with_capacity(state.slope_fronts.len()),
            slope_relations: Vec::with_capacity(state.slope_relations.len()),
            relations: Vec::with_capacity(state.relations.len()),
            charging: Vec::with_capacity(state.charging.len()),
            numeric_cost: 0.,
            resource_objective: 0.,
            accounting: None,
        };
        let prepared = &self.transitions.arcs[arc];
        for (domain, &allowed) in next.selection.iter_mut().zip(&prepared.selection) {
            *domain &= allowed;
            if *domain == 0 {
                return Ok(None);
            }
        }
        for (index, ((load, &step), &capacity)) in next
            .loads
            .iter_mut()
            .zip(&prepared.loads)
            .zip(&self.load_capacities)
            .enumerate()
        {
            if let Some(peaks) = &mut next.reversal_peaks {
                peaks[index] = finite(peaks[index] + step[1])?.max(0.);
            }
            *load = if forward {
                combine_load(*load, step)?
            } else {
                combine_load(step, *load)?
            };
            if finite(load[0] + load[2])? > capacity
                && !(reversal_union
                    && forward
                    && next
                        .reversal_peaks
                        .as_ref()
                        .is_some_and(|peaks| load[0] + peaks[index] <= capacity))
            {
                return Ok(None);
            }
        }
        for (old, &i) in state.fronts.iter().zip(&self.front_indices) {
            let CoupledResource::IntervalCost {
                bounds,
                increments,
                functions,
                ..
            } = &self.definitions[i]
            else {
                unreachable!()
            };
            let Some(front) = old.extend(
                increments[arc],
                &functions[arc],
                *bounds,
                forward,
                self.cost_scale,
            )?
            else {
                return Ok(None);
            };
            next.fronts.push(front);
        }
        for ((old, &index), charges) in state
            .slope_fronts
            .iter()
            .zip(&self.slope_indices)
            .zip(&self.slope_charges)
        {
            let Some(front) = old.extend(
                &self.definitions[index],
                &charges[arc],
                arc,
                forward,
                self.cost_scale,
            )?
            else {
                return Ok(None);
            };
            next.slope_fronts.push(front);
        }
        for (old, prepared) in state.slope_relations.iter().zip(&self.slope_paired) {
            let Some(front) = prepared.extend(old, arc, forward, self.cost_scale)? else {
                return Ok(None);
            };
            next.slope_relations.push(front);
        }
        let first_backward = state
            .suffix
            .as_ref()
            .is_some_and(|s| s.program.iter().next().is_none());
        for (old, prepared) in state.relations.iter().zip(&self.paired) {
            let Some(label) =
                prepared.extend(old, arc, forward, first_backward, &mut workspace.relation)?
            else {
                return Ok(None);
            };
            next.relations.push(label);
        }
        for (old, prepared) in state.charging.iter().zip(&self.charging) {
            let Some(label) = prepared.extend(old, arc, forward)? else {
                return Ok(None);
            };
            next.charging.push(label);
        }
        let cost = if forward {
            let mut cost = state.numeric_cost;
            for operation in &self.transitions.arcs[arc].operations {
                if !operation.apply(&mut next.numeric, self.cost_scale, &mut cost)?
                    || !operation.feasible(&next.numeric)
                {
                    return Ok(None);
                }
            }
            cost
        } else {
            let old = state.suffix.as_ref().unwrap();
            let n = self.initial.len();
            let transition = self.transitions.arcs[arc]
                .backward
                .as_ref()
                .ok_or_else(|| invalid("nonfinite coupled resource arithmetic"))?;
            let matrix = &transition.matrix;
            let offsets = &transition.offsets;
            let mut coefficients = StateValues::<f64, 4>::from_slice(&transition.coefficients);
            let mut constant = transition.constant;
            let mut bounds_out = transition.bounds.clone();
            // Prepend this arc to the suffix: transform every entry condition
            // and its cost function through the arc's triangular affine map.
            for bound in &old.bounds {
                let mut row = StateValues::<f64, 4>::filled(n, 0.);
                for (j, value) in row.iter_mut().enumerate() {
                    for (i, entry) in matrix.iter().enumerate() {
                        *value = finite(*value + finite(bound.coefficients[i] * entry[j])?)?;
                    }
                }
                bounds_out.push(AffineBound {
                    coefficients: row,
                    offset: finite(bound.offset + dot(&bound.coefficients, offsets)?)?,
                    bounds: bound.bounds,
                });
            }
            if self.transitions.deferred {
                for (value, &previous) in coefficients.iter_mut().zip(&old.coefficients) {
                    *value = finite(*value + previous)?;
                }
            } else {
                for (j, value) in coefficients.iter_mut().enumerate() {
                    for (i, entry) in matrix.iter().enumerate() {
                        *value = finite(*value + finite(old.coefficients[i] * entry[j])?)?;
                    }
                }
            }
            constant = finite(constant + finite(old.constant + dot(&old.coefficients, offsets)?)?)?;
            let new_cost = finite(constant + dot(&coefficients, &self.initial)?)?;
            let state_costs = self.transitions.arcs[arc]
                .state_costs
                .combine(&old.state_costs);
            next.suffix = Some(Box::new(Suffix {
                bounds: bounds_out,
                state_costs,
                coefficients,
                constant,
                program: if self.transitions.deferred
                    || !self.relation_indices.is_empty()
                    || self.cost_resource
                    || !self.slope_paired.is_empty()
                    || !self.charging.is_empty()
                {
                    old.program.prepend(arc)
                } else {
                    Default::default()
                },
            }));
            new_cost
        };
        next.numeric_cost = cost;
        let mut objective = next.numeric_cost;
        for front in &next.fronts {
            objective = finite(objective + front.minimum())?;
        }
        for front in &next.slope_fronts {
            objective = finite(objective + front.minimum)?;
        }
        for front in &next.slope_relations {
            objective = finite(objective + front.minimum)?;
        }
        for (prepared, label) in self.paired.iter().zip(&next.relations) {
            objective = finite(objective + prepared.objective(label, self.cost_scale)?)?;
        }
        next.resource_objective = objective;
        Ok(Some((
            Box::new(next),
            finite(objective - state.resource_objective)?,
        )))
    }
    /// Join at one physical vertex. All arc services are accounted for once.
    pub fn join(&self, f: &CoupledLabelState, b: &CoupledLabelState) -> Result<Option<f64>> {
        self.check_state(f)?;
        self.check_state(b)?;
        if f.suffix.is_some() || b.suffix.is_none() {
            return Err(invalid("coupled join direction mismatch"));
        }
        if self.has_slope_fronts() {
            let Some(total) = self.join_resource_objective(f, b)? else {
                return Ok(None);
            };
            return Ok(Some(finite(
                finite(total - f.resource_objective)? - b.resource_objective,
            )?));
        }
        if f.selection
            .iter()
            .zip(&b.selection)
            .any(|(a, b)| a & b == 0)
        {
            return Ok(None);
        }
        for ((&left, &right), &capacity) in f.loads.iter().zip(&b.loads).zip(&self.load_capacities)
        {
            let load = combine_load(left, right)?;
            if finite(load[0] + load[2])? > capacity {
                return Ok(None);
            }
        }
        let mut front_correction = 0.;
        for (left, right) in f.fronts.iter().zip(&b.fronts) {
            let Some(joined) = left.joined_minimum(right)? else {
                return Ok(None);
            };
            front_correction = finite(
                front_correction + finite(finite(joined - left.minimum())? - right.minimum())?,
            )?;
        }
        let mut workspace = CoupledWorkspace::default();
        for ((left, right), prepared) in f.relations.iter().zip(&b.relations).zip(&self.paired) {
            if !paired::domains_may_intersect(left, right, self.front_bucket_policy) {
                return Ok(None);
            }
            let joined = match prepared.cached_join(left, right) {
                Some(Some(value)) => value,
                Some(None) => return Ok(None),
                None => {
                    let Some(value) = prepared.replay_suffix(
                        left,
                        &b.suffix.as_ref().unwrap().program,
                        &mut workspace.relation,
                    )?
                    else {
                        return Ok(None);
                    };
                    value
                }
            };
            if self.cost_scale != 0. {
                let baseline = left
                    .forward()
                    .ok_or_else(|| invalid("paired forward join expected"))?
                    .minimum();
                // Feasibility-only paired relations have no objective.
                if prepared.has_costs() {
                    front_correction = finite(
                        front_correction + finite(self.cost_scale * finite(joined - baseline)?)?,
                    )?;
                }
            }
        }
        let suffix = b.suffix.as_ref().unwrap();
        for (front, prepared) in f.charging.iter().zip(&self.charging) {
            if !prepared.join(front, &suffix.program)? {
                return Ok(None);
            }
        }
        let mut state_cost = 0.;
        if self.transitions.deferred {
            let mut numeric = f.numeric.clone();

            for arc in suffix.program.iter() {
                for operation in &self.transitions.arcs[arc].operations {
                    let scale =
                        if matches!(operation, transitions::NumericOperation::StateCost { .. }) {
                            self.cost_scale
                        } else {
                            0.
                        };
                    if !operation.apply(&mut numeric, scale, &mut state_cost)?
                        || !operation.feasible(&numeric)
                    {
                        return Ok(None);
                    }
                }
            }
        }
        for bound in &suffix.bounds {
            let value = finite(bound.offset + dot(&bound.coefficients, &f.numeric)?)?;
            if value < bound.bounds.lower || value > bound.bounds.upper {
                return Ok(None);
            }
        }
        let entry_cost = dot(&suffix.coefficients, &f.numeric)?;
        let baseline = dot(&suffix.coefficients, &self.initial)?;
        Ok(Some(finite(
            finite(finite(entry_cost - baseline)? + state_cost)? + front_correction,
        )?))
    }
    /// Physical suffix arcs in forward traversal order. Charged resources
    /// retain this chain even when their numeric transition is affine.
    pub(crate) fn suffix_arcs<'a>(
        &self,
        b: &'a CoupledLabelState,
    ) -> impl Iterator<Item = usize> + 'a {
        b.suffix
            .as_ref()
            .into_iter()
            .flat_map(|suffix| suffix.program.iter())
    }
    /// Return the complete joined resource objective directly. Reconstructing
    /// it from two rounded minima plus a correction loses scalar information.
    pub(crate) fn join_resource_objective(
        &self,
        f: &CoupledLabelState,
        b: &CoupledLabelState,
    ) -> Result<Option<f64>> {
        self.check_state(f)?;
        self.check_state(b)?;
        if f.suffix.is_some() || b.suffix.is_none() {
            return Err(invalid("coupled join direction mismatch"));
        }
        if f.selection
            .iter()
            .zip(&b.selection)
            .any(|(a, b)| a & b == 0)
        {
            return Ok(None);
        }
        for ((&left, &right), &capacity) in f.loads.iter().zip(&b.loads).zip(&self.load_capacities)
        {
            let load = combine_load(left, right)?;
            if finite(load[0] + load[2])? > capacity {
                return Ok(None);
            }
        }
        // Numeric-only canonical joins retain one state vector and literal
        // cost accumulator. Rebuilding a boxed physical label per suffix arc
        // would only copy already-checked selection and load summaries.
        if self.front_indices.is_empty()
            && !self.has_slope_fronts()
            && self.relation_indices.is_empty()
            && self.charging.is_empty()
        {
            let mut numeric = f.numeric.clone();
            let mut cost = f.numeric_cost;
            for arc in self.suffix_arcs(b) {
                let prepared = &self.transitions.arcs[arc];
                if prepared.impossible {
                    return Ok(None);
                }
                for operation in &prepared.operations {
                    if !operation.apply(&mut numeric, self.cost_scale, &mut cost)?
                        || !operation.feasible(&numeric)
                    {
                        return Ok(None);
                    }
                }
            }
            return Ok(Some(cost));
        }
        // In the restricted exact bounded-cost specialization the cached
        // final dependent minimum is already canonical; avoid suffix replay.
        if self.front_indices.is_empty()
            && !self.has_slope_fronts()
            && self.initial.is_empty()
            && !self.paired.is_empty()
            && self.charging.is_empty()
        {
            let mut total = 0.;
            let mut known = true;
            for ((left, right), prepared) in f.relations.iter().zip(&b.relations).zip(&self.paired)
            {
                match prepared.cached_join(left, right) {
                    Some(Some(final_minimum)) => {
                        total = finite(
                            total + prepared.final_objective(final_minimum, self.cost_scale)?,
                        )?
                    }
                    Some(None) => return Ok(None),
                    None => {
                        known = false;
                        break;
                    }
                }
            }
            if known {
                return Ok(Some(total));
            }
        }
        let mut workspace = CoupledWorkspace::default();
        let mut current: Option<Box<CoupledLabelState>> = None;
        for arc in self.suffix_arcs(b) {
            let input = current.as_deref().unwrap_or(f);
            let Some((next, _)) =
                self.extend_internal(input, arc, true, false, false, &mut workspace)?
            else {
                return Ok(None);
            };
            current = Some(next);
        }
        Ok(Some(current.as_deref().unwrap_or(f).resource_objective))
    }
    pub fn replay(&self, arcs: &[usize]) -> Result<Option<f64>> {
        Ok(self.replay_state(arcs)?.map(|(_, cost)| cost))
    }
    /// Replay a physical arc-position path and return its final coupled state.
    pub fn replay_state(&self, arcs: &[usize]) -> Result<Option<(Box<CoupledLabelState>, f64)>> {
        let mut state = self.root(true);
        let mut workspace = CoupledWorkspace::default();
        for &arc in arcs {
            let Some((next, _)) =
                self.extend_internal(&state, arc, true, false, false, &mut workspace)?
            else {
                return Ok(None);
            };
            state = next;
        }
        let cost = state.resource_objective;
        Ok(Some((state, cost)))
    }
    fn check_state(&self, state: &CoupledLabelState) -> Result<()> {
        if state.charging.len() != self.charging.len()
            || state.charging.iter().any(|label| {
                matches!(label, charging::ChargingLabel::Forward(_)) != state.suffix.is_none()
            })
            || state.relations.len() != self.relation_indices.len()
            || state.fronts.len() != self.front_indices.len()
            || state.slope_fronts.len() != self.slope_indices.len()
            || state.slope_relations.len() != self.slope_paired.len()
            || state
                .slope_relations
                .iter()
                .any(|label| label.front.is_some() != state.suffix.is_none())
            || state
                .slope_fronts
                .iter()
                .any(|label| label.front.is_some() != state.suffix.is_none())
            || state.numeric.len() != self.initial.len()
            || state.selection.len() != self.selection_initial.len()
            || state.loads.len() != self.load_capacities.len()
            || state.reversal_peaks.as_ref().is_some_and(|p| {
                state.suffix.is_some()
                    || p.len() != state.loads.len()
                    || p.iter().any(|x| !x.is_finite())
            })
            || state.suffix.as_ref().is_some_and(|s| {
                s.coefficients.len() != self.initial.len()
                    || s.bounds
                        .iter()
                        .any(|b| b.coefficients.len() != self.initial.len())
            })
        {
            return Err(invalid("coupled resource state shape mismatch"));
        }
        Ok(())
    }
}

impl CoupledLabelState {
    /// Cost knots for each interval resource in declaration order. Forward
    /// functions price prefixes at the current arrival value; backward
    /// functions price suffixes at their entry value. Singleton domains are
    /// represented by one knot. Values include the current objective scale.
    pub fn interval_cost_fronts(&self) -> impl Iterator<Item = &[[f64; 2]]> {
        self.fronts.iter().map(front::CostFront::points)
    }

    /// Affine `[base,dependent]` endpoint pairs for each retained paired
    /// frontier. Disconnected pieces are returned separately. Backward labels
    /// retain direction-specific suffix programs/caches, so this accessor
    /// returns None for backward labels.
    pub fn paired_resource_fronts(&self) -> Option<Vec<Vec<[[f64; 2]; 2]>>> {
        self.suffix.is_none().then(|| {
            self.relations
                .iter()
                .filter_map(|label| label.forward())
                .map(|front| front.pieces().iter().map(|p| [p.left, p.right]).collect())
                .collect()
        })
    }

    pub fn selection_domains(&self) -> &[u64] {
        &self.selection
    }
    /// One [total deliveries, net pickups-minus-deliveries, peak net load]
    /// summary for each simultaneous resource, in declaration order.
    pub fn load_summaries(&self) -> &[[f64; 3]] {
        &self.loads
    }
    /// Forward numeric states, in numeric resource declaration order.
    /// Backward labels carry entry conditions instead of actual arrivals.
    pub fn numeric_values(&self) -> Option<&[f64]> {
        self.suffix.is_none().then_some(&self.numeric[..])
    }
}

impl<const D: usize> Graph<D> {
    /// Attach specialized resource operations to this physical graph. No
    /// vertices or arcs are added. Reinstall after changing topology, terminal
    /// vertices, elementarity or physical arc vector order.
    pub fn set_coupled_resources(&mut self, definitions: &[CoupledResource]) -> Result<()> {
        let kernel = CoupledKernel::prepare(self, definitions)?;
        self.coupled = (!definitions.is_empty()).then(|| Box::new(kernel));
        Ok(())
    }
    pub fn coupled_kernel(&self) -> Option<&CoupledKernel> {
        self.coupled.as_deref()
    }
    pub fn has_coupled_resources(&self) -> bool {
        self.coupled.is_some()
    }
    #[cfg(feature = "clp")]
    pub(crate) fn suppress_coupled_costs(&mut self) {
        if let Some(kernel) = self.coupled.as_mut() {
            kernel.suppress_costs();
        }
    }
    pub fn coupled_path_cost(&self, arcs: &[usize]) -> Result<Option<f64>> {
        match self.coupled_kernel() {
            Some(k) => k.replay(arcs),
            None => Ok(Some(0.)),
        }
    }
}

#[cfg(all(test, feature = "clp"))]
mod nonlinear_phase_tests {
    use super::*;
    fn model(cost_upper: f64) -> CoupledKernel {
        let mut graph: Graph<2> = Graph::new(3, 0).unwrap();
        graph.add_arc(0, 1, 0.).unwrap();
        graph.add_arc(1, 2, 0.).unwrap();
        CoupledKernel::prepare(
            &graph,
            &[
                CoupledResource::PiecewiseLinearResource {
                    initial: 0.,
                    bounds: Window {
                        lower: 0.,
                        upper: 2.,
                    },
                    functions: vec![
                        PiecewiseLinearFunction::new(vec![[0., 1.], [1., 2.], [2., 2.]])
                            .unwrap();
                        2
                    ],
                },
                CoupledResource::PiecewiseLinearCost {
                    parent: 0,
                    functions: vec![
                        PiecewiseLinearFunction::new(vec![
                            [0., 0.],
                            [cost_upper, 2. * cost_upper]
                        ])
                        .unwrap();
                        2
                    ],
                },
            ],
        )
        .unwrap()
    }
    #[test]
    fn phase_i_suppresses_nonlinear_costs_and_retains_all_numeric_constraints() {
        let mut kernel = model(2.);
        assert_eq!(kernel.replay(&[0, 1]).unwrap(), Some(6.));
        let (valid, [lower, upper]) = kernel.certify(&[0, 1]).unwrap();
        assert!(valid && lower <= 6. && upper >= 6.);
        kernel.suppress_costs();
        assert!(!kernel.has_costs());
        let (state, cost) = kernel.replay_state(&[0, 1]).unwrap().unwrap();
        assert_eq!(state.numeric_values().unwrap(), [2.]);
        assert_eq!(cost, 0.);
        let (f, _) = kernel.extend(&kernel.root(true), 0, true).unwrap().unwrap();
        let (b, bc) = kernel
            .extend(&kernel.root(false), 1, false)
            .unwrap()
            .unwrap();
        assert_eq!(bc, 0.);
        assert_eq!(kernel.join(&f, &b).unwrap(), Some(0.));
        assert_eq!(kernel.certify(&[0, 1]), Some((true, [0., 0.])));
        let mut restricted = model(1.5);
        assert!(restricted.replay(&[0, 1]).unwrap().is_none());
        restricted.suppress_costs();
        assert!(restricted.replay(&[0, 1]).unwrap().is_none());
        assert_eq!(restricted.certify(&[0, 1]), Some((false, [0., 0.])));
    }
}

#[cfg(test)]
mod canonical_numeric_join_tests {
    use super::*;
    #[test]
    fn numeric_suffix_join_retains_literal_order_without_rebuilding_labels() {
        let mut graph = Graph::<2>::new(4, 0).unwrap();
        for (u, v) in [(0, 1), (1, 2), (2, 3)] {
            graph.add_arc(u, v, 0.).unwrap();
        }
        let kernel = CoupledKernel::prepare(
            &graph,
            &[CoupledResource::CumulativeCost {
                initial: 0.,
                increments: vec![1.; 3],
                weights: vec![1e16, -1e16, 1.],
            }],
        )
        .unwrap();
        let expected = kernel.replay(&[0, 1, 2]).unwrap();
        for split in 0..=3 {
            let f = kernel
                .replay_state(&(0..split).collect::<Vec<_>>())
                .unwrap()
                .unwrap()
                .0;
            let mut b = kernel.root(false);
            for arc in (split..3).rev() {
                b = kernel.extend(&b, arc, false).unwrap().unwrap().0;
            }
            assert_eq!(kernel.join_resource_objective(&f, &b).unwrap(), expected);
        }
    }
}
