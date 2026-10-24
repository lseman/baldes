//! Direct resource operations shared by physical-graph labels and path replay.
use super::*;
mod accounting;
pub(super) mod bounds;
mod bucketing;
mod budgeted;
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

mod extension;
mod joining;
mod preparation;
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
    /// Budgeted worst-case vectors, concatenated in declaration order.
    budgets: Vec<f64>,
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
    budgeted: Vec<budgeted::PreparedBudgeted>,
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
            || state.budgets.len() != budgeted::PreparedBudgeted::width(&self.budgeted)
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
#[path = "../../../../tests/unit/graph/coupled/kernel/nonlinear_phase_tests.rs"]
mod nonlinear_phase_tests;

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/kernel/canonical_numeric_join_tests.rs"]
mod canonical_numeric_join_tests;
