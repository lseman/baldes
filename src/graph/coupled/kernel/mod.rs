//! Direct resource operations shared by physical-graph labels and path replay.
use super::*;
mod accounting;
mod activities;
pub(super) mod bounds;
mod bucketing;
mod budgeted;
mod certificate;
mod continuation;
mod dependency_order;
mod domains;
mod dominance;
mod front;
mod frozen;
mod objective;
mod pricing;
pub(super) mod reversal;
mod selection;
mod slope_interval;
pub(super) mod slope_paired;
pub use slope_paired::{PreparedMachineArc, PreparedMachineBackwardInputs};
/// Diagnostic machine history bound to the installed machine program that
/// produced it. This is not a route certificate.
#[derive(Clone, Debug)]
pub struct BoundMachinePathHistory {
    history: descriptor::MachinePathHistory,
    resource: usize,
    path: Vec<usize>,
    inputs: std::sync::Arc<[PreparedMachineArc]>,
    identity: std::sync::Arc<()>,
}
impl BoundMachinePathHistory {
    /// The unreduced root and forward snapshots retained by replay.
    pub fn history(&self) -> &descriptor::MachinePathHistory {
        &self.history
    }
}
/// Explicit, already prepared inputs for one physical arc. The caller is
/// responsible for preparing `truncated_base` and `dependent` according to
/// the intended machine program; neither is inferred from the source model.
#[derive(Clone, Debug)]
pub struct MachineArcPreparationInput<'a> {
    /// Index in the graph's physical arc vector, not the arc ID.
    pub physical_arc: usize,
    /// Tail bounds used to truncate the supplied base vector.
    pub tail_bounds: Window,
    /// The exact per-arc waiting bounds, including any vertex intersection.
    pub waiting_bounds: Window,
    pub waiting_rate: i32,
    pub truncated_base: &'a PreparedSlopeFunction,
    pub dependent: &'a PreparedSlopeFunction,
    pub dependent_lower: &'a PreparedSlopeFunction,
    pub dependent_upper: &'a PreparedSlopeFunction,
}

/// Per-arc function inputs before the front program's window preparation.
#[derive(Clone, Copy, Debug)]
pub struct MachineArcFunctionInput<'a> {
    pub physical_arc: usize,
    pub tail_bounds: Window,
    pub waiting_bounds: Window,
    pub waiting_rate: i32,
    /// Base consumption vector, before truncation to the tail window and
    /// conversion to the arrival mapping.
    pub base: &'a PreparedSlopeFunction,
    pub dependent: descriptor::MachineRootInput<'a>,
    pub dependent_lower: descriptor::MachineRootInput<'a>,
    pub dependent_upper: descriptor::MachineRootInput<'a>,
}
mod storage;
pub use activities::CoupledPathCosts;
use storage::StateValues;
mod transitions;

mod extension;
mod joining;
mod machine;
#[cfg(feature = "clp")]
mod objective_lower;
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
    completion_identity: std::sync::Arc<()>,
    selection: StateValues<u64, 2>,
    loads: StateValues<[f64; 3], 1>,
    /// Budgeted worst-case vectors, concatenated in declaration order.
    budgets: Vec<f64>,
    reversal_peaks: Option<Box<StateValues<f64, 1>>>,
    /// Created by a consumption-join sweep: this prefix can also be joined as
    /// a reversed suffix, so dominance must preserve that role too.
    reversed_use: bool,
    numeric: StateValues<f64, 4>,
    custom_machine: Vec<crate::graph::coupled::CustomMachineTrace>,
    custom_machine_identity: Option<std::sync::Arc<()>>,
    suffix: Option<Box<Suffix>>,
    fronts: Vec<front::CostFront>,
    slope_fronts: Vec<slope_interval::SlopeLabel>,
    slope_relations: Vec<slope_interval::SlopeLabel>,
    plugins: Vec<crate::graph::coupled::PluginState>,
    pricing_activities: Option<Box<[Box<CoupledLabelState>]>>,
    numeric_cost: f64,
    resource_objective: f64,
    paired_ordered_key: bool,
    accounting: Option<accounting::ScalarCosts>,
}
/// Prepared resource definitions over the unchanged physical arc vector.
#[derive(Clone, Debug, PartialEq)]
pub struct CoupledKernel {
    // Shared by semantic clones; cost suppression receives a fresh token.
    identity: std::sync::Arc<()>,
    // Machine operands and roots outlive changes to pricing-only state.
    machine_identity: std::sync::Arc<()>,
    definitions: std::sync::Arc<[CoupledResource]>,
    custom_indices: Vec<usize>,
    slots: Vec<Option<usize>>,
    initial: Vec<f64>,
    cost_scale: f64,
    objective_coefficients: std::sync::Arc<[f64]>,
    // Original associated-variable charges do not scale raw paired pricing.
    paired_associated_costs: std::sync::Arc<[(usize, f64)]>,
    paired_owner_registration_bounds: std::sync::Arc<[(usize, [f64; 2])]>,
    weighted_objective: bool,
    nonnegative_paired_weights_only: bool,
    negative_paired_objective: bool,
    pricing_adjustments: std::sync::Arc<[(usize, f64)]>,
    pricing_activities: std::sync::Arc<[CoupledKernel]>,
    /// Explicit MODEL domain choice, never inferred from a numerical policy.
    front_program_domain: bool,
    selection_initial: Vec<u64>,
    /// Directional vertex masks and forgetting memory; None for exact arc-only
    /// resources, which need only the transition's arc intersection.
    selection_memory: Vec<Option<selection::PreparedSelection>>,
    /// Physical arc heads and the source, for exact vertex masks.
    selection_heads: std::sync::Arc<[usize]>,
    selection_source: usize,
    /// Pickup/delivery capacity plus its tolerance: the largest admitted load.
    load_capacities: Vec<f64>,
    /// Pickup/delivery dominance tolerance, as in the capacity checks.
    load_tolerances: Vec<f64>,
    budgeted: Vec<budgeted::PreparedBudgeted>,
    reversal_union: bool,
    cost_resource: bool,
    front_indices: Vec<usize>,
    slope_indices: Vec<usize>,
    slope_roots: Vec<slope_interval::SlopeLabel>,
    slope_charges: Vec<std::sync::Arc<[slope_interval::PreparedSlopeCharge]>>,
    slope_dominance: Vec<slope_interval::PreparedSlopeDominance>,
    slope_paired: Vec<slope_paired::PreparedSlopePaired>,
    plugins: Vec<crate::graph::coupled::PluginResource>,
    front_bucket_policy: bucketing::FrontBucketPolicy,
    frozen: frozen::PreparedFrozen,
    arc_count: usize,
    optimization: PreparedBounds,
    numeric_cost_order: dominance::NumericCostOrder,
    transitions: PreparedTransitions,
    route_domains: domains::RouteDomains,
}
/// Certificate for the explicitly named alternate native machine-front model.
/// It is not a source-control feasibility witness for a default graph. The
/// objective interval also includes outward aggregation with other resources.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrontProgramCertificate {
    pub feasible: bool,
    pub objective: [f64; 2],
}
/// Reusable per-thread resource-extension scratch. Retained labels own all
/// their state and never borrow temporary workspace storage.
#[derive(Default)]
pub struct CoupledWorkspace {}
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
        if state
            .pricing_activities
            .as_ref()
            .map_or(0, |states| states.len())
            != self.pricing_activities.len()
            || state.plugins.len() != self.plugins.len()
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
            || state.custom_machine.len() != self.custom_indices.len()
            || (!state.custom_machine.is_empty()
                && !state
                    .custom_machine_identity
                    .as_ref()
                    .is_some_and(|identity| {
                        std::sync::Arc::ptr_eq(identity, &self.machine_identity)
                    }))
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
    /// Apply the recovered custom vertex callback to this label's raw machine
    /// traces. Labeling and enumeration call this at their vertex step;
    /// callers extending standalone labels may call it at the same point.
    pub fn custom_machine_enter_vertex(&mut self) {
        for trace in &mut self.custom_machine {
            if trace.accepted {
                trace.wrapper_cost = 0. + trace.wrapper_cost;
            }
        }
    }

    /// Raw custom callback traces in descriptor order. A rejected machine
    /// trace does not reject the independently evaluated source label.
    pub fn custom_machine_traces(&self) -> &[crate::graph::coupled::CustomMachineTrace] {
        &self.custom_machine
    }

    /// Cost knots for each interval resource in declaration order. Forward
    /// functions price prefixes at the current arrival value; backward
    /// functions price suffixes at their entry value. Singleton domains are
    /// represented by one knot. Values include the current objective scale.
    pub fn interval_cost_fronts(&self) -> impl Iterator<Item = &[[f64; 2]]> {
        self.fronts.iter().map(front::CostFront::points)
    }

    /// Piecewise affine paired fronts at the current forward label.
    pub fn paired_resource_fronts(&self) -> Option<Vec<Vec<[[f64; 2]; 2]>>> {
        if self.suffix.is_some() {
            return None;
        }
        self.slope_relations
            .iter()
            .map(|label| {
                let front = label.front.as_ref()?;
                let mut pieces = Vec::new();
                for function in front.components() {
                    let pairs = function.pairs();
                    let (mut value, mut coordinate) = *pairs.first()?;
                    if pairs.len() == 1 {
                        pieces.push([[coordinate, value], [coordinate, value]]);
                    }
                    for &(slope, end) in &pairs[1..] {
                        let next = value + (end - coordinate) * slope;
                        if !next.is_finite() {
                            return None;
                        }
                        pieces.push([[coordinate, value], [end, next]]);
                        coordinate = end;
                        value = next;
                    }
                }
                Some(pieces)
            })
            .collect()
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
    /// Set explicit per-arc accumulated-base waiting windows for one paired
    /// resource, indexed by its position in the resource definitions. The
    /// windows must already include the destination vertex intersection and
    /// follow the physical arc order. Passing `None` removes the override.
    /// Replacing resources or changing graph topology requires reinstalling it.
    pub fn set_paired_waiting_windows(
        &mut self,
        resource: usize,
        windows: Option<&[Window]>,
    ) -> Result<()> {
        let kernel = self
            .coupled
            .as_mut()
            .ok_or_else(|| invalid("no coupled resources"))?;
        if !matches!(
            kernel.definitions.get(resource),
            Some(CoupledResource::PairedFront { .. })
        ) {
            return Err(invalid("waiting windows require a paired slope resource"));
        }
        let index = kernel.definitions[..resource]
            .iter()
            .filter(|r| matches!(r, CoupledResource::PairedFront { .. }))
            .count();
        let changed = kernel.slope_paired[index].set_waiting_windows(windows)?;
        let changed = kernel.slope_paired[index].clear_base_windows() || changed;
        if changed {
            kernel.identity = std::sync::Arc::new(());
        }
        Ok(())
    }

    /// Set waiting windows from a scalar resource without replacing an explicitly supplied front program.
    pub fn set_paired_waiting_windows_from_resource(
        &mut self,
        resource: usize,
        base_resource: usize,
    ) -> Result<()> {
        if base_resource >= self.resources.len() {
            return Err(invalid("paired waiting base resource is out of range"));
        }
        #[cfg(feature = "multi-time-windows")]
        if self
            .time_windows
            .keys()
            .any(|&(_, index)| index == base_resource)
        {
            return Err(invalid(
                "paired waiting windows cannot represent multiple time windows",
            ));
        }
        let mut windows = Vec::with_capacity(self.arcs.len());
        for arc in &self.arcs {
            let vertex = self
                .vertices
                .get(arc.head)
                .ok_or_else(|| invalid("paired waiting arc head is out of range"))?;
            let vertex_window = *vertex
                .bounds
                .get(base_resource)
                .ok_or_else(|| invalid("paired waiting vertex resource is out of range"))?;
            let arc_window = arc
                .bounds
                .get(base_resource)
                .ok_or_else(|| invalid("paired waiting arc resource is out of range"))?
                .unwrap_or(Window {
                    lower: -1e12,
                    upper: 1e12,
                });
            Window::new(vertex_window.lower, vertex_window.upper)?;
            Window::new(arc_window.lower, arc_window.upper)?;
            // MAXSD/MINSD select the arc operand on equal values, including
            // signed zero. Both inputs have already been validated as finite.
            let lower = if vertex_window.lower > arc_window.lower {
                vertex_window.lower
            } else {
                arc_window.lower
            };
            let upper = if vertex_window.upper < arc_window.upper {
                vertex_window.upper
            } else {
                arc_window.upper
            };
            windows.push(Window::new(lower, upper)?);
        }
        let snapshot = slope_paired::BaseWindows {
            resource: base_resource,
            vertices: self
                .vertices
                .iter()
                .map(|vertex| {
                    let window = vertex
                        .bounds
                        .get(base_resource)
                        .ok_or_else(|| invalid("paired waiting vertex resource is out of range"))?;
                    Ok((window.lower.to_bits(), window.upper.to_bits()))
                })
                .collect::<Result<Vec<_>>>()?,
            arcs: self
                .arcs
                .iter()
                .map(|arc| {
                    Ok(arc
                        .bounds
                        .get(base_resource)
                        .ok_or_else(|| invalid("paired waiting arc resource is out of range"))?
                        .map(|window| (window.lower.to_bits(), window.upper.to_bits())))
                })
                .collect::<Result<Vec<_>>>()?,
        };
        self.set_paired_waiting_windows(resource, Some(&windows))?;
        let kernel = self
            .coupled
            .as_mut()
            .ok_or_else(|| invalid("no coupled resources"))?;
        let index = kernel.definitions[..resource]
            .iter()
            .filter(|definition| matches!(definition, CoupledResource::PairedFront { .. }))
            .count();
        if kernel.slope_paired[index].set_base_windows(snapshot) {
            kernel.identity = std::sync::Arc::new(());
        }
        Ok(())
    }
    /// Select the ordered front domain for raw resources. Paired resources
    /// always use this domain; it cannot be disabled while one is installed.
    pub fn set_front_program_domain(&mut self, enabled: bool) -> Result<()> {
        let Some(kernel) = self.coupled.as_mut() else {
            return if enabled {
                Err(invalid(
                    "native front program domain requires native raw resources",
                ))
            } else {
                Ok(())
            };
        };
        if !enabled
            && kernel
                .definitions
                .iter()
                .any(|definition| matches!(definition, CoupledResource::PairedFront { .. }))
        {
            return Err(invalid("paired resources require the native front domain"));
        }
        if enabled && kernel.has_negative_paired_objectives() {
            return Err(invalid(
                "negative paired objectives are not supported by paired front programs",
            ));
        }
        if enabled && !kernel.has_machine_front_contracts() {
            return Err(invalid(
                "native front program domain requires native raw resources",
            ));
        }
        if kernel.front_program_domain != enabled {
            kernel.front_program_domain = enabled;
            kernel.identity = std::sync::Arc::new(());
        }
        Ok(())
    }
    /// Whether native raw feasibility explicitly uses the machine-front model.
    pub fn front_program_domain(&self) -> bool {
        self.coupled
            .as_ref()
            .is_some_and(|kernel| kernel.front_program_domain)
    }
    /// Retain selection eliminations at every vertex of violating windows.
    pub(crate) fn strengthen_selection(&mut self, arcs: &[usize]) -> bool {
        self.coupled
            .as_mut()
            .is_some_and(|kernel| kernel.strengthen_selection(arcs))
    }
    /// The reference schedule: one first window per resource, length-capped.
    pub(crate) fn strengthen_selection_first_windows(
        &mut self,
        arcs: &[usize],
        max_length: i64,
    ) -> bool {
        self.coupled
            .as_mut()
            .is_some_and(|kernel| kernel.strengthen_selection_first_windows(arcs, max_length))
    }
    pub(crate) fn selection_relaxed(&self) -> bool {
        self.coupled_kernel()
            .is_some_and(CoupledKernel::selection_relaxed)
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
            Some(k)
                if k.machine_paired_join_complete()
                    && (arcs.is_empty() && self.source == self.sink
                        || arcs
                            .first()
                            .and_then(|&index| self.arcs.get(index))
                            .is_some_and(|arc| arc.tail == self.source)
                            && arcs
                                .last()
                                .and_then(|&index| self.arcs.get(index))
                                .is_some_and(|arc| arc.head == self.sink)) =>
            {
                self.coupled_route_cost(arcs)
            }
            Some(k) => k.replay(arcs),
            None => Ok(Some(0.)),
        }
    }
    /// Complete-route check for the explicit machine-feasibility domain.
    /// Partial-path callers retain the forward replay contract above.
    pub(crate) fn coupled_route_cost(&self, arcs: &[usize]) -> Result<Option<f64>> {
        match self.coupled_kernel() {
            Some(k) if k.machine_paired_join_complete() => {
                k.validate_for(self)?;
                let mut vertex = self.source;
                for &index in arcs {
                    let arc = self
                        .arcs
                        .get(index)
                        .ok_or_else(|| invalid("unknown coupled route arc"))?;
                    if arc.tail != vertex {
                        return Ok(None);
                    }
                    vertex = arc.head;
                }
                if vertex != self.sink {
                    return Ok(None);
                }
                Ok(k.joined_associated_machine_path_matching(arcs, None)?
                    .map(|(cost, _)| cost))
            }
            _ => self.coupled_path_cost(arcs),
        }
    }
}

#[cfg(all(test, feature = "clp"))]
#[path = "../../../../tests/unit/graph/coupled/kernel/nonlinear_phase_tests.rs"]
mod nonlinear_phase_tests;

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/kernel/canonical_numeric_join_tests.rs"]
mod canonical_numeric_join_tests;

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/kernel/custom_resource_tests.rs"]
mod custom_resource_tests;

pub use machine::PairedBaseDomains;
