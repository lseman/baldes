//! Direct resource operations shared by physical-graph labels and path replay.
use super::*;
mod accounting;
mod activities;
pub(super) mod bounds;
mod bucketing;
mod budgeted;
#[cfg(feature = "clp")]
mod certificate;
#[cfg(feature = "clp")]
pub use certificate::PairedControlCertificate;
mod continuation;
mod dependency_order;
mod domains;
mod dominance;
mod front;
mod frozen;
mod objective;
mod paired;
mod pricing;
pub(super) mod relation;
mod reversal;
mod selection;
mod slope_interval;
pub(super) mod slope_paired;
pub use slope_paired::{PreparedMachineArc, PreparedMachineBackwardInputs};
/// Diagnostic machine history bound to the exact installed sidecar and
/// kernel identity that produced it. This is not a route certificate.
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
mod storage;
pub use activities::{CoupledPathCosts, CoupledPathValues, PairedControl};
use storage::StateValues;
mod transitions;

mod extension;
mod joining;
mod machine_reconstruction;
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
    selection: StateValues<u64, 2>,
    loads: StateValues<[f64; 3], 1>,
    /// Budgeted worst-case vectors, concatenated in declaration order.
    budgets: Vec<f64>,
    reversal_peaks: Option<Box<StateValues<f64, 1>>>,
    /// Created by a consumption-join sweep: this prefix can also be joined as
    /// a reversed suffix, so dominance must preserve that role too.
    reversed_use: bool,
    numeric: StateValues<f64, 4>,
    suffix: Option<Box<Suffix>>,
    fronts: Vec<front::CostFront>,
    slope_fronts: Vec<slope_interval::SlopeLabel>,
    slope_relations: Vec<slope_interval::SlopeLabel>,
    relations: Vec<paired::PairedLabel>,
    plugins: Vec<crate::graph::coupled::PluginState>,
    pricing_activities: Option<Box<[Box<CoupledLabelState>]>>,
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
    objective_coefficients: std::sync::Arc<[f64]>,
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
    relation_indices: Vec<usize>,
    paired: Vec<paired::PreparedPaired>,
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
#[cfg(feature = "clp")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrontProgramCertificate {
    pub feasible: bool,
    pub objective: [f64; 2],
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
        if state
            .pricing_activities
            .as_ref()
            .map_or(0, |states| states.len())
            != self.pricing_activities.len()
            || state.plugins.len() != self.plugins.len()
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
            Some(CoupledResource::SlopePaired { .. })
        ) {
            return Err(invalid("waiting windows require a paired slope resource"));
        }
        let index = kernel.definitions[..resource]
            .iter()
            .filter(|r| matches!(r, CoupledResource::SlopePaired { .. }))
            .count();
        if kernel.slope_paired[index].set_waiting_windows(windows)? {
            kernel.identity = std::sync::Arc::new(());
        }
        Ok(())
    }

    /// Install exact prepared machine operands in physical arc order for a
    /// slope-paired resource. This is an explicit data sidecar; it does not
    /// select a feasibility domain or change pricing by itself. Pass `None`
    /// to remove it. Replacing its contents invalidates prepared graph state.
    pub fn set_paired_machine_inputs(
        &mut self,
        resource: usize,
        inputs: Option<&[PreparedMachineArc]>,
    ) -> Result<()> {
        let topology = inputs.map(|_| slope_paired::MachineTopology {
            source: self.source,
            sink: self.sink,
            vertex_count: self.vertices.len(),
            arcs: self
                .arcs
                .iter()
                .map(|arc| (arc.id, arc.tail, arc.head))
                .collect(),
        });
        let kernel = self
            .coupled
            .as_mut()
            .ok_or_else(|| invalid("no coupled resources"))?;
        if !matches!(
            kernel.definitions.get(resource),
            Some(CoupledResource::SlopePaired { .. })
        ) {
            return Err(invalid("machine inputs require a paired slope resource"));
        }
        let index = kernel.definitions[..resource]
            .iter()
            .filter(|r| matches!(r, CoupledResource::SlopePaired { .. }))
            .count();
        if kernel.slope_paired[index].set_machine_inputs(inputs, topology)? {
            kernel.identity = std::sync::Arc::new(());
        }
        Ok(())
    }

    /// Set the already prepared dependent vector copied into the machine root.
    ///
    /// The caller must prepare this vector using the intended machine input
    /// and bounds. It is separate from the source model's initial frontier.
    /// Replacing or removing it invalidates prepared graph state.
    pub fn set_paired_machine_root(
        &mut self,
        resource: usize,
        root: Option<&PreparedSlopeFunction>,
    ) -> Result<()> {
        let kernel = self
            .coupled
            .as_mut()
            .ok_or_else(|| invalid("no coupled resources"))?;
        if !matches!(
            kernel.definitions.get(resource),
            Some(CoupledResource::SlopePaired { .. })
        ) {
            return Err(invalid("machine root requires a paired slope resource"));
        }
        let index = kernel.definitions[..resource]
            .iter()
            .filter(|r| matches!(r, CoupledResource::SlopePaired { .. }))
            .count();
        if kernel.slope_paired[index].set_machine_root_input(root)? {
            kernel.identity = std::sync::Arc::new(());
        }
        Ok(())
    }

    /// Prepare and install explicit per-arc machine operands atomically.
    ///
    /// `inputs` must be in physical arc-vector order. Its tail windows are
    /// caller-supplied reference inputs, not inferred from source resources;
    /// arcs with the same tail must supply identical window bits. The base
    /// vectors must already be truncated for those windows, and the dependent
    /// vectors and waiting windows must already be prepared. The caller must
    /// reinstall after any graph or model changes affecting those inputs.
    ///
    /// This operation does not select machine pricing or certify a source
    /// model. Unsupported arithmetic, invalid input, and interrupted preparation
    /// return unknown without changing the installed sidecar. Work is divided
    /// conservatively across arcs and all stages share one elapsed deadline.
    pub fn prepare_paired_machine_inputs(
        &mut self,
        resource: usize,
        inputs: &[MachineArcPreparationInput<'_>],
        budget: descriptor::FrontierBudget,
    ) -> descriptor::FrontierOutcome<()> {
        use descriptor::{FrontierOutcome, FrontierUnknown};
        let started = std::time::Instant::now();
        let Some(kernel) = self.coupled.as_ref() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("no coupled resources"));
        };
        if !matches!(
            kernel.definitions.get(resource),
            Some(CoupledResource::SlopePaired {
                policy: FrontOperationPolicy::FrontAlgebra,
                ..
            })
        ) {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine preparation requires a front-algebra paired resource",
            ));
        }
        if inputs.len() != self.arcs.len() || self.source >= self.vertices.len() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine preparation graph source or arc count",
            ));
        }
        let index = kernel.definitions[..resource]
            .iter()
            .filter(|definition| matches!(definition, CoupledResource::SlopePaired { .. }))
            .count();
        let Some(scans) = inputs.len().checked_add(self.vertices.len()) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let Some(remaining_work) = budget.max_operations.checked_sub(scans) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        // Reserve half of the remaining allowance for sidecar validation,
        // comparison and construction before its atomic publication.
        let preparation_work = remaining_work / 2;
        let per_arc = if inputs.is_empty() {
            preparation_work
        } else {
            preparation_work / inputs.len()
        };
        let installation_budget = remaining_work - per_arc * inputs.len();
        let mut installation_work = 0usize;
        let mut tails: Vec<Option<Window>> = vec![None; self.vertices.len()];
        let mut prepared = Vec::with_capacity(inputs.len());
        for (position, input) in inputs.iter().enumerate() {
            let remaining_time = budget
                .max_elapsed
                .map(|limit| limit.saturating_sub(started.elapsed()));
            if remaining_time.is_some_and(|remaining| remaining.is_zero()) {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            let arc = &self.arcs[position];
            if input.dependent_lower.pairs().len() > budget.max_pairs
                || input.dependent_upper.pairs().len() > budget.max_pairs
            {
                return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
            }
            if input.physical_arc != position
                || arc.tail >= self.vertices.len()
                || arc.head >= self.vertices.len()
                || Window::new(input.tail_bounds.lower, input.tail_bounds.upper).is_err()
                || Window::new(input.waiting_bounds.lower, input.waiting_bounds.upper).is_err()
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine preparation arc order or bounds",
                ));
            }
            if let Some(previous) = tails[arc.tail] {
                if previous.lower.to_bits() != input.tail_bounds.lower.to_bits()
                    || previous.upper.to_bits() != input.tail_bounds.upper.to_bits()
                {
                    return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                        "machine preparation inconsistent tail bounds",
                    ));
                }
            } else {
                tails[arc.tail] = Some(input.tail_bounds);
            }
            let arc_budget = descriptor::FrontierBudget {
                max_operations: per_arc,
                max_elapsed: remaining_time,
                ..budget
            };
            let result = descriptor::prepare_machine_arc(
                input.truncated_base,
                input.dependent,
                input.tail_bounds.lower,
                input.tail_bounds.upper,
                input.waiting_bounds.upper,
                arc_budget,
            );
            let value = match result {
                FrontierOutcome::Complete(value) => value,
                FrontierOutcome::Empty => {
                    return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                        "machine preparation unavailable",
                    ));
                }
                FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
            };
            // Charge cloning, validation and potential bitwise comparison
            // before copying the caller's dependent bound functions.
            let arc_work = [
                &value.inputs.forward,
                &value.inputs.inverse,
                &value.inputs.sum,
                input.dependent_lower,
                input.dependent_upper,
                input.dependent,
            ]
            .iter()
            .try_fold(6usize, |total, function| {
                function
                    .pairs()
                    .len()
                    .checked_mul(4)
                    .and_then(|pairs| total.checked_add(pairs))
            });
            let Some(next_work) = arc_work.and_then(|work| installation_work.checked_add(work))
            else {
                return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
            };
            if next_work > installation_budget {
                return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
            }
            installation_work = next_work;
            prepared.push(PreparedMachineArc {
                tail_limit: value.limits.tail_limit,
                head_limit: input.waiting_bounds.upper,
                waiting_lower: input.waiting_bounds.lower,
                waiting_upper: input.waiting_bounds.upper,
                waiting_rate: input.waiting_rate,
                forward_mapping: value.inputs.forward,
                inverse: value.inputs.inverse,
                sum_vector: value.inputs.sum,
                dependent_lower: input.dependent_lower.clone(),
                dependent_upper: input.dependent_upper.clone(),
                backward: Some(PreparedMachineBackwardInputs {
                    tail_lower: input.tail_bounds.lower,
                    tail_upper: input.tail_bounds.upper,
                    forward_at_tail_lower: value.limits.forward_at_tail_lower,
                    mapped_upper: value.limits.mapped_upper,
                    dependent: input.dependent.clone(),
                }),
            });
        }
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        // Stage all allocations and validation before publication so a
        // deadline or failure cannot leave a partially installed sidecar.
        let topology = slope_paired::MachineTopology {
            source: self.source,
            sink: self.sink,
            vertex_count: self.vertices.len(),
            arcs: self
                .arcs
                .iter()
                .map(|arc| (arc.id, arc.tail, arc.head))
                .collect(),
        };
        let mut staged = kernel.slope_paired[index].clone();
        let changed = match staged.set_machine_inputs(Some(&prepared), Some(topology)) {
            Ok(changed) => changed,
            Err(_) => {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine preparation could not install inputs",
                ))
            }
        };
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        if changed {
            let kernel = self.coupled.as_mut().expect("kernel was present");
            kernel.slope_paired[index] = staged;
            kernel.identity = std::sync::Arc::new(());
        }
        FrontierOutcome::Complete(())
    }

    /// Prepare a backward machine root from installed sink-arc upper vectors.
    ///
    /// The supplied terminal bounds must be the intended machine bounds. The
    /// installed operands and graph are explicit inputs, not a source-model
    /// certificate. Unsupported or stale data returns unknown.
    pub fn prepare_paired_machine_backward_root(
        &self,
        resource: usize,
        terminal_bounds: Window,
        budget: descriptor::FrontierBudget,
    ) -> descriptor::FrontierOutcome<descriptor::MachineRootState> {
        use descriptor::{FrontierBudget, FrontierOutcome, FrontierUnknown, MachineRootState};
        let started = std::time::Instant::now();
        let Some(kernel) = self.coupled.as_ref() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("no coupled resources"));
        };
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let Some(validation_work) = self
            .arcs
            .len()
            .checked_mul(2)
            .and_then(|work| work.checked_add(resource))
        else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let Some(remaining_work) = budget.max_operations.checked_sub(validation_work) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        if !matches!(
            kernel.definitions.get(resource),
            Some(CoupledResource::SlopePaired { .. })
        ) {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine backward root requires a paired slope resource",
            ));
        }
        let index = kernel.definitions[..resource]
            .iter()
            .filter(|item| matches!(item, CoupledResource::SlopePaired { .. }))
            .count();
        let paired = &kernel.slope_paired[index];
        let Some(inputs) = paired.machine_input_handle() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired machine inputs unavailable",
            ));
        };
        if self.sink >= self.vertices.len() || !paired.machine_topology_matches(self) {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired machine graph topology changed",
            ));
        }
        let mut upper_vectors = Vec::new();
        for (arc, input) in self.arcs.iter().zip(inputs.iter()) {
            if budget
                .max_elapsed
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            if arc.head == self.sink {
                upper_vectors.push(&input.dependent_upper);
            }
        }
        let remaining_time = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        MachineRootState::from_backward_inputs(
            &upper_vectors,
            terminal_bounds.lower,
            terminal_bounds.upper,
            FrontierBudget {
                max_operations: remaining_work,
                max_elapsed: remaining_time,
                ..budget
            },
        )
    }

    /// Install the prepared backward root for machine label creation.
    /// A later arc-input replacement clears the root and requires reinstalling.
    pub fn install_paired_machine_backward_root(
        &mut self,
        resource: usize,
        terminal_bounds: Window,
        budget: descriptor::FrontierBudget,
    ) -> descriptor::FrontierOutcome<()> {
        use descriptor::{FrontierOutcome, FrontierUnknown};
        let started = std::time::Instant::now();
        let root_work = budget.max_operations / 2;
        let root = match self.prepare_paired_machine_backward_root(
            resource,
            terminal_bounds,
            descriptor::FrontierBudget {
                max_operations: root_work,
                ..budget
            },
        ) {
            FrontierOutcome::Complete(root) => root,
            FrontierOutcome::Empty => return FrontierOutcome::Empty,
            FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
        };
        let Some(kernel) = self.coupled.as_mut() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("no coupled resources"));
        };
        let index = kernel.definitions[..resource]
            .iter()
            .filter(|item| matches!(item, CoupledResource::SlopePaired { .. }))
            .count();
        let remaining_time = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        match kernel.slope_paired[index].set_machine_backward_root(
            root,
            descriptor::FrontierBudget {
                max_operations: budget.max_operations - root_work,
                max_elapsed: remaining_time,
                ..budget
            },
        ) {
            FrontierOutcome::Complete(()) => {}
            FrontierOutcome::Empty => return FrontierOutcome::Empty,
            FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
        }
        kernel.identity = std::sync::Arc::new(());
        FrontierOutcome::Complete(())
    }

    /// Replay an explicit prepared-machine paired resource from a caller-
    /// supplied machine root. `path` contains physical arc-vector indices,
    /// starts at the graph source and may stop at any vertex. The `mode` is
    /// the reference resource mode integer, supplied explicitly by the caller.
    ///
    /// This bounded diagnostic does not prove source feasibility, validate a
    /// route witness, or affect pricing. Missing sidecar data, stale physical
    /// topology, noncontiguous paths and unsupported machine stages return
    /// unknown. Changing the graph requires reinstalling the sidecar.
    pub fn replay_paired_machine_path(
        &self,
        resource: usize,
        root: &descriptor::MachineRootState,
        path: &[usize],
        mode: i32,
        max_states: usize,
        budget: descriptor::FrontierBudget,
    ) -> descriptor::FrontierOutcome<descriptor::MachinePathHistory> {
        let Some(kernel) = self.coupled.as_ref() else {
            return descriptor::FrontierOutcome::Unknown(descriptor::FrontierUnknown::Unsupported(
                "no coupled resources",
            ));
        };
        if !matches!(
            kernel.definitions.get(resource),
            Some(CoupledResource::SlopePaired { .. })
        ) {
            return descriptor::FrontierOutcome::Unknown(descriptor::FrontierUnknown::Unsupported(
                "machine replay requires a paired slope resource",
            ));
        }
        let index = kernel.definitions[..resource]
            .iter()
            .filter(|r| matches!(r, CoupledResource::SlopePaired { .. }))
            .count();
        kernel.slope_paired[index].machine_replay(self, root, path, mode, max_states, budget)
    }

    /// Replay and bind a diagnostic history to the exact installed machine
    /// inputs. This does not validate the supplied root or certify a route.
    pub fn replay_bound_paired_machine_path(
        &self,
        resource: usize,
        root: &descriptor::MachineRootState,
        path: &[usize],
        mode: i32,
        max_states: usize,
        budget: descriptor::FrontierBudget,
    ) -> descriptor::FrontierOutcome<BoundMachinePathHistory> {
        use descriptor::{FrontierBudget, FrontierOutcome, FrontierUnknown};
        let started = std::time::Instant::now();
        if budget.max_elapsed.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let Some(kernel) = self.coupled.as_ref() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("no coupled resources"));
        };
        if !matches!(
            kernel.definitions.get(resource),
            Some(CoupledResource::SlopePaired { .. })
        ) {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine replay requires a paired slope resource",
            ));
        }
        let Some(wrapper_work) = resource.checked_add(path.len()) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let Some(remaining_work) = budget.max_operations.checked_sub(wrapper_work) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let mut index = 0;
        for definition in &kernel.definitions[..resource] {
            if budget
                .max_elapsed
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            if matches!(definition, CoupledResource::SlopePaired { .. }) {
                index += 1;
            }
        }
        let Some(inputs) = kernel.slope_paired[index].machine_input_handle() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired machine inputs unavailable",
            ));
        };
        let remaining_time = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|remaining| remaining.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let history = match kernel.slope_paired[index].machine_replay(
            self,
            root,
            path,
            mode,
            max_states,
            FrontierBudget {
                max_operations: remaining_work,
                max_elapsed: remaining_time,
                ..budget
            },
        ) {
            FrontierOutcome::Complete(value) => value,
            FrontierOutcome::Empty => {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine replay rejected",
                ))
            }
            FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
        };
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let path = path.to_vec();
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        FrontierOutcome::Complete(BoundMachinePathHistory {
            history,
            resource,
            path,
            inputs,
            identity: kernel.identity.clone(),
        })
    }

    /// Backtrack a sidecar-bound history using its installed inverse vectors.
    /// `root_is_entry_zero` explicitly selects the unproved correspondence
    /// between this history's root and reference path entry zero. This is a
    /// diagnostic, not a route, objective, or source-feasibility certificate.
    pub fn backtrack_bound_paired_machine_path(
        &self,
        history: &BoundMachinePathHistory,
        terminal: f64,
        root_is_entry_zero: bool,
        budget: descriptor::FrontierBudget,
    ) -> descriptor::FrontierOutcome<Vec<f64>> {
        use descriptor::{FrontierBudget, FrontierOutcome, FrontierUnknown};
        let started = std::time::Instant::now();
        if budget.max_elapsed.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let Some(kernel) = self.coupled.as_ref() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("no coupled resources"));
        };
        if !std::sync::Arc::ptr_eq(&kernel.identity, &history.identity)
            || !matches!(
                kernel.definitions.get(history.resource),
                Some(CoupledResource::SlopePaired { .. })
            )
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired machine kernel identity changed",
            ));
        }
        let Some(remaining_work) = budget.max_operations.checked_sub(history.resource) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let mut index = 0;
        for definition in &kernel.definitions[..history.resource] {
            if budget
                .max_elapsed
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            if matches!(definition, CoupledResource::SlopePaired { .. }) {
                index += 1;
            }
        }
        let remaining_time = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|remaining| remaining.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let result = kernel.slope_paired[index].machine_backtrack_bound(
            self,
            history,
            terminal,
            root_is_entry_zero,
            FrontierBudget {
                max_operations: remaining_work,
                max_elapsed: remaining_time,
                ..budget
            },
        );
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        result
    }
    /// Reconstruct a retained backward label's machine suffix from explicitly
    /// supplied joined values. The installed terminal root and arc operands
    /// must still match the history. This returns machine values, not a source
    /// feasibility certificate or a normalized route objective.
    pub fn reconstruct_paired_machine_suffix(
        &self,
        resource: usize,
        backward: &CoupledLabelState,
        initial_base: f64,
        initial_dependent: f64,
        budget: descriptor::FrontierBudget,
    ) -> descriptor::FrontierOutcome<descriptor::MachinePathReconstruction> {
        use descriptor::{
            FrontierBudget, FrontierOutcome, FrontierUnknown, PreparedSlopeRelationArc,
        };
        let started = std::time::Instant::now();
        if budget.max_elapsed.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let Some(kernel) = self.coupled.as_ref() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("no coupled resources"));
        };
        if kernel.check_state(backward).is_err()
            || !matches!(
                kernel.definitions.get(resource),
                Some(CoupledResource::SlopePaired { .. })
            )
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine suffix resource or state",
            ));
        }
        let Some(suffix) = backward.suffix.as_ref() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine suffix requires a backward label",
            ));
        };
        let index = kernel.definitions[..resource]
            .iter()
            .filter(|definition| matches!(definition, CoupledResource::SlopePaired { .. }))
            .count();
        let prepared = &kernel.slope_paired[index];
        let Some(history) = backward.slope_relations[index].machine_history.as_ref() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine suffix history unavailable",
            ));
        };
        let validation_work = resource
            .saturating_add(self.arcs.len())
            .saturating_add(suffix.program.len().saturating_mul(4))
            .saturating_add(history.len())
            .saturating_add(1);
        let Some(remaining_work) = budget.max_operations.checked_sub(validation_work) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let path: Vec<_> = suffix.program.iter().collect();
        if !prepared.matches_backward_root(history) {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine suffix terminal root changed",
            ));
        }
        let (Some(inputs), Some(topology)) =
            (prepared.machine_input_handle(), prepared.machine_topology())
        else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine suffix operands unavailable",
            ));
        };
        if self.source != topology.source
            || self.sink != topology.sink
            || self.vertices.len() != topology.vertex_count
            || self.arcs.len() != topology.arcs.len()
            || self.arcs.len() != inputs.len()
            || history.len().checked_sub(1) != Some(path.len())
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine suffix topology changed",
            ));
        }
        for (arc, &(id, tail, head)) in self.arcs.iter().zip(topology.arcs.iter()) {
            if budget
                .max_elapsed
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            if (arc.id, arc.tail, arc.head) != (id, tail, head)
                || tail >= self.vertices.len()
                || head >= self.vertices.len()
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine suffix topology changed",
                ));
            }
        }
        let mut vertex = path
            .first()
            .and_then(|&index| self.arcs.get(index))
            .map_or(self.sink, |arc| arc.tail);
        let mut steps = Vec::with_capacity(path.len());
        for &arc in &path {
            if budget
                .max_elapsed
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            let Some(physical) = self.arcs.get(arc).filter(|arc| arc.tail == vertex) else {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine suffix path continuity",
                ));
            };
            let Some(input) = inputs[arc].backward_inputs() else {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine suffix backward operands unavailable",
                ));
            };
            vertex = physical.head;
            steps.push((arc, input));
        }
        if vertex != self.sink {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine suffix does not reach sink",
            ));
        }
        let remaining_time = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        PreparedSlopeRelationArc::machine_reconstruct_backward_path(
            history,
            &steps,
            initial_base,
            initial_dependent,
            prepared.machine_mode(),
            FrontierBudget {
                max_operations: remaining_work,
                max_elapsed: remaining_time,
                ..budget
            },
        )
    }

    /// Install paired machine waiting windows from a physical scalar resource.
    /// The caller identifies the scalar resource representing the paired base
    /// coordinate. Each window is the intersection of its destination vertex
    /// bounds and physical arc bounds, in physical arc order. Absent arc bounds
    /// use the reference's finite -1e12/+1e12 sentinels; ties retain the arc
    /// bound's binary64 bits. Disconnected multi-time windows and empty
    /// intersections are not representable here.
    /// Reinstall after changing the graph or paired resource definitions.
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
        self.set_paired_waiting_windows(resource, Some(&windows))
    }
    /// Select a different original resource domain for native raw fronts.
    ///
    /// With `true`, native raw interval and paired feasibility is defined by
    /// their checked ordered IEEE front program. This is a MODEL choice, not
    /// a proof-strength hint: it need not equal source-control or exact-real
    /// feasibility. The default is `false`; strict and other families do not
    /// change. Clones, branches and cost suppression preserve the choice.
    /// Replacing resource descriptors resets it to the default.
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
        if enabled && kernel.has_negative_paired_objectives() {
            return Err(invalid("negative paired objectives require a source model"));
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

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/kernel/custom_resource_tests.rs"]
mod custom_resource_tests;
