//! Direct coupled label resources over the physical graph.
//!
//! Install with `Graph::set_coupled_resources` or the corresponding Solver
//! method. Forward/backward extension, joins and replay share a prepared kernel.
use super::{Graph, Window};
use crate::{invalid, Result};

pub mod descriptor;
mod encoded;
mod custom_machine;
mod kernel;
pub mod plugin;
mod pwl;
pub use descriptor::{
    EvaluationPolicy, FrontOperationPolicy, PairedCostNormalization, PreparedSlopeFunction,
    SlopeRelationArc,
};
pub use encoded::EncodedFunction;
pub use custom_machine::{CustomMachineState, CustomMachineTrace};
pub use kernel::{
    BoundMachinePathHistory, CoupledKernel, CoupledLabelState, CoupledPathCosts, CoupledPathValues,
    CoupledWorkspace, MachineArcFunctionInput, MachineArcPreparationInput, PairedControl,
    PreparedMachineArc, PreparedMachineBackwardInputs,
};
pub(crate) use kernel::{CoupledPairSummary, CoupledStateBounds};
pub use kernel::{FrontProgramCertificate, PairedControlCertificate};
pub use plugin::{CoupledPlugin, PluginResource, PluginState};
pub use pwl::PiecewiseLinearFunction;

/// Diagnostics for bounded graph-wide resource-domain propagation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoupledPreprocessing {
    pub passes: usize,
    pub converged: bool,
    pub arcs: usize,
    pub usable_arcs: usize,
}

/// Meaning of the dependent coordinate in a paired function relation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "clp", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "clp", serde(rename_all = "snake_case"))]
pub enum PairedResourceMode {
    /// Dependent coordinate constrains feasibility and adds no objective.
    Feasibility,
    /// Signed cost increments; dependent bounds are constant across arcs.
    CostUnbounded,
    /// Constant bounds, positive increments and nonnegative waiting slopes.
    CostBounded,
}

/// One physical arc's function-valued base/dependent relation.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "clp", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "clp", serde(deny_unknown_fields))]
pub struct PairedResourceArc {
    /// Total base arrival q(t), strictly increasing, with q(t)>=t.
    pub arrival: PiecewiseLinearFunction,
    /// Dependent increment phi(t), evaluated at the departure base value.
    pub increment: PiecewiseLinearFunction,
    /// Lower dependent bound as a function of the head base value.
    pub lower: PiecewiseLinearFunction,
    /// Upper dependent bound as a function of the head base value.
    pub upper: PiecewiseLinearFunction,
    /// Dependent increment per unit of waiting after base arrival.
    pub waiting: i32,
}

/// Default pickup/delivery capacity tolerance.
pub const DEFAULT_LOAD_TOLERANCE: f64 = 1e-6;
#[cfg(feature = "clp")]
fn default_load_tolerance() -> f64 {
    DEFAULT_LOAD_TOLERANCE
}

/// Resource inputs use the original graph's arc vector order, not arc IDs.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "clp", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(
    feature = "clp",
    serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)
)]
pub enum CoupledResource {
    /// A resource supplied by an extension (for example EVRP charging).
    /// Objective costs require explicit opt-in; it cannot be a numeric
    /// dependency parent. Feasibility-only plugins can supply an independent
    /// proof callback; cost-bearing plugins currently have no directed cost
    /// certificate. Plugins are not serialized.
    #[cfg_attr(feature = "clp", serde(skip))]
    Plugin(PluginResource),
    /// Numeric base transition retaining its literal function encoding.
    EncodedPiecewiseLinearResource {
        initial: f64,
        bounds: Window,
        functions: Vec<EncodedFunction>,
    },
    /// Numeric dependency retaining its literal function encoding and policy.
    EncodedPiecewiseLinearDependent {
        parent: usize,
        initial: f64,
        bounds: Window,
        functions: Vec<EncodedFunction>,
    },
    /// State-dependent objective retaining its literal encoding. Its domain
    /// remains a feasibility constraint when resource costs are suppressed.
    EncodedPiecewiseLinearCost {
        parent: usize,
        functions: Vec<EncodedFunction>,
    },
    /// Independent free-initial-value interval priced by ordered slope-front
    /// operations. This is the minimum of the computed operator front, not a
    /// promise of the global minimum of a sequential binary64 source program.
    /// Unknown operations fail explicitly; they never prove an empty route set.
    SlopeIntervalCost {
        initial: Window,
        bounds: Window,
        increments: Vec<f64>,
        functions: Vec<PreparedSlopeFunction>,
        policy: FrontOperationPolicy,
    },
    /// Paired resource using raw slope operators and an explicit objective
    /// normalization. Unsupported operators fail; no knot conversion is made.
    SlopePaired {
        initial: PreparedSlopeFunction,
        initial_base: Window,
        arcs: Vec<SlopeRelationArc>,
        mode: PairedResourceMode,
        normalization: PairedCostNormalization,
        policy: FrontOperationPolicy,
    },
    /// A general paired base/dependent frontier. For departure t and head x,
    /// x>=arrival(t), and the next dependent value is at least
    /// max(previous+increment(t)+waiting*(x-arrival(t)),lower(x)), subject to
    /// upper(x). All waiting choices and disconnected feasible domains survive.
    /// Cost modes use the explicit dependent objective normalization.
    /// An empty path costs zero under either convention.
    /// This independent relation cannot be used as a numeric dependency parent.
    PairedPiecewiseLinear {
        initial: PiecewiseLinearFunction,
        initial_base: Window,
        arcs: Vec<PairedResourceArc>,
        mode: PairedResourceMode,
        #[cfg_attr(feature = "clp", serde(default))]
        normalization: PairedCostNormalization,
    },

    /// Minimize total piecewise-linear charges over one shared free initial
    /// value. On each arc x += increment, then charge function(x). The full
    /// cost frontier is retained; no locally optimal point is substituted.
    /// Bounds and charge domains constrain every arrival. This independent
    /// resource cannot serve as a scalar dependency parent.
    IntervalCost {
        initial: Window,
        bounds: Window,
        increments: Vec<f64>,
        functions: Vec<PiecewiseLinearFunction>,
    },
    /// Values still possible for a route. Each arc intersects this domain with
    /// its mask; an empty intersection is infeasible. A two-value domain is a
    /// binary selection resource. Up to 64 distinct values are supported.
    ///
    /// `allowed` holds arc incompatibilities (one mask per arc) and
    /// `vertices` vertex incompatibilities (one mask per vertex, or empty
    /// for none); a route keeps the values every arc and visited vertex
    /// allow. With `relaxed`, each arc and arriving vertex first check
    /// feasibility, then an unremembered vertex restores the initial domain,
    /// forgetting prior eliminations. Forward extension checks the head;
    /// backward extension checks the tail. The master remembers every vertex
    /// of a minimal violating interval, including both endpoints, so subsequent
    /// labels retain those eliminations. Certification, incumbents and final
    /// routes always apply the exact rule.
    Selection {
        initial: u64,
        allowed: Vec<u64>,
        #[cfg_attr(feature = "clp", serde(default))]
        vertices: Vec<u64>,
        #[cfg_attr(feature = "clp", serde(default))]
        relaxed: bool,
    },
    /// Deliveries leave the depot loaded; each customer unloads before picking
    /// up. Arc quantities describe the service at that arc's head. Capacity is
    /// enforced at departure and after every service, including earlier stops
    /// when a later delivery increases the required departure load.
    SimultaneousPickupDelivery {
        capacity: f64,
        deliveries: Vec<f64>,
        pickups: Vec<f64>,
        /// Loads up to `capacity + tolerance` are feasible (default 1e-6).
        /// Use 0 for exact capacity.
        #[cfg_attr(feature = "clp", serde(default = "default_load_tolerance"))]
        tolerance: f64,
    },
    /// Worst-case load when at most `budget` served demands take their
    /// deviation (budgeted uncertainty). Arc quantities describe the service
    /// at that arc's head. Labels keep the worst load for each 0..=budget.
    BudgetedLoad {
        capacity: f64,
        budget: usize,
        demands: Vec<f64>,
        deviations: Vec<f64>,
    },
    /// Worst-case service start when at most `budget` arcs take their
    /// duration deviation. On arc i the duration (tail service plus travel)
    /// is `durations[i]`, or `durations[i] + deviations[i]`; service waits
    /// for and must start within `windows[i]`, the head's window. Waiting
    /// can absorb deviations, so this is not the sum of the largest ones.
    BudgetedTime {
        initial: f64,
        budget: usize,
        durations: Vec<f64>,
        deviations: Vec<f64>,
        windows: Vec<Window>,
    },
    /// On arc i, `x += increments[i]`, then `cost += x * weights[i]`. The cost
    /// correction is added to both original and reduced arc costs. Signed,
    /// finite data are supported; no monotone dominance is assumed.
    CumulativeCost {
        initial: f64,
        increments: Vec<f64>,
        weights: Vec<f64>,
        /// Optional upper bounds of this resource at the sink and at the
        /// source. Dominance reads them as promises, for nonnegative
        /// increments and weights: `sink_upper` bounds the total weight of
        /// any completion, so a forward label with a larger x pays
        /// (x_a - x_b) * sink_upper; `source_upper` bounds the increments
        /// of any prefix, which bounds the entry x of a backward label.
        #[cfg_attr(feature = "clp", serde(default))]
        sink_upper: Option<f64>,
        #[cfg_attr(feature = "clp", serde(default))]
        source_upper: Option<f64>,
    },
    /// Zero-initialized cumulative resource with independent dominance constants.
    /// On arc i, `x += increments[i]`, then `cost += x * weights[i]`. There are
    /// no vertex transitions or hard resource bounds. Signed finite data are
    /// supported. This fixed numeric resource is separate from `Plugin`.
    CustomResource {
        increments: Vec<f64>,
        weights: Vec<f64>,
        /// Optional promise bounding the total weight of any completion.
        /// Used for forward dominance with nonnegative increments and weights.
        #[cfg_attr(feature = "clp", serde(default))]
        forward_bound: Option<f64>,
        /// Optional promise bounding the increments of any prefix. Used for
        /// backward dominance with nonnegative increments and weights.
        #[cfg_attr(feature = "clp", serde(default))]
        backward_bound: Option<f64>,
    },
    /// Bounded affine dependency: `y += offsets[i] + factors[i] * parent_after`.
    /// The parent must be an earlier numeric resource.
    /// Bounds are checked without waiting after every arc, including the sink.
    /// This is an explicit affine contract, not a claim that all external
    /// dependency-type conventions have been recovered.
    Dependent {
        parent: usize,
        initial: f64,
        bounds: Window,
        offsets: Vec<f64>,
        factors: Vec<f64>,
    },
    /// Bounded dependency y += function(parent_after), one continuous
    /// piecewise-linear function per physical arc. Outside-domain arguments
    /// are infeasible. Parents must be earlier numeric resources.
    PiecewiseLinearDependent {
        parent: usize,
        initial: f64,
        bounds: Window,
        functions: Vec<PiecewiseLinearFunction>,
    },
    /// Direct bounded base-state transition `x = function[arc](x_before)`.
    /// This can encode state-dependent consumption, waiting and reset maps
    /// explicitly. Function-domain and output-window violations are infeasible.
    PiecewiseLinearResource {
        initial: f64,
        bounds: Window,
        functions: Vec<PiecewiseLinearFunction>,
    },
    /// Add `function[arc](parent_after)` to both original and reduced objectives.
    /// The earlier numeric parent's state is unchanged by this descriptor.
    /// Function argument domains remain constraints even in phase I.
    PiecewiseLinearCost {
        parent: usize,
        functions: Vec<PiecewiseLinearFunction>,
    },
}

fn finite(x: f64) -> Result<f64> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(invalid("nonfinite coupled resource arithmetic"))
    }
}
fn numbers(xs: &[f64], count: usize, nonnegative: bool) -> Result<()> {
    if xs.len() != count
        || xs
            .iter()
            .any(|x| !x.is_finite() || (nonnegative && *x < 0.))
    {
        return Err(invalid("invalid coupled resource arc quantities"));
    }
    Ok(())
}

impl CoupledResource {
    fn is_numeric(&self) -> bool {
        matches!(
            self,
            Self::CumulativeCost { .. }
                | Self::CustomResource { .. }
                | Self::Dependent { .. }
                | Self::PiecewiseLinearDependent { .. }
                | Self::PiecewiseLinearResource { .. }
                | Self::EncodedPiecewiseLinearResource { .. }
                | Self::EncodedPiecewiseLinearDependent { .. }
        )
    }
    fn validate<const D: usize>(
        &self,
        g: &Graph<D>,
        index: usize,
        resources: &[Self],
    ) -> Result<()> {
        let n = g.arcs.len();
        match self {
            Self::Plugin(plugin) => plugin.0.validate(n)?,
            Self::SlopePaired {
                initial,
                initial_base,
                arcs,
                mode,
                normalization,
                policy,
            } => {
                Window::new(initial_base.lower, initial_base.upper)?;
                let domain = initial.domain();
                if !initial_base.lower.is_finite()
                    || !initial_base.upper.is_finite()
                    || initial_base.lower < domain.0
                    || initial_base.upper > domain.1
                    || arcs.len() != n
                {
                    return Err(invalid("raw paired resource requires an initial interval inside its descriptor and one relation per physical arc"));
                }
                // Mode/operator validation is performed by the prepared
                // paired constructor below, before kernel installation. Do
                // not construct every inverse/front twice at this boundary.
                let _ = (mode, normalization, policy);
            }
            Self::PairedPiecewiseLinear {
                initial,
                initial_base,
                arcs,
                mode,
                ..
            } => {
                initial.validate()?;
                Window::new(initial_base.lower, initial_base.upper)?;
                if !initial_base.lower.is_finite()
                    || !initial_base.upper.is_finite()
                    || arcs.len() != n
                    || initial_base.lower < initial.prepared_domain().lower
                    || initial_base.upper > initial.prepared_domain().upper
                {
                    return Err(invalid("paired resource needs a finite initial base interval inside its initial front and one relation per arc"));
                }
                for arc in arcs {
                    arc.validate()?;
                }
                if *mode != PairedResourceMode::Feasibility {
                    kernel::relation::validate_cost_mode(
                        arcs,
                        kernel::relation::RelationFront::initial(initial, *initial_base)?
                            .ok_or_else(|| invalid("empty paired initial front"))?
                            .minimum(),
                        *mode == PairedResourceMode::CostBounded,
                    )?;
                }
            }

            Self::SlopeIntervalCost {
                initial,
                bounds,
                increments,
                functions,
                ..
            } => {
                Window::new(initial.lower, initial.upper)?;
                Window::new(bounds.lower, bounds.upper)?;
                if ![initial.lower, initial.upper, bounds.lower, bounds.upper]
                    .iter()
                    .all(|x| x.is_finite())
                    || initial.lower < bounds.lower
                    || initial.upper > bounds.upper
                    || functions.len() != n
                {
                    return Err(invalid("slope interval cost needs finite nested windows and one descriptor per physical arc"));
                }
                numbers(increments, n, false)?;
                for function in functions {
                    PreparedSlopeFunction::from_prepared_pairs(function.pairs().to_vec())
                        .map_err(|error| invalid(error.to_string()))?;
                }
            }
            Self::IntervalCost {
                initial,
                bounds,
                increments,
                functions,
            } => {
                Window::new(initial.lower, initial.upper)?;
                Window::new(bounds.lower, bounds.upper)?;
                if ![initial.lower, initial.upper, bounds.lower, bounds.upper]
                    .iter()
                    .all(|x| x.is_finite())
                    || initial.lower < bounds.lower
                    || initial.upper > bounds.upper
                    || functions.len() != n
                {
                    return Err(invalid("interval cost needs finite nested initial/state windows and one charge per arc"));
                }
                numbers(increments, n, false)?;
                for function in functions {
                    function.validate()?;
                }
            }
            Self::Selection {
                initial,
                allowed,
                vertices,
                ..
            } => {
                if *initial == 0 || allowed.len() != n {
                    return Err(invalid(
                        "selection needs a nonempty initial domain and one mask per arc",
                    ));
                }
                if !vertices.is_empty()
                    && (vertices.len() != g.vertices.len() || initial & vertices[g.source] == 0)
                {
                    return Err(invalid(
                        "selection vertex masks need one per vertex and must admit the source",
                    ));
                }
            }
            Self::SimultaneousPickupDelivery {
                capacity,
                deliveries,
                pickups,
                tolerance,
            } => {
                if !capacity.is_finite() || *capacity < 0. {
                    return Err(invalid("invalid pickup/delivery capacity"));
                }
                if !tolerance.is_finite() || *tolerance < 0. {
                    return Err(invalid("invalid pickup/delivery tolerance"));
                }
                numbers(deliveries, n, true)?;
                numbers(pickups, n, true)?;
            }
            Self::BudgetedLoad {
                capacity,
                demands,
                deviations,
                ..
            } => {
                if !capacity.is_finite() || *capacity < 0. {
                    return Err(invalid("invalid budgeted load capacity"));
                }
                numbers(demands, n, true)?;
                numbers(deviations, n, true)?;
            }
            Self::BudgetedTime {
                initial,
                durations,
                deviations,
                windows,
                ..
            } => {
                if !initial.is_finite() || windows.len() != n {
                    return Err(invalid(
                        "budgeted time needs a finite start and one head window per arc",
                    ));
                }
                numbers(durations, n, true)?;
                numbers(deviations, n, true)?;
                for window in windows {
                    if !window.lower.is_finite() || !window.upper.is_finite() {
                        return Err(invalid("budgeted time windows must be finite"));
                    }
                    Window::new(window.lower, window.upper)?;
                }
            }
            Self::CumulativeCost {
                increments,
                weights,
                sink_upper,
                source_upper,
                ..
            }
            | Self::CustomResource {
                increments,
                weights,
                forward_bound: sink_upper,
                backward_bound: source_upper,
            } => {
                if let Self::CumulativeCost { initial, .. } = self {
                    finite(*initial)?;
                }
                numbers(increments, n, false)?;
                numbers(weights, n, false)?;
                if [sink_upper, source_upper]
                    .into_iter()
                    .flatten()
                    .any(|k| !k.is_finite() || *k < 0.)
                {
                    return Err(invalid(
                        "cumulative dominance bounds must be finite and nonnegative",
                    ));
                }
            }
            Self::PiecewiseLinearResource {
                initial,
                bounds,
                functions,
            } => {
                Window::new(bounds.lower, bounds.upper)?;
                if !initial.is_finite()
                    || *initial < bounds.lower
                    || *initial > bounds.upper
                    || functions.len() != n
                {
                    return Err(invalid("piecewise base resources need a bounded finite initial value and one function per arc"));
                }
                for function in functions {
                    function.validate()?;
                }
            }
            Self::PiecewiseLinearCost { parent, functions } => {
                if *parent >= index || !resources[*parent].is_numeric() || functions.len() != n {
                    return Err(invalid(
                        "piecewise cost needs an earlier numeric parent and one function per arc",
                    ));
                }
                for function in functions {
                    function.validate()?;
                }
            }
            Self::PiecewiseLinearDependent {
                parent,
                initial,
                bounds,
                functions,
            } => {
                Window::new(bounds.lower, bounds.upper)?;
                if *parent >= index
                    || !resources[*parent].is_numeric()
                    || !initial.is_finite()
                    || *initial < bounds.lower
                    || *initial > bounds.upper
                    || functions.len() != n
                {
                    return Err(invalid("piecewise-linear dependency needs an earlier numeric parent, bounded initial value and one function per arc"));
                }
                for function in functions {
                    function.validate()?;
                }
            }
            Self::EncodedPiecewiseLinearResource {
                initial,
                bounds,
                functions,
            } => {
                Window::new(bounds.lower, bounds.upper)?;
                if !initial.is_finite()
                    || *initial < bounds.lower
                    || *initial > bounds.upper
                    || functions.len() != n
                {
                    return Err(invalid("piecewise base resources need a bounded finite initial value and one function per arc"));
                }
                for function in functions {
                    function.validate()?;
                }
            }
            Self::EncodedPiecewiseLinearCost { parent, functions } => {
                if *parent >= index || !resources[*parent].is_numeric() || functions.len() != n {
                    return Err(invalid(
                        "piecewise cost needs an earlier numeric parent and one function per arc",
                    ));
                }
                for function in functions {
                    function.validate()?;
                }
            }
            Self::EncodedPiecewiseLinearDependent {
                parent,
                initial,
                bounds,
                functions,
            } => {
                Window::new(bounds.lower, bounds.upper)?;
                if *parent >= index
                    || !resources[*parent].is_numeric()
                    || !initial.is_finite()
                    || *initial < bounds.lower
                    || *initial > bounds.upper
                    || functions.len() != n
                {
                    return Err(invalid("piecewise-linear dependency needs an earlier numeric parent, bounded initial value and one function per arc"));
                }
                for function in functions {
                    function.validate()?;
                }
            }
            Self::Dependent {
                parent,
                initial,
                bounds,
                offsets,
                factors,
            } => {
                Window::new(bounds.lower, bounds.upper)?;
                if *parent >= index
                    || !resources[*parent].is_numeric()
                    || !initial.is_finite()
                    || *initial < bounds.lower
                    || *initial > bounds.upper
                {
                    return Err(invalid("dependent resource needs an earlier numeric parent and a bounded initial value"));
                }
                numbers(offsets, n, false)?;
                numbers(factors, n, false)?;
            }
        }
        Ok(())
    }
}
