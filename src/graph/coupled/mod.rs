//! Direct coupled label resources over the physical graph.
//!
//! Install with `Graph::set_coupled_resources` or the corresponding Solver
//! method. Forward/backward extension, joins and replay share a prepared kernel.
use super::{Graph, Window};
use crate::{invalid, Result};

pub mod descriptor;
mod encoded;
mod kernel;
mod pwl;
pub use descriptor::{
    EvaluationPolicy, FrontOperationPolicy, PairedCostNormalization, PreparedSlopeFunction,
    SlopeRelationArc,
};
pub use encoded::EncodedFunction;
pub use kernel::{CoupledKernel, CoupledLabelState, CoupledWorkspace};
pub(crate) use kernel::{CoupledPairSummary, CoupledStateBounds};
pub use pwl::PiecewiseLinearFunction;

/// Travel followed by service and optional continuous charging at the head.
/// Charging maps energy to cumulative time from empty. Customers constrain
/// service start; a charging stop also constrains completion by closing time.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "clp", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "clp", serde(deny_unknown_fields))]
pub struct ChargingResourceArc {
    pub travel_time: f64,
    pub energy: f64,
    pub window: Window,
    pub service: f64,
    pub charging: Option<PiecewiseLinearFunction>,
}

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

/// Resource inputs use the original graph's arc vector order, not arc IDs.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "clp", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(
    feature = "clp",
    serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)
)]
pub enum CoupledResource {
    /// Earliest departure time as a continuous function of remaining energy.
    /// All partial charging choices survive. Adds no objective, cannot be a
    /// numeric dependency parent, and does not claim directed certification.
    /// `max_work` bounds each frontier extension, including envelope work;
    /// exhaustion is an error, never a proof of infeasibility.
    Charging {
        battery: f64,
        initial_energy: f64,
        initial_time: f64,
        arcs: Vec<ChargingResourceArc>,
        max_work: usize,
    },
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
    /// Cost modes minimize the final dependent coordinate minus the minimum
    /// initial dependent coordinate; an empty path therefore costs zero.
    /// This independent relation cannot be used as a numeric dependency parent.
    PairedPiecewiseLinear {
        initial: PiecewiseLinearFunction,
        initial_base: Window,
        arcs: Vec<PairedResourceArc>,
        mode: PairedResourceMode,
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
    Selection { initial: u64, allowed: Vec<u64> },
    /// Deliveries leave the depot loaded; each customer unloads before picking
    /// up. Arc quantities describe the service at that arc's head. Capacity is
    /// enforced at departure and after every service, including earlier stops
    /// when a later delivery increases the required departure load.
    SimultaneousPickupDelivery {
        capacity: f64,
        deliveries: Vec<f64>,
        pickups: Vec<f64>,
    },
    /// On arc i, x += increments[i], then cost += x * weights[i]. The cost
    /// correction is added to both original and reduced arc costs. Signed,
    /// finite data are supported; no monotone dominance is assumed.
    CumulativeCost {
        initial: f64,
        increments: Vec<f64>,
        weights: Vec<f64>,
    },
    /// Bounded affine dependency: y += offsets[i] + factors[i] * parent_after.
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
    /// Direct bounded base-state transition x = function[arc](x_before).
    /// This can encode state-dependent consumption, waiting and reset maps
    /// explicitly. Function-domain and output-window violations are infeasible.
    PiecewiseLinearResource {
        initial: f64,
        bounds: Window,
        functions: Vec<PiecewiseLinearFunction>,
    },
    /// Add function[arc](parent_after) to both original and reduced objectives.
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
            Self::Charging {
                battery,
                initial_energy,
                initial_time,
                arcs,
                ..
            } => {
                if !battery.is_finite()
                    || *battery <= 0.
                    || !initial_energy.is_finite()
                    || *initial_energy < 0.
                    || initial_energy > battery
                    || !initial_time.is_finite()
                    || *initial_time < 0.
                    || arcs.len() != n
                {
                    return Err(invalid(
                        "invalid charging resource initial state or arc count",
                    ));
                }
                for arc in arcs {
                    Window::new(arc.window.lower, arc.window.upper)?;
                    if [
                        arc.travel_time,
                        arc.energy,
                        arc.service,
                        arc.window.lower,
                        arc.window.upper,
                    ]
                    .iter()
                    .any(|x| !x.is_finite() || *x < 0.)
                    {
                        return Err(invalid(
                            "invalid charging resource travel, service or window",
                        ));
                    }
                    if let Some(curve) = &arc.charging {
                        curve.validate()?;
                        if curve.points[0] != [0., 0.]
                            || curve.points.last().unwrap()[0] != *battery
                            || curve.points.windows(2).any(|w| w[0][1] >= w[1][1])
                        {
                            return Err(invalid("charging curve must strictly increase from zero to battery capacity"));
                        }
                    }
                }
            }
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
            Self::Selection { initial, allowed } => {
                if *initial == 0 || allowed.len() != n {
                    return Err(invalid(
                        "selection needs a nonempty initial domain and one mask per arc",
                    ));
                }
            }
            Self::SimultaneousPickupDelivery {
                capacity,
                deliveries,
                pickups,
            } => {
                if !capacity.is_finite() || *capacity < 0. {
                    return Err(invalid("invalid pickup/delivery capacity"));
                }
                numbers(deliveries, n, true)?;
                numbers(pickups, n, true)?;
            }
            Self::CumulativeCost {
                initial,
                increments,
                weights,
            } => {
                finite(*initial)?;
                numbers(increments, n, false)?;
                numbers(weights, n, false)?;
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
