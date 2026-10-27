//! Arc-local numeric programs and affine maps, shared by all physical labels.
use super::*;

#[derive(Clone, Debug, PartialEq)]
pub(super) enum NumericOperation {
    Cumulative {
        slot: usize,
        increment: f64,
        weight: f64,
        cost_coefficient: f64,
    },
    Dependent {
        slot: usize,
        parent: usize,
        offset: f64,
        factor: f64,
        bounds: Window,
    },
    PiecewiseLinear {
        slot: usize,
        parent: usize,
        bounds: Window,
        function: std::sync::Arc<EncodedFunction>,
    },
    BaseMap {
        slot: usize,
        bounds: Window,
        function: std::sync::Arc<EncodedFunction>,
    },
    StateCost {
        parent: usize,
        function: std::sync::Arc<EncodedFunction>,
    },
}
impl NumericOperation {
    pub(super) fn apply(&self, numeric: &mut [f64], scale: f64, cost: &mut f64) -> Result<bool> {
        match *self {
            Self::Cumulative {
                slot,
                increment,
                weight,
                cost_coefficient,
            } => {
                numeric[slot] = finite(numeric[slot] + increment)?;
                let scale = finite(scale * cost_coefficient)?;
                if scale != 0. {
                    *cost = finite(*cost + finite(scale * finite(numeric[slot] * weight)?)?)?;
                }
            }
            Self::BaseMap {
                slot, ref function, ..
            } => {
                let Some(value) = function.evaluate(numeric[slot])? else {
                    return Ok(false);
                };
                numeric[slot] = value;
            }
            Self::StateCost {
                parent,
                ref function,
            } => {
                let Some(value) = function.evaluate(numeric[parent])? else {
                    return Ok(false);
                };
                if scale != 0. {
                    *cost = finite(*cost + finite(scale * value)?)?;
                }
            }
            Self::PiecewiseLinear {
                slot,
                parent,
                ref function,
                ..
            } => {
                let Some(delta) = function.evaluate(numeric[parent])? else {
                    return Ok(false);
                };
                numeric[slot] = finite(numeric[slot] + delta)?;
            }
            Self::Dependent {
                slot,
                parent,
                offset,
                factor,
                ..
            } => {
                numeric[slot] =
                    finite(numeric[slot] + finite(offset + finite(factor * numeric[parent])?)?)?;
            }
        }
        Ok(true)
    }
    pub(super) fn feasible(&self, numeric: &[f64]) -> bool {
        match *self {
            Self::Dependent { slot, bounds, .. }
            | Self::PiecewiseLinear { slot, bounds, .. }
            | Self::BaseMap { slot, bounds, .. } => {
                numeric[slot] >= bounds.lower && numeric[slot] <= bounds.upper
            }
            Self::Cumulative { .. } | Self::StateCost { .. } => true,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct ArcTransition {
    pub(super) matrix: Vec<Vec<f64>>,
    pub(super) offsets: Vec<f64>,
    pub(super) coefficients: Vec<f64>,
    pub(super) constant: f64,
    pub(super) bounds: Vec<AffineBound>,
}
impl ArcTransition {
    fn prepare(
        operations: &[NumericOperation],
        n: usize,
        scale: f64,
        deferred: bool,
    ) -> Result<Self> {
        if deferred {
            // Numeric constraints are evaluated by the shared continuation.
            // Cumulative objectives depend only on their own additive state,
            // so their map is diagonal: no dense matrix is needed here.
            let mut offsets = vec![0.; n];
            let mut coefficients = vec![0.; n];
            let mut constant = 0.;
            for operation in operations {
                if let NumericOperation::Cumulative {
                    slot,
                    increment,
                    weight,
                    cost_coefficient,
                } = *operation
                {
                    offsets[slot] = finite(offsets[slot] + increment)?;
                    let w = finite(finite(scale * cost_coefficient)? * weight)?;
                    coefficients[slot] = finite(coefficients[slot] + w)?;
                    constant = finite(constant + finite(w * offsets[slot])?)?;
                }
            }
            return Ok(Self {
                matrix: Vec::new(),
                offsets,
                coefficients,
                constant,
                bounds: Vec::new(),
            });
        }
        let mut matrix = vec![vec![0.; n]; n];
        for (i, row) in matrix.iter_mut().enumerate() {
            row[i] = 1.;
        }
        let mut offsets = vec![0.; n];
        let mut coefficients = vec![0.; n];
        let mut constant = 0.;
        let mut bounds_out = Vec::new();
        for operation in operations {
            match *operation {
                NumericOperation::Cumulative {
                    slot: i,
                    increment,
                    weight,
                    cost_coefficient,
                } => {
                    offsets[i] = finite(offsets[i] + increment)?;
                    let w = finite(finite(scale * cost_coefficient)? * weight)?;
                    for (coefficient, &value) in coefficients.iter_mut().zip(&matrix[i]) {
                        *coefficient = finite(*coefficient + finite(w * value)?)?;
                    }
                    constant = finite(constant + finite(w * offsets[i])?)?;
                }
                NumericOperation::PiecewiseLinear { .. }
                | NumericOperation::BaseMap { .. }
                | NumericOperation::StateCost { .. } => {}
                NumericOperation::Dependent {
                    slot: i,
                    parent: p,
                    offset,
                    factor: f,
                    bounds,
                } => {
                    if deferred {
                        continue;
                    }
                    let (earlier, current) = matrix.split_at_mut(i);
                    for (value, &parent) in current[0].iter_mut().zip(&earlier[p]) {
                        *value = finite(*value + finite(f * parent)?)?;
                    }
                    offsets[i] = finite(offsets[i] + finite(offset + finite(f * offsets[p])?)?)?;
                    bounds_out.push(AffineBound {
                        coefficients: matrix[i].clone().into(),
                        offset: offsets[i],
                        bounds,
                    });
                }
            }
        }
        Ok(Self {
            matrix,
            offsets,
            coefficients,
            constant,
            bounds: bounds_out,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct PreparedArc {
    pub(super) operations: Vec<NumericOperation>,
    pub(super) selection: Vec<u64>,
    pub(super) loads: Vec<[f64; 3]>,
    // An overflow in a symbolic map must still be reported only when a
    // backward label actually uses this arc. Forward replay may remain finite.
    pub(super) backward: Option<ArcTransition>,
    pub(super) impossible: bool,
    pub(super) state_costs: CostEnvelope,
}
#[derive(Clone, Debug, PartialEq)]
pub(super) struct PreparedTransitions {
    /// Original descriptor identity of each ordered numeric operation.
    /// This order is identical on every arc and is stored only once.
    pub(super) operation_resources: std::sync::Arc<[usize]>,
    pub(super) arcs: std::sync::Arc<[PreparedArc]>,
    pub(super) deferred: bool,
}
impl PreparedTransitions {
    pub(super) fn new(
        definitions: &[CoupledResource],
        slots: &[Option<usize>],
        n: usize,
        arcs: usize,
        scale: f64,
        objective_coefficients: &[f64],
    ) -> Self {
        let operation_resources = definitions
            .iter()
            .enumerate()
            .filter(|(_, definition)| {
                definition.is_numeric()
                    || matches!(
                        definition,
                        CoupledResource::PiecewiseLinearCost { .. }
                            | CoupledResource::EncodedPiecewiseLinearCost { .. }
                    )
            })
            .map(|(resource, _)| resource)
            .collect::<std::sync::Arc<[usize]>>();
        let deferred = definitions.iter().any(|r| {
            matches!(
                r,
                CoupledResource::PiecewiseLinearDependent { .. }
                    | CoupledResource::PiecewiseLinearResource { .. }
                    | CoupledResource::PiecewiseLinearCost { .. }
                    | CoupledResource::EncodedPiecewiseLinearResource { .. }
                    | CoupledResource::EncodedPiecewiseLinearDependent { .. }
                    | CoupledResource::EncodedPiecewiseLinearCost { .. }
            )
        });
        let arcs = (0..arcs)
            .map(|arc| {
                let operations: Vec<_> = definitions
                    .iter()
                    .enumerate()
                    .filter_map(|(r, definition)| match definition {
                        CoupledResource::PiecewiseLinearResource {
                            bounds, functions, ..
                        } => Some(NumericOperation::BaseMap {
                            slot: slots[r].unwrap(),
                            bounds: *bounds,
                            function: std::sync::Arc::new(functions[arc].clone().into()),
                        }),
                        CoupledResource::PiecewiseLinearCost { parent, functions } => {
                            Some(NumericOperation::StateCost {
                                parent: slots[*parent].unwrap(),
                                function: std::sync::Arc::new(functions[arc].clone().into()),
                            })
                        }
                        CoupledResource::EncodedPiecewiseLinearResource {
                            bounds,
                            functions,
                            ..
                        } => Some(NumericOperation::BaseMap {
                            slot: slots[r].unwrap(),
                            bounds: *bounds,
                            function: std::sync::Arc::new(functions[arc].clone().into()),
                        }),
                        CoupledResource::EncodedPiecewiseLinearCost { parent, functions } => {
                            Some(NumericOperation::StateCost {
                                parent: slots[*parent].unwrap(),
                                function: std::sync::Arc::new(functions[arc].clone().into()),
                            })
                        }
                        CoupledResource::CumulativeCost {
                            increments,
                            weights,
                            ..
                        }
                        | CoupledResource::CustomResource {
                            increments,
                            weights,
                            ..
                        } => Some(NumericOperation::Cumulative {
                            slot: slots[r].unwrap(),
                            increment: increments[arc],
                            weight: weights[arc],
                            cost_coefficient: objective_coefficients[r],
                        }),
                        CoupledResource::PiecewiseLinearDependent {
                            parent,
                            bounds,
                            functions,
                            ..
                        } => Some(NumericOperation::PiecewiseLinear {
                            slot: slots[r].unwrap(),
                            parent: slots[*parent].unwrap(),
                            bounds: *bounds,
                            function: std::sync::Arc::new(functions[arc].clone().into()),
                        }),
                        CoupledResource::EncodedPiecewiseLinearDependent {
                            parent,
                            bounds,
                            functions,
                            ..
                        } => Some(NumericOperation::PiecewiseLinear {
                            slot: slots[r].unwrap(),
                            parent: slots[*parent].unwrap(),
                            bounds: *bounds,
                            function: std::sync::Arc::new(functions[arc].clone().into()),
                        }),
                        CoupledResource::Dependent {
                            parent,
                            bounds,
                            offsets,
                            factors,
                            ..
                        } => Some(NumericOperation::Dependent {
                            slot: slots[r].unwrap(),
                            parent: slots[*parent].unwrap(),
                            offset: offsets[arc],
                            factor: factors[arc],
                            bounds: *bounds,
                        }),
                        _ => None,
                    })
                    .collect();
                let backward = ArcTransition::prepare(&operations, n, scale, deferred).ok();
                let impossible = dependency_impossible(&operations, n);
                let state_costs = state_cost_envelope(&operations, scale);
                PreparedArc {
                    selection: definitions
                        .iter()
                        .filter_map(|r| match r {
                            CoupledResource::Selection { allowed, .. } => Some(allowed[arc]),
                            _ => None,
                        })
                        .collect(),
                    loads: definitions
                        .iter()
                        .filter_map(|r| match r {
                            CoupledResource::SimultaneousPickupDelivery {
                                deliveries,
                                pickups,
                                ..
                            } => {
                                let net = pickups[arc] - deliveries[arc];
                                Some([deliveries[arc], net, net.max(0.)])
                            }
                            _ => None,
                        })
                        .collect(),
                    operations,
                    backward,
                    impossible,
                    state_costs,
                }
            })
            .collect::<Vec<_>>();
        Self {
            operation_resources,
            arcs: arcs.into(),
            deferred,
        }
    }
    #[cfg(feature = "clp")]
    pub(super) fn suppress_costs(&mut self, n: usize) {
        for arc in std::sync::Arc::make_mut(&mut self.arcs) {
            arc.state_costs = Default::default();
            arc.backward = ArcTransition::prepare(&arc.operations, n, 0., self.deferred).ok();
        }
    }
}

/// Necessary entry/exit-window propagation, without assuming reachable numeric
/// values or bounding the length of a walk. Unknown arithmetic retains the arc.
fn dependency_impossible(operations: &[NumericOperation], n: usize) -> bool {
    use crate::core::rounding::{add_down, add_up, multiply_down};
    let mut exits: Vec<Option<Window>> = vec![None; n];
    for operation in operations {
        if let NumericOperation::Dependent {
            slot,
            parent,
            offset,
            factor,
            bounds,
        } = *operation
        {
            // The earlier parent has already been updated on this arc. Its
            // declared window is valid at that point, including at the root.
            let image = (|| {
                let (product_lo, product_hi) = if factor == 0. {
                    (0., 0.)
                } else {
                    let parent = exits[parent]?;
                    let (lo, hi) = if factor >= 0. {
                        (parent.lower, parent.upper)
                    } else {
                        (parent.upper, parent.lower)
                    };
                    (multiply_down(factor, lo)?, -multiply_down(-factor, hi)?)
                };
                let lower = add_down(bounds.lower, add_down(offset, product_lo)?)?;
                let upper = add_up(bounds.upper, add_up(offset, product_hi)?)?;
                Some(Window { lower, upper })
            })();
            if let Some(image) = image {
                if image.upper < bounds.lower || image.lower > bounds.upper {
                    return true;
                }
                exits[slot] = Some(Window {
                    lower: image.lower.max(bounds.lower),
                    upper: image.upper.min(bounds.upper),
                });
            } else {
                exits[slot] = Some(bounds);
            }
        }
    }
    false
}

fn state_cost_envelope(operations: &[NumericOperation], scale: f64) -> CostEnvelope {
    operations
        .iter()
        .fold(CostEnvelope::default(), |sum, operation| {
            if let NumericOperation::StateCost { function, .. } = operation {
                let range = match function.as_ref() {
                    // Preserve the legacy clamped-knot envelope exactly.
                    EncodedFunction::Knots { function } => function.points.iter().fold(
                        bounds::Range {
                            lo: f64::INFINITY,
                            hi: f64::NEG_INFINITY,
                        },
                        |range, point| bounds::Range {
                            lo: range.lo.min(point[1]),
                            hi: range.hi.max(point[1]),
                        },
                    ),
                    // Endpoint values do not bound a literal slope evaluator
                    // under every policy. Unknown disables completion pruning.
                    EncodedFunction::SlopeSegments { .. } => bounds::Range::unknown(),
                }
                .multiply(bounds::Range::point(scale));
                sum.combine(&CostEnvelope::charge(range))
            } else {
                sum
            }
        })
}
