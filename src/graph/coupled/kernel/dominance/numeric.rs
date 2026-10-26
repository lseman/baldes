//! Per-operation zero-compensation numeric cost orders.
//!
//! This metadata never spends a mixed-key gap. Independently ordered physical
//! scalar bases and numeric prefix costs are still required by the caller.
use super::super::bounds::Range;
use super::super::*;
use crate::core::rounding::add_up;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Slot {
    // Other numeric operations retain DependencyOrder's closed feasibility
    // proof; every function-valued cost parent is required equal there.
    Other,
    Uncharged { increment_direction: i8 },
    Positive,
    Negative,
    Equal,
}
#[derive(Clone, Debug, PartialEq)]
pub(in crate::graph::coupled::kernel) struct NumericCostOrder {
    slots: Vec<Slot>,
    // Optional user bounds on any completion's total weight, per slot.
    future: Vec<Option<Range>>,
    nonnegative_charges: bool,
    deploy_unequal: bool,
}
impl NumericCostOrder {
    pub(in crate::graph::coupled::kernel) fn new(
        definitions: &[CoupledResource],
        slots: &[Option<usize>],
        numeric_count: usize,
    ) -> Self {
        let mut order = vec![Slot::Other; numeric_count];
        let mut future = vec![None; numeric_count];
        let mut nonnegative_charges = true;
        for (r, definition) in definitions.iter().enumerate() {
            match definition {
                CoupledResource::CumulativeCost {
                    increments,
                    weights,
                    sink_upper,
                    ..
                }
                | CoupledResource::CustomResource {
                    increments,
                    weights,
                    forward_bound: sink_upper,
                    ..
                } => {
                    let increasing = increments.iter().all(|&x| x >= 0.);
                    let decreasing = increments.iter().all(|&x| x <= 0.);
                    let kind = if weights.iter().all(|&x| x == 0.) {
                        Slot::Uncharged {
                            increment_direction: if increasing && decreasing {
                                0
                            } else if increasing {
                                1
                            } else if decreasing {
                                -1
                            } else {
                                2
                            },
                        }
                    } else if increasing && weights.iter().all(|&w| w >= 0.) {
                        Slot::Positive
                    } else if decreasing && weights.iter().all(|&w| w <= 0.) {
                        Slot::Negative
                    } else {
                        nonnegative_charges = false;
                        Slot::Equal
                    };
                    let slot = slots[r].expect("validated numeric resource slot");
                    order[slot] = kind;
                    // Like the reference: [0, sink bound] in the positive cone.
                    if kind == Slot::Positive {
                        future[slot] = sink_upper.map(|k| Range { lo: 0., hi: k });
                    }
                }
                CoupledResource::PiecewiseLinearCost { functions, .. } => {
                    // Its argument stays equal under DependencyOrder. Ordered
                    // interpolation with endpoint clamps preserves this sign.
                    nonnegative_charges &= functions
                        .iter()
                        .all(|f| f.points.iter().all(|p| p[1] >= 0.));
                }
                CoupledResource::EncodedPiecewiseLinearCost { .. } => {
                    // Parent equality is certified independently. Encoded
                    // values may be signed, so retain equal-prefix-cost only.
                    nonnegative_charges = false;
                }
                _ => {}
            }
        }
        // The order is also safe with a piecewise cost: DependencyOrder
        // requires its parent scalar to remain equal, so its future charges
        // are identical in both labels. Other rich dependency programs retain
        // the baseline identity path until their composition is established.
        let deploy_unequal = definitions.iter().all(|definition| {
            matches!(
                definition,
                CoupledResource::CumulativeCost { .. }
                    | CoupledResource::CustomResource { .. }
                    | CoupledResource::PiecewiseLinearCost { .. }
                    | CoupledResource::EncodedPiecewiseLinearCost { .. }
                    | CoupledResource::Selection { .. }
                    | CoupledResource::SimultaneousPickupDelivery { .. }
                    | CoupledResource::BudgetedLoad { .. }
                    | CoupledResource::BudgetedTime { .. }
            )
        });
        Self {
            slots: order,
            future,
            nonnegative_charges,
            deploy_unequal,
        }
    }

    #[inline]
    pub(in crate::graph::coupled::kernel) fn deploys_unequal(&self) -> bool {
        self.deploy_unequal
    }

    /// Every future numeric objective operation preserves A_cost <= B_cost.
    /// This must be combined with DependencyOrder::preserves: function domains,
    /// dependent bounds, and function-cost parent equality are not checked here.
    #[inline]
    pub(in crate::graph::coupled::kernel) fn preserves(
        &self,
        a: &[f64],
        b: &[f64],
        a_cost: f64,
        b_cost: f64,
    ) -> bool {
        if a.len() != self.slots.len()
            || b.len() != self.slots.len()
            || !a_cost.is_finite()
            || !b_cost.is_finite()
            || a_cost > b_cost
        {
            return false;
        }
        if self.nonnegative_charges && a_cost < 0. {
            // This fast path requires a nonnegative prefix as well as future
            // charges. Earlier histories outside the cone, and foreign public
            // states, remain on the conservative fallback.
            return false;
        }
        if !self.nonnegative_charges && a_cost != b_cost {
            // Common negative charges can overflow a smaller prefix while
            // the comparator stays finite. Equal inputs and equal cost keep
            // all signed objective operations literally identical instead.
            return false;
        }
        // Identical numeric inputs and identical accumulated numeric costs
        // replay every future objective operation identically. This common
        // case needs no per-slot sign/cone dispatch. Retain finite validation
        // for standalone/public resource-state comparisons.
        if a_cost == b_cost && a == b {
            return a.iter().all(|x| x.is_finite());
        }
        // A bounded slot may put A on the wrong side of B: along a common
        // completion the states keep their difference, so A's extra future
        // cost is at most max(0, (a - b) * W). The numeric prefix pays it.
        let mut penalty = 0.;
        let ordered = self
            .slots
            .iter()
            .zip(&self.future)
            .zip(a.iter().zip(b))
            .all(|((slot, future), (&a, &b))| {
                if !a.is_finite() || !b.is_finite() {
                    return false;
                }
                match slot {
                    Slot::Other => true,
                    Slot::Uncharged {
                        increment_direction,
                    } => {
                        // Preserve existence of a finite dominator update even
                        // for zero-weight numeric states and unbounded walks.
                        match increment_direction {
                            0 => true,
                            1 => a <= b,
                            -1 => a >= b,
                            _ => a == b,
                        }
                    }
                    Slot::Equal => a == b,
                    Slot::Positive | Slot::Negative => {
                        if !self.nonnegative_charges {
                            return a == b;
                        }
                        let positive = *slot == Slot::Positive;
                        // State, product, scaling and charge keep the cone's
                        // sign. If B's literal operation is finite, A's is too.
                        let in_cone = if positive {
                            0. <= a.min(b)
                        } else {
                            a.max(b) <= 0.
                        };
                        let ordered = if positive {
                            0. <= a && a <= b
                        } else {
                            b <= a && a <= 0.
                        };
                        if ordered {
                            return true;
                        }
                        let Some(weight) = future else {
                            return false;
                        };
                        let extra = Range::point(a)
                            .subtract(Range::point(b))
                            .multiply(*weight)
                            .hi
                            .max(0.);
                        match add_up(penalty, extra) {
                            Some(sum) if in_cone && sum.is_finite() => {
                                penalty = sum;
                                true
                            }
                            _ => false,
                        }
                    }
                }
            });
        ordered && (penalty == 0. || add_up(a_cost, penalty).is_some_and(|c| c <= b_cost))
    }
}

#[cfg(test)]
#[path = "../../../../../tests/unit/graph/coupled/kernel/dominance/numeric/tests.rs"]
mod tests;
