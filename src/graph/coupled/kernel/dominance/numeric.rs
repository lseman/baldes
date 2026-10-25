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
                    nonnegative_charges = false;
                }
                _ => {}
            }
        }
        // Measured activation policy: the per-operation proof is deployed on
        // independent additive numeric resources. Rich dependency programs
        // keep the baseline identity path rather than paying for extra proof
        // checks that did not improve their measured labeling workload.
        let deploy_unequal = definitions.iter().all(|definition| {
            matches!(
                definition,
                CoupledResource::CumulativeCost { .. }
                    | CoupledResource::CustomResource { .. }
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
mod tests {
    use super::*;
    fn cumulative(increments: Vec<f64>, weights: Vec<f64>) -> CoupledResource {
        CoupledResource::CumulativeCost {
            initial: 0.,
            increments,
            weights,
            sink_upper: None,
            source_upper: None,
        }
    }
    #[test]
    fn accepted_positive_and_negative_cones_preserve_literal_costs_over_all_small_words() {
        for sign in [-1., 1.] {
            let increments = [sign, 2. * sign];
            let weights = [sign, 3. * sign];
            let order = NumericCostOrder::new(
                &[cumulative(increments.to_vec(), weights.to_vec())],
                &[Some(0)],
                1,
            );
            for smaller in 0..=3 {
                for larger in smaller..=3 {
                    let a0 = sign * f64::from(smaller);
                    let b0 = sign * f64::from(larger);
                    for ac in 0..=3 {
                        for bc in ac..=3 {
                            assert!(order.preserves(&[a0], &[b0], f64::from(ac), f64::from(bc)));
                            for length in 0..=4 {
                                for word in 0..(1usize << length) {
                                    let (mut a, mut b) = (a0, b0);
                                    let (mut ca, mut cb) = (f64::from(ac), f64::from(bc));
                                    // Independent physical recurrence; this
                                    // does not call NumericOperation::apply.
                                    for step in 0..length {
                                        let arc = (word >> step) & 1;
                                        a += increments[arc];
                                        b += increments[arc];
                                        ca += a * weights[arc];
                                        cb += b * weights[arc];
                                        assert!(ca.is_finite() && cb.is_finite() && ca <= cb);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn sink_bound_lets_a_larger_state_pay_for_every_short_completion() {
        let (increments, weights) = ([1., 2.], [1., 3.]);
        let bounded = |sink_upper| {
            NumericCostOrder::new(
                &[CoupledResource::CumulativeCost {
                    initial: 0.,
                    increments: increments.to_vec(),
                    weights: weights.to_vec(),
                    sink_upper,
                    source_upper: None,
                }],
                &[Some(0)],
                1,
            )
        };
        // At most four future arcs, each weighing at most 3.
        let order = bounded(Some(12.));
        assert!(order.preserves(&[3.], &[0.], 0., 36.));
        assert!(!order.preserves(&[3.], &[0.], 0., 35.));
        assert!(!bounded(None).preserves(&[3.], &[0.], 0., 36.));
        let mut accepted = 0;
        for a0 in 0..=3 {
            for b0 in 0..=3 {
                for ac in 0..=40 {
                    for bc in ac..=40 {
                        let (a0, b0) = (f64::from(a0), f64::from(b0));
                        if !order.preserves(&[a0], &[b0], f64::from(ac), f64::from(bc)) {
                            continue;
                        }
                        accepted += usize::from(a0 > b0);
                        for length in 0..=4 {
                            for word in 0..(1usize << length) {
                                let (mut a, mut b) = (a0, b0);
                                let (mut ca, mut cb) = (f64::from(ac), f64::from(bc));
                                for step in 0..length {
                                    let arc = (word >> step) & 1;
                                    a += increments[arc];
                                    b += increments[arc];
                                    ca += a * weights[arc];
                                    cb += b * weights[arc];
                                }
                                assert!(ca <= cb, "{a0} {b0} {ac} {bc} {word}");
                            }
                        }
                    }
                }
            }
        }
        assert!(accepted > 0);
    }
    #[test]
    fn signed_shared_charges_require_identical_cost_inputs_and_prefix_costs() {
        let definitions = [
            cumulative(vec![0., 1.], vec![-1., 1.]),
            cumulative(vec![0., 0.], vec![0., 0.]),
        ];
        let order = NumericCostOrder::new(&definitions, &[Some(0), Some(1)], 2);
        assert!(order.preserves(&[2., -7.], &[2., 9.], 3., 3.));
        assert!(!order.preserves(&[1., -7.], &[2., 9.], 3., 3.));
        assert!(!order.preserves(&[2., -7.], &[2., 9.], 2., 3.));
        for length in 0..=5 {
            for word in 0..(1usize << length) {
                let (mut a, mut b) = (2., 2.);
                let (mut ca, mut cb) = (3., 3.);
                for step in 0..length {
                    let arc = (word >> step) & 1;
                    a += [0., 1.][arc];
                    b += [0., 1.][arc];
                    ca += a * [-1., 1.][arc];
                    cb += b * [-1., 1.][arc];
                    assert_eq!(ca, cb);
                }
            }
        }
    }
    #[test]
    fn large_finite_products_and_zero_weight_state_updates_keep_overflow_guards() {
        let order = NumericCostOrder::new(&[cumulative(vec![0.], vec![f64::MAX])], &[Some(0)], 1);
        // The wrong-sign dominator product overflows while B stays finite.
        let a: f64 = -2.;
        let b: f64 = 0.;
        assert!(!(a * f64::MAX).is_finite());
        assert!((b * f64::MAX).is_finite());
        assert!(!order.preserves(&[a], &[b], 0., 0.));
        assert!(order.preserves(&[0.], &[0.5], 0., 0.));
        let zero_weight =
            NumericCostOrder::new(&[cumulative(vec![f64::MAX], vec![0.])], &[Some(0)], 1);
        let a = f64::MAX;
        assert!(!(a + f64::MAX).is_finite());
        assert!((0. + f64::MAX).is_finite());
        assert!(!zero_weight.preserves(&[a], &[0.], 0., 0.));
        assert!(zero_weight.preserves(&[0.], &[a], 0., 0.));
    }
    #[test]
    fn signed_zero_and_reachable_nonnegative_prefix_policy_are_explicit() {
        let order =
            NumericCostOrder::new(&[cumulative(vec![-0., 1.], vec![1., 2.])], &[Some(0)], 1);
        assert!(order.preserves(&[-0.], &[0.], -0., 0.));
        assert!(order.preserves(&[0.], &[f64::from_bits(1)], 0., f64::from_bits(1)));
        assert!(!order.preserves(&[0.], &[1.], -1., 0.));
        let signed = NumericCostOrder::new(&[cumulative(vec![0.], vec![-1.])], &[Some(0)], 1);
        // Prefix equality prevents a shared negative charge from overflowing
        // only the putative dominator's already very negative cost.
        assert!(!signed.preserves(&[2.], &[2.], -f64::MAX, 0.));
    }
}
