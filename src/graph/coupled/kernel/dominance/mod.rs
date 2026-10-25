//! Directional resource order and conservative cumulative cost compensation.
use super::bounds::Range;
use super::*;
use crate::core::rounding::add_up;

mod numeric;
pub(super) use numeric::NumericCostOrder;

/// `tolerances` admits the same per-resource slack as the capacity checks;
/// missing entries compare exactly.
pub(super) fn loads_dominate(
    a: &CoupledLabelState,
    b: &CoupledLabelState,
    tolerances: &[f64],
) -> bool {
    if a.loads.len() != b.loads.len() || a.suffix.is_some() != b.suffix.is_some() {
        return false;
    }
    if a.reversal_peaks.is_some() != b.reversal_peaks.is_some() {
        return false;
    }
    let backward = a.suffix.is_some();
    let reversed = a.reversed_use || b.reversed_use;
    a.loads
        .iter()
        .zip(&b.loads)
        .enumerate()
        .all(|(index, (la, lb))| {
            let tolerance = tolerances.get(index).copied().unwrap_or(0.);
            let fits = |x: f64, y: f64| x <= y + tolerance;
            // A suffix is prepended to and joined only through its deliveries
            // D and peak D+M; a prefix is appended to and joined only through
            // its pickups D+N and peak D+M.
            let peak = fits(la[0] + la[2], lb[0] + lb[2]);
            if backward {
                return fits(la[0], lb[0]) && peak;
            }
            let prefix = fits(la[0] + la[1], lb[0] + lb[1]) && peak;
            if !reversed {
                return prefix;
            }
            // A reversed prefix acts as a suffix whose peak is its reversal
            // peak when mixed-sign states retain one, otherwise M.
            let (ra, rb) = match (&a.reversal_peaks, &b.reversal_peaks) {
                (Some(pa), Some(pb)) => (pa[index], pb[index]),
                _ => (la[2], lb[2]),
            };
            prefix && fits(la[0], lb[0]) && fits(la[0] + ra, lb[0] + rb)
        })
}

fn weaker_bounds(a: &Suffix, b: &Suffix) -> bool {
    a.bounds.iter().all(|a| {
        b.bounds.iter().any(|b| {
            a.coefficients == b.coefficients
                && a.offset == b.offset
                && a.bounds.lower <= b.bounds.lower
                && a.bounds.upper >= b.bounds.upper
        })
    })
}
impl CoupledKernel {
    /// A nonnegative upper bound on the additional continuation cost of a
    /// relative to b. None means resource feasibility/order is not established.
    /// Nonnegative penalties preserve the existing hot-key scan ordering.
    pub fn dominance_penalty(
        &self,
        a: &CoupledLabelState,
        b: &CoupledLabelState,
        forward: bool,
    ) -> Option<f64> {
        self.dominance_penalty_with_frozen_fronts(a, b, forward, false)
    }

    /// `frozen_fronts` is a caller-established forward-only proof: every
    /// reachable future interval-resource transition has zero increment and
    /// zero scaled charge, and preserves all of a's current front domains.
    /// Consequently every retained minimum and every future min-delta remain
    /// unchanged. This location-specific proof must not be inferred from a
    /// current normalized-function difference or a rounded reconstructed key.
    /// The caller must additionally establish that the current comparison keys
    /// preserve the canonical physical-prefix cost order. Freezing future
    /// operations cannot repair roundoff already accumulated in earlier deltas.
    pub(super) fn dominance_penalty_with_frozen_fronts(
        &self,
        a: &CoupledLabelState,
        b: &CoupledLabelState,
        forward: bool,
        frozen_fronts: bool,
    ) -> Option<f64> {
        // Raw front operator certificates are not established. Retain every
        // candidate rather than spending a rounded scalar-key margin.
        if self.has_slope_fronts() {
            return None;
        }
        // Reject cheap direction/discrete/component mismatches before
        // traversing paired front pieces or validating retained suffix rows.
        if a.suffix.is_none() != forward
            || b.suffix.is_none() != forward
            || !loads_dominate(a, b, &self.load_tolerances)
            || !self.budgets_dominate(&a.budgets, &b.budgets, forward)
            || a.selection
                .iter()
                .zip(&b.selection)
                .any(|(a, b)| a & b != *b)
        {
            return None;
        }
        // Compare every active resource-cost family's independent scalar and
        // resource components before the hot mixed key. Different prefix
        // histories can otherwise round to the same key while their physical
        // scalar sums differ, even when the final front objects are identical.
        let charged = self.has_costs();
        // Delay the per-slot proof until cheap scalar/component checks pass.
        let numeric_order = || {
            !charged
                // Baseline contract: identical numeric inputs replay the same
                // future operations. Common finite additions preserve prefix
                // cost order; unsupported overflow aborts pricing globally.
                || a.numeric == b.numeric
                || (forward && self.numeric_cost_order.deploys_unequal()
                    && self.numeric_cost_order.preserves(
                        &a.numeric, &b.numeric, a.numeric_cost, b.numeric_cost,
                    ))
        };
        let tracked_order =
            if !charged && self.front_indices.is_empty() && self.relation_indices.is_empty() {
                false
            } else {
                match (a.accounting, b.accounting) {
                    (Some(a_cost), Some(b_cost)) => {
                        if a_cost.reduced > b_cost.reduced
                            || a.numeric_cost > b.numeric_cost
                            || !numeric_order()
                            || a.fronts
                                .iter()
                                .zip(&b.fronts)
                                .any(|(a, b)| a.minimum() > b.minimum())
                        {
                            return None;
                        }
                        true
                    }
                    (None, None) => {
                        // Public resource-only states carry no scalar history,
                        // but their already accumulated numeric objective is
                        // still part of the continuation component contract.
                        if (charged && a.numeric_cost > b.numeric_cost) || !numeric_order() {
                            return None;
                        }
                        false
                    }
                    _ => return None,
                }
            };
        let frozen_fronts = frozen_fronts && tracked_order;
        if !forward && a.suffix.as_ref()?.program != b.suffix.as_ref()?.program {
            return None;
        }
        // Retain full validation before any accepted dominance result. A
        // malformed state rejected by an earlier necessary condition also
        // yields None, exactly as failed validation did before reordering.
        self.check_state(a).ok()?;
        self.check_state(b).ok()?;
        let mut penalty = 0.;
        if a.fronts.len() != b.fronts.len() {
            return None;
        }
        for ((a, b), &index) in a.fronts.iter().zip(&b.fronts).zip(&self.front_indices) {
            if a == b {
                continue;
            }
            if a.domain().lower > b.domain().lower || a.domain().upper < b.domain().upper {
                return None;
            }
            if frozen_fronts && forward {
                continue;
            }
            // A bound on the CURRENT normalized difference is not a bound on
            // future rounded additions. Even gap 1 can become gap 2 after a
            // common charge at 2^53. Preserve general unequal costly fronts.
            // An all-zero front whose every future charge is suppressed/zero
            // is a safe broader subset: all subsequent minima/deltas stay 0,
            // including after clipping or translating its feasible domain.
            let CoupledResource::IntervalCost { functions, .. } = &self.definitions[index] else {
                unreachable!()
            };
            let zero_charges = self.cost_scale == 0.
                || functions
                    .iter()
                    .all(|function| function.points.iter().all(|p| p[1] == 0.));
            if !zero_charges || a.points().iter().chain(b.points()).any(|p| p[1] != 0.) {
                return None;
            }
        }
        // Deep paired equality is deliberately last among these necessary
        // preconditions: rejected scalar/discrete/program pairs need no walk
        // through the retained function and backward-adjoint pieces.
        if a.relations != b.relations || a.charging != b.charging {
            return None;
        }
        if !forward {
            let a = a.suffix.as_ref()?;
            let b = b.suffix.as_ref()?;
            if a.bounds != b.bounds && !weaker_bounds(a, b) {
                return None;
            }
            return add_up(penalty, self.backward_penalty(a, b)?);
        }
        if !self.optimization.dependency_order.preserves(
            &a.numeric,
            &b.numeric,
            &self.optimization.numeric,
        ) {
            return None;
        }
        if charged {
            // The prepared per-operation proof above establishes literal
            // numeric component order. No current key gap or positive
            // compensation participates in charged-resource dominance.
            return Some(0.);
        }
        for (i, (&a, &b)) in a.numeric.iter().zip(&b.numeric).enumerate() {
            if a == b {
                continue;
            }
            if self.cost_scale == 0. {
                continue;
            }
            let weights = self.optimization.weights[i];
            // These orders are preserved even by rounded arithmetic because
            // the same additive numeric transition is applied to both labels.
            if (weights.lo >= 0. && a <= b) || (weights.hi <= 0. && a >= b) {
                continue;
            }
            // Cost tradeoffs use finite walk-weight bounds only where all
            // cumulative state/product sums remain exact binary64 integers.
            // Other signed/unbounded domains retain the conservative rule.
            if !self.optimization.exact_costs {
                return None;
            }
            let domain = self.optimization.numeric[i];
            if a.fract() != 0.
                || b.fract() != 0.
                || a < domain.lo
                || a > domain.hi
                || b < domain.lo
                || b > domain.hi
            {
                return None;
            }
            let gap = Range::point(a).subtract(Range::point(b)).multiply(weights);
            if !gap.hi.is_finite() {
                return None;
            }
            penalty = add_up(penalty, gap.hi.max(0.))?;
        }
        Some(penalty)
    }
}
impl<const D: usize> Graph<D> {
    pub(crate) fn coupled_dominance_penalty(
        &self,
        a: Option<&CoupledLabelState>,
        b: Option<&CoupledLabelState>,
        forward: bool,
        vertex: usize,
    ) -> Option<f64> {
        match (self.coupled_kernel(), a, b) {
            (None, None, None) => Some(0.),
            (Some(kernel), Some(a), Some(b)) => {
                let frozen = forward
                    && a.fronts.iter().enumerate().all(|(slot, front)| {
                        kernel
                            .frozen
                            .contains(slot, vertex, front.domain(), kernel.cost_scale)
                    });
                kernel.dominance_penalty_with_frozen_fronts(a, b, forward, frozen)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod front_continuation_tests {
    use super::*;
    use crate::graph::coupled::PiecewiseLinearFunction;

    #[test]
    fn rounded_future_charge_cannot_prune_the_better_physical_route() {
        let large = 9_007_199_254_740_994.;
        let mut graph = Graph::<2>::new(4, 0).unwrap();
        for (tail, head, cost) in [(0, 1, -1.), (0, 1, 0.), (1, 2, -large + 2.), (2, 3, -10.)] {
            graph.add_arc(tail, head, cost).unwrap();
        }
        let function = |points| PiecewiseLinearFunction::new(points).unwrap();
        graph
            .set_coupled_resources(&[CoupledResource::IntervalCost {
                initial: Window::new(0., 1.).unwrap(),
                bounds: Window::new(0., 1.).unwrap(),
                increments: vec![0.; 4],
                functions: vec![
                    function(vec![[0., 0.], [1., 1.]]),
                    function(vec![[0., 0.], [1., 0.]]),
                    function(vec![[1., large], [2., large]]),
                    function(vec![[0., 0.], [1., 0.]]),
                ],
            }])
            .unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        let root = kernel.root(true);
        let (a, _) = kernel.extend(&root, 0, true).unwrap().unwrap();
        let (b, _) = kernel.extend(&root, 1, true).unwrap().unwrap();
        // A snapshot difference of one grows to two after common addition.
        assert_eq!(kernel.dominance_penalty(&a, &b, true), None);
        let (_, ad) = kernel.extend(&a, 2, true).unwrap().unwrap();
        let (_, bd) = kernel.extend(&b, 2, true).unwrap().unwrap();
        assert_eq!(ad - bd, 2.);
        // The scalar sums are independently evaluated in physical arc order.
        // Resource values below are literal endpoint sums, not front replay.
        let scalar_a = ((0. - 1.) + (-large + 2.)) - 10.;
        let scalar_b = ((0. + 0.) + (-large + 2.)) - 10.;
        assert_eq!(scalar_a + (1. + large), -6.);
        assert_eq!(scalar_b + (0. + large), -8.);
        for direction in [0., 1., 2.] {
            let mut solver = crate::pricing::Solver::<2>::new(4, 0).unwrap();
            solver.graph = graph.clone();
            solver
                .set_option("BALDESuseBidirectionalSearch", direction)
                .unwrap();
            let result = solver.price_phase(0, 100, false).unwrap();
            assert_eq!(result.minimum_reduced_cost, -8.);
            assert!(result
                .columns
                .iter()
                .any(|r| r.arcs == [1, 2, 3] && r.original_cost == -8.));
        }
    }

    #[test]
    fn frozen_future_does_not_repair_an_already_rounded_prefix_key() {
        let large = 9_007_199_254_740_994.;
        let mut graph = Graph::<2>::new(4, 0).unwrap();
        for (tail, head, cost) in [(0, 1, -large), (1, 2, 0.), (0, 2, 3.), (2, 3, -10.)] {
            graph.add_arc(tail, head, cost).unwrap();
        }
        let constant =
            |value| PiecewiseLinearFunction::new(vec![[0., value], [1., value]]).unwrap();
        graph
            .set_coupled_resources(&[CoupledResource::IntervalCost {
                initial: Window::new(0., 1.).unwrap(),
                bounds: Window::new(0., 1.).unwrap(),
                increments: vec![0.; 4],
                functions: vec![constant(1.), constant(large), constant(0.), constant(0.)],
            }])
            .unwrap();
        // Prefix A's interleaved scalar/min-delta key is four, but its
        // physical scalar sum plus final resource minimum is two. B's key
        // and canonical cost are three. Zero future charges cannot repair this
        // discrepancy before choosing which complete route to retain.
        assert_eq!((-large + 1.) + ((1. + large) - 1.), 4.);
        assert_eq!(-large + (1. + large), 2.);
        for direction in [0., 1., 2.] {
            let mut solver = crate::pricing::Solver::<2>::new(4, 0).unwrap();
            solver.graph = graph.clone();
            solver
                .set_option("BALDESuseBidirectionalSearch", direction)
                .unwrap();
            let result = solver.price_phase(0, 100, false).unwrap();
            assert_eq!(result.minimum_reduced_cost, -8.);
            assert!(result
                .columns
                .iter()
                .any(|r| r.arcs == [0, 1, 3] && r.original_cost == -8.));
        }
    }

    #[test]
    fn equal_final_fronts_do_not_hide_different_physical_scalar_prefixes() {
        let large = 9_007_199_254_740_994.;
        let mut graph = Graph::<2>::new(5, 0).unwrap();
        for (tail, head, cost) in [
            (0, 1, -large),
            (1, 3, 0.),
            (0, 2, -(large + 2.)),
            (2, 3, 3.),
            (3, 4, -10.),
        ] {
            graph.add_arc(tail, head, cost).unwrap();
        }
        let constant =
            |value| PiecewiseLinearFunction::new(vec![[0., value], [1., value]]).unwrap();
        graph
            .set_coupled_resources(&[CoupledResource::IntervalCost {
                initial: Window::new(0., 1.).unwrap(),
                bounds: Window::new(0., 1.).unwrap(),
                increments: vec![0.; 5],
                functions: vec![
                    constant(1.),
                    constant(large),
                    constant(large + 2.),
                    constant(0.),
                    constant(0.),
                ],
            }])
            .unwrap();
        // Both final resource functions are the same constant large+2.
        // Independent physical scalar sums distinguish the complete costs.
        assert_eq!((-large - 10.) + (1. + large), -8.);
        assert_eq!(((-(large + 2.) + 3.) - 10.) + (large + 2.), -6.);
        for direction in [0., 1., 2.] {
            let mut solver = crate::pricing::Solver::<2>::new(5, 0).unwrap();
            solver.graph = graph.clone();
            solver
                .set_option("BALDESuseBidirectionalSearch", direction)
                .unwrap();
            let result = solver.price_phase(0, 100, false).unwrap();
            assert_eq!(result.minimum_reduced_cost, -8.);
            assert!(result
                .columns
                .iter()
                .any(|r| r.arcs == [0, 1, 4] && r.original_cost == -8.));
        }
    }

    #[test]
    fn numeric_only_piecewise_costs_preserve_independent_scalar_components() {
        let large = 9_007_199_254_740_994.;
        let mut graph = Graph::<2>::new(4, 0).unwrap();
        for (tail, head, cost) in [(0, 1, -1.), (0, 1, 0.), (1, 2, -large + 2.), (2, 3, -10.)] {
            graph.add_arc(tail, head, cost).unwrap();
        }
        let constant =
            |value| PiecewiseLinearFunction::new(vec![[-1., value], [1., value]]).unwrap();
        graph
            .set_coupled_resources(&[
                CoupledResource::CumulativeCost {
                    initial: 0.,
                    increments: vec![0.; 4],
                    weights: vec![0.; 4],
                },
                CoupledResource::PiecewiseLinearCost {
                    parent: 0,
                    functions: vec![constant(1.), constant(0.), constant(large), constant(0.)],
                },
            ])
            .unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        let root = kernel.root(true);
        let (a, _) = kernel.extend(&root, 0, true).unwrap().unwrap();
        let (b, _) = kernel.extend(&root, 1, true).unwrap().unwrap();
        assert_eq!(a.numeric, b.numeric);
        assert!(!CoupledKernel::dominates(Some(&a), Some(&b)));
        assert_eq!(kernel.dominance_penalty(&a, &b, true), None);
        // Independent scalar sum plus literal numeric charge accumulation.
        let scalar_a = ((0. - 1.) + (-large + 2.)) - 10.;
        let scalar_b = (0. + (-large + 2.)) - 10.;
        assert_eq!(scalar_a + (1. + large), -6.);
        assert_eq!(scalar_b + large, -8.);
        for direction in [0., 1., 2.] {
            let mut solver = crate::pricing::Solver::<2>::new(4, 0).unwrap();
            solver.graph = graph.clone();
            solver
                .set_option("BALDESuseBidirectionalSearch", direction)
                .unwrap();
            let result = solver.price_phase(0, 100, false).unwrap();
            assert_eq!(result.minimum_reduced_cost, -8.);
            assert!(result
                .columns
                .iter()
                .any(|r| r.arcs == [1, 2, 3] && r.original_cost == -8.));
        }
    }

    #[test]
    fn nonnegative_numeric_continuations_restore_zero_compensation_state_dominance() {
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        for (tail, head) in [(0, 1), (0, 1), (1, 2)] {
            graph.add_arc(tail, head, -20.).unwrap();
        }
        graph
            .set_coupled_resources(&[CoupledResource::CumulativeCost {
                initial: 0.,
                increments: vec![1., 2., 1.],
                weights: vec![1., 1., 3.],
            }])
            .unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        let root = kernel.root(true);
        let (a, _) = kernel.extend(&root, 0, true).unwrap().unwrap();
        let (b, _) = kernel.extend(&root, 1, true).unwrap().unwrap();
        assert_ne!(a.numeric, b.numeric);
        assert_eq!(kernel.dominance_penalty(&a, &b, true), Some(0.));
        assert_eq!(kernel.dominance_penalty(&b, &a, true), None);
        // Independent resource recurrences: route A pays 1 + 2*3 = 7;
        // route B pays 2 + 3*3 = 11. Physical scalar sums are both -40.
        for direction in [0., 1., 2.] {
            let mut solver = crate::pricing::Solver::<2>::new(3, 0).unwrap();
            solver.graph = graph.clone();
            solver
                .set_option("BALDESuseBidirectionalSearch", direction)
                .unwrap();
            let result = solver.price_phase(0, 100, false).unwrap();
            assert_eq!(result.minimum_reduced_cost, -33.);
            assert!(result
                .columns
                .iter()
                .any(|r| r.arcs == [0, 2] && r.original_cost == -33.));
        }
    }

    #[test]
    fn identical_numeric_inputs_keep_finite_signed_and_nonnegative_prefix_order() {
        for [a_charge, b_charge, future] in [[1., 2., 3.], [-3., -1., -2.]] {
            let mut graph = Graph::<2>::new(3, 0).unwrap();
            for (tail, head) in [(0, 1), (0, 1), (1, 2)] {
                graph.add_arc(tail, head, 0.).unwrap();
            }
            let constant =
                |value| PiecewiseLinearFunction::new(vec![[-1., value], [1., value]]).unwrap();
            graph
                .set_coupled_resources(&[
                    CoupledResource::CumulativeCost {
                        initial: 0.,
                        increments: vec![0.; 3],
                        weights: vec![0.; 3],
                    },
                    CoupledResource::PiecewiseLinearCost {
                        parent: 0,
                        functions: vec![constant(a_charge), constant(b_charge), constant(future)],
                    },
                ])
                .unwrap();
            let kernel = graph.coupled_kernel().unwrap();
            let root = kernel.root(true);
            let (a, _) = kernel.extend(&root, 0, true).unwrap().unwrap();
            let (b, _) = kernel.extend(&root, 1, true).unwrap().unwrap();
            assert_eq!(a.numeric, b.numeric);
            assert_eq!(kernel.dominance_penalty(&a, &b, true), Some(0.));
            assert_eq!(kernel.dominance_penalty(&b, &a, true), None);
            // Independent literal objective recurrence on each physical path.
            assert_eq!(kernel.replay(&[0, 2]).unwrap(), Some(a_charge + future));
            assert_eq!(kernel.replay(&[1, 2]).unwrap(), Some(b_charge + future));
        }
    }

    #[test]
    fn identical_numeric_signed_costs_preserve_baseline_and_overflow_is_global_unknown() {
        let maximum = f64::MAX;
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        for (tail, head, cost) in [(0, 1, -1.), (0, 1, 0.), (1, 2, 0.)] {
            graph.add_arc(tail, head, cost).unwrap();
        }
        let constant =
            |value| PiecewiseLinearFunction::new(vec![[-1., value], [1., value]]).unwrap();
        graph
            .set_coupled_resources(&[
                CoupledResource::CumulativeCost {
                    initial: 0.,
                    increments: vec![0.; 3],
                    weights: vec![0.; 3],
                },
                CoupledResource::PiecewiseLinearCost {
                    parent: 0,
                    functions: vec![
                        constant(-maximum),
                        constant(-maximum / 2.),
                        constant(-maximum / 2.),
                    ],
                },
            ])
            .unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        let root = kernel.root(true);
        let (a, _) = kernel.extend(&root, 0, true).unwrap().unwrap();
        let (b, _) = kernel.extend(&root, 1, true).unwrap().unwrap();
        assert_eq!(a.numeric, b.numeric);
        assert!(a.numeric_cost < b.numeric_cost);
        assert_eq!(kernel.dominance_penalty(&a, &b, true), Some(0.));
        assert!(kernel.extend(&a, 2, true).is_err());
        assert_eq!(kernel.replay(&[1, 2]).unwrap(), Some(-maximum));
        for direction in [0., 1., 2.] {
            let mut solver = crate::pricing::Solver::<2>::new(3, 0).unwrap();
            solver.graph = graph.clone();
            solver
                .set_option("BALDESuseBidirectionalSearch", direction)
                .unwrap();
            if let Ok(result) = solver.price_phase(0, 100, false) {
                // A scheduler that never explores the unsupported branch can
                // return the valid finite best route. It cannot silently
                // turn overflow into empty/complete infeasibility evidence.
                assert_eq!(result.minimum_reduced_cost, -maximum);
                assert!(!result.columns.is_empty());
                assert!(result
                    .columns
                    .iter()
                    .all(|r| r.arcs == [1, 2] && r.original_cost == -maximum));
            }
        }
    }

    #[test]
    fn complex_numeric_programs_keep_baseline_identity_activation_policy() {
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        for (tail, head) in [(0, 1), (0, 1), (1, 2)] {
            graph.add_arc(tail, head, -20.).unwrap();
        }
        graph
            .set_coupled_resources(&[
                CoupledResource::CumulativeCost {
                    initial: 0.,
                    increments: vec![1., 2., 1.],
                    weights: vec![1., 1., 3.],
                },
                CoupledResource::Dependent {
                    parent: 0,
                    initial: 0.,
                    bounds: Window::new(0., 100.).unwrap(),
                    offsets: vec![0.; 3],
                    factors: vec![1.; 3],
                },
            ])
            .unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        let root = kernel.root(true);
        let (a, _) = kernel.extend(&root, 0, true).unwrap().unwrap();
        let (b, _) = kernel.extend(&root, 1, true).unwrap().unwrap();
        // The proof can establish the arithmetic order; deployment keeps
        // richer programs on their measured baseline identity path.
        assert!(kernel.numeric_cost_order.preserves(
            &a.numeric,
            &b.numeric,
            a.numeric_cost,
            b.numeric_cost
        ));
        assert!(!kernel.numeric_cost_order.deploys_unequal());
        assert_eq!(kernel.dominance_penalty(&a, &b, true), None);
        assert_eq!(kernel.dominance_penalty(&a, &a, true), Some(0.));
    }

    #[test]
    fn permanently_zero_fronts_keep_domain_inclusion_dominance() {
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        for (tail, head) in [(0, 1), (0, 1), (1, 2)] {
            graph.add_arc(tail, head, 0.).unwrap();
        }
        let zero = |lo, hi| PiecewiseLinearFunction::new(vec![[lo, 0.], [hi, 0.]]).unwrap();
        graph
            .set_coupled_resources(&[CoupledResource::IntervalCost {
                initial: Window::new(0., 4.).unwrap(),
                bounds: Window::new(0., 4.).unwrap(),
                increments: vec![0.; 3],
                functions: vec![zero(0., 4.), zero(1., 3.), zero(0., 4.)],
            }])
            .unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        let root = kernel.root(true);
        let (a, _) = kernel.extend(&root, 0, true).unwrap().unwrap();
        let (b, _) = kernel.extend(&root, 1, true).unwrap().unwrap();
        assert_eq!(kernel.dominance_penalty(&a, &b, true), Some(0.));
        assert_eq!(kernel.dominance_penalty(&b, &a, true), None);
    }
}

#[cfg(test)]
mod load_dominance_tests {
    use super::*;

    fn kernel_graph() -> Graph<2> {
        let mut graph = Graph::new(4, 1).unwrap();
        for (u, v) in [(0, 1), (0, 2), (1, 3), (2, 3)] {
            graph.add_arc(u, v, 0.).unwrap();
        }
        graph
            .set_coupled_resources(&[CoupledResource::SimultaneousPickupDelivery {
                capacity: 20.,
                deliveries: vec![5., 0., 0., 0.],
                pickups: vec![0., 6., 0., 0.],
                tolerance: 0.,
            }])
            .unwrap();
        graph
    }

    #[test]
    fn prefixes_compare_only_pickups_and_peak() {
        let graph = kernel_graph();
        let kernel = graph.coupled_kernel().unwrap();
        let root = kernel.root(true);
        let deliver = kernel.extend(&root, 0, true).unwrap().unwrap().0;
        let pickup = kernel.extend(&root, 1, true).unwrap().unwrap().0;
        // (D,N,M) = (5,-5,0) vs (0,6,6): fewer pickups and a lower peak.
        assert!(CoupledKernel::dominates(Some(&deliver), Some(&pickup)));
        assert!(!CoupledKernel::dominates(Some(&pickup), Some(&deliver)));
    }

    #[test]
    fn reversed_prefixes_also_compare_their_suffix_role() {
        let graph = kernel_graph();
        let kernel = graph.coupled_kernel().unwrap();
        let mut workspace = CoupledWorkspace::default();
        let root = kernel.reversal_root(true);
        let mut extend = |arc| {
            kernel
                .extend_reversal_with_workspace(&root, arc, true, &mut workspace)
                .unwrap()
                .unwrap()
                .0
        };
        let deliver = extend(0);
        let pickup = extend(1);
        // As a reversed suffix the delivering prefix carries more deliveries.
        assert!(!CoupledKernel::dominates(Some(&deliver), Some(&pickup)));
    }
}

#[cfg(test)]
mod load_tolerance_tests {
    use super::*;

    fn kernel_graph(tolerance: f64) -> Graph<2> {
        let mut graph = Graph::new(3, 1).unwrap();
        graph.add_arc(0, 1, 0.).unwrap();
        graph.add_arc(1, 2, 0.).unwrap();
        graph
            .set_coupled_resources(&[CoupledResource::SimultaneousPickupDelivery {
                capacity: 1.,
                deliveries: vec![0., 0.],
                pickups: vec![0.6, 0.4000005],
                tolerance,
            }])
            .unwrap();
        graph
    }

    #[test]
    fn capacity_admits_exactly_the_configured_tolerance() {
        let graph = kernel_graph(1e-6);
        let kernel = graph.coupled_kernel().unwrap();
        assert!(kernel.replay_state(&[0, 1]).unwrap().is_some());
        let graph = kernel_graph(0.);
        let kernel = graph.coupled_kernel().unwrap();
        assert!(kernel.replay_state(&[0, 1]).unwrap().is_none());
    }

    #[test]
    fn dominance_uses_the_resource_tolerance() {
        let a = [[0.5000005, 0.5000005, 0.5000005]];
        let b = [[0.5, 0.5, 0.5]];
        for (tolerance, expected) in [(1e-6, true), (0., false)] {
            let graph = kernel_graph(tolerance);
            let kernel = graph.coupled_kernel().unwrap();
            let mut x = kernel.root(true);
            let mut y = kernel.root(true);
            x.loads = StateValues::from_slice(&a);
            y.loads = StateValues::from_slice(&b);
            assert_eq!(loads_dominate(&x, &y, &kernel.load_tolerances), expected);
        }
    }
}
