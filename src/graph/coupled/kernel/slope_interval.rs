//! Independent raw slope-front intervals with ordered suffix-operator replay.
use super::*;
use descriptor::{FrontierBudget, FrontierOutcome, SlopeFront};

#[derive(Clone, Debug)]
pub(super) struct SlopeLabel {
    pub(super) front: Option<SlopeFront>,
    pub(super) minimum: f64,
    pub(super) upper: Option<relation::RelationFront>,
}
impl PartialEq for SlopeLabel {
    fn eq(&self, other: &Self) -> bool {
        if self.minimum.to_bits() != other.minimum.to_bits() || self.upper != other.upper {
            return false;
        }
        match (&self.front, &other.front) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                a.used_tolerant_alignment() == b.used_tolerant_alignment()
                    && a.components().len() == b.components().len()
                    && a.components().iter().zip(b.components()).all(|(a, b)| {
                        a.pairs().len() == b.pairs().len()
                            && a.pairs().iter().zip(b.pairs()).all(|(a, b)| {
                                a.0.to_bits() == b.0.to_bits() && a.1.to_bits() == b.1.to_bits()
                            })
                    })
            }
            _ => false,
        }
    }
}
/// Shared leaf fronts prepared once per arc, including phase-I domain-only
/// fronts. The zero objective field denotes unexecuted metadata, not its value.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct PreparedSlopeCharge {
    normal: SlopeLabel,
    suppressed: SlopeLabel,
}
impl PreparedSlopeCharge {
    pub(super) fn new(function: &PreparedSlopeFunction) -> Result<Self> {
        let normal = SlopeFront::from_prepared(function.clone());
        let zeros = function
            .pairs()
            .iter()
            .map(|&(_, coordinate)| (0., coordinate))
            .collect();
        let zero = PreparedSlopeFunction::from_prepared_pairs(zeros)
            .map_err(|error| invalid(error.to_string()))?;
        let suppressed = SlopeFront::from_prepared(zero);
        Ok(Self {
            normal: SlopeLabel::root(&normal, true),
            suppressed: SlopeLabel::root(&suppressed, true),
        })
    }
    fn front(&self, scale: f64) -> Result<&SlopeFront> {
        let label = if scale == 0. {
            &self.suppressed
        } else if scale == 1. {
            &self.normal
        } else {
            return Err(crate::incomplete(
                "unsupported raw slope-front objective scale",
            ));
        };
        Ok(label.front.as_ref().expect("prepared charge front"))
    }
}
/// Per-resource facts behind the domain-only dominance rule. Unequal costly
/// fronts never compare: a pointwise order is not preserved by future
/// operations, because translation re-rounds every segment span and addition
/// re-accumulates split segments, so an earlier gap can reverse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct PreparedSlopeDominance {
    strict: bool,
    zero_charges: bool,
}
impl PreparedSlopeDominance {
    pub(super) fn new(definition: &CoupledResource) -> Self {
        let CoupledResource::SlopeIntervalCost {
            functions, policy, ..
        } = definition
        else {
            unreachable!()
        };
        Self {
            strict: *policy == FrontOperationPolicy::StrictDomain,
            zero_charges: functions.iter().all(zero_valued),
        }
    }
    /// Forward a dominates b when both fronts are identically zero, every
    /// future charge is zero (or suppressed) and each component of b lies in
    /// one component of a. Zero values survive every operation exactly, and
    /// literal translation, restriction and strict-domain intersection are
    /// monotone in each coordinate, so a stays feasible wherever b does with
    /// the same zero minimum. Tolerant native alignment can snap a and b to
    /// different boundaries and is excluded.
    pub(super) fn dominates(&self, a: &SlopeLabel, b: &SlopeLabel, scale: f64) -> bool {
        let (Some(a), Some(b)) = (&a.front, &b.front) else {
            return false;
        };
        self.strict
            && (scale == 0. || self.zero_charges)
            && a.components().iter().chain(b.components()).all(zero_valued)
            && b.components().iter().all(|b| {
                let (lo, hi) = b.domain();
                a.components().iter().any(|a| {
                    let (start, end) = a.domain();
                    start <= lo && hi <= end
                })
            })
    }
}
fn zero_valued(function: &PreparedSlopeFunction) -> bool {
    function.pairs().iter().all(|pair| pair.0 == 0.)
}
pub(super) fn outcome<T>(value: FrontierOutcome<T>) -> Result<Option<T>> {
    match value {
        FrontierOutcome::Complete(value) => Ok(Some(value)),
        FrontierOutcome::Empty => Ok(None),
        FrontierOutcome::Unknown(reason) => Err(crate::incomplete(format!(
            "raw slope-front operation is unknown: {reason:?}"
        ))),
    }
}
impl SlopeLabel {
    pub(super) fn root(front: &SlopeFront, forward: bool) -> Self {
        Self {
            front: forward.then(|| front.clone()),
            minimum: 0.,
            upper: None,
        }
    }
    pub(super) fn extend(
        &self,
        definition: &CoupledResource,
        charge: &PreparedSlopeCharge,
        arc: usize,
        forward: bool,
        scale: f64,
    ) -> Result<Option<Self>> {
        if !forward {
            return Ok(Some(self.clone()));
        }
        let CoupledResource::SlopeIntervalCost {
            bounds,
            increments,
            policy,
            ..
        } = definition
        else {
            unreachable!()
        };
        let budget = FrontierBudget::default();
        let old = self
            .front
            .as_ref()
            .ok_or_else(|| invalid("missing forward raw slope front"))?;
        let Some(translated) = outcome(old.translate(increments[arc], budget))? else {
            return Ok(None);
        };
        let Some(restricted) =
            outcome(translated.restrict(bounds.lower, bounds.upper, *policy, budget))?
        else {
            return Ok(None);
        };
        // The phase-I leaf is precomputed with identical full domain and
        // zero values. No raw pair Vec is cloned or revalidated per extension.
        let charge = charge.front(scale)?;
        let Some(front) = outcome(restricted.add(charge, *policy, budget))? else {
            return Ok(None);
        };
        let Some(witness) = outcome(front.minimum(budget))? else {
            return Ok(None);
        };
        Ok(Some(Self {
            front: Some(front),
            minimum: finite(witness.value)?,
            upper: None,
        }))
    }
}
impl CoupledKernel {
    pub(crate) fn has_slope_fronts(&self) -> bool {
        !self.slope_indices.is_empty() || !self.slope_paired.is_empty()
    }
}
impl CoupledLabelState {
    /// Computed raw operator fronts, in slope-interval declaration order.
    /// Backward labels retain an arc program and expose no invented front.
    pub fn slope_cost_fronts(&self) -> impl Iterator<Item = Option<&SlopeFront>> {
        self.slope_fronts.iter().map(|label| label.front.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn constant(value: f64, lo: f64, hi: f64) -> PreparedSlopeFunction {
        descriptor::RawAffine::new(value, 0.)
            .unwrap()
            .prepare(lo, hi)
            .unwrap()
    }
    fn definition(functions: Vec<PreparedSlopeFunction>) -> CoupledResource {
        CoupledResource::SlopeIntervalCost {
            initial: Window::new(0., 2.).unwrap(),
            bounds: Window::new(0., 2.).unwrap(),
            increments: vec![0.; functions.len()],
            functions,
            policy: FrontOperationPolicy::StrictDomain,
        }
    }
    fn negative_suffix_graph() -> Graph<2> {
        let mut graph = Graph::new(4, 1).unwrap();
        for v in 0..4 {
            graph.set_bounds(v, 0, 0., 2.).unwrap();
        }
        for (u, v, cost) in [(0, 1, 0.), (1, 3, 100.), (0, 2, -5.), (2, 3, -5.)] {
            let a = graph.add_arc(u, v, cost).unwrap();
            graph.arc_mut(a).unwrap().consumption[0] = 1.;
        }
        graph
            .set_coupled_resources(&[definition(vec![
                constant(0., 0., 2.),
                constant(-200., 0., 2.),
                constant(0., 0., 2.),
                constant(0., 0., 2.),
            ])])
            .unwrap();
        graph
    }
    #[test]
    fn negative_raw_suffix_is_never_hidden_by_zero_backward_baseline() {
        let graph = negative_suffix_graph();
        let kernel = graph.coupled_kernel().unwrap();
        assert!(kernel.has_costs());
        assert_eq!(kernel.replay(&[0, 1]).unwrap(), Some(-200.));
        for split in 0..=2 {
            let f = kernel.replay_state(&[0, 1][..split]).unwrap().unwrap().0;
            let mut b = kernel.root(false);
            for &arc in [0, 1][split..].iter().rev() {
                b = kernel.extend(&b, arc, false).unwrap().unwrap().0;
            }
            assert!(b.slope_cost_fronts().all(|front| front.is_none()));
            assert_eq!(kernel.join_resource_objective(&f, &b).unwrap(), Some(-200.));
            let mut hull = CoupledStateBounds::default();
            hull.include(Some(&b));
            assert_eq!(
                kernel.completion_correction_lower(true, Some(&f), &hull),
                f64::NEG_INFINITY
            );
            assert!(kernel.completion_feasible(true, Some(&f), &hull));
        }
        for mode in [0., 1., 2.] {
            let mut solver = crate::pricing::Solver::<2>::new(4, 1).unwrap();
            solver.graph = graph.clone();
            solver.enumeration_preserve_arc_paths = true;
            solver
                .set_option("BALDESuseBidirectionalSearch", mode)
                .unwrap();
            let result = solver.price_phase(0, 20, false).unwrap();
            assert_eq!(result.minimum_reduced_cost, -100.);
            let mut costs: Vec<_> = solver
                .enumerate(0.)
                .unwrap()
                .into_iter()
                .map(|route| {
                    assert_eq!(route.cost, route.original_cost);
                    route.cost
                })
                .collect();
            costs.sort_by(f64::total_cmp);
            assert_eq!(costs, vec![-100., -10.]);
            solver.fix(-50.).unwrap();
            assert_eq!(solver.enumerate(-50.).unwrap().len(), 1);
            let mut child = solver.fork_for_node();
            child
                .set_coupled_resources(&[definition(vec![constant(0., 0., 2.); 4])])
                .unwrap();
            assert_eq!(
                child
                    .price_phase(0, 20, false)
                    .unwrap()
                    .minimum_reduced_cost,
                -10.
            );
            assert_eq!(
                solver
                    .price_phase(0, 20, false)
                    .unwrap()
                    .minimum_reduced_cost,
                -100.
            );
        }
    }
    #[test]
    fn raw_interval_cut_pricing_keeps_resource_and_physical_objectives_separate() {
        for mode in [0., 1., 2.] {
            let mut solver = crate::pricing::Solver::<2>::new(4, 1).unwrap();
            solver.graph = negative_suffix_graph();
            solver
                .set_option("BALDESuseBidirectionalSearch", mode)
                .unwrap();
            solver
                .cuts
                .add_strong_k_path_cut(4, &[1, 2], 1, -1.25)
                .unwrap();
            let priced = solver.price_phase(0, 20, false).unwrap();
            assert_eq!(priced.minimum_reduced_cost, -98.75);
            for route in priced.columns {
                if route.cost == -98.75 {
                    assert_eq!(route.original_cost, -100.);
                }
            }
        }
    }
    #[test]
    fn raw_interval_uses_computed_native_front_not_global_source_program_minimum() {
        let functions = vec![
            PreparedSlopeFunction::from_prepared_pairs(vec![(2f64.powi(30), 0.), (1., 1.)])
                .unwrap(),
            constant(-2f64.powi(30), 0., 1.),
            PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (-1., 1.)]).unwrap(),
        ];
        let mut graph = Graph::<2>::new(4, 0).unwrap();
        for (u, v) in [(0, 1), (1, 2), (2, 3)] {
            graph.add_arc(u, v, 0.).unwrap();
        }
        let resource = CoupledResource::SlopeIntervalCost {
            initial: Window::new(0., 1.).unwrap(),
            bounds: Window::new(0., 1.).unwrap(),
            increments: vec![0.; 3],
            functions: functions.clone(),
            policy: FrontOperationPolicy::NativeFrontAlgebra,
        };
        graph.set_coupled_resources(&[resource]).unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        let (state, cost) = kernel.replay_state(&[0, 1, 2]).unwrap().unwrap();
        assert_eq!(cost, 0.);
        let front = state.slope_cost_fronts().next().unwrap().unwrap();
        assert_eq!(
            front
                .minimum(FrontierBudget::default())
                .complete_for_test()
                .value,
            0.
        );
        let x = 2f64.powi(-24);
        let witness = descriptor::SourceReplayWitness::evaluate_shared_argument(
            &functions,
            x,
            EvaluationPolicy::StrictDomain,
        )
        .unwrap()
        .unwrap();
        assert_eq!(witness.value(), -x);
        for split in 0..=3 {
            let f = kernel.replay_state(&[0, 1, 2][..split]).unwrap().unwrap().0;
            let mut b = kernel.root(false);
            for &a in [0, 1, 2][split..].iter().rev() {
                b = kernel.extend(&b, a, false).unwrap().unwrap().0;
            }
            assert_eq!(kernel.join_resource_objective(&f, &b).unwrap(), Some(0.));
        }
    }
    #[test]
    fn raw_interval_translation_restriction_and_singleton_survive_replay() {
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        graph.add_arc(0, 1, 0.).unwrap();
        graph.add_arc(1, 2, 0.).unwrap();
        graph
            .set_coupled_resources(&[CoupledResource::SlopeIntervalCost {
                initial: Window::new(0., 1.).unwrap(),
                bounds: Window::new(0., 3.).unwrap(),
                increments: vec![1., 1.],
                functions: vec![constant(2., 1., 1.), constant(-5., 2., 2.)],
                policy: FrontOperationPolicy::StrictDomain,
            }])
            .unwrap();
        let (state, cost) = graph
            .coupled_kernel()
            .unwrap()
            .replay_state(&[0, 1])
            .unwrap()
            .unwrap();
        assert_eq!(cost, -3.);
        assert_eq!(
            state
                .slope_cost_fronts()
                .next()
                .unwrap()
                .unwrap()
                .components()[0]
                .domain(),
            (2., 2.)
        );
    }
    #[test]
    fn raw_front_unknown_is_an_error_and_proven_empty_is_infeasible() {
        let mut graph = Graph::<2>::new(2, 0).unwrap();
        graph.add_arc(0, 1, 0.).unwrap();
        let large = 9_007_199_254_740_992.;
        graph
            .set_coupled_resources(&[CoupledResource::SlopeIntervalCost {
                initial: Window::new(0., 1.).unwrap(),
                bounds: Window::new(0., large + 2.).unwrap(),
                increments: vec![large],
                functions: vec![constant(0., 0., large + 2.)],
                policy: FrontOperationPolicy::StrictDomain,
            }])
            .unwrap();
        assert!(graph.coupled_kernel().unwrap().replay(&[0]).is_err());
        graph
            .set_coupled_resources(&[CoupledResource::SlopeIntervalCost {
                initial: Window::new(0., 1.).unwrap(),
                bounds: Window::new(0., 3.).unwrap(),
                increments: vec![0.],
                functions: vec![constant(0., 2., 3.)],
                policy: FrontOperationPolicy::StrictDomain,
            }])
            .unwrap();
        assert_eq!(graph.coupled_kernel().unwrap().replay(&[0]).unwrap(), None);
    }
    #[cfg(feature = "clp")]
    #[test]
    fn raw_interval_phase_i_keeps_domains_and_declines_certification() {
        let graph = negative_suffix_graph();
        let mut kernel = graph.coupled_kernel().unwrap().clone();
        kernel.suppress_costs();
        assert!(!kernel.has_costs());
        assert_eq!(kernel.replay(&[0, 1]).unwrap(), Some(0.));
        assert!(kernel.certify(&[0, 1]).is_none());
        let json = serde_json::to_string(kernel.definitions()).unwrap();
        let decoded: Vec<CoupledResource> = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, kernel.definitions());
        let mut restricted = Graph::<2>::new(3, 0).unwrap();
        restricted.add_arc(0, 1, 0.).unwrap();
        restricted.add_arc(1, 2, 0.).unwrap();
        restricted
            .set_coupled_resources(&[definition(vec![
                constant(0., 0., 0.5),
                constant(0., 1., 2.),
            ])])
            .unwrap();
        let mut restricted = restricted.coupled_kernel().unwrap().clone();
        restricted.suppress_costs();
        assert_eq!(restricted.replay(&[0, 1]).unwrap(), None);
    }
    fn parallel_prefixes(
        functions: Vec<PreparedSlopeFunction>,
        costs: [f64; 2],
    ) -> (Graph<2>, Box<CoupledLabelState>, Box<CoupledLabelState>) {
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        for (tail, head, cost) in [(0, 1, costs[0]), (0, 1, costs[1]), (1, 2, 0.)] {
            graph.add_arc(tail, head, cost).unwrap();
        }
        graph
            .set_coupled_resources(&[definition(functions)])
            .unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        let root = kernel.root(true);
        let mut prefix = |arc: usize| {
            let mut state = kernel.extend(&root, arc, true).unwrap().unwrap().0;
            state
                .account_extension(&root, costs[arc], costs[arc])
                .unwrap();
            state
        };
        let (a, b) = (prefix(0), prefix(1));
        (graph, a, b)
    }
    #[test]
    fn equal_raw_fronts_dominate_by_independent_scalar_order() {
        let charges = vec![
            constant(3., 0., 2.),
            constant(3., 0., 2.),
            constant(0., 0., 2.),
        ];
        let (graph, cheap, dear) = parallel_prefixes(charges, [-10., -9.]);
        let kernel = graph.coupled_kernel().unwrap();
        assert_eq!(cheap.slope_fronts, dear.slope_fronts);
        assert_eq!(kernel.dominance_penalty(&cheap, &dear, true), Some(0.));
        assert_eq!(kernel.dominance_penalty(&dear, &cheap, true), None);
        for mode in [0., 1., 2.] {
            let mut solver = crate::pricing::Solver::<2>::new(3, 0).unwrap();
            solver.graph = graph.clone();
            solver
                .set_option("BALDESuseBidirectionalSearch", mode)
                .unwrap();
            let result = solver.price_phase(0, 20, false).unwrap();
            assert_eq!(result.minimum_reduced_cost, -7.);
            assert!(result
                .columns
                .iter()
                .any(|r| r.arcs == [0, 2] && r.original_cost == -7.));
        }
    }
    #[test]
    fn zero_raw_fronts_dominate_by_domain_coverage() {
        let charges = vec![
            constant(0., 0., 2.),
            constant(0., 0., 1.),
            constant(0., 0., 2.),
        ];
        let (graph, wide, narrow) = parallel_prefixes(charges, [0., 0.]);
        let kernel = graph.coupled_kernel().unwrap();
        assert_ne!(wide.slope_fronts, narrow.slope_fronts);
        assert_eq!(kernel.dominance_penalty(&wide, &narrow, true), Some(0.));
        assert_eq!(kernel.dominance_penalty(&narrow, &wide, true), None);
        // The narrow prefix is physically cheaper and must survive pricing.
        let charges = vec![
            constant(0., 0., 2.),
            constant(0., 0., 1.),
            constant(0., 0., 2.),
        ];
        let (graph, ..) = parallel_prefixes(charges, [0., -5.]);
        for mode in [0., 1., 2.] {
            let mut solver = crate::pricing::Solver::<2>::new(3, 0).unwrap();
            solver.graph = graph.clone();
            solver
                .set_option("BALDESuseBidirectionalSearch", mode)
                .unwrap();
            let result = solver.price_phase(0, 20, false).unwrap();
            assert_eq!(result.minimum_reduced_cost, -5.);
            assert!(result.columns.iter().any(|r| r.arcs == [1, 2]));
        }
    }
    #[test]
    fn unequal_costly_raw_fronts_stay_conservative() {
        // The cheaper front is pointwise lower now, but future translations
        // and additions re-round its segment spans; no order is retained.
        let charges = vec![
            constant(1., 0., 2.),
            constant(2., 0., 2.),
            constant(0., 0., 2.),
        ];
        let (graph, low, high) = parallel_prefixes(charges, [0., 0.]);
        let kernel = graph.coupled_kernel().unwrap();
        assert_eq!(kernel.dominance_penalty(&low, &high, true), None);
        assert_eq!(kernel.dominance_penalty(&high, &low, true), None);
        // A future nonzero charge forbids the domain rule even for zero fronts.
        let charges = vec![
            constant(0., 0., 2.),
            constant(0., 0., 1.),
            constant(1., 0., 2.),
        ];
        let (graph, wide, narrow) = parallel_prefixes(charges, [0., 0.]);
        let kernel = graph.coupled_kernel().unwrap();
        assert_eq!(kernel.dominance_penalty(&wide, &narrow, true), None);
    }
    #[cfg(feature = "clp")]
    #[test]
    fn phase_i_raw_fronts_dominate_by_domain_coverage() {
        let charges = vec![
            constant(1., 0., 2.),
            constant(2., 0., 1.),
            constant(5., 0., 2.),
        ];
        let (graph, ..) = parallel_prefixes(charges, [0., 0.]);
        let mut kernel = graph.coupled_kernel().unwrap().clone();
        kernel.suppress_costs();
        let root = kernel.root(true);
        let mut prefix = |arc| {
            let mut state = kernel.extend(&root, arc, true).unwrap().unwrap().0;
            state.account_extension(&root, 0., 0.).unwrap();
            state
        };
        let (wide, narrow) = (prefix(0), prefix(1));
        assert_eq!(kernel.dominance_penalty(&wide, &narrow, true), Some(0.));
        assert_eq!(kernel.dominance_penalty(&narrow, &wide, true), None);
    }
    trait Complete<T> {
        fn complete_for_test(self) -> T;
    }
    impl<T> Complete<T> for FrontierOutcome<T> {
        fn complete_for_test(self) -> T {
            match self {
                FrontierOutcome::Complete(value) => value,
                _ => panic!("expected complete frontier"),
            }
        }
    }
}
