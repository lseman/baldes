use super::*;

fn problem() -> RoutingContext<2> {
    let mut graph = crate::Graph::<2>::new(4, 0).unwrap();
    for (tail, head) in [(0, 1), (0, 2), (1, 2), (2, 1), (1, 3), (2, 3)] {
        graph.add_arc(tail, head, 1.).unwrap();
    }
    RoutingContext::set_partitioning(graph, 2.).unwrap()
}

#[test]
fn deterministic_domain_failures_are_not_repeated_but_larger_budgets_retry() {
    let problem = problem();
    let generous = PathDomainLimits {
        deadline: None,
        ..PathDomainLimits::default()
    };
    let domain = ConservativeCoverRelaxation::collect(&problem, generous).unwrap();
    let exact = PathDomainLimits {
        max_paths: domain.paths().paths().len(),
        max_stored_arcs: domain.paths().stored_arcs(),
        max_work: domain.work_used(),
        deadline: None,
    };
    for insufficient in [
        PathDomainLimits {
            max_work: exact.max_work - 1,
            ..exact
        },
        PathDomainLimits {
            max_paths: exact.max_paths - 1,
            ..exact
        },
        PathDomainLimits {
            max_stored_arcs: exact.max_stored_arcs - 1,
            ..exact
        },
    ] {
        let cache = SearchCache::default();
        assert!(cache.collect(&problem, insufficient).is_none());
        assert_eq!(cache.attempts.get(), 1);
        assert!(cache.collect(&problem, insufficient).is_none());
        assert_eq!(cache.attempts.get(), 1);
        let recovered = cache.collect(&problem, exact).unwrap();
        assert_eq!(cache.attempts.get(), 2);
        assert_eq!(recovered.paths().paths().len(), exact.max_paths);
        assert_eq!(recovered.work_used(), exact.max_work);
    }
}

#[test]
fn expired_deadline_does_not_poison_a_later_certificate_attempt() {
    let problem = problem();
    let cache = SearchCache::default();
    let limits = PathDomainLimits {
        deadline: Some(Instant::now()),
        ..PathDomainLimits::default()
    };
    assert!(cache.collect(&problem, limits).is_none());
    assert!(cache.failed_domain_limits.get().is_none());
    assert!(cache
        .collect(
            &problem,
            PathDomainLimits {
                deadline: None,
                ..limits
            }
        )
        .is_some());
    assert_eq!(cache.attempts.get(), 2);
}

#[test]
fn paired_front_program_never_uses_a_source_control_root_lower_bound() {
    use crate::graph::coupled::descriptor::{
        FrontierBudget, FrontierOutcome, MachineRootInput, RawAffine,
    };
    use crate::graph::coupled::{
        CoupledResource, MachineArcFunctionInput, PairedResourceMode, PreparedSlopeFunction,
    };
    use crate::graph::Window;
    let mut graph = crate::Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::PairedFront {
            mode: PairedResourceMode::CostUnbounded,
        }])
        .unwrap();
    let bounds = Window::new(0., 2.).unwrap();
    let base = PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (0., 2.)]).unwrap();
    let root =
        PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (-0.00005, 1.), (-0.00001, 2.)])
            .unwrap();
    let affine = |value| MachineRootInput::Affine(RawAffine::new(value, 0.).unwrap());
    let inputs: Vec<_> = (0..2)
        .map(|physical_arc| MachineArcFunctionInput {
            physical_arc,
            tail_bounds: bounds,
            waiting_bounds: bounds,
            waiting_rate: 0,
            base: &base,
            dependent: affine(0.),
            dependent_lower: affine(-10.),
            dependent_upper: affine(10.),
        })
        .collect();
    let result = graph.prepare_paired_front_program(
        0,
        &inputs,
        MachineRootInput::Vector(&root),
        bounds,
        bounds,
        FrontierBudget::default(),
    );
    assert!(
        matches!(result, FrontierOutcome::Complete(())),
        "{result:?}"
    );
    assert!(graph
        .coupled_kernel()
        .unwrap()
        .replay(&[0, 1])
        .unwrap()
        .unwrap()
        .is_finite());
    let problem = RoutingContext::set_partitioning(graph, 1.).unwrap();
    assert!(ConservativeCoverRelaxation::collect(&problem, PathDomainLimits::default()).is_err());
    let mut options = RootOptions::default();
    options.capture_state = true;
    options.enumerate = false;
    options.cg.stabilize = false;
    let output = super::super::solve(problem, options).unwrap();
    let snapshot = output.snapshot.unwrap();
    assert!(snapshot.problem.model.networks[0]
        .graph
        .front_program_domain());
    let state = State {
        solver: snapshot.pricing.fork_for_node(),
        routes: snapshot.routes.clone(),
        route_tags: Default::default(),
        incumbent: f64::INFINITY,
        enumerate: false,
        rounded: false,
        pricing: adapter::RootPricing::default(),
        previous_fixing_gap: 1e12,
        deadline: None,
        pricing_seconds: 0.,
        cut: snapshot.cut.clone(),
        fixing_calls: 0,
        enumerated_routes: 0,
    };
    assert_eq!(
        lower_bound(
            &snapshot.problem,
            &state,
            &snapshot.master,
            &SearchCache::default()
        ),
        None
    );
}
