use super::*;

fn routing() -> (RoutingContext<2>, State<2>) {
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    graph.elementary = true;
    graph.add_elementarity_set(&[1]).unwrap();
    for v in 0..3 {
        graph.set_bounds(v, 0, 0., 10.).unwrap();
    }
    for (tail, head) in [(0, 1), (1, 2)] {
        let id = graph.add_arc(tail, head, 1.).unwrap();
        graph.arc_mut(id).unwrap().consumption[0] = 1.;
    }
    let mut options = RootOptions::default();
    options.capture_state = true;
    options.enumerate = false;
    let output = crate::master::routing::solve(
        RoutingContext::set_partitioning(graph, 1.).unwrap(),
        options,
    )
    .unwrap();
    let snapshot = output.snapshot.unwrap();
    let state = State {
        solver: snapshot.pricing.fork_for_node(),
        routes: snapshot.routes.clone(),
        route_tags: snapshot
            .routes
            .iter()
            .enumerate()
            .map(|(p, r)| (r.identity(), p as i32))
            .collect(),
        incumbent: f64::INFINITY,
        enumerate: false,
        rounded: true,
        pricing: RootPricing::default(),
        previous_fixing_gap: f64::INFINITY,
        deadline: None,
        pricing_seconds: 0.,
        cut: snapshot.cut.clone(),
        fixing_calls: 0,
        enumerated_routes: 0,
    };
    (snapshot.problem.clone(), state)
}

#[test]
fn live_and_retained_routing_domains_use_the_same_batch_contribution_and_raw_rows() {
    let (problem, mut state) = routing();
    state.pricing.rhs = problem.model.rows.iter().map(|r| r.rhs).collect();
    let phase = state.solver.phases.iter().position(|p| p.exact()).unwrap();
    let duals = [0., 3.];
    let live = state.price(&problem, phase, &duals, 0).unwrap();
    assert!(live.exact);
    assert_eq!(live.minimum, -1.);
    assert_eq!(state.pricing.correction, Some(-1.));
    assert_eq!(state.pricing.gradient, Some(vec![0., 0.]));
    assert_eq!(state.pricing.oracle_calls, 1);
    state.solver.enumerate(1000.).unwrap();
    assert!(state.solver.enumeration.as_ref().unwrap().completed);
    let retained = state.price(&problem, phase, &duals, 0).unwrap();
    assert!(retained.exact);
    assert_eq!(retained.minimum, live.minimum);
    assert_eq!(state.pricing.correction, Some(-1.));
    assert_eq!(state.pricing.gradient, Some(vec![0., 0.]));
    assert_eq!(state.pricing.oracle_calls, 2);
    assert_eq!(retained.columns.len(), live.columns.len());
    assert_eq!(retained.columns[0].cost, 2.);
    assert_eq!(retained.columns[0].entries, live.columns[0].entries);
    assert_eq!(
        retained.columns[0].reduced_cost,
        live.columns[0].reduced_cost
    );
}

#[test]
fn routing_timeout_clears_batch_proof_and_coverage_at_the_current_duals() {
    let (problem, mut state) = routing();
    state.pricing.rhs = problem.model.rows.iter().map(|r| r.rhs).collect();
    let phase = state.solver.phases.iter().position(|p| p.exact()).unwrap();
    state.price(&problem, phase, &[0., 3.], 0).unwrap();
    assert!(state.pricing.complete());
    state.deadline = Some(std::time::Instant::now());
    assert!(state.price(&problem, phase, &[0., 4.], 0).is_err());
    assert!(!state.pricing.complete());
    assert_eq!(state.pricing.correction, None);
    assert_eq!(state.pricing.gradient, None);
}
