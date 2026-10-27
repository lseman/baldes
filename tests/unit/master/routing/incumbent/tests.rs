use super::*;
#[test]
fn rounded_witness_uses_original_cost_and_exact_rows() {
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    for v in 0..3 {
        graph.set_bounds(v, 0, 0., 10.).unwrap();
    }
    graph.add_arc(0, 1, 1e6).unwrap();
    graph.add_arc(1, 2, 0.25).unwrap();
    let mut problem = RoutingContext::set_partitioning(graph, 1.).unwrap();
    let ids = problem.model.networks[0]
        .graph
        .arcs
        .iter()
        .map(|a| a.id)
        .collect::<Vec<_>>();
    let route = problem.elementary_route(&ids).unwrap();
    assert_eq!(
        validated_upper(&problem, &[route.clone()], &[0], &[1. - 1e-12]),
        Some(1_000_000.25)
    );
    assert!(validated_upper(&problem, &[route.clone()], &[0], &[2.]).is_none());
    problem.model.rows[1].rhs = 1. + 1e-12;
    assert!(validated_upper(&problem, &[route], &[0], &[1.]).is_none());
}
#[test]
fn exact_sum_preserves_cancellation_and_rounds_objective_up() {
    let mut sum = ExactSum::default();
    for x in [1e16, 1., -1e16] {
        sum.add(x).unwrap();
    }
    assert_eq!(sum.upper(), Some(1.));
    sum.add(-1.).unwrap();
    assert_eq!(sum.sign(), 0.);
    sum.add(1.).unwrap();
    sum.add(f64::EPSILON / 4.).unwrap();
    assert_eq!(sum.upper(), Some(f64::from_bits(1f64.to_bits() + 1)));
}
fn chain(consumption: &[f64], upper: f64) -> (RoutingContext<2>, Route) {
    let n = consumption.len() + 1;
    let mut graph = Graph::<2>::new(n, 1).unwrap();
    graph.initial[0] = Some(0.);
    for v in 0..n {
        graph.set_bounds(v, 0, 0., upper).unwrap();
    }
    for (v, &c) in consumption.iter().enumerate() {
        let id = graph.add_arc(v, v + 1, 1.).unwrap();
        graph.arc_mut(id).unwrap().consumption[0] = c;
    }
    let problem = RoutingContext::set_partitioning(graph, 1.).unwrap();
    // Profiles and rounded costs are untrusted payloads. Feasibility and
    // objective are reconstructed from physical arcs and original inputs.
    let route = Route {
        coupled_controls: Vec::new(),
        coupled_values: Vec::new(),
        nodes: (0..n as i32).collect(),
        arcs: (0..n as i32 - 1).collect(),
        cost: 0.,
        original_cost: 0.,
        resources: vec![0.],
        resource_profile: vec![vec![0.]; n],
    };
    (problem, route)
}
#[test]
fn exact_resources_reject_false_feasibility_and_recover_false_rejection() {
    let (p, r) = chain(&[1., f64::EPSILON / 2.], 1.);
    assert!(p.elementary_route(&r.arcs).is_some());
    assert!(validated_upper(&p, &[r], &[0], &[1.]).is_none());
    let delta = 3. * 2f64.powi(-53);
    let upper = f64::from_bits(1f64.to_bits() + 3);
    let (p, r) = chain(&[1., delta, delta], upper);
    assert!(p.elementary_route(&r.arcs).is_none());
    assert_eq!(validated_upper(&p, &[r], &[0], &[1.]), Some(3.));
}
#[test]
fn exact_resource_waiting_windows_and_equality_are_distinct() {
    let (mut p, r) = chain(&[0., 0.], f64::MAX);
    p.model.networks[0].graph.initial[0] = Some(-f64::MAX);
    p.model.networks[0]
        .graph
        .set_bounds(0, 0, -f64::MAX, -f64::MAX)
        .unwrap();
    p.model.networks[0]
        .graph
        .set_bounds(1, 0, f64::MAX, f64::MAX)
        .unwrap();
    p.model.networks[0]
        .graph
        .set_bounds(2, 0, f64::MAX, f64::MAX)
        .unwrap();
    assert!(exact_route_feasible(&p, &r).is_some());
    p.model.networks[0].graph.resources[0] = crate::graph::ResourceKind::Equality;
    assert!(exact_route_feasible(&p, &r).is_none());
    let (mut p, r) = chain(&[0.7, 0.3], 2.);
    p.model.networks[0].graph.arcs[0].bounds[0] = Some(crate::Window::new(1., 1.).unwrap());
    assert!(exact_route_feasible(&p, &r).is_some());
    p.model.networks[0].graph.set_bounds(2, 0, 0., 1.).unwrap();
    assert!(exact_route_feasible(&p, &r).is_none());
}
#[cfg(feature = "multi-time-windows")]
#[test]
fn exact_witness_waits_across_gaps_and_checks_arc_intersections() {
    let (mut problem, route) = chain(&[4., 2.], 20.);
    problem.model.networks[0]
        .graph
        .set_time_windows(
            1,
            0,
            &[
                crate::Window::new(0., 2.).unwrap(),
                crate::Window::new(8., 12.).unwrap(),
            ],
        )
        .unwrap();
    assert_eq!(exact_route_feasibility(&problem, &route), Some(true));
    problem.model.networks[0]
        .graph
        .set_bounds(2, 0, 0., 9.)
        .unwrap();
    assert_eq!(exact_route_feasibility(&problem, &route), Some(false));
    problem.model.networks[0]
        .graph
        .set_bounds(2, 0, 0., 20.)
        .unwrap();
    problem.model.networks[0].graph.arcs[0].bounds[0] = Some(crate::Window::new(3., 7.).unwrap());
    assert_eq!(exact_route_feasibility(&problem, &route), Some(false));
}

#[test]
fn fixed_resource_paths_and_recorded_node_chains_are_revalidated() {
    let (mut p, r) = chain(&[1., f64::EPSILON / 2.], 1.);
    p.fixed_routes.push((r, 1));
    for row in &mut p.model.rows {
        row.rhs = 0.;
    }
    assert!(validated_upper(&p, &[], &[], &[]).is_none());
    let (mut p, r) = chain(&[0.5, 0.5], 1.);
    p.fixed_routes.push((r, 1));
    for row in &mut p.model.rows {
        row.rhs = 0.;
    }
    assert_eq!(validated_upper(&p, &[], &[], &[]), Some(2.));
    let (mut p, mut r) = chain(&[0., 0., 0.], 1.);
    p.branch(BranchConstraint::RyanFoster(
        crate::pricing::restrictions::RyanFosterConstraint {
            first: 1,
            second: 2,
            together: false,
        },
    ))
    .unwrap();
    r.nodes[2] = 3; // Cannot hide the second endpoint from the branch row.
    assert!(validated_upper(&p, &[r], &[0], &[1.]).is_none());
}

#[test]
fn fractional_product_upper_preserves_subnormal_remainders_and_cancellation() {
    let unit = f64::from_bits(1);
    for n in -31i32..=31 {
        for k in -31..=31 {
            let mut sum = ExactSum::default();
            sum.product_upper(f64::from(n) * unit, f64::from(k) / 8.)
                .unwrap();
            let rounded_units = sum.upper().unwrap() / unit;
            let numerator = n * k;
            let ceiling = -(-numerator).div_euclid(8);
            assert!(rounded_units >= f64::from(ceiling), "n={n}, k={k}");
            assert!(rounded_units <= f64::from(ceiling + 1), "n={n}, k={k}");
        }
    }
    let mut sum = ExactSum::default();
    sum.product_upper(unit, 2.5).unwrap();
    sum.add(-2. * unit).unwrap();
    assert_eq!(sum.upper(), Some(unit));

    let mut exact = ExactSum::default();
    exact.product_upper(f64::MIN_POSITIVE, 0.5).unwrap();
    assert_eq!(exact.upper(), Some(f64::MIN_POSITIVE / 2.));
    exact.product_upper(0., 0.5).unwrap();
    assert_eq!(exact.upper(), Some(f64::MIN_POSITIVE / 2.));
}

#[test]
fn signed_disposable_incumbents_apply_exact_waiting_and_upper_windows() {
    let (mut problem, route) = chain(&[0.5, -0.75, 0.25], 2.);
    assert_eq!(exact_route_feasibility(&problem, &route), Some(true));
    assert_eq!(
        validated_upper(&problem, &[route.clone()], &[0], &[1.]),
        Some(3.)
    );
    problem.model.networks[0]
        .graph
        .set_bounds(2, 0, 1., 2.)
        .unwrap();
    problem.model.networks[0]
        .graph
        .set_bounds(3, 0, 0., 1.)
        .unwrap();
    assert_eq!(exact_route_feasibility(&problem, &route), Some(false));
    problem.model.networks[0].graph.resources[0] = crate::graph::ResourceKind::Equality;
    assert_eq!(exact_route_feasibility(&problem, &route), Some(false));
}

#[test]
fn signed_resource_replay_retains_exact_cancellation_beyond_binary64_profiles() {
    let (mut problem, route) = chain(&[1., f64::EPSILON / 2., -1.], 2.);
    problem.model.networks[0]
        .graph
        .set_bounds(3, 0, 0., 0.)
        .unwrap();
    // A rounded replay loses the half-ulp before subtracting one. The actual
    // sum is still positive and violates the terminal zero upper bound.
    assert!(problem.elementary_route(&route.arcs).is_some());
    assert_eq!(exact_route_feasibility(&problem, &route), Some(false));
    assert!(validated_upper(&problem, &[route], &[0], &[1.]).is_none());
}
