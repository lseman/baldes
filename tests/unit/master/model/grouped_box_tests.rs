use super::*;

fn model() -> format::Model {
    format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/multi_network_original_variables.json"
    )))
    .unwrap()
}
fn options() -> Options {
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.preprocessing.enabled = false;
    options
}
fn envelope(id: i32, upper: f64) -> OriginalVariableBound {
    OriginalVariableBound {
        network: 7,
        id,
        lower: 0.,
        upper: Some(upper),
    }
}

#[test]
fn alternative_original_box_avoids_repricing_without_double_counting_aliases() {
    for mode in [cg::BoundMode::Numerical, cg::BoundMode::Certified] {
        let (problem, registry) = model().compile::<2>().unwrap();
        let mut plain = options();
        plain.cg.bound_mode = mode;
        let baseline = solve(problem.clone(), plain.clone()).unwrap();
        registry
            .configure_boxes(&mut plain, &[envelope(0, 1.)])
            .unwrap();
        let grouped = solve(problem, plain).unwrap();
        assert_eq!(grouped.master.status, cg::Status::Optimal);
        assert_eq!(grouped.master.lp, -1.);
        assert_eq!(grouped.master.bound, -1.);
        assert!(grouped.oracle_calls[0].1 < baseline.oracle_calls[0].1);
        assert!(
            grouped.oracle_calls[0].1 > 0,
            "a negative optional path needs an actual witness"
        );
    }
}

#[test]
fn overlapping_original_variables_decompose_one_arc_cost_once() {
    let mut model = model();
    let mut other = model.pricing_variables[0].clone();
    other.id = 1;
    other.cost = -2.;
    other.integer = false;
    other.arcs = vec![(0, 0.25), (2, 0.25)];
    model.pricing_variables.push(other);
    let (problem, registry) = model.compile::<2>().unwrap();
    let mut configured = options();
    registry
        .configure_boxes(&mut configured, &[envelope(0, 1.), envelope(1, 0.25)])
        .unwrap();
    let result = solve(problem, configured).unwrap();
    assert_eq!(result.master.lp, -1.5);
    assert_eq!(result.master.numerical_bound, -1.5);
    assert_eq!(registry.activity(7, 1, &result).unwrap().lower, 0.25);
}

#[test]
fn residual_cost_enclosure_covers_every_independent_dyadic_path_cost() {
    let (problem, registry) = model().compile::<2>().unwrap();
    let mut configured = options();
    registry
        .configure_boxes(&mut configured, &[envelope(0, 1.)])
        .unwrap();
    let mut output = solve(problem, configured.clone()).unwrap();
    let oracle = &mut output.snapshot.state.oracles[0];
    let groups = grouped_boxes::prepare(
        &configured.grouped_variable_boxes,
        0,
        std::iter::once((7, oracle.network.graph.as_ref(), &[][..])),
    )
    .unwrap();
    let groups = &groups[&7];
    for a in [-2., -0.5, 0., 1.] {
        for b in [-2., -0.5, 0., 1.] {
            for c in [-2., -0.5, 0., 1.] {
                for d in [-2., -0.5, 0., 1.] {
                    oracle.costs = vec![a, b, c, d];
                    for offset in [0., 0.5] {
                        let bound =
                            grouped_variable_box(oracle, offset, groups, None, &[], 0).unwrap();
                        let exact = (a + b).min(c + d) + offset;
                        assert!(
                            bound.minimum <= exact,
                            "costs={:?}, offset={offset}",
                            oracle.costs
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn original_space_cancellation_does_not_use_the_physical_arc_zero_shortcut() {
    let mut model = model();
    model.pricing_variables[0].cost = 1.;
    model.pricing_variables[0].integer = false;
    let mut other = model.pricing_variables[0].clone();
    other.id = 1;
    other.cost = -1.;
    model.pricing_variables.push(other);
    let (problem, registry) = model.compile::<2>().unwrap();
    let baseline = solve(problem.clone(), options()).unwrap();
    assert_eq!(baseline.oracle_calls[0].1, 0);
    let mut configured = options();
    registry
        .configure_boxes(&mut configured, &[envelope(0, 1.), envelope(1, 1.)])
        .unwrap();
    let result = solve(problem, configured).unwrap();
    assert_eq!(result.master.lp, 0.);
    assert!(result.oracle_calls[0].1 > 0);
}

#[test]
fn invalid_group_envelopes_fail_before_a_zero_budget_and_identity_edits_are_atomic() {
    let (problem, registry) = model().compile::<2>().unwrap();
    let mut configured = options();
    registry
        .configure_boxes(&mut configured, &[envelope(0, 0.5)])
        .unwrap();
    configured.cg.time_limit = std::time::Duration::ZERO;
    assert!(solve(problem, configured).is_err());
    let mut configured = options();
    assert!(registry
        .configure_boxes(&mut configured, &[envelope(0, 1.), envelope(100, 1.)])
        .is_err());
    assert!(configured.grouped_variable_boxes.is_empty());
}

#[test]
fn resource_identity_proves_a_cyclic_group_envelope() {
    let mut graph = Graph::<2>::new(3, 2).unwrap();
    for v in 0..3 {
        graph.set_bounds(v, 0, 0., 8.).unwrap();
        graph.set_bounds(v, 1, 0., 2.).unwrap();
    }
    for (tail, head, weight) in [(0, 1, 0.), (1, 1, 1.), (1, 2, 0.)] {
        let arc = graph.add_arc(tail, head, 0.).unwrap();
        graph.arc_mut(arc).unwrap().consumption[0] = 1.;
        graph.arc_mut(arc).unwrap().consumption[1] = weight;
    }
    let variable = OriginalPricingVariable {
        accumulated_resource: None,
        network: 7,
        id: 0,
        cost: -1.,
        entries: vec![],
        arcs: vec![(1, 1.)],
        integer: false,
        priority: 0.,
    };
    let boxes = grouped_boxes::prepare(
        &[GroupedVariableBound {
            variable,
            lower: 0.,
            upper: Some(2.),
        }],
        0,
        std::iter::once((7, &graph, &[][..])),
    )
    .unwrap();
    assert_eq!(boxes[&7].variables[0].bounds.upper, 2.);
    assert_eq!(boxes[&7].caps[1], 2.);
}

#[test]
fn registered_cancellation_keeps_implicit_original_coordinates_on_resume() {
    for mode in [cg::BoundMode::Numerical, cg::BoundMode::Certified] {
        let mut model = model();
        model.pricing_variables[0].cost = 1.;
        model.pricing_variables[0].integer = false;
        let mut other = model.pricing_variables[0].clone();
        other.id = 1;
        other.cost = -1.;
        model.pricing_variables.push(other);
        let (problem, registry) = model.compile::<2>().unwrap();
        let mut configured = options();
        configured.cg.bound_mode = mode;
        registry.configure_pricing(&mut configured);
        let result = solve(problem, configured).unwrap();
        assert_eq!(result.master.lp, 0.);
        assert_eq!(result.master.bound, 0.);
        assert_eq!(result.oracle_calls, vec![(7, 3)]);
        let groups = &result.snapshot.state.grouped_variable_boxes[&7];
        assert_eq!(groups.variables.len(), 2);
        assert!(groups
            .variables
            .iter()
            .all(|v| v.bounds.lower == 0. && v.bounds.upper == f64::INFINITY));
        let resumed = result.snapshot.solve(options()).unwrap();
        assert_eq!(resumed.master.lp, 0.);
        assert_eq!(
            resumed.snapshot.state.grouped_variable_boxes[&7]
                .variables
                .len(),
            2
        );
    }
}

#[test]
fn implicit_original_orthant_never_restricts_negative_graph_activity() {
    let mut model = model();
    model.pricing_variables[0].cost = 1.;
    model.pricing_variables[0].integer = false;
    model.pricing_variables[0].arcs = vec![(0, -1.), (2, -1.)];
    let (problem, registry) = model.compile::<2>().unwrap();
    let mut configured = options();
    registry.configure_pricing(&mut configured);
    let result = solve(problem, configured).unwrap();
    assert_eq!(result.master.lp, -1.);
    assert!(!result
        .snapshot
        .state
        .grouped_variable_boxes
        .contains_key(&7));
    assert_eq!(registry.activity(7, 0, &result).unwrap().lower, -1.);
}

#[test]
fn implicit_original_orthant_proves_signed_complete_path_cancellation() {
    let mut model = model();
    model.pricing_variables[0].cost = -1.;
    model.pricing_variables[0].integer = false;
    model.pricing_variables[0].arcs = vec![(0, -1.), (1, 1.), (2, -1.), (3, 1.)];
    let (problem, registry) = model.compile::<2>().unwrap();
    let mut configured = options();
    registry.configure_pricing(&mut configured);
    let result = solve(problem, configured).unwrap();
    assert_eq!(result.master.lp, 0.);
    let groups = &result.snapshot.state.grouped_variable_boxes[&7];
    assert_eq!(groups.variables.len(), 1);
    assert_eq!(groups.variables[0].bounds.lower, 0.);
    assert_eq!(groups.variables[0].bounds.upper, f64::INFINITY);
}
