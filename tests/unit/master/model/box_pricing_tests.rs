use super::*;

fn positive_network(lower: f64, stabilized: bool) -> Output<2> {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 1.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    let mut options = Options::default();
    options.cg.stabilize = stabilized;
    options.cg.artificial_cost = 100.;
    options.preprocessing.enabled = false;
    solve(
        Problem {
            rows: vec![],
            networks: vec![Network {
                id: 7,
                graph,
                multiplicity: Window { lower, upper: 2. },
                arc_entries: vec![vec![]; 2],
                route_entries: vec![],
            }],
        },
        options,
    )
    .unwrap()
}

#[test]
fn zero_lower_nonnegative_network_needs_no_pricing_witness() {
    let output = positive_network(0., false);
    assert_eq!(output.master.status, cg::Status::Optimal);
    assert_eq!(output.master.lp, 0.);
    assert_eq!(output.oracle_calls, vec![(7, 0)]);
    assert!(output.snapshot.state.routes.is_empty());
    assert_eq!(output.snapshot.state.correction, Some(0.));
}

#[test]
fn positive_multiplicity_still_requires_a_priced_route() {
    let output = positive_network(1., false);
    assert_eq!(output.master.status, cg::Status::Optimal);
    assert_eq!(output.master.lp, 1.);
    assert!(output.oracle_calls[0].1 > 0);
    assert!(!output.snapshot.state.routes.is_empty());
}

#[test]
fn explicit_stabilization_bypasses_native_box_admission() {
    let output = positive_network(0., true);
    assert!(output.oracle_calls[0].1 > 0);
    let mut options = Options::default();
    options.cg.stabilize = false;
    let resumed = output.snapshot.solve(options).unwrap();
    assert_eq!(resumed.oracle_calls, vec![(7, 0)]);
}

#[test]
fn box_admission_rejects_negative_nonfinite_and_omitted_cost_terms() {
    let mut output = positive_network(0., false);
    let oracle = &mut output.snapshot.state.oracles[0];
    oracle.costs = vec![1., 0.];
    assert!(nonnegative_box(oracle, 0.));
    assert!(!nonnegative_box(oracle, -1.));
    assert!(!nonnegative_box(oracle, f64::NAN));
    oracle.costs[0] = -1e-15;
    assert!(!nonnegative_box(oracle, 0.));
    oracle.costs[0] = f64::INFINITY;
    assert!(!nonnegative_box(oracle, 0.));
    oracle.costs[0] = 1.;
    oracle.nonlinear_rows.push((0, 0));
    assert!(!nonnegative_box(oracle, 0.));
    oracle.nonlinear_rows.clear();
    oracle.knapsack_rows.push((0, 0));
    assert!(!nonnegative_box(oracle, 0.));
    oracle.knapsack_rows.clear();
    oracle.presence_rows.push((0, 0));
    assert!(!nonnegative_box(oracle, 0.));
}

#[test]
fn changed_shared_duals_restore_pricing_and_fresh_zero_bound_admission() {
    let root = positive_network(0., false);
    let mut node = root
        .snapshot
        .with_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 1.,
            },
            &[RowCoefficients {
                network: 7,
                arcs: vec![1., 0.],
                per_route: 0.,
            }],
        )
        .unwrap();
    let phase = node.state.oracles[0]
        .solver
        .phases
        .iter()
        .position(|phase| phase.exact())
        .unwrap();
    let mut duals = vec![0.; node.state.rhs.len()];
    duals[node.state.shared_offset] = 2.;
    let improving = node.state.price(phase, &duals).unwrap();
    assert!(improving.exact);
    assert!(improving.minimum < 0.);
    assert!(!improving.columns.is_empty());
    assert_eq!(node.state.oracles[0].calls, 1);
    duals[node.state.shared_offset] = 0.;
    let nonnegative = node.state.price(phase, &duals).unwrap();
    assert!(nonnegative.exact);
    assert_eq!(nonnegative.minimum, 0.);
    assert!(nonnegative.columns.is_empty());
    assert_eq!(node.state.oracles[0].calls, 1);
    assert_eq!(node.state.correction, Some(0.));
}

#[test]
fn box_admission_preserves_heuristic_phase_contract() {
    let mut output = positive_network(0., false);
    let phase = output.snapshot.state.oracles[0]
        .solver
        .phases
        .iter()
        .position(|phase| !phase.exact())
        .unwrap();
    let duals = vec![0.; output.snapshot.state.rhs.len()];
    let answer = output.snapshot.state.price(phase, &duals).unwrap();
    assert!(!answer.exact);
    assert_eq!(output.snapshot.state.oracles[0].calls, 0);
}

fn near_zero_point(upper: Option<f64>) -> (NodeSnapshot<2>, Vec<f64>) {
    let output = positive_network(0., false);
    let mut node = output
        .snapshot
        .with_row(
            Row {
                sense: RowSense::Equal,
                rhs: 0.,
            },
            &[RowCoefficients {
                network: 7,
                arcs: vec![1., 0.],
                per_route: 0.,
            }],
        )
        .unwrap();
    // Zero route quantities give a feasible RMP point of objective zero;
    // a deliberately loose upper estimate exercises stronger-bound fallback.
    node.state.pricing_master = upper.map(|value| (value, 0.));
    let mut duals = vec![0.; node.state.rhs.len()];
    duals[node.state.shared_offset] = 1. + f64::EPSILON;
    (node, duals)
}

#[test]
fn near_zero_admission_retains_negative_correction_without_a_fabricated_gradient() {
    for exact in [false, true] {
        let (mut node, duals) = near_zero_point(Some(0.));
        let phase = node.state.oracles[0]
            .solver
            .phases
            .iter()
            .position(|phase| phase.exact() == exact)
            .unwrap();
        let answer = node.state.price(phase, &duals).unwrap();
        assert_eq!(answer.exact, exact);
        assert!(answer.columns.is_empty());
        assert_eq!(answer.minimum, -2. * f64::EPSILON);
        assert_eq!(node.state.correction, Some(-4. * f64::EPSILON));
        assert!(node.state.gradient.is_none());
        assert_eq!(node.state.oracles[0].calls, 0);
        assert!(node.state.scheduler.complete);
    }
}

#[test]
fn weak_near_zero_estimate_or_missing_master_point_falls_back_to_real_pricing() {
    for upper in [Some(100.), Some(f64::INFINITY), None] {
        let (mut node, duals) = near_zero_point(upper);
        let phase = node.state.oracles[0]
            .solver
            .phases
            .iter()
            .position(|phase| phase.exact())
            .unwrap();
        let answer = node.state.price(phase, &duals).unwrap();
        assert!(answer.exact);
        assert!(!answer.columns.is_empty());
        assert_eq!(answer.minimum, -f64::EPSILON);
        assert_eq!(node.state.oracles[0].calls, 1);
        assert!(node.state.gradient.is_some());
    }
}

#[test]
fn near_zero_admission_bounds_every_pricing_walk_and_its_quantity() {
    let mut output = positive_network(0., false);
    let oracle = &mut output.snapshot.state.oracles[0];
    oracle.costs = vec![-f64::EPSILON, 0.];
    assert!(near_zero_box(oracle, 0., 1e-6).is_some());
    Arc::make_mut(&mut oracle.network.graph)
        .add_arc(1, 1, 0.)
        .unwrap();
    oracle.costs.push(0.);
    // An original physical-simple domain alone does not bound its NG walks.
    assert!(near_zero_box(oracle, 0., 1e-6).is_none());
    Arc::make_mut(&mut oracle.network.graph)
        .set_route_domain(crate::graph::RouteDomain::BoundedWalks { max_arcs: 3 })
        .unwrap();
    assert!(near_zero_box(oracle, 0., 1e-6).is_some());
    oracle.network.multiplicity.upper = f64::INFINITY;
    assert!(near_zero_box(oracle, 0., 1e-6).is_none());
    oracle.network.multiplicity.upper = 1.5;
    assert!(near_zero_box(oracle, 0., 1e-6).is_none());
}

#[test]
fn individually_small_negative_costs_cannot_hide_an_improving_long_walk() {
    let mut output = positive_network(0., false);
    let oracle = &mut output.snapshot.state.oracles[0];
    Arc::make_mut(&mut oracle.network.graph)
        .add_arc(1, 1, -1e-7)
        .unwrap();
    Arc::make_mut(&mut oracle.network.graph)
        .set_route_domain(crate::graph::RouteDomain::BoundedWalks { max_arcs: 100 })
        .unwrap();
    oracle.costs = vec![-1e-7, 0., -1e-7];
    assert!(near_zero_box(oracle, 0., 1e-6).is_none());
}

#[test]
fn an_improving_known_tag_does_not_suppress_stronger_bound_fallback() {
    let problem = format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem::<2>()
    .unwrap();
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 100.;
    options.preprocessing.enabled = false;
    let output = solve(problem, options).unwrap();
    let mut node = output.snapshot.fork().unwrap();
    assert_eq!(node.state.routes.len(), 9);
    node.state.pricing_master = Some((100., 0.));
    let mut duals = vec![0.; node.state.rhs.len()];
    duals[node.state.shared_offset] = 1. + f64::EPSILON;
    // A large negative per-route offset prices network42. Its paths already
    // have tags, so their returned negative costs do not guarantee a new LP
    // column. They cannot hide a weak bound-only estimate on network7.
    duals[1] = 1.;
    let phase = node.state.oracles[0]
        .solver
        .phases
        .iter()
        .position(|phase| phase.exact())
        .unwrap();
    let answer = node.state.price(phase, &duals).unwrap();
    assert!(answer.exact);
    assert!(answer
        .columns
        .iter()
        .any(|column| column.reduced_cost < -1e-6));
    assert_eq!(node.state.oracles[0].calls, 1);
    assert_eq!(node.state.oracles[1].calls, 2);
    assert!(node.state.gradient.is_some());
}

fn box_bounds(oracle: &Oracle<2>, upper: f64) -> Result<Vec<crate::master::preprocessing::Bounds>> {
    let definition = VariableBox {
        network: oracle.network.id,
        variables: oracle
            .network
            .graph
            .arcs
            .iter()
            .map(|arc| ArcVariableBound {
                arc: arc.id,
                lower: 0.,
                upper: Some(upper),
            })
            .collect(),
    };
    Ok(variable_boxes::prepare(
        &[definition],
        std::iter::once((
            oracle.network.id,
            oracle.network.graph.as_ref(),
            oracle.solver.forbidden_arcs.as_slice(),
        )),
    )?
    .remove(&oracle.network.id)
    .unwrap())
}

#[test]
fn finite_variable_box_balances_negative_arc_costs_without_extra_pricing() {
    for mode in [cg::BoundMode::Numerical, cg::BoundMode::Certified] {
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        graph.add_arc(0, 1, -1.).unwrap();
        graph.add_arc(1, 2, 0.).unwrap();
        let mut options = Options::default();
        options.cg.stabilize = false;
        options.cg.bound_mode = mode;
        options.preprocessing.enabled = false;
        options.variable_boxes = vec![VariableBox {
            network: 7,
            variables: vec![
                ArcVariableBound {
                    arc: 0,
                    lower: 0.,
                    upper: Some(1.),
                },
                ArcVariableBound {
                    arc: 1,
                    lower: 0.,
                    upper: Some(1.),
                },
            ],
        }];
        let output = solve(
            Problem {
                rows: vec![],
                networks: vec![Network {
                    id: 7,
                    graph,
                    multiplicity: Window {
                        lower: 0.,
                        upper: 1.,
                    },
                    arc_entries: vec![vec![]; 2],
                    route_entries: vec![],
                }],
            },
            options,
        )
        .unwrap();
        assert_eq!(output.master.status, cg::Status::Optimal);
        assert_eq!(output.master.lp, -1.);
        assert_eq!(output.oracle_calls, vec![(7, 1)]);
        assert_eq!(output.snapshot.state.correction, Some(-1.));
        assert!(output.snapshot.state.gradient.is_none());
    }
}

#[test]
fn finite_box_uses_exact_cancellation_and_actual_walk_limits() {
    let mut output = positive_network(0., false);
    let oracle = &mut output.snapshot.state.oracles[0];
    let bounds = box_bounds(oracle, 1.).unwrap();
    oracle.costs = vec![-1., -f64::EPSILON / 4.];
    assert!(finite_variable_box(oracle, 1., &bounds).unwrap().minimum < 0.);
    assert_eq!(
        finite_variable_box(oracle, 1. + f64::EPSILON, &bounds),
        Some(BoxAdmission::zero())
    );
    assert!(finite_variable_box(oracle, f64::NAN, &bounds).is_none());
    Arc::make_mut(&mut oracle.network.graph)
        .add_arc(1, 1, 0.)
        .unwrap();
    oracle.costs = vec![0., 0., -1.];
    assert!(box_bounds(oracle, 1.).is_err());
    Arc::make_mut(&mut oracle.network.graph)
        .set_route_domain(crate::graph::RouteDomain::BoundedWalks { max_arcs: 3 })
        .unwrap();
    assert!(box_bounds(oracle, 0.).is_err());
    assert!(box_bounds(oracle, 1.).is_ok());
    let bounds = box_bounds(oracle, 3.).unwrap();
    assert_eq!(
        finite_variable_box(oracle, 2., &bounds).unwrap().minimum,
        -1.
    );
    assert_eq!(
        finite_variable_box(oracle, 3., &bounds),
        Some(BoxAdmission::zero())
    );
    oracle.network.multiplicity.lower = 1.;
    let required = finite_variable_box(oracle, 3., &bounds).unwrap();
    assert_eq!(required.minimum, 0.);
    assert!(!required.zero_quantity);
}

#[test]
fn forbidden_negative_arc_has_zero_box_upper_bound() {
    let mut output = positive_network(0., false);
    let oracle = &mut output.snapshot.state.oracles[0];
    let bounds = box_bounds(oracle, 1.).unwrap();
    oracle.costs = vec![-1., 0.];
    assert!(finite_variable_box(oracle, 0., &bounds).unwrap().minimum < 0.);
    oracle.solver.forbidden_arcs.resize(2, false);
    oracle.solver.forbidden_arcs[0] = true;
    assert_eq!(
        finite_variable_box(oracle, 0., &bounds),
        Some(BoxAdmission::zero())
    );
}

#[test]
fn every_admission_gate_declines_omitted_grouped_resource_terms() {
    let mut output = positive_network(0., false);
    let oracle = &mut output.snapshot.state.oracles[0];
    let bounds = box_bounds(oracle, 1.).unwrap();
    oracle.resource_groups.push((
        0,
        crate::master::branching::AccumulatedResourceGroupBranch::new(0, vec![0], 0, 1., true)
            .unwrap(),
    ));
    oracle.costs = vec![1., 0.];
    assert!(!nonnegative_box(oracle, 0.));
    oracle.costs = vec![-1., 0.];
    assert!(finite_variable_box(oracle, 1., &bounds).is_none());
    oracle.costs = vec![-f64::EPSILON, 0.];
    assert!(near_zero_box(oracle, 0., 1e-6).is_none());
}

#[test]
fn weak_or_missing_master_point_for_finite_box_really_prices_the_network() {
    for point in [None, Some((0., 0.))] {
        let mut output = positive_network(0., false);
        let node = &mut output.snapshot;
        Arc::make_mut(&mut node.state.oracles[0].network.graph).arcs[0].cost = -1.;
        node.state.pricing_master = point;
        node.state
            .variable_boxes
            .insert(7, box_bounds(&node.state.oracles[0], 1.).unwrap());
        let phase = node.state.oracles[0]
            .solver
            .phases
            .iter()
            .position(|p| p.exact())
            .unwrap();
        let duals = vec![0.; node.state.rhs.len()];
        let answer = node.state.price(phase, &duals).unwrap();
        assert!(answer.exact);
        assert_eq!(answer.minimum, -1.);
        assert!(!answer.columns.is_empty());
        assert_eq!(node.state.oracles[0].calls, 1);
    }
}

#[test]
fn registered_loose_bounds_are_not_silently_replaced_by_stronger_graph_caps() {
    let mut output = positive_network(0., false);
    let oracle = &mut output.snapshot.state.oracles[0];
    let bounds = box_bounds(oracle, 99.).unwrap();
    oracle.costs = vec![-1., 0.];
    assert_eq!(
        finite_variable_box(oracle, 0., &bounds).unwrap().minimum,
        -99.
    );
    let bounds = vec![
        crate::master::preprocessing::Bounds {
            lower: -2.,
            upper: 1.
        };
        2
    ];
    oracle.costs = vec![1., 0.];
    assert_eq!(
        finite_variable_box(oracle, 0., &bounds).unwrap().minimum,
        -2.
    );
}

#[test]
fn variable_boxes_reject_unknown_or_unproved_inputs_before_zero_budget_returns() {
    for definition in [
        VariableBox {
            network: 99,
            variables: vec![],
        },
        VariableBox {
            network: 7,
            variables: vec![ArcVariableBound {
                arc: 99,
                lower: 0.,
                upper: Some(1.),
            }],
        },
        VariableBox {
            network: 7,
            variables: vec![ArcVariableBound {
                arc: 0,
                lower: 0.,
                upper: Some(0.),
            }],
        },
        VariableBox {
            network: 7,
            variables: vec![ArcVariableBound {
                arc: 0,
                lower: 2.,
                upper: Some(2.),
            }],
        },
    ] {
        let mut options = ModelTreeOptions::default();
        options.node_limit = 0;
        options.node.variable_boxes = vec![definition];
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        graph.add_arc(0, 1, 1.).unwrap();
        graph.add_arc(1, 2, 0.).unwrap();
        let problem = Problem {
            rows: vec![],
            networks: vec![Network {
                id: 7,
                graph,
                multiplicity: Window {
                    lower: 0.,
                    upper: 1.,
                },
                arc_entries: vec![vec![]; 2],
                route_entries: vec![],
            }],
        };
        assert!(crate::master::tree::solve(problem, options).is_err());
    }
}

#[test]
fn variable_box_zero_budget_validation_rejects_bad_graph_without_walking_it() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 1.).unwrap();
    graph.arcs[0].head = 99;
    let problem = Problem {
        rows: vec![],
        networks: vec![Network {
            id: 7,
            graph,
            multiplicity: Window {
                lower: 0.,
                upper: 1.,
            },
            arc_entries: vec![vec![]],
            route_entries: vec![],
        }],
    };
    let mut options = ModelTreeOptions::default();
    options.node_limit = 0;
    options.node.variable_boxes = vec![VariableBox {
        network: 7,
        variables: vec![ArcVariableBound {
            arc: 0,
            lower: 0.,
            upper: Some(1.),
        }],
    }];
    assert!(crate::master::tree::solve(problem, options).is_err());
}

#[test]
fn required_zero_box_needs_a_master_point_and_a_closing_gap() {
    for point in [None, Some((100., 0.))] {
        let mut node = positive_network(1., false).snapshot;
        node.state.oracles[0].calls = 0;
        node.state.pricing_master = point;
        let phase = node.state.oracles[0]
            .solver
            .phases
            .iter()
            .position(|p| p.exact())
            .unwrap();
        let duals = vec![0.; node.state.rhs.len()];
        // With these duals the implicit nonnegative box has lower estimate0,
        // but route quantity1 is required. Neither a missing nor a weak RMP
        // point may turn that estimate into an attaining zero selection.
        let _ = node.state.price(phase, &duals);
        assert_eq!(node.state.oracles[0].calls, 1);
    }
}

#[test]
fn positive_box_correction_uses_required_quantity_without_inventing_a_gradient() {
    let mut node = positive_network(1., false).snapshot;
    let definitions: Vec<VariableBox> = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/multi_network_native_positive_boxes.json"
    )))
    .unwrap();
    node.state.variable_boxes = variable_boxes::prepare(
        &definitions,
        node.state.oracles.iter().map(|o| {
            (
                o.network.id,
                o.network.graph.as_ref(),
                o.solver.forbidden_arcs.as_slice(),
            )
        }),
    )
    .unwrap();
    node.state.oracles[0].calls = 0;
    node.state.pricing_master = Some((1., 0.));
    let phase = node.state.oracles[0]
        .solver
        .phases
        .iter()
        .position(|p| p.exact())
        .unwrap();
    let duals = vec![0.; node.state.rhs.len()];
    let answer = node.state.price(phase, &duals).unwrap();
    assert!(answer.exact);
    assert_eq!(answer.minimum, 1.);
    assert!(answer.columns.is_empty());
    // Quantity1 is required and quantity2 is allowed. Multiplying a positive
    // per-pattern lower bound by2 would overstate the true optimum1.
    assert_eq!(node.state.correction, Some(1.));
    assert!(node.state.gradient.is_none());
    assert_eq!(node.state.oracles[0].calls, 0);
}

#[test]
fn an_empty_required_domain_cannot_become_feasible_from_a_box_bound() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 1.).unwrap(); // No completion to sink2.
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 1.;
    options.preprocessing.enabled = false;
    options.variable_boxes = vec![VariableBox {
        network: 7,
        variables: vec![ArcVariableBound {
            arc: 0,
            lower: 1.,
            upper: Some(1.),
        }],
    }];
    let result = solve(
        Problem {
            rows: vec![],
            networks: vec![Network {
                id: 7,
                graph,
                multiplicity: Window {
                    lower: 1.,
                    upper: 1.,
                },
                arc_entries: vec![vec![]],
                route_entries: vec![],
            }],
        },
        options,
    );
    if let Ok(output) = result {
        assert_ne!(output.master.status, cg::Status::Optimal);
    }
}

#[test]
fn fractional_multiplicity_keeps_outward_subnormal_box_corrections() {
    let mut output = positive_network(0., false);
    let oracle = &mut output.snapshot.state.oracles[0];
    let unit = f64::from_bits(1);
    for (quantity, expected) in [(0.5, -unit), (2.5, -3. * unit)] {
        oracle.network.multiplicity.upper = quantity;
        let admission = box_correction(oracle, -unit).unwrap();
        assert_eq!(admission.minimum, -unit);
        assert_eq!(admission.correction, expected);
        assert!(!admission.zero_quantity);
    }
    oracle.network.multiplicity.upper = 2.;
    assert_eq!(
        box_correction(oracle, -unit).unwrap().correction,
        -2. * unit
    );
}
