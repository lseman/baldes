use super::*;

fn root() -> Output<2> {
    let mut source: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/joint-interior-model.json"
    ))
    .unwrap();
    source["networks"][0]["resources"] = serde_json::json!([
        {"kind":"disposable","bounds":[[0.,0.],[0.,10.]]}
    ]);
    let (problem, registry) = format::Model::from_json(&source.to_string())
        .unwrap()
        .compile::<2>()
        .unwrap();
    let mut options = Options::default();
    options.integer_primal_termination = true;
    options.cg.stabilize = false;
    registry.configure_pricing(&mut options);
    solve(problem, options).unwrap()
}

#[test]
fn joint_proposal_witness_requires_exact_whole_model_rows_and_residual_caps() {
    let output = root();
    let mut state = output.snapshot.state.fork_state();
    let tag = |values| {
        state
            .routes
            .iter()
            .position(|route| route.route.coupled_values == values)
            .unwrap()
    };
    let tags = [
        tag(vec![(0, 0.), (1, 0.)]) as i32,
        tag(vec![(0, 8.), (1, 8.)]) as i32,
    ];
    let interior = tag(vec![(0, 4.), (1, 4.)]);
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    for _ in 0..2 {
        let column = master.add_column(999., &[(0, 1.)]).unwrap();
        master.set_upper(column, 0.5).unwrap();
    }
    assert!(master.solve());
    let metadata = |column| tags.get(column).copied();
    let witness = repair_witness(&state, &master, metadata, 1e-7)
        .unwrap()
        .unwrap();
    assert_eq!(witness.0, -10.);
    assert_eq!(
        witness.1[0].route.route.coupled_values,
        vec![(0, 4.), (1, 4.)]
    );
    assert!(
        !output
            .snapshot
            .enumerate(&EnumerationOptions::default())
            .unwrap()
            .enumeration_report()
            .complete
    );
    state.route_upper_bounds.insert(interior, 0.);
    assert!(repair_witness(&state, &master, metadata, 1e-7)
        .unwrap()
        .is_none());
    state.route_upper_bounds.clear();
    state.shared_rows[0].rhs = 9.;
    let changed = repair_witness(&state, &master, metadata, 1e-7)
        .unwrap()
        .unwrap();
    assert_eq!(
        changed.1[0].route.route.coupled_values,
        vec![(0, 4.5), (1, 4.5)]
    );
    state.shared_rows[0].rhs = 17.;
    assert!(repair_witness(&state, &master, metadata, 1e-7)
        .unwrap()
        .is_none());
    state.shared_rows[0].rhs = 8.;
    state.universe = Some(vec![tags[0] as usize, tags[1] as usize]);
    assert!(repair_witness(&state, &master, metadata, 1e-7)
        .unwrap()
        .is_none());
}

#[test]
fn joint_control_targets_share_work_iterations_and_deadlines() {
    let output = root();
    let mut state = output.snapshot.state.fork_state();
    let oracle = &state.oracles[0];
    let mut work = 1_000_000;
    let mut iterations = 100_000;
    let priced = path(
        oracle,
        state.shared_offset,
        &vec![0.; state.rhs.len()],
        &[0],
        &mut work,
        &mut iterations,
        state.deadline,
    )
    .unwrap()
    .unwrap();
    let initial = priced
        .route
        .as_ref()
        .unwrap()
        .clone()
        .with_paired_controls(&oracle.network.graph, sorted(priced.greedy.clone()))
        .unwrap();
    assert!(joint_controls(
        &state,
        oracle,
        &[0],
        &priced,
        &initial,
        1.,
        (&mut 0, &mut iterations),
    )
    .is_none());
    assert!(joint_controls(
        &state,
        oracle,
        &[0],
        &priced,
        &initial,
        1.,
        (&mut work, &mut 0),
    )
    .is_none());
    let controls = joint_controls(
        &state,
        oracle,
        &[0],
        &priced,
        &initial,
        1.,
        (&mut work, &mut iterations),
    )
    .unwrap();
    let route = initial
        .clone()
        .with_paired_controls(&oracle.network.graph, sorted(controls))
        .unwrap();
    assert_eq!(route.coupled_values, vec![(0, 4.), (1, 4.)]);
    state.deadline = Instant::now();
    assert!(joint_controls(
        &state,
        &state.oracles[0],
        &[0],
        &priced,
        &initial,
        1.,
        (&mut work, &mut iterations),
    )
    .is_none());
}

#[test]
fn scheduled_repair_checks_caps_and_coverage_after_profile_changes() {
    let output = root();
    let mut state = output.snapshot.state.fork_state();
    let original = state
        .routes
        .iter()
        .position(|r| r.route.coupled_values == vec![(0, 4.), (1, 4.)])
        .unwrap();
    let mut changed = state.routes[original].as_ref().clone();
    changed.route.resource_profile.last_mut().unwrap()[0] = 5.;
    let copy = FixedRoute {
        route: changed.clone(),
        quantity: 1.,
    };
    let tag = state.routes.len();
    state.routes.push(Arc::new(changed));
    state.route_upper_bounds.insert(tag, 0.);
    assert!(!scheduled_identities_allowed(&state, &[copy.clone()]));
    // Fixed copies are outside this residual-cap check, including a consumed cap0.
    assert!(scheduled_identities_allowed(&state, &[]));
    state.route_upper_bounds.insert(tag, 1.);
    assert!(scheduled_identities_allowed(&state, &[copy.clone()]));
    assert!(!scheduled_identities_allowed(
        &state,
        &[copy.clone(), copy.clone()]
    ));
    state.universe = Some(vec![original]);
    assert!(!scheduled_identities_allowed(&state, &[copy.clone()]));
    state.universe.as_mut().unwrap().push(tag);
    assert!(scheduled_identities_allowed(&state, &[copy.clone()]));
    state.route_upper_bounds.insert(tag, f64::INFINITY);
    assert!(scheduled_identities_allowed(&state, &[copy]));
}

#[test]
fn paired_witness_allocations_split_counts_across_capped_identities() {
    let output = root();
    let mut state = output.snapshot.state.fork_state();
    let first = state
        .routes
        .iter()
        .position(|route| route.route.coupled_values == vec![(0, 0.), (1, 0.)])
        .unwrap();
    let second = state
        .routes
        .iter()
        .position(|route| route.route.coupled_values == vec![(0, 8.), (1, 8.)])
        .unwrap();
    assert_eq!(
        state.routes[first].route.arcs,
        state.routes[second].route.arcs
    );
    state.route_upper_bounds.insert(first, 1.);
    state.route_upper_bounds.insert(second, 1.);
    let choices = [first, second].map(|tag| (tag, 0.5, state.routes[tag].as_ref().clone()));

    let allocations = identity_allocations(&state, &choices, 2., 128);

    assert!(allocations.iter().any(|allocation| {
        allocation.len() == 2
            && allocation.iter().all(|route| route.quantity == 1.)
            && allocation
                .iter()
                .any(|route| route.route.route.coupled_values == vec![(0, 0.), (1, 0.)])
            && allocation
                .iter()
                .any(|route| route.route.route.coupled_values == vec![(0, 8.), (1, 8.)])
    }));
}

#[test]
fn paired_witness_repair_recovers_path_count_from_capped_identities() {
    let output = root();
    let mut state = output.snapshot.state.fork_state();
    state.oracles[0].total_multiplicity = Window::new(2., 2.).unwrap();
    let first = state
        .routes
        .iter()
        .position(|route| route.route.coupled_values == vec![(0, 0.), (1, 0.)])
        .unwrap();
    let second = state
        .routes
        .iter()
        .position(|route| route.route.coupled_values == vec![(0, 4.), (1, 4.)])
        .unwrap();
    let arcs = state.routes[first].route.arcs.clone();
    assert_eq!(arcs, state.routes[second].route.arcs);
    for (tag, route) in state.routes.iter().enumerate() {
        if route.network == state.routes[first].network && route.route.arcs == arcs {
            state.route_upper_bounds.insert(tag, 1.);
        }
    }

    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 2., &[]).unwrap();
    let tags = [first, second, first, second].map(|tag| tag as i32);
    for _ in 0..4 {
        let column = master.add_column(999., &[(0, 1.)]).unwrap();
        master.set_upper(column, 0.5).unwrap();
    }
    assert!(master.solve());

    let witness = repair_witness(&state, &master, |column| tags.get(column).copied(), 1e-7)
        .unwrap()
        .expect("the path count of two can use two identities capped at one each");

    assert_eq!(
        witness.1.iter().map(|route| route.quantity).sum::<f64>(),
        2.
    );
    assert!(witness
        .1
        .iter()
        .any(|route| route.route.route.coupled_values == vec![(0, 0.), (1, 0.)]));
    assert!(witness
        .1
        .iter()
        .any(|route| route.route.route.coupled_values == vec![(0, 4.), (1, 4.)]));
}

#[test]
fn joint_recovery_keeps_fixed_copies_and_signed_master_contributions() {
    use crate::graph::coupled::PairedControl;
    let output = root();
    for sign in [1_f64, -1.] {
        let mut state = output.snapshot.state.fork_state();
        let tag = |values| {
            state
                .routes
                .iter()
                .position(|r| r.route.coupled_values == values)
                .unwrap() as i32
        };
        let tags = [tag(vec![(0, 0.), (1, 0.)]), tag(vec![(0, 8.), (1, 8.)])];
        let mut fixed = state.routes[0].as_ref().clone();
        fixed.route = fixed
            .route
            .with_paired_controls(
                &state.oracles[0].network.graph,
                vec![
                    PairedControl {
                        resource: 0,
                        base: vec![2., 3.],
                    },
                    PairedControl {
                        resource: 1,
                        base: vec![2., 3.],
                    },
                ],
            )
            .unwrap();
        state.fixed_routes.push(FixedRoute {
            route: fixed,
            quantity: 1.,
        });
        state.oracles[0].total_multiplicity = Window::new(2., 2.).unwrap();
        state.shared_rows[0].rhs = 13. + sign;
        state
            .master_variables
            .push(master_variables::MasterVariable {
                id: 77,
                cost: 2.,
                bounds: crate::master::preprocessing::Bounds {
                    lower: sign.min(0.),
                    upper: sign.max(0.),
                },
                integer: false,
                entries: vec![(0, 2.)],
            });
        let mut master = ClpMaster::new().unwrap();
        master.add_row(RowSense::Equal, 1., &[]).unwrap();
        master.add_row(RowSense::Equal, 0.5, &[]).unwrap();
        for _ in 0..2 {
            let column = master.add_column(999., &[(0, 1.)]).unwrap();
            master.set_upper(column, 0.5).unwrap();
        }
        master.add_column(999., &[(1, 1.)]).unwrap();
        master.add_column(0., &[]).unwrap();
        assert!(master.solve());
        let metadata = |column| match column {
            0 | 1 => Some(tags[column]),
            2 => Some(if sign > 0. { -2 } else { -3 }),
            _ => None,
        };
        // Numerical artificial residuals do not become original source activity.
        master.values[3] = -1e-12;
        let witness = repair_witness(&state, &master, metadata, 1e-7)
            .unwrap()
            .unwrap();
        assert_eq!(witness.0, -20. + sign);
        assert_eq!(witness.2, vec![(77, sign * 0.5)]);
        assert_eq!(witness.1.len(), 2);
        assert_eq!(
            witness.1[0].route.route.coupled_values,
            vec![(0, 2.), (1, 2.)]
        );
        assert_eq!(
            witness.1[1].route.route.coupled_values,
            vec![(0, 4.5), (1, 4.5)]
        );
        for artifact in [-0.5, 0.5] {
            master.values[3] = artifact;
            assert!(repair_witness(&state, &master, metadata, 1e-7)
                .unwrap()
                .is_none());
        }
    }
}
