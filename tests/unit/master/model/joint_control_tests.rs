use super::*;

fn root() -> Output<2> {
    let (problem, registry) = format::Model::from_json(include_str!(
        "../../../fixtures/coupled/joint-interior-model.json"
    ))
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
