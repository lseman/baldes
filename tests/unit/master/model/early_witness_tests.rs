use super::*;

fn root() -> Output<2> {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 1.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 100.;
    options.preprocessing.enabled = false;
    solve(
        Problem {
            rows: vec![Row {
                sense: RowSense::Equal,
                rhs: 1.,
            }],
            networks: vec![Network {
                id: 7,
                graph,
                multiplicity: Window {
                    lower: 0.,
                    upper: 2.,
                },
                arc_entries: vec![vec![(0, 1.)], vec![]],
                route_entries: vec![],
            }],
        },
        options,
    )
    .unwrap()
}

#[test]
fn inherited_original_bound_closes_valid_integer_point_before_pricing() {
    for mode in [cg::BoundMode::Numerical, cg::BoundMode::Certified] {
        let mut output = root();
        // The unique route costs one and must cover this equality once.
        output.snapshot.warm.bound_mode = mode;
        output.snapshot.warm.bound = 1.;
        output.snapshot.warm.numerical_bound = 1.;
        output.snapshot.warm.certified_bound = 1.;
        let mut options = Options::default();
        options.integer_primal_termination = true;
        options.cg.stabilize = false;
        options.cg.bound_mode = mode;
        options.preprocessing.enabled = false;
        let resumed = output.snapshot.solve(options).unwrap();
        assert_eq!(resumed.master.status, cg::Status::Optimal);
        assert!(resumed.master.pruned_by_bound);
        assert_eq!(resumed.master.iterations, 1);
        assert_eq!(resumed.master.pricing_calls, 0);
        assert_eq!(resumed.oracle_calls, vec![(7, 0)]);
        assert_eq!(resumed.master.bound, 1.);
        assert_eq!(
            resumed.snapshot.integer_witness(1e-6).unwrap().unwrap().0,
            1.
        );
    }
}

#[test]
fn ordinary_root_lp_and_open_integer_gap_still_price() {
    for (enabled, bound) in [(false, 1.), (true, 0.5)] {
        let mut output = root();
        output.snapshot.warm.bound = bound;
        output.snapshot.warm.numerical_bound = bound;
        let mut options = Options::default();
        options.integer_primal_termination = enabled;
        options.cg.stabilize = false;
        options.preprocessing.enabled = false;
        let resumed = output.snapshot.solve(options).unwrap();
        assert!(resumed.master.pricing_calls > 0);
        assert_eq!(resumed.master.bound, 1.);
    }
}

#[test]
fn retained_original_primal_survives_master_invalidation_but_not_new_rows() {
    let mut output = root();
    let witness = output.snapshot.integer_witness(1e-6).unwrap().unwrap();
    output.snapshot.state.pricing_witness = Some(witness);
    let column = output
        .snapshot
        .warm
        .columns
        .iter()
        .position(|c| c.is_some())
        .unwrap();
    output.snapshot.master.set_cost(column, 2.).unwrap();
    assert_eq!(
        output.snapshot.integer_witness(1e-6).unwrap().unwrap().0,
        1.
    );
    let child = output
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
    assert!(child.integer_witness(1e-6).unwrap().is_none());
}

#[test]
fn artificial_or_unknown_metadata_never_becomes_an_original_primal() {
    let output = root();
    assert!(tree_adapter::replay_integer_witness(
        &output.snapshot.state,
        &output.snapshot.master,
        |_| Some(-1),
        1e-6
    )
    .unwrap()
    .is_none());
    assert!(tree_adapter::replay_integer_witness(
        &output.snapshot.state,
        &output.snapshot.master,
        |_| Some(i32::MAX),
        1e-6
    )
    .unwrap()
    .is_none());
}
