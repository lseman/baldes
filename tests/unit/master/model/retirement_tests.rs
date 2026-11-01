use super::*;

fn fixture() -> Output<2> {
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    graph.add_arc(0, 1, 1.).unwrap();
    graph.add_arc(1, 2, 1.).unwrap();
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.gap_enumeration = false;
    solve::solve_with_master_variables(
        Problem::set_partitioning(graph, 1.).unwrap(),
        vec![],
        options,
    )
    .unwrap()
}

#[test]
fn fixed_cover_row_retires_its_artificials_and_keeps_parent_replay() {
    let parent = fixture();
    let tag = parent.values.iter().position(|&q| q > 0.).unwrap();
    let fixed = parent.snapshot.fix_route(tag, 1.).unwrap();
    let child = fixed.preprocess().unwrap();
    let row = child.state.shared_offset;
    assert!(child.state.retired_rows.contains(&row));
    assert!(!parent.snapshot.state.retired_rows.contains(&row));
    assert_eq!(child.state.shared_rows[0].rhs, 1.);
    assert_eq!(child.state.rhs[row], 0.);
    assert!(child
        .master
        .column_entries(0)
        .unwrap()
        .iter()
        .all(|&(r, _)| r != row));
    for &column in &parent.snapshot.warm.artificials {
        if parent
            .snapshot
            .master
            .column_entries(column)
            .unwrap()
            .iter()
            .any(|&(r, _)| r == row)
        {
            assert_eq!(child.master.upper[column], 0.);
        }
    }
    let solved = child.solve(Options::default()).unwrap();
    assert_eq!(solved.master.lp, 2.);
    assert_eq!(solved.fixed_routes.len(), 1);
}

#[test]
fn missing_future_columns_do_not_make_a_cover_row_redundant() {
    let parent = fixture();
    assert!(parent.snapshot.state.retired_rows.is_empty());
    assert_eq!(parent.master.lp, 2.);
}

#[test]
fn nonzero_redundant_rhs_survives_backend_retirement_and_child_rows() {
    use crate::master::preprocessing::Bounds;
    let problem = Problem::<2> {
        networks: vec![],
        rows: vec![Row {
            sense: RowSense::LessEqual,
            rhs: 3.,
        }],
    };
    let variable = MasterVariable {
        id: 17,
        cost: -1.,
        integer: false,
        bounds: Bounds {
            lower: 0.,
            upper: 2.,
        },
        entries: vec![(0, 1.)],
    };
    let output =
        solve::solve_with_master_variables(problem, vec![variable], Options::default()).unwrap();
    let node = &output.snapshot;
    assert!(node.state.retired_rows.contains(&0));
    assert_eq!(node.master.rhs[0], 0.);
    assert_eq!(node.state.rhs[0], 3.);
    assert_eq!(output.master.lp, -2.);
    let restricted = node
        .with_master_variable_bounds(
            17,
            Bounds {
                lower: 0.,
                upper: 1.,
            },
        )
        .unwrap();
    assert_eq!(restricted.state.rhs[0], 3.);
    let child = restricted.solve(Options::default()).unwrap();
    assert_eq!(child.master.lp, -1.);
    assert_eq!(child.snapshot.state.rhs[0], 3.);
    assert_eq!(node.master.rhs[0], 0.);
    assert_eq!(node.state.master_variables[0].bounds.upper, 2.);
}

#[test]
fn nonlinear_retirement_requires_complete_coverage_and_preserves_fixed_coefficients() {
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    for (u, v, cost) in [(0, 1, 1.), (1, 2, 1.), (2, 3, 0.)] {
        graph.add_arc(u, v, cost).unwrap();
    }
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.gap_enumeration = false;
    let parent = solve::solve_with_master_variables(
        Problem::set_partitioning(graph, 1.).unwrap(),
        vec![],
        options,
    )
    .unwrap();
    let added = parent
        .snapshot
        .with_rank_one_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 3.,
            },
            &[RankOneCoefficients {
                network: 0,
                weights: vec![0, 1, 1, 0],
                denominator: 2,
            }],
        )
        .unwrap();
    let shared = added.state.shared_rows.len() - 1;
    let physical = added.state.shared_offset + shared;
    let restricted = added.preprocess().unwrap();
    assert!(!restricted.state.retired_rows.contains(&physical));
    let complete = added.enumerate(&EnumerationOptions::default()).unwrap();
    assert!(complete.enumeration_report().complete);
    let retired = complete.preprocess().unwrap();
    assert!(retired.state.retired_rows.contains(&physical));
    assert_eq!(retired.master.rhs[physical], 0.);
    assert_eq!(retired.state.rhs[physical], 3.);
    assert!(retired
        .warm
        .columns
        .iter()
        .flatten()
        .all(|column| column.entries.iter().all(|&(r, _)| r != physical)));
    // Original row data remains available to the fixed-solution reduction.
    let fixed = retired.fix_route(0, 1.).unwrap();
    assert_eq!(fixed.state.rhs[physical], 2.);
    assert_eq!(fixed.master.rhs[physical], 0.);
    let solved = fixed.solve(Options::default()).unwrap();
    assert_eq!(solved.master.lp, 2.);
    assert_eq!(solved.fixed_routes.len(), 1);
}

#[test]
fn retiring_user_rows_keeps_positive_lower_and_negative_upper_bound_encodings() {
    use crate::master::preprocessing::Bounds;
    for (bounds, cost, sense, rhs, expected) in [
        (
            Bounds {
                lower: 2.,
                upper: 4.,
            },
            1.,
            RowSense::LessEqual,
            5.,
            2.,
        ),
        (
            Bounds {
                lower: -4.,
                upper: -2.,
            },
            -1.,
            RowSense::GreaterEqual,
            -5.,
            -2.,
        ),
    ] {
        let problem = Problem::<2> {
            networks: vec![],
            rows: vec![Row { sense, rhs }],
        };
        let variable = MasterVariable {
            id: 17,
            cost,
            integer: false,
            bounds,
            entries: vec![(0, 1.)],
        };
        let solved =
            solve::solve_with_master_variables(problem, vec![variable], Options::default())
                .unwrap();
        assert!(solved.snapshot.state.retired_rows.contains(&0));
        assert!(!solved.snapshot.state.master_variable_bound_rows.is_empty());
        assert!(solved
            .snapshot
            .state
            .retired_rows
            .is_disjoint(&solved.snapshot.state.master_variable_bound_rows));
        assert_eq!(solved.master_values, vec![(17, expected)]);
        assert_eq!(solved.master.lp, 2.);
        let resumed = solved.snapshot.solve(Options::default()).unwrap();
        assert_eq!(resumed.master_values, vec![(17, expected)]);
        assert_eq!(resumed.master.lp, 2.);
    }
}

#[test]
fn inferred_network_quantity_does_not_remove_its_only_master_encoding() {
    let mut graph = Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, -1.).unwrap();
    let problem = Problem {
        rows: vec![Row {
            sense: RowSense::LessEqual,
            rhs: 2.,
        }],
        networks: vec![Network {
            id: 7,
            graph,
            multiplicity: Window::new(0., 5.).unwrap(),
            arc_entries: vec![vec![]],
            route_entries: vec![(0, 1.)],
            activity_entries: vec![],
        }],
    };
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.gap_enumeration = false;
    let solved = solve::solve_with_master_variables(problem, vec![], options).unwrap();
    let row = solved.snapshot.state.shared_offset;
    assert!(!solved.snapshot.state.retired_rows.contains(&row));
    assert_eq!(solved.master.lp, -2.);
    assert_eq!(
        solved.snapshot.state.oracles[0].network.multiplicity.upper,
        5.
    );
    assert_eq!(
        solved.snapshot.state.oracles[0].proved_multiplicity_upper,
        Some(2.)
    );
}

#[test]
fn clearing_the_last_coefficient_reoptimizes_old_positive_column_activity() {
    let quantity = 2_000_000_000_001.;
    let mut master = ClpMaster::new().unwrap();
    let row = master.add_row(RowSense::Equal, quantity, &[]).unwrap();
    let column = master.add_column(1., &[(row, 1.)]).unwrap();
    master.set_upper(column, quantity).unwrap();
    assert!(master.solve());
    assert_eq!(master.values[column], quantity);
    let iterations = master.simplex_iterations;
    master.set_upper(column, 0.).unwrap();
    master.set_rhs(&[0.]).unwrap();
    master.set_coefficient(row, column, 0.).unwrap();
    assert_eq!(master.simplex_iterations, iterations);
    assert!(master.solve());
    assert_eq!(master.values[column], 0.);
    assert_eq!(master.objective, 0.);
    // The same bound-only model must optimize costs, not merely zero values.
    master.set_cost(column, -1.).unwrap();
    master.set_upper(column, 2.).unwrap();
    assert!(master.solve());
    assert_eq!(master.values[column], 2.);
    assert_eq!(master.objective, -2.);
}
