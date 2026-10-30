use super::*;
use baldes::master::{preprocessing::Bounds, tree::TreeStatus};
use model::{
    imported::{self, Column, Subproblem},
    MasterVariable,
};
fn bounds(lower: f64, upper: f64) -> Bounds {
    Bounds { lower, upper }
}
fn column(id: i32, cost: f64, value: f64) -> Column {
    Column {
        id,
        cost,
        bounds: bounds(0., f64::INFINITY),
        integer: true,
        entries: if value == 0. {
            vec![]
        } else {
            vec![(0, value)]
        },
    }
}
fn fixture(mode: BoundMode) -> imported::ImportedOutput<2> {
    let p = Problem {
        rows: vec![Row {
            sense: RowSense::GreaterEqual,
            rhs: 1.,
        }],
        networks: vec![],
    };
    let vars = vec![MasterVariable {
        id: 0,
        cost: 0.,
        bounds: bounds(0., f64::INFINITY),
        integer: false,
        entries: vec![(0, 1.)],
    }];
    let groups = vec![
        Subproblem {
            id: 7,
            variables: vec![0],
            multiplicity: bounds(0., 1.),
            columns: vec![column(0, 10., 1.)],
        },
        Subproblem {
            id: 42,
            variables: vec![0],
            multiplicity: bounds(0., 1.),
            columns: vec![column(0, 5., 1.)],
        },
    ];
    let mut opts = options();
    opts.cg.bound_mode = mode;
    imported::solve(p, vars, groups, opts).unwrap()
}

#[test]
fn imported_expansion_resets_old_certificates_and_derived_bounds_but_replays_qualified_fixing() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let mut opts = options();
        opts.cg.bound_mode = mode;
        let root = fixture(mode);
        optimal(&root.output, 5.);
        let restricted = root
            .snapshot()
            .unwrap()
            .with_column_bounds(42, 0, bounds(0., 0.))
            .unwrap()
            .solve(opts.clone())
            .unwrap();
        optimal(&restricted.output, 10.);
        assert_eq!(
            restricted
                .output
                .snapshot
                .preprocessing_report()
                .master_bounds[0]
                .1
                .upper,
            1.
        );
        let node = restricted.snapshot().unwrap();
        let mut zero_budget = opts.clone();
        zero_budget.cg.time_limit = std::time::Duration::ZERO;
        let pending = node
            .solve_with_columns(7, vec![column(1, -1., 4.)], zero_budget)
            .unwrap();
        assert_eq!(pending.output.master.status, Status::TimeLimit);
        assert_eq!(pending.output.master.bound, f64::NEG_INFINITY);
        assert_eq!(pending.output.master.certified_bound, f64::NEG_INFINITY);
        assert_eq!(pending.output.master.numerical_bound, f64::NEG_INFINITY);
        let expanded = node
            .solve_with_columns(7, vec![column(1, -1., 4.)], opts.clone())
            .unwrap();
        optimal(&expanded.output, -1.);
        assert_eq!(expanded.values.original, vec![(0, 4.)]);
        assert_eq!(
            expanded
                .values
                .columns
                .iter()
                .find(|v| v.subproblem == 42)
                .unwrap()
                .value,
            0.
        );
        assert_eq!(
            expanded
                .values
                .columns
                .iter()
                .find(|v| v.subproblem == 7 && v.column == 1)
                .unwrap()
                .value,
            1.
        );
        assert!(node.values(&expanded.output.master_values).is_err());
        let third = expanded
            .snapshot()
            .unwrap()
            .solve_with_columns(42, vec![column(1, -2., 8.)], opts.clone())
            .unwrap();
        optimal(&third.output, -3.);
        assert_eq!(third.values.original, vec![(0, 12.)]);
        optimal(&node.solve(opts.clone()).unwrap().output, 10.);
        optimal(&root.snapshot().unwrap().solve(opts).unwrap().output, 5.);
    }
}

#[test]
fn imported_expansion_keeps_original_column_and_multiplicity_restrictions_in_lp_and_mip() {
    let root = fixture(BoundMode::Certified);
    let mut opts = options();
    opts.cg.bound_mode = BoundMode::Certified;
    let node = root
        .snapshot()
        .unwrap()
        .with_column_bounds(42, 0, bounds(0., 0.))
        .unwrap()
        .with_variable_bounds(0, bounds(1., 1.))
        .unwrap();
    let expanded = node
        .solve_with_columns(7, vec![column(1, -1., 4.)], opts.clone())
        .unwrap();
    optimal(&expanded.output, -0.25);
    assert_eq!(expanded.values.original.len(), 1);
    assert_eq!(expanded.values.original[0].0, 0);
    assert!((expanded.values.original[0].1 - 1.).abs() < 1e-12);
    #[cfg(feature = "highs")]
    {
        let child = expanded.snapshot().unwrap();
        let mip = child
            .node()
            .solve_enumerated_mip(&Default::default())
            .unwrap()
            .unwrap();
        assert_eq!(
            mip.mip
                .solution
                .as_ref()
                .expect(&format!("{mip:?}"))
                .objective,
            10.
        );
        assert_eq!(
            child.values(&mip.master_values).unwrap().original,
            vec![(0, 1.)]
        );
    }
    let mut integer_policy = opts.clone();
    integer_policy.preprocessing.integer_domains = true;
    let rounded = node
        .solve_with_columns(7, vec![column(1, -1., 4.)], integer_policy.clone())
        .unwrap();
    optimal(&rounded.output, 10.);
    let reopened = rounded
        .snapshot()
        .unwrap()
        .solve_with_columns(42, vec![column(1, 0., -3.)], integer_policy)
        .unwrap();
    optimal(&reopened.output, -1.);
    assert_eq!(reopened.values.original, vec![(0, 1.)]);
    assert_eq!(
        reopened
            .values
            .columns
            .iter()
            .find(|v| v.subproblem == 7 && v.column == 1)
            .unwrap()
            .value,
        1.
    );
    let fixed = root
        .snapshot()
        .unwrap()
        .with_column_bounds(7, 0, bounds(1., 1.))
        .unwrap()
        .with_column_bounds(42, 0, bounds(0., 0.))
        .unwrap()
        .with_multiplicity(7, bounds(1., 1.))
        .unwrap();
    let expanded = fixed
        .solve_with_columns(7, vec![column(1, -1., 4.)], opts.clone())
        .unwrap();
    optimal(&expanded.output, 10.);
    assert_eq!(
        expanded
            .values
            .columns
            .iter()
            .find(|v| v.subproblem == 7 && v.column == 1)
            .unwrap()
            .value,
        0.
    );
    let disabled = root
        .snapshot()
        .unwrap()
        .with_multiplicity(7, bounds(0., 0.))
        .unwrap();
    let expanded = disabled
        .solve_with_columns(7, vec![column(1, -1., 4.)], opts)
        .unwrap();
    optimal(&expanded.output, 5.);
}

#[test]
fn imported_expansion_reprices_routing_and_recovers_from_an_empty_import_domain() {
    for routing in [false, true] {
        let p = Problem::<2> {
            rows: vec![Row {
                sense: RowSense::GreaterEqual,
                rhs: 1.,
            }],
            networks: if routing {
                vec![network(99, &[(5., &[0])], 1)]
            } else {
                vec![]
            },
        };
        let vars = vec![MasterVariable {
            id: 0,
            cost: 0.,
            bounds: bounds(0., f64::INFINITY),
            integer: false,
            entries: vec![(0, 1.)],
        }];
        let groups = vec![Subproblem {
            id: 7,
            variables: vec![0],
            multiplicity: bounds(0., 1.),
            columns: vec![],
        }];
        let root = imported::solve(p, vars, groups, options()).unwrap();
        if routing {
            optimal(&root.output, 5.);
        } else {
            assert!(
                root.output
                    .snapshot
                    .preprocessing_report()
                    .propagation
                    .infeasible
            );
        }
        let expanded = root
            .snapshot()
            .unwrap()
            .solve_with_columns(7, vec![column(0, 1., 1.)], options())
            .unwrap();
        optimal(&expanded.output, 1.);
        assert_eq!(expanded.values.original.len(), 1);
        assert_eq!(expanded.values.original[0].0, 0);
        assert!((expanded.values.original[0].1 - 1.).abs() < 1e-12);
        if routing {
            assert!(!expanded.output.oracle_calls.is_empty());
        } else {
            let tree = expanded
                .snapshot()
                .unwrap()
                .node()
                .solve_tree(Default::default())
                .unwrap();
            assert_eq!(tree.status, TreeStatus::Optimal);
            assert_eq!(tree.incumbent, 1.);
        }
    }
}

#[test]
fn rejected_extensions_leave_the_old_import_domain_and_restrictions_unchanged() {
    let root = fixture(BoundMode::Numerical);
    let node = root
        .snapshot()
        .unwrap()
        .with_column_bounds(42, 0, bounds(0., 0.))
        .unwrap();
    assert!(node
        .solve_with_columns(7, vec![column(0, -1., 4.)], options())
        .is_err());
    assert!(node.solve_with_columns(999, vec![], options()).is_err());
    let mut bad = column(1, 1., 1.);
    bad.entries.push((999, 1.));
    assert!(node.solve_with_columns(7, vec![bad], options()).is_err());
    optimal(
        &node
            .solve_with_columns(7, vec![], options())
            .unwrap()
            .output,
        10.,
    );
    optimal(&node.solve(options()).unwrap().output, 10.);
    optimal(
        &root.snapshot().unwrap().solve(options()).unwrap().output,
        5.,
    );
}

#[test]
fn imported_expansion_preserves_explicit_routing_multiplicity_and_parent_isolation() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let mut opts = options();
        opts.cg.bound_mode = mode;
        let root = imported::solve(
            Problem::<2> {
                rows: vec![Row {
                    sense: RowSense::GreaterEqual,
                    rhs: 1.,
                }],
                networks: vec![network(99, &[(5., &[0])], 1)],
            },
            vec![MasterVariable {
                id: 0,
                cost: 0.,
                bounds: bounds(0., f64::INFINITY),
                integer: false,
                entries: vec![(0, 1.)],
            }],
            vec![Subproblem {
                id: 7,
                variables: vec![0],
                multiplicity: bounds(0., 1.),
                columns: vec![],
            }],
            opts.clone(),
        )
        .unwrap();
        optimal(&root.output, 5.);
        let parent = root.snapshot().unwrap();
        assert!(parent.with_routing_multiplicity(7, bounds(0., 1.)).is_err());
        assert!(parent
            .with_routing_multiplicity(99, bounds(-1., 1.))
            .is_err());
        assert!(parent.fix_route(usize::MAX, 1.).is_err());
        assert!(parent.fix_route(0, 0.5).is_err());
        let fixed = parent
            .fix_route(0, 1.)
            .unwrap()
            .with_routing_multiplicity(99, bounds(1., 1.))
            .unwrap();
        let expanded_fixed = fixed
            .solve_with_columns(7, vec![column(0, -2., 1.)], opts.clone())
            .unwrap();
        optimal(&expanded_fixed.output, 3.);
        assert_eq!(expanded_fixed.output.fixed_routes.len(), 1);
        assert_eq!(expanded_fixed.output.fixed_routes[0].quantity, 1.);
        assert_eq!(
            expanded_fixed.output.fixed_routes[0].route.route.arcs,
            root.output.routes[0].route.arcs
        );
        let again = expanded_fixed
            .snapshot()
            .unwrap()
            .solve_with_columns(7, vec![column(1, -3., 1.)], opts.clone())
            .unwrap();
        optimal(&again.output, 2.);
        assert_eq!(again.output.fixed_routes.len(), 1);
        #[cfg(feature = "highs")]
        {
            let complete = again
                .snapshot()
                .unwrap()
                .node()
                .enumerate(&Default::default())
                .unwrap();
            let mip = complete
                .solve_enumerated_mip(&Default::default())
                .unwrap()
                .unwrap();
            assert_eq!(mip.mip.solution.as_ref().unwrap().objective, 2.);
        }
        let required = parent
            .with_routing_multiplicity(99, bounds(1., 1.))
            .unwrap();
        let expanded = required
            .solve_with_columns(7, vec![column(0, 1., 1.)], opts.clone())
            .unwrap();
        optimal(&expanded.output, 5.);
        let repeated = expanded
            .snapshot()
            .unwrap()
            .solve_with_columns(7, vec![column(1, -2., 1.)], opts.clone())
            .unwrap();
        optimal(&repeated.output, 3.);
        let disabled = parent
            .with_routing_multiplicity(99, bounds(0., 0.))
            .unwrap();
        let recovered = disabled
            .solve_with_columns(7, vec![column(0, 1., 1.)], opts.clone())
            .unwrap();
        optimal(&recovered.output, 1.);
        optimal(&parent.solve(opts).unwrap().output, 5.);
    }
}

#[test]
fn imported_shared_rows_restrict_future_vectors_and_reject_internal_ids() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let root = fixture(mode);
        let mut opts = options();
        opts.cg.bound_mode = mode;
        let parent = root.snapshot().unwrap();
        // Generated imported-column ID 1 is not an original coordinate ID.
        assert!(parent
            .with_row(
                Row {
                    sense: RowSense::Equal,
                    rhs: 0.
                },
                &[],
                &[(1, 1.)]
            )
            .is_err());
        assert!(parent
            .with_row(
                Row {
                    sense: RowSense::Equal,
                    rhs: 0.
                },
                &[],
                &[(0, 1.), (0, 2.)]
            )
            .is_err());
        assert!(parent
            .with_row(
                Row {
                    sense: RowSense::Equal,
                    rhs: 0.
                },
                &[],
                &[(0, f64::NAN)]
            )
            .is_err());
        for (sense, rhs, coefficient) in [
            (RowSense::LessEqual, 1., 1.),
            (RowSense::GreaterEqual, -1., -1.),
            (RowSense::Equal, 1., 1.),
        ] {
            let node = parent
                .with_row(Row { sense, rhs }, &[], &[(0, coefficient)])
                .unwrap();
            let expanded = node
                .solve_with_columns(7, vec![column(1, -4., 4.)], opts.clone())
                .unwrap();
            optimal(&expanded.output, -1.);
            assert_eq!(expanded.values.original.len(), 1);
            assert_eq!(expanded.values.original[0].0, 0);
            assert!((expanded.values.original[0].1 - 1.).abs() < 1e-12);
            let again = expanded
                .snapshot()
                .unwrap()
                .solve_with_columns(42, vec![column(1, -8., 4.)], opts.clone())
                .unwrap();
            optimal(&again.output, -2.);
            assert_eq!(again.values.original.len(), 1);
            assert_eq!(again.values.original[0].0, 0);
            assert!((again.values.original[0].1 - 1.).abs() < 1e-12);
            optimal(&node.solve(opts.clone()).unwrap().output, 5.);
        }
        optimal(&parent.solve(opts).unwrap().output, 5.);
    }
}

#[test]
fn imported_shared_routing_rows_replay_fixed_contributions_in_both_operation_orders() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let mut opts = options();
        opts.cg.bound_mode = mode;
        let root = imported::solve(
            Problem::<2> {
                rows: vec![Row {
                    sense: RowSense::GreaterEqual,
                    rhs: 1.,
                }],
                networks: vec![network(99, &[(5., &[0])], 1)],
            },
            vec![MasterVariable {
                id: 0,
                cost: 0.,
                bounds: bounds(0., f64::INFINITY),
                integer: false,
                entries: vec![(0, 1.)],
            }],
            vec![Subproblem {
                id: 7,
                variables: vec![0],
                multiplicity: bounds(0., 1.),
                columns: vec![],
            }],
            opts.clone(),
        )
        .unwrap();
        let parent = root.snapshot().unwrap();
        let routing = [RowCoefficients {
            network: 99,
            arcs: vec![0.5, 0.],
            per_route: 0.5,
        }];
        let row = Row {
            sense: RowSense::LessEqual,
            rhs: 1.,
        };
        for fix_first in [false, true] {
            let child = if fix_first {
                parent
                    .fix_route(0, 1.)
                    .unwrap()
                    .with_row(row.clone(), &routing, &[(0, 1.)])
                    .unwrap()
            } else {
                parent
                    .with_row(row.clone(), &routing, &[(0, 1.)])
                    .unwrap()
                    .fix_route(0, 1.)
                    .unwrap()
            };
            let expanded = child
                .solve_with_columns(7, vec![column(0, -2., 1.)], opts.clone())
                .unwrap();
            optimal(&expanded.output, 5.);
            assert_eq!(expanded.values.original, vec![(0, 0.)]);
            assert_eq!(expanded.output.fixed_routes.len(), 1);
            let again = expanded
                .snapshot()
                .unwrap()
                .solve_with_columns(7, vec![column(1, -3., 1.)], opts.clone())
                .unwrap();
            optimal(&again.output, 5.);
            #[cfg(feature = "highs")]
            {
                let node = again.snapshot().unwrap();
                let complete = node.node().enumerate(&Default::default()).unwrap();
                let mip = complete
                    .solve_enumerated_mip(&Default::default())
                    .unwrap()
                    .unwrap();
                assert_eq!(mip.mip.solution.as_ref().unwrap().objective, 5.);
            }
        }
        optimal(&parent.solve(opts).unwrap().output, 5.);
    }
}

#[test]
fn imported_presence_rows_survive_expansion_and_replay_fixed_nonlinear_contributions() {
    use baldes::cuts::soft_conflicts::{Mode, PresenceRow};
    use model::PresenceCoefficients;
    for bound_mode in [BoundMode::Numerical, BoundMode::Certified] {
        for mode in [Mode::PairCount, Mode::AnyPair] {
            let mut opts = options();
            opts.cg.bound_mode = bound_mode;
            let root = imported::solve(
                Problem::<2> {
                    rows: vec![Row {
                        sense: RowSense::GreaterEqual,
                        rhs: 1.,
                    }],
                    networks: vec![network(99, &[(5., &[0])], 1)],
                },
                vec![MasterVariable {
                    id: 0,
                    cost: 0.,
                    bounds: bounds(0., f64::INFINITY),
                    integer: false,
                    entries: vec![(0, 1.)],
                }],
                vec![Subproblem {
                    id: 7,
                    variables: vec![0],
                    multiplicity: bounds(0., 1.),
                    columns: vec![],
                }],
                opts.clone(),
            )
            .unwrap();
            let parent = root.snapshot().unwrap();
            let routing = [PresenceCoefficients {
                network: 99,
                row: PresenceRow::new(2, vec![(0, 1), (0, 1)], mode).unwrap(),
                arc_events: vec![vec![0], vec![1]],
            }];
            let coefficient = if mode == Mode::PairCount { 2. } else { 1. };
            let row = Row {
                sense: RowSense::LessEqual,
                rhs: coefficient,
            };
            assert!(parent
                .with_presence_row(row.clone(), &routing, &[(1, 1.)])
                .is_err());
            let unrestricted = parent
                .with_presence_row(row.clone(), &routing, &[(0, 1.)])
                .unwrap()
                .solve_with_columns(7, vec![column(0, -2., 1.)], opts.clone())
                .unwrap();
            optimal(&unrestricted.output, -2.);
            for fix_first in [false, true] {
                let child = if fix_first {
                    parent
                        .fix_route(0, 1.)
                        .unwrap()
                        .with_presence_row(row.clone(), &routing, &[(0, 1.)])
                        .unwrap()
                } else {
                    parent
                        .with_presence_row(row.clone(), &routing, &[(0, 1.)])
                        .unwrap()
                        .fix_route(0, 1.)
                        .unwrap()
                };
                let expanded = child
                    .solve_with_columns(7, vec![column(0, -2., 1.)], opts.clone())
                    .unwrap();
                optimal(&expanded.output, 5.);
                assert_eq!(expanded.values.original, vec![(0, 0.)]);
                assert_eq!(expanded.output.fixed_routes.len(), 1);
                let again = expanded
                    .snapshot()
                    .unwrap()
                    .solve_with_columns(7, vec![column(1, -3., 1.)], opts.clone())
                    .unwrap();
                optimal(&again.output, 5.);
                let streamed = again
                    .snapshot()
                    .unwrap()
                    .begin_import(7)
                    .unwrap()
                    .collect(&imported::ImportOptions::default(), |_, _| {
                        Ok(imported::ColumnBatch {
                            columns: vec![column(2, -4., 1.)],
                            complete: true,
                        })
                    })
                    .unwrap()
                    .session
                    .solve(opts.clone())
                    .unwrap();
                optimal(&streamed.output, 5.);
                assert_eq!(streamed.output.fixed_routes.len(), 1);

                #[cfg(feature = "highs")]
                {
                    let snapshot = again.snapshot().unwrap();
                    let complete = snapshot.node().enumerate(&Default::default()).unwrap();
                    let mip = complete
                        .solve_enumerated_mip(&Default::default())
                        .unwrap()
                        .unwrap();
                    assert_eq!(mip.mip.solution.as_ref().unwrap().objective, 5.);
                }
            }
            optimal(&parent.solve(opts).unwrap().output, 5.);
        }
    }
}

#[test]
fn imported_source_requires_completion_and_resumes_without_inheriting_finite_domain_proofs() {
    use imported::{ColumnBatch, ImportOptions, ImportStatus};
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let root = fixture(mode);
        let mut opts = options();
        opts.cg.bound_mode = mode;
        let node = root
            .snapshot()
            .unwrap()
            .with_column_bounds(42, 0, bounds(0., 0.))
            .unwrap();
        let session = node.begin_import(7).unwrap();
        assert!(!session.is_complete());
        assert!(session.solve(opts.clone()).is_err());
        let budget = ImportOptions {
            max_batches: 1,
            time_limit: std::time::Duration::from_secs(10),
        };
        let first = session
            .collect(&budget, |group, remaining| {
                assert_eq!(group.id, 7);
                assert_eq!(group.columns.len(), 1);
                assert!(!remaining.is_zero());
                Ok(ColumnBatch {
                    columns: vec![column(1, -2., 2.)],
                    complete: false,
                })
            })
            .unwrap();
        assert_eq!(first.status, ImportStatus::BatchLimit);
        assert_eq!(first.batches, 1);
        assert_eq!(first.added_columns, 1);
        assert!(first.session.solve(opts.clone()).is_err());
        assert_eq!(session.subproblem().columns.len(), 1);
        let done = first
            .session
            .collect(&budget, |group, _| {
                assert_eq!(
                    group.columns.iter().map(|c| c.id).collect::<Vec<_>>(),
                    vec![0, 1]
                );
                Ok(ColumnBatch {
                    columns: vec![column(2, -4., 4.)],
                    complete: true,
                })
            })
            .unwrap();
        assert_eq!(done.status, ImportStatus::Complete);
        assert!(done.session.is_complete());
        let solved = done.session.solve(opts.clone()).unwrap();
        optimal(&solved.output, -4.);
        assert_eq!(solved.values.original, vec![(0, 4.)]);
        assert_eq!(
            solved
                .values
                .columns
                .iter()
                .find(|c| c.subproblem == 42)
                .unwrap()
                .value,
            0.
        );
        let already = done
            .session
            .collect(
                &ImportOptions {
                    max_batches: 0,
                    time_limit: std::time::Duration::ZERO,
                },
                |_, _| panic!("complete source called again"),
            )
            .unwrap();
        assert_eq!(already.status, ImportStatus::Complete);
        assert_eq!(already.batches, 0);
        optimal(&node.solve(opts.clone()).unwrap().output, 10.);
        optimal(&root.snapshot().unwrap().solve(opts).unwrap().output, 5.);
    }
}

#[test]
fn imported_source_budget_errors_and_invalid_completion_leave_the_session_reusable() {
    use imported::{ColumnBatch, ImportOptions, ImportStatus};
    let root = fixture(BoundMode::Numerical);
    let node = root.snapshot().unwrap();
    assert!(node.begin_import(999).is_err());
    let source = node.begin_import(7).unwrap();
    let stopped = source
        .collect(
            &ImportOptions {
                max_batches: 2,
                time_limit: std::time::Duration::ZERO,
            },
            |_, _| panic!("zero-time callback"),
        )
        .unwrap();
    assert_eq!(stopped.status, ImportStatus::TimeLimit);
    assert!(stopped.session.solve(options()).is_err());
    let stopped = source
        .collect(
            &ImportOptions {
                max_batches: 0,
                time_limit: std::time::Duration::from_secs(1),
            },
            |_, _| panic!("zero-batch callback"),
        )
        .unwrap();
    assert_eq!(stopped.status, ImportStatus::BatchLimit);
    let budget = ImportOptions::default();
    let mut attempts = 0;
    assert!(source
        .collect(&budget, |known, _| {
            attempts += 1;
            assert_eq!(known.columns.len(), attempts);
            Ok(ColumnBatch {
                columns: vec![column(1, -2., 2.)],
                complete: attempts == 2,
            })
        })
        .is_err());
    assert_eq!(attempts, 2);
    assert_eq!(source.subproblem().columns.len(), 1);
    let empty = source
        .collect(
            &ImportOptions {
                max_batches: 2,
                time_limit: std::time::Duration::from_secs(1),
            },
            |_, _| {
                Ok(ColumnBatch {
                    columns: vec![],
                    complete: false,
                })
            },
        )
        .unwrap();
    assert_eq!(empty.status, ImportStatus::BatchLimit);
    assert_eq!(empty.batches, 2);
    assert!(empty.session.solve(options()).is_err());

    assert!(source
        .collect(&budget, |_, _| Ok(ColumnBatch {
            columns: vec![column(0, -9., 1.)],
            complete: true
        }))
        .is_err());
    let mut bad = column(1, -9., 1.);
    bad.entries.push((999, 1.));
    assert!(source
        .collect(&budget, |_, _| Ok(ColumnBatch {
            columns: vec![bad.clone()],
            complete: true
        }))
        .is_err());
    let mut error = Some(node.with_variable_bounds(999, bounds(0., 1.)).unwrap_err());
    assert!(source
        .collect(&budget, |_, _| Err(error.take().unwrap()))
        .is_err());
    assert_eq!(source.subproblem().columns.len(), 1);
    assert!(!source.is_complete());
    let complete = source
        .collect(&budget, |_, _| {
            Ok(ColumnBatch {
                columns: vec![],
                complete: true,
            })
        })
        .unwrap();
    assert_eq!(complete.status, ImportStatus::Complete);
    optimal(&complete.session.solve(options()).unwrap().output, 5.);
}

#[test]
fn imported_pricing_points_use_original_links_and_keep_external_completion_unknown() {
    use imported::{ColumnBatch, ImportOptions};
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let root = fixture(mode);
        let mut opts = options();
        opts.cg.bound_mode = mode;
        let node = root
            .snapshot()
            .unwrap()
            .with_column_bounds(42, 0, bounds(0., 0.))
            .unwrap();
        let session = node.begin_import(7).unwrap();
        let prices = session.pricing(opts.clone()).unwrap();
        assert_eq!(prices.subproblem().id, 7);
        let costs = prices.costs().unwrap();
        assert_eq!(costs.coordinates.len(), 1);
        let predicted = -1. + costs.per_column + 4. * costs.coordinates[0].1;
        assert!((predicted - prices.reduced_cost(column(1, -1., 4.)).unwrap()).abs() < 1e-8);

        assert!(prices.reduced_cost(column(1, -1., 4.)).unwrap() < -1.);
        assert!(prices.reduced_cost(column(0, -1., 4.)).is_err());
        let mut forced = column(1, -1., 4.);
        forced.bounds.lower = 1.;
        assert!(prices.reduced_cost(forced).is_err());
        assert!(session.solve(opts.clone()).is_err());
        let pending = session
            .collect(
                &ImportOptions {
                    max_batches: 1,
                    ..Default::default()
                },
                |_, _| {
                    Ok(ColumnBatch {
                        columns: vec![column(1, -1., 4.)],
                        complete: false,
                    })
                },
            )
            .unwrap();
        let updated = pending.session.pricing(opts.clone()).unwrap();
        assert!(updated.reduced_cost(column(2, -2., 4.)).unwrap() < 0.);
        assert!(pending.session.solve(opts.clone()).is_err());
        let complete = pending
            .session
            .collect(&ImportOptions::default(), |_, _| {
                Ok(ColumnBatch {
                    columns: vec![column(2, -2., 4.)],
                    complete: true,
                })
            })
            .unwrap();
        let solved = complete.session.solve(opts.clone()).unwrap();
        optimal(&solved.output, -2.);
        assert_eq!(solved.values.original, vec![(0, 4.)]);
        let mut no_time = opts.clone();
        no_time.cg.time_limit = std::time::Duration::ZERO;
        assert!(session.pricing(no_time).is_err());
        optimal(&node.solve(opts).unwrap().output, 10.);
    }
}

#[test]
fn imported_pricing_includes_explicit_multiplicity_and_original_objective_terms() {
    let root = imported::solve(
        Problem::<2> {
            rows: vec![],
            networks: vec![],
        },
        vec![MasterVariable {
            id: 0,
            cost: 3.,
            bounds: bounds(0., f64::INFINITY),
            integer: false,
            entries: vec![],
        }],
        vec![Subproblem {
            id: 7,
            variables: vec![0],
            multiplicity: bounds(0., 2.),
            columns: vec![column(0, 10., 1.)],
        }],
        options(),
    )
    .unwrap();
    let required = root
        .snapshot()
        .unwrap()
        .with_multiplicity(7, bounds(1., 1.))
        .unwrap();
    let session = required.begin_import(7).unwrap();
    let prices = session.pricing(options()).unwrap();
    let costs = prices.costs().unwrap();
    assert_eq!(costs.coordinates, vec![(0, 3.)]);
    assert_eq!(costs.per_column, -13.);
    let enclosed = prices
        .reduced_cost_bounds(column(1, 1., 1.))
        .unwrap()
        .unwrap();
    assert_eq!(enclosed.lower, -9.);
    assert_eq!(enclosed.upper, -9.);

    // Existing cost 10+3*x = 13; candidate 1+3*x = 4, each one copy.
    assert!((prices.reduced_cost(column(1, 1., 1.)).unwrap() + 9.).abs() < 1e-8);
    let complete = session
        .collect(&imported::ImportOptions::default(), |_, _| {
            Ok(imported::ColumnBatch {
                columns: vec![column(1, 1., 1.)],
                complete: true,
            })
        })
        .unwrap();
    optimal(&complete.session.solve(options()).unwrap().output, 4.);
    optimal(&required.solve(options()).unwrap().output, 13.);
}

#[test]
fn imported_pricing_does_not_turn_an_infeasible_restricted_domain_into_a_global_claim() {
    let root = imported::solve(
        Problem::<2> {
            rows: vec![Row {
                sense: RowSense::GreaterEqual,
                rhs: 1.,
            }],
            networks: vec![],
        },
        vec![MasterVariable {
            id: 0,
            cost: 0.,
            bounds: bounds(0., f64::INFINITY),
            integer: false,
            entries: vec![(0, 1.)],
        }],
        vec![Subproblem {
            id: 7,
            variables: vec![0],
            multiplicity: bounds(0., 1.),
            columns: vec![],
        }],
        options(),
    )
    .unwrap();
    let session = root.snapshot().unwrap().begin_import(7).unwrap();
    assert!(session.pricing(options()).is_err());
    assert!(!session.is_complete());
    assert!(session.solve(options()).is_err());
    let complete = session
        .collect(&imported::ImportOptions::default(), |_, _| {
            Ok(imported::ColumnBatch {
                columns: vec![column(0, 1., 1.)],
                complete: true,
            })
        })
        .unwrap();
    optimal(&complete.session.solve(options()).unwrap().output, 1.);
}

#[test]
fn imported_phase_one_prices_restore_missing_columns_without_claiming_global_infeasibility() {
    use imported::{ColumnBatch, ImportOptions, ImportPricingKind};
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let mut opts = options();
        opts.cg.bound_mode = mode;
        let root = imported::solve(
            Problem::<2> {
                rows: vec![Row {
                    sense: RowSense::GreaterEqual,
                    rhs: 1.,
                }],
                networks: vec![],
            },
            vec![MasterVariable {
                id: 0,
                cost: 7.,
                bounds: bounds(0., f64::INFINITY),
                integer: false,
                entries: vec![(0, 1.)],
            }],
            vec![Subproblem {
                id: 7,
                variables: vec![0],
                multiplicity: bounds(0., 1.),
                columns: vec![],
            }],
            opts.clone(),
        )
        .unwrap();
        let session = root.snapshot().unwrap().begin_import(7).unwrap();
        let phase = session.feasibility_pricing(opts.clone()).unwrap();
        assert_eq!(phase.kind(), ImportPricingKind::Feasibility);
        assert_eq!(phase.restricted_violation(), Some(1.));
        assert!(phase.reduced_cost(column(0, 100., 1.)).unwrap() < 0.);
        assert_eq!(
            phase.reduced_cost(column(0, 100., 1.)).unwrap(),
            phase.reduced_cost(column(0, -100., 1.)).unwrap()
        );
        let costs = phase.costs().unwrap();
        assert_eq!(costs.kind, ImportPricingKind::Feasibility);
        let enclosed = phase
            .reduced_cost_bounds(column(0, 100., 1.))
            .unwrap()
            .unwrap();
        assert!(enclosed.upper < 0.);

        assert!(
            (costs.per_column + costs.coordinates[0].1
                - phase.reduced_cost(column(0, 100., 1.)).unwrap())
            .abs()
                < 1e-8
        );
        assert!(session.solve(opts.clone()).is_err());
        let first = session
            .collect(
                &ImportOptions {
                    max_batches: 1,
                    ..Default::default()
                },
                |_, _| {
                    Ok(ColumnBatch {
                        columns: vec![column(0, 100., 1.)],
                        complete: false,
                    })
                },
            )
            .unwrap();
        let feasible = first.session.feasibility_pricing(opts.clone()).unwrap();
        assert_eq!(feasible.restricted_violation(), Some(0.));
        let objective = first.session.pricing(opts.clone()).unwrap();
        assert_eq!(objective.kind(), ImportPricingKind::Objective);
        assert_eq!(objective.restricted_violation(), None);
        assert!(objective.reduced_cost(column(1, 1., 1.)).unwrap() < -90.);
        assert!(first.session.solve(opts.clone()).is_err());
        let complete = first
            .session
            .collect(&ImportOptions::default(), |_, _| {
                Ok(ColumnBatch {
                    columns: vec![column(1, 1., 1.)],
                    complete: true,
                })
            })
            .unwrap();
        optimal(&complete.session.solve(opts.clone()).unwrap().output, 8.);
        let mut zero = opts;
        zero.cg.time_limit = std::time::Duration::ZERO;
        assert!(session.feasibility_pricing(zero).is_err());
    }
}

#[test]
fn imported_phase_one_keeps_fixed_routes_and_nonlinear_rows_in_the_feasibility_model() {
    use baldes::cuts::soft_conflicts::{Mode, PresenceRow};
    use model::PresenceCoefficients;
    let root = imported::solve(
        Problem::<2> {
            rows: vec![Row {
                sense: RowSense::GreaterEqual,
                rhs: 1.,
            }],
            networks: vec![network(99, &[(5., &[0])], 1)],
        },
        vec![MasterVariable {
            id: 0,
            cost: 123.,
            bounds: bounds(0., f64::INFINITY),
            integer: false,
            entries: vec![(0, 1.)],
        }],
        vec![Subproblem {
            id: 7,
            variables: vec![0],
            multiplicity: bounds(0., 1.),
            columns: vec![],
        }],
        options(),
    )
    .unwrap();
    let fixed = root
        .snapshot()
        .unwrap()
        .fix_route(0, 1.)
        .unwrap()
        .with_presence_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 1.,
            },
            &[PresenceCoefficients {
                network: 99,
                row: PresenceRow::new(2, vec![(0, 1)], Mode::AnyPair).unwrap(),
                arc_events: vec![vec![0], vec![1]],
            }],
            &[(0, 1.)],
        )
        .unwrap();
    let session = fixed.begin_import(7).unwrap();
    let point = session.feasibility_pricing(options()).unwrap();
    assert_eq!(point.restricted_violation(), Some(0.));
    assert!(session.solve(options()).is_err());
    // The fixed routing cost remains in the ordinary model, not in phase I.
    optimal(&fixed.solve(options()).unwrap().output, 5.);
}

#[test]
fn imported_priced_collection_switches_phases_and_resumes_at_the_current_domain() {
    use imported::{ColumnBatch, ImportOptions, ImportPricingKind, ImportStatus};
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let mut opts = options();
        opts.cg.bound_mode = mode;
        let root = imported::solve(
            Problem::<2> {
                rows: vec![Row {
                    sense: RowSense::GreaterEqual,
                    rhs: 1.,
                }],
                networks: vec![],
            },
            vec![MasterVariable {
                id: 0,
                cost: 7.,
                bounds: bounds(0., f64::INFINITY),
                integer: false,
                entries: vec![(0, 1.)],
            }],
            vec![Subproblem {
                id: 7,
                variables: vec![0],
                multiplicity: bounds(0., 1.),
                columns: vec![],
            }],
            opts.clone(),
        )
        .unwrap();
        let session = root.snapshot().unwrap().begin_import(7).unwrap();
        // This source's entire intended domain consists of these two columns.
        let mut seen = vec![];
        let mut source = |point: &imported::ImportPricing<2>, remaining: std::time::Duration| {
            assert!(!remaining.is_zero());
            seen.push(point.kind());
            match point.subproblem().columns.len() {
                0 => {
                    assert_eq!(point.kind(), ImportPricingKind::Feasibility);
                    assert_eq!(point.restricted_violation(), Some(1.));
                    assert!(point.reduced_cost(column(0, 100., 1.)).unwrap() < 0.);
                    Ok(ColumnBatch {
                        columns: vec![column(0, 100., 1.)],
                        complete: false,
                    })
                }
                1 => {
                    assert_eq!(point.kind(), ImportPricingKind::Objective);
                    assert!(point.reduced_cost(column(1, 1., 1.)).unwrap() < 0.);
                    Ok(ColumnBatch {
                        columns: vec![column(1, 1., 1.)],
                        complete: true,
                    })
                }
                _ => panic!("completed source called again"),
            }
        };
        let first = session
            .collect_priced(
                &ImportOptions {
                    max_batches: 1,
                    ..Default::default()
                },
                opts.clone(),
                &mut source,
            )
            .unwrap();
        assert_eq!(first.status, ImportStatus::BatchLimit);
        assert_eq!(first.batches, 1);
        assert!(first.session.solve(opts.clone()).is_err());
        let done = first
            .session
            .collect_priced(&ImportOptions::default(), opts.clone(), &mut source)
            .unwrap();
        assert_eq!(done.status, ImportStatus::Complete);
        assert_eq!(done.batches, 1);
        assert_eq!(done.added_columns, 1);
        drop(source);
        assert_eq!(
            seen,
            vec![ImportPricingKind::Feasibility, ImportPricingKind::Objective]
        );
        optimal(&done.session.solve(opts.clone()).unwrap().output, 8.);
        let already = done
            .session
            .collect_priced(
                &ImportOptions {
                    max_batches: 0,
                    time_limit: std::time::Duration::ZERO,
                },
                opts,
                |_, _| panic!("completed session called source"),
            )
            .unwrap();
        assert_eq!(already.status, ImportStatus::Complete);
        assert_eq!(already.batches, 0);
        assert!(session.subproblem().columns.is_empty());
    }
}

#[test]
fn imported_priced_collection_respects_limits_and_rolls_back_invalid_later_batches() {
    use imported::{ColumnBatch, ImportOptions, ImportStatus};
    let root = fixture(BoundMode::Numerical);
    let session = root.snapshot().unwrap().begin_import(7).unwrap();
    let zero = session
        .collect_priced(
            &ImportOptions {
                max_batches: 2,
                time_limit: std::time::Duration::ZERO,
            },
            options(),
            |_, _| panic!("zero-time source called"),
        )
        .unwrap();
    assert_eq!(zero.status, ImportStatus::TimeLimit);
    assert_eq!(zero.batches, 0);
    let zero = session
        .collect_priced(
            &ImportOptions {
                max_batches: 0,
                ..Default::default()
            },
            options(),
            |_, _| panic!("zero-batch source called"),
        )
        .unwrap();
    assert_eq!(zero.status, ImportStatus::BatchLimit);
    let mut stopped_lp = options();
    stopped_lp.cg.time_limit = std::time::Duration::ZERO;
    assert!(session
        .collect_priced(&ImportOptions::default(), stopped_lp, |_, _| panic!(
            "unsolved pricing point used"
        ))
        .is_err());
    let mut calls = 0;
    assert!(session
        .collect_priced(&ImportOptions::default(), options(), |point, _| {
            calls += 1;
            assert_eq!(point.subproblem().columns.len(), calls);
            Ok(ColumnBatch {
                columns: vec![column(1, -1., 1.)],
                complete: calls == 2,
            })
        })
        .is_err());
    assert_eq!(calls, 2);
    assert_eq!(session.subproblem().columns.len(), 1);
    assert!(!session.is_complete());
    let pending = session
        .collect_priced(
            &ImportOptions {
                max_batches: 2,
                ..Default::default()
            },
            options(),
            |_, _| {
                Ok(ColumnBatch {
                    columns: vec![],
                    complete: false,
                })
            },
        )
        .unwrap();
    assert_eq!(pending.status, ImportStatus::BatchLimit);
    assert_eq!(pending.batches, 2);
    assert!(pending.session.solve(options()).is_err());
    optimal(
        &root.snapshot().unwrap().solve(options()).unwrap().output,
        5.,
    );
}

#[test]
fn imported_objective_pricing_recovers_feasibility_above_the_artificial_penalty() {
    for routing in [false, true] {
        let root = imported::solve(
            Problem::<2> {
                rows: vec![Row {
                    sense: RowSense::GreaterEqual,
                    rhs: 1.,
                }],
                networks: if routing {
                    vec![network(99, &[(100., &[0])], 1)]
                } else {
                    vec![]
                },
            },
            vec![MasterVariable {
                id: 0,
                cost: 7.,
                bounds: bounds(0., f64::INFINITY),
                integer: false,
                entries: vec![(0, 1.)],
            }],
            vec![Subproblem {
                id: 7,
                variables: vec![0],
                multiplicity: bounds(0., 1.),
                columns: if routing {
                    vec![]
                } else {
                    vec![column(0, 100., 1.)]
                },
            }],
            options(),
        )
        .unwrap();
        let parent_cost = if routing { 100. } else { 107. };
        optimal(&root.output, parent_cost);
        let mut parent = root.snapshot().unwrap();
        if routing {
            use baldes::cuts::soft_conflicts::{Mode, PresenceRow};
            parent = parent
                .with_presence_row(
                    Row {
                        sense: RowSense::LessEqual,
                        rhs: 1.,
                    },
                    &[model::PresenceCoefficients {
                        network: 99,
                        row: PresenceRow::new(2, vec![(0, 1)], Mode::AnyPair).unwrap(),
                        arc_events: vec![vec![0], vec![1]],
                    }],
                    &[(0, 1.)],
                )
                .unwrap();
        }
        let session = parent.begin_import(7).unwrap();
        let mut low_penalty = options();
        low_penalty.cg.artificial_cost = 1.;
        low_penalty.cg.max_penalty_updates = 0;
        let point = session.pricing(low_penalty.clone()).unwrap();
        assert!(point.reduced_cost(column(1, 1., 1.)).unwrap() < -1.);
        assert!(!session.is_complete());
        let done = session
            .collect_priced(
                &imported::ImportOptions::default(),
                low_penalty,
                |point, _| {
                    assert_eq!(point.kind(), imported::ImportPricingKind::Objective);
                    Ok(imported::ColumnBatch {
                        columns: vec![column(1, 1., 1.)],
                        complete: true,
                    })
                },
            )
            .unwrap();
        optimal(&done.session.solve(options()).unwrap().output, 8.);
        optimal(
            &root.snapshot().unwrap().solve(options()).unwrap().output,
            parent_cost,
        );
    }
}

#[test]
fn imported_envelopes_bound_unseen_vectors_and_survive_completion_and_extension() {
    use imported::{ColumnBatch, ColumnEnvelope, ImportOptions};
    let root = fixture(BoundMode::Numerical);
    let parent = root
        .snapshot()
        .unwrap()
        .with_column_bounds(42, 0, bounds(0., 0.))
        .unwrap();
    let session = parent
        .begin_import(7)
        .unwrap()
        .with_envelope(ColumnEnvelope {
            cost_lower: 1.,
            coordinates: vec![(0, bounds(1., 4.))],
        })
        .unwrap();
    let point = session.pricing(options()).unwrap();
    let lower = point.envelope_reduced_cost_bound().unwrap().unwrap();
    assert!(lower.is_finite() && lower < 0.);
    let corner = point
        .reduced_cost_bounds(column(1, 1., 4.))
        .unwrap()
        .unwrap();
    assert_eq!(lower, corner.lower);
    for cost in [1., 2., 10., 100.] {
        for value in [1., 1.5, 2., 3.5, 4.] {
            let candidate = point
                .reduced_cost_bounds(column(1, cost, value))
                .unwrap()
                .unwrap();
            assert!(lower <= candidate.lower);
        }
    }
    assert!(session
        .collect(&ImportOptions::default(), |_, _| Ok(ColumnBatch {
            columns: vec![column(1, 0., 2.)],
            complete: true,
        }))
        .is_err());
    assert!(session
        .collect(&ImportOptions::default(), |_, _| Ok(ColumnBatch {
            columns: vec![column(1, 2., 5.)],
            complete: true,
        }))
        .is_err());
    assert_eq!(session.subproblem().columns.len(), 1);
    assert!(session
        .with_envelope(ColumnEnvelope {
            cost_lower: 11.,
            coordinates: vec![(0, bounds(1., 4.))]
        })
        .is_err());
    assert!(session
        .with_envelope(ColumnEnvelope {
            cost_lower: 1.,
            coordinates: vec![(999, bounds(1., 4.))]
        })
        .is_err());
    let widened = session
        .with_envelope(ColumnEnvelope {
            cost_lower: f64::NEG_INFINITY,
            coordinates: vec![(0, bounds(f64::NEG_INFINITY, f64::INFINITY))],
        })
        .unwrap();
    let envelope = widened.envelope().unwrap();
    assert_eq!(envelope.cost_lower, 1.);
    assert_eq!(envelope.coordinates[0].1.lower, 1.);
    assert_eq!(envelope.coordinates[0].1.upper, 4.);
    let done = session
        .collect(&ImportOptions::default(), |_, _| {
            Ok(ColumnBatch {
                columns: vec![column(1, 1., 2.)],
                complete: true,
            })
        })
        .unwrap();
    let solved = done.session.solve(options()).unwrap();
    optimal(&solved.output, 0.5);
    let snapshot = solved.snapshot().unwrap();
    #[cfg(feature = "highs")]
    {
        let mip = snapshot
            .node()
            .solve_enumerated_mip(&Default::default())
            .unwrap()
            .unwrap();
        assert_eq!(mip.mip.solution.as_ref().unwrap().objective, 1.);
    }

    let retained = snapshot.begin_import(7).unwrap();
    assert_eq!(retained.envelope().unwrap().cost_lower, 1.);
    assert!(snapshot
        .solve_with_columns(7, vec![column(2, 0., 1.)], options())
        .is_err());
    assert!(snapshot
        .solve_with_columns(7, vec![column(2, 2., 5.)], options())
        .is_err());
    optimal(
        &snapshot
            .solve_with_columns(7, vec![column(2, 2., 3.)], options())
            .unwrap()
            .output,
        0.5,
    );
    assert!(parent.begin_import(7).unwrap().envelope().is_none());
    optimal(&parent.solve(options()).unwrap().output, 10.);
}

#[test]
fn imported_envelope_bounds_distinguish_absent_unbounded_and_phase_one_costs() {
    use imported::ColumnEnvelope;
    let root = fixture(BoundMode::Numerical);
    let base = root
        .snapshot()
        .unwrap()
        .with_column_bounds(42, 0, bounds(0., 0.))
        .unwrap()
        .begin_import(7)
        .unwrap();
    assert_eq!(
        base.pricing(options())
            .unwrap()
            .envelope_reduced_cost_bound()
            .unwrap(),
        None
    );
    let no_cost = base
        .with_envelope(ColumnEnvelope {
            cost_lower: f64::NEG_INFINITY,
            coordinates: vec![(0, bounds(1., 4.))],
        })
        .unwrap();
    assert_eq!(
        no_cost
            .pricing(options())
            .unwrap()
            .envelope_reduced_cost_bound()
            .unwrap(),
        Some(f64::NEG_INFINITY)
    );
    assert!(no_cost
        .feasibility_pricing(options())
        .unwrap()
        .envelope_reduced_cost_bound()
        .unwrap()
        .unwrap()
        .is_finite());
    let unbounded_vector = base
        .with_envelope(ColumnEnvelope {
            cost_lower: 1.,
            coordinates: vec![(0, bounds(f64::NEG_INFINITY, f64::INFINITY))],
        })
        .unwrap();
    assert_eq!(
        unbounded_vector
            .pricing(options())
            .unwrap()
            .envelope_reduced_cost_bound()
            .unwrap(),
        Some(f64::NEG_INFINITY)
    );
    for envelope in [
        ColumnEnvelope {
            cost_lower: f64::NAN,
            coordinates: vec![(0, bounds(1., 4.))],
        },
        ColumnEnvelope {
            cost_lower: 1.,
            coordinates: vec![],
        },
        ColumnEnvelope {
            cost_lower: 1.,
            coordinates: vec![(0, bounds(4., 1.))],
        },
    ] {
        assert!(base.with_envelope(envelope).is_err());
    }
    assert!(base.envelope().is_none());
    let empty = imported::solve(
        Problem::<2> {
            rows: vec![Row {
                sense: RowSense::GreaterEqual,
                rhs: 1.,
            }],
            networks: vec![],
        },
        vec![MasterVariable {
            id: 0,
            cost: 7.,
            bounds: bounds(0., f64::INFINITY),
            integer: false,
            entries: vec![(0, 1.)],
        }],
        vec![Subproblem {
            id: 7,
            variables: vec![0],
            multiplicity: bounds(0., 1.),
            columns: vec![],
        }],
        options(),
    )
    .unwrap();
    let missing = empty
        .snapshot()
        .unwrap()
        .begin_import(7)
        .unwrap()
        .with_envelope(ColumnEnvelope {
            cost_lower: 100.,
            coordinates: vec![(0, bounds(1., 2.))],
        })
        .unwrap();
    let phase = missing.feasibility_pricing(options()).unwrap();
    assert_eq!(phase.restricted_violation(), Some(1.));
    assert_eq!(phase.envelope_reduced_cost_bound().unwrap(), Some(-2.));
    assert!(!missing.is_complete());
}

#[test]
fn imported_envelope_box_bound_covers_signed_multidimensional_future_columns() {
    use imported::ColumnEnvelope;
    let root = imported::solve(
        Problem::<7> {
            rows: vec![],
            networks: vec![],
        },
        vec![
            MasterVariable {
                id: i32::MAX,
                cost: 2.,
                bounds: bounds(f64::NEG_INFINITY, f64::INFINITY),
                integer: false,
                entries: vec![],
            },
            MasterVariable {
                id: 3,
                cost: -3.,
                bounds: bounds(f64::NEG_INFINITY, f64::INFINITY),
                integer: false,
                entries: vec![],
            },
        ],
        vec![Subproblem {
            id: 42,
            variables: vec![i32::MAX, 3],
            multiplicity: bounds(1., 1.),
            columns: vec![Column {
                id: 0,
                cost: 10.,
                bounds: bounds(0., f64::INFINITY),
                integer: true,
                entries: vec![(i32::MAX, 1.), (3, -1.)],
            }],
        }],
        options(),
    )
    .unwrap();
    optimal(&root.output, 15.);
    let session = root
        .snapshot()
        .unwrap()
        .begin_import(42)
        .unwrap()
        .with_envelope(ColumnEnvelope {
            cost_lower: -1.,
            coordinates: vec![(3, bounds(-3., 5.)), (i32::MAX, bounds(-2., 4.))],
        })
        .unwrap();
    let point = session.pricing(options()).unwrap();
    assert_eq!(
        point.costs().unwrap().coordinates,
        vec![(i32::MAX, 2.), (3, -3.)]
    );
    let lower = point.envelope_reduced_cost_bound().unwrap().unwrap();
    assert_eq!(lower, -35.);
    for cost in [-1., 0., 10.] {
        for x in [-2., -1.5, 0., 1., 4.] {
            for y in [-3., -1., 0., 1.5, 5.] {
                let enclosed = point
                    .reduced_cost_bounds(Column {
                        id: 1,
                        cost,
                        bounds: bounds(0., f64::INFINITY),
                        integer: true,
                        entries: vec![(i32::MAX, x), (3, y)],
                    })
                    .unwrap()
                    .unwrap();
                assert!(lower <= enclosed.lower);
            }
        }
    }
    assert!(!session.is_complete());
}

#[test]
fn imported_open_domain_objective_certificate_covers_unenumerated_envelope_columns() {
    use imported::{ColumnBatch, ColumnEnvelope, ImportOptions, ImportPricingKind};
    let root = imported::solve(
        Problem::<7> {
            rows: vec![],
            networks: vec![],
        },
        vec![
            MasterVariable {
                id: 0,
                cost: 2.,
                bounds: bounds(f64::NEG_INFINITY, f64::INFINITY),
                integer: false,
                entries: vec![],
            },
            MasterVariable {
                id: 1,
                cost: -3.,
                bounds: bounds(f64::NEG_INFINITY, f64::INFINITY),
                integer: false,
                entries: vec![],
            },
        ],
        vec![Subproblem {
            id: 7,
            variables: vec![0, 1],
            multiplicity: bounds(1., 1.),
            columns: vec![Column {
                id: 0,
                cost: 10.,
                bounds: bounds(0., f64::INFINITY),
                integer: true,
                entries: vec![(0, 1.), (1, -1.)],
            }],
        }],
        options(),
    )
    .unwrap();
    let session = root
        .snapshot()
        .unwrap()
        .begin_import(7)
        .unwrap()
        .with_envelope(ColumnEnvelope {
            cost_lower: -1.,
            coordinates: vec![(0, bounds(-2., 4.)), (1, bounds(-3., 5.))],
        })
        .unwrap();
    let certificate = session.objective_certificate(options()).unwrap().unwrap();
    assert_eq!(certificate.kind(), ImportPricingKind::Objective);
    assert_eq!(certificate.lower_bound(), -20.);
    assert!(!certificate.proves_infeasible());
    for cost in [-1., 0., 10.] {
        for x in [-2., -1., 0., 4.] {
            for y in [-3., 0., 1., 5.] {
                assert!(certificate.lower_bound() <= cost + 2. * x - 3. * y);
            }
        }
    }
    let improved = session
        .collect(
            &ImportOptions {
                max_batches: 1,
                ..Default::default()
            },
            |_, _| {
                Ok(ColumnBatch {
                    columns: vec![Column {
                        id: 1,
                        cost: -1.,
                        bounds: bounds(0., f64::INFINITY),
                        integer: true,
                        entries: vec![(0, -2.), (1, 5.)],
                    }],
                    complete: false,
                })
            },
        )
        .unwrap();
    let improved_bound = improved
        .session
        .objective_certificate(options())
        .unwrap()
        .unwrap();
    assert_eq!(improved_bound.lower_bound(), -20.);
    assert!(!improved.session.is_complete());
    assert!(improved.session.solve(options()).is_err());
    optimal(
        &root.snapshot().unwrap().solve(options()).unwrap().output,
        15.,
    );
}

#[test]
fn imported_open_domain_phase_one_certificate_proves_only_envelope_infeasibility() {
    use imported::{ColumnBatch, ColumnEnvelope, ImportOptions};
    let root = imported::solve(
        Problem::<2> {
            rows: vec![Row {
                sense: RowSense::GreaterEqual,
                rhs: 1.,
            }],
            networks: vec![],
        },
        vec![MasterVariable {
            id: 0,
            cost: 0.,
            bounds: bounds(0., 1.),
            integer: false,
            entries: vec![(0, 1.)],
        }],
        vec![Subproblem {
            id: 7,
            variables: vec![0],
            multiplicity: bounds(0., 1.),
            columns: vec![],
        }],
        options(),
    )
    .unwrap();
    let base = root.snapshot().unwrap().begin_import(7).unwrap();
    assert!(base.infeasibility_certificate(options()).unwrap().is_none());
    let impossible = base
        .with_envelope(ColumnEnvelope {
            cost_lower: f64::NEG_INFINITY,
            coordinates: vec![(0, bounds(0., 0.))],
        })
        .unwrap();
    let certificate = impossible
        .infeasibility_certificate(options())
        .unwrap()
        .unwrap();
    assert_eq!(certificate.lower_bound(), 1.);
    assert!(certificate.proves_infeasible());
    assert!(!impossible.is_complete());
    assert!(impossible
        .collect(&ImportOptions::default(), |_, _| Ok(ColumnBatch {
            columns: vec![column(0, 0., 1.)],
            complete: true,
        }))
        .is_err());
    let possible = base
        .with_envelope(ColumnEnvelope {
            cost_lower: 100.,
            coordinates: vec![(0, bounds(1., 2.))],
        })
        .unwrap();
    let certificate = possible
        .feasibility_certificate(options())
        .unwrap()
        .unwrap();
    assert!(certificate.lower_bound() <= 0.);
    assert!(!certificate.proves_infeasible());
    assert!(possible
        .infeasibility_certificate(options())
        .unwrap()
        .is_none());
    let recovered = possible
        .collect(
            &ImportOptions {
                max_batches: 1,
                ..Default::default()
            },
            |_, _| {
                Ok(ColumnBatch {
                    columns: vec![column(0, 100., 1.)],
                    complete: false,
                })
            },
        )
        .unwrap();
    assert!(!recovered
        .session
        .feasibility_certificate(options())
        .unwrap()
        .unwrap()
        .proves_infeasible());
}

#[test]
fn imported_open_domain_certificate_handles_infinite_and_explicit_multiplicity_caps() {
    use imported::ColumnEnvelope;
    fn model(cost: f64) -> imported::ImportedOutput<2> {
        imported::solve(
            Problem::<2> {
                rows: vec![],
                networks: vec![],
            },
            vec![MasterVariable {
                id: 0,
                cost: 0.,
                bounds: bounds(0., f64::INFINITY),
                integer: false,
                entries: vec![],
            }],
            vec![Subproblem {
                id: 7,
                variables: vec![0],
                multiplicity: bounds(0., f64::INFINITY),
                columns: vec![column(0, cost, 0.)],
            }],
            options(),
        )
        .unwrap()
    }
    let root = model(0.);
    let snapshot = root.snapshot().unwrap();
    let adverse = snapshot
        .begin_import(7)
        .unwrap()
        .with_envelope(ColumnEnvelope {
            cost_lower: -1.,
            coordinates: vec![(0, bounds(0., 0.))],
        })
        .unwrap();
    assert!(adverse.objective_certificate(options()).unwrap().is_none());
    let capped = snapshot
        .with_multiplicity(7, bounds(0., 2.))
        .unwrap()
        .begin_import(7)
        .unwrap()
        .with_envelope(ColumnEnvelope {
            cost_lower: -1.,
            coordinates: vec![(0, bounds(0., 0.))],
        })
        .unwrap();
    assert_eq!(
        capped
            .objective_certificate(options())
            .unwrap()
            .unwrap()
            .lower_bound(),
        -2.
    );
    let disabled = snapshot
        .with_multiplicity(7, bounds(0., 0.))
        .unwrap()
        .begin_import(7)
        .unwrap();
    assert_eq!(
        disabled
            .objective_certificate(options())
            .unwrap()
            .unwrap()
            .lower_bound(),
        0.
    );
    let positive = model(1.)
        .snapshot()
        .unwrap()
        .begin_import(7)
        .unwrap()
        .with_envelope(ColumnEnvelope {
            cost_lower: 1.,
            coordinates: vec![(0, bounds(0., 0.))],
        })
        .unwrap();
    assert_eq!(
        positive
            .objective_certificate(options())
            .unwrap()
            .unwrap()
            .lower_bound(),
        0.
    );
    let mut zero = options();
    zero.cg.time_limit = std::time::Duration::ZERO;
    assert!(positive.objective_certificate(zero).is_err());
}

#[test]
fn imported_open_domain_certificates_retain_fixed_routing_and_nonlinear_contributions() {
    use baldes::cuts::soft_conflicts::{Mode, PresenceRow};
    use imported::ColumnEnvelope;
    use model::PresenceCoefficients;
    let root = imported::solve(
        Problem::<2> {
            rows: vec![Row {
                sense: RowSense::GreaterEqual,
                rhs: 1.,
            }],
            networks: vec![network(99, &[(5., &[0])], 1)],
        },
        vec![
            MasterVariable {
                id: 0,
                cost: 0.,
                bounds: bounds(0., 1.),
                integer: false,
                entries: vec![(0, 1.)],
            },
            MasterVariable {
                id: 1,
                cost: 0.,
                bounds: bounds(0., 1.),
                integer: false,
                entries: vec![],
            },
        ],
        vec![Subproblem {
            id: 7,
            variables: vec![0, 1],
            multiplicity: bounds(0., 1.),
            columns: vec![],
        }],
        options(),
    )
    .unwrap();
    let fixed = root.snapshot().unwrap().fix_route(0, 1.).unwrap();
    for rhs in [0., 1.] {
        let child = fixed
            .with_presence_row(
                Row {
                    sense: RowSense::LessEqual,
                    rhs,
                },
                &[PresenceCoefficients {
                    network: 99,
                    row: PresenceRow::new(2, vec![(0, 1)], Mode::AnyPair).unwrap(),
                    arc_events: vec![vec![0], vec![1]],
                }],
                &[(1, 1.)],
            )
            .unwrap();
        let open = child
            .begin_import(7)
            .unwrap()
            .with_envelope(ColumnEnvelope {
                cost_lower: 0.,
                coordinates: vec![(0, bounds(0., 1.)), (1, bounds(0., 1.))],
            })
            .unwrap();
        if rhs == 0. {
            assert!(open
                .infeasibility_certificate(options())
                .unwrap()
                .unwrap()
                .proves_infeasible());
        } else {
            assert_eq!(
                open.objective_certificate(options())
                    .unwrap()
                    .unwrap()
                    .lower_bound(),
                5.
            );
            assert!(open.infeasibility_certificate(options()).unwrap().is_none());
        }
        assert!(!open.is_complete());
    }
    optimal(
        &root.snapshot().unwrap().solve(options()).unwrap().output,
        5.,
    );
}
