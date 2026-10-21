//! Open-domain tree conclusions versus explicit original-coordinate models.
use super::imported_multiple_tests::{bounds, column, envelope, fixture};
use super::*;
use multi_pricing::{
    imported::{
        self, ColumnBatch, ColumnEnvelope, ImportOptions, ImportPricingKind, ImportTreeStatus,
    },
    MasterVariable,
};
use std::time::Duration;

fn tree_options() -> multi_pricing::AssignmentTreeOptions {
    multi_pricing::AssignmentTreeOptions {
        node: options(),
        ..Default::default()
    }
}
#[test]
fn open_tree_closes_from_a_joint_certificate_without_enumerating_sources() {
    let root = fixture(false);
    let session = root
        .snapshot()
        .unwrap()
        .begin_imports(&[7, 42])
        .unwrap()
        .with_envelope(7, envelope(2., 0, 1.))
        .unwrap()
        .with_envelope(42, envelope(1., 1, 1.))
        .unwrap();
    let tree = session.begin_tree();
    let mut calls = 0;
    let result = tree
        .run(
            &tree_options(),
            &ImportOptions::default(),
            |point, remaining| {
                calls += 1;
                assert!(!remaining.is_zero());
                assert_eq!(point.kind(), ImportPricingKind::Objective);
                assert_eq!(point.subproblem().id, 7);
                Ok(ColumnBatch {
                    columns: vec![column(1, 2., 0)],
                    complete: false,
                })
            },
        )
        .unwrap();
    assert_eq!(calls, 1);
    assert_eq!(result.status, ImportTreeStatus::Optimal);
    assert_eq!(result.open_nodes, 0);
    assert_eq!(result.bound, 3.);
    assert_eq!(result.incumbent, 3.);
    assert_eq!(
        result.tree.solution().unwrap().values.original,
        vec![(0, 1.), (1, 1.)]
    );
    assert!(tree.solution().is_none());
    assert!(!session.is_complete());
    optimal(&root.output, 101.);
}
#[test]
fn open_tree_phase_one_covers_all_groups_before_proving_infeasibility() {
    let root = fixture(true);
    let base = root.snapshot().unwrap().begin_imports(&[7, 42]).unwrap();
    let impossible = base
        .with_envelope(7, envelope(0., 0, 0.))
        .unwrap()
        .with_envelope(42, envelope(0., 1, 0.))
        .unwrap();
    let result = impossible
        .begin_tree()
        .run(&tree_options(), &ImportOptions::default(), |_, _| {
            panic!("positive original-domain feasibility proof needs no callback")
        })
        .unwrap();
    assert_eq!(result.status, ImportTreeStatus::Infeasible);
    assert_eq!(result.open_nodes, 0);
    assert_eq!(result.incumbent, f64::INFINITY);
    let possible = base
        .with_envelope(7, envelope(2., 0, 1.))
        .unwrap()
        .with_envelope(42, envelope(1., 1, 1.))
        .unwrap();
    let mut kinds = vec![];
    let result = possible
        .begin_tree()
        .run(&tree_options(), &ImportOptions::default(), |point, _| {
            kinds.push(point.kind());
            let (id, cost, coordinate) = if point.subproblem().id == 7 {
                (7, 2., 0)
            } else {
                (42, 1., 1)
            };
            assert!(point.reduced_cost(column(0, cost, coordinate)).unwrap() < 0.);
            assert_eq!(id, point.subproblem().id);
            Ok(ColumnBatch {
                columns: vec![column(0, cost, coordinate)],
                complete: false,
            })
        })
        .unwrap();
    assert_eq!(kinds, vec![ImportPricingKind::Feasibility; 2]);
    assert_eq!(result.status, ImportTreeStatus::Optimal);
    assert_eq!(result.incumbent, 3.);
}
fn fractional_fixture(original_integer: bool) -> imported::ImportedOutput<2> {
    imported::solve(
        Problem::<2> {
            rows: vec![Row {
                sense: RowSense::GreaterEqual,
                rhs: 1.5,
            }],
            networks: vec![],
        },
        vec![MasterVariable {
            id: 0,
            cost: 1.,
            bounds: bounds(0., f64::INFINITY),
            integer: original_integer,
            entries: vec![(0, 1.)],
        }],
        vec![imported::Subproblem {
            id: 7,
            variables: vec![0],
            multiplicity: bounds(0., 3.),
            columns: vec![column(0, 0., 0)],
        }],
        options(),
    )
    .unwrap()
}
#[test]
fn open_tree_original_integer_branches_use_certificates_and_preserve_the_parent() {
    let root = fractional_fixture(true);
    assert_eq!(root.values.original, vec![(0, 1.5)]);
    let session = root
        .snapshot()
        .unwrap()
        .begin_imports(&[7])
        .unwrap()
        .with_envelope(7, envelope(0., 0, 1.))
        .unwrap();
    let result = session
        .begin_tree()
        .run(
            &tree_options(),
            &ImportOptions {
                max_batches: 0,
                ..Default::default()
            },
            |_, _| panic!("both integer children close from original-domain certificates"),
        )
        .unwrap();
    assert_eq!(result.status, ImportTreeStatus::Optimal);
    assert_eq!(result.incumbent, 2.);
    assert_eq!(result.bound, 2.);
    assert!(result.nodes >= 3);
    assert_eq!(
        result.tree.solution().unwrap().values.original,
        vec![(0, 2.)]
    );
    assert_eq!(root.values.original, vec![(0, 1.5)]);
    assert!(!session.is_complete());
}
#[test]
fn open_tree_column_branches_replay_into_callbacks_and_share_completion_with_siblings() {
    let root = fractional_fixture(false);
    let session = root
        .snapshot()
        .unwrap()
        .begin_imports(&[7])
        .unwrap()
        .with_envelope(
            7,
            ColumnEnvelope {
                cost_lower: 0.,
                coordinates: vec![(0, bounds(1., 2.))],
            },
        )
        .unwrap();
    let mut calls = 0;
    let result = session
        .begin_tree()
        .run(&tree_options(), &ImportOptions::default(), |point, _| {
            calls += 1;
            // The <=1 child cannot cover x>=1.5 with its known pattern. Its
            // sibling must receive this globally valid pattern and completion too.
            assert_eq!(point.kind(), ImportPricingKind::Feasibility);
            assert_eq!(point.subproblem().columns.len(), 1);
            let mut new = column(1, 0., 0);
            new.entries[0].1 = 2.;
            assert!(point.reduced_cost(new.clone()).unwrap() < 0.);
            Ok(ColumnBatch {
                columns: vec![new],
                complete: true,
            })
        })
        .unwrap();
    assert_eq!(calls, 1);
    assert_eq!(result.status, ImportTreeStatus::Optimal);
    assert_eq!(result.incumbent, 2.);
    assert_eq!(result.open_nodes, 0);
    let solution = result.tree.solution().unwrap();
    assert_eq!(solution.values.original, vec![(0, 2.)]);
    assert!(solution
        .values
        .columns
        .iter()
        .any(|v| v.column == 1 && v.value == 1.));
}
#[test]
fn open_tree_batch_limits_resume_without_promoting_the_restricted_objective() {
    let root = fixture(false);
    let session = root.snapshot().unwrap().begin_imports(&[7]).unwrap();
    let start = session.begin_tree();
    let limits = ImportOptions {
        max_batches: 1,
        ..Default::default()
    };
    let step = start
        .run(&tree_options(), &limits, |_, _| {
            Ok(ColumnBatch {
                columns: vec![column(1, 2., 0)],
                complete: false,
            })
        })
        .unwrap();
    assert_eq!(step.status, ImportTreeStatus::BatchLimit);
    assert_eq!(step.incumbent, 3.);
    assert_eq!(step.bound, f64::NEG_INFINITY);
    assert_eq!(step.open_nodes, 1);
    let done = step
        .tree
        .run(&tree_options(), &limits, |point, _| {
            assert_eq!(point.subproblem().columns.len(), 2);
            Ok(ColumnBatch {
                columns: vec![],
                complete: true,
            })
        })
        .unwrap();
    assert_eq!(done.status, ImportTreeStatus::Optimal);
    assert_eq!(done.incumbent, 3.);
    assert_eq!(done.bound, 3.);
    assert!(start.solution().is_none());
    assert!(!session.is_complete());
    let mut changed = tree_options();
    changed.node.cg.bound_mode = BoundMode::Certified;
    assert!(step
        .tree
        .run(&changed, &limits, |_, _| unreachable!())
        .is_err());
}
#[test]
fn open_tree_empty_batches_are_fair_and_errors_leave_the_frontier_reusable() {
    let root = fixture(false);
    let session = root.snapshot().unwrap().begin_imports(&[42, 7]).unwrap();
    let start = session.begin_tree();
    let mut seen = vec![];
    let progress = start
        .run(&tree_options(), &ImportOptions::default(), |point, _| {
            seen.push(point.subproblem().id);
            Ok(ColumnBatch {
                columns: vec![],
                complete: false,
            })
        })
        .unwrap();
    assert_eq!(seen, vec![42, 7]);
    assert_eq!(progress.status, ImportTreeStatus::Incomplete);
    assert_eq!(progress.bound, f64::NEG_INFINITY);
    let mut calls = 0;
    let failure = start.run(&tree_options(), &ImportOptions::default(), |point, _| {
        calls += 1;
        if calls == 1 {
            Ok(ColumnBatch {
                columns: vec![column(1, 2., 1)],
                complete: true,
            })
        } else {
            assert_eq!(point.subproblem().id, 7);
            Ok(ColumnBatch {
                columns: vec![column(0, 0., 0)],
                complete: true,
            })
        }
    });
    assert!(failure.is_err());
    assert_eq!(calls, 2);
    let mut mandatory = column(1, 2., 1);
    mandatory.bounds.lower = 1.;
    assert!(start
        .run(&tree_options(), &ImportOptions::default(), |_, _| Ok(
            ColumnBatch {
                columns: vec![mandatory.clone()],
                complete: true
            }
        ))
        .is_err());
    let recovered = start
        .run(&tree_options(), &ImportOptions::default(), |point, _| {
            assert_eq!(point.subproblem().columns.len(), 1);
            Ok(ColumnBatch {
                columns: vec![],
                complete: true,
            })
        })
        .unwrap();
    assert_eq!(recovered.status, ImportTreeStatus::Optimal);
    assert_eq!(recovered.incumbent, 101.);
    for zero in [true, false] {
        let mut configured = tree_options();
        if zero {
            configured.node_limit = 0;
        } else {
            configured.time_limit = Duration::ZERO;
        }
        let result = start
            .run(&configured, &ImportOptions::default(), |_, _| {
                panic!("zero budget callback")
            })
            .unwrap();
        assert_eq!(
            result.status,
            if zero {
                ImportTreeStatus::NodeLimit
            } else {
                ImportTreeStatus::TimeLimit
            }
        );
        assert_eq!(result.open_nodes, 1);
    }
}
#[test]
fn open_session_restrictions_apply_to_future_vectors_and_reject_invalid_inputs() {
    let root = fixture(false);
    let session = root.snapshot().unwrap().begin_imports(&[7]).unwrap();
    assert!(session.with_variable_bounds(999, bounds(0., 1.)).is_err());
    assert!(session.with_variable_bounds(0, bounds(2., 1.)).is_err());
    assert!(session.with_column_bounds(7, 999, bounds(0., 1.)).is_err());
    assert!(session.with_multiplicity(999, bounds(0., 1.)).is_err());
    assert!(session
        .with_routing_multiplicity(999, bounds(0., 1.))
        .is_err());
    assert!(session
        .with_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 0.
            },
            &[],
            &[(0, 1.), (0, 1.)]
        )
        .is_err());
    assert!(session
        .with_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: f64::NAN
            },
            &[],
            &[]
        )
        .is_err());
    let child = session
        .with_multiplicity(7, bounds(1., 1.))
        .unwrap()
        .with_variable_bounds(0, bounds(0., 1.))
        .unwrap()
        .with_column_bounds(7, 0, bounds(0., 0.))
        .unwrap()
        .with_row(
            Row {
                sense: RowSense::GreaterEqual,
                rhs: 1.,
            },
            &[],
            &[(0, 1.)],
        )
        .unwrap();
    let step = child
        .collect(&ImportOptions::default(), |_, _| {
            Ok(ColumnBatch {
                columns: vec![column(1, 2., 0)],
                complete: true,
            })
        })
        .unwrap();
    let solved = step.session.solve(options()).unwrap();
    optimal(&solved.output, 3.);
    assert_eq!(
        solved
            .values
            .columns
            .iter()
            .find(|v| v.subproblem == 7 && v.column == 0)
            .unwrap()
            .value,
        0.
    );
    optimal(&root.output, 101.);
}

#[test]
fn open_tree_branches_routing_multiplicity_and_physical_activity() {
    for arcs in [false, true] {
        let problem = if arcs {
            Problem {
                rows: vec![
                    Row {
                        sense: RowSense::Equal,
                        rhs: 1.,
                    },
                    Row {
                        sense: RowSense::GreaterEqual,
                        rhs: 0.5,
                    },
                ],
                networks: vec![network(7, &[(0., &[0]), (1., &[0, 1])], 2)],
            }
        } else {
            Problem {
                rows: vec![Row {
                    sense: RowSense::GreaterEqual,
                    rhs: 0.5,
                }],
                networks: vec![network(7, &[(1., &[0])], 1)],
            }
        };
        let root = imported::solve(
            problem,
            vec![],
            vec![imported::Subproblem {
                id: 42,
                variables: vec![],
                multiplicity: bounds(0., 0.),
                columns: vec![],
            }],
            options(),
        )
        .unwrap();
        let session = root.snapshot().unwrap().begin_imports(&[42]).unwrap();
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            let mut configured = tree_options();
            configured.node.cg.bound_mode = mode;
            let mut limited = configured.clone();
            limited.node_limit = 1;
            let first = session
                .begin_tree()
                .run(&limited, &ImportOptions::default(), |_, _| {
                    panic!("zero-cap source")
                })
                .unwrap();
            assert_eq!(first.status, ImportTreeStatus::NodeLimit);
            assert_eq!(first.open_nodes, 2);
            assert!((first.bound - 0.5).abs() < 1e-8);
            let done = first
                .tree
                .run(&configured, &ImportOptions::default(), |_, _| {
                    panic!("children close by certificate")
                })
                .unwrap();
            assert_eq!(done.status, ImportTreeStatus::Optimal);
            assert_eq!(done.incumbent, 1.);
            assert_eq!(done.bound, 1.);
        }
        optimal(&root.output, 0.5);
    }
}
#[test]
fn open_tree_replays_fixed_routes_and_new_presence_restrictions_into_phase_one() {
    use baldes::cuts::soft_conflicts::{Mode, PresenceRow};
    use multi_pricing::PresenceCoefficients;
    let root = imported::solve(
        Problem::<2> {
            rows: vec![
                Row {
                    sense: RowSense::Equal,
                    rhs: 1.,
                },
                Row {
                    sense: RowSense::GreaterEqual,
                    rhs: 1.,
                },
            ],
            networks: vec![network(7, &[(5., &[0])], 2)],
        },
        vec![MasterVariable {
            id: 0,
            cost: 0.,
            bounds: bounds(0., 1.),
            integer: false,
            entries: vec![(1, 1.)],
        }],
        vec![imported::Subproblem {
            id: 42,
            variables: vec![0],
            multiplicity: bounds(0., 1.),
            columns: vec![column(0, 100., 0)],
        }],
        options(),
    )
    .unwrap();
    let route = root.output.values.iter().position(|&v| v > 0.).unwrap();
    let parent = root.snapshot().unwrap().fix_route(route, 1.).unwrap();
    let session = parent
        .begin_imports(&[42])
        .unwrap()
        .with_envelope(42, envelope(2., 0, 1.))
        .unwrap()
        .with_routing_multiplicity(7, bounds(1., 1.))
        .unwrap()
        .with_presence_row(
            Row {
                sense: RowSense::GreaterEqual,
                rhs: 2.,
            },
            &[PresenceCoefficients {
                network: 7,
                row: PresenceRow::new(2, vec![(0, 1)], Mode::AnyPair).unwrap(),
                arc_events: vec![vec![0], vec![1]],
            }],
            &[(0, 1.)],
        )
        .unwrap()
        .with_column_bounds(42, 0, bounds(0., 0.))
        .unwrap();
    let mut calls = 0;
    let done = session
        .begin_tree()
        .run(&tree_options(), &ImportOptions::default(), |point, _| {
            calls += 1;
            assert_eq!(point.kind(), ImportPricingKind::Feasibility);
            Ok(ColumnBatch {
                columns: vec![column(1, 2., 0)],
                complete: false,
            })
        })
        .unwrap();
    assert_eq!(calls, 1);
    assert_eq!(done.status, ImportTreeStatus::Optimal);
    assert_eq!(done.incumbent, 7.);
    assert_eq!(done.bound, 7.);
    assert_eq!(done.tree.solution().unwrap().routes.len(), 1);
    assert_eq!(done.tree.solution().unwrap().routes[0].quantity, 1.);
    optimal(&parent.solve(options()).unwrap().output, 105.);
}
