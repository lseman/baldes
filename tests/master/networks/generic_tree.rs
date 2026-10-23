use super::*;
use baldes::master::{preprocessing::Bounds, tree::TreeStatus};
use mp::{AssignmentTreeOptions, MasterVariable};

#[test]
fn tree_from_snapshot_branches_signed_integer_master_variables() {
    for sign in [1., -1.] {
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            let p = Problem {
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
                networks: vec![network(7, &[(0., &[0])], 2)],
            };
            let variable = MasterVariable {
                id: 99,
                cost: sign,
                integer: true,
                bounds: if sign > 0. {
                    Bounds {
                        lower: 0.,
                        upper: 1.,
                    }
                } else {
                    Bounds {
                        lower: -1.,
                        upper: 0.,
                    }
                },
                entries: vec![(1, sign)],
            };
            let mut configured = options();
            configured.cg.bound_mode = mode;
            let root =
                mp::solve_with_master_variables(p, vec![variable], configured.clone()).unwrap();
            optimal(&root, 0.5);
            let mut zero = AssignmentTreeOptions {
                node: configured.clone(),
                node_limit: 0,
                ..Default::default()
            };
            zero.node.cg.bound_mode = BoundMode::Certified;
            let stopped = root.snapshot.solve_assignment_tree(zero).unwrap();
            assert_eq!(stopped.bound, root.master.certified_bound);
            assert_eq!(stopped.open_nodes, 1);

            let tree = root
                .snapshot
                .solve_assignment_tree(AssignmentTreeOptions {
                    node: configured.clone(),
                    ..Default::default()
                })
                .unwrap();
            assert_eq!(tree.status, TreeStatus::Optimal);
            assert_eq!(tree.incumbent, 1.);
            assert_eq!(tree.master_values, vec![(99, sign)]);
            assert_eq!(tree.nodes, 3);
            optimal(&root.snapshot.solve(configured).unwrap(), 0.5);
        }
    }
}

#[test]
fn tree_branches_multiplicity_and_original_arc_activity_without_assignment_candidates() {
    for arcs in [false, true] {
        let p = if arcs {
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
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            let mut configured = options();
            configured.cg.bound_mode = mode;
            let root = mp::solve(p.clone(), configured.clone()).unwrap();
            optimal(&root, 0.5);
            assert!(root
                .snapshot
                .fractional_assignments(1e-6)
                .unwrap()
                .is_empty());
            let tree = root
                .snapshot
                .solve_assignment_tree(AssignmentTreeOptions {
                    node: configured.clone(),
                    ..Default::default()
                })
                .unwrap();
            assert_eq!(tree.status, TreeStatus::Optimal);
            assert_eq!(tree.incumbent, 1.);
            assert_eq!(tree.nodes, 3);
            let limited = root
                .snapshot
                .solve_assignment_tree(AssignmentTreeOptions {
                    node: configured.clone(),
                    node_limit: 1,
                    ..Default::default()
                })
                .unwrap();
            assert_eq!(limited.status, TreeStatus::NodeLimit);
            assert_eq!(limited.open_nodes, 2);
            assert!((limited.bound - 0.5).abs() < 1e-8);
            optimal(&root.snapshot.solve(configured).unwrap(), 0.5);
        }
    }
}

#[test]
fn continuous_master_values_do_not_trigger_integer_branching() {
    let p = Problem {
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
        networks: vec![network(7, &[(0., &[0])], 2)],
    };
    let tree = mp::solve_assignment_tree_with_master_variables(
        p,
        vec![MasterVariable {
            id: 99,
            cost: 1.,
            integer: false,
            bounds: Bounds {
                lower: 0.,
                upper: 1.,
            },
            entries: vec![(1, 1.)],
        }],
        AssignmentTreeOptions {
            node: options(),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(tree.status, TreeStatus::Optimal);
    assert_eq!(tree.incumbent, 0.5);
    assert_eq!(tree.nodes, 1);
}
