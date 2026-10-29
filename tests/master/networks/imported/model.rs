use super::*;
use baldes::master::{preprocessing::Bounds, tree::TreeStatus};
use model::{
    imported::{self, Column, Subproblem},
    MasterVariable,
};

fn quantity() -> Bounds {
    Bounds {
        lower: 0.,
        upper: f64::INFINITY,
    }
}
fn column(id: i32, cost: f64, entries: Vec<(i32, f64)>) -> Column {
    Column {
        id,
        cost,
        entries,
        bounds: quantity(),
        integer: true,
    }
}
fn fixture() -> (Problem<2>, Vec<MasterVariable>, Vec<Subproblem>) {
    let problem = Problem {
        rows: vec![
            Row {
                sense: RowSense::GreaterEqual,
                rhs: 1.
            };
            2
        ],
        networks: vec![],
    };
    let variables = [i32::MAX, 3]
        .into_iter()
        .enumerate()
        .map(|(row, id)| MasterVariable {
            id,
            cost: 0.,
            bounds: quantity(),
            integer: false,
            entries: vec![(row, 1.)],
        })
        .collect();
    let groups = vec![
        Subproblem {
            id: 7,
            variables: vec![i32::MAX, 3],
            multiplicity: Bounds {
                lower: 0.,
                upper: 2.,
            },
            columns: vec![
                column(0, 1., vec![(i32::MAX, 1.)]),
                column(1, 4., vec![(3, 1.)]),
            ],
        },
        Subproblem {
            id: 42,
            variables: vec![i32::MAX, 3],
            multiplicity: Bounds {
                lower: 0.,
                upper: 2.,
            },
            columns: vec![
                column(0, 1., vec![(3, 1.)]),
                column(1, 3., vec![(i32::MAX, 1.)]),
            ],
        },
    ];
    (problem, variables, groups)
}

#[test]
fn imported_original_vectors_fix_columns_multiplicities_and_restore_the_parent() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let (problem, variables, groups) = fixture();
        let mut opts = options();
        opts.cg.bound_mode = mode;
        let root = imported::solve(problem, variables, groups, opts.clone()).unwrap();
        optimal(&root.output, 2.);
        assert!(root.output.routes.is_empty() && root.output.oracle_calls.is_empty());
        assert_eq!(root.values.original, vec![(i32::MAX, 1.), (3, 1.)]);
        assert_eq!(
            root.values
                .columns
                .iter()
                .map(|v| (v.subproblem, v.column, v.value))
                .collect::<Vec<_>>(),
            vec![(7, 0, 1.), (7, 1, 0.), (42, 0, 1.), (42, 1, 0.)]
        );
        let node = root.snapshot().unwrap();
        let excluded = node
            .with_column_bounds(
                7,
                0,
                Bounds {
                    lower: 0.,
                    upper: 0.,
                },
            )
            .unwrap();
        optimal(&excluded.solve(opts.clone()).unwrap().output, 4.);
        optimal(
            &node
                .with_multiplicity(
                    7,
                    Bounds {
                        lower: 0.,
                        upper: 0.,
                    },
                )
                .unwrap()
                .solve(opts.clone())
                .unwrap()
                .output,
            4.,
        );
        optimal(
            &node
                .with_variable_bounds(
                    i32::MAX,
                    Bounds {
                        lower: 2.,
                        upper: 2.,
                    },
                )
                .unwrap()
                .solve(opts.clone())
                .unwrap()
                .output,
            3.,
        );
        let fixed = node
            .with_column_bounds(
                7,
                0,
                Bounds {
                    lower: 1.,
                    upper: 1.,
                },
            )
            .unwrap();
        let both = fixed
            .with_multiplicity(
                7,
                Bounds {
                    lower: 2.,
                    upper: 2.,
                },
            )
            .unwrap();
        let solved = both.solve(opts.clone()).unwrap();
        optimal(&solved.output, 5.);
        assert_eq!(solved.values.columns[0].value, 1.);
        assert_eq!(solved.values.columns[1].value, 1.);
        #[cfg(feature = "highs")]
        {
            assert!(node.node().enumeration_report().complete);
            let mip = node
                .node()
                .solve_enumerated_mip(&Default::default())
                .unwrap()
                .unwrap();
            assert_eq!(mip.mip.solution.unwrap().objective, 2.);
            assert_eq!(node.values(&mip.master_values).unwrap(), root.values);
        }
        let tree = node.node().solve_tree(Default::default()).unwrap();
        assert_eq!(tree.status, TreeStatus::Optimal);
        assert_eq!(tree.incumbent, 2.);
        assert_eq!(node.values(&tree.master_values).unwrap(), root.values);
        optimal(&node.solve(opts).unwrap().output, 2.);
        assert!(node.with_column_bounds(42, 99, quantity()).is_err());
        assert!(node.with_variable_bounds(0, quantity()).is_err());
        assert!(node.values(&[]).is_err());
    }
}

#[test]
fn imported_columns_mix_with_real_routing_and_do_not_fake_complete_pricing() {
    let (mut problem, variables, groups) = fixture();
    problem.rows.push(Row {
        sense: RowSense::GreaterEqual,
        rhs: 1.,
    });
    problem.networks.push(network(99, &[(0.25, &[2])], 3));
    let root = imported::solve(problem, variables, groups, options()).unwrap();
    optimal(&root.output, 2.25);
    assert!(!root.output.oracle_calls.is_empty());
    assert!(!root.output.snapshot.enumeration_report().complete);
    let node = root.snapshot().unwrap();
    let complete = node.node().enumerate(&Default::default()).unwrap();
    assert!(complete.enumeration_report().complete);
    #[cfg(feature = "highs")]
    {
        let mip = complete
            .solve_enumerated_mip(&Default::default())
            .unwrap()
            .unwrap();
        assert_eq!(mip.mip.solution.unwrap().objective, 2.25);
        assert_eq!(node.values(&mip.master_values).unwrap(), root.values);
    }
    optimal(&node.solve(options()).unwrap().output, 2.25);
}

#[test]
fn signed_fractional_imports_keep_declared_costs_and_continuous_quantities() {
    let p = Problem::<7> {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: -1.5,
        }],
        networks: vec![],
    };
    let variables = vec![MasterVariable {
        id: 90,
        cost: 1.,
        bounds: Bounds {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
        },
        integer: false,
        entries: vec![(0, 1.)],
    }];
    let mut col = column(4, 0.5, vec![(90, -1.)]);
    col.integer = false;
    let groups = vec![Subproblem {
        id: 1,
        variables: vec![90],
        multiplicity: quantity(),
        columns: vec![col],
    }];
    let mut opts = options();
    opts.cg.bound_mode = BoundMode::Certified;
    let root = imported::solve(p, variables, groups, opts.clone()).unwrap();
    optimal(&root.output, -0.75);
    assert_eq!(root.values.original, vec![(90, -1.5)]);
    assert_eq!(root.values.columns[0].value, 1.5);
    let node = root
        .snapshot()
        .unwrap()
        .with_column_bounds(
            1,
            4,
            Bounds {
                lower: 1.5,
                upper: 1.5,
            },
        )
        .unwrap();
    optimal(&node.solve(opts).unwrap().output, -0.75);
    #[cfg(feature = "highs")]
    {
        let mip = node
            .node()
            .solve_enumerated_mip(&Default::default())
            .unwrap()
            .unwrap();
        assert_eq!(mip.mip.solution.unwrap().objective, -0.75);
        assert_eq!(node.values(&mip.master_values).unwrap(), root.values);
    }
}

#[test]
fn imports_validate_original_domains_before_generated_links_can_hide_errors() {
    let (p, v, g) = fixture();
    let mut bad = v.clone();
    bad[0].entries.push((2, 1.));
    assert!(imported::solve(p.clone(), bad, g.clone(), options()).is_err());
    for variant in 0..6 {
        let mut bad = g.clone();
        match variant {
            0 => bad[0].variables.push(i32::MAX),
            1 => bad[0].columns[0].entries.push((i32::MAX, 1.)),
            2 => bad[0].columns[0].entries.push((999, 1.)),
            3 => bad[0].columns[0].bounds.lower = -1.,
            4 => bad[0].columns[0].cost = f64::NAN,
            _ => bad[1].id = 7,
        }
        assert!(
            imported::solve(p.clone(), v.clone(), bad, options()).is_err(),
            "variant {variant}"
        );
    }
    let mut empty = g;
    empty[0].columns.clear();
    empty[1].columns.clear();
    let root = imported::solve(p, v, empty, options()).unwrap();
    assert!(
        root.output
            .snapshot
            .preprocessing_report()
            .propagation
            .infeasible
    );
    assert_eq!(
        root.snapshot()
            .unwrap()
            .node()
            .solve_tree(Default::default())
            .unwrap()
            .status,
        TreeStatus::Infeasible
    );
}

#[test]
fn imported_empty_coordinate_domains_keep_large_multiplicity_and_infinite_caps() {
    let amount = 2_f64.powi(41);
    let group = Subproblem {
        id: 5,
        variables: vec![],
        multiplicity: Bounds {
            lower: amount,
            upper: f64::INFINITY,
        },
        columns: vec![column(2, 1., vec![])],
    };
    for stabilize in [false, true] {
        let mut opts = options();
        opts.cg.bound_mode = BoundMode::Certified;
        opts.cg.stabilize = stabilize;
        let root = imported::solve(
            Problem::<2> {
                rows: vec![],
                networks: vec![],
            },
            vec![],
            vec![group.clone()],
            opts,
        )
        .unwrap();
        optimal(&root.output, amount);
        assert!(root.values.original.is_empty());
        assert_eq!(root.values.columns[0].value, amount);
        let node = root.snapshot().unwrap();
        assert!(node
            .with_multiplicity(
                5,
                Bounds {
                    lower: -1.,
                    upper: amount
                }
            )
            .is_err());
        assert!(node
            .with_column_bounds(
                5,
                2,
                Bounds {
                    lower: f64::INFINITY,
                    upper: f64::INFINITY
                }
            )
            .is_err());
    }
}

#[test]
fn imported_column_feature_rows_propagate_across_groups_after_fixing() {
    let p = Problem::<2> {
        rows: vec![
            Row {
                sense: RowSense::Equal,
                rhs: 1.
            };
            2
        ],
        networks: vec![],
    };
    let variables = [10, 11, 12]
        .into_iter()
        .enumerate()
        .map(|(i, id)| MasterVariable {
            id,
            cost: 0.,
            bounds: quantity(),
            integer: false,
            entries: if i < 2 { vec![(i, 1.)] } else { vec![] },
        })
        .collect();
    let groups = [(7, 10), (42, 11)]
        .into_iter()
        .map(|(id, x)| Subproblem {
            id,
            variables: vec![x, 12],
            multiplicity: quantity(),
            columns: vec![
                column(0, 1., vec![(x, 1.), (12, 1.)]),
                column(1, 3., vec![(x, 1.)]),
            ],
        })
        .collect();
    let root = imported::solve(p, variables, groups, options()).unwrap();
    optimal(&root.output, 2.);
    let parent = root.snapshot().unwrap();
    let limited = parent
        .with_variable_bounds(
            12,
            Bounds {
                lower: 0.,
                upper: 1.,
            },
        )
        .unwrap();
    optimal(&limited.solve(options()).unwrap().output, 4.);
    let fixed = limited
        .with_column_bounds(
            7,
            0,
            Bounds {
                lower: 1.,
                upper: 1.,
            },
        )
        .unwrap();
    let propagated = fixed.node().preprocess().unwrap();
    assert!(propagated.preprocessing_report().fixed_columns >= 2);
    let mut certified = options();
    certified.cg.bound_mode = BoundMode::Certified;
    let solved = propagated.solve(certified).unwrap();
    optimal(&solved, 4.);
    let values = fixed.values(&solved.master_values).unwrap();
    assert_eq!(
        values
            .columns
            .iter()
            .find(|v| v.subproblem == 42 && v.column == 0)
            .unwrap()
            .value,
        0.
    );
    #[cfg(feature = "highs")]
    assert_eq!(
        propagated
            .solve_enumerated_mip(&Default::default())
            .unwrap()
            .unwrap()
            .mip
            .solution
            .unwrap()
            .objective,
        4.
    );
    optimal(&parent.solve(options()).unwrap().output, 2.);
}
