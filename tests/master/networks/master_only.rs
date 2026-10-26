use super::*;
use baldes::master::tree::ModelTreeOptions;
use baldes::master::{preprocessing::Bounds, tree::TreeStatus};
use model::MasterVariable;

fn variable(
    id: i32,
    cost: f64,
    lower: f64,
    upper: f64,
    integer: bool,
    entries: Vec<(usize, f64)>,
) -> MasterVariable {
    MasterVariable {
        id,
        cost,
        bounds: Bounds { lower, upper },
        integer,
        entries,
    }
}

#[test]
fn master_only_signed_free_variables_fix_and_restore_without_a_dummy_network() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let mut opts = options();
        opts.cg.bound_mode = mode;
        let root = model::solve_master(
            vec![Row {
                sense: RowSense::Equal,
                rhs: -1.5,
            }],
            vec![
                variable(
                    7,
                    1.,
                    f64::NEG_INFINITY,
                    f64::INFINITY,
                    false,
                    vec![(0, 1.)],
                ),
                variable(42, 2., 0., 2., false, vec![(0, -1.)]),
            ],
            opts.clone(),
        )
        .unwrap();
        optimal(&root, -1.5);
        assert!(root.routes.is_empty() && root.oracle_calls.is_empty());
        assert_eq!(root.master_values, vec![(7, -1.5), (42, 0.)]);
        assert!(root.snapshot.enumeration_report().complete);
        assert_eq!(root.snapshot.enumeration_report().attempts, 0);
        assert!(root.snapshot.preprocessing_report().complete_domain);
        let fixed = root
            .snapshot
            .with_master_variable_bounds(
                42,
                Bounds {
                    lower: 1.,
                    upper: 1.,
                },
            )
            .unwrap();
        optimal(&fixed.solve(opts.clone()).unwrap(), 1.5);
        let sibling = root
            .snapshot
            .with_master_variable_bounds(
                7,
                Bounds {
                    lower: -1.,
                    upper: -1.,
                },
            )
            .unwrap();
        optimal(&sibling.solve(opts.clone()).unwrap(), 0.);
        optimal(&root.snapshot.solve(opts).unwrap(), -1.5);
    }
}

#[test]
fn master_only_integer_policy_tree_and_mip_share_the_complete_domain() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let mut opts = options();
        opts.cg.bound_mode = mode;
        let root = model::solve_master(
            vec![Row {
                sense: RowSense::GreaterEqual,
                rhs: 0.5,
            }],
            vec![variable(9, 1., 0., 3., true, vec![(0, 1.)])],
            opts.clone(),
        )
        .unwrap();
        optimal(&root, 0.5);
        let mut integer_policy = model::PreprocessingOptions::default();
        integer_policy.integer_domains = true;
        let integral = root
            .snapshot
            .preprocess_with_options(&integer_policy)
            .unwrap();
        optimal(&integral.solve(opts.clone()).unwrap(), 1.);
        let tree = root
            .snapshot
            .solve_tree(ModelTreeOptions {
                node: opts.clone(),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(tree.status, TreeStatus::Optimal);
        assert_eq!(tree.incumbent, 1.);
        assert_eq!(tree.master_values, vec![(9, 1.)]);
        assert_eq!(tree.open_nodes, 0);
        assert!(tree.oracle_calls.is_empty());
        #[cfg(feature = "highs")]
        {
            let mip = root
                .snapshot
                .solve_enumerated_mip(&Default::default())
                .unwrap()
                .unwrap();
            assert_eq!(mip.mip.solution.unwrap().objective, 1.);
            assert_eq!(mip.master_values, vec![(9, 1.)]);
            assert!(mip.route_values.is_empty());
        }
        optimal(&root.snapshot.solve(opts).unwrap(), 0.5);
    }
}

#[test]
fn empty_master_domain_and_original_infeasibility_have_explicit_outcomes() {
    let empty = model::solve_master(vec![], vec![], options()).unwrap();
    optimal(&empty, 0.);
    assert!(empty.master_values.is_empty());
    let impossible = model::solve_master(
        vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        vec![],
        options(),
    )
    .unwrap();
    assert!(
        impossible
            .snapshot
            .preprocessing_report()
            .propagation
            .infeasible
    );
    let tree = impossible.snapshot.solve_tree(Default::default()).unwrap();
    assert_eq!(tree.status, TreeStatus::Infeasible);
    assert_eq!(tree.open_nodes, 0);
    let malformed = model::solve_master(
        vec![Row {
            sense: RowSense::Equal,
            rhs: f64::NAN,
        }],
        vec![],
        options(),
    );
    assert!(malformed.is_err());
}

#[test]
fn master_only_preserves_unbounded_directions_and_arbitrary_snapshot_dimensions() {
    let unbounded = model::solve_master(
        vec![],
        vec![variable(
            3,
            -1.,
            f64::NEG_INFINITY,
            f64::INFINITY,
            false,
            vec![],
        )],
        options(),
    )
    .unwrap();
    assert_eq!(unbounded.master.status, Status::Unbounded);
    let p = Problem::<7> {
        rows: vec![],
        networks: vec![],
    };
    let bounded = model::solve_with_master_variables(
        p,
        vec![variable(3, 1., -2., 3., false, vec![])],
        options(),
    )
    .unwrap();
    optimal(&bounded, -2.);
    assert!(bounded.oracle_calls.is_empty());
}

#[test]
fn inferred_integer_bounds_publish_both_signs_in_fresh_and_warm_masters() {
    for negative in [false, true] {
        let (sense, rhs, cost, lower, upper) = if negative {
            (RowSense::LessEqual, -0.5, -1., -3., 0.)
        } else {
            (RowSense::GreaterEqual, 0.5, 1., 0., 3.)
        };
        for routing in [false, true] {
            let p = Problem::<2> {
                rows: vec![Row { sense, rhs }],
                networks: if routing {
                    vec![network(7, &[(0., &[])], 1)]
                } else {
                    vec![]
                },
            };
            let vars = vec![variable(9, cost, lower, upper, true, vec![(0, 1.)])];
            let root =
                model::solve_with_master_variables(p.clone(), vars.clone(), options()).unwrap();
            optimal(&root, 0.5);
            let mut integer_options = options();
            integer_options.preprocessing.integer_domains = true;
            let fresh =
                model::solve_with_master_variables(p, vars, integer_options.clone()).unwrap();
            optimal(&fresh, 1.);
            let warm = root.snapshot.solve(integer_options).unwrap();
            optimal(&warm, 1.);
            let policy = model::PreprocessingOptions {
                integer_domains: true,
                ..Default::default()
            };
            let child = root.snapshot.preprocess_with_options(&policy).unwrap();
            optimal(&child.solve(options()).unwrap(), 1.);
            let mut certified = options();
            certified.cg.bound_mode = BoundMode::Certified;
            optimal(&child.solve(certified).unwrap(), 1.);
            #[cfg(feature = "highs")]
            {
                let complete = child.enumerate(&Default::default()).unwrap();
                assert_eq!(
                    complete
                        .solve_enumerated_mip(&Default::default())
                        .unwrap()
                        .unwrap()
                        .mip
                        .solution
                        .unwrap()
                        .objective,
                    1.
                );
            }
            optimal(&root.snapshot.solve(options()).unwrap(), 0.5);
        }
    }
}

#[test]
fn master_only_original_multiplicity_links_propagate_and_survive_fixing() {
    let rows = vec![
        Row {
            sense: RowSense::GreaterEqual,
            rhs: 1.5,
        },
        Row {
            sense: RowSense::Equal,
            rhs: 0.,
        },
    ];
    let variables = vec![
        variable(1, 1., 0., f64::INFINITY, true, vec![(0, 1.), (1, -1.)]),
        variable(2, 2., 0., f64::INFINITY, true, vec![(0, 1.), (1, -1.)]),
        variable(3, 0., 0., f64::INFINITY, true, vec![(1, 1.)]),
    ];
    let mut opts = options();
    opts.cg.bound_mode = BoundMode::Certified;
    let root = model::solve_master(rows, variables, opts.clone()).unwrap();
    optimal(&root, 1.5);
    assert_eq!(root.master_values, vec![(1, 1.5), (2, 0.), (3, 1.5)]);
    let policy = model::PreprocessingOptions {
        integer_domains: true,
        ..Default::default()
    };
    let propagated = root.snapshot.preprocess_with_options(&policy).unwrap();
    assert_eq!(
        propagated
            .preprocessing_report()
            .master_bounds
            .iter()
            .find(|(id, _)| *id == 3)
            .unwrap()
            .1
            .lower,
        2.
    );
    optimal(&propagated.solve(opts.clone()).unwrap(), 2.);
    let fixed = root
        .snapshot
        .with_master_variable_bounds(
            3,
            Bounds {
                lower: 3.,
                upper: 3.,
            },
        )
        .unwrap();
    optimal(&fixed.solve(opts.clone()).unwrap(), 3.);
    let tree = root
        .snapshot
        .solve_tree(ModelTreeOptions {
            node: opts.clone(),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(tree.status, TreeStatus::Optimal);
    assert_eq!(tree.incumbent, 2.);
    optimal(&root.snapshot.solve(opts).unwrap(), 1.5);
}
