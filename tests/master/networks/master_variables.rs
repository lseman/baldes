use super::*;
use baldes::master::preprocessing::Bounds;
use model::{EnumerationOptions, MasterVariable};

fn optional_cover(sign: f64, integer: bool) -> MasterVariable {
    MasterVariable {
        id: 100,
        cost: 0.25 * sign,
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
        integer,
        entries: vec![(0, sign)],
    }
}

#[test]
fn independent_master_variables_link_both_networks_and_keep_siblings() {
    for sign in [1., -1.] {
        let root = model::solve_with_master_variables(
            triangle(),
            vec![optional_cover(sign, true)],
            options(),
        )
        .unwrap();
        optimal(&root, 1.25);
        assert_eq!(root.master_values, vec![(100, sign)]);
        let fixed = root
            .snapshot
            .with_master_variable_bounds(
                100,
                Bounds {
                    lower: 0.,
                    upper: 0.,
                },
            )
            .unwrap();
        optimal(&fixed.solve(options()).unwrap(), 1.5);
        let opposite = root
            .snapshot
            .with_master_variable_bounds(
                100,
                Bounds {
                    lower: sign,
                    upper: sign,
                },
            )
            .unwrap();
        optimal(&opposite.solve(options()).unwrap(), 1.25);
        // A network restriction reaches the same master variable and both
        // pricing domains; adding a nonlinear row must not treat static tags
        // as route-registry indices.
        let child = opposite
            .with_multiplicity(
                7,
                Bounds {
                    lower: 0.,
                    upper: 0.,
                },
            )
            .unwrap();
        optimal(&child.solve(options()).unwrap(), 2.25);
        optimal(&root.snapshot.solve(options()).unwrap(), 1.25);
        let complete = fixed
            .enumerate(&EnumerationOptions::default())
            .unwrap()
            .preprocess()
            .unwrap();
        optimal(&complete.solve(options()).unwrap(), 1.5);
    }
}

#[test]
fn continuous_master_values_and_integer_route_mip_are_distinct() {
    let root =
        model::solve_with_master_variables(triangle(), vec![optional_cover(1., false)], options())
            .unwrap();
    let half = root
        .snapshot
        .with_master_variable_bounds(
            100,
            Bounds {
                lower: 0.5,
                upper: 0.5,
            },
        )
        .unwrap();
    let output = half.solve(options()).unwrap();
    optimal(&output, 1.375);
    assert_eq!(output.master_values, vec![(100, 0.5)]);
    #[cfg(feature = "highs")]
    {
        let complete = half.enumerate(&EnumerationOptions::default()).unwrap();
        let result = complete
            .solve_enumerated_mip(&Default::default())
            .unwrap()
            .unwrap();
        // Integer routes cannot supply exactly half a unit of row zero, so
        // this mixed domain is infeasible despite its feasible LP relaxation.
        assert_eq!(
            result.mip.status,
            baldes::master::solvers::mip::MipStatus::Infeasible
        );
        assert!(result.mip.solution.is_none());
        assert!(result.master_values.is_empty());
    }
}

#[test]
fn static_master_variable_certificate_compensates_negative_reduced_cost() {
    let mut configured = options();
    configured.cg.bound_mode = BoundMode::Certified;
    // A negative objective variable absent from all cover rows must contribute
    // -3 to a certificate; route-only corrections would incorrectly report 1.5.
    let variable = MasterVariable {
        id: 5,
        cost: -3.,
        bounds: Bounds {
            lower: 0.,
            upper: 1.,
        },
        integer: true,
        entries: vec![],
    };
    let output =
        model::solve_with_master_variables(triangle(), vec![variable], configured).unwrap();
    optimal(&output, -1.5);
    assert!(output.master.certified_bound <= -1.5);
    assert_eq!(output.master_values, vec![(5, 1.)]);
}

#[test]
fn static_integer_values_survive_enumeration_mip_and_assignment_tree() {
    let root =
        model::solve_with_master_variables(triangle(), vec![optional_cover(1., true)], options())
            .unwrap();
    #[cfg(feature = "highs")]
    {
        let complete = root
            .snapshot
            .enumerate(&EnumerationOptions::default())
            .unwrap();
        let mip = complete
            .solve_enumerated_mip(&Default::default())
            .unwrap()
            .unwrap();
        assert!((mip.mip.solution.unwrap().objective - 1.25).abs() < 1e-8);
        assert_eq!(mip.master_values, vec![(100, 1.)]);
    }
    let mut config = baldes::master::tree::ModelTreeOptions::default();
    config.node = options();
    let tree =
        baldes::master::tree::solve((triangle(), vec![optional_cover(1., true)]), config).unwrap();
    assert_eq!(tree.status, baldes::master::tree::TreeStatus::Optimal);
    assert_eq!(tree.incumbent, 1.25);
    assert_eq!(tree.master_values, vec![(100, 1.)]);
    optimal(&root.snapshot.solve(options()).unwrap(), 1.25);
}

#[test]
fn independent_master_fixing_propagates_through_a_shared_nonlinear_row() {
    use model::RankOneCoefficients;
    let problem = Problem {
        rows: (0..4)
            .map(|_| Row {
                sense: RowSense::Equal,
                rhs: 1.,
            })
            .collect(),
        networks: vec![nonlinear_network(7, 0), nonlinear_network(42, 2)],
    };
    let master = MasterVariable {
        id: 1,
        cost: 0.25,
        bounds: Bounds {
            lower: 0.,
            upper: 1.,
        },
        integer: true,
        entries: vec![],
    };
    let root = model::solve_with_master_variables(problem, vec![master], options()).unwrap();
    optimal(&root, 4.);
    let floors = [7, 42].map(|network| RankOneCoefficients {
        network,
        weights: vec![0, 1, 1, 0],
        denominator: 2,
    });
    let linked = root
        .snapshot
        .with_rank_one_master_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 0.,
            },
            &floors,
            &[(1, -1.)],
        )
        .unwrap();
    let output = linked.solve(options()).unwrap();
    // y=0 needs four singletons (8). y=1 permits one pair and two singletons
    // (6 + 1/4). Enumerating both choices independently proves 25/4.
    optimal(&output, 6.25);
    assert_eq!(output.master_values, vec![(1, 1.)]);
    let zero = linked
        .with_master_variable_bounds(
            1,
            Bounds {
                lower: 0.,
                upper: 0.,
            },
        )
        .unwrap();
    optimal(&zero.solve(options()).unwrap(), 8.);
    let pair = output
        .routes
        .iter()
        .position(|r| r.network == 7 && r.route.arcs.len() == 3)
        .unwrap();
    let fixed = output.snapshot.fix_route(pair, 1.).unwrap();
    optimal(&fixed.solve(options()).unwrap(), 6.25);
    let complete = fixed
        .enumerate(&EnumerationOptions::default())
        .unwrap()
        .preprocess()
        .unwrap();
    optimal(&complete.solve(options()).unwrap(), 6.25);
    #[cfg(feature = "highs")]
    {
        let mip = complete
            .solve_enumerated_mip(&Default::default())
            .unwrap()
            .unwrap();
        assert_eq!(mip.mip.solution.unwrap().objective, 6.25);
        assert_eq!(mip.master_values, vec![(1, 1.)]);
    }
    optimal(&root.snapshot.solve(options()).unwrap(), 4.);
    optimal(&linked.solve(options()).unwrap(), 6.25);
}

#[test]
fn integer_domain_policy_proves_mixed_cover_contradiction_without_changing_default_lp() {
    let mut variable = optional_cover(1., false);
    variable.bounds = Bounds {
        lower: 0.5,
        upper: 0.5,
    };
    let root =
        model::solve_with_master_variables(triangle(), vec![variable.clone()], options()).unwrap();
    optimal(&root, 1.375);
    assert!(!root.snapshot.preprocessing_report().propagation.infeasible);
    let mut config = baldes::master::tree::ModelTreeOptions::default();
    config.node = options();
    config.node.preprocessing.integer_domains = true;
    let tree = baldes::master::tree::solve((triangle(), vec![variable]), config).unwrap();
    assert_eq!(tree.status, baldes::master::tree::TreeStatus::Infeasible);
    assert_eq!(tree.open_nodes, 0);
    assert!(tree.master_values.is_empty());
    assert!(tree.incumbent.is_infinite());
}

#[test]
fn mixed_mip_retains_a_feasible_fractional_continuous_master_value() {
    let variable = MasterVariable {
        id: 9,
        cost: 0.25,
        bounds: Bounds {
            lower: 0.5,
            upper: 0.5,
        },
        integer: false,
        entries: vec![(0, 2.)],
    };
    let root =
        model::solve_with_master_variables(triangle(), vec![variable.clone()], options()).unwrap();
    optimal(&root, 1.125);
    assert_eq!(root.master_values, vec![(9, 0.5)]);
    #[cfg(feature = "highs")]
    {
        let complete = root
            .snapshot
            .enumerate(&EnumerationOptions::default())
            .unwrap()
            .preprocess()
            .unwrap();
        let mip = complete
            .solve_enumerated_mip(&Default::default())
            .unwrap()
            .unwrap();
        assert_eq!(
            mip.mip.status,
            baldes::master::solvers::mip::MipStatus::Optimal
        );
        assert_eq!(mip.mip.solution.unwrap().objective, 1.125);
        assert_eq!(mip.master_values, vec![(9, 0.5)]);
    }
    let mut config = baldes::master::tree::ModelTreeOptions::default();
    config.node = options();
    config.node.preprocessing.integer_domains = true;
    let tree = baldes::master::tree::solve((triangle(), vec![variable]), config).unwrap();
    assert_eq!(tree.status, baldes::master::tree::TreeStatus::Optimal);
    assert_eq!(tree.incumbent, 1.125);
    assert_eq!(tree.master_values, vec![(9, 0.5)]);
}

#[test]
fn master_variable_validation_does_not_confuse_id_spaces_or_partial_definitions() {
    let v = optional_cover(1., true);
    assert!(
        model::solve_with_master_variables(triangle(), vec![v.clone(), v.clone()], options())
            .is_err()
    );
    let mut invalid = v.clone();
    invalid.entries.push((0, 2.));
    assert!(model::solve_with_master_variables(triangle(), vec![invalid], options()).is_err());
    let root = model::solve_with_master_variables(triangle(), vec![v], options()).unwrap();
    assert!(root
        .snapshot
        .with_master_variable_bounds(
            7,
            Bounds {
                lower: 0.,
                upper: 0.
            }
        )
        .is_err());
    assert!(root
        .snapshot
        .with_master_row(
            Row {
                sense: RowSense::Equal,
                rhs: 0.
            },
            &[],
            &[(100, f64::NAN)]
        )
        .is_err());
    assert!(root
        .snapshot
        .with_master_row(
            Row {
                sense: RowSense::Equal,
                rhs: 0.
            },
            &[],
            &[(100, 1.), (100, 1.)]
        )
        .is_err());
    optimal(&root.snapshot.solve(options()).unwrap(), 1.25);
}
