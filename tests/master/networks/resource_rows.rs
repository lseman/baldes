use super::*;
use baldes::cuts::knapsack::{KnapsackKind, RouteLoadKnapsackCut};
use baldes::master::preprocessing::Bounds;
use multi_pricing::{EnumerationOptions, MasterVariable, ResourceKnapsackCoefficients};

fn problem() -> Problem<2> {
    let mut networks = vec![
        network(7, &[(1., &[0]), (3., &[0])], 2),
        network(42, &[(1., &[1]), (3., &[1])], 2),
    ];
    for n in &mut networks {
        n.graph.arcs[0].consumption[0] = 2.;
        n.graph.arcs[2].consumption[0] = 1.;
    }
    Problem {
        rows: vec![
            Row {
                sense: RowSense::Equal,
                rhs: 1.
            };
            2
        ],
        networks,
    }
}
fn coefficients() -> Vec<ResourceKnapsackCoefficients> {
    [(7, 2., 3), (42, 1., 2)]
        .into_iter()
        .map(|(network, scale, threshold)| ResourceKnapsackCoefficients {
            network,
            resource: 0,
            scale,
            cut: RouteLoadKnapsackCut {
                kind: KnapsackKind::Steps,
                numerator: 1,
                denominator: 1,
                rhs: 1,
                lhs: 0.,
                violation: 0.,
                thresholds: vec![threshold],
                coefficients: std::collections::BTreeMap::from([(threshold, 1)]),
            },
        })
        .collect()
}
#[test]
fn shared_resource_steps_price_fix_enumerate_and_restore() {
    let root = multi_pricing::solve(problem(), options()).unwrap();
    optimal(&root, 2.);
    let row = Row {
        sense: RowSense::LessEqual,
        rhs: 1.,
    };
    let child = root
        .snapshot
        .with_resource_knapsack_row(row.clone(), &coefficients())
        .unwrap();
    optimal(&child.solve(options()).unwrap(), 4.);
    let cheap = root
        .routes
        .iter()
        .position(|r| r.network == 7 && r.route.arcs == [0, 1])
        .unwrap();
    let fixed = child.fix_route(cheap, 1.).unwrap();
    optimal(&fixed.solve(options()).unwrap(), 4.);
    let reordered = root
        .snapshot
        .fix_route(cheap, 1.)
        .unwrap()
        .with_resource_knapsack_row(row, &coefficients())
        .unwrap();
    optimal(&reordered.solve(options()).unwrap(), 4.);
    let complete = fixed
        .enumerate(&EnumerationOptions::default())
        .unwrap()
        .preprocess()
        .unwrap();
    let mut certified = options();
    certified.cg.bound_mode = BoundMode::Certified;
    optimal(&complete.solve(certified).unwrap(), 4.);
    #[cfg(feature = "highs")]
    assert_eq!(
        complete
            .solve_enumerated_mip(&Default::default())
            .unwrap()
            .unwrap()
            .mip
            .solution
            .unwrap()
            .objective,
        4.
    );
    optimal(&root.snapshot.solve(options()).unwrap(), 2.);
}
#[test]
fn resource_steps_propagate_with_unbounded_static_variables() {
    let root = multi_pricing::solve_with_master_variables(
        problem(),
        vec![MasterVariable {
            id: 99,
            cost: 0.25,
            integer: true,
            bounds: Bounds {
                lower: 0.,
                upper: f64::INFINITY,
            },
            entries: vec![],
        }],
        options(),
    )
    .unwrap();
    let linked = root
        .snapshot
        .with_resource_knapsack_master_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 0.,
            },
            &coefficients(),
            &[(99, -1.)],
        )
        .unwrap();
    let untouched = linked.preprocess().unwrap();
    assert_eq!(untouched.preprocessing_report().forbidden_arcs, 0);
    let compensated = linked
        .with_master_variable_bounds(
            99,
            Bounds {
                lower: 0.,
                upper: 1.,
            },
        )
        .unwrap()
        .preprocess()
        .unwrap();
    assert_eq!(compensated.preprocessing_report().forbidden_arcs, 0);
    optimal(&compensated.solve(options()).unwrap(), 4.25);
    let solved = linked.solve(options()).unwrap();
    optimal(&solved, 2.5);
    assert_eq!(solved.master_values, vec![(99, 2.)]);
    let fixed = linked
        .with_master_variable_bounds(
            99,
            Bounds {
                lower: 0.,
                upper: 0.,
            },
        )
        .unwrap();
    let propagated = fixed.preprocess().unwrap();
    assert!(!propagated.preprocessing_report().complete_domain);
    assert_eq!(propagated.preprocessing_report().forbidden_arcs, 4);
    assert!(propagated.preprocessing_report().fixed_columns >= 2);
    let mut original_domain = options();
    original_domain.cg.bound_mode = BoundMode::Certified;
    optimal(&propagated.solve(original_domain).unwrap(), 6.);
    optimal(&fixed.solve(options()).unwrap(), 6.);
    let complete = fixed
        .enumerate(&EnumerationOptions::default())
        .unwrap()
        .preprocess()
        .unwrap();
    assert!(complete.preprocessing_report().fixed_columns >= 2);
    let mut certified = options();
    certified.cg.bound_mode = BoundMode::Certified;
    optimal(&complete.solve(certified).unwrap(), 6.);
    optimal(&linked.solve(options()).unwrap(), 2.5);
}
#[test]
fn resource_steps_reject_unproved_load_and_row_contracts() {
    let root = multi_pricing::solve(problem(), options()).unwrap();
    assert!(root
        .snapshot
        .with_resource_knapsack_row(
            Row {
                sense: RowSense::GreaterEqual,
                rhs: 1.
            },
            &coefficients()
        )
        .is_err());
    let mut bad = coefficients();
    bad[0].scale = -1.;
    assert!(root
        .snapshot
        .with_resource_knapsack_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 1.
            },
            &bad
        )
        .is_err());
    let mut waiting = problem();
    waiting.networks[0].graph.arcs[0].bounds[0] = Some(Window::new(0.1, 100000.).unwrap());
    let root_waiting = multi_pricing::solve(waiting, options()).unwrap();
    assert!(root_waiting
        .snapshot
        .with_resource_knapsack_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 1.
            },
            &coefficients()
        )
        .is_err());
    let mut p = problem();
    p.networks[0].graph.arcs[0].consumption[0] = 0.1;
    let root = multi_pricing::solve(p, options()).unwrap();
    assert!(root
        .snapshot
        .with_resource_knapsack_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 1.
            },
            &coefficients()
        )
        .is_err());
}

#[test]
fn scheduled_resource_rows_replay_initial_usage_waiting_and_equality() {
    for variant in ["initial", "vertex_wait", "arc_wait", "equality"] {
        let mut p = problem();
        let mut c = coefficients();
        for (i, n) in p.networks.iter_mut().enumerate() {
            n.graph.initial[0] = Some(if i == 0 { 1. } else { 2. });
            c[i].cut.coefficients.clear();
            let threshold = if i == 0 { 5 } else { 4 };
            c[i].cut.coefficients.insert(threshold, 1);
            if variant == "vertex_wait" {
                n.graph.vertices[1].bounds[0] = Window::new(5., 10.).unwrap();
                c[i].cut.coefficients.clear();
                c[i].cut.coefficients.insert(if i == 0 { 9 } else { 5 }, 1);
            } else if variant == "arc_wait" {
                n.graph.arcs[0].bounds[0] = Some(Window::new(5., 10.).unwrap());
                c[i].cut.coefficients.clear();
                c[i].cut.coefficients.insert(if i == 0 { 9 } else { 5 }, 1);
            } else if variant == "equality" {
                n.graph.resources[0] = baldes::graph::ResourceKind::Equality;
            }
        }
        let mut forward = options();
        forward
            .pricing_options
            .push(("BALDESuseBidirectionalSearch".into(), 0.));
        let root = multi_pricing::solve(p, forward).unwrap();
        optimal(&root, 2.);
        let row = Row {
            sense: RowSense::LessEqual,
            rhs: 1.,
        };
        let child = root
            .snapshot
            .with_resource_knapsack_row(row.clone(), &c)
            .unwrap();
        let mut requested_backward = options();
        requested_backward
            .pricing_options
            .push(("BALDESuseBidirectionalSearch".into(), 1.));
        optimal(&child.solve(requested_backward).unwrap(), 4.);
        let cheap = root
            .routes
            .iter()
            .position(|r| r.network == 7 && r.route.arcs == [0, 1])
            .unwrap();
        let fixed = child.fix_route(cheap, 1.).unwrap();
        optimal(&fixed.solve(options()).unwrap(), 4.);
        let reordered = root
            .snapshot
            .fix_route(cheap, 1.)
            .unwrap()
            .with_resource_knapsack_row(row, &c)
            .unwrap();
        optimal(&reordered.solve(options()).unwrap(), 4.);
        let complete = fixed.enumerate(&EnumerationOptions::default()).unwrap();
        let mut certified = options();
        certified.cg.bound_mode = BoundMode::Certified;
        optimal(&complete.solve(certified).unwrap(), 4.);
        #[cfg(feature = "highs")]
        assert_eq!(
            complete
                .solve_enumerated_mip(&Default::default())
                .unwrap()
                .unwrap()
                .mip
                .solution
                .unwrap()
                .objective,
            4.
        );
        let zero = root
            .snapshot
            .with_resource_knapsack_row(
                Row {
                    sense: RowSense::LessEqual,
                    rhs: 0.,
                },
                &c,
            )
            .unwrap()
            .preprocess()
            .unwrap();
        assert_eq!(zero.preprocessing_report().forbidden_arcs, 4, "{variant}");
        let mut certified = options();
        certified.cg.bound_mode = BoundMode::Certified;
        optimal(&zero.solve(certified).unwrap(), 6.);
    }
}

#[test]
fn equality_enumeration_filters_paths_that_only_a_waiting_relaxation_can_use() {
    let mut p = problem();
    for (i, n) in p.networks.iter_mut().enumerate() {
        n.graph.resources[0] = baldes::graph::ResourceKind::Equality;
        n.graph.initial[0] = Some(if i == 0 { 1. } else { 2. });
        n.graph.vertices[1].bounds[0] = Window::new(5., 10.).unwrap();
    }
    let mut forward = options();
    forward
        .pricing_options
        .push(("BALDESuseBidirectionalSearch".into(), 0.));
    let root = multi_pricing::solve(p, forward).unwrap();
    optimal(&root, 6.);
    let complete = root
        .snapshot
        .enumerate(&EnumerationOptions::default())
        .unwrap();
    assert!(complete.enumeration_report().complete);
    assert_eq!(complete.enumeration_report().routes, 2);
    let mut certified = options();
    certified.cg.bound_mode = BoundMode::Certified;
    // The complete universe bypasses pricing, so this also works with the
    // default backward option: no equality relaxation enters the master.
    optimal(&complete.solve(certified).unwrap(), 6.);
    #[cfg(feature = "highs")]
    assert_eq!(
        complete
            .solve_enumerated_mip(&Default::default())
            .unwrap()
            .unwrap()
            .mip
            .solution
            .unwrap()
            .objective,
        6.
    );
}

#[test]
fn automatic_resource_flow_policy_reaches_a_cross_network_fixed_point_with_one_budget() {
    let root = multi_pricing::solve_with_master_variables(
        problem(),
        vec![MasterVariable {
            id: 99,
            cost: 0.,
            integer: false,
            bounds: Bounds {
                lower: 0.,
                upper: f64::INFINITY,
            },
            entries: vec![],
        }],
        options(),
    )
    .unwrap();
    let coeffs = coefficients();
    // First nonlinear row removes network 7's cheap route. Its original arc
    // activity then bounds y, enabling the second network's nonlinear fixing.
    let first = root
        .snapshot
        .with_resource_knapsack_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 0.,
            },
            &coeffs[..1],
        )
        .unwrap();
    let linked = first
        .with_master_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 0.,
            },
            &[RowCoefficients {
                network: 7,
                arcs: vec![-1., 0., 0., 0.],
                per_route: 0.,
            }],
            &[(99, 1.)],
        )
        .unwrap();
    let chain = linked
        .with_resource_knapsack_master_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 0.,
            },
            &coeffs[1..],
            &[(99, -1.)],
        )
        .unwrap();
    let propagated = chain.preprocess().unwrap();
    let report = propagated.preprocessing_report();
    assert!(!report.complete_domain);
    assert!(report.propagation.complete);
    assert!(!report.propagation.infeasible);
    assert_eq!(report.forbidden_arcs, 4);
    assert_eq!(
        report.master_bounds,
        vec![(
            99,
            Bounds {
                lower: 0.,
                upper: 0.
            }
        )]
    );
    let mut certified = options();
    certified.cg.bound_mode = BoundMode::Certified;
    optimal(&propagated.solve(certified).unwrap(), 6.);
    for budget in [0, 64, 128, 256, 512, 1024, 4096] {
        let partial = chain
            .preprocess_with_options(&multi_pricing::PreprocessingOptions {
                work_limit: budget,
                ..Default::default()
            })
            .unwrap();
        let r = partial.preprocessing_report();
        assert!(r.propagation.work_used <= budget);
        assert!(!r.propagation.infeasible);
        if r.propagation.complete {
            assert_eq!(r.forbidden_arcs, 4);
        }
        let mut disabled = options();
        disabled.preprocessing.enabled = false;
        optimal(&partial.solve(disabled).unwrap(), 6.);
    }
    optimal(&root.snapshot.solve(options()).unwrap(), 2.);
}
