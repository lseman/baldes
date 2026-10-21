use super::*;
use baldes::cuts::soft_conflicts::{Mode, PresenceRow};
use multi_pricing::{EnumerationOptions, PresenceCoefficients};

fn problem() -> Problem<2> {
    Problem {
        rows: vec![
            Row {
                sense: RowSense::Equal,
                rhs: 1.
            };
            2
        ],
        networks: vec![
            network(7, &[(1., &[0]), (3., &[0])], 2),
            network(42, &[(1., &[1]), (3., &[1])], 2),
        ],
    }
}
fn coefficients(mode: Mode) -> Vec<PresenceCoefficients> {
    [7, 42]
        .into_iter()
        .map(|network| PresenceCoefficients {
            network,
            row: PresenceRow::new(2, vec![(0, 1)], mode).unwrap(),
            arc_events: vec![vec![0, 1, 0], vec![], vec![0], vec![]],
        })
        .collect()
}

fn priced(output: &Output, expected: f64, mode: BoundMode) {
    if mode == BoundMode::Numerical {
        optimal(output, expected);
    } else {
        // The original-domain relaxation may drop a positive presence penalty.
        // A loose proof must remain Stalled, never promote the numerical LP.
        assert!(matches!(
            output.master.status,
            Status::Optimal | Status::Stalled
        ));
        assert!((output.master.lp - expected).abs() < 1e-7);
        assert!(output.master.certified_bound.is_finite());
        assert!(output.master.certified_bound <= expected + 1e-7);
    }
}

#[test]
fn presence_shared_rows_price_fix_enumerate_and_restore() {
    for mode in [Mode::PairCount, Mode::AnyPair] {
        for bound_mode in [BoundMode::Certified, BoundMode::Numerical] {
            let mut opts = options();
            opts.cg.bound_mode = bound_mode;
            let root = multi_pricing::solve(problem(), opts.clone()).unwrap();
            optimal(&root, 2.);
            let row = Row {
                sense: RowSense::LessEqual,
                rhs: 1.,
            };
            let child = root
                .snapshot
                .with_presence_row(row.clone(), &coefficients(mode))
                .unwrap();
            priced(&child.solve(opts.clone()).unwrap(), 4., bound_mode);
            let cheap = root
                .routes
                .iter()
                .position(|r| r.network == 7 && r.route.arcs == [0, 1])
                .unwrap();
            let fixed = child.fix_route(cheap, 1.).unwrap();
            priced(&fixed.solve(opts.clone()).unwrap(), 4., bound_mode);
            let reordered = root
                .snapshot
                .fix_route(cheap, 1.)
                .unwrap()
                .with_presence_row(row, &coefficients(mode))
                .unwrap();
            priced(&reordered.solve(opts.clone()).unwrap(), 4., bound_mode);
            let complete = fixed
                .enumerate(&EnumerationOptions::default())
                .unwrap()
                .preprocess()
                .unwrap();
            optimal(&complete.solve(opts.clone()).unwrap(), 4.);
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
            optimal(&root.snapshot.solve(opts).unwrap(), 2.);
        }
    }
}

#[test]
fn presence_lower_and_equal_rows_keep_the_original_flow_relaxation_valid() {
    for sense in [RowSense::GreaterEqual, RowSense::Equal] {
        let root = multi_pricing::solve(problem(), options()).unwrap();
        let child = root
            .snapshot
            .with_presence_row(Row { sense, rhs: 2. }, &coefficients(Mode::PairCount))
            .unwrap();
        optimal(&child.solve(options()).unwrap(), 2.);
    }
}

#[test]
fn presence_propagation_respects_signed_master_compensation_and_budget() {
    use baldes::master::preprocessing::Bounds;
    use multi_pricing::MasterVariable;
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
        .with_presence_master_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 0.,
            },
            &coefficients(Mode::AnyPair),
            &[(99, -1.)],
        )
        .unwrap();
    assert_eq!(
        linked
            .preprocess()
            .unwrap()
            .preprocessing_report()
            .forbidden_arcs,
        0
    );
    optimal(&linked.solve(options()).unwrap(), 2.5);
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
    assert_eq!(propagated.preprocessing_report().forbidden_arcs, 4);
    assert!(propagated.preprocessing_report().propagation.complete);
    let mut certified = options();
    certified.cg.bound_mode = BoundMode::Certified;
    optimal(&propagated.solve(certified).unwrap(), 6.);
    let budget = multi_pricing::PreprocessingOptions {
        work_limit: 1,
        ..Default::default()
    };
    let limited = fixed.preprocess_with_options(&budget).unwrap();
    assert_eq!(limited.preprocessing_report().forbidden_arcs, 0);
    assert!(limited.preprocessing_report().propagation.work_used <= 1);
    optimal(&root.snapshot.solve(options()).unwrap(), 2.);
}

#[test]
fn shared_presence_modes_distinguish_duplicate_pairs() {
    let root = multi_pricing::solve(problem(), options()).unwrap();
    for (mode, expected) in [(Mode::PairCount, 4.), (Mode::AnyPair, 2.)] {
        let mut coefficients = coefficients(mode);
        for c in &mut coefficients {
            c.row = PresenceRow::new(2, vec![(0, 1), (0, 1)], mode).unwrap();
        }
        let child = root
            .snapshot
            .with_presence_row(
                Row {
                    sense: RowSense::LessEqual,
                    rhs: 2.,
                },
                &coefficients,
            )
            .unwrap();
        optimal(&child.solve(options()).unwrap(), expected);
    }
    let mut bad = coefficients(Mode::AnyPair);
    bad[0].arc_events[0].push(2);
    assert!(root
        .snapshot
        .with_presence_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 1.
            },
            &bad
        )
        .is_err());
    optimal(&root.snapshot.solve(options()).unwrap(), 2.);
}

#[test]
fn presence_rows_apply_before_any_route_is_priced() {
    let mut cold = options();
    cold.cg.time_limit = std::time::Duration::ZERO;
    let root = multi_pricing::solve(problem(), cold).unwrap();
    assert_eq!(root.master.pricing_calls, 0);
    let child = root
        .snapshot
        .with_presence_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 1.,
            },
            &coefficients(Mode::AnyPair),
        )
        .unwrap();
    optimal(&child.solve(options()).unwrap(), 4.);
}

#[test]
fn exhausted_presence_rows_propagate_conflicts_across_separate_arcs() {
    for (mode, sense) in [
        (Mode::PairCount, RowSense::LessEqual),
        (Mode::AnyPair, RowSense::LessEqual),
        (Mode::PairCount, RowSense::Equal),
        (Mode::AnyPair, RowSense::Equal),
    ] {
        let root = multi_pricing::solve(problem(), options()).unwrap();
        let mut emissions = coefficients(mode);
        for c in &mut emissions {
            c.arc_events[0] = vec![0];
            c.arc_events[1] = vec![1];
        }
        let constrained = root
            .snapshot
            .with_presence_row(Row { sense, rhs: 0. }, &emissions)
            .unwrap();
        let propagated = constrained.preprocess().unwrap();
        assert!(propagated.preprocessing_report().propagation.complete);
        assert_eq!(propagated.preprocessing_report().forbidden_arcs, 4);
        for n in &propagated.preprocessing_report().networks {
            assert_eq!(n.arcs[0].upper, 0.);
            assert_eq!(n.arcs[1].upper, 0.);
            assert!(n.arcs[2].upper > 0.);
            assert!(n.arcs[3].upper > 0.);
        }
        let mut certified = options();
        certified.cg.bound_mode = BoundMode::Certified;
        optimal(&propagated.solve(certified).unwrap(), 6.);
        optimal(&root.snapshot.solve(options()).unwrap(), 2.);
    }
}

#[test]
fn positive_fractional_presence_budget_keeps_fractional_route_activity() {
    let root = multi_pricing::solve(problem(), options()).unwrap();
    let mut emissions = coefficients(Mode::AnyPair);
    for c in &mut emissions {
        c.arc_events[0] = vec![0];
        c.arc_events[1] = vec![1];
    }
    let child = root
        .snapshot
        .with_presence_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 0.5,
            },
            &emissions,
        )
        .unwrap()
        .preprocess()
        .unwrap();
    assert_eq!(child.preprocessing_report().forbidden_arcs, 0);
    optimal(&child.solve(options()).unwrap(), 5.);
}

#[test]
fn presence_walk_support_reaches_a_cross_network_static_link_fixed_point() {
    use baldes::master::preprocessing::Bounds;
    use multi_pricing::MasterVariable;
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
    let mut emissions = coefficients(Mode::AnyPair);
    for c in &mut emissions {
        c.arc_events[0] = vec![0];
        c.arc_events[1] = vec![1];
    }
    let linked = root
        .snapshot
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
        .unwrap()
        .with_presence_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 0.,
            },
            &emissions[..1],
        )
        .unwrap()
        .with_presence_master_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 0.,
            },
            &emissions[1..],
            &[(99, -1.)],
        )
        .unwrap();
    let propagated = linked.preprocess().unwrap();
    assert!(propagated.preprocessing_report().propagation.complete);
    assert_eq!(propagated.preprocessing_report().forbidden_arcs, 4);
    assert_eq!(
        propagated.preprocessing_report().master_bounds[0].1.upper,
        0.
    );
    let mut certified = options();
    certified.cg.bound_mode = BoundMode::Certified;
    optimal(&propagated.solve(certified).unwrap(), 6.);
    optimal(&root.snapshot.solve(options()).unwrap(), 2.);
}

#[test]
fn joint_presence_certificates_close_shared_roots_and_fixed_descendants_without_enumeration() {
    for mode in [Mode::PairCount, Mode::AnyPair] {
        let mut configured = options();
        configured.cg.bound_mode = BoundMode::Certified;
        let root = multi_pricing::solve(problem(), configured.clone()).unwrap();
        let node = root
            .snapshot
            .with_presence_row(
                Row {
                    sense: RowSense::LessEqual,
                    rhs: 1.,
                },
                &coefficients(mode),
            )
            .unwrap();
        let solved = node.solve(configured.clone()).unwrap();
        optimal(&solved, 4.);
        assert!(solved.snapshot.enumeration_report().networks.is_empty());
        let cheap = root
            .routes
            .iter()
            .position(|r| r.network == 7 && r.route.arcs == [0, 1])
            .unwrap();
        optimal(
            &node
                .fix_route(cheap, 1.)
                .unwrap()
                .solve(configured.clone())
                .unwrap(),
            4.,
        );
        let tree = node
            .solve_assignment_tree(multi_pricing::AssignmentTreeOptions {
                node: configured.clone(),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(tree.status, baldes::master::tree::TreeStatus::Optimal);
        assert_eq!(tree.incumbent, 4.);
        assert_eq!(tree.bound, 4.);
        optimal(&root.snapshot.solve(configured).unwrap(), 2.);
    }
}

#[cfg(feature = "highs")]
#[test]
fn cutoff_presence_rows_keep_both_dual_signs_fixed_copies_and_descendants() {
    use baldes::master::solvers::mip::{MipOptions, MipStatus};
    for mode in [Mode::PairCount, Mode::AnyPair] {
        for lower in [false, true] {
            let root = multi_pricing::solve(problem(), options()).unwrap();
            let mut coefficients = coefficients(mode);
            if lower {
                // Require one expensive route; its presence dual is positive.
                for c in &mut coefficients {
                    c.arc_events.swap(0, 2);
                }
            }
            let row = Row {
                sense: if lower {
                    RowSense::GreaterEqual
                } else {
                    RowSense::LessEqual
                },
                rhs: 1.,
            };
            let child = root.snapshot.with_presence_row(row, &coefficients).unwrap();
            let priced = child.solve(options()).unwrap();
            optimal(&priced, 4.);
            let presence_dual = *priced.duals.last().unwrap();
            assert!(if lower {
                presence_dual > 0.
            } else {
                presence_dual < 0.
            });
            let cut = priced
                .snapshot
                .enumerate_below(5., &EnumerationOptions::default())
                .unwrap();
            assert_eq!(cut.report().coverage_cutoff, Some(5.));
            if lower {
                assert!(cut
                    .cost_bounds()
                    .iter()
                    .any(|b| b.route_constant_lower_bound < 0.));
            }
            assert_eq!(
                cut.solve_mip(&MipOptions::default())
                    .unwrap()
                    .mip
                    .solution
                    .unwrap()
                    .objective,
                4.
            );
            assert_eq!(
                cut.solve_mip(&MipOptions {
                    objective_cutoff: 4.,
                    ..Default::default()
                })
                .unwrap()
                .mip
                .status,
                MipStatus::ObjectiveCutoff
            );
            let cheap = root
                .routes
                .iter()
                .position(|r| r.network == 7 && r.route.arcs == [0, 1])
                .unwrap();
            let fixed = child
                .fix_route(cheap, 1.)
                .unwrap()
                .solve(options())
                .unwrap();
            let fixed_cut = fixed
                .snapshot
                .enumerate_below(5., &EnumerationOptions::default())
                .unwrap();
            let mip = fixed_cut.solve_mip(&MipOptions::default()).unwrap();
            assert_eq!(mip.mip.solution.unwrap().objective, 4.);
            assert!(mip
                .route_values
                .iter()
                .any(|r| r.route.network == 7 && r.route.route.arcs == [0, 1] && r.quantity >= 1.));
            if !lower {
                let descendant = child
                    .with_presence_row(
                        Row {
                            sense: RowSense::LessEqual,
                            rhs: 0.,
                        },
                        &coefficients,
                    )
                    .unwrap()
                    .solve(options())
                    .unwrap();
                optimal(&descendant, 6.);
                let next = descendant
                    .snapshot
                    .enumerate_below(7., &EnumerationOptions::default())
                    .unwrap();
                assert_eq!(
                    next.solve_mip(&MipOptions::default())
                        .unwrap()
                        .mip
                        .solution
                        .unwrap()
                        .objective,
                    6.
                );
                // The old view retains its own rows and source, not the descendant's.
                assert_eq!(
                    cut.solve_mip(&MipOptions::default())
                        .unwrap()
                        .mip
                        .solution
                        .unwrap()
                        .objective,
                    4.
                );
            }
            optimal(&root.snapshot.solve(options()).unwrap(), 2.);
        }
    }
}
