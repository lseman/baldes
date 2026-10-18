use baldes::master::{
    preprocessing::{propagate, Bounds, Constraint},
    RowSense::*,
};

fn row(entries: &[(usize, f64)], sense: baldes::master::RowSense, rhs: f64) -> Constraint {
    Constraint {
        entries: entries.to_vec(),
        sense,
        rhs,
    }
}

#[test]
fn signed_bounds_propagate_through_a_worklist() {
    let mut bounds = vec![
        Bounds {
            lower: 0.,
            upper: f64::INFINITY
        };
        3
    ];
    let report = propagate(
        &mut bounds,
        &[
            row(&[(1, 1.), (2, 1.)], Equal, 1.),
            row(&[(0, 1.), (1, -1.)], Equal, 0.),
            row(&[(0, 1.)], GreaterEqual, 1.),
        ],
        1e-8,
        1000,
    )
    .unwrap();
    assert!(report.complete && !report.infeasible);
    assert_eq!(
        bounds,
        vec![
            Bounds {
                lower: 1.,
                upper: 1.
            },
            Bounds {
                lower: 1.,
                upper: 1.
            },
            Bounds {
                lower: 0.,
                upper: 0.
            },
        ]
    );
}

#[test]
fn duplicate_entries_infinities_and_infeasibility() {
    let mut b = vec![Bounds {
        lower: f64::NEG_INFINITY,
        upper: f64::INFINITY,
    }];
    let r = propagate(
        &mut b,
        &[
            row(&[(0, 2.), (0, -1.)], LessEqual, -2.),
            row(&[(0, -1.)], LessEqual, 1.),
        ],
        1e-8,
        100,
    )
    .unwrap();
    assert!(r.infeasible);
    assert!(
        propagate(&mut [], &[row(&[], GreaterEqual, 1.)], 1e-8, 10)
            .unwrap()
            .infeasible
    );
    let mut b = vec![Bounds {
        lower: 0.,
        upper: 1.,
    }];
    assert!(
        !propagate(&mut b, &[row(&[(0, 1.), (0, -1.)], Equal, 0.)], 1e-8, 10)
            .unwrap()
            .infeasible
    );
}

#[test]
fn work_limit_and_invalid_inputs_are_distinct_from_infeasibility() {
    let mut b = vec![Bounds {
        lower: 0.,
        upper: 1.,
    }];
    let report = propagate(&mut b, &[row(&[(0, 1.)], LessEqual, 0.)], 1e-8, 0).unwrap();
    assert!(!report.complete && !report.infeasible);
    assert_eq!(b[0].upper, 1.);
    assert!(propagate(&mut b, &[row(&[(1, 1.)], Equal, 0.)], 1e-8, 100).is_err());
    assert!(propagate(&mut b, &[], f64::NAN, 100).is_err());
}

#[test]
fn propagation_preserves_every_feasible_point_on_a_small_grid() {
    // Signed coefficients, every sense, and multiple constraints. Exhaustive
    // witnesses catch a reversed residual bound or an unsound infeasibility.
    for a in [-2., -1., 1., 2.] {
        for c in [-2., -1., 1., 2.] {
            for sense in [Equal, LessEqual, GreaterEqual] {
                for rhs in -3..=3 {
                    let constraints = [
                        row(&[(0, a), (1, c)], sense, rhs as f64),
                        row(&[(0, 1.), (1, -1.)], LessEqual, 1.),
                    ];
                    let mut b = vec![
                        Bounds {
                            lower: -2.,
                            upper: 2.
                        };
                        2
                    ];
                    let report = propagate(&mut b, &constraints, 1e-8, 1000).unwrap();
                    for x in -4..=4 {
                        for y in -4..=4 {
                            let point = [x as f64 / 2., y as f64 / 2.];
                            let feasible = constraints.iter().all(|r| {
                                let v: f64 = r.entries.iter().map(|&(i, a)| a * point[i]).sum();
                                match r.sense {
                                    Equal => v == r.rhs,
                                    LessEqual => v <= r.rhs,
                                    GreaterEqual => v >= r.rhs,
                                }
                            });
                            if feasible {
                                assert!(!report.infeasible, "{constraints:?}, {point:?}");
                                for i in 0..2 {
                                    assert!(
                                        point[i] >= b[i].lower - 1e-7
                                            && point[i] <= b[i].upper + 1e-7,
                                        "{constraints:?}, {point:?}, {b:?}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn solver() -> baldes::pricing::Solver {
    let mut s = baldes::pricing::Solver::new(4, 1).unwrap();
    for v in 0..4 {
        s.graph.set_bounds(v, 0, 0., 2.).unwrap();
    }
    for (u, v) in [(0, 1), (1, 2), (2, 3), (0, 2), (2, 1), (1, 3)] {
        let id = s.graph.add_arc(u, v, 1.).unwrap();
        s.graph.arc_mut(id).unwrap().consumption[0] = if v == 3 { 0. } else { 1. };
    }
    s
}

#[test]
fn arc_fixing_applies_to_pricing_enumeration_rebuilds_and_siblings() {
    for bidirectional in [0., 1.] {
        let mut parent = solver();
        parent
            .set_option("BALDESuseBidirectionalSearch", bidirectional)
            .unwrap();
        parent.enumeration_preserve_arc_paths = true;
        assert_eq!(parent.enumerate(100.).unwrap().len(), 4);
        let mut child = parent.fork_for_node();
        child.set_forbidden_arcs(vec![1]).unwrap();
        let routes = child.inspect_enumeration(100.).unwrap().unwrap();
        assert_eq!(routes.len(), 3);
        assert!(routes.iter().all(|r| !r.arcs.contains(&1)));
        // Rebuilding the engine must keep the node's original arc fixing.
        child.restore_arcs();
        let result = child.price_phase(0, 100, false).unwrap();
        assert!(result.exact);
        assert!(result.columns.iter().all(|r| !r.arcs.contains(&1)));
        assert_eq!(child.enumerate(100.).unwrap().len(), 3);
        child.set_forbidden_arcs(vec![]).unwrap();
        assert_eq!(child.enumerate(100.).unwrap().len(), 4);
        assert_eq!(parent.inspect_enumeration(100.).unwrap().unwrap().len(), 4);
    }
}

#[cfg(feature = "clp")]
#[test]
fn original_flow_preprocessing_rejects_contradictions_before_pricing() {
    use baldes::master::{
        branching::BranchConstraint,
        root::{self, RootOptions, RootProblem},
    };
    let mut p = RootProblem::set_partitioning(solver().graph, 2.).unwrap();
    p.branch(BranchConstraint::ArcFlow {
        arc: 1,
        sense: GreaterEqual,
        rhs: 1.,
    })
    .unwrap();
    p.branch(BranchConstraint::ArcFlow {
        arc: 1,
        sense: LessEqual,
        rhs: 0.,
    })
    .unwrap();
    let result = root::solve(p, RootOptions::default()).unwrap();
    assert!(result.preprocessing.infeasible);
    assert_eq!(result.master.pricing_calls, 0);
    assert_eq!(result.simplex_iterations, 0);
}

#[cfg(feature = "clp")]
#[test]
fn a_required_arc_propagates_to_other_incoming_and_outgoing_arcs() {
    use baldes::master::{
        branching::BranchConstraint,
        root::{self, RootOptions, RootProblem},
    };
    let mut p = RootProblem::set_partitioning(solver().graph, 2.).unwrap();
    p.branch(BranchConstraint::ArcFlow {
        arc: 1,
        sense: GreaterEqual,
        rhs: 1.,
    })
    .unwrap();
    let mut options = RootOptions::default();
    options.cg.stabilize = false;
    options.enumerate = false;
    let result = root::solve(p, options).unwrap();
    assert!(!result.preprocessing.infeasible);
    assert!(result.preprocessing.bound_updates > 0);
    assert_eq!(
        result.master.status,
        baldes::master::column_generation::Status::Optimal
    );
    // 1->2 required: 0->2 and 1->sink are fixed out by cover/conservation.
    assert!(
        result
            .routes
            .iter()
            .all(|r| !r.arcs.contains(&3) && !r.arcs.contains(&5)),
        "{:?}",
        result.routes
    );
}

#[cfg(feature = "clp")]
#[test]
fn tree_reports_proved_infeasibility_without_an_incumbent() {
    use baldes::master::{
        root::RootProblem,
        tree::{self, TreeOptions, TreeStatus},
    };
    let mut graph = baldes::Graph::new(4, 1).unwrap();
    for (u, v) in [(0, 1), (1, 3), (0, 2), (2, 3)] {
        graph.add_arc(u, v, 1.).unwrap();
    }
    let p = RootProblem::set_partitioning(graph, 0.5).unwrap();
    let result = tree::solve(p, TreeOptions::default()).unwrap();
    assert_eq!(result.status, TreeStatus::Infeasible, "{result:?}");
    assert_eq!(result.open_nodes, 0);
    assert_eq!(result.pricing_calls, 0);
    assert!(!result.incumbent.is_finite());
}

#[cfg(feature = "clp")]
#[test]
fn positive_aggregate_bounds_do_not_filter_unrelated_routes() {
    use baldes::master::{
        branching::BranchConstraint,
        root::{self, RootOptions, RootProblem},
    };
    let mut s = baldes::pricing::Solver::new(5, 1).unwrap();
    for (u, v) in [(0, 1), (1, 2), (2, 4), (0, 3), (3, 4)] {
        s.graph.add_arc(u, v, 1.).unwrap();
    }
    let mut p = RootProblem::set_partitioning(s.graph, 2.).unwrap();
    p.branch(BranchConstraint::ArcFlow {
        arc: 1,
        sense: GreaterEqual,
        rhs: 1.,
    })
    .unwrap();
    let mut options = RootOptions::default();
    options.cg.stabilize = false;
    let result = root::solve(p, options).unwrap();
    assert_eq!(
        result.master.status,
        baldes::master::column_generation::Status::Optimal
    );
    assert!((result.master.lp - 5.).abs() < 1e-6);
    assert!(result.routes.iter().any(|r| r.arcs == [3, 4]));
}
