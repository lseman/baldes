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

fn solver() -> baldes::pricing::Solver<2> {
    let mut s = baldes::pricing::Solver::<2>::new(4, 1).unwrap();
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
mod network {
    use super::*;
    use baldes::master::{
        cg::Status,
        formulation::Row,
        model::{self, NodeSnapshot, Options, Output, Problem, RowCoefficients, RyanFosterMember},
        RowSense,
    };
    use std::time::Duration;

    /// A snapshot without priced columns: restrictions apply before pricing
    /// as a fresh-problem branch did.
    pub(super) fn cold(problem: Problem<2>, options: &Options) -> NodeSnapshot<2> {
        let mut cold = options.clone();
        cold.cg.time_limit = Duration::ZERO;
        let root = model::solve(problem, cold).unwrap();
        assert_eq!(root.master.pricing_calls, 0);
        root.snapshot
    }

    pub(super) fn arc_row(
        node: &NodeSnapshot<2>,
        arcs: usize,
        arc: usize,
        sense: RowSense,
        rhs: f64,
    ) -> NodeSnapshot<2> {
        let mut coefficients = vec![0.; arcs];
        coefficients[arc] = 1.;
        node.with_row(
            Row { sense, rhs },
            &[RowCoefficients {
                network: 0,
                arcs: coefficients,
                per_route: 0.,
            }],
        )
        .unwrap()
    }

    fn assert_rejected_before_pricing(output: &Output<2>) {
        assert_eq!(output.master.status, Status::Infeasible, "{output:?}");
        assert_eq!(output.master.pricing_calls, 0, "{output:?}");
    }

    #[test]
    fn original_flow_preprocessing_rejects_contradictions_before_pricing() {
        let graph = solver().graph;
        let arcs = graph.arcs.len();
        let options = Options::default();
        let node = cold(Problem::set_partitioning(graph, 2.).unwrap(), &options);
        let node = arc_row(&node, arcs, 1, GreaterEqual, 1.);
        let node = arc_row(&node, arcs, 1, LessEqual, 0.);
        assert_rejected_before_pricing(&node.solve(options).unwrap());
    }

    #[test]
    fn a_required_arc_propagates_to_other_incoming_and_outgoing_arcs() {
        let graph = solver().graph;
        let arcs = graph.arcs.len();
        let mut options = Options::default();
        options.cg.stabilize = false;
        options.gap_enumeration = false;
        let node = cold(Problem::set_partitioning(graph, 2.).unwrap(), &options);
        let result = arc_row(&node, arcs, 1, GreaterEqual, 1.)
            .solve(options)
            .unwrap();
        assert_eq!(result.master.status, Status::Optimal);
        // 1->2 required: 0->2 and 1->sink are fixed out by cover/conservation.
        assert!(
            result
                .routes
                .iter()
                .zip(&result.values)
                .filter(|(_, &x)| x > 1e-9)
                .all(|(r, _)| !r.route.arcs.contains(&3) && !r.route.arcs.contains(&5)),
            "{:?}",
            result.routes
        );
    }

    #[test]
    fn tree_reports_proved_infeasibility_without_an_incumbent() {
        use baldes::master::tree::{self, ModelTreeOptions, TreeStatus};
        let mut graph = baldes::Graph::<2>::new(4, 1).unwrap();
        for (u, v) in [(0, 1), (1, 3), (0, 2), (2, 3)] {
            graph.add_arc(u, v, 1.).unwrap();
        }
        let p = Problem::set_partitioning(graph, 0.5).unwrap();
        let result = tree::solve(p, ModelTreeOptions::default()).unwrap();
        assert_eq!(result.status, TreeStatus::Infeasible, "{result:?}");
        assert_eq!(result.open_nodes, 0);
        assert_eq!(result.pricing_calls, 0);
        assert!(!result.incumbent.is_finite());
    }

    #[test]
    fn positive_aggregate_bounds_do_not_filter_unrelated_routes() {
        let mut s = baldes::pricing::Solver::<2>::new(5, 1).unwrap();
        for (u, v) in [(0, 1), (1, 2), (2, 4), (0, 3), (3, 4)] {
            s.graph.add_arc(u, v, 1.).unwrap();
        }
        let arcs = s.graph.arcs.len();
        let mut options = Options::default();
        options.cg.stabilize = false;
        let node = cold(Problem::set_partitioning(s.graph, 2.).unwrap(), &options);
        let result = arc_row(&node, arcs, 1, GreaterEqual, 1.)
            .solve(options)
            .unwrap();
        assert_eq!(result.master.status, Status::Optimal);
        assert!((result.master.lp - 5.).abs() < 1e-6);
        assert!(result.routes.iter().any(|r| r.route.arcs == [3, 4]));
    }

    #[test]
    fn inherited_capacity_rows_detect_a_child_contradiction_before_pricing() {
        use baldes::inputs::{Instance, Rounding};
        use baldes::master::{model::separation::CapacityCutConfig, preprocessing::Bounds};
        let instance = Instance::parse("triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
        for complement in [false, true] {
            let graph = instance.graph(3, Rounding::None).unwrap();
            let arcs = graph.arcs.len();
            let start = graph
                .arcs
                .iter()
                .position(|a| a.tail == graph.source && a.head == 3)
                .unwrap();
            let end = graph
                .arcs
                .iter()
                .position(|a| a.tail == 3 && a.head == graph.sink)
                .unwrap();
            let problem = Problem::set_partitioning(graph, 3.).unwrap();
            let mut config = CapacityCutConfig::new(vec![0., 1., 1., 1.], 2.);
            config.parameters.complement_form = complement;
            let mut options = Options::default();
            options.separation.capacity = Some(config);
            options.gap_enumeration = false;
            options.cg.stabilize = false;
            let parent = model::solve(problem, options.clone()).unwrap();
            assert!(parent.cuts > 0);
            let snapshot = &parent.snapshot;
            let one = snapshot
                .with_multiplicity(
                    0,
                    Bounds {
                        lower: 0.,
                        upper: 1.,
                    },
                )
                .unwrap();
            let child = arc_row(
                &arc_row(&one, arcs, start, GreaterEqual, 1.),
                arcs,
                end,
                GreaterEqual,
                1.,
            )
            .solve(options.clone())
            .unwrap();
            assert_rejected_before_pricing(&child);
            // Neither an infeasible child nor its bound propagation mutates the
            // parent pool, physical rows or a feasible sibling's warm start.
            let sibling = snapshot
                .with_multiplicity(
                    0,
                    Bounds {
                        lower: 2.,
                        upper: 3.,
                    },
                )
                .unwrap()
                .solve(options.clone())
                .unwrap();
            assert_eq!(sibling.master.status, Status::Optimal, "{sibling:?}");
            let replay = snapshot.solve(options).unwrap();
            assert!((replay.master.lp - parent.master.lp).abs() < 1e-7);
            assert!(replay.cuts >= parent.cuts);
        }
    }

    #[test]
    fn transitive_ryan_foster_contradictions_are_proved_infeasible() {
        let mut graph = baldes::Graph::<2>::new(5, 0).unwrap();
        graph.elementary = true;
        for u in 0..4 {
            for v in 1..5 {
                if u != v {
                    graph.add_arc(u, v, 1.).unwrap();
                }
            }
        }
        let options = Options::default();
        let mut node = cold(Problem::set_partitioning(graph, 3.).unwrap(), &options);
        for (first, second, together) in [(1, 2, true), (2, 3, true), (1, 3, false)] {
            node = node
                .with_ryan_foster(
                    &[RyanFosterMember {
                        network: 0,
                        first,
                        second,
                    }],
                    together,
                )
                .unwrap();
        }
        // The removed routing root also rejected this before pricing (a
        // BALDES transitive-closure rule, not native); phase I proves it here.
        let output = node.solve(options).unwrap();
        assert_eq!(output.master.status, Status::Infeasible, "{output:?}");
        assert_eq!(output.master.bound, f64::INFINITY);
    }
}
