use baldes::master::{
    cg::{BoundMode, Status},
    model::{self, format::Model, EnumerationOptions, Options},
    routing::Row,
    RowSense,
};
use std::time::Duration;
fn problem() -> model::Problem<2> {
    Model::from_json(include_str!("../../examples/networks/triangle.json"))
        .unwrap()
        .problem()
        .unwrap()
}
fn options() -> Options {
    let mut options = Options::default();
    options.cg.artificial_cost = 100.;
    options.cg.stabilize = false;
    options
}

#[test]
fn adaptive_transfer_runs_after_pricing_and_preserves_global_counters() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let mut configured = options();
        configured.enumeration = Some(EnumerationOptions::default());
        configured.cg.bound_mode = mode;
        let output = model::solve(problem(), configured).unwrap();
        assert_eq!(output.master.status, Status::Optimal);
        assert!(!output.master.enumeration_requested);
        assert!(output.snapshot.enumeration_report().complete);
        assert_eq!(output.snapshot.enumeration_report().attempts, 1);
        // A pre-CG transfer bypasses all routing oracle calls. This handoff
        // must instead follow an actual exact pricing point.
        assert!(output.oracle_calls.iter().all(|(_, calls)| *calls > 0));
        assert!((output.master.lp - 1.5).abs() < 1e-8);
        assert_eq!(output.master.iterations, output.master.history.len());
        assert_eq!(
            output.master.history.last().unwrap().pricing_calls,
            output.master.pricing_calls
        );
        for (index, row) in output.master.history.iter().enumerate() {
            assert_eq!(row.iteration, index + 1);
        }
        let mut resumed_options = options();
        resumed_options.enumeration = Some(EnumerationOptions::default());
        let resumed = output.snapshot.solve(resumed_options).unwrap();
        assert_eq!(resumed.snapshot.enumeration_report().attempts, 1);
        assert_eq!(resumed.master.status, Status::Optimal);
    }
}

#[test]
fn adaptive_budget_declines_once_then_resumes_and_can_retry() {
    for limits in [
        EnumerationOptions {
            max_routes: 5,
            ..Default::default()
        },
        EnumerationOptions {
            max_work: 1,
            ..Default::default()
        },
        EnumerationOptions {
            time_limit: Duration::ZERO,
            ..Default::default()
        },
    ] {
        let mut configured = options();
        configured.enumeration = Some(limits);
        let output = model::solve(problem(), configured).unwrap();
        assert_eq!(output.master.status, Status::Optimal);
        assert!((output.master.lp - 1.5).abs() < 1e-8);
        let report = output.snapshot.enumeration_report();
        assert_eq!(report.attempts, 1);
        assert!(!report.complete);
        assert!(report.failure.is_some());
        assert_eq!(report.transferred, 0);
        let mut retry = options();
        retry.enumeration = Some(EnumerationOptions::default());
        let resumed = output.snapshot.solve(retry).unwrap();
        assert_eq!(resumed.master.status, Status::Optimal);
        assert!(resumed.snapshot.enumeration_report().complete);
        assert_eq!(resumed.snapshot.enumeration_report().attempts, 2);
    }
}

#[test]
fn adaptive_transfer_does_not_extend_iteration_or_time_limits() {
    let baseline = model::solve(problem(), options()).unwrap();
    let mut configured = options();
    configured.enumeration = Some(EnumerationOptions::default());
    configured.cg.max_iterations = baseline.master.iterations;
    let limited = model::solve(problem(), configured).unwrap();
    assert_eq!(limited.master.status, Status::IterationLimit);
    assert_eq!(limited.master.iterations, baseline.master.iterations);
    assert!(!limited.snapshot.enumeration_report().complete);
    assert!(!limited.master.enumeration_requested);
    let mut configured = options();
    configured.enumeration = Some(EnumerationOptions::default());
    configured.cg.time_limit = Duration::ZERO;
    let limited = model::solve(problem(), configured).unwrap();
    assert_eq!(limited.master.status, Status::TimeLimit);
    assert_eq!(limited.master.iterations, 0);
    assert_eq!(limited.snapshot.enumeration_report().attempts, 0);
}
#[test]
fn complete_transfer_is_atomic_repeatable_and_resumable() {
    let parent = model::solve(problem(), options()).unwrap();
    let prepared = parent
        .snapshot
        .enumerate(&EnumerationOptions::default())
        .unwrap();
    let report = prepared.enumeration_report();
    assert!(report.complete, "{report:?}");
    assert_eq!(report.routes, 9);
    assert_eq!(report.networks, [(7, 5), (42, 4)]);
    assert_eq!(report.attempts, 1);
    let repeated = prepared.enumerate(&EnumerationOptions::default()).unwrap();
    assert_eq!(repeated.enumeration_report().attempts, 1);
    assert_eq!(
        repeated.enumeration_report().transferred,
        report.transferred
    );
    for node in [&prepared, &repeated, &parent.snapshot] {
        let output = node.solve(options()).unwrap();
        assert_eq!(output.master.status, Status::Optimal);
        assert!((output.master.lp - 1.5).abs() < 1e-8);
    }
    assert!(!parent.snapshot.enumeration_report().complete);
}
#[test]
fn any_network_or_budget_failure_leaves_pricing_available() {
    let parent = model::solve(problem(), options()).unwrap();
    for limits in [
        EnumerationOptions {
            max_routes: 5,
            ..Default::default()
        },
        EnumerationOptions {
            max_work: 1,
            ..Default::default()
        },
        EnumerationOptions {
            max_stored_arcs: 1,
            ..Default::default()
        },
        EnumerationOptions {
            time_limit: Duration::ZERO,
            ..Default::default()
        },
    ] {
        let partial = parent.snapshot.enumerate(&limits).unwrap();
        assert!(!partial.enumeration_report().complete);
        assert!(partial.enumeration_report().failure.is_some());
        assert_eq!(partial.enumeration_report().transferred, 0);
        let resumed = partial.solve(options()).unwrap();
        assert_eq!(resumed.master.status, Status::Optimal);
        assert!((resumed.master.lp - 1.5).abs() < 1e-8);
    }
}
#[test]
fn explicit_root_transfer_and_additive_descendants_keep_all_coefficients() {
    let mut configured = options();
    configured.enumeration = Some(EnumerationOptions::default());
    configured.cg.bound_mode = BoundMode::Certified;
    let parent = model::solve(problem(), configured).unwrap();
    assert_eq!(parent.master.status, Status::Optimal);
    assert!(parent.snapshot.enumeration_report().complete);
    let coeffs = problem()
        .networks
        .iter()
        .map(|n| model::RowCoefficients {
            network: n.id,
            arcs: vec![0.; n.graph.arcs.len()],
            per_route: 1.,
        })
        .collect::<Vec<_>>();
    let child = parent
        .snapshot
        .with_row(
            Row {
                sense: RowSense::GreaterEqual,
                rhs: 2.,
            },
            &coeffs,
        )
        .unwrap()
        .branch_assignment(0, 7, false)
        .unwrap()
        .solve(options())
        .unwrap();
    assert_eq!(child.master.status, Status::Optimal);
    assert!((child.master.lp - 2.).abs() < 1e-8);
    assert_eq!(child.snapshot.enumeration_report().attempts, 1);
}

#[cfg(feature = "highs")]
mod mip {
    use super::*;
    use baldes::master::{
        solvers::mip::{MipOptions, MipStatus},
        tree::TreeStatus,
    };
    #[test]
    fn all_network_integer_master_matches_exhaustive_optimum_and_keeps_fixed_rows() {
        let parent = model::solve(problem(), options()).unwrap();
        assert!(parent
            .snapshot
            .solve_enumerated_mip(&MipOptions::default())
            .unwrap()
            .is_none());
        let node = parent
            .snapshot
            .enumerate(&EnumerationOptions::default())
            .unwrap();
        let result = node
            .solve_enumerated_mip(&MipOptions::default())
            .unwrap()
            .unwrap();
        assert_eq!(result.mip.status, MipStatus::Optimal);
        assert_eq!(result.mip.solution.as_ref().unwrap().objective, 2.);
        assert_eq!(
            result.route_values.iter().map(|r| r.quantity).sum::<f64>(),
            2.
        );
        assert!(result.coverage_cutoff.is_infinite());
        let fixed = parent
            .routes
            .iter()
            .position(|r| r.network == 42 && r.route.arcs == [0, 1])
            .unwrap();
        let child = parent
            .snapshot
            .fix_route(fixed, 1.)
            .unwrap()
            .branch_assignment(0, 7, false)
            .unwrap();
        let coeffs = problem()
            .networks
            .iter()
            .map(|n| model::RowCoefficients {
                network: n.id,
                arcs: vec![0.; n.graph.arcs.len()],
                per_route: 1.,
            })
            .collect::<Vec<_>>();
        let child = child
            .with_row(
                Row {
                    sense: RowSense::Equal,
                    rhs: 2.,
                },
                &coeffs,
            )
            .unwrap()
            .enumerate(&EnumerationOptions::default())
            .unwrap();
        let fixed_mip = child
            .solve_enumerated_mip(&MipOptions::default())
            .unwrap()
            .unwrap();
        assert_eq!(fixed_mip.mip.status, MipStatus::Optimal);
        assert_eq!(fixed_mip.mip.solution.unwrap().objective, 2.);
        assert_eq!(
            fixed_mip
                .route_values
                .iter()
                .map(|r| r.quantity)
                .sum::<f64>(),
            2.
        );
        assert!((parent.snapshot.solve(options()).unwrap().master.lp - 1.5).abs() < 1e-8);
    }
    #[test]
    fn cutoff_limits_and_retry_never_change_the_source() {
        let parent = model::solve(problem(), options()).unwrap();
        let node = parent
            .snapshot
            .enumerate(&EnumerationOptions::default())
            .unwrap();
        for configured in [
            MipOptions {
                time_limit: Duration::ZERO,
                ..Default::default()
            },
            MipOptions {
                node_limit: 0,
                ..Default::default()
            },
        ] {
            let stopped = node.solve_enumerated_mip(&configured).unwrap().unwrap();
            assert!(matches!(
                stopped.mip.status,
                MipStatus::TimeLimit | MipStatus::NodeLimit
            ));
            assert!(stopped.mip.solution.is_none());
        }
        let cutoff = node
            .solve_enumerated_mip(&MipOptions {
                objective_cutoff: 2.,
                ..Default::default()
            })
            .unwrap()
            .unwrap();
        assert_eq!(cutoff.mip.status, MipStatus::ObjectiveCutoff);
        assert!(cutoff.mip.solution.is_none());
        let failure = node.solve_enumerated_mip(&MipOptions {
            node_limit: usize::MAX,
            ..Default::default()
        });
        assert!(failure.is_err());
        let retry = node
            .solve_enumerated_mip(&MipOptions::default())
            .unwrap()
            .unwrap();
        assert_eq!(retry.mip.status, MipStatus::Optimal);
        assert!((node.solve(options()).unwrap().master.lp - 1.5).abs() < 1e-8);
    }
    #[test]
    fn tree_uses_full_transfer_and_falls_back_on_partial_enumeration() {
        let mut configured = baldes::master::tree::ModelTreeOptions::default();
        configured.node = options();
        configured.enumerated_mip = Some(MipOptions::default());
        let tree = baldes::master::tree::solve(problem(), configured.clone()).unwrap();
        assert_eq!(tree.status, TreeStatus::Optimal);
        assert_eq!(tree.incumbent, 2.);
        assert_eq!(tree.nodes, 1);
        assert_eq!(tree.enumerated_mip_calls, 1);
        configured.node.enumeration = Some(EnumerationOptions {
            max_routes: 5,
            ..Default::default()
        });
        let fallback = baldes::master::tree::solve(problem(), configured).unwrap();
        assert_eq!(fallback.status, TreeStatus::Optimal);
        assert_eq!(fallback.incumbent, 2.);
        // Direct source/sink routes now participate in full route coverage,
        // changing the partial-universe fallback tree and allowing complete
        // child universes to use the enumerated MIP path.
        assert_eq!(fallback.nodes, 6);
        assert_eq!(fallback.enumerated_mip_calls, 2);
    }
}

#[test]
fn global_limit_and_equality_relaxation_are_explicit() {
    let root = model::solve(problem(), options()).unwrap();
    assert!(
        root.snapshot
            .enumerate(&EnumerationOptions {
                max_routes: 9,
                ..Default::default()
            })
            .unwrap()
            .enumeration_report()
            .complete
    );
    assert!(
        !root
            .snapshot
            .enumerate(&EnumerationOptions {
                max_routes: 8,
                ..Default::default()
            })
            .unwrap()
            .enumeration_report()
            .complete
    );
    let mut equality = problem();
    equality.networks[1].graph.resources[0] = baldes::graph::ResourceKind::Equality;
    for vertex in 0..equality.networks[1].graph.vertices.len() {
        equality.networks[1]
            .graph
            .set_bounds(vertex, 0, 0., 0.)
            .unwrap();
    }
    let root = model::solve(equality, options()).unwrap();
    let prepared = root
        .snapshot
        .enumerate(&EnumerationOptions::default())
        .unwrap();
    assert!(prepared.enumeration_report().complete);
    assert_eq!(prepared.enumeration_report().routes, 9);
    assert!(prepared.enumeration_report().failure.is_none());
    assert_eq!(
        prepared.solve(options()).unwrap().master.status,
        Status::Optimal
    );
}

#[cfg(feature = "highs")]
#[test]
fn phase_one_and_enumerated_mip_close_original_activity_children() {
    use baldes::master::{solvers::mip::MipOptions, tree::TreeStatus};
    let mut model = problem();
    // Merge every path into one pricing graph while retaining distinct path IDs.
    let first = model.networks.remove(0);
    let mut graph = first.graph.clone();
    let old_sink = graph.sink;
    let id = graph.add_arc(0, 3, 1.).unwrap();
    let mut entries = first.arc_entries.clone();
    assert_eq!(id as usize, entries.len());
    entries.push(vec![(0, 1.), (2, 1.)]);
    // The new direct-to-route-vertex path reaches the same sink through arc 5.
    assert_eq!(graph.arcs[5].head, old_sink);
    model.networks = vec![model::Network {
        graph,
        arc_entries: entries,
        ..first
    }];
    let mut configured = baldes::master::tree::ModelTreeOptions::default();
    configured.node = options();
    let plain = baldes::master::tree::solve(model.clone(), configured.clone()).unwrap();
    // General all-network phase I now proves the combined-row infeasible
    // children that previously required a complete finite MIP fallback.
    assert_eq!(plain.status, TreeStatus::Optimal);
    assert_eq!(plain.incumbent, 2.);
    assert_eq!(plain.bound, 2.);
    assert_eq!(plain.open_nodes, 0);
    configured.enumerated_mip = Some(MipOptions::default());
    let solved = baldes::master::tree::solve(model, configured).unwrap();
    assert_eq!(solved.status, TreeStatus::Optimal);
    assert_eq!(solved.incumbent, 2.);
    assert_eq!(solved.nodes, 1);
}

#[test]
fn direct_source_sink_routes_are_included_in_complete_coverage() {
    let mut graph = baldes::Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, 1.).unwrap();
    let model = model::Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        networks: vec![model::Network {
            id: 7,
            graph,
            multiplicity: baldes::graph::Window::new(0., 1.).unwrap(),
            arc_entries: vec![vec![]],
            route_entries: vec![(0, 1.)],
        }],
    };
    let parent = model::solve(model, options()).unwrap();
    assert_eq!(parent.master.status, Status::Optimal);
    let prepared = parent
        .snapshot
        .enumerate(&EnumerationOptions::default())
        .unwrap();
    assert!(prepared.enumeration_report().complete);
    assert_eq!(prepared.enumeration_report().routes, 1);
    assert_eq!(prepared.solve(options()).unwrap().master.lp, 1.);
}

#[cfg(feature = "highs")]
#[test]
fn material_objective_rounding_does_not_publish_a_false_integer_proof() {
    use baldes::master::solvers::mip::{MipOptions, MipStatus};
    let mut graph = baldes::Graph::<2>::new(4, 0).unwrap();
    graph.add_arc(0, 1, 1e16).unwrap();
    graph.add_arc(1, 2, 1.).unwrap();
    graph.add_arc(2, 3, -1e16).unwrap();
    let model = model::Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        networks: vec![model::Network {
            id: 7,
            graph,
            multiplicity: baldes::graph::Window::new(0., 1.).unwrap(),
            arc_entries: vec![vec![(0, 1.)], vec![], vec![]],
            route_entries: vec![],
        }],
    };
    let parent = model::solve(model, options()).unwrap();
    let node = parent
        .snapshot
        .enumerate(&EnumerationOptions::default())
        .unwrap();
    let mip = node
        .solve_enumerated_mip(&MipOptions::default())
        .unwrap()
        .unwrap();
    assert_eq!(mip.mip.status, MipStatus::Incomplete);
    assert_eq!(mip.mip.solution.unwrap().objective, 1.);
    assert_eq!(mip.mip.bound, f64::NEG_INFINITY);
    let cutoff = node
        .solve_enumerated_mip(&MipOptions {
            objective_cutoff: 1.,
            ..Default::default()
        })
        .unwrap()
        .unwrap();
    assert_eq!(cutoff.mip.status, MipStatus::Incomplete);
    assert!(cutoff.mip.solution.is_none());
}

#[cfg(feature = "highs")]
#[test]
fn tree_cutoff_fallback_solves_when_full_storage_limit_is_exceeded() {
    use baldes::master::{solvers::mip::MipOptions, tree::TreeStatus};
    let mut problem = problem();
    let first = &mut problem.networks[0];
    let old = first.graph.arcs[0].clone();
    first.graph.add_arc(old.tail, old.head, 20.).unwrap();
    first.arc_entries.push(first.arc_entries[0].clone());
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let mut configured = baldes::master::tree::ModelTreeOptions::default();
        configured.node = options();
        configured.node.cg.bound_mode = mode;
        configured.node.enumeration = Some(EnumerationOptions {
            max_routes: 9,
            ..Default::default()
        });
        configured.enumerated_mip = Some(MipOptions {
            objective_cutoff: 3.,
            ..Default::default()
        });
        configured.node_limit = 1;
        let tree = baldes::master::tree::solve(problem.clone(), configured).unwrap();
        assert_eq!(tree.incumbent, 2.);
        assert_eq!(tree.enumerated_mip_calls, 1);
        assert!(tree
            .enumeration_reports
            .iter()
            .any(|r| !r.complete && r.coverage_cutoff.is_none()));
        let cutoff = tree
            .enumeration_reports
            .iter()
            .find(|r| r.complete && r.coverage_cutoff == Some(3.))
            .unwrap();
        assert_eq!(cutoff.routes, 9);
        if mode == BoundMode::Numerical {
            assert_eq!(tree.status, TreeStatus::Optimal);
            assert_eq!(tree.open_nodes, 0);
        } else {
            assert_eq!(tree.status, TreeStatus::NodeLimit);
            assert!(tree.open_nodes > 0);
            assert!(tree.bound < tree.incumbent);
        }
    }
}

#[cfg(feature = "highs")]
#[test]
fn tree_cutoff_without_incumbent_never_means_original_infeasibility() {
    use baldes::master::{solvers::mip::MipOptions, tree::TreeStatus};
    let mut configured = baldes::master::tree::ModelTreeOptions::default();
    configured.node = options();
    configured.node.enumeration = Some(EnumerationOptions {
        max_routes: 3,
        ..Default::default()
    });
    configured.enumerated_mip = Some(MipOptions {
        objective_cutoff: 1.75,
        ..Default::default()
    });
    configured.node_limit = 1;
    let tree = baldes::master::tree::solve(problem(), configured.clone()).unwrap();
    assert_eq!(tree.status, TreeStatus::NodeLimit);
    assert!(tree.incumbent.is_infinite());
    assert!(tree.open_nodes > 0);
    assert_eq!(tree.enumerated_mip_calls, 1);
    assert!(tree
        .enumeration_reports
        .iter()
        .any(|r| r.complete && r.coverage_cutoff == Some(1.75) && r.routes == 3));
    // Exhausting enumeration proof work must leave ordinary branching usable.
    configured.node.enumeration.as_mut().unwrap().max_work = 0;
    configured.node_limit = 20;
    let fallback = baldes::master::tree::solve(problem(), configured).unwrap();
    assert_eq!(fallback.status, TreeStatus::Optimal);
    assert_eq!(fallback.incumbent, 2.);
    assert_eq!(fallback.enumerated_mip_calls, 0);
    assert!(fallback
        .enumeration_reports
        .iter()
        .any(|r| r.coverage_cutoff == Some(1.75) && r.failure.is_some()));
}

#[cfg(feature = "highs")]
#[test]
fn equality_with_free_sink_uses_forward_pricing_and_original_enumeration() {
    use baldes::master::solvers::mip::{MipOptions, MipStatus};
    let text = r#"{"rows":[{"sense":"eq","rhs":1}],"networks":[{
        "id":7,"vertices":4,"multiplicity":[0,1],
        "resources":[{"kind":"equality","initial":1,"bounds":[[0,10],[5,10],[0,10],[0,10]]}],
        "arcs":[{"tail":0,"head":1,"cost":1,"consumption":[2],"entries":[[0,1]]},
        {"tail":1,"head":3,"cost":0,"consumption":[0]},
        {"tail":0,"head":2,"cost":3,"consumption":[1],"entries":[[0,1]]},
        {"tail":2,"head":3,"cost":0,"consumption":[0]}]}]}"#;
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        for direction in [0., 1., 2.] {
            let problem = Model::from_json(text).unwrap().problem::<5>().unwrap();
            let mut configured = options();
            configured.cg.bound_mode = mode;
            configured.enumeration = Some(EnumerationOptions::default());
            configured
                .pricing_options
                .push(("BALDESuseBidirectionalSearch".into(), direction));
            let root = model::solve(problem, configured).unwrap();
            assert_eq!(root.master.status, Status::Optimal);
            assert_eq!(root.master.lp, 3.);
            assert_eq!(root.master.bound, 3.);
            assert_eq!(root.snapshot.enumeration_report().routes, 1);
            assert!(root
                .routes
                .iter()
                .zip(&root.values)
                .all(|(r, &x)| x == 0. || r.route.arcs == [2, 3]));
            let mip = root
                .snapshot
                .solve_enumerated_mip(&MipOptions::default())
                .unwrap()
                .unwrap();
            assert_eq!(mip.mip.status, MipStatus::Optimal);
            assert_eq!(mip.mip.solution.unwrap().objective, 3.);
        }
    }
}

#[cfg(feature = "highs")]
#[test]
fn rounded_up_route_objective_cannot_prove_no_original_improvement() {
    use baldes::master::solvers::mip::{MipOptions, MipStatus};
    let mut graph = baldes::Graph::<2>::new(4, 0).unwrap();
    graph.add_arc(0, 1, 1e16).unwrap();
    graph.add_arc(1, 2, 1.5).unwrap();
    graph.add_arc(2, 3, -1e16).unwrap();
    let problem = model::Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        networks: vec![model::Network {
            id: 7,
            graph,
            multiplicity: baldes::graph::Window::new(0., 1.).unwrap(),
            arc_entries: vec![vec![(0, 1.)], vec![], vec![]],
            route_entries: vec![],
        }],
    };
    let root = model::solve(problem, options()).unwrap();
    let node = root
        .snapshot
        .enumerate(&EnumerationOptions::default())
        .unwrap();
    let result = node
        .solve_enumerated_mip(&MipOptions {
            objective_cutoff: 1.75,
            ..Default::default()
        })
        .unwrap()
        .unwrap();
    // The rounded route coefficient is two, but its original arc sum is 1.5.
    // No returned witness means replay alone cannot detect this false cutoff.
    assert_eq!(result.mip.status, MipStatus::Incomplete);
    assert_eq!(result.mip.bound, f64::NEG_INFINITY);
    let cutoff = root
        .snapshot
        .enumerate_below(1.75, &EnumerationOptions::default())
        .unwrap();
    let scoped = cutoff.solve_mip(&MipOptions::default()).unwrap();
    assert_eq!(scoped.mip.status, MipStatus::Incomplete);
    assert_eq!(scoped.mip.bound, f64::NEG_INFINITY);
}

#[cfg(feature = "highs")]
#[test]
fn rounded_route_coefficient_cannot_prove_original_integer_infeasibility() {
    use baldes::master::solvers::mip::{MipOptions, MipStatus};
    let mut graph = baldes::Graph::<2>::new(4, 0).unwrap();
    graph.add_arc(0, 1, 1.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    graph.add_arc(2, 3, 0.).unwrap();
    let problem = model::Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 1.5,
        }],
        networks: vec![model::Network {
            id: 7,
            graph,
            multiplicity: baldes::graph::Window::new(0., 1.).unwrap(),
            arc_entries: vec![vec![(0, 1e16)], vec![(0, 1.5)], vec![(0, -1e16)]],
            route_entries: vec![],
        }],
    };
    let mut configured = options();
    configured.preprocessing.enabled = false;
    let root = model::solve(problem, configured).unwrap();
    let node = root
        .snapshot
        .enumerate(&EnumerationOptions::default())
        .unwrap();
    let result = node
        .solve_enumerated_mip(&MipOptions::default())
        .unwrap()
        .unwrap();
    // Original coefficient 1.5 admits quantity one. Its encoded coefficient
    // two forces quantity 0.75, so rounded-model infeasibility proves nothing.
    assert_eq!(result.mip.status, MipStatus::Incomplete);
    assert_eq!(result.mip.bound, f64::NEG_INFINITY);
    assert!(result.mip.solution.is_none());
}

#[test]
fn interval_cost_full_transfer_replays_fronts_and_declines_unproved_translations_atomically() {
    #[cfg(feature = "highs")]
    use baldes::master::solvers::mip::MipOptions;
    for fractional in [false, true] {
        let mut model: serde_json::Value =
            serde_json::from_str(include_str!("../../examples/networks/interval_cost.json"))
                .unwrap();
        // This parallel arc is cheap in the scalar relaxation, but its
        // charge domain is disjoint from the original interval state window.
        model["networks"][0]["arcs"].as_array_mut().unwrap().push(
            serde_json::json!({"tail":0,"head":1,"cost":-100,"consumption":[0],"entries":[[0,1]]}),
        );
        model["networks"][0]["coupled_resources"][0]["increments"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!(0));
        model["networks"][0]["coupled_resources"][0]["functions"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({"points":[[5,0],[6,0]]}));
        if fractional {
            model["networks"][0]["coupled_resources"][0]["increments"] =
                serde_json::json!([0.25, -0.25, 0.25, -0.25, 0.]);
        }
        let problem = Model::from_json(&model.to_string())
            .unwrap()
            .problem::<2>()
            .unwrap();
        let parent = model::solve(problem, options()).unwrap();
        assert_eq!(parent.master.status, Status::Optimal);
        let prepared = parent
            .snapshot
            .enumerate(&EnumerationOptions::default())
            .unwrap();
        let report = prepared.enumeration_report();
        assert_eq!(report.complete, !fractional, "{report:?}");
        assert!(!parent.snapshot.enumeration_report().complete);
        let resumed = prepared.solve(options()).unwrap();
        assert_eq!(resumed.master.status, Status::Optimal);
        assert!((resumed.master.lp - parent.master.lp).abs() < 1e-8);
        if fractional {
            assert_eq!(report.transferred, 0);
            assert!(report.failure.as_ref().unwrap().contains("unknown"));
        } else {
            assert_eq!(report.routes, 2);
            assert!(resumed
                .routes
                .iter()
                .all(|route| route.route.original_cost == 20.));
            #[cfg(feature = "highs")]
            {
                let mip = prepared
                    .solve_enumerated_mip(&MipOptions::default())
                    .unwrap()
                    .unwrap();
                assert!((mip.mip.solution.unwrap().objective - 40.).abs() < 1e-8);
            }
        }
    }
}
