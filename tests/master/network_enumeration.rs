use baldes::master::{
    column_generation::{BoundMode, Status},
    networks::{self, format::Model, EnumerationOptions, Options},
    root::Row,
    RowSense,
};
use std::time::Duration;
fn problem() -> networks::Problem<2> {
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
fn complete_transfer_is_atomic_repeatable_and_resumable() {
    let parent = networks::solve(problem(), options()).unwrap();
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
    let parent = networks::solve(problem(), options()).unwrap();
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
    let parent = networks::solve(problem(), configured).unwrap();
    assert_eq!(parent.master.status, Status::Optimal);
    assert!(parent.snapshot.enumeration_report().complete);
    let coeffs = problem()
        .networks
        .iter()
        .map(|n| networks::RowCoefficients {
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
        let parent = networks::solve(problem(), options()).unwrap();
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
            .map(|n| networks::RowCoefficients {
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
        let parent = networks::solve(problem(), options()).unwrap();
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
        let mut configured = networks::AssignmentTreeOptions::default();
        configured.node = options();
        configured.enumerated_mip = Some(MipOptions::default());
        let tree = networks::solve_assignment_tree(problem(), configured.clone()).unwrap();
        assert_eq!(tree.status, TreeStatus::Optimal);
        assert_eq!(tree.incumbent, 2.);
        assert_eq!(tree.nodes, 1);
        assert_eq!(tree.enumerated_mip_calls, 1);
        configured.node.enumeration = Some(EnumerationOptions {
            max_routes: 5,
            ..Default::default()
        });
        let fallback = networks::solve_assignment_tree(problem(), configured).unwrap();
        assert_eq!(fallback.status, TreeStatus::Optimal);
        assert_eq!(fallback.incumbent, 2.);
        assert_eq!(fallback.nodes, 3);
        assert_eq!(fallback.enumerated_mip_calls, 0);
    }
}

#[test]
fn global_limit_equality_and_unsupported_resources_are_explicit() {
    let root = networks::solve(problem(), options()).unwrap();
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
    let mut unsupported = problem();
    unsupported.networks[1].graph.resources[0] = baldes::graph::ResourceKind::Equality;
    for vertex in 0..unsupported.networks[1].graph.vertices.len() {
        unsupported.networks[1]
            .graph
            .set_bounds(vertex, 0, 0., 0.)
            .unwrap();
    }
    let root = networks::solve(unsupported, options()).unwrap();
    let prepared = root
        .snapshot
        .enumerate(&EnumerationOptions::default())
        .unwrap();
    assert!(!prepared.enumeration_report().complete);
    assert!(prepared
        .enumeration_report()
        .failure
        .as_ref()
        .unwrap()
        .contains("UnsupportedResources"));
    assert_eq!(
        prepared.solve(options()).unwrap().master.status,
        Status::Optimal
    );
}

#[cfg(feature = "highs")]
#[test]
fn enumeration_mip_closes_a_fractional_node_without_assignment_candidates() {
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
    model.networks = vec![networks::Network {
        graph,
        arc_entries: entries,
        ..first
    }];
    let mut configured = networks::AssignmentTreeOptions::default();
    configured.node = options();
    let plain = networks::solve_assignment_tree(model.clone(), configured.clone()).unwrap();
    assert_eq!(plain.status, TreeStatus::Incomplete);
    configured.enumerated_mip = Some(MipOptions::default());
    let solved = networks::solve_assignment_tree(model, configured).unwrap();
    assert_eq!(solved.status, TreeStatus::Optimal);
    assert_eq!(solved.incumbent, 2.);
    assert_eq!(solved.nodes, 1);
}

#[test]
fn direct_source_sink_routes_are_not_silently_omitted_from_coverage() {
    let mut graph = baldes::Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, 1.).unwrap();
    let model = networks::Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        networks: vec![networks::Network {
            id: 7,
            graph,
            multiplicity: baldes::graph::Window::new(0., 1.).unwrap(),
            arc_entries: vec![vec![]],
            route_entries: vec![(0, 1.)],
        }],
    };
    let parent = networks::solve(model, options()).unwrap();
    assert_eq!(parent.master.status, Status::Optimal);
    let prepared = parent
        .snapshot
        .enumerate(&EnumerationOptions::default())
        .unwrap();
    assert!(!prepared.enumeration_report().complete);
    assert!(prepared
        .enumeration_report()
        .failure
        .as_ref()
        .unwrap()
        .contains("direct source/sink"));
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
    let model = networks::Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        networks: vec![networks::Network {
            id: 7,
            graph,
            multiplicity: baldes::graph::Window::new(0., 1.).unwrap(),
            arc_entries: vec![vec![(0, 1.)], vec![], vec![]],
            route_entries: vec![],
        }],
    };
    let parent = networks::solve(model, options()).unwrap();
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
