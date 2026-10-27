//! End-to-end synchronization proofs through the public native interface.
#![cfg(feature = "clp")]
use baldes::{
    graph::{Graph, Window},
    master::{
        cg::BoundMode,
        model::{self, Network, Problem, SynchronizedService, SynchronizedServiceMember},
        tree::TreeStatus,
    },
};
use std::time::Duration;

fn network(id: i32, lower: f64, upper: f64, count: f64) -> Network<2> {
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    graph.set_bounds(0, 0, 0., 0.).unwrap();
    graph.set_bounds(1, 0, lower, upper).unwrap();
    graph.set_bounds(2, 0, 0., 100.).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    Network {
        id,
        graph,
        multiplicity: Window::new(0., count).unwrap(),
        arc_entries: vec![vec![], vec![]],
        route_entries: vec![],
        activity_entries: vec![],
    }
}
fn service(members: &[(i32, usize)]) -> SynchronizedService {
    SynchronizedService {
        id: 17,
        resource: 0,
        vehicles_required: 2,
        reward: 1.,
        start: [0, 10],
        members: members
            .iter()
            .map(|&(network, vertex)| SynchronizedServiceMember { network, vertex })
            .collect(),
    }
}
fn options(services: Vec<SynchronizedService>) -> baldes::master::tree::ModelTreeOptions {
    let mut options = baldes::master::tree::ModelTreeOptions::default();
    options.time_limit = Duration::from_secs(20);
    options.node.cg.time_limit = options.time_limit;
    options.node.cg.stabilize = false;
    options.node.cg.artificial_cost = 100.;
    options.node.synchronized_services = services;
    options
}
fn solve(
    networks: Vec<Network<2>>,
    services: Vec<SynchronizedService>,
) -> baldes::master::tree::ModelTreeOutput {
    baldes::master::tree::solve(
        Problem {
            rows: vec![],
            networks,
        },
        options(services),
    )
    .unwrap()
}
#[test]
fn waiting_synchronizes_and_counts_reward_once() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let mut opts = options(vec![service(&[(7, 1), (42, 1)])]);
        opts.node.cg.bound_mode = mode;
        let result = baldes::master::tree::solve(
            Problem {
                rows: vec![],
                networks: vec![network(7, 0., 9., 1.), network(42, 5., 10., 1.)],
            },
            opts,
        )
        .unwrap();
        assert_eq!(result.status, TreeStatus::Optimal, "{result:?}");
        assert_eq!(result.incumbent, -1.);
        assert_eq!(result.bound, -1.);
        assert_eq!(result.routes.len(), 2);
        for fixed in &result.routes {
            assert_eq!(fixed.route.route.resource_profile[1][0], 5.);
        }
    }
}
#[test]
fn disjoint_vehicle_windows_cannot_be_served() {
    let result = solve(
        vec![network(7, 0., 4., 1.), network(42, 6., 10., 1.)],
        vec![service(&[(7, 1), (42, 1)])],
    );
    assert_eq!(result.status, TreeStatus::Optimal, "{result:?}");
    assert_eq!(result.incumbent, 0.);
    assert!(result.routes.is_empty());
    assert!(result.nodes > 1);
}
#[test]
fn interchangeable_route_copies_are_distinct_vehicles() {
    let result = solve(vec![network(7, 2., 6., 2.)], vec![service(&[(7, 1)])]);
    assert_eq!(result.status, TreeStatus::Optimal, "{result:?}");
    assert_eq!(result.incumbent, -1.);
    assert_eq!(result.routes.iter().map(|r| r.quantity).sum::<f64>(), 2.);
}
#[test]
fn insufficient_fleet_does_not_earn_partial_reward() {
    let result = solve(vec![network(7, 0., 10., 1.)], vec![service(&[(7, 1)])]);
    assert_eq!(result.status, TreeStatus::Optimal, "{result:?}");
    assert_eq!(result.incumbent, 0.);
}
#[test]
fn integer_domain_rejects_only_fractional_common_times() {
    let result = solve(vec![network(7, 0.2, 0.8, 2.)], vec![service(&[(7, 1)])]);
    assert_eq!(result.status, TreeStatus::Optimal, "{result:?}");
    assert_eq!(result.incumbent, 0.);
}
#[test]
fn singleton_start_is_enforced_from_root() {
    let mut spec = service(&[(7, 1)]);
    spec.start = [8, 8];
    let result = solve(vec![network(7, 0., 6., 2.)], vec![spec]);
    assert_eq!(result.status, TreeStatus::Optimal, "{result:?}");
    assert_eq!(result.incumbent, 0.);
}
#[test]
fn domains_and_fixed_routes_survive_snapshot_forks() {
    let mut opts = options(vec![service(&[(7, 1), (42, 1)])]).node;
    let root = model::solve(
        Problem {
            rows: vec![],
            networks: vec![network(7, 0., 10., 1.), network(42, 0., 10., 1.)],
        },
        opts.clone(),
    )
    .unwrap();
    opts.synchronized_services.clear();
    let child = root.snapshot.with_synchronized_start(17, 7, 9).unwrap();
    let index = root.routes.iter().position(|r| r.network == 7).unwrap();
    let fixed = child.fix_route(index, 1.).unwrap();
    let result = fixed
        .solve_tree(baldes::master::tree::ModelTreeOptions {
            node: opts,
            ..options(vec![])
        })
        .unwrap();
    assert_eq!(result.status, TreeStatus::Optimal, "{result:?}");
    for r in result.routes {
        assert_eq!(r.route.route.resource_profile[1][0], 7.);
    }
    let parent = root.snapshot.solve_tree(options(vec![])).unwrap();
    for r in parent.routes {
        assert_eq!(r.route.route.resource_profile[1][0], 0.);
    }
}
#[test]
fn zero_node_budget_never_publishes_unsynchronized_incumbent() {
    let mut opts = options(vec![service(&[(7, 1), (42, 1)])]);
    opts.node_limit = 0;
    let result = baldes::master::tree::solve(
        Problem {
            rows: vec![],
            networks: vec![network(7, 0., 4., 1.), network(42, 6., 10., 1.)],
        },
        opts,
    )
    .unwrap();
    assert_eq!(result.status, TreeStatus::NodeLimit);
    assert!(!result.incumbent.is_finite());
}
#[test]
fn rejects_duplicate_occurrences_and_invalid_resources() {
    for spec in [
        SynchronizedService {
            members: vec![
                SynchronizedServiceMember {
                    network: 7,
                    vertex: 1
                };
                2
            ],
            ..service(&[])
        },
        SynchronizedService {
            resource: 1,
            ..service(&[(7, 1)])
        },
        SynchronizedService {
            start: [3, 2],
            ..service(&[(7, 1)])
        },
        SynchronizedService {
            vehicles_required: 1,
            ..service(&[(7, 1)])
        },
    ] {
        assert!(model::solve(
            Problem {
                rows: vec![],
                networks: vec![network(7, 0., 10., 2.)]
            },
            options(vec![spec]).node
        )
        .is_err());
    }
}
#[cfg(feature = "multi-time-windows")]
#[test]
fn nonconvex_windows_preserve_holes_and_waiting() {
    let mut a = network(7, 0., 10., 1.);
    a.graph
        .set_time_windows(
            1,
            0,
            &[Window::new(0., 2.).unwrap(), Window::new(8., 10.).unwrap()],
        )
        .unwrap();
    let result = solve(
        vec![a, network(42, 4., 9., 1.)],
        vec![service(&[(7, 1), (42, 1)])],
    );
    assert_eq!(result.status, TreeStatus::Optimal, "{result:?}");
    assert_eq!(result.incumbent, -1.);
    for r in result.routes {
        assert_eq!(r.route.route.resource_profile[1][0], 8.);
    }
}
#[test]
fn jointly_inconsistent_jobs_are_not_certified_independently() {
    let mut networks = vec![];
    for (id, order) in [(7, [1, 2]), (42, [2, 1])] {
        let mut graph = Graph::<2>::new(4, 1).unwrap();
        graph.set_bounds(0, 0, 0., 0.).unwrap();
        for v in 1..4 {
            graph.set_bounds(v, 0, 0., 6.).unwrap();
        }
        graph.add_arc(0, order[0], 0.).unwrap();
        let arc = graph.add_arc(order[0], order[1], 0.).unwrap();
        graph.arc_mut(arc).unwrap().consumption[0] = 1.;
        graph.add_arc(order[1], 3, 0.).unwrap();
        networks.push(Network {
            id,
            graph,
            multiplicity: Window::new(0., 1.).unwrap(),
            arc_entries: vec![vec![]; 3],
            route_entries: vec![],
            activity_entries: vec![],
        });
    }
    let mut a = service(&[(7, 1), (42, 1)]);
    a.start = [0, 6];
    let mut b = service(&[(7, 2), (42, 2)]);
    b.id = 18;
    b.start = [0, 6];
    let result = solve(networks, vec![a, b]);
    assert_eq!(result.status, TreeStatus::Optimal, "{result:?}");
    assert_eq!(result.incumbent, 0.);
    assert!(result.routes.is_empty());
}

#[test]
fn integer_domain_endpoints_are_validated_even_with_zero_node_budget() {
    let t = 1_i64 << 52;
    let mut spec = service(&[(7, 1)]);
    spec.start = [t, t];
    let mut opts = options(vec![spec.clone()]);
    opts.node_limit = 0;
    let problem = Problem {
        rows: vec![],
        networks: vec![network(7, 0., 10., 2.)],
    };
    let result = baldes::master::tree::solve(problem.clone(), opts).unwrap();
    assert_eq!(result.status, TreeStatus::NodeLimit);
    spec.start = [t + 1, t + 1];
    let mut opts = options(vec![spec]);
    opts.node_limit = 0;
    assert!(baldes::master::tree::solve(problem, opts).is_err());
}

#[test]
fn decimal_consumptions_can_wait_for_a_common_integer_start() {
    let mut a = network(7, 0., 10., 1.);
    a.graph.arc_mut(0).unwrap().consumption[0] = 0.1;
    let mut b = network(42, 0., 10., 1.);
    b.graph.arc_mut(0).unwrap().consumption[0] = 0.3;
    let result = solve(vec![a, b], vec![service(&[(7, 1), (42, 1)])]);
    assert_eq!(result.status, TreeStatus::Optimal, "{result:?}");
    for r in result.routes {
        assert_eq!(r.route.route.resource_profile[1][0], 1.);
    }
}

#[cfg(feature = "highs")]
#[test]
fn enumerated_mip_defers_until_common_starts_are_fixed() {
    let problem = Problem {
        rows: vec![],
        networks: vec![network(7, 0., 10., 2.)],
    };
    let root = model::solve(problem.clone(), options(vec![service(&[(7, 1)])]).node).unwrap();
    let enumerated = root
        .snapshot
        .enumerate(&model::EnumerationOptions::default())
        .unwrap();
    let mip = baldes::master::solvers::mip::MipOptions::default();
    assert!(enumerated.solve_enumerated_mip(&mip).unwrap().is_none());
    let mut spec = service(&[(7, 1)]);
    spec.start = [5, 5];
    let root = model::solve(problem, options(vec![spec]).node).unwrap();
    let enumerated = root
        .snapshot
        .enumerate(&model::EnumerationOptions::default())
        .unwrap();
    let result = enumerated.solve_enumerated_mip(&mip).unwrap().unwrap();
    assert_eq!(result.mip.solution.unwrap().objective, -1.);
    for fixed in result.route_values {
        assert_eq!(fixed.route.route.resource_profile[1][0], 5.);
    }
}

#[test]
fn a_conflict_with_a_fixed_route_proves_infeasibility() {
    let root = model::solve(
        Problem {
            rows: vec![],
            networks: vec![network(7, 0., 4., 1.), network(42, 0., 10., 1.)],
        },
        options(vec![service(&[(7, 1), (42, 1)])]).node,
    )
    .unwrap();
    let index = root.routes.iter().position(|r| r.network == 7).unwrap();
    let child = root
        .snapshot
        .fix_route(index, 1.)
        .unwrap()
        .with_synchronized_start(17, 6, 8)
        .unwrap();
    let result = child.solve_tree(options(vec![])).unwrap();
    assert_eq!(result.status, TreeStatus::Infeasible, "{result:?}");
    assert!(!result.incumbent.is_finite());
}

#[test]
fn stopping_after_an_unsynchronized_root_retains_a_feasible_witness_and_relaxation_bound() {
    let mut opts = options(vec![service(&[(7, 1), (42, 1)])]);
    opts.node_limit = 1;
    let result = baldes::master::tree::solve(
        Problem {
            rows: vec![],
            networks: vec![network(7, 0., 4., 1.), network(42, 6., 10., 1.)],
        },
        opts,
    )
    .unwrap();
    assert_eq!(result.status, TreeStatus::NodeLimit, "{result:?}");
    // Pricing retains the feasible empty solution before the relaxed root
    // serves an unsynchronizable job. No service means no earned reward.
    assert_eq!(result.incumbent, 0., "{result:?}");
    assert!(result.routes.is_empty());
    assert_eq!(result.master_values.len(), 1);
    assert!(result.master_values.iter().all(|&(_, value)| value == 0.));
    assert_eq!(result.nodes, 1);
    assert_eq!(result.processed_nodes, 1);
    assert_eq!(result.strong_branching_trials, 0);
    assert_eq!(result.bound, -1.);
    assert_eq!(result.open_nodes, 2);
}
