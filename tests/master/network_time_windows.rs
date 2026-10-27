#![cfg(all(feature = "clp", feature = "multi-time-windows"))]
//! Network-tree branching on time-window selection counts.
use baldes::{
    graph::{Graph, Window},
    master::{
        model::{self, Network, Problem, SynchronizedService, SynchronizedServiceMember},
        routing::Row,
        tree::TreeStatus,
        RowSense,
    },
};
use std::time::Duration;

fn windows(pairs: &[(f64, f64)]) -> Vec<Window> {
    pairs
        .iter()
        .map(|&(a, b)| Window::new(a, b).unwrap())
        .collect()
}

fn options(selection: bool) -> baldes::master::tree::ModelTreeOptions {
    let mut options = baldes::master::tree::ModelTreeOptions::default();
    options.time_limit = Duration::from_secs(30);
    options.node.cg.time_limit = options.time_limit;
    options.node.cg.stabilize = false;
    options.node.cg.artificial_cost = 1000.;
    options.time_window_selection = selection.then_some(0);
    options
}

struct Instance {
    n: usize,
    travel: Vec<Vec<f64>>,
    windows: Vec<Vec<(f64, f64)>>,
}

/// Customers 1..=n with two disjoint windows each, depot 0 and sink n + 1.
fn instance(n: usize, seed: u64) -> Instance {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut next = |m: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) % m
    };
    let points: Vec<_> = (0..n + 2)
        .map(|i| {
            if i == 0 || i == n + 1 {
                (5., 5.)
            } else {
                (next(11) as f64, next(11) as f64)
            }
        })
        .collect();
    let travel = points
        .iter()
        .map(|a| {
            points
                .iter()
                .map(|b| ((a.0 - b.0) as f64).hypot(a.1 - b.1).round() + 1.)
                .collect()
        })
        .collect();
    let windows = (0..n + 2)
        .map(|i| {
            if i == 0 || i == n + 1 {
                vec![(0., 100.)]
            } else {
                let a = next(30) as f64;
                let b = a + 6. + next(30) as f64;
                vec![(a, a + 4.), (b, b + 4.)]
            }
        })
        .collect();
    Instance { n, travel, windows }
}

/// One network covering every customer exactly once with at most `vehicles` routes.
fn problem(inst: &Instance, vehicles: f64) -> Problem<2> {
    let sink = inst.n + 1;
    let mut graph = Graph::<2>::new(inst.n + 2, 1).unwrap();
    graph.source = 0;
    graph.sink = sink;
    for v in 0..=sink {
        graph.set_bounds(v, 0, 0., 100.).unwrap();
    }
    for v in 1..=inst.n {
        let w: Vec<_> = inst.windows[v].clone();
        graph.set_time_windows(v, 0, &windows(&w)).unwrap();
    }
    let mut entries = vec![];
    for u in 0..sink {
        for v in 1..=sink {
            if u == v || (u == 0 && v == sink) {
                continue;
            }
            let id = graph.add_arc(u, v, inst.travel[u][v]).unwrap();
            graph.arc_mut(id).unwrap().consumption[0] = inst.travel[u][v];
            entries.push(if v == sink { vec![] } else { vec![(v - 1, 1.)] });
        }
    }
    Problem {
        rows: (0..inst.n)
            .map(|_| Row {
                sense: RowSense::Equal,
                rhs: 1.,
            })
            .collect(),
        networks: vec![Network {
            id: 3,
            graph,
            multiplicity: Window::new(0., vehicles).unwrap(),
            arc_entries: entries,
            route_entries: vec![],
            activity_entries: vec![],
        }],
    }
}

/// Exhaustive optimum under earliest MTW schedules (see time_window_selection.rs).
fn exhaustive(inst: &Instance, vehicles: usize) -> f64 {
    let n = inst.n;
    let full = (1usize << n) - 1;
    let mut best = vec![f64::INFINITY; full + 1];
    fn walk(inst: &Instance, at: usize, time: f64, mask: usize, cost: f64, best: &mut [f64]) {
        let sink = inst.n + 1;
        if mask != 0 && time + inst.travel[at][sink] <= 100. {
            best[mask] = best[mask].min(cost + inst.travel[at][sink]);
        }
        for v in 1..=inst.n {
            if mask >> (v - 1) & 1 == 1 {
                continue;
            }
            let arrive = time + inst.travel[at][v];
            if let Some(start) = inst.windows[v]
                .iter()
                .find_map(|&(a, b)| (arrive.max(a) <= b).then_some(arrive.max(a)))
            {
                walk(
                    inst,
                    v,
                    start,
                    mask | 1 << (v - 1),
                    cost + inst.travel[at][v],
                    best,
                );
            }
        }
    }
    walk(inst, 0, 0., 0, 0., &mut best);
    let mut cover = vec![vec![f64::INFINITY; full + 1]; vehicles + 1];
    cover[0][0] = 0.;
    for k in 1..=vehicles {
        for mask in 1..=full {
            let low = mask & mask.wrapping_neg();
            let mut sub = mask;
            while sub > 0 {
                if sub & low != 0 && best[sub].is_finite() {
                    cover[k][mask] = cover[k][mask].min(cover[k - 1][mask ^ sub] + best[sub]);
                }
                sub = (sub - 1) & mask;
            }
        }
    }
    (1..=vehicles)
        .map(|k| cover[k][full])
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn network_window_selection_matches_exhaustive_optima_in_input_arc_ids() {
    // closed[selection]: instances proved optimal without / with window counts.
    let (mut feasible, mut closed) = (0, [0, 0]);
    for seed in 0..12 {
        let inst = instance(6, seed);
        let optimum = exhaustive(&inst, 3);
        if !optimum.is_finite() {
            continue;
        }
        feasible += 1;
        let base = problem(&inst, 3.);
        let input: std::collections::HashSet<_> =
            base.networks[0].graph.arcs.iter().map(|a| a.id).collect();
        for selection in [false, true] {
            let output = baldes::master::tree::solve(base.clone(), options(selection)).unwrap();
            // A stalled column generation leaves nodes open in either mode.
            if output.status == TreeStatus::Incomplete {
                continue;
            }
            assert_eq!(
                output.status,
                TreeStatus::Optimal,
                "seed {seed} {selection}: {output:?}"
            );
            closed[selection as usize] += 1;
            assert!(
                (output.incumbent - optimum).abs() < 1e-6,
                "seed {seed} {selection}: {} vs {optimum}",
                output.incumbent
            );
            for fixed in &output.routes {
                assert!(fixed.route.route.arcs.iter().all(|id| input.contains(id)));
            }
        }
    }
    eprintln!("feasible {feasible}, closed without/with window counts {closed:?}");
    assert!(feasible >= 6, "only {feasible} feasible instances");
    assert!(
        closed[1] >= closed[0] && closed[1] > 0,
        "{closed:?} of {feasible}"
    );
}

/// Members in networks 7 and 42 with different window sets that overlap
/// only in [6, 8]; both vehicles must start the service there.
fn synchronized(
    first: &[(f64, f64)],
    second: &[(f64, f64)],
) -> baldes::master::tree::ModelTreeOutput {
    let network = |id: i32, w: &[(f64, f64)]| {
        let mut graph = Graph::<2>::new(3, 1).unwrap();
        graph.set_bounds(0, 0, 0., 0.).unwrap();
        graph.set_bounds(1, 0, 0., 20.).unwrap();
        graph.set_bounds(2, 0, 0., 100.).unwrap();
        graph.set_time_windows(1, 0, &windows(w)).unwrap();
        graph.add_arc(0, 1, 0.).unwrap();
        graph.add_arc(1, 2, 0.).unwrap();
        Network {
            id,
            graph,
            multiplicity: Window::new(0., 1.).unwrap(),
            arc_entries: vec![vec![], vec![]],
            route_entries: vec![],
            activity_entries: vec![],
        }
    };
    let mut opts = options(true);
    opts.node.synchronized_services = vec![SynchronizedService {
        id: 17,
        resource: 0,
        vehicles_required: 2,
        reward: 1.,
        start: [0, 20],
        members: vec![
            SynchronizedServiceMember {
                network: 7,
                vertex: 1,
            },
            SynchronizedServiceMember {
                network: 42,
                vertex: 1,
            },
        ],
    }];
    baldes::master::tree::solve(
        Problem {
            rows: vec![],
            networks: vec![network(7, first), network(42, second)],
        },
        opts,
    )
    .unwrap()
}

#[test]
fn synchronized_members_meet_in_their_only_common_window() {
    let result = synchronized(&[(0., 2.), (6., 8.)], &[(3., 5.), (6., 8.)]);
    assert_eq!(result.status, TreeStatus::Optimal, "{result:?}");
    assert_eq!(result.incumbent, -1.);
    assert_eq!(result.routes.len(), 2);
    let starts: Vec<_> = result
        .routes
        .iter()
        .map(|r| r.route.route.resource_profile[1][0])
        .collect();
    assert_eq!(starts[0], starts[1]);
    assert!((6. ..=8.).contains(&starts[0]), "{starts:?}");
    for fixed in &result.routes {
        assert!(fixed.route.route.arcs.iter().all(|&id| id < 2), "{fixed:?}");
    }
}

#[test]
fn synchronized_members_without_a_common_window_are_not_served() {
    let result = synchronized(&[(0., 2.), (9., 11.)], &[(3., 5.), (6., 8.)]);
    assert_eq!(result.status, TreeStatus::Optimal, "{result:?}");
    assert_eq!(result.incumbent, 0.);
    assert!(result.routes.is_empty());
}

#[test]
fn window_selection_requires_a_fresh_problem() {
    let inst = instance(4, 1);
    let root = model::solve(problem(&inst, 2.), options(false).node).unwrap();
    assert!(root.snapshot.solve_tree(options(true)).is_err());
}
