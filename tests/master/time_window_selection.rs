#![cfg(all(feature = "clp", feature = "multi-time-windows"))]
//! Root-tree branching on time-window selection counts.
use baldes::graph::Window;
use baldes::master::{
    root::{RootOptions, RootProblem},
    tree::{self, TreeOptions, TreeStatus},
};
use baldes::Graph;
use std::time::Duration;

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

fn graph(inst: &Instance) -> Graph<2> {
    let sink = inst.n + 1;
    let mut g = Graph::new(inst.n + 2, 1).unwrap();
    g.source = 0;
    g.sink = sink;
    for v in 0..=sink {
        g.set_bounds(v, 0, 0., 100.).unwrap();
    }
    for v in 1..=inst.n {
        let windows: Vec<_> = inst.windows[v]
            .iter()
            .map(|&(a, b)| Window::new(a, b).unwrap())
            .collect();
        g.set_time_windows(v, 0, &windows).unwrap();
    }
    for u in 0..sink {
        for v in 1..=sink {
            if u == v || (u == 0 && v == sink) {
                continue;
            }
            let id = g.add_arc(u, v, inst.travel[u][v]).unwrap();
            g.arc_mut(id).unwrap().consumption[0] = inst.travel[u][v];
        }
    }
    g
}

/// Exhaustive optimum: every elementary route under earliest MTW schedules,
/// then a minimum-cost exact cover of the customers by at most `vehicles` routes.
fn exhaustive(inst: &Instance, vehicles: usize) -> f64 {
    let n = inst.n;
    let full = (1usize << n) - 1;
    let mut best_route = vec![f64::INFINITY; full + 1];
    fn walk(inst: &Instance, at: usize, time: f64, mask: usize, cost: f64, best: &mut [f64]) {
        let sink = inst.n + 1;
        let back = time + inst.travel[at][sink];
        if mask != 0 && back <= 100. {
            best[mask] = best[mask].min(cost + inst.travel[at][sink]);
        }
        for v in 1..=inst.n {
            if mask >> (v - 1) & 1 == 1 {
                continue;
            }
            let arrive = time + inst.travel[at][v];
            let start = inst.windows[v]
                .iter()
                .find_map(|&(a, b)| (arrive.max(a) <= b).then_some(arrive.max(a)));
            if let Some(start) = start {
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
    walk(inst, 0, 0., 0, 0., &mut best_route);
    // cover[k][mask]: cheapest exact cover of `mask` with k routes.
    let mut cover = vec![vec![f64::INFINITY; full + 1]; vehicles + 1];
    cover[0][0] = 0.;
    for k in 1..=vehicles {
        for mask in 1..=full {
            let low = mask & mask.wrapping_neg();
            let mut sub = mask;
            while sub > 0 {
                if sub & low != 0 && best_route[sub].is_finite() {
                    let rest = cover[k - 1][mask ^ sub];
                    cover[k][mask] = cover[k][mask].min(rest + best_route[sub]);
                }
                sub = (sub - 1) & mask;
            }
        }
    }
    (1..=vehicles)
        .map(|k| cover[k][full])
        .fold(f64::INFINITY, f64::min)
}

fn options(selection: bool) -> TreeOptions {
    let mut root = RootOptions::default();
    root.cg.stabilize = false;
    let mut options = TreeOptions {
        root,
        time_limit: Duration::from_secs(30),
        ..Default::default()
    };
    options.branching.time_window_selection = selection.then_some(0);
    options
}

#[test]
fn window_selection_branching_matches_exhaustive_optima_in_input_arc_ids() {
    let mut solved = 0;
    for seed in 0..12 {
        let inst = instance(6, seed);
        let vehicles = 3;
        let optimum = exhaustive(&inst, vehicles);
        if !optimum.is_finite() {
            continue;
        }
        solved += 1;
        let g = graph(&inst);
        let input_arcs: std::collections::HashSet<_> = g.arcs.iter().map(|a| a.id).collect();
        for selection in [false, true] {
            let problem = RootProblem::set_partitioning(g.clone(), vehicles as f64).unwrap();
            let output = tree::solve(problem, options(selection)).unwrap();
            assert_eq!(
                output.status,
                TreeStatus::Optimal,
                "seed {seed}: {output:?}"
            );
            assert!(
                (output.incumbent - optimum).abs() < 1e-6,
                "seed {seed} {selection}: {output:?} vs {optimum}"
            );
            let mut covered = vec![0; inst.n + 2];
            for route in &output.routes {
                assert!(
                    route.arcs.iter().all(|id| input_arcs.contains(id)),
                    "{route:?}"
                );
                for &v in &route.nodes[1..route.nodes.len() - 1] {
                    covered[v as usize] += 1;
                }
            }
            assert!(
                covered[1..=inst.n].iter().all(|&c| c == 1),
                "seed {seed}: {covered:?}"
            );
        }
    }
    assert!(solved >= 6, "only {solved} feasible instances");
}

#[test]
fn window_selection_requires_a_disposable_resource() {
    let inst = instance(4, 1);
    let problem = RootProblem::set_partitioning(graph(&inst), 2.).unwrap();
    let mut controls = options(true);
    controls.branching.time_window_selection = Some(3);
    assert!(tree::solve(problem, controls).is_err());
}

#[test]
fn split_trees_branch_on_summed_arc_flows_then_window_counts() {
    use baldes::master::branching::{BranchingOptions, WindowGroup};
    use baldes::master::tree::{candidates::candidates, BranchConstraint};
    use baldes::master::RowSense;
    let (mut fractional_windows, mut summed_arcs) = (0, 0);
    for seed in 0..40 {
        let inst = instance(6, seed);
        if !exhaustive(&inst, 3).is_finite() {
            continue;
        }
        let mut g = graph(&inst);
        let split = g.split_time_window_arcs(0).unwrap();
        let problem = RootProblem::set_partitioning(g, 3.).unwrap();
        let mut root = RootOptions::default();
        root.cg.stabilize = false;
        root.enumerate = false;
        let result = baldes::master::root::solve(problem.clone(), root).unwrap();
        let values = result.route_values();
        let flow = |ids: &[i32]| -> f64 {
            values
                .iter()
                .map(|(r, x)| x * r.arcs.iter().filter(|a| ids.contains(a)).count() as f64)
                .sum()
        };
        let mut options = BranchingOptions::arc_flow_only();
        options.time_window_selection = Some(0);
        options.window_arcs = Some(split.clone());
        // Default order: arc flows summed over each input arc's copies.
        let copied: std::collections::HashSet<i32> = split
            .arcs
            .iter()
            .filter(|a| {
                split
                    .arcs
                    .iter()
                    .filter(|b| b.original == a.original)
                    .count()
                    > 1
            })
            .map(|a| a.arc)
            .collect();
        for c in candidates(&problem, &result, &options) {
            match &c.down {
                BranchConstraint::ArcFlow { arc, .. } => {
                    assert!(
                        !copied.contains(&problem.graph.arcs[*arc].id),
                        "seed {seed}: copy {arc}"
                    )
                }
                BranchConstraint::WindowFlow {
                    group: WindowGroup::Arc { original },
                    ..
                } => {
                    let ids: Vec<_> = split
                        .arcs
                        .iter()
                        .filter(|a| a.original == *original)
                        .map(|a| a.arc)
                        .collect();
                    assert!((c.value - flow(&ids)).abs() < 1e-9);
                    summed_arcs += 1;
                }
                other => panic!("unexpected family {other:?}"),
            }
        }
        // Window counts, with their family moved first.
        options.priorities.time_window_selection = 20.;
        let found = candidates(&problem, &result, &options);
        for ((vertex, window), ids) in split.selections() {
            let value = flow(&ids);
            if (value - value.round()).abs() <= 1e-6 {
                continue;
            }
            fractional_windows += 1;
            let c = found
                .iter()
                .find(|c| matches!(c.down, BranchConstraint::WindowFlow {
                    group: WindowGroup::Selection { vertex: v, window: w }, .. } if v == vertex && w == window))
                .unwrap_or_else(|| panic!("seed {seed}: no candidate ({vertex}, {window})"));
            assert!((c.value - value).abs() < 1e-9);
            assert!(
                matches!(&c.down, BranchConstraint::WindowFlow { sense: RowSense::LessEqual, rhs, arcs, .. }
                if *rhs == value.floor() && arcs.len() == ids.len())
            );
            let others = split
                .selections()
                .into_iter()
                .filter(|&((v, w), _)| v == vertex && w != window)
                .map(|(_, ids)| ids.len())
                .sum::<usize>();
            if value.ceil() == 1. {
                // Right child of Li et al.: only this window remains for the customer.
                assert!(
                    matches!(&c.up, BranchConstraint::WindowFlow { sense: RowSense::LessEqual, rhs, arcs, .. }
                    if *rhs == 0. && arcs.len() == others)
                );
            } else {
                assert!(
                    matches!(&c.up, BranchConstraint::WindowFlow { sense: RowSense::GreaterEqual, rhs, .. }
                    if *rhs == value.ceil())
                );
            }
        }
    }
    assert!(
        fractional_windows > 0,
        "no fractional window count at any root"
    );
    assert!(summed_arcs > 0, "no fractional summed arc flow at any root");
}
