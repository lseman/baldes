use super::*;
use crate::graph::Window;
use crate::Graph;

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

fn node_options() -> Options {
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 1000.;
    options.gap_enumeration = false;
    options
}

fn flow(output: &Output<2>, ids: &[i32]) -> f64 {
    output
        .routes
        .iter()
        .zip(&output.values)
        .map(|(r, x)| x * r.route.arcs.iter().filter(|id| ids.contains(id)).count() as f64)
        .sum()
}

fn assert_row(node: &NodeSnapshot<2>, index: usize, sense: RowSense, rhs: f64, ids: &[i32]) {
    let row = &node.state.shared_rows[index];
    assert_eq!(row.sense, sense);
    assert_eq!(row.rhs, rhs);
    let oracle = &node.state.oracles[0];
    for (arc, entries) in oracle
        .network
        .graph
        .arcs
        .iter()
        .zip(oracle.network.arc_entries.iter())
    {
        let coefficient: f64 = entries
            .iter()
            .filter(|&&(r, _)| r == index)
            .map(|&(_, v)| v)
            .sum();
        assert_eq!(
            coefficient,
            if ids.contains(&arc.id) { 1. } else { 0. },
            "arc {}",
            arc.id
        );
    }
}

#[test]
fn time_window_selection_generators_sum_arc_copies_and_restrict_both_children() {
    let (mut fractional_windows, mut summed_arcs, mut feasible_children) = (0, 0, 0);
    for seed in 0..40 {
        let inst = instance(6, seed);
        if !exhaustive(&inst, 3).is_finite() {
            continue;
        }
        let mut problem = Problem::set_partitioning(graph(&inst), 3.).unwrap();
        let windows = split(&mut problem, 0).unwrap();
        let result = crate::master::model::solve(problem, node_options()).unwrap();
        let node = &result.snapshot;
        let parent_rows = node.state.shared_rows.len();
        let oracle = &node.state.oracles[0];
        let activities = HashMap::from([(
            oracle.network.id,
            oracle
                .network
                .graph
                .arcs
                .iter()
                .map(|a| flow(&result, &[a.id]))
                .collect::<Vec<_>>(),
        )]);
        let split = &windows[0].split;
        let mut copies = BTreeMap::<i32, Vec<i32>>::new();
        for a in &split.arcs {
            copies.entry(a.original).or_default().push(a.arc);
        }
        let mut best_arc: Option<(f64, Vec<i32>, f64)> = None;
        for ids in copies.into_values().filter(|ids| ids.len() > 1) {
            let value = flow(&result, &ids);
            if (value - value.round()).abs() > 1e-6
                && best_arc
                    .as_ref()
                    .map_or(true, |b| fraction_score(value) < b.0)
            {
                best_arc = Some((fraction_score(value), ids, value));
            }
        }
        let children = arc_children(node, &windows, &activities, 1e-6).unwrap();
        assert_eq!(children.is_some(), best_arc.is_some(), "seed {seed}");
        if let Some((_, ids, value)) = best_arc {
            summed_arcs += 1;
            let children = children.unwrap();
            assert_eq!(children.len(), 2);
            assert_row(
                &children[0],
                parent_rows + usize::from(value.floor() != 0.),
                if value.floor() == 0. {
                    RowSense::Equal
                } else {
                    RowSense::LessEqual
                },
                value.floor(),
                &ids,
            );
            assert_row(
                &children[1],
                parent_rows,
                RowSense::GreaterEqual,
                value.ceil(),
                &ids,
            );
            for (up, child) in children.iter().enumerate() {
                let solved = child.solve(node_options()).unwrap();
                if !crate::master::model::model_strong::proven_infeasible(&solved)
                    && solved.master.status == cg::Status::Optimal
                {
                    feasible_children += 1;
                    let actual = flow(&solved, &ids);
                    assert!(if up == 0 {
                        actual <= value.floor() + 1e-6
                    } else {
                        actual >= value.ceil() - 1e-6
                    });
                }
            }
        }
        let selections = split.selections();
        let mut best_selection: Option<(f64, usize, usize, f64)> = None;
        for (&(vertex, window), ids) in &selections {
            let value = flow(&result, ids);
            if (value - value.round()).abs() > 1e-6
                && best_selection
                    .as_ref()
                    .map_or(true, |b| fraction_score(value) < b.0)
            {
                best_selection = Some((fraction_score(value), vertex, window, value));
            }
        }
        let children = selection_children(node, &windows, &activities, 1e-6).unwrap();
        assert_eq!(children.is_some(), best_selection.is_some(), "seed {seed}");
        if let Some((_, vertex, window, value)) = best_selection {
            fractional_windows += 1;
            let children = children.unwrap();
            let ids = &selections[&(vertex, window)];
            let other_ids = others(&selections, vertex, window);
            assert_row(
                &children[0],
                parent_rows + usize::from(value.floor() != 0.),
                if value.floor() == 0. {
                    RowSense::Equal
                } else {
                    RowSense::LessEqual
                },
                value.floor(),
                ids,
            );
            assert_row(
                &children[1],
                parent_rows,
                RowSense::GreaterEqual,
                value.ceil(),
                ids,
            );
            assert_row(
                &children[1],
                parent_rows + 1,
                RowSense::Equal,
                0.,
                &other_ids,
            );
            for (up, child) in children.iter().enumerate() {
                let solved = child.solve(node_options()).unwrap();
                if !crate::master::model::model_strong::proven_infeasible(&solved)
                    && solved.master.status == cg::Status::Optimal
                {
                    feasible_children += 1;
                    let actual = flow(&solved, ids);
                    assert!(if up == 0 {
                        actual <= value.floor() + 1e-6
                    } else {
                        actual >= value.ceil() - 1e-6
                    });
                    if up == 1 {
                        assert!(flow(&solved, &other_ids).abs() <= 1e-6);
                    }
                }
            }
        }
        assert_eq!(
            node.state.shared_rows.len(),
            parent_rows,
            "children changed parent"
        );
    }
    assert!(
        fractional_windows > 0,
        "no fractional window count at any root"
    );
    assert!(summed_arcs > 0, "no fractional summed arc flow at any root");
    assert!(feasible_children > 0, "no feasible child solved");
}
