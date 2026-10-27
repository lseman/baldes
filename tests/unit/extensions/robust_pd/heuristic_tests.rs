use super::*;
use crate::extensions::robust_pd::{Instance, Parameters};

/// Random paired instance with integer data and tight windows.
fn instance(n: usize, seed: u64, gq: usize, gt: usize) -> Problem {
    let mut state = seed.wrapping_mul(6364136223846793005) ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = |m: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) % m
    };
    let mut pickups = vec![];
    let mut deliveries = vec![];
    for _ in 0..n {
        let demand = 1 + next(5) as i64;
        let open = next(40);
        pickups.push((next(30), next(30), demand, open, open + 20 + next(40)));
        let open = 10 + next(60);
        deliveries.push((next(30), next(30), -demand, open, open + 20 + next(50)));
    }
    let mut text = format!("n {n} Q 10 r_mod 2\n0 0 0 0 0 0 220\n");
    for (k, r) in pickups.iter().chain(&deliveries).enumerate() {
        text += &format!("{} {} {} 1 {} {} {}\n", k + 1, r.0, r.1, r.2, r.3, r.4);
    }
    text += &format!("{} 0 0 0 0 0 220\n", 2 * n + 1);
    Problem::new(
        Instance::from_text(&text).unwrap(),
        Parameters {
            demand_budget: gq,
            time_budget: gt,
            demand_deviation: 0.5,
            time_deviation: 0.3,
            vehicle_cost: 100.,
            adjust_demands: false,
            ..Default::default()
        },
    )
    .unwrap()
}

fn search(p: &Problem) -> Search<'_> {
    let n = p.instance.requests;
    Search {
        problem: p,
        n,
        budget: p.parameters.time_budget.min(2 * n + 1),
        rng: Xoroshiro128Plus::from_seed(7),
        deadline: Instant::now() + Duration::from_secs(60),
    }
}

#[test]
fn cached_insertion_checks_match_full_route_replay() {
    let mut compared = 0;
    let mut accepted = 0;
    for seed in 0..25 {
        for (gq, gt) in [(0, 0), (1, 1), (2, 3), (5, 5)] {
            let p = instance(5, seed, gq, gt);
            let s = search(&p);
            // Feasible routes grown by verified insertions, then every
            // placement of every absent request compared with replay.
            let mut routes = vec![];
            for r in 0..5 {
                let allow_new = routes.is_empty() || r % 2 == 0;
                let _ = s.insert(&mut routes, r, allow_new);
            }
            for (index, route) in routes.iter().enumerate() {
                for r in 0..5 {
                    if route.nodes.contains(&(r + 1)) {
                        continue;
                    }
                    let mut fast = vec![];
                    s.placements(route, index, r, &mut fast);
                    let fast: Vec<_> = fast
                        .iter()
                        .map(|c| (c.pickup_after, c.delivery_after))
                        .collect();
                    let m = route.nodes.len() - 1;
                    for i in 0..m {
                        for j in i..m {
                            let nodes = s.insert_nodes(&route.nodes, r, i, j);
                            let feasible = p.replay(&nodes).is_ok();
                            assert_eq!(
                                fast.contains(&(i, j)),
                                feasible,
                                "seed={seed} ({gq},{gt}) {nodes:?}"
                            );
                            compared += 1;
                            accepted += usize::from(feasible);
                        }
                    }
                }
            }
        }
    }
    assert!(compared > 2_000 && accepted > 100, "{compared} {accepted}");
}

/// Every precedence-respecting elementary route of an n-request instance.
fn paired_routes(n: usize) -> Vec<Vec<usize>> {
    fn visit(route: &mut Vec<usize>, n: usize, out: &mut Vec<Vec<usize>>) {
        let open = (1..=n).any(|i| route.contains(&i) && !route.contains(&(i + n)));
        if route.len() > 1 && !open {
            let mut done = route.clone();
            done.push(2 * n + 1);
            out.push(done);
        }
        for v in 1..=2 * n {
            if !route.contains(&v) && (v <= n || route.contains(&(v - n))) {
                route.push(v);
                visit(route, n, out);
                route.pop();
            }
        }
    }
    let mut out = vec![];
    visit(&mut vec![0], n, &mut out);
    out
}

fn optimum(p: &Problem, all: &[Vec<usize>]) -> f64 {
    let n = p.instance.requests;
    let full = (1 << n) - 1;
    let mut cheapest = vec![f64::INFINITY; full + 1];
    for r in all {
        if let Ok(schedule) = p.replay(r) {
            let mask = r
                .iter()
                .filter(|v| (1..=n).contains(*v))
                .fold(0, |m, v| m | 1 << (v - 1));
            cheapest[mask] = cheapest[mask].min(schedule.cost);
        }
    }
    let mut best = vec![f64::INFINITY; full + 1];
    best[0] = 0.;
    for mask in 1..=full {
        let low = mask & mask.wrapping_neg();
        let mut sub = mask;
        while sub > 0 {
            if sub & low != 0 {
                best[mask] = best[mask].min(cheapest[sub] + best[mask ^ sub]);
            }
            sub = (sub - 1) & mask;
        }
    }
    best[full]
}

#[test]
fn heuristics_return_valid_partitions_close_to_the_exact_optimum() {
    for method in [
        HeuristicMethod::IteratedLocalSearch,
        HeuristicMethod::Genetic,
    ] {
        partitions_close_to_optimum(method);
    }
}

fn partitions_close_to_optimum(method: HeuristicMethod) {
    let all = paired_routes(4);
    let (mut solved, mut optimal) = (0, 0);
    for seed in 0..30 {
        for (gq, gt) in [(0, 0), (1, 1), (2, 2)] {
            let p = instance(4, seed, gq, gt);
            let exact = optimum(&p, &all);
            let options = HeuristicOptions {
                time_limit: Duration::from_secs(2),
                iterations: 200,
                seed,
                method,
            };
            let Some(solution) = p.heuristic(options) else {
                assert!(
                    (0..4).any(|r| search(&p).alone(r).is_none()),
                    "seed={seed}: heuristic failed on a servable instance"
                );
                continue;
            };
            let mut covered = [0; 4];
            let mut cost = 0.;
            for route in &solution.routes {
                cost += p.replay(route).unwrap().cost;
                for &v in route.iter().filter(|v| (1..=4).contains(*v)) {
                    covered[v - 1] += 1;
                }
            }
            assert_eq!(covered, [1; 4], "seed={seed}");
            assert!((cost - solution.cost).abs() < 1e-9);
            assert!(solution.cost >= exact - 1e-9, "seed={seed}: below optimum");
            solved += 1;
            optimal += usize::from(solution.cost <= exact + 1e-9);
        }
    }
    assert!(solved > 40, "{solved}");
    assert!(
        optimal * 10 >= solved * 9,
        "{method:?}: {optimal}/{solved} optimal"
    );
}
