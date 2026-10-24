use super::reference::{price, PricingOptions, PricingStatus};
use super::{Node, Problem, Travel};

pub(super) struct Lcg(pub u64);
impl Lcg {
    pub fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Random complete instance with paper-style truncated deviations.
pub(super) fn random(seed: u64, customers: usize, gq: usize, gt: usize, alpha: f64) -> Problem {
    let mut rng = Lcg(seed);
    let points = (0..=customers)
        .map(|_| ((50.0 * rng.next()).floor(), (50.0 * rng.next()).floor()))
        .collect::<Vec<_>>();
    let nodes = (0..=customers)
        .map(|i| {
            if i == 0 {
                return Node {
                    demand: 0.0,
                    demand_deviation: 0.0,
                    window: (0.0, 230.0),
                    service: 0.0,
                };
            }
            let open = (100.0 * rng.next()).floor();
            Node {
                demand: (1.0 + 9.0 * rng.next()).floor(),
                demand_deviation: 0.0,
                window: (open, open + 30.0 + (60.0 * rng.next()).floor()),
                service: 5.0,
            }
        })
        .collect();
    let travel = points
        .iter()
        .enumerate()
        .map(|(i, a)| {
            points
                .iter()
                .enumerate()
                .map(|(j, b)| {
                    let d = ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
                    let d = (10.0 * d).trunc() / 10.0;
                    (i != j).then_some(Travel {
                        time: d,
                        time_deviation: 0.0,
                        cost: d,
                    })
                })
                .collect()
        })
        .collect();
    Problem {
        nodes,
        travel,
        depot: 0,
        capacity: 20.0,
        demand_budget: gq,
        time_budget: gt,
    }
    .with_relative_deviations(alpha, alpha)
    .unwrap()
}

/// Every robust feasible elementary route, by depth-first prefix extension.
/// Infeasible prefixes are pruned: both worst-case vectors only grow.
pub(super) fn enumerate(p: &Problem) -> Vec<(Vec<usize>, f64)> {
    fn dfs(p: &Problem, path: &mut Vec<usize>, out: &mut Vec<(Vec<usize>, f64)>) {
        let mut closed = path.clone();
        closed.push(p.depot);
        if path.len() > 1 {
            if let Ok(s) = p.replay(&closed) {
                out.push((closed, s.cost));
            }
        }
        for next in 0..p.nodes.len() {
            if next == p.depot || path.contains(&next) {
                continue;
            }
            path.push(next);
            let mut prefix = path.clone();
            prefix.push(p.depot);
            let ok = prefix
                .windows(2)
                .take(path.len() - 1)
                .try_fold(p.initial_state(), |s, w| p.extend(&s, w[0], w[1]));
            if ok.is_some() {
                dfs(p, path, out);
            }
            path.pop();
        }
    }
    let mut out = Vec::new();
    dfs(p, &mut vec![p.depot], &mut out);
    out
}

fn figure_one(v3_close: f64) -> Problem {
    let node = |window, service| Node {
        demand: 0.0,
        demand_deviation: 0.0,
        window,
        service,
    };
    let mut travel = vec![vec![None; 4]; 4];
    for (i, j, time, time_deviation) in [
        (0, 1, 2.0, 3.0),
        (1, 2, 5.0, 1.0),
        (2, 3, 3.0, 2.0),
        (3, 0, 3.0, 1.0),
    ] {
        travel[i][j] = Some(Travel {
            time,
            time_deviation,
            cost: time,
        });
    }
    Problem {
        nodes: vec![
            node((0.0, 22.0), 0.0),
            node((5.0, 8.0), 1.0),
            node((8.0, 12.0), 1.0),
            node((11.0, v3_close), 1.0),
        ],
        travel,
        depot: 0,
        capacity: 0.0,
        demand_budget: 0,
        time_budget: 1,
    }
}

#[test]
fn figure_one_worst_case_starts_skip_the_largest_deviation() {
    let s = figure_one(17.0).replay(&[0, 1, 2, 3, 0]).unwrap();
    let starts = |g: usize| s.states[1..].iter().map(|x| x.start[g]).collect::<Vec<_>>();
    assert_eq!(starts(0), [5.0, 11.0, 15.0, 19.0]);
    assert_eq!(starts(1), [5.0, 12.0, 17.0, 21.0]);
}

#[test]
fn figure_one_tight_window_is_nominal_but_not_robust_feasible() {
    let mut p = figure_one(15.0);
    assert!(p.replay(&[0, 1, 2, 3, 0]).is_err());
    p.time_budget = 0;
    assert!(p.replay(&[0, 1, 2, 3, 0]).is_ok());
}

#[test]
fn worst_case_load_adds_the_largest_demand_deviations() {
    let mut rng = Lcg(7);
    for _ in 0..200 {
        let k = 1 + (8.0 * rng.next()) as usize;
        let gamma = (6.0 * rng.next()) as usize;
        let mut p = random(rng.0, k, gamma, 0, 0.0);
        p.capacity = 1e9;
        for node in &mut p.nodes[1..] {
            node.demand_deviation = (20.0 * rng.next()).floor();
        }
        for node in &mut p.nodes {
            node.window = (0.0, 1e9);
        }
        let route = (0..=k).chain([0]).collect::<Vec<_>>();
        let load = p.replay(&route).unwrap().states[k].load.clone();
        let mut deviations = p.nodes[1..]
            .iter()
            .map(|n| n.demand_deviation)
            .collect::<Vec<_>>();
        deviations.sort_by(|a, b| b.total_cmp(a));
        let nominal = p.nodes.iter().map(|n| n.demand).sum::<f64>();
        for (g, &value) in load.iter().enumerate() {
            assert_eq!(value, nominal + deviations.iter().take(g).sum::<f64>());
        }
    }
}

#[test]
fn larger_budgets_never_admit_more_routes() {
    for seed in 1..6 {
        let mut previous = usize::MAX;
        for gamma in 0..4 {
            let count = enumerate(&random(seed, 6, gamma, gamma, 0.5)).len();
            assert!(count <= previous, "seed {seed}, gamma {gamma}");
            previous = count;
        }
    }
}

#[test]
fn pricing_matches_enumeration_for_every_budget() {
    for seed in 1..9 {
        for (gq, gt) in [(0, 0), (1, 0), (0, 1), (2, 2), (6, 6)] {
            let p = random(seed, 6, gq, gt, 0.5);
            let mut rng = Lcg(seed * 31 + gq as u64);
            let duals = (0..p.nodes.len())
                .map(|i| {
                    if i == 0 {
                        0.0
                    } else {
                        (80.0 * rng.next()).floor()
                    }
                })
                .collect::<Vec<_>>();
            let best = enumerate(&p)
                .iter()
                .map(|(r, c)| r.iter().fold(*c - 3.0, |c, &v| c - duals[v]))
                .fold(f64::INFINITY, f64::min);
            for dominance in [true, false] {
                let options = PricingOptions {
                    dominance,
                    ..Default::default()
                };
                let result = price(&p, &duals, 3.0, options).unwrap();
                assert_eq!(result.status, PricingStatus::Complete);
                let found = result
                    .routes
                    .first()
                    .map_or(f64::INFINITY, |r| r.reduced_cost);
                assert!(
                    (found - best).abs() < 1e-9 || found == best,
                    "seed {seed} budgets ({gq},{gt}) dominance {dominance}: {found} vs {best}"
                );
                assert!(result
                    .routes
                    .windows(2)
                    .all(|w| w[0].reduced_cost <= w[1].reduced_cost));
            }
        }
    }
}

#[test]
fn label_limit_is_reported() {
    let p = random(3, 6, 1, 1, 0.25);
    let options = PricingOptions {
        max_labels: 3,
        ..Default::default()
    };
    let result = price(&p, &vec![0.0; p.nodes.len()], 0.0, options).unwrap();
    assert_eq!(result.status, PricingStatus::LabelLimit);
}
