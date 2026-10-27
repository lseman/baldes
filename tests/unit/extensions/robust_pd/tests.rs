use super::*;

fn fixture(capacity: f64, close: f64, gq: usize, gt: usize) -> Problem {
    let text = format!("n 2 Q {capacity} r_mod 2\n0 0 0 0 0 0 100\n1 1 0 0 4 0 50\n2 2 0 0 4 0 50\n3 3 0 0 -4 0 {close}\n4 4 0 0 -4 0 {close}\n5 0 0 0 0 0 100\n");
    Problem::new(
        Instance::from_text(&text).unwrap(),
        Parameters {
            demand_budget: gq,
            time_budget: gt,
            demand_deviation: 0.5,
            time_deviation: 0.5,
            adjust_demands: false,
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
#[ignore = "requires benchmark download: python3 scripts/download_robust_pd.py --download"]
fn all_downloaded_instances_parse_and_keep_their_pairs() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("instances/RobustPDPTW");
    let mut count = 0;
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|s| s.to_str()) != Some("txt") {
            continue;
        }
        let instance = Instance::read(path).unwrap();
        assert!(instance.requests() >= 5 && instance.requests() <= 75);
        Problem::new(instance, Parameters::default()).unwrap();
        count += 1;
    }
    assert_eq!(count, 60);
}

#[test]
fn parser_rejects_missing_rows_bad_pairs_and_nonfinite_values() {
    for text in [
        "",
        "n 1 Q NaN r_mod 2",
        "n 1 Q 10 r_mod 2\n0 0 0 0 0 0 20",
        "n 1 Q 10 r_mod 2\n0 0 0 0 0 0 20\n1 0 0 0 4 0 20\n2 0 0 0 -3 0 20\n3 0 0 0 0 0 20",
    ] {
        assert!(Instance::from_text(text).is_err());
    }
}

#[test]
fn paper_adjustment_precedes_ceiling_deviations_even_in_nominal_case() {
    let text =
        "n 1 Q 15 r_mod 2\n0 0 0 0 0 0 20\n1 0 0 0 14 0 20\n2 0 0 0 -14 0 20\n3 0 0 0 0 0 20";
    let instance = Instance::from_text(text).unwrap();
    let p = Problem::new(
        instance.clone(),
        Parameters {
            demand_deviation: 0.3,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(p.demands(), [10.]);
    assert_eq!(p.deviations(), [3.]);
    let p = Problem::new(
        instance,
        Parameters {
            adjust_demands: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(p.demands(), [14.]);
}

#[test]
fn distance_rounding_is_explicit_and_full_precision_is_available() {
    let instance = Instance::from_text(
        "n 1 Q 10 r_mod 2\n0 0 0 0 0 0 100\n1 1 1 0 1 0 100\n2 2 2 0 -1 0 100\n3 0 0 0 0 0 100",
    )
    .unwrap();
    let rounded = Problem::new(instance.clone(), Parameters::default()).unwrap();
    assert_eq!(rounded.travel[0][1], 1.41);
    let full = Problem::new(
        instance,
        Parameters {
            distance_rounding: DistanceRounding::FullPrecision,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(full.travel[0][1], 2f64.sqrt());
}

#[test]
fn independent_certificate_accepts_valid_routes_rejects_repeats_and_respects_limits() {
    use std::time::Instant;
    let p = fixture(10., 100., 1, 1);
    let graph = p.pricing_graph(&[0., 0.], 0.).unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let arc_ids = |route: &[usize]| {
        route
            .windows(2)
            .map(|w| {
                graph
                    .arcs
                    .iter()
                    .position(|a| a.tail == w[0] && a.head == w[1])
                    .unwrap()
            })
            .collect::<Vec<_>>()
    };
    let valid = arc_ids(&[0, 1, 3, 2, 4, 5]);
    let mut work = 10_000;
    assert!(
        kernel
            .certify_with_budget(&valid, &mut work, None)
            .unwrap()
            .0
    );
    assert!(work < 10_000);
    let repeated = arc_ids(&[0, 1, 3, 2, 4, 1, 3, 5]);
    assert!(
        !kernel
            .certify_with_budget(&repeated, &mut 10_000, None)
            .unwrap()
            .0
    );
    assert!(kernel.certify_with_budget(&valid, &mut 0, None).is_none());
    assert!(kernel
        .certify_with_budget(&valid, &mut 10_000, Some(Instant::now()))
        .is_none());
}

#[test]
fn capacity_uses_only_onboard_requests_and_releases_the_same_deviation() {
    let p = fixture(6., 100., 1, 0);
    let s = p.replay(&[0, 1, 3, 2, 4, 5]).unwrap();
    assert_eq!(s.loads, [0., 6., 0., 6., 0., 0.]);
    assert!(p.replay(&[0, 1, 2, 3, 4, 5]).is_err());
    for route in [
        &[0, 3, 1, 5][..],
        &[0, 1, 5],
        &[0, 1, 3, 1, 3, 5],
        &[0, 1, 3, 5, 2, 4, 5],
    ] {
        assert!(p.replay(route).is_err(), "{route:?}");
    }
}

#[test]
fn delivery_replaces_the_largest_onboard_deviation_with_the_next_largest() {
    let mut p = fixture(20., 100., 1, 0);
    p.deviations = vec![1., 3.];
    assert_eq!(
        p.replay(&[0, 1, 2, 4, 3, 5]).unwrap().loads,
        [0., 5., 11., 5., 0., 0.]
    );
}

#[test]
fn waiting_absorbs_delays_service_is_counted_and_return_window_is_robust() {
    let mut p = fixture(20., 20., 0, 2);
    p.instance.nodes[1].open = 10.;
    p.instance.nodes[1].close = 10.;
    p.instance.nodes[1].service = 3.;
    p.instance.nodes[5].close = 21.;
    let schedule = p.replay(&[0, 1, 3, 5]).unwrap();
    assert_eq!(schedule.starts[1], [10., 10., 10.]);
    assert_eq!(schedule.starts[2], [15., 16., 16.]);
    assert_eq!(schedule.starts[3], [18., 20., 21.]);
    p.instance.nodes[5].close = 20.;
    assert!(p.replay(&[0, 1, 3, 5]).is_err());
}

/// Exhaustively evaluate uncertainty-set extreme points, independently of the DP.
fn scenarios(p: &Problem, route: &[usize]) -> bool {
    let n = p.instance.requests;
    for qmask in 0usize..1 << n {
        if qmask.count_ones() as usize > p.parameters.demand_budget {
            continue;
        }
        for tmask in 0usize..1 << (route.len() - 1) {
            if tmask.count_ones() as usize > p.parameters.time_budget {
                continue;
            }
            let mut picked = vec![false; n];
            let mut delivered = vec![false; n];
            let (mut load, mut time) = (0., p.instance.nodes[0].open);
            for (k, edge) in route.windows(2).enumerate() {
                let (u, v) = (edge[0], edge[1]);
                let travel = p.travel[u][v];
                time += p.instance.nodes[u].service + travel;
                if tmask & (1 << k) != 0 {
                    time += (p.parameters.time_deviation * travel).ceil();
                }
                time = time.max(p.instance.nodes[v].open);
                if time > p.instance.nodes[v].close {
                    return false;
                }
                if v <= n {
                    let r = v - 1;
                    if picked[r] {
                        return false;
                    }
                    picked[r] = true;
                    load += p.demands[r]
                        + if qmask & (1 << r) != 0 {
                            p.deviations[r]
                        } else {
                            0.
                        };
                } else if v <= 2 * n {
                    let r = v - n - 1;
                    if !picked[r] || delivered[r] {
                        return false;
                    }
                    delivered[r] = true;
                    load -= p.demands[r]
                        + if qmask & (1 << r) != 0 {
                            p.deviations[r]
                        } else {
                            0.
                        };
                } else if picked != delivered {
                    return false;
                }
                if load > p.instance.capacity || load < 0. {
                    return false;
                }
            }
        }
    }
    true
}

fn routes() -> Vec<Vec<usize>> {
    fn visit(prefix: Vec<usize>, remaining: Vec<usize>, out: &mut Vec<Vec<usize>>) {
        let mut route = prefix.clone();
        route.push(5);
        if prefix.len() > 1 {
            out.push(route);
        }
        for (i, &v) in remaining.iter().enumerate() {
            let mut next = prefix.clone();
            next.push(v);
            let mut rest = remaining.clone();
            rest.remove(i);
            visit(next, rest, out);
        }
    }
    let mut out = vec![];
    visit(vec![0], vec![1, 2, 3, 4], &mut out);
    out
}

#[test]
fn replay_and_every_native_join_match_exhaustive_scenarios() {
    let mut counts = [0, 0];
    for (gq, gt) in [(0, 0), (1, 0), (0, 1), (1, 2), (2, 2), (99, 99)] {
        for capacity in [6., 10., 12.] {
            let p = fixture(capacity, 6., gq, gt);
            let graph = p.pricing_graph(&[0., 0.], 0.).unwrap();
            let kernel = graph.coupled_kernel().unwrap();
            for route in routes() {
                let feasible = scenarios(&p, &route);
                counts[usize::from(feasible)] += 1;
                assert_eq!(p.replay(&route).is_ok(), feasible, "{route:?} ({gq},{gt})");
                let Some(arcs): Option<Vec<usize>> = route
                    .windows(2)
                    .map(|w| {
                        graph
                            .arcs
                            .iter()
                            .position(|a| a.tail == w[0] && a.head == w[1])
                    })
                    .collect()
                else {
                    assert!(!feasible);
                    continue;
                };
                for split in 0..=arcs.len() {
                    let mut f = Some(kernel.root(true));
                    for &a in &arcs[..split] {
                        f = f.and_then(|s| kernel.extend(&s, a, true).unwrap().map(|x| x.0));
                    }
                    let mut b = Some(kernel.root(false));
                    for &a in arcs[split..].iter().rev() {
                        b = b.and_then(|s| kernel.extend(&s, a, false).unwrap().map(|x| x.0));
                    }
                    let joined = match (f, b) {
                        (Some(f), Some(b)) => kernel.join(&f, &b).unwrap().is_some(),
                        _ => false,
                    };
                    assert_eq!(
                        joined, feasible,
                        "{route:?} split={split} ({gq},{gt}) cap={capacity}"
                    );
                }
            }
        }
    }
    assert!(counts[0] > 100 && counts[1] > 10, "{counts:?}");
}

#[test]
fn native_pricing_matches_exhaustive_routes_in_both_directions() {
    use crate::labeling::{
        engine::{EngineControls, LabelingEngine},
        Controls,
    };
    use crate::pricing::phases::Phase;
    for (gq, gt) in [(0, 0), (1, 1), (2, 2)] {
        let p = fixture(10., 7., gq, gt);
        let duals = [20_000., 20_000.];
        let best = routes()
            .iter()
            .filter(|r| scenarios(&p, r))
            .map(|r| {
                p.replay(r).unwrap().cost
                    - r.iter().filter(|v| (1..=2).contains(*v)).count() as f64 * 20_000.
            })
            .fold(f64::INFINITY, f64::min);
        for bidirectional in [false, true] {
            let graph = p.pricing_graph(&duals, 0.).unwrap();
            let mut engine = LabelingEngine::prepare(
                &graph,
                EngineControls {
                    bidirectional,
                    ..Default::default()
                },
                Phase {
                    check_ng: true,
                    jump_arcs: true,
                    labels_per_bucket: i64::MAX,
                },
            )
            .unwrap();
            let result = engine
                .solve(
                    &graph,
                    &Controls {
                        max_columns: 1000,
                        ..Default::default()
                    },
                )
                .unwrap();
            assert!(
                (result.minimum_reduced_cost - best).abs() < 1e-6,
                "({gq},{gt}) bidir={bidirectional}: {} vs {best}",
                result.minimum_reduced_cost
            );
            for route in result.columns {
                assert!(p
                    .replay(&route.nodes.iter().map(|v| *v as usize).collect::<Vec<_>>())
                    .is_ok());
            }
        }
    }
}

#[cfg(feature = "clp")]
#[test]
fn native_master_tree_covers_every_request_and_matches_exact_partition() {
    use crate::master::tree::ModelTreeOptions;
    use crate::master::tree::TreeStatus;
    for (gq, gt) in [(0, 0), (1, 1), (2, 2)] {
        let p = fixture(10., 7., gq, gt);
        let all = routes();
        let cheapest = |mask: usize| {
            all.iter()
                .filter(|r| {
                    scenarios(&p, r)
                        && r.iter()
                            .filter(|v| (1..=2).contains(*v))
                            .fold(0usize, |m, v| m | (1 << (v - 1)))
                            == mask
                })
                .map(|r| p.replay(r).unwrap().cost)
                .fold(f64::INFINITY, f64::min)
        };
        let expected = cheapest(3).min(cheapest(1) + cheapest(2));
        let mut options = ModelTreeOptions::default();
        options.node.cg.artificial_cost = 1e6;
        let result = crate::master::tree::solve(p.master_problem().unwrap(), options).unwrap();
        assert_eq!(result.status, TreeStatus::Optimal, "{result:?}");
        assert!((result.incumbent - expected).abs() < 1e-6, "{result:?}");
        let mut covered = [0.; 2];
        for fixed in &result.routes {
            let route: Vec<_> = fixed
                .route
                .route
                .nodes
                .iter()
                .map(|v| *v as usize)
                .collect();
            p.replay(&route).unwrap();
            for v in route {
                if (1..=2).contains(&v) {
                    covered[v - 1] += fixed.quantity;
                }
            }
        }
        assert_eq!(covered, [1., 1.]);
    }
}

/// Random paired instance with tight windows, so unreachable marking fires.
fn random_problem(n: usize, seed: u64, gq: usize, gt: usize, ng_size: usize) -> Problem {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut next = |m: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) % m
    };
    let mut text = format!("n {n} Q 10 r_mod 2\n0 0 0 0 0 0 200\n");
    let mut rows = vec![];
    for i in 1..=n {
        let demand = 1 + next(5);
        let open = next(40);
        rows.push((
            next(30),
            next(30),
            demand as i64,
            open,
            open + 15 + next(40),
        ));
        let open = 10 + next(60);
        rows.push((
            next(30),
            next(30),
            -(demand as i64),
            open,
            open + 15 + next(50),
        ));
        let _ = i;
    }
    for (k, r) in rows
        .iter()
        .step_by(2)
        .chain(rows.iter().skip(1).step_by(2))
        .enumerate()
    {
        text += &format!("{} {} {} 1 {} {} {}\n", k + 1, r.0, r.1, r.2, r.3, r.4);
    }
    text += &format!("{} 0 0 0 0 0 200\n", 2 * n + 1);
    Problem::new(
        Instance::from_text(&text).unwrap(),
        Parameters {
            demand_budget: gq,
            time_budget: gt,
            demand_deviation: 0.5,
            time_deviation: 0.3,
            vehicle_cost: 0.,
            adjust_demands: false,
            ng_size,
            ..Default::default()
        },
    )
    .unwrap()
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

fn reduced_cost(p: &Problem, route: &[usize], duals: &[f64]) -> f64 {
    let n = p.instance.requests;
    p.replay(route).unwrap().cost
        - route
            .iter()
            .filter(|v| (1..=n).contains(*v))
            .map(|v| duals[v - 1])
            .sum::<f64>()
}

fn price(p: &Problem, duals: &[f64], bidirectional: bool) -> crate::labeling::PricingResult {
    use crate::labeling::{
        engine::{EngineControls, LabelingEngine},
        Controls,
    };
    use crate::pricing::phases::Phase;
    let graph = p.pricing_graph(duals, 0.).unwrap();
    LabelingEngine::prepare(
        &graph,
        EngineControls {
            bidirectional,
            ..Default::default()
        },
        Phase {
            check_ng: true,
            jump_arcs: true,
            labels_per_bucket: i64::MAX,
        },
    )
    .unwrap()
    .solve(
        &graph,
        &Controls {
            max_columns: 1000,
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn subset_dominance_and_unreachable_marking_keep_pricing_exact() {
    let all = paired_routes(4);
    let mut checked = 0;
    for seed in 0..12 {
        for (gq, gt) in [(0, 0), (1, 1), (2, 3)] {
            let p = random_problem(4, seed, gq, gt, 4);
            let duals: Vec<f64> = (0..4)
                .map(|i| 40. + 9. * ((seed as usize + i) % 5) as f64)
                .collect();
            let best = all
                .iter()
                .filter(|r| p.replay(r).is_ok())
                .map(|r| reduced_cost(&p, r, &duals))
                .fold(f64::INFINITY, f64::min);
            if !best.is_finite() {
                continue;
            }
            for bidirectional in [false, true] {
                let result = price(&p, &duals, bidirectional);
                assert!(
                    (result.minimum_reduced_cost - best).abs() < 1e-6,
                    "seed={seed} ({gq},{gt}) bidir={bidirectional}: {} vs {best}",
                    result.minimum_reduced_cost
                );
                for column in &result.columns {
                    let route: Vec<usize> = column.nodes.iter().map(|v| *v as usize).collect();
                    assert!(p.replay(&route).is_ok(), "{route:?}");
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 40, "{checked}");
}

#[test]
fn ng_pricing_is_a_relaxation_of_elementary_pricing() {
    let all = paired_routes(4);
    for seed in 0..12 {
        for (gq, gt) in [(0, 0), (1, 1), (2, 3)] {
            let duals: Vec<f64> = (0..4)
                .map(|i| 40. + 9. * ((seed as usize + i) % 5) as f64)
                .collect();
            let exact = random_problem(4, seed, gq, gt, 4);
            let best = all
                .iter()
                .filter(|r| exact.replay(r).is_ok())
                .map(|r| reduced_cost(&exact, r, &duals))
                .fold(f64::INFINITY, f64::min);
            for ng in [1, 2] {
                let p = random_problem(4, seed, gq, gt, ng);
                for bidirectional in [false, true] {
                    let result = price(&p, &duals, bidirectional);
                    assert!(
                        result.minimum_reduced_cost <= best + 1e-6,
                        "seed={seed} ng={ng}: {} > {best}",
                        result.minimum_reduced_cost
                    );
                    for column in &result.columns {
                        let route: Vec<usize> = column.nodes.iter().map(|v| *v as usize).collect();
                        let mut pickups = route
                            .iter()
                            .filter(|v| (1..=4).contains(*v))
                            .collect::<Vec<_>>();
                        pickups.sort();
                        pickups.dedup();
                        if pickups.len() * 2 + 2 == route.len() {
                            assert!(p.replay(&route).is_ok(), "{route:?}");
                        }
                    }
                }
            }
        }
    }
}

#[cfg(feature = "clp")]
#[test]
fn ng_master_tree_matches_exact_partition_on_random_instances() {
    use crate::master::tree::{ModelTreeOptions, TreeStatus};
    // Fractional points with integral arc activity arise only from NG walks;
    // fallback branching must keep them in the domain instead of stalling.
    for n in [3, 4] {
        let all = paired_routes(n);
        let full = (1 << n) - 1;
        for seed in 0..10 {
            for (gq, gt) in [(0, 0), (1, 1), (2, 2)] {
                let mut p = random_problem(n, seed, gq, gt, 1);
                p.parameters.vehicle_cost = 100.;
                let mut cheapest = vec![f64::INFINITY; full + 1];
                for r in &all {
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
                let mut options = ModelTreeOptions::default();
                options.node.cg.artificial_cost = 1e6;
                let result =
                    crate::master::tree::solve(p.master_problem().unwrap(), options).unwrap();
                if !best[full].is_finite() {
                    assert_ne!(result.status, TreeStatus::Optimal, "n={n} seed={seed}");
                    continue;
                }
                assert_eq!(
                    result.status,
                    TreeStatus::Optimal,
                    "n={n} seed={seed} ({gq},{gt}) {result:?}"
                );
                assert!(
                    (result.incumbent - best[full]).abs() < 1e-6,
                    "n={n} seed={seed} ({gq},{gt}): {} vs {}",
                    result.incumbent,
                    best[full]
                );
                for fixed in &result.routes {
                    let route: Vec<_> = fixed
                        .route
                        .route
                        .nodes
                        .iter()
                        .map(|v| *v as usize)
                        .collect();
                    p.replay(&route).unwrap();
                }
            }
        }
    }
}

/// Fallback branching on these instances meets NG walks; they previously
/// stalled because the route domain and certificate rejected the walks.
#[cfg(feature = "clp")]
#[test]
#[ignore = "requires benchmark download: python3 scripts/download_robust_pd.py --download"]
fn ng_benchmark_trees_close_after_fallback_branching() {
    use crate::master::tree::{ModelTreeOptions, TreeStatus};
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("instances/RobustPDPTW");
    for (name, budget, sigma, expected) in [
        ("CC15.txt", 5, 0.3, 20589.98),
        ("DD15.txt", 1, 0.1, 20683.07),
    ] {
        let p = Problem::new(
            Instance::read(root.join(name)).unwrap(),
            Parameters {
                demand_budget: budget,
                time_budget: budget,
                demand_deviation: sigma,
                time_deviation: sigma,
                ..Default::default()
            },
        )
        .unwrap();
        let mut options = ModelTreeOptions::default();
        options.node.cg.artificial_cost = (p.parameters.vehicle_cost + 100_000.) * 15.;
        let result = crate::master::tree::solve(p.master_problem().unwrap(), options).unwrap();
        assert_eq!(result.status, TreeStatus::Optimal, "{name}: {result:?}");
        assert!(
            (result.incumbent - expected).abs() < 1e-6,
            "{name}: {}",
            result.incumbent
        );
    }
}
