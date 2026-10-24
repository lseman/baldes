use super::tests::{enumerate, random, Lcg};
use super::*;
use crate::labeling::engine::{EngineControls, LabelingEngine};
use crate::labeling::Controls;
use crate::pricing::phases::Phase;

fn duals(p: &Problem, seed: u64) -> Vec<f64> {
    let mut rng = Lcg(seed);
    (0..p.nodes.len())
        .map(|i| {
            if i == p.depot {
                0.
            } else {
                (80. * rng.next()).floor()
            }
        })
        .collect()
}

fn reduced(route: &[usize], cost: f64, duals: &[f64], fleet: f64) -> f64 {
    route.iter().fold(cost - fleet, |c, &v| c - duals[v])
}

fn engine_minimum(
    p: &Problem,
    duals: &[f64],
    fleet: f64,
    bidirectional: bool,
) -> (f64, Vec<Vec<usize>>) {
    let graph = p.native_graph(duals, fleet, p.nodes.len()).unwrap();
    let phase = Phase {
        check_ng: true,
        jump_arcs: true,
        labels_per_bucket: i64::MAX,
    };
    let mut engine = LabelingEngine::prepare(
        &graph,
        EngineControls {
            bidirectional,
            ..Default::default()
        },
        phase,
    )
    .unwrap();
    let controls = Controls {
        max_columns: 1000,
        ..Default::default()
    };
    let result = engine.solve(&graph, &controls).unwrap();
    let routes = result
        .columns
        .iter()
        .map(|c| p.native_route(&c.nodes))
        .collect();
    (result.minimum_reduced_cost, routes)
}

#[test]
fn bucket_engine_matches_robust_enumeration_in_both_directions() {
    let mut binding = 0;
    for seed in 1..13 {
        for (gq, gt) in [(0, 0), (1, 0), (0, 1), (1, 2), (2, 2), (3, 1), (8, 8)] {
            let p = random(seed, 7, gq, gt, 0.5);
            let duals = duals(&p, seed * 101 + (gq * 7 + gt) as u64);
            let fleet = 3.;
            let best = enumerate(&p)
                .iter()
                .map(|(r, c)| reduced(r, *c, &duals, fleet))
                .fold(f64::INFINITY, f64::min);
            let mut nominal = p.clone();
            (nominal.demand_budget, nominal.time_budget) = (0, 0);
            let nominal_best = enumerate(&nominal)
                .iter()
                .map(|(r, c)| reduced(r, *c, &duals, fleet))
                .fold(f64::INFINITY, f64::min);
            binding += usize::from(best > nominal_best + 1e-9);
            for bidirectional in [false, true] {
                let (found, routes) = engine_minimum(&p, &duals, fleet, bidirectional);
                for route in &routes {
                    let schedule = p.replay(route).unwrap_or_else(|e| {
                        panic!("seed {seed} ({gq},{gt}) bidirectional {bidirectional}: {route:?} {e:?}")
                    });
                    assert!(reduced(route, schedule.cost, &duals, fleet) >= best - 1e-9);
                }
                if best < -1e-9 {
                    assert!(
                        (found - best).abs() < 1e-6,
                        "seed {seed} ({gq},{gt}) bidirectional {bidirectional}: {found} vs {best}"
                    );
                } else {
                    assert!(found >= -1e-9 || (found - best).abs() < 1e-6);
                }
            }
        }
    }
    // Robustness must change the optimum often enough to be tested.
    assert!(binding >= 20, "only {binding} binding cases");
}

#[test]
fn every_forward_backward_split_joins_exactly_when_the_route_is_robust() {
    let mut rng = Lcg(17);
    let mut counts = [0usize; 2];
    for seed in 1..9 {
        for (gq, gt) in [(1, 1), (2, 3), (0, 2), (4, 0)] {
            let mut p = random(seed, 6, gq, gt, 0.5);
            // Tighten capacity so the load vector also binds.
            p.capacity = 14.;
            let graph = p.native_graph(&vec![0.; p.nodes.len()], 0., 7).unwrap();
            let kernel = graph.coupled_kernel().unwrap();
            let sink = p.nodes.len();
            let arc_of = |u: usize, v: usize| {
                let v = if v == p.depot { sink } else { v };
                graph.arcs.iter().position(|a| a.tail == u && a.head == v)
            };
            for _ in 0..60 {
                let k = 1 + (5. * rng.next()) as usize;
                let mut customers: Vec<usize> = (1..p.nodes.len()).collect();
                for i in (1..customers.len()).rev() {
                    customers.swap(i, (rng.next() * (i + 1) as f64) as usize);
                }
                let mut route = vec![p.depot];
                route.extend(&customers[..k]);
                route.push(p.depot);
                let feasible = p.replay(&route).is_ok();
                let mut nominal = p.clone();
                (nominal.demand_budget, nominal.time_budget) = (0, 0);
                if nominal.replay(&route).is_ok() {
                    counts[usize::from(feasible)] += 1;
                }
                let Some(arcs) = route
                    .windows(2)
                    .map(|w| arc_of(w[0], w[1]))
                    .collect::<Option<Vec<_>>>()
                else {
                    assert!(!feasible, "a robust route lost an arc: {route:?}");
                    continue;
                };
                for split in 0..=arcs.len() {
                    let mut forward = Some(kernel.root(true));
                    for &arc in &arcs[..split] {
                        forward = forward
                            .and_then(|s| kernel.extend(&s, arc, true).unwrap().map(|x| x.0));
                    }
                    let mut backward = Some(kernel.root(false));
                    for &arc in arcs[split..].iter().rev() {
                        backward = backward
                            .and_then(|s| kernel.extend(&s, arc, false).unwrap().map(|x| x.0));
                    }
                    let joined = match (forward, backward) {
                        (Some(f), Some(b)) => kernel.join(&f, &b).unwrap().is_some(),
                        _ => false,
                    };
                    assert_eq!(
                        joined, feasible,
                        "seed {seed} ({gq},{gt}) route {route:?} split {split}"
                    );
                }
            }
        }
    }
    // Nominally feasible routes, both robust infeasible and robust feasible.
    assert!(counts[0] >= 30 && counts[1] >= 30, "{counts:?}");
}

#[test]
fn backward_labels_completed_at_the_source_respect_the_worst_departure() {
    // Nominal arrival 2 meets the window [0, 3]; one deviation (+3) does not.
    let node = |window| Node {
        demand: 0.,
        demand_deviation: 0.,
        window,
        service: 0.,
    };
    let arc = Some(Travel {
        time: 2.,
        time_deviation: 3.,
        cost: 1.,
    });
    let mut p = Problem {
        nodes: vec![node((0., 50.)), node((0., 3.))],
        travel: vec![vec![None, arc], vec![arc, None]],
        depot: 0,
        capacity: 1.,
        demand_budget: 0,
        time_budget: 0,
    };
    for (budget, feasible) in [(0, true), (1, false)] {
        p.time_budget = budget;
        assert_eq!(p.replay(&[0, 1, 0]).is_ok(), feasible);
        // Build the graph nominally so the arc filter keeps the source arc.
        let mut nominal = p.clone();
        nominal.time_budget = 0;
        let mut graph = nominal.native_graph(&[0., 0.], 0., 1).unwrap();
        let arcs = [(0, 1), (1, 2)];
        graph
            .set_coupled_resources(&p.budgeted_resources(&arcs, 2).unwrap())
            .unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        let id = |t, h| {
            graph
                .arcs
                .iter()
                .position(|a| a.tail == t && a.head == h)
                .unwrap()
        };
        let (last, _) = kernel
            .extend(&kernel.root(false), id(1, 2), false)
            .unwrap()
            .unwrap();
        let first = kernel.extend(&last, id(0, 1), false).unwrap();
        assert_eq!(first.is_some(), feasible, "budget {budget}");
    }
}

#[cfg(feature = "clp")]
#[test]
fn directed_certificates_agree_with_robust_replay() {
    let mut rng = Lcg(5);
    let mut counts = [0usize; 2];
    for seed in 1..7 {
        let mut p = random(seed, 6, 2, 2, 0.5);
        p.capacity = 14.;
        let graph = p.native_graph(&vec![0.; p.nodes.len()], 0., 7).unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        let sink = p.nodes.len();
        for _ in 0..80 {
            let k = 1 + (4. * rng.next()) as usize;
            let mut route = vec![p.depot];
            let mut customers: Vec<usize> = (1..p.nodes.len()).collect();
            for i in (1..customers.len()).rev() {
                customers.swap(i, (rng.next() * (i + 1) as f64) as usize);
            }
            route.extend(&customers[..k]);
            route.push(p.depot);
            let Some(arcs) = route
                .windows(2)
                .map(|w| {
                    let head = if w[1] == p.depot { sink } else { w[1] };
                    graph
                        .arcs
                        .iter()
                        .position(|a| a.tail == w[0] && a.head == head)
                })
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            let feasible = p.replay(&route).is_ok();
            // Integer data: every interval is a point, so the answer is known.
            let (valid, _) = kernel.certify(&arcs).unwrap();
            assert_eq!(valid, feasible, "{route:?}");
            counts[usize::from(feasible)] += 1;
        }
    }
    assert!(counts[0] > 20 && counts[1] > 20, "{counts:?}");
}

#[test]
fn fixing_and_enumeration_keep_every_robust_route_under_the_cutoff() {
    for (seed, ng, slack) in (1..13).flat_map(|s| [(s, 3, 0.5), (s, 9, 2.), (s, 4, 25.)]) {
        for (gq, gt) in [(0, 0), (0, 1), (1, 1), (2, 3), (3, 0)] {
            let p = random(seed, 9, gq, gt, 0.5);
            let duals = duals(&p, seed * 13 + (gq * 5 + gt) as u64);
            let fleet = 3.;
            let mut all: Vec<_> = enumerate(&p)
                .into_iter()
                .map(|(r, c)| (reduced(&r, c, &duals, fleet), r))
                .collect();
            all.sort_by(|a, b| a.0.total_cmp(&b.0));
            let Some(best) = all.first().map(|r| r.0) else {
                continue;
            };
            let cutoff = best + slack;
            let mut expected: Vec<_> = all
                .iter()
                .filter(|(rc, _)| *rc <= cutoff - 1e-6)
                .map(|(_, r)| r.clone())
                .collect();
            expected.sort();
            let mut solver = crate::pricing::Solver::<2>::new(2, 0).unwrap();
            solver.graph = p.native_graph(&duals, fleet, ng).unwrap();
            solver.price_phase(0, 10, false).unwrap();
            solver.fix(cutoff).unwrap();
            let mut found: Vec<_> = solver
                .enumerate(cutoff)
                .unwrap()
                .iter()
                .map(|r| p.native_route(&r.nodes))
                .filter(|r| reduced(r, p.replay(r).unwrap().cost, &duals, fleet) <= cutoff - 1e-6)
                .collect();
            found.sort();
            let missing: Vec<_> = expected.iter().filter(|r| !found.contains(r)).collect();
            assert!(
                missing.is_empty(),
                "seed {seed} ng {ng} slack {slack} ({gq},{gt}): {} of {} missing, e.g. {:?}",
                missing.len(),
                expected.len(),
                &missing[..missing.len().min(3)]
            );
        }
    }
}

#[test]
fn bucket_directions_match_reference_on_asymmetric_reordered_depots() {
    use super::reference::{price, PricingOptions, PricingStatus};
    for seed in 1..=96 {
        let mut rng = Lcg(seed * 43);
        let (gq, gt) = ((seed % 5) as usize, ((seed / 5) % 5) as usize);
        let mut p = random(seed, 6, gq, gt, 0.5);
        // Directed, sparse arcs exercise more than symmetric Euclidean data.
        for row in &mut p.travel {
            for arc in row {
                if let Some(travel) = arc {
                    if rng.next() < 0.15 {
                        *arc = None;
                    } else {
                        travel.time += (5. * rng.next()).floor();
                        travel.cost += (5. * rng.next()).floor();
                    }
                }
            }
        }
        let depot = 1 + seed as usize % (p.nodes.len() - 1);
        p.nodes.swap(0, depot);
        p.travel.swap(0, depot);
        for row in &mut p.travel {
            row.swap(0, depot);
        }
        p.depot = depot;
        let duals = duals(&p, seed * 101);
        let fleet = 3.;
        let best = enumerate(&p)
            .iter()
            .map(|(r, c)| reduced(r, *c, &duals, fleet))
            .fold(f64::INFINITY, f64::min);
        for dominance in [false, true] {
            let reference = price(
                &p,
                &duals,
                fleet,
                PricingOptions {
                    max_routes: 1,
                    dominance,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(reference.status, PricingStatus::Complete);
            let found = reference
                .routes
                .first()
                .map_or(f64::INFINITY, |r| r.reduced_cost);
            assert!(found == best || (found - best).abs() < 1e-6, "seed {seed}");
        }
        for bidirectional in [false, true] {
            let (found, routes) = engine_minimum(&p, &duals, fleet, bidirectional);
            for route in routes {
                p.replay(&route).unwrap();
            }
            if best < -1e-9 {
                assert!(
                    (found - best).abs() < 1e-6,
                    "seed {seed}, depot {depot}, bidirectional {bidirectional}: {found} vs {best}"
                );
            } else {
                assert!(found >= -1e-9, "seed {seed}: {found} vs {best}");
            }
        }
    }
}
