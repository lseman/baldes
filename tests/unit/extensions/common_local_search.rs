use super::*;
use crate::inputs::{Customer, Rounding};

fn instance() -> Instance {
    Instance {
        name: "route-replay".into(),
        vehicles: 2,
        capacity: 4.,
        customers: (0..6)
            .map(|i| Customer {
                x: i as f64,
                y: 0.,
                demand: if i == 0 { 0. } else { 1. },
                ready: if i == 2 { 8. } else { 0. },
                due: if i == 3 { 18. } else { 100. },
                service: 0.5,
            })
            .collect(),
        rounding: Rounding::None,
        distances: Some(
            (0..6)
                .map(|i| (0..6).map(|j| ((i * 7 + j * 3) % 9 + 1) as f64).collect())
                .collect(),
        ),
    }
}

#[test]
fn cached_suffix_replay_matches_independent_directed_route_replay() {
    let instance = instance();
    // Every distinct route of length <= 5 includes changed prefixes, reversals,
    // waiting, capacity violations, and customer/depot time-window violations.
    let old = vec![1, 2, 4];
    let cache = Replay::new(&instance, &old);
    fn visit(instance: &Instance, old: &[usize], cache: &Replay, route: &mut Vec<usize>) {
        let result = cache.trial(instance, old, route);
        if route.is_empty() {
            assert_eq!(result, Some(0.));
        } else {
            assert_eq!(
                result.is_some(),
                super::super::validate_route(instance, route).is_ok(),
                "{route:?}"
            );
            if let Some(cost) = result {
                let nodes: Vec<_> = std::iter::once(0)
                    .chain(route.iter().copied())
                    .chain(std::iter::once(0))
                    .collect();
                let expected: f64 = nodes
                    .windows(2)
                    .map(|e| instance.distance(e[0], e[1], instance.rounding))
                    .sum();
                assert_eq!(cost, expected, "{route:?}");
            }
        }
        for v in 1..6 {
            if !route.contains(&v) {
                route.push(v);
                visit(instance, old, cache, route);
                route.pop();
            }
        }
    }
    visit(&instance, &old, &cache, &mut vec![]);
}

#[test]
fn cached_replay_rejects_invalid_replacement_arcs_and_depot_return() {
    let mut instance = instance();
    let old = vec![1];
    let cache = Replay::new(&instance, &old);
    for bad in [f64::NAN, f64::INFINITY, -1.] {
        instance.distances.as_mut().unwrap()[1][2] = bad;
        assert!(cache.trial(&instance, &old, &[1, 2]).is_none());
        instance.distances.as_mut().unwrap()[1][2] = 1.;
        instance.distances.as_mut().unwrap()[2][0] = bad;
        assert!(cache.trial(&instance, &old, &[1, 2]).is_none());
        instance.distances.as_mut().unwrap()[2][0] = 1.;
    }
}

#[test]
fn relocating_to_empty_slot_respects_fleet_limit() {
    let mut instance = instance();
    instance.vehicles = 1;
    instance.customers.iter_mut().for_each(|c| {
        c.ready = 0.;
        c.due = 1000.;
    });
    instance.distances = Some(vec![vec![1.; 6]; 6]);
    instance.distances.as_mut().unwrap()[1][2] = 50.;
    let routes = vec![vec![1, 2], vec![]];
    let replay = routes
        .iter()
        .map(|r| Replay::new(&instance, r))
        .collect::<Vec<_>>();
    assert!(improving(
        &instance,
        &routes,
        &replay,
        Move {
            a: 0,
            b: 1,
            first: vec![1],
            second: vec![2]
        },
        1
    )
    .is_none());
    instance.vehicles = 2;
    assert!(improving(
        &instance,
        &routes,
        &replay,
        Move {
            a: 0,
            b: 1,
            first: vec![1],
            second: vec![2]
        },
        1
    )
    .is_some());
}

#[test]
fn granular_membership_is_symmetric_and_keeps_depot_boundaries() {
    let mut instance = instance();
    instance.customers = (0..80)
        .map(|i| Customer {
            x: i as f64,
            y: 0.,
            demand: 1.,
            ready: 0.,
            due: 1000.,
            service: 0.,
        })
        .collect();
    instance.distances = None;
    let neighbours = Neighbours::new(
        &instance,
        Instant::now() + std::time::Duration::from_secs(1),
    )
    .unwrap();
    assert!(!neighbours.contains(1, 79));
    assert!(neighbours.contains(1, 2));
    assert!(neighbours.insertion(1, &[79], 0));
    assert!(neighbours.insertion(1, &[79], 1));
    for u in 0..80 {
        for v in 0..80 {
            assert_eq!(neighbours.contains(u, v), neighbours.contains(v, u));
        }
    }
}

#[test]
fn versioned_pair_scans_revisit_routes_until_relocation_and_swap_optimal() {
    for seed in 0..12 {
        let mut instance = instance();
        instance.customers.iter_mut().for_each(|c| {
            c.ready = 0.;
            c.due = 1000.;
        });
        instance.distances = Some(
            (0..6)
                .map(|i| {
                    (0..6)
                        .map(|j| ((i * 17 + j * 31 + i * j * (seed + 1)) % 43 + 1) as f64)
                        .collect()
                })
                .collect(),
        );
        let mut routes = vec![vec![1, 2, 3], vec![4, 5], vec![]];
        improve(
            &instance,
            &mut routes,
            Instant::now() + std::time::Duration::from_secs(2),
            &[true, true, true],
        );
        let cost = validate_plan(&instance, &routes).unwrap();
        let check = |trial: Vec<Vec<usize>>| {
            if let Ok(next) = super::super::plan_cost(&instance, &trial, true) {
                assert!(
                    next >= cost - 1e-8,
                    "seed {seed}: missed {routes:?} -> {trial:?}"
                );
            }
        };
        for a in 0..routes.len() {
            for b in 0..routes.len() {
                for i in 0..routes[a].len() {
                    for pos in 0..=routes[b].len() {
                        if a == b && (pos == i || pos == i + 1) {
                            continue;
                        }
                        let mut trial = routes.clone();
                        let u = trial[a].remove(i);
                        trial[b].insert(if a == b && pos > i { pos - 1 } else { pos }, u);
                        check(trial);
                    }
                    for j in 0..routes[b].len() {
                        let mut trial = routes.clone();
                        trial[a][i] = routes[b][j];
                        trial[b][j] = routes[a][i];
                        check(trial);
                    }
                }
            }
        }
    }
}

#[test]
fn reused_neighbours_match_fresh_search_across_different_route_plans() {
    let mut instance = instance();
    instance.customers.iter_mut().for_each(|c| {
        c.ready = 0.;
        c.due = 1000.;
    });
    let deadline = Instant::now() + std::time::Duration::from_secs(5);
    let search = LocalSearch::new(&instance, deadline).unwrap();
    // Only neighbour data is shared: route counts, loads, and modification
    // markers must come from each new plan, including after a no-op call.
    for (plan, changed) in [
        (vec![vec![1, 2, 3], vec![4, 5]], vec![true, true]),
        (vec![vec![5, 3], vec![2, 4, 1], vec![]], vec![false; 3]),
        (vec![vec![5, 3], vec![2, 4, 1], vec![]], vec![true; 3]),
        (
            vec![vec![], vec![1, 4], vec![2, 5, 3]],
            vec![false, true, false],
        ),
    ] {
        let mut reused = plan.clone();
        let mut fresh = plan;
        search.improve(&mut reused, deadline, &changed);
        improve(&instance, &mut fresh, deadline, &changed);
        assert_eq!(reused, fresh);
        assert!(validate_plan(&instance, &reused).is_ok());
    }
}

#[test]
fn reusable_search_observes_expired_deadlines_and_rejects_invalid_instances() {
    let instance = instance();
    assert!(LocalSearch::new(&instance, Instant::now()).is_none());
    let deadline = Instant::now() + std::time::Duration::from_secs(1);
    let search = LocalSearch::new(&instance, deadline).unwrap();
    let original = vec![vec![1, 2, 4], vec![3, 5]];
    let mut routes = original.clone();
    search.improve(&mut routes, Instant::now(), &[true, true]);
    assert_eq!(routes, original);
    let mut invalid = instance.clone();
    invalid.distances = Some(vec![vec![0.]]);
    assert!(LocalSearch::new(&invalid, deadline).is_none());
}
