use super::{improve_modified_routes, nearest_positions};
use crate::inputs::{Customer, Instance, Rounding};

#[test]
fn cached_metrics_preserve_rounding_bits_and_explicit_directed_costs() {
    let mut instance = Instance {
        name: "cached-metric".into(),
        vehicles: 2,
        capacity: 10.,
        customers: [(0., 0.), (0.1, 0.2), (10.123, -3.457)]
            .into_iter()
            .map(|(x, y)| Customer {
                x,
                y,
                demand: 1.,
                ready: 0.,
                due: f64::INFINITY,
                service: 0.,
            })
            .collect(),
        rounding: Rounding::None,
        distances: None,
    };
    for rounding in [
        Rounding::None,
        Rounding::Tenth,
        Rounding::Nearest,
        Rounding::Ceil,
    ] {
        instance.rounding = rounding;
        let cached = instance.cached_distances();
        for i in 0..instance.customers.len() {
            for j in 0..instance.customers.len() {
                assert_eq!(
                    cached.distance(i, j, rounding).to_bits(),
                    instance.distance(i, j, rounding).to_bits()
                );
            }
        }
    }
    instance.distances = Some(vec![
        vec![10., 1., 2.],
        vec![3., 20., 4.],
        vec![5., 6., 30.],
    ]);
    let cached = instance.cached_distances();
    assert!(matches!(cached, std::borrow::Cow::Borrowed(_)));
    assert_eq!(cached.distance(2, 1, Rounding::Ceil), 6.);
    assert_eq!(cached.distance(1, 2, Rounding::None), 4.);
    assert_eq!(cached.distance(1, 1, Rounding::None), 20.);
}

#[test]
fn fast_plan_replay_retains_coverage_capacity_windows_and_fleet_checks() {
    let instance = Instance {
        name: "replay".into(),
        vehicles: 1,
        capacity: 2.,
        rounding: Rounding::None,
        distances: None,
        customers: (0..3)
            .map(|i| Customer {
                x: i as f64,
                y: 0.,
                demand: if i == 0 { 0. } else { 1. },
                ready: 0.,
                due: 10.,
                service: 0.,
            })
            .collect(),
    };
    assert_eq!(
        super::plan_cost(&instance, &[vec![1, 2]], true).unwrap(),
        4.
    );
    for routes in [
        vec![vec![1, 1, 2]],
        vec![vec![1]],
        vec![vec![1], vec![2]],
        vec![vec![9]],
    ] {
        assert!(super::plan_cost(&instance, &routes, true).is_err());
    }
    let mut constrained = instance.clone();
    constrained.capacity = 1.;
    assert!(super::plan_cost(&constrained, &[vec![1, 2]], true).is_err());
    constrained.capacity = 2.;
    constrained.customers[1].due = 0.5;
    assert!(super::plan_cost(&constrained, &[vec![1, 2]], true).is_err());
    let mut invalid = instance.clone();
    invalid.distances = Some(vec![vec![0.]]);
    let original = vec![vec![1, 2]];
    let mut routes = original.clone();
    improve_modified_routes(
        &invalid,
        &mut routes,
        std::time::Instant::now() + std::time::Duration::from_secs(1),
        &[true],
    );
    assert_eq!(routes, original);
}

#[test]
fn nearest_positions_limits_and_orders_neighbors_deterministically() {
    let customers = (0..=45)
        .map(|i| Customer {
            x: i as f64,
            y: 0.,
            demand: 1.,
            ready: 0.,
            due: f64::INFINITY,
            service: 0.,
        })
        .collect();
    let instance = Instance {
        name: "neighbors".into(),
        vehicles: 1,
        capacity: 100.,
        customers,
        rounding: Rounding::None,
        distances: None,
    };
    let route: Vec<_> = (1..=45).rev().collect();
    let selected = nearest_positions(&instance, 0, &route, 40);
    assert_eq!(selected.len(), 40);
    assert_eq!(
        selected.iter().map(|&i| route[i]).collect::<Vec<_>>(),
        (1..=40).collect::<Vec<_>>()
    );
    assert!(nearest_positions(&instance, 0, &route, 0).is_empty());
}

#[test]
fn restricted_search_only_starts_from_marked_routes() {
    let customers = [(0., 0.), (1., 0.), (10., 0.), (2., 0.), (11., 0.)]
        .into_iter()
        .enumerate()
        .map(|(i, (x, y))| Customer {
            x,
            y,
            demand: if i == 0 { 0. } else { 1. },
            ready: 0.,
            due: 100.,
            service: 0.,
        })
        .collect();
    let instance = Instance {
        name: "restricted".into(),
        vehicles: 2,
        capacity: 2.,
        customers,
        rounding: Rounding::None,
        distances: None,
    };
    let original = vec![vec![1, 2], vec![3, 4]];
    let mut routes = original.clone();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    improve_modified_routes(&instance, &mut routes, deadline, &[false, false]);
    assert_eq!(routes, original);
    improve_modified_routes(&instance, &mut routes, deadline, &[true, false]);
    assert_ne!(routes, original);
}
