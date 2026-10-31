use super::{improve_modified_routes, nearest_positions};
use crate::inputs::{Customer, Instance, Rounding};

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
