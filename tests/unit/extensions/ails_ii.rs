use super::{canonical_routes, edge_distance, pass_deadline};

#[test]
fn elite_comparison_ignores_route_order_and_empty_routes() {
    let a = vec![vec![1, 2], vec![], vec![3, 4]];
    let b = vec![vec![3, 4], vec![1, 2]];
    let reversed = vec![vec![4, 3], vec![1, 2]];
    assert_eq!(canonical_routes(&a), canonical_routes(&b));
    assert_ne!(canonical_routes(&a), canonical_routes(&reversed));
}

#[test]
fn solution_distance_counts_both_depot_edges_of_singleton_routes() {
    let separate = vec![vec![1], vec![2]];
    let joined = vec![vec![1, 2]];
    assert_eq!(edge_distance(&separate, &joined), 2);
    assert_eq!(edge_distance(&separate, &separate), 0);
}

#[test]
fn local_search_pass_respects_its_overall_deadline() {
    let now = std::time::Instant::now();
    let overall = now + std::time::Duration::from_secs(2);
    assert_eq!(pass_deadline(overall, 0.0), overall);
    let pass = pass_deadline(overall, 1.0);
    assert!(pass >= now && pass <= overall);
    let expired = std::time::Instant::now();
    assert_eq!(pass_deadline(expired, 1.0), expired);
}

#[test]
fn batch_perturbations_preserve_feasible_coverage() {
    use crate::extensions::common::validate_plan;
    use crate::inputs::{Customer, Instance, Rounding};
    use crate::rng::DefaultRng;
    use std::time::{Duration, Instant};

    let customers = (0..=8)
        .map(|i| Customer {
            x: (i % 3) as f64,
            y: (i / 3) as f64,
            demand: if i == 0 { 0. } else { 1. },
            ready: 0.,
            due: 100.,
            service: 0.,
        })
        .collect();
    let instance = Instance {
        name: "batch".into(),
        vehicles: 2,
        capacity: 4.,
        customers,
        rounding: Rounding::None,
        distances: None,
    };
    let original = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];
    for sequence in [false, true] {
        let mut expired = original.clone();
        super::perturbation::perturb(
            &instance,
            &mut expired,
            &mut DefaultRng::new(0),
            4,
            sequence,
            Instant::now(),
        );
        assert_eq!(expired, original);
        for seed in 0..20 {
            let mut routes = original.clone();
            super::perturbation::perturb(
                &instance,
                &mut routes,
                &mut DefaultRng::new(seed),
                4,
                sequence,
                Instant::now() + Duration::from_secs(1),
            );
            validate_plan(&instance, &routes).unwrap();
        }
    }
}
