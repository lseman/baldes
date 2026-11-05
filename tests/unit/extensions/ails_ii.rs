use super::{canonical_routes, edge_distance, pass_deadline};

#[test]
fn shared_engine_observes_caller_deadline_and_cancellation() {
    use crate::inputs::{Customer, Instance, Rounding};
    let instance = Instance {
        name: "controlled".into(),
        vehicles: 1,
        capacity: 2.,
        rounding: Rounding::None,
        distances: None,
        customers: [0., 1.]
            .into_iter()
            .enumerate()
            .map(|(i, x)| Customer {
                x,
                y: 0.,
                demand: i as f64,
                ready: 0.,
                due: 100.,
                service: 0.,
            })
            .collect(),
    };
    let seed = vec![vec![1]];
    let options = super::AilsIiOptions {
        time_limit: 0.,
        ..Default::default()
    };
    for cancelled in [false, true] {
        let deadline = (!cancelled).then(std::time::Instant::now);
        let result =
            super::run_ails_ii_controlled(&instance, &seed, &options, deadline, || cancelled)
                .unwrap();
        assert_eq!(result.iterations, 0);
        assert_eq!(result.routes, seed);
        assert_eq!(result.best_cost, 2.);
    }
}

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
