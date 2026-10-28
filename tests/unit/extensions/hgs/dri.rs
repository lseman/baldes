use super::*;
use crate::inputs::{Customer, Rounding};

fn instance() -> Instance {
    Instance {
        name: "tiny".into(),
        vehicles: 4,
        capacity: 2.0,
        customers: [0.0, 1.0, 2.0, 9.0, 10.0]
            .iter()
            .enumerate()
            .map(|(i, &x)| Customer {
                x,
                y: 0.0,
                demand: if i == 0 { 0.0 } else { 1.0 },
                ready: 0.0,
                due: 100.0,
                service: 0.0,
            })
            .collect(),
        rounding: Rounding::None,
        distances: None,
    }
}
fn options() -> DriOptions {
    DriOptions {
        time_limit: 2.0,
        cluster_size: 2,
        workers: 2,
        hgs: HgsOptions {
            time_limit: 0.0,
            nb_iter: 10,
            population_size: 3,
            seed: 3,
            verbose: false,
        },
        ..DriOptions::default()
    }
}
#[test]
fn clusters_cover_customers_once_and_respect_size() {
    let data = instance();
    let c = cluster(&data, 2, 5, 2.0, Instant::now() + Duration::from_secs(2)).unwrap();
    assert_eq!(c.clusters.len(), 2);
    assert!(c.clusters.iter().all(|x| !x.is_empty() && x.len() <= 2));
    let mut all: Vec<_> = c.clusters.iter().flatten().copied().collect();
    all.sort_unstable();
    assert_eq!(all, vec![1, 2, 3, 4]);
    for (i, route) in c.clusters.iter().enumerate() {
        assert!(route.contains(&c.medoids[i]));
        for &customer in route {
            assert_eq!(c.assignment[customer], i);
        }
    }
    for row in &c.memberships[1..] {
        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(row.iter().all(|&u| (0.0..=1.0).contains(&u)));
    }
    for (i, row) in c.memberships.iter().enumerate().skip(1) {
        let a = std_dissimilarity(&data, i, c.medoids[0]);
        let b = std_dissimilarity(&data, i, c.medoids[1]);
        if a > 1e-12 && b > 1e-12 {
            assert!((row[0] / row[1] - (b / a).powi(2)).abs() < 1e-8);
        }
    }
}

#[test]
fn coincident_medoids_share_membership_without_dividing_by_zero() {
    let mut data = instance();
    for c in &mut data.customers {
        c.x = 0.0;
    }
    let c = cluster(&data, 2, 2, 2.0, Instant::now() + Duration::from_secs(2)).unwrap();
    for row in &c.memberships[1..] {
        assert_eq!(row, &[0.5, 0.5]);
    }
}

#[test]
fn fuzziness_exponent_changes_membership_ratio() {
    let data = instance();
    let c = cluster(&data, 2, 2, 3.0, Instant::now() + Duration::from_secs(2)).unwrap();
    for (i, row) in c.memberships.iter().enumerate().skip(1) {
        let a = std_dissimilarity(&data, i, c.medoids[0]);
        let b = std_dissimilarity(&data, i, c.medoids[1]);
        if a > 1e-12 && b > 1e-12 {
            assert!((row[0] / row[1] - b / a).abs() < 1e-8);
        }
    }
}

#[test]
fn fuzzy_threshold_limits_cross_cluster_search_to_boundary_customers() {
    let mut data = instance();
    data.capacity = 4.0;
    let clusters = Clustering {
        medoids: vec![1, 2],
        clusters: vec![vec![1], vec![2, 3, 4]],
        assignment: vec![usize::MAX, 0, 1, 1, 1],
        memberships: vec![
            vec![0.0, 0.0],
            vec![0.9, 0.1],
            vec![0.1, 0.9],
            vec![0.1, 0.9],
            vec![0.1, 0.9],
        ],
    };
    let original = vec![vec![1], vec![2, 3, 4]];
    let mut routes = original.clone();
    improve(
        &data,
        &clusters,
        &mut routes,
        &options(),
        Instant::now() + Duration::from_secs(1),
    );
    assert_eq!(routes, original);
    let unrestricted = DriOptions {
        fuzzy_threshold: 1.0,
        ..options()
    };
    improve(
        &data,
        &clusters,
        &mut routes,
        &unrestricted,
        Instant::now() + Duration::from_secs(1),
    );
    assert!(validate_plan(&data, &routes).unwrap() < validate_plan(&data, &original).unwrap());
}

#[test]
fn improvement_can_repair_cluster_solution_exceeding_fleet() {
    let mut data = instance();
    data.vehicles = 2;
    let clusters = Clustering {
        medoids: vec![1, 3],
        clusters: vec![vec![1, 2], vec![3, 4]],
        assignment: vec![usize::MAX, 0, 0, 1, 1],
        memberships: vec![
            vec![0.0, 0.0],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
        ],
    };
    let mut routes = vec![vec![1], vec![2], vec![3], vec![4]];
    assert!(validate_plan(&data, &routes).is_err());
    improve(
        &data,
        &clusters,
        &mut routes,
        &options(),
        Instant::now() + Duration::from_secs(1),
    );
    assert!(validate_plan(&data, &routes).is_ok());
    assert!(routes.len() <= 2);
}

#[test]
fn improvement_applies_single_route_two_opt() {
    let mut data = instance();
    data.capacity = 4.0;
    let clusters = Clustering {
        medoids: vec![1, 3],
        clusters: vec![vec![1, 2], vec![3, 4]],
        assignment: vec![usize::MAX, 0, 0, 1, 1],
        memberships: vec![
            vec![0.0, 0.0],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
        ],
    };
    let mut routes = vec![vec![1, 3, 2, 4]];
    let before = validate_plan(&data, &routes).unwrap();
    improve(
        &data,
        &clusters,
        &mut routes,
        &options(),
        Instant::now() + Duration::from_secs(1),
    );
    assert!(validate_plan(&data, &routes).unwrap() < before);
}
#[test]
fn cross_cluster_relocation_reduces_cost_without_losing_customers() {
    let mut data = instance();
    data.capacity = 4.0;
    let clusters = Clustering {
        medoids: vec![1, 2],
        clusters: vec![vec![1], vec![2, 3, 4]],
        assignment: vec![usize::MAX, 0, 1, 1, 1],
        memberships: vec![
            vec![0.0, 0.0],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
        ],
    };
    let mut routes = vec![vec![1], vec![2, 3, 4]];
    let initial = validate_plan(&data, &routes).unwrap();
    improve(
        &data,
        &clusters,
        &mut routes,
        &options(),
        Instant::now() + Duration::from_secs(1),
    );
    assert!(validate_plan(&data, &routes).unwrap() < initial);
}
#[test]
fn cluster_solve_produces_a_feasible_complete_plan() {
    let data = instance();
    let result = run_dri(&data, None, &options()).unwrap();
    assert_eq!(
        validate_plan(&data, &result.routes).unwrap(),
        result.best_cost.unwrap()
    );
}

#[test]
fn supports_unbounded_time_windows_and_asymmetric_subset_mapping() {
    let mut data = instance();
    for customer in &mut data.customers {
        customer.due = f64::INFINITY;
    }
    data.distances = Some(
        (0..5)
            .map(|i| {
                (0..5)
                    .map(|j| if i == j { 0.0 } else { (10 * i + j) as f64 })
                    .collect()
            })
            .collect(),
    );
    let (sub, map) = subset(&data, &[4, 2]);
    assert_eq!(map, vec![0, 4, 2]);
    assert_eq!(sub.distance(1, 2, sub.rounding), 42.0);
    assert_eq!(sub.distance(2, 1, sub.rounding), 24.0);
    let result = run_dri(&data, None, &options()).unwrap();
    assert_eq!(
        validate_plan(&data, &result.routes).unwrap(),
        result.best_cost.unwrap()
    );
}
#[test]
fn seed_survives_cluster_fleet_failure_and_expired_budget() {
    let mut data = instance();
    data.vehicles = 2;
    let seed = vec![vec![1, 2], vec![3, 4]];
    let opts = DriOptions {
        cluster_size: 1,
        ..options()
    };
    let result = run_dri(&data, Some(&seed), &opts).unwrap();
    assert_eq!(result.routes, seed);
    let opts = DriOptions {
        time_limit: 1e-12,
        ..options()
    };
    let result = run_dri(&data, Some(&seed), &opts).unwrap();
    assert_eq!(result.routes, seed);
}
#[test]
fn rejects_invalid_inputs_and_controls() {
    let data = instance();
    assert!(run_dri(
        &data,
        None,
        &DriOptions {
            fuzziness: 1.0,
            ..options()
        }
    )
    .is_err());
    assert!(run_dri(
        &data,
        None,
        &DriOptions {
            fuzzy_threshold: 1.1,
            ..options()
        }
    )
    .is_err());
    assert!(run_dri(
        &data,
        None,
        &DriOptions {
            workers: 0,
            ..options()
        }
    )
    .is_err());
    let mut data = data;
    data.customers[1].demand = 0.5;
    assert!(run_dri(&data, None, &options()).is_err());
}
