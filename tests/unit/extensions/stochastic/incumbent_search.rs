use super::*;
use crate::extensions::stochastic::Demand;
use std::time::Duration;

fn problem() -> Problem {
    Problem::new(
        4,
        vec![
            Demand::new(vec![(2, 1.)]).unwrap(),
            Demand::new(vec![(0, 0.5), (4, 0.5)]).unwrap(),
            Demand::new(vec![(2, 1.)]).unwrap(),
            Demand::new(vec![(0, 0.5), (4, 0.5)]).unwrap(),
        ],
        (0..5)
            .map(|i| (0..5).map(|j| if i == j { 0. } else { 1. }).collect())
            .collect(),
    )
    .unwrap()
}

#[test]
fn improves_true_recourse_with_identical_travel() {
    let p = problem();
    let initial = vec![vec![1, 2], vec![3, 4]];
    let before: f64 = initial.iter().map(|r| travel(&p, r)).sum();
    let (answer, stats) = p
        .search_incumbent(
            Policy::OptimalRestocking,
            2,
            initial,
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap();
    assert!(stats.accepted_moves > 0);
    assert!(stats.final_objective < stats.initial_objective);
    assert_eq!(
        answer.routes.iter().map(|r| travel(&p, r)).sum::<f64>(),
        before
    );
    let mut coverage: Vec<_> = answer.routes.iter().flatten().copied().collect();
    coverage.sort_unstable();
    assert_eq!(coverage, vec![1, 2, 3, 4]);
    assert!(answer.routes.len() <= 2);
    let replay: f64 = answer
        .routes
        .iter()
        .map(|r| {
            p.evaluate_route(r, Policy::OptimalRestocking)
                .unwrap()
                .total_cost
        })
        .sum();
    assert_eq!(answer.total_cost, replay);
    assert_eq!(stats.final_objective, replay);
}

#[test]
fn expired_budget_preserves_valid_plan() {
    let routes = vec![vec![1, 2], vec![3, 4]];
    let (answer, stats) = problem()
        .search_incumbent(Policy::OptimalRestocking, 2, routes.clone(), Instant::now())
        .unwrap();
    assert_eq!(answer.routes, routes);
    assert_eq!(stats.evaluated_moves, 0);
    assert_eq!(stats.accepted_moves, 0);
    assert_eq!(stats.initial_objective, stats.final_objective);
}

#[test]
fn rejects_invalid_coverage_fleet_and_mean_load() {
    for (fleet, routes) in [
        (2, vec![vec![1, 2], vec![1, 4]]),
        (1, vec![vec![1, 2], vec![3, 4]]),
        (2, vec![vec![1, 2, 3], vec![4]]),
        (2, vec![vec![1, 2, 3, 4], vec![]]),
    ] {
        assert!(problem()
            .search_incumbent(Policy::OptimalRestocking, fleet, routes, Instant::now())
            .is_err());
    }
}
