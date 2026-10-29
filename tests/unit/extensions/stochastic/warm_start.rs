use super::*;
use crate::extensions::stochastic::Demand;

#[test]
fn rejects_rounded_overload_and_incomplete_or_repeated_plan() {
    let demand = Demand::new(vec![(0, 0.4996), (1, 0.5004)]).unwrap();
    let problem = Problem::new(
        1,
        vec![demand.clone(), demand],
        vec![vec![0., 1., 1.], vec![1., 0., 1.], vec![1., 1., 0.]],
    )
    .unwrap();
    let found = HgsResult {
        routes: vec![vec![1], vec![1]],
        columns: vec![vec![1, 2], vec![2], vec![0], vec![1, 1]],
        best_cost: Some(4.),
        generations: 0,
        elapsed_secs: 0.,
    };
    let (columns, solution, summary) = problem.validate_hgs_result(Policy::Classical, 2, found);
    assert_eq!(columns, BTreeSet::from([vec![1], vec![2]]));
    assert_eq!(summary.rejected_columns, 3);
    assert!(!summary.accepted_plan);
    assert!(solution.is_none());

    let found = HgsResult {
        routes: vec![vec![1], vec![2]],
        columns: vec![vec![1, 2]],
        best_cost: Some(999.),
        generations: 0,
        elapsed_secs: 0.,
    };
    let (_, solution, summary) = problem.validate_hgs_result(Policy::OptimalRestocking, 2, found);
    assert!(summary.accepted_plan);
    let solution = solution.unwrap();
    let replay: f64 = solution
        .routes
        .iter()
        .map(|route| {
            problem
                .evaluate_route(route, Policy::OptimalRestocking)
                .unwrap()
                .total_cost
        })
        .sum();
    assert_eq!(solution.total_cost, replay);
    assert_ne!(solution.total_cost, 999.);
}
