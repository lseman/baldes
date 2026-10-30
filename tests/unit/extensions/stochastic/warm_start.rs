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

#[test]
fn route_orientation_is_selected_by_stochastic_policy_not_travel_cost() {
    let problem = Problem::new(
        4,
        vec![
            Demand::new(vec![(0, 0.7), (4, 0.3)]).unwrap(),
            Demand::new(vec![(0, 0.3), (4, 0.7)]).unwrap(),
        ],
        vec![vec![0., 1., 10.], vec![1., 0., 9.], vec![10., 9., 0.]],
    )
    .unwrap();
    for (policy, expected, reversed) in [
        (Policy::Classical, 20.6, 0),
        (Policy::OptimalRestocking, 20.42, 1),
    ] {
        let found = HgsResult {
            routes: vec![vec![1, 2]],
            columns: vec![vec![1, 2]],
            best_cost: Some(20.),
            generations: 0,
            elapsed_secs: 0.,
        };
        let (columns, solution, summary) = problem.validate_hgs_result(policy, 1, found);
        assert!(summary.accepted_plan);
        assert!((summary.proposal_objective.unwrap() - 20.6).abs() < 1e-9);
        assert!((summary.initial_objective.unwrap() - expected).abs() < 1e-9);
        assert_eq!(summary.reversed_routes, reversed);
        let solution = solution.unwrap();
        assert!((solution.total_cost - expected).abs() < 1e-9);
        assert!(columns.contains(&vec![1, 2]));
        assert!(columns.contains(&solution.routes[0]));
        assert_eq!(
            solution.routes[0],
            if reversed == 1 {
                vec![2, 1]
            } else {
                vec![1, 2]
            }
        );
    }
}
