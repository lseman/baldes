use super::*;
use crate::extensions::stochastic::Demand;
use std::time::Duration;

#[test]
fn proof_first_sweep_bounds_elementary_pricing_and_respects_limits() {
    let demands = vec![Demand::new(vec![(0, 0.5), (2, 0.5)]).unwrap(); 3];
    let costs = vec![
        vec![0., 2., 4., 3.],
        vec![2., 0., 2., 2.],
        vec![4., 2., 0., 1.],
        vec![3., 2., 1., 0.],
    ];
    let problem = Problem::new(3, demands, costs).unwrap();
    let duals = vec![4., 4., 4.];
    let exact = problem
        .price(
            Policy::OptimalRestocking,
            &duals,
            0.,
            &[],
            &[],
            false,
            PricingOptions::default(),
        )
        .unwrap();
    assert_eq!(exact.status, PricingStatus::Complete);
    let ng = problem
        .price_ng_scheduled(
            Policy::OptimalRestocking,
            &duals,
            0.,
            &[],
            &[],
            false,
            2,
            PricingOptions::default(),
            true,
        )
        .unwrap();
    assert!(ng.lower_bound.unwrap() <= exact.minimum_reduced_cost.unwrap() + 1e-6);
    let interrupted = problem
        .price_ng_scheduled(
            Policy::OptimalRestocking,
            &duals,
            0.,
            &[],
            &[],
            false,
            2,
            PricingOptions {
                time_limit: Duration::ZERO,
                ..Default::default()
            },
            true,
        )
        .unwrap();
    assert_eq!(interrupted.status, PricingStatus::TimeLimit);
    assert_eq!(interrupted.lower_bound, None);

    // More than nine customers normally triggers the bounded heuristic stages.
    // Capacity one restricts every feasible route to one customer, making the
    // complete result easy to verify without relying on the solver itself.
    let demand = Demand::new(vec![(1, 1.)]).unwrap();
    let costs: Vec<Vec<f64>> = (0..11)
        .map(|i| (0..11).map(|j| if i == j { 0. } else { 1. }).collect())
        .collect();
    let problem = Problem::new(1, vec![demand; 10], costs).unwrap();
    let ng = problem
        .price_ng_scheduled(
            Policy::Classical,
            &[3.; 10],
            0.,
            &[],
            &[],
            false,
            4,
            PricingOptions::default(),
            true,
        )
        .unwrap();
    assert_eq!(ng.status, PricingStatus::Complete);
    let bound = ng.lower_bound.unwrap();
    assert!(bound <= -1. && bound > -1.00001);
}
