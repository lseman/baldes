use super::*;
use crate::extensions::stochastic::{Demand, Policy, Problem};

#[test]
fn adaptive_ng_bound_covers_every_elementary_route_and_requires_completed_sweeps() {
    for nonmetric in [false, true] {
        let mut c = vec![vec![3.; 5]; 5];
        for (i, row) in c.iter_mut().enumerate() {
            row[i] = 0.;
        }
        if nonmetric {
            c[1][2] = 100.;
        }
        let p = Problem::new(
            4,
            vec![Demand::new(vec![(0, 0.1), (1, 0.2), (2, 0.4), (3, 0.2), (4, 0.1)]).unwrap(); 4],
            c,
        )
        .unwrap();
        let arcs: Vec<_> = (0..5)
            .flat_map(|i| (0..5).filter(move |&j| i != j).map(move |j| (i, j)))
            .collect();
        let weights: Vec<_> = arcs
            .iter()
            .map(|&(i, j)| p.costs()[i][j] - if j == 0 { 0. } else { 10. })
            .collect();
        let resource = RestockingResource::new(p.clone(), arcs).unwrap();
        let bound = minimum(
            &resource,
            &weights,
            &mut 1_000_000,
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap();
        for i in 1..5 {
            for j in 1..5 {
                if i == j {
                    continue;
                }
                let true_cost = p
                    .evaluate_route(&[i, j], Policy::OptimalRestocking)
                    .unwrap()
                    .total_cost
                    - 20.;
                assert!(bound <= true_cost, "{bound} > {true_cost}");
            }
        }
        assert!(minimum(
            &resource,
            &weights,
            &mut 0,
            Instant::now() + Duration::from_secs(1)
        )
        .is_none());
        assert!(minimum(&resource, &weights, &mut 1_000_000, Instant::now()).is_none());
    }
}
