use super::tests::{enumerate, random};
use super::*;
use crate::master::{clp::ClpMaster, LpMaster};

fn solve(p: &Problem) -> (RobustColumnGenerationResult, Vec<f64>) {
    let mut master = RobustMaster::new(p.clone(), ClpMaster::new().unwrap(), 1e4).unwrap();
    let result = master
        .optimize(RobustColumnGenerationOptions::default())
        .unwrap();
    let duals = (0..p.nodes.len())
        .map(|v| if v == p.depot { 0.0 } else { master.lp().duals()[v - 1] })
        .collect();
    (result, duals)
}

#[test]
fn column_generation_converges_with_no_negative_route_left() {
    for seed in 1..5 {
        let p = random(seed, 7, 2, 2, 0.25);
        let (result, duals) = solve(&p);
        assert_eq!(result.status, RobustColumnGenerationStatus::Converged);
        assert!(result.artificial_usage < 1e-9);
        let minimum = enumerate(&p)
            .iter()
            .map(|(r, c)| r.iter().fold(*c, |c, &v| c - duals[v]))
            .fold(f64::INFINITY, f64::min);
        assert!(minimum > -1e-6, "seed {seed}: {minimum}");
    }
}

#[test]
fn lp_bound_grows_with_the_time_budget() {
    for seed in 1..5 {
        let mut previous = f64::NEG_INFINITY;
        for gamma in 0..4 {
            let (result, _) = solve(&random(seed, 7, 0, gamma, 0.5));
            assert_eq!(result.status, RobustColumnGenerationStatus::Converged);
            let objective = result.lp_objective.unwrap();
            assert!(objective >= previous - 1e-6, "seed {seed} gamma {gamma}");
            previous = objective;
        }
    }
}

#[test]
fn diag_coverage() {
    for seed in 1..5 {
        let p = random(seed, 7, 2, 2, 0.25);
        let routes = enumerate(&p);
        let uncovered: Vec<_> = (1..p.nodes.len()).filter(|v| !routes.iter().any(|(r, _)| r.contains(v))).collect();
        eprintln!("seed {seed}: uncovered {uncovered:?}");
    }
}
