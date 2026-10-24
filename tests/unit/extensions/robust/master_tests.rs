use super::reference::{
    RobustColumnGenerationOptions, RobustColumnGenerationResult, RobustColumnGenerationStatus,
    RobustMaster,
};
use super::tests::{enumerate, random};
use super::Problem;
use crate::master::clp::ClpMaster;

fn solve(p: &Problem) -> (RobustColumnGenerationResult, Vec<f64>) {
    let mut master = RobustMaster::new(p.clone(), ClpMaster::new().unwrap(), 1e4).unwrap();
    let result = master
        .optimize(RobustColumnGenerationOptions::default())
        .unwrap();
    let duals = (0..p.nodes.len())
        .map(|v| {
            if v == p.depot {
                0.0
            } else {
                master.lp().duals()[v - 1]
            }
        })
        .collect();
    (result, duals)
}

#[test]
fn column_generation_converges_with_no_negative_route_left() {
    for seed in 1..5 {
        let p = random(seed, 7, 2, 2, 0.25);
        let (result, duals) = solve(&p);
        assert_eq!(result.status, RobustColumnGenerationStatus::Converged);
        // Artificials cover exactly the customers no robust route can serve.
        let routes = enumerate(&p);
        let uncovered = (1..p.nodes.len())
            .filter(|v| !routes.iter().any(|(r, _)| r.contains(v)))
            .count();
        assert!((result.artificial_usage - uncovered as f64).abs() < 1e-9);
        let minimum = routes
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
fn bucket_root_lp_matches_the_independent_reference_master() {
    use crate::master::cg::Status;
    use crate::master::root::{self, RootOptions, RootProblem};
    let mut feasible = 0;
    for seed in 1..=8 {
        for (gq, gt) in [(0, 0), (1, 2), (3, 1), (8, 8)] {
            let p = random(seed, 6, gq, gt, 0.5);
            let (reference, _) = solve(&p);
            assert_eq!(reference.status, RobustColumnGenerationStatus::Converged);
            let graph = p
                .native_graph(&vec![0.; p.nodes.len()], 0., p.nodes.len())
                .unwrap();
            let customers = p.nodes.len() - 1;
            let mut options = RootOptions::default();
            options.enumerate = false;
            options.cg.stabilize = false;
            options.cg.multiplicity = customers as f64;
            options.cg.artificial_cost = 1e4;
            let result = root::solve(
                RootProblem::set_partitioning(graph, customers as f64).unwrap(),
                options,
            )
            .unwrap();
            let artificial: f64 = result
                .master
                .artificials
                .iter()
                .map(|&c| result.values[c].max(0.))
                .sum();
            if reference.artificial_usage > 1e-6 {
                assert!(artificial > 1e-6 || result.master.status == Status::Infeasible);
                continue;
            }
            feasible += 1;
            assert_eq!(
                result.master.status,
                Status::Optimal,
                "seed {seed} ({gq},{gt})"
            );
            assert!(artificial <= 1e-6);
            assert!(
                (result.master.lp - reference.lp_objective.unwrap()).abs() < 1e-6,
                "seed {seed} ({gq},{gt}): {} vs {:?}",
                result.master.lp,
                reference.lp_objective
            );
            for (route, value) in result.route_values() {
                if value > 1e-6 {
                    p.replay(&p.native_route(&route.nodes)).unwrap();
                }
            }
        }
    }
    assert!(feasible >= 8, "only {feasible} feasible LP comparisons");
}
