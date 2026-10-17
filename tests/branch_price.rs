#![cfg(feature = "clp")]
use baldes::{
    master::{
        root::{RootOptions, RootProblem},
        tree::{self, TreeOptions, TreeStatus},
    },
    vrptw::{Instance, Rounding},
};
use std::time::Duration;

fn triangle() -> Instance {
    Instance::parse("triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap()
}
fn options() -> TreeOptions {
    let mut root = RootOptions::default();
    root.cg.stabilize = false;
    TreeOptions {
        root,
        strong_branching: None,
        diving: None,
        time_limit: Duration::from_secs(10),
        ..Default::default()
    }
}
#[test]
fn tree_closes_fractional_root_with_and_without_integer_heuristic() {
    let instance = triangle();
    for (mip, fleet) in [(0, 2.), (0, 3.), (10000, 2.), (10000, 3.)] {
        let problem =
            RootProblem::set_partitioning(instance.graph(1, Rounding::None).unwrap(), fleet)
                .unwrap();
        let mut controls = options();
        controls.mip_node_limit = mip;
        let out = tree::solve(problem, controls).unwrap();
        assert_eq!(out.status, TreeStatus::Optimal, "{out:?}");
        assert!(
            out.nodes > 1,
            "must branch rather than certify a restricted MIP optimum"
        );
        assert!((out.incumbent - (4. + 2f64.sqrt())).abs() < 1e-6, "{out:?}");
        assert_eq!(out.bound, out.incumbent);
        assert_eq!(out.open_nodes, 0);
        assert!(out.routes.len() as f64 <= fleet);
        let mut visited: Vec<_> = out
            .routes
            .iter()
            .flat_map(|r| r.nodes[1..r.nodes.len() - 1].iter().copied())
            .collect();
        visited.sort_unstable();
        assert_eq!(visited, vec![1, 2, 3]);
    }
}
#[test]
fn node_limit_preserves_open_bound_and_never_certifies_restricted_mip() {
    let problem =
        RootProblem::set_partitioning(triangle().graph(1, Rounding::None).unwrap(), 3.).unwrap();
    let mut controls = options();
    controls.node_limit = 1;
    let out = tree::solve(problem, controls).unwrap();
    assert_eq!(out.status, TreeStatus::NodeLimit);
    assert_eq!(out.nodes, 1);
    assert_eq!(out.open_nodes, 2);
    assert!(out.bound < out.incumbent - 0.1, "{out:?}");
    assert!((out.incumbent - (4. + 2f64.sqrt())).abs() < 1e-6);
}
#[test]
fn time_limit_and_cost_only_incumbent_do_not_claim_a_solution() {
    let problem =
        RootProblem::set_partitioning(triangle().graph(1, Rounding::None).unwrap(), 3.).unwrap();
    let mut controls = options();
    controls.time_limit = Duration::ZERO;
    controls.root.incumbent = 0.;
    let out = tree::solve(problem, controls).unwrap();
    assert_eq!(out.status, TreeStatus::TimeLimit);
    assert_eq!(out.nodes, 0);
    assert_eq!(out.bound, 0.);
    assert!(out.incumbent.is_infinite());
    assert!(out.routes.is_empty());
}
// Independently enumerate every feasible elementary tour, then solve the
// full set-partitioning problem by subset DP (no pricing or tree involved).
fn exhaustive(instance: &Instance) -> f64 {
    let n = instance.customers.len() - 1;
    let mut costs = vec![f64::INFINITY; 1 << n];
    fn extend(
        instance: &Instance,
        at: usize,
        mask: usize,
        load: f64,
        cost: f64,
        costs: &mut [f64],
    ) {
        if mask != 0 {
            costs[mask] = costs[mask].min(cost + instance.distance(at, 0, Rounding::None));
        }
        for v in 1..instance.customers.len() {
            if mask & (1 << (v - 1)) != 0 || load + instance.customers[v].demand > instance.capacity
            {
                continue;
            }
            extend(
                instance,
                v,
                mask | (1 << (v - 1)),
                load + instance.customers[v].demand,
                cost + instance.distance(at, v, Rounding::None),
                costs,
            );
        }
    }
    extend(instance, 0, 0, 0., 0., &mut costs);
    let mut best = vec![f64::INFINITY; 1 << n];
    best[0] = 0.;
    for mask in 1..1 << n {
        let mut subset = mask;
        while subset > 0 {
            best[mask] = best[mask].min(costs[subset] + best[mask ^ subset]);
            subset = (subset - 1) & mask;
        }
    }
    best[(1 << n) - 1]
}
#[test]
fn small_trees_agree_with_exhaustive_elementary_route_universe() {
    for (n, capacity) in [(4, 2), (5, 2), (5, 3), (6, 3)] {
        let mut text=format!("small\nVEHICLE\nNUMBER CAPACITY\n{n} {capacity}\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 1000 0\n");
        for i in 1..=n {
            text.push_str(&format!(
                "{i} {} {} 1 0 1000 0\n",
                (i * 7) % 11,
                (i * 3) % 13
            ));
        }
        let instance = Instance::parse(&text).unwrap();
        let exact = exhaustive(&instance);
        let problem =
            RootProblem::set_partitioning(instance.graph(1, Rounding::None).unwrap(), n as f64)
                .unwrap();
        let mut advanced = options();
        advanced.strong_branching = Some(Default::default());
        advanced.diving = Some(Default::default());
        let out = tree::solve(problem, advanced).unwrap();
        assert_eq!(
            out.status,
            TreeStatus::Optimal,
            "n={n} capacity={capacity}: {out:?}"
        );
        assert!(
            (out.incumbent - exact).abs() < 1e-5,
            "n={n} capacity={capacity}: {} != {exact}",
            out.incumbent
        );
    }
}


#[test]
fn strong_branching_solves_with_certified_child_bounds() {
    use baldes::master::tree::StrongBranchingOptions;
    let problem =
        RootProblem::set_partitioning(triangle().graph(1, Rounding::None).unwrap(), 2.).unwrap();
    let mut controls = options();
    controls.strong_branching = Some(StrongBranchingOptions::default());
    let out = tree::solve(problem, controls).unwrap();
    assert_eq!(out.status, TreeStatus::Optimal, "{out:?}");
    assert!(out.strong_branching_trials > 0);
    assert!((out.incumbent - (4. + 2f64.sqrt())).abs() < 1e-6);
    assert_eq!(out.bound, out.incumbent);
}

#[test]
fn diving_without_mip_finds_incumbent_but_does_not_certify_original_node() {
    use baldes::master::tree::DivingOptions;
    let problem =
        RootProblem::set_partitioning(triangle().graph(1, Rounding::None).unwrap(), 2.).unwrap();
    let mut controls = options();
    controls.node_limit = 1;
    controls.mip_node_limit = 0;
    controls.diving = Some(DivingOptions::default());
    let out = tree::solve(problem, controls).unwrap();
    assert_eq!(out.status, TreeStatus::NodeLimit, "{out:?}");
    assert!(out.diving_nodes > 0);
    assert!((out.incumbent - (4. + 2f64.sqrt())).abs() < 1e-6, "{out:?}");
    assert!(
        out.bound < out.incumbent - 0.1,
        "dive bounds cannot close the original node: {out:?}"
    );
}

#[test]
fn heuristic_pricing_trial_never_certifies_a_bound() {
    let instance = triangle();
    let mut root = RootOptions::default();
    root.enumerate = false;
    root.pricing_phase_floor = 1;
    let out = instance.solve_root(1, Rounding::None, root).unwrap();
    assert!(!out.master.bound.is_finite(), "{out:?}");
    assert!(!out.master.pruned_by_bound);
}
