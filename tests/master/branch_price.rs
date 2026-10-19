#![cfg(feature = "clp")]
use baldes::{
    master::{
        root::{RootOptions, RootProblem},
        tree::{self, TreeOptions, TreeStatus},
    },
    inputs::{Instance, Rounding},
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
fn integer_heuristic_incumbent_bounds_the_original_arc_sum() {
    let mut graph = baldes::Graph::new(3, 1).unwrap();
    for vertex in 0..3 {
        graph.set_bounds(vertex, 0, 0., 10.).unwrap();
    }
    graph.add_arc(0, 1, 1e16).unwrap();
    graph.add_arc(1, 2, 1.).unwrap();
    let problem = RootProblem::set_partitioning(graph, 1.).unwrap();
    let mut controls = options();
    controls.node_limit = 0;
    let output = tree::solve(problem, controls).unwrap();
    // The exact original sum is 10^16 + 1, between adjacent doubles.
    // An ordinary route-cost sum rounds down and is not an upper bound.
    assert_eq!(output.incumbent, f64::from_bits(1e16f64.to_bits() + 1));
    assert_eq!(output.routes.len(), 1);
    assert_eq!(output.status, TreeStatus::NodeLimit);
}
#[test]
fn tolerance_closed_tree_keeps_its_lower_certificate() {
    let mut graph = baldes::Graph::new(4, 1).unwrap();
    for vertex in 0..4 {
        graph.set_bounds(vertex, 0, 0., 10.).unwrap();
    }
    for (tail, head, cost) in [(0, 1, 0.), (1, 3, 1.), (0, 2, 0.), (2, 3, 1.), (1, 2, 0.)] {
        graph.add_arc(tail, head, cost).unwrap();
    }
    let problem = RootProblem::set_partitioning(graph, 2.).unwrap();
    let mut controls = options();
    controls.absolute_gap = 3.;
    controls.relative_gap = 0.;
    let output = tree::solve(problem, controls).unwrap();
    // Initial singleton heuristic costs 2. The unpriced pair route costs 1.
    // Closing at the requested gap cannot turn that incumbent into a lower bound.
    assert_eq!(output.status, TreeStatus::Optimal);
    assert_eq!(output.nodes, 0);
    assert_eq!(output.incumbent, 2.);
    assert_eq!(output.bound, 0.);
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
        assert_eq!(out.bound, 4. + 2f64.sqrt());
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
    assert_eq!(out.bound, 4. + 2f64.sqrt());
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

#[test]
fn fork_and_carry_inherits_cuts_basis_and_protects_branch_rows() {
    use baldes::master::{
        root::{self, CapacityCutConfig},
        tree::{DivingOptions, StrongBranchingOptions},
        RowSense,
    };
    let mut text = String::from(
        "five\nVEHICLE\nNUMBER CAPACITY\n5 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 1000 0\n",
    );
    for i in 1..=5 {
        text.push_str(&format!("{i} {} {} 1 0 1000 0\n", (i * 7) % 11, (i * 3) % 13));
    }
    let instance = Instance::parse(&text).unwrap();
    let problem =
        RootProblem::set_partitioning(instance.graph(1, Rounding::None).unwrap(), 5.).unwrap();
    let demands: Vec<f64> = instance.customers.iter().map(|c| c.demand).collect();
    let mut root_opts = RootOptions::default()
        .with_capacity_cuts(CapacityCutConfig::new(demands, instance.capacity))
        .with_rank_one_cuts();
    root_opts.capture_state = true;
    root_opts.cg.stabilize = false;
    root_opts.enumerate = false;

    let parent_out = root::solve(problem.clone(), root_opts.clone()).unwrap();
    let snapshot = parent_out
        .snapshot
        .as_ref()
        .expect("root::solve must capture NodeSnapshot");
    assert_eq!(snapshot.branch_depth(), 0);
    assert!(
        snapshot.active_cuts() > 0,
        "expected active cuts at root, got {snapshot:?}"
    );

    // Pick a fractional customer-to-customer arc from the root solution to branch on.
    let mut branch_arc = 0;
    for (index, arc) in problem.graph.arcs.iter().enumerate() {
        if arc.tail >= 1 && arc.head >= 1 {
            branch_arc = index;
            break;
        }
    }
    let delta = [(branch_arc, RowSense::LessEqual, 0.)];
    let child_down = root::inherit(snapshot, &delta, root_opts.clone()).unwrap();
    assert_eq!(child_down.inherited_cuts, snapshot.active_cuts());
    let child_snap = child_down
        .snapshot
        .as_ref()
        .expect("root::inherit must capture child NodeSnapshot");
    assert_eq!(child_snap.branch_depth(), 1);

    // Fork a grandchild with a second branch delta to verify incremental delta slicing
    // and branch-row protection across cut cleanup rounds.
    let second_arc = (branch_arc + 1) % problem.graph.arcs.len();
    let grandchild_delta = [(second_arc, RowSense::LessEqual, 0.)];
    let grandchild = root::inherit(child_snap, &grandchild_delta, root_opts.clone()).unwrap();
    assert_eq!(
        grandchild.snapshot.as_ref().unwrap().branch_depth(),
        2,
        "grandchild must have exactly 2 branch rows"
    );

    // Verify full tree solve with cuts + strong branching + diving agrees with exact optimum.
    let exact = exhaustive(&instance);
    let mut controls = options();
    controls.root = root_opts;
    controls.strong_branching = Some(StrongBranchingOptions::default());
    controls.diving = Some(DivingOptions::default());
    let tree_out = tree::solve(problem, controls).unwrap();
    assert_eq!(tree_out.status, TreeStatus::Optimal, "{tree_out:?}");
    assert!((tree_out.incumbent - exact).abs() < 1e-5, "{tree_out:?}");
}


#[test]
fn ryan_foster_tree_agrees_with_exhaustive_routes_and_strong_trials() {
    use baldes::master::branching::BranchingOptions;
    for (n,capacity) in [(3,2),(4,2),(5,3),(6,3)] {
        let mut text=format!("small\nVEHICLE\nNUMBER CAPACITY\n{n} {capacity}\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 1000 0\n");
        for i in 1..=n {text.push_str(&format!("{i} {} {} 1 0 1000 0\n",(i*7)%11,(i*3)%13));}
        let instance=Instance::parse(&text).unwrap();let exact=exhaustive(&instance);
        for enumerate in [false,true] {
            let problem=RootProblem::set_partitioning(instance.graph(1,Rounding::None).unwrap(),n as f64).unwrap();
            let mut controls=options();controls.root.enumerate=enumerate;
            controls.branching=BranchingOptions{ryan_foster:true,..BranchingOptions::arc_flow_only()};
            controls.strong_branching=Some(Default::default());controls.diving=Some(Default::default());
            let out=tree::solve(problem,controls).unwrap_or_else(|e|panic!("n={n} enum={enumerate}: {e:?}"));
            assert_eq!(out.status,TreeStatus::Optimal,"n={n}, enum={enumerate} {out:?}");
            assert!((out.incumbent-exact).abs()<1e-5,"n={n}, enum={enumerate} {out:?}");
        }
    }
}
