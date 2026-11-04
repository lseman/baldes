#![cfg(feature = "clp")]
use baldes::{
    inputs::{Instance, Rounding},
    master::{
        cg::BoundMode,
        clp::MasterSolver,
        model::Problem as RoutingContext,
        tree::{self, ModelTreeOptions as TreeOptions, RestrictedMasterMip, TreeStatus},
    },
};
use std::time::Duration;

fn triangle() -> Instance {
    Instance::parse("triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap()
}
fn options() -> TreeOptions {
    let mut options = TreeOptions {
        diving: None,
        time_limit: Duration::from_secs(10),
        ..Default::default()
    };
    options.node.cg.stabilize = false;
    options
}

/// Restricted-master integer heuristic with this HiGHS node budget (0: off).
fn restricted_mip(nodes: usize) -> Option<RestrictedMasterMip> {
    (nodes > 0).then(|| RestrictedMasterMip {
        node_limit: nodes,
        ..Default::default()
    })
}

/// Branch on route count, then on the flow of every arc.
fn arc_flow_only(problem: &RoutingContext, controls: &mut TreeOptions) {
    controls
        .branching_registry
        .register_batch(
            (problem.arc_flow_variables(1.))
                .into_iter()
                .map(::baldes::master::model::BranchingRegistration::Mapped),
        )
        .unwrap();
}

fn customers_of(route: &baldes::master::model::FixedRoute) -> Vec<i32> {
    let nodes = &route.route.route.nodes;
    nodes[1..nodes.len() - 1].to_vec()
}

#[test]
fn parallel_tree_matches_exhaustive_routes_in_both_bound_modes() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        for backend in [MasterSolver::Clp, MasterSolver::Ipm]
            .into_iter()
            .chain(cfg!(feature = "highs").then_some(MasterSolver::Highs))
        {
            for threads in [1, 2, 4] {
                let instance = triangle();
                let problem = RoutingContext::set_partitioning(
                    instance.graph(1, Rounding::None).unwrap(),
                    3.,
                )
                .unwrap();
                let mut controls = options();
                controls.threads = threads;
                controls.node.cg.bound_mode = mode;
                controls.node.master_solver = backend;
                controls.node.gap_enumeration = false;
                let output = tree::solve(problem, controls).unwrap();
                let optimum = exhaustive(&instance);
                assert_eq!(output.status, TreeStatus::Optimal, "{threads}: {output:?}");
                assert!((output.incumbent - optimum).abs() < 1e-6, "{output:?}");
                assert!((output.bound - optimum).abs() < 1e-6, "{output:?}");
                assert_eq!(output.open_nodes, 0);
            }
        }
    }
}

#[test]
fn parallel_tree_preserves_global_node_and_time_budgets() {
    let instance = triangle();
    let optimum = exhaustive(&instance);
    for limit in [0, 1, 2, 3] {
        let problem =
            RoutingContext::set_partitioning(instance.graph(1, Rounding::None).unwrap(), 3.)
                .unwrap();
        let mut controls = options();
        controls.threads = 4;
        controls.node_limit = limit;
        controls.node.gap_enumeration = false;
        let output = tree::solve(problem, controls).unwrap();
        assert!(output.nodes <= limit, "{output:?}");
        assert!(output.bound <= optimum + 1e-6, "{output:?}");
        if output.status != TreeStatus::Optimal {
            assert_eq!(output.status, TreeStatus::NodeLimit, "{output:?}");
            assert!(output.open_nodes > 0, "{output:?}");
        }
    }
    let problem =
        RoutingContext::set_partitioning(instance.graph(1, Rounding::None).unwrap(), 3.).unwrap();
    let mut controls = options();
    controls.threads = 2;
    controls.time_limit = Duration::ZERO;
    let output = tree::solve(problem.clone(), controls).unwrap();
    assert_eq!(output.status, TreeStatus::TimeLimit);
    assert_eq!(output.nodes, 0);
    assert_eq!(output.open_nodes, 1);
    let mut controls = options();
    controls.threads = 0;
    assert!(tree::solve(problem, controls)
        .unwrap_err()
        .message
        .contains("threads positive"));
}
#[test]
fn integer_heuristic_incumbent_bounds_the_original_arc_sum() {
    let mut graph = baldes::Graph::<2>::new(3, 1).unwrap();
    for vertex in 0..3 {
        graph.set_bounds(vertex, 0, 0., 10.).unwrap();
    }
    graph.add_arc(0, 1, 1e16).unwrap();
    graph.add_arc(1, 2, 1.).unwrap();
    let problem = RoutingContext::set_partitioning(graph, 1.).unwrap();
    let mut controls = options();
    controls.node_limit = 1;
    let output = tree::solve(problem, controls).unwrap();
    // The exact original sum is 10^16 + 1, between adjacent doubles.
    // An ordinary route-cost sum rounds down and is not an upper bound.
    assert_eq!(output.incumbent, f64::from_bits(1e16f64.to_bits() + 1));
    assert_eq!(output.routes.len(), 1);
}
#[test]
fn tolerance_closed_tree_keeps_its_lower_certificate() {
    let mut graph = baldes::Graph::<2>::new(4, 1).unwrap();
    for vertex in 0..4 {
        graph.set_bounds(vertex, 0, 0., 10.).unwrap();
    }
    for (tail, head, cost) in [(0, 1, 0.), (1, 3, 1.), (0, 2, 0.), (2, 3, 1.), (1, 2, 0.)] {
        graph.add_arc(tail, head, cost).unwrap();
    }
    let problem = RoutingContext::set_partitioning(graph, 2.).unwrap();
    let mut controls = options();
    controls.node.cg.bound_mode = BoundMode::Certified;
    controls.absolute_gap = 3.;
    controls.relative_gap = 0.;
    let output = tree::solve(problem, controls).unwrap();
    // Initial singleton heuristic costs 2. The unpriced pair route costs 1.
    // Closing at the requested gap cannot turn that incumbent into a lower bound.
    assert_eq!(output.status, TreeStatus::Optimal, "{output:?}");
    // The root is solved; its integer witness closes the tree at the gap.
    assert_eq!(output.nodes, 1);
    assert!(output.incumbent <= 2., "{output:?}");
    assert!(output.bound <= 1. + 1e-6, "{output:?}");
}

#[test]
fn gap_closed_tree_does_not_publish_the_suboptimal_initial_incumbent_as_a_bound() {
    let mut graph = baldes::Graph::<2>::new(4, 1).unwrap();
    for vertex in 0..4 {
        graph.set_bounds(vertex, 0, 0., 10.).unwrap();
    }
    for (tail, head, cost) in [(0, 1, 0.), (1, 3, 1.), (0, 2, 0.), (2, 3, 1.), (1, 2, 0.)] {
        graph.add_arc(tail, head, cost).unwrap();
    }
    let problem = RoutingContext::set_partitioning(graph, 2.).unwrap();
    let root = baldes::master::model::solve(problem.clone(), options().node).unwrap();
    let singletons: Vec<_> = root
        .routes
        .iter()
        .filter(|route| route.route.nodes.len() == 3)
        .map(|route| baldes::master::model::FixedRoute {
            route: route.clone(),
            quantity: 1.,
        })
        .collect();
    assert_eq!(singletons.len(), 2);
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        for threads in [1, 4] {
            for (absolute_gap, relative_gap) in [(3., 0.), (0., 0.75)] {
                let mut controls = options();
                controls.node.cg.bound_mode = mode;
                controls.threads = threads;
                controls.absolute_gap = absolute_gap;
                controls.relative_gap = relative_gap;
                controls.initial_solution = Some((2., singletons.clone()));
                let output = tree::solve(problem.clone(), controls).unwrap();
                assert_eq!(output.status, TreeStatus::Optimal, "{output:?}");
                assert_eq!(output.incumbent, 2.);
                assert_eq!(output.open_nodes, 0);
                // The pair route costs one. Gap closure may retain the more
                // expensive incumbent, but must retain the root's proof too.
                assert!(output.bound.is_finite() && output.bound <= 1., "{output:?}");
            }
        }
    }
}
#[test]
fn tree_closes_fractional_root_with_and_without_integer_heuristic() {
    let instance = triangle();
    for (mip, fleet) in [(0, 2.), (0, 3.), (10000, 2.), (10000, 3.)]
        .into_iter()
        .filter(|(mip, _)| *mip == 0 || cfg!(feature = "highs"))
    {
        let problem =
            RoutingContext::set_partitioning(instance.graph(1, Rounding::None).unwrap(), fleet)
                .unwrap();
        let mut controls = options();
        controls.restricted_master_mip = restricted_mip(mip);
        let out = tree::solve(problem, controls).unwrap();
        assert_eq!(out.status, TreeStatus::Optimal, "{out:?}");
        assert!(
            out.nodes > 1,
            "must branch rather than certify a restricted MIP optimum"
        );
        assert!((out.incumbent - (4. + 2f64.sqrt())).abs() < 1e-6, "{out:?}");
        assert!((out.bound - (4. + 2f64.sqrt())).abs() < 1e-9, "{out:?}");
        assert_eq!(out.open_nodes, 0);
        assert!(out.routes.len() as f64 <= fleet);
        let mut visited: Vec<_> = out.routes.iter().flat_map(customers_of).collect();
        visited.sort_unstable();
        assert_eq!(visited, vec![1, 2, 3]);
    }
}

#[test]
fn tree_rejects_nonfinite_family_priorities_before_solving() {
    let problem =
        RoutingContext::set_partitioning(triangle().graph(1, Rounding::None).unwrap(), 3.).unwrap();
    for priority in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut controls = options();
        controls.node_limit = 0;
        controls
            .branching_registry
            .register(::baldes::master::model::BranchingRegistration::Priorities(
                ::baldes::master::tree::NetworkBranchPriorities {
                    arc: priority,
                    ..Default::default()
                },
            ))
            .unwrap();
        let error = tree::solve(problem.clone(), controls).unwrap_err();
        assert!(error.to_string().contains("priorities"), "{error}");
    }
}

#[test]
fn quota_one_records_trials_without_ranking_history_or_weakening_bounds() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let problem =
            RoutingContext::set_partitioning(triangle().graph(1, Rounding::None).unwrap(), 3.)
                .unwrap();
        let mut controls = options();
        controls.node.cg.bound_mode = mode;
        arc_flow_only(&problem, &mut controls);
        let mut strong = tree::StrongBranchingOptions::default();
        strong.phases.retain(|phase| phase.is_exact());
        strong.phases[0].candidates = 1;
        controls
            .branching_registry
            .register(::baldes::master::model::BranchingRegistration::StrongBranching(Some(strong)))
            .unwrap();
        let out = tree::solve(problem, controls).unwrap();
        assert_eq!(out.status, TreeStatus::Optimal, "{out:?}");
        assert!((out.incumbent - (4. + 2f64.sqrt())).abs() < 1e-6, "{out:?}");
        // Directed lower bounds and upward-rounded witnesses can differ by
        // several ulps even when the requested optimality gap is closed.
        assert!(out.bound <= out.incumbent);
        assert!(out.incumbent - out.bound < 1e-6);
        assert!(out.strong_branching_trials > 0, "{mode:?}: {out:?}");
        assert!(!out.mapped_branch_evaluations.is_empty());
        assert_eq!(out.mapped_history_ranked, 0);
    }
}
#[test]
fn adaptive_trial_budget_changes_work_without_changing_the_optimum() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let problem =
            RoutingContext::set_partitioning(triangle().graph(1, Rounding::None).unwrap(), 3.)
                .unwrap();
        let mut counts = vec![];
        for ratio in [0., 0.5e-12] {
            let mut controls = options();
            controls.node.cg.bound_mode = mode;
            arc_flow_only(&problem, &mut controls);
            let mut strong = tree::StrongBranchingOptions::default();
            strong.phases[0].subtree_ratio = ratio;
            controls
                .branching_registry
                .register(
                    ::baldes::master::model::BranchingRegistration::StrongBranching(Some(strong)),
                )
                .unwrap();
            let out = tree::solve(problem.clone(), controls).unwrap();
            assert_eq!(out.status, TreeStatus::Optimal, "{out:?}");
            assert!((out.incumbent - (4. + 2f64.sqrt())).abs() < 1e-6, "{out:?}");
            assert!(out.bound <= out.incumbent);
            assert!(out.incumbent - out.bound < 1e-6);
            counts.push(out.strong_branching_trials);
        }
        // A tighter subtree budget never adds trials.
        assert!(counts[1] <= counts[0], "{mode:?}: {counts:?}");
    }
}

#[test]
fn completed_depth_history_keeps_numerical_and_certified_tree_bounds() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let problem =
            RoutingContext::set_partitioning(triangle().graph(1, Rounding::None).unwrap(), 3.)
                .unwrap();
        let mut controls = options();
        controls.node.cg.bound_mode = mode;
        arc_flow_only(&problem, &mut controls);
        let mut strong = tree::StrongBranchingOptions::default();
        strong.use_history_subtree_size = true;
        controls
            .branching_registry
            .register(::baldes::master::model::BranchingRegistration::StrongBranching(Some(strong)))
            .unwrap();
        let out = tree::solve(problem, controls).unwrap();
        assert_eq!(out.status, TreeStatus::Optimal, "{out:?}");
        assert!((out.incumbent - (4. + 2f64.sqrt())).abs() < 1e-6, "{out:?}");
        assert!(out.bound <= out.incumbent);
        assert!(out.incumbent - out.bound < 1e-6);
    }
}

#[test]
fn invalid_subtree_ratios_are_rejected_before_solving() {
    let problem =
        RoutingContext::set_partitioning(triangle().graph(1, Rounding::None).unwrap(), 3.).unwrap();
    for ratio in [-1., f64::NAN, f64::INFINITY] {
        let mut controls = options();
        controls.node_limit = 0;
        let mut strong = tree::StrongBranchingOptions::default();
        strong.phases[0].subtree_ratio = ratio;
        controls
            .branching_registry
            .register(::baldes::master::model::BranchingRegistration::StrongBranching(Some(strong)))
            .unwrap();
        assert!(tree::solve(problem.clone(), controls)
            .unwrap_err()
            .to_string()
            .contains("valid nonempty phases"));
    }
}

#[test]
fn singleton_skips_ranking_phases_but_keeps_exact_trials_and_bounds() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let problem =
            RoutingContext::set_partitioning(triangle().graph(1, Rounding::None).unwrap(), 3.)
                .unwrap();
        let mut controls = options();
        controls.node.cg.bound_mode = mode;
        arc_flow_only(&problem, &mut controls);
        let mut strong = tree::StrongBranchingOptions::default();
        strong.phases[0].candidates = 1;
        controls
            .branching_registry
            .register(::baldes::master::model::BranchingRegistration::StrongBranching(Some(strong)))
            .unwrap();
        let out = tree::solve(problem, controls).unwrap();
        assert_eq!(out.status, TreeStatus::Optimal, "{out:?}");
        assert!((out.incumbent - (4. + 2f64.sqrt())).abs() < 1e-6, "{out:?}");
        assert!(out.bound <= out.incumbent);
        assert!(out.incumbent - out.bound < 1e-6);
        assert!(out.strong_branching_trials > 0);
        let stages: Vec<_> = out
            .mapped_branch_evaluations
            .iter()
            .map(|evaluation| evaluation.stage)
            .collect();
        assert!(!stages.is_empty(), "{out:?}");
        assert!(stages.iter().all(|&stage| stage == 1), "{stages:?}");
        assert_eq!(out.mapped_history_ranked, 0);
    }
}
#[cfg(feature = "highs")]
#[test]
fn node_limit_preserves_open_bound_and_never_certifies_restricted_mip() {
    let problem =
        RoutingContext::set_partitioning(triangle().graph(1, Rounding::None).unwrap(), 3.).unwrap();
    let mut controls = options();
    controls.node_limit = 1;
    // Native RestrictedMasterIpHeuristic at the root supplies the incumbent.
    controls.restricted_master_mip = restricted_mip(100_000);
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
        RoutingContext::set_partitioning(triangle().graph(1, Rounding::None).unwrap(), 3.).unwrap();
    let mut controls = options();
    controls.time_limit = Duration::ZERO;
    // An objective value without a witness only authorizes fixing.
    controls.node.fixing_upper_bound = Some(0.);
    let out = tree::solve(problem, controls).unwrap();
    assert_eq!(out.status, TreeStatus::TimeLimit);
    assert_eq!(out.nodes, 0);
    assert!(out.bound <= 0., "{out:?}");
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
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            let problem = RoutingContext::set_partitioning(
                instance.graph(1, Rounding::None).unwrap(),
                n as f64,
            )
            .unwrap();
            let mut advanced = options();
            advanced.node.cg.bound_mode = mode;
            advanced
                .branching_registry
                .register(
                    ::baldes::master::model::BranchingRegistration::StrongBranching(Some(
                        Default::default(),
                    )),
                )
                .unwrap();
            advanced.diving = Some(Default::default());
            let out = tree::solve(problem, advanced).unwrap();
            assert_eq!(out.bound_mode, mode);
            assert_eq!(
                out.status,
                TreeStatus::Optimal,
                "n={n} capacity={capacity} mode={mode:?}: {out:?}"
            );
            assert!(
                (out.incumbent - exact).abs() < 1e-5,
                "n={n} capacity={capacity} mode={mode:?}: {} != {exact}",
                out.incumbent
            );
            assert!(out.bound <= exact + 1e-7, "{out:?}, exact={exact}");
        }
    }
}

#[test]
fn strong_branching_solves_with_certified_child_bounds() {
    use baldes::master::tree::StrongBranchingOptions;
    let problem =
        RoutingContext::set_partitioning(triangle().graph(1, Rounding::None).unwrap(), 2.).unwrap();
    let mut controls = options();
    controls.node.cg.bound_mode = BoundMode::Certified;
    controls
        .branching_registry
        .register(
            ::baldes::master::model::BranchingRegistration::StrongBranching(Some(
                StrongBranchingOptions::default(),
            )),
        )
        .unwrap();
    let out = tree::solve(problem, controls).unwrap();
    assert_eq!(out.status, TreeStatus::Optimal, "{out:?}");
    assert!(out.strong_branching_trials > 0);
    assert!((out.incumbent - (4. + 2f64.sqrt())).abs() < 1e-6);
    assert!((out.bound - (4. + 2f64.sqrt())).abs() < 1e-9, "{out:?}");
}

#[test]
fn diving_without_mip_finds_incumbent_but_does_not_certify_original_node() {
    use baldes::master::tree::DivingOptions;
    let problem =
        RoutingContext::set_partitioning(triangle().graph(1, Rounding::None).unwrap(), 2.).unwrap();
    let mut controls = options();
    controls.node_limit = 1;
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
fn fork_and_carry_inherits_cuts_basis_and_protects_branch_rows() {
    use baldes::master::{
        formulation::Row,
        model::{self, separation::CapacityCutConfig, NodeSnapshot, RowCoefficients},
        tree::{DivingOptions, StrongBranchingOptions},
        RowSense,
    };
    let mut text = String::from(
        "five\nVEHICLE\nNUMBER CAPACITY\n5 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 1000 0\n",
    );
    for i in 1..=5 {
        text.push_str(&format!(
            "{i} {} {} 1 0 1000 0\n",
            (i * 7) % 11,
            (i * 3) % 13
        ));
    }
    let instance = Instance::parse(&text).unwrap();
    let problem =
        RoutingContext::set_partitioning(instance.graph(1, Rounding::None).unwrap(), 5.).unwrap();
    let arcs = problem.networks[0].graph.arcs.len();
    let demands: Vec<f64> = instance.customers.iter().map(|c| c.demand).collect();
    let mut root_opts = model::Options::default();
    root_opts.separation.capacity = Some(CapacityCutConfig::new(demands, instance.capacity));
    root_opts.separation.rank_one.enabled = true;
    root_opts.cg.stabilize = false;
    root_opts.gap_enumeration = false;

    let parent_out = model::solve(problem.clone(), root_opts.clone()).unwrap();
    assert!(parent_out.cuts > 0, "expected active cuts at root");
    let arc_row = |node: &NodeSnapshot<2>, arc: usize| {
        let mut coefficients = vec![0.; arcs];
        coefficients[arc] = 1.;
        node.with_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 0.,
            },
            &[RowCoefficients {
                network: 0,
                arcs: coefficients,
                per_route: 0.,
            }],
        )
        .unwrap()
    };
    // Branch on a customer-to-customer arc.
    let branch_arc = problem.networks[0]
        .graph
        .arcs
        .iter()
        .position(|arc| arc.tail >= 1 && arc.head >= 1)
        .unwrap();
    let child_down = arc_row(&parent_out.snapshot, branch_arc)
        .solve(root_opts.clone())
        .unwrap();
    // Inherited cut rows remain in the child master.
    assert!(child_down.cuts >= parent_out.cuts, "{child_down:?}");
    // A grandchild adds a second branch row; both survive cut cleanup.
    let second_arc = (branch_arc + 1) % arcs;
    let grandchild = arc_row(&child_down.snapshot, second_arc)
        .solve(root_opts.clone())
        .unwrap();
    assert!(grandchild.master.lp >= child_down.master.lp - 1e-6);
    assert!(child_down.master.lp >= parent_out.master.lp - 1e-6);

    // Full tree solve with cuts, strong branching and diving agrees with the
    // exact optimum.
    let exact = exhaustive(&instance);
    for threads in [1, 2, 4] {
        for strong in [false, true] {
            let mut controls = options();
            controls.threads = threads;
            controls.node = root_opts.clone();
            controls
                .branching_registry
                .register(
                    ::baldes::master::model::BranchingRegistration::StrongBranching(
                        strong.then(StrongBranchingOptions::default),
                    ),
                )
                .unwrap();
            controls.diving = Some(DivingOptions::default());
            let tree_out = tree::solve(problem.clone(), controls).unwrap();
            assert_eq!(tree_out.status, TreeStatus::Optimal, "{tree_out:?}");
            assert!((tree_out.incumbent - exact).abs() < 1e-5, "{tree_out:?}");
        }
    }
}

#[test]
fn ryan_foster_tree_agrees_with_exhaustive_routes_and_strong_trials() {
    use baldes::master::model::RyanFosterBranching;
    for (n, capacity) in [(3, 2), (4, 2), (5, 3), (6, 3)] {
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
        for enumerate in [false, true] {
            let problem = RoutingContext::set_partitioning(
                instance.graph(1, Rounding::None).unwrap(),
                n as f64,
            )
            .unwrap();
            let mut controls = options();
            controls.node.gap_enumeration = enumerate;
            arc_flow_only(&problem, &mut controls);
            if let Some(value) = Some(RyanFosterBranching::new(2.)) {
                controls
                    .branching_registry
                    .register(::baldes::master::model::BranchingRegistration::RyanFoster(
                        value,
                    ))
                    .unwrap();
            }
            controls
                .branching_registry
                .register(
                    ::baldes::master::model::BranchingRegistration::StrongBranching(Some(
                        Default::default(),
                    )),
                )
                .unwrap();
            controls.diving = Some(Default::default());
            let out = tree::solve(problem.clone(), controls)
                .unwrap_or_else(|e| panic!("n={n} enum={enumerate}: {e:?}"));
            assert_eq!(
                out.status,
                TreeStatus::Optimal,
                "n={n}, enum={enumerate} {out:?}"
            );
            assert!(
                (out.incumbent - exact).abs() < 1e-5,
                "n={n}, enum={enumerate} {out:?}"
            );
        }
    }
}

#[test]
fn cluster_resource_and_history_trees_preserve_exhaustive_optimum() {
    use baldes::master::{
        model::{format::SolveResult, NetworkClusterBranching},
        tree::NetworkBranchingPolicy,
    };
    let inst = triangle();
    let optimum = exhaustive(&inst);
    for family in ["cluster", "resource", "hybrid"] {
        for strong in [false, true] {
            let graph = inst.graph(8, Rounding::None).unwrap();
            let problem = RoutingContext::set_partitioning(graph, inst.vehicles as f64).unwrap();
            let mut controls = options();
            controls.node.gap_enumeration = false;
            arc_flow_only(&problem, &mut controls);
            if family != "resource" {
                let mut clusters = NetworkClusterBranching::new(0);
                clusters.partition = Some(vec![vec![1], vec![2, 3]]);
                controls
                    .branching_registry
                    .register_batch(
                        (vec![clusters])
                            .into_iter()
                            .map(::baldes::master::model::BranchingRegistration::Cluster),
                    )
                    .unwrap();
            }
            if family != "cluster" {
                controls
                    .branching_registry
                    .register(::baldes::master::model::BranchingRegistration::Policy(
                        NetworkBranchingPolicy::ResourceFirst,
                    ))
                    .unwrap();
            }
            if strong {
                controls
                    .branching_registry
                    .register(
                        ::baldes::master::model::BranchingRegistration::StrongBranching(Some(
                            Default::default(),
                        )),
                    )
                    .unwrap();
            }
            controls
                .branching_registry
                .register(::baldes::master::model::BranchingRegistration::History(
                    true,
                ))
                .unwrap();
            let result = tree::solve(problem, controls).unwrap();
            assert_eq!(
                result.status,
                TreeStatus::Optimal,
                "{family} strong={strong}: {result:?}"
            );
            assert!((result.incumbent - optimum).abs() < 1e-6);
            assert!(SolveResult::tree(&result, 0.)
                .to_json()
                .unwrap()
                .contains("\"mapped_branch_evaluations\":["));
        }
    }
}

#[test]
fn root_and_exact_incumbent_validation_support_more_than_two_resources() {
    let mut graph = baldes::Graph::<5>::new(3, 5).unwrap();
    for v in 0..3 {
        for r in 0..5 {
            graph.set_bounds(v, r, 0., 10.).unwrap();
        }
    }
    let a = graph.add_arc(0, 1, 2.).unwrap();
    let b = graph.add_arc(1, 2, 3.).unwrap();
    for r in 0..5 {
        graph.arc_mut(a).unwrap().consumption[r] = 1.;
        graph.arc_mut(b).unwrap().consumption[r] = 2.;
    }
    let problem = RoutingContext::set_partitioning(graph, 1.).unwrap();
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let mut controls = options();
        controls.node.cg.bound_mode = mode;
        let output = tree::solve(problem.clone(), controls).unwrap();
        assert_eq!(output.status, TreeStatus::Optimal);
        assert_eq!(output.incumbent, 5.);
        assert_eq!(output.routes[0].route.route.resources, vec![3.; 5]);
        assert!(output.bound <= 5.);
    }
}

#[test]
fn edge_and_registered_aggregate_branching_reach_the_exhaustive_optimum() {
    use baldes::master::model::{MappedBranchVariable, MappedNetworkVariable};
    let problem =
        RoutingContext::set_partitioning(triangle().graph(1, Rounding::None).unwrap(), 3.).unwrap();
    let graph = &problem.networks[0].graph;
    // A registered generic variable: the route count, as source departures.
    let departures: Vec<_> = graph
        .arcs
        .iter()
        .filter(|a| a.tail == graph.source && a.head != graph.sink)
        .map(|a| (a.id, 1.))
        .collect();
    let routes = MappedBranchVariable {
        name: "routes".into(),
        priority: 2.,
        networks: vec![MappedNetworkVariable {
            network: 0,
            arcs: departures,
            activities: vec![],
            per_route: 0.,
        }],
        masters: vec![],
    };
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        for aggregate in [false, true] {
            let mut controls = options();
            controls.node.cg.bound_mode = mode;
            controls
                .branching_registry
                .register_batch(
                    (problem.edge_flow_variables(1.))
                        .into_iter()
                        .map(::baldes::master::model::BranchingRegistration::Mapped),
                )
                .unwrap();
            if aggregate {
                controls
                    .branching_registry
                    .register(::baldes::master::model::BranchingRegistration::Mapped(
                        routes.clone(),
                    ))
                    .unwrap();
            }
            let out = tree::solve(problem.clone(), controls).unwrap();
            assert_eq!(out.status, TreeStatus::Optimal, "{out:?}");
            assert!((out.incumbent - (4. + 2f64.sqrt())).abs() < 1e-6, "{out:?}");
            assert!(out.bound <= out.incumbent);
        }
    }
}
