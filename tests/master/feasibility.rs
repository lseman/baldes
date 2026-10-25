#![cfg(feature = "clp")]

use baldes::{
    master::{
        branching::BranchConstraint,
        cg::{FeasibilityStatus, Status},
        root::{self, BoundMode, FixedRoute, MasterSolver, RootOptions, RootProblem},
        tree::{self, TreeOptions, TreeStatus},
        RowSense,
    },
    Graph,
};
use std::time::Duration;

fn singleton_problem() -> RootProblem<2> {
    let mut graph = Graph::<2>::new(4, 1).unwrap();
    for vertex in 0..4 {
        graph.set_bounds(vertex, 0, 0., 10.).unwrap();
    }
    for (a, b, cost) in [(0, 1, 100.), (1, 3, 100.), (0, 2, 150.), (2, 3, 150.)] {
        graph.add_arc(a, b, cost).unwrap();
    }
    RootProblem::set_partitioning(graph, 2.).unwrap()
}

fn options(mode: BoundMode) -> RootOptions {
    let mut options = RootOptions::default();
    options.capture_state = true;
    options.enumerate = false;
    options.round_reduced_costs = false;
    options.bound_mode = mode;
    options.cg.artificial_cost = 1.;
    options.cg.max_penalty_updates = 0;
    options.cg.insert_non_negative = false;
    options.cg.stabilize = false;
    options.cg.time_limit = Duration::from_secs(10);
    options
}

#[test]
fn expensive_columns_are_recovered_then_optimized_at_original_costs() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        for stabilized in [false, true] {
            let mut options = options(mode);
            options.cg.stabilize = stabilized;
            let output = root::solve(singleton_problem(), options).unwrap();
            assert_eq!(output.master.status, Status::Optimal, "{output:?}");
            assert!((output.master.lp - 500.).abs() < 1e-7, "{output:?}");
            assert!(
                !output.master.feasibility.is_empty(),
                "phase I must execute"
            );
            assert_eq!(
                output.master.feasibility[0].status,
                FeasibilityStatus::Feasible
            );
            assert!(output
                .master
                .artificials
                .iter()
                .all(|&c| output.values[c].abs() < 1e-7));
            assert_eq!(output.route_values().len(), 2);
            assert_eq!(
                output
                    .route_values()
                    .iter()
                    .map(|(r, q)| r.original_cost * q)
                    .sum::<f64>(),
                500.
            );
            assert!(output.master.bound <= 500. + 1e-7);
            assert!(output.master.history.iter().any(|h| h.lp >= 500.));
        }
    }
}

#[test]
fn phase_one_recovers_ryan_foster_children_before_original_cost_optimization() {
    use baldes::pricing::restrictions::RyanFosterConstraint;

    let mut problem = singleton_problem();
    problem.graph.add_arc(1, 2, 50.).unwrap();
    // Rebuild arc coefficients after adding the pair route.
    let problem = RootProblem::set_partitioning(problem.graph, 2.).unwrap();
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        for stabilized in [false, true] {
            let mut controls = options(mode);
            controls.cg.stabilize = stabilized;
            let parent = root::solve(problem.clone(), controls.clone()).unwrap();
            for together in [false, true] {
                let rule = RyanFosterConstraint {
                    first: 1,
                    second: 2,
                    together,
                };
                let branch = BranchConstraint::RyanFoster(rule);
                let mut fresh = problem.clone();
                fresh.branch(branch.clone()).unwrap();
                let fresh = root::solve(fresh, controls.clone()).unwrap();
                assert!(
                    fresh
                        .master
                        .feasibility
                        .iter()
                        .any(|report| { report.status == FeasibilityStatus::Feasible }),
                    "{fresh:?}"
                );
                for output in [
                    fresh,
                    root::inherit_branches(
                        parent.snapshot.as_ref().unwrap(),
                        &[branch],
                        controls.clone(),
                    )
                    .unwrap(),
                ] {
                    let optimum = if together { 300. } else { 500. };
                    assert_eq!(output.master.status, Status::Optimal, "{output:?}");
                    assert!((output.master.lp - optimum).abs() < 1e-6, "{output:?}");
                    assert!(output.master.bound <= optimum + 1e-6, "{output:?}");
                    if !output.master.pruned_by_bound {
                        assert!((output.master.bound - optimum).abs() < 1e-6, "{output:?}");
                    }
                    assert!(output
                        .master
                        .artificials
                        .iter()
                        .all(|&c| output.values[c] < 1e-7));
                    assert!(output
                        .route_values()
                        .iter()
                        .all(|(route, value)| *value < 1e-7
                            || rule.satisfied_by(route.nodes.iter().map(|&v| v as usize))));
                }
            }
        }
    }
}

#[test]
fn phase_one_closes_infeasible_ryan_foster_child_without_pair_route() {
    use baldes::pricing::restrictions::RyanFosterConstraint;

    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let mut problem = singleton_problem();
        problem
            .branch(BranchConstraint::RyanFoster(RyanFosterConstraint {
                first: 1,
                second: 2,
                together: true,
            }))
            .unwrap();
        let output = root::solve(problem, options(mode)).unwrap();
        assert_eq!(output.master.status, Status::Infeasible, "{output:?}");
        assert_eq!(output.master.bound, f64::INFINITY);
        assert!(
            output.master.feasibility.iter().any(|report| {
                report.status == FeasibilityStatus::Infeasible
                    && (mode == BoundMode::Numerical || report.certified_infeasible)
            }),
            "{output:?}"
        );
    }
}

#[test]
fn recovery_uses_all_supported_lp_backends() {
    let mut backends = vec![MasterSolver::Clp, MasterSolver::Ipm];
    if cfg!(feature = "highs") {
        backends.push(MasterSolver::Highs);
    }
    for backend in backends {
        let mut options = options(BoundMode::Numerical);
        options.master_solver = backend;
        let output = root::solve(singleton_problem(), options).unwrap();
        assert_eq!(
            output.master.status,
            Status::Optimal,
            "{backend:?}: {output:?}"
        );
        assert!(
            (output.master.lp - 500.).abs() < 1e-5,
            "{backend:?}: {output:?}"
        );
        assert!(output
            .master
            .feasibility
            .iter()
            .any(|round| round.status == FeasibilityStatus::Feasible));
    }
}

fn fractional_infeasible_problem() -> RootProblem<2> {
    let mut graph = Graph::<2>::new(5, 1).unwrap();
    for vertex in 0..5 {
        graph.set_bounds(vertex, 0, 0., 2.).unwrap();
    }
    for tail in 0..4 {
        for head in 1..5 {
            if tail == head {
                continue;
            }
            if tail == 0 && head == 4 {
                continue;
            }
            let id = graph.add_arc(tail, head, 100.).unwrap();
            graph.arc_mut(id).unwrap().consumption[0] = if head == 4 { 0. } else { 1. };
        }
    }
    // Every route serves at most two customers: covering three needs at least
    // 1.5 routes even fractionally, while individual original flows remain viable.
    RootProblem::set_partitioning(graph, 1.4).unwrap()
}

#[test]
fn true_infeasibility_requires_a_positive_whole_domain_feasibility_bound() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let output = root::solve(fractional_infeasible_problem(), options(mode)).unwrap();
        assert!(
            !output.preprocessing.infeasible,
            "exercise phase I rather than preprocessing"
        );
        assert_eq!(output.master.status, Status::Infeasible, "{output:?}");
        assert_eq!(output.master.bound, f64::INFINITY);
        let report = output.master.feasibility.last().unwrap();
        assert_eq!(report.status, FeasibilityStatus::Infeasible);
        assert!(report.lower_bound.unwrap() > 1e-6);
        assert!(report.certified_infeasible);
    }
}

#[test]
fn rounding_boundary_without_a_directed_proof_stays_unknown_in_both_modes() {
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    let upper = f64::from_bits(1f64.to_bits() - 1);
    graph.initial[0] = Some(0.);
    for vertex in 0..3 {
        graph.set_bounds(vertex, 0, 0., upper).unwrap();
    }
    for (tail, head, load) in [(0, 1, 0.7), (1, 2, 0.3)] {
        let id = graph.add_arc(tail, head, 100.).unwrap();
        graph.arc_mut(id).unwrap().consumption[0] = load;
    }
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let result = root::solve(
            RootProblem::set_partitioning(graph.clone(), 1.).unwrap(),
            options(mode),
        )
        .unwrap();
        assert_eq!(result.master.status, Status::Stalled, "{result:?}");
        let phase = result.master.feasibility.last().unwrap();
        assert_eq!(phase.status, FeasibilityStatus::Stalled);
        assert!(phase.lower_bound.unwrap() > 1e-6);
        assert!(!phase.certified_infeasible);
        assert!(result.incumbent.is_infinite());
        assert_ne!(result.master.bound, f64::INFINITY);
    }
}

#[test]
fn recovered_parent_fixed_routes_children_and_sibling_keep_original_objective() {
    let problem = singleton_problem();
    let original = problem.clone();
    let options = options(BoundMode::Certified);
    let parent = root::solve(problem, options.clone()).unwrap();
    let snapshot = parent.snapshot.as_ref().unwrap();
    let fixed = root::inherit_fixed_routes(
        snapshot,
        &[FixedRoute {
            arcs: parent.route_values()[0].0.arcs.clone(),
            quantity: 1,
        }],
        options.clone(),
    )
    .unwrap();
    assert_eq!(fixed.master.status, Status::Optimal, "{fixed:?}");
    assert!((fixed.master.lp - 500.).abs() < 1e-7);
    assert!(fixed.fixed_cost > 0.);
    let branch = BranchConstraint::NumberOfVehicles {
        graph_id: 0,
        sense: RowSense::GreaterEqual,
        rhs: 2.,
    };
    let child = root::inherit_branches(snapshot, &[branch], options.clone()).unwrap();
    assert_eq!(child.master.status, Status::Optimal, "{child:?}");
    assert!((child.master.lp - 500.).abs() < 1e-7);
    let sibling = root::inherit_branches(snapshot, &[], options.clone()).unwrap();
    assert_eq!(sibling.master.status, Status::Optimal);
    assert!((sibling.master.lp - parent.master.lp).abs() < 1e-7);
    let fresh = root::solve(original, options).unwrap();
    assert!((fresh.master.lp - parent.master.lp).abs() < 1e-7);
}

#[test]
fn iteration_limited_recovery_retains_original_columns_and_can_resume() {
    let mut limited = None;
    for budget in 2..12 {
        let mut controls = options(BoundMode::Numerical);
        controls.cg.max_iterations = budget;
        let output = root::solve(singleton_problem(), controls).unwrap();
        assert!(output.master.iterations <= budget, "{output:?}");
        if output
            .master
            .feasibility
            .iter()
            .any(|r| r.status == FeasibilityStatus::IterationLimit && r.pricing_calls > 0)
            && !output.master.inserted_tags.is_empty()
        {
            limited = Some(output);
            break;
        }
    }
    let output = limited.expect("find a deterministic partially completed phase I");
    assert_eq!(output.master.status, Status::IterationLimit);
    for column in output.master.warm_start.columns.iter().flatten() {
        assert!(column.cost >= 200., "never retain zero phase-I costs");
    }
    let resumed = root::inherit_branches(
        output.snapshot.as_ref().unwrap(),
        &[],
        options(BoundMode::Numerical),
    )
    .unwrap();
    assert_eq!(resumed.master.status, Status::Optimal, "{resumed:?}");
    assert!((resumed.master.lp - 500.).abs() < 1e-7);
}

#[test]
fn recovered_parent_keeps_both_arc_branch_children_isolated() {
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    for vertex in 0..3 {
        graph.set_bounds(vertex, 0, 0., 10.).unwrap();
    }
    let cheap = graph.add_arc(0, 1, 100.).unwrap();
    let expensive = graph.add_arc(0, 1, 400.).unwrap();
    graph.add_arc(1, 2, 100.).unwrap();
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let controls = options(mode);
        let parent = root::solve(
            RootProblem::set_partitioning(graph.clone(), 1.).unwrap(),
            controls.clone(),
        )
        .unwrap();
        assert_eq!(parent.master.lp, 200.);
        let snapshot = parent.snapshot.as_ref().unwrap();
        let down = root::inherit_branches(
            snapshot,
            &[BranchConstraint::ArcFlow {
                arc: 0,
                sense: RowSense::LessEqual,
                rhs: 0.,
            }],
            controls.clone(),
        )
        .unwrap();
        assert_eq!(down.master.status, Status::Optimal, "{down:?}");
        assert_eq!(down.master.lp, 500.);
        for (route, _) in down.route_values() {
            assert!(route.arcs.contains(&expensive));
            assert!(!route.arcs.contains(&cheap));
        }
        let up = root::inherit_branches(
            snapshot,
            &[BranchConstraint::ArcFlow {
                arc: 0,
                sense: RowSense::GreaterEqual,
                rhs: 1.,
            }],
            controls.clone(),
        )
        .unwrap();
        assert_eq!(up.master.status, Status::Optimal, "{up:?}");
        assert_eq!(up.master.lp, 200.);
        let restored = root::inherit_branches(snapshot, &[], controls).unwrap();
        assert_eq!(restored.master.lp, 200.);
    }
}

#[test]
fn feasibility_infeasibility_closes_tree_nodes_in_both_bound_modes() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let output = tree::solve(
            fractional_infeasible_problem(),
            TreeOptions {
                root: options(mode),
                strong_branching: None,
                diving: None,
                time_limit: Duration::from_secs(10),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(output.status, TreeStatus::Infeasible, "{output:?}");
        assert_eq!(output.open_nodes, 0);
        assert!(!output.incumbent.is_finite());
    }
}
