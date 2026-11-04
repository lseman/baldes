#![cfg(feature = "clp")]

use baldes::{
    master::{
        cg::{BoundMode, FeasibilityStatus, Status},
        clp::MasterSolver,
        formulation::Row,
        model::{self, NodeSnapshot, Options, Output, Problem, RowCoefficients, RyanFosterMember},
        preprocessing::Bounds,
        tree::{self, ModelTreeOptions, TreeStatus},
        RowSense,
    },
    Graph,
};
use std::time::Duration;

fn singleton_graph() -> Graph<2> {
    let mut graph = Graph::<2>::new(4, 1).unwrap();
    for vertex in 0..4 {
        graph.set_bounds(vertex, 0, 0., 10.).unwrap();
    }
    for (a, b, cost) in [(0, 1, 100.), (1, 3, 100.), (0, 2, 150.), (2, 3, 150.)] {
        graph.add_arc(a, b, cost).unwrap();
    }
    graph
}

fn singleton_problem() -> Problem<2> {
    Problem::set_partitioning(singleton_graph(), 2.).unwrap()
}

fn options(mode: BoundMode) -> Options {
    let mut options = Options::default();
    options.cg.bound_mode = mode;
    options.cg.artificial_cost = 1.;
    options.cg.max_penalty_updates = 0;
    options.cg.insert_non_negative = false;
    options.cg.stabilize = false;
    options.cg.time_limit = Duration::from_secs(10);
    options
}

/// Registered routes with a positive restricted-master quantity.
fn route_values(output: &Output<2>) -> Vec<(&baldes::Route, f64)> {
    output
        .routes
        .iter()
        .zip(&output.values)
        .filter(|(_, &x)| x > 1e-9)
        .map(|(r, &x)| (&r.route, x))
        .collect()
}

/// A snapshot without any priced column, so a restriction applies before
/// the first pricing round as a fresh-problem branch would.
fn cold_snapshot(problem: Problem<2>, mode: BoundMode) -> NodeSnapshot<2> {
    let mut cold = options(mode);
    cold.cg.time_limit = Duration::ZERO;
    let root = model::solve(problem, cold).unwrap();
    assert_eq!(root.master.pricing_calls, 0);
    root.snapshot
}

fn ryan_foster(first: usize, second: usize) -> [RyanFosterMember; 1] {
    [RyanFosterMember {
        network: 0,
        first,
        second,
    }]
}

fn satisfies(route: &baldes::Route, first: i32, second: i32, together: bool) -> bool {
    let a = route.nodes.contains(&first);
    let b = route.nodes.contains(&second);
    if together {
        a == b
    } else {
        !(a && b)
    }
}

fn arc_row(
    snapshot: &NodeSnapshot<2>,
    arcs: usize,
    arc: usize,
    sense: RowSense,
    rhs: f64,
) -> NodeSnapshot<2> {
    let mut coefficients = vec![0.; arcs];
    coefficients[arc] = 1.;
    snapshot
        .with_row(
            Row { sense, rhs },
            &[RowCoefficients {
                network: 0,
                arcs: coefficients,
                per_route: 0.,
            }],
        )
        .unwrap()
}

#[test]
fn expensive_columns_are_recovered_then_optimized_at_original_costs() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        for stabilized in [false, true] {
            let mut options = options(mode);
            options.cg.stabilize = stabilized;
            let output = model::solve(singleton_problem(), options).unwrap();
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
            // No artificial activity remains: routes alone pay the LP value.
            let paid: f64 = route_values(&output)
                .iter()
                .map(|(r, q)| r.original_cost * q)
                .sum();
            assert!((paid - output.master.lp).abs() < 1e-7, "{output:?}");
            assert_eq!(route_values(&output).len(), 2);
            assert_eq!(
                route_values(&output)
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
    let mut graph = singleton_graph();
    graph.add_arc(1, 2, 50.).unwrap();
    let problem = Problem::set_partitioning(graph, 2.).unwrap();
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        for stabilized in [false, true] {
            let mut controls = options(mode);
            controls.cg.stabilize = stabilized;
            let parent = model::solve(problem.clone(), controls.clone()).unwrap();
            for together in [false, true] {
                let members = ryan_foster(1, 2);
                let fresh = cold_snapshot(problem.clone(), mode)
                    .with_ryan_foster(&members, together)
                    .unwrap()
                    .solve(controls.clone())
                    .unwrap();
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
                    parent
                        .snapshot
                        .with_ryan_foster(&members, together)
                        .unwrap()
                        .solve(controls.clone())
                        .unwrap(),
                ] {
                    let optimum = if together { 300. } else { 500. };
                    assert_eq!(output.master.status, Status::Optimal, "{output:?}");
                    assert!((output.master.lp - optimum).abs() < 1e-6, "{output:?}");
                    assert!(output.master.bound <= optimum + 1e-6, "{output:?}");
                    if !output.master.pruned_by_bound {
                        assert!((output.master.bound - optimum).abs() < 1e-6, "{output:?}");
                    }
                    let paid: f64 = route_values(&output)
                        .iter()
                        .map(|(r, q)| r.original_cost * q)
                        .sum();
                    assert!((paid - output.master.lp).abs() < 1e-6, "{output:?}");
                    assert!(route_values(&output)
                        .iter()
                        .all(|(route, _)| satisfies(route, 1, 2, together)));
                }
            }
        }
    }
}

#[test]
fn phase_one_closes_infeasible_ryan_foster_child_without_pair_route() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let output = cold_snapshot(singleton_problem(), mode)
            .with_ryan_foster(&ryan_foster(1, 2), true)
            .unwrap()
            .solve(options(mode))
            .unwrap();
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
        let output = model::solve(singleton_problem(), options).unwrap();
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

fn fractional_infeasible_problem() -> Problem<2> {
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
    Problem::set_partitioning(graph, 1.4).unwrap()
}

#[test]
fn true_infeasibility_requires_a_positive_whole_domain_feasibility_bound() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let output = model::solve(fractional_infeasible_problem(), options(mode)).unwrap();
        assert!(
            output.master.pricing_calls > 0,
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
fn rounding_boundary_without_a_directed_proof_stays_unknown_when_certified() {
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    let upper = f64::from_bits(1f64.to_bits() - 1);
    graph.edit().initial[0] = Some(0.);
    for vertex in 0..3 {
        graph.set_bounds(vertex, 0, 0., upper).unwrap();
    }
    for (tail, head, load) in [(0, 1, 0.7), (1, 2, 0.3)] {
        let id = graph.add_arc(tail, head, 100.).unwrap();
        graph.arc_mut(id).unwrap().consumption[0] = load;
    }
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let result = model::solve(
            Problem::set_partitioning(graph.clone(), 1.).unwrap(),
            options(mode),
        )
        .unwrap();
        // 0.7 + 0.3 exceeds the window by 2^-54 in exact reals. Exhausted exact
        // pricing settles it numerically (native phase-I rule), but no
        // directed certificate covers the rounding boundary.
        let phase = result.master.feasibility.last().unwrap();
        assert!(!phase.certified_infeasible, "{result:?}");
        assert!(!phase.lower_bound.is_some_and(|b| b > 1e-6), "{result:?}");
        if mode == BoundMode::Numerical {
            assert_eq!(result.master.status, Status::Infeasible, "{result:?}");
            assert_eq!(phase.status, FeasibilityStatus::Infeasible);
        } else {
            assert_eq!(result.master.status, Status::Stalled, "{result:?}");
            assert_eq!(phase.status, FeasibilityStatus::Stalled);
            assert_ne!(result.master.bound, f64::INFINITY);
        }
    }
}

#[test]
fn recovered_parent_fixed_routes_children_and_sibling_keep_original_objective() {
    let problem = singleton_problem();
    let original = problem.clone();
    let options = options(BoundMode::Certified);
    let parent = model::solve(problem, options.clone()).unwrap();
    let first = parent.values.iter().position(|&x| x > 1e-9).unwrap();
    let fixed = parent
        .snapshot
        .fix_route(first, 1.)
        .unwrap()
        .solve(options.clone())
        .unwrap();
    assert_eq!(fixed.master.status, Status::Optimal, "{fixed:?}");
    assert!((fixed.master.lp - 500.).abs() < 1e-7);
    assert!(fixed
        .fixed_routes
        .iter()
        .any(|f| f.route.route.original_cost * f.quantity > 0.));
    let child = parent
        .snapshot
        .with_multiplicity(
            0,
            Bounds {
                lower: 2.,
                upper: 2.,
            },
        )
        .unwrap()
        .solve(options.clone())
        .unwrap();
    assert_eq!(child.master.status, Status::Optimal, "{child:?}");
    assert!((child.master.lp - 500.).abs() < 1e-7);
    let sibling = parent.snapshot.solve(options.clone()).unwrap();
    assert_eq!(sibling.master.status, Status::Optimal);
    assert!((sibling.master.lp - parent.master.lp).abs() < 1e-7);
    let fresh = model::solve(original, options).unwrap();
    assert!((fresh.master.lp - parent.master.lp).abs() < 1e-7);
}

#[test]
fn iteration_limited_recovery_retains_original_columns_and_can_resume() {
    let mut limited = None;
    for budget in 2..12 {
        let mut controls = options(BoundMode::Numerical);
        controls.cg.max_iterations = budget;
        let output = model::solve(singleton_problem(), controls).unwrap();
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
    let resumed = output
        .snapshot
        .solve(options(BoundMode::Numerical))
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
        let arcs = graph.arcs.len();
        let parent = model::solve(
            Problem::set_partitioning(graph.clone(), 1.).unwrap(),
            controls.clone(),
        )
        .unwrap();
        assert_eq!(parent.master.lp, 200.);
        let snapshot = &parent.snapshot;
        let down = arc_row(snapshot, arcs, 0, RowSense::LessEqual, 0.)
            .solve(controls.clone())
            .unwrap();
        assert_eq!(down.master.status, Status::Optimal, "{down:?}");
        assert_eq!(down.master.lp, 500.);
        for (route, _) in route_values(&down) {
            assert!(route.arcs.contains(&expensive));
            assert!(!route.arcs.contains(&cheap));
        }
        let up = arc_row(snapshot, arcs, 0, RowSense::GreaterEqual, 1.)
            .solve(controls.clone())
            .unwrap();
        assert_eq!(up.master.status, Status::Optimal, "{up:?}");
        assert_eq!(up.master.lp, 200.);
        let restored = snapshot.solve(controls).unwrap();
        assert_eq!(restored.master.lp, 200.);
    }
}

#[test]
fn feasibility_infeasibility_closes_tree_nodes_in_both_bound_modes() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let output = tree::solve(
            fractional_infeasible_problem(),
            ModelTreeOptions {
                node: options(mode),
                diving: None,
                time_limit: Duration::from_secs(10),

                branching_registry: ::baldes::master::model::BranchingRegistry::default(),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(output.status, TreeStatus::Infeasible, "{output:?}");
        assert_eq!(output.open_nodes, 0);
        assert!(!output.incumbent.is_finite());
    }
}
