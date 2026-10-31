use super::*;
use crate::inputs::{Instance, Rounding};

fn rc101(customers: usize, rounding: Rounding) -> Problem {
    let root = env!("CARGO_MANIFEST_DIR");
    let mut instance = Instance::read(format!("{root}/instances/Solomon/RC101.txt")).unwrap();
    instance.truncate(customers).unwrap();
    let graph = instance.graph(8, rounding).unwrap();
    Problem::set_partitioning(graph, customers as f64).unwrap()
}

fn options(problem: &Problem) -> ModelTreeOptions {
    let mut configured = ModelTreeOptions::default();
    configured.mapped_variables = problem.arc_flow_variables(1.);
    configured.node_limit = 2000;
    configured
}

#[test]
fn gaps_close_nodes_within_the_native_tolerances() {
    let configured = ModelTreeOptions::default();
    assert_eq!(
        (configured.absolute_gap, configured.relative_gap),
        (1e-6, 1e-9)
    );
    assert!(configured.closes(100. - 5e-7, 100.));
    assert!(!configured.closes(99., 100.));
    let relative = ModelTreeOptions {
        relative_gap: 0.02,
        ..Default::default()
    };
    assert!(relative.closes(98.5, 100.));
    assert!(!relative.closes(97.5, 100.));
    assert!(!relative.closes(1e9, f64::INFINITY));
    let integer = ModelTreeOptions {
        integer_objective: true,
        ..Default::default()
    };
    assert!(integer.closes(99.2, 100.));
    assert!(!integer.closes(98.9, 100.));
    // A bound numerically just above an integer is not rounded past it.
    assert!(!integer.closes(98. + 1e-9, 100.));
}

#[test]
fn integer_objective_bounds_require_and_preserve_integer_costs() {
    let decimal = rc101(25, Rounding::None);
    let mut configured = options(&decimal);
    configured.integer_objective = true;
    assert!(crate::master::tree::solve(decimal, configured).is_err());
    let integral = rc101(25, Rounding::Nearest);
    let plain = crate::master::tree::solve(integral.clone(), options(&integral)).unwrap();
    let mut configured = options(&integral);
    configured.integer_objective = true;
    let rounded = crate::master::tree::solve(integral, configured).unwrap();
    assert_eq!(rounded.status, crate::master::tree::TreeStatus::Optimal);
    assert_eq!(rounded.incumbent, plain.incumbent);
    assert!(rounded.processed_nodes <= plain.processed_nodes);
}

#[test]
fn history_toggle_and_relative_gap_keep_valid_results() {
    let problem = rc101(25, Rounding::None);
    let plain = crate::master::tree::solve(problem.clone(), options(&problem)).unwrap();
    let mut configured = options(&problem);
    configured.use_branching_history = false;
    let unranked = crate::master::tree::solve(problem.clone(), configured).unwrap();
    assert_eq!(unranked.mapped_history_ranked, 0);
    assert!((unranked.incumbent - plain.incumbent).abs() < 1e-6);
    let mut configured = options(&problem);
    configured.relative_gap = 0.05;
    let gapped = crate::master::tree::solve(problem, configured).unwrap();
    assert!(gapped.incumbent <= plain.incumbent * 1.05 + 1e-6);
    assert!(gapped.incumbent >= plain.incumbent - 1e-6);
}

#[cfg(feature = "highs")]
#[test]
fn restricted_master_mip_supplies_a_replayed_root_incumbent() {
    let problem = rc101(25, Rounding::None);
    let mut configured = options(&problem);
    configured.mapped_strong_branching = None;
    configured.node_limit = 1;
    let plain = crate::master::tree::solve(problem.clone(), configured.clone()).unwrap();
    assert!(plain.incumbent.is_infinite());
    configured.restricted_master_mip = Some(RestrictedMasterMip::default());
    let heuristic = crate::master::tree::solve(problem, configured).unwrap();
    assert_eq!(heuristic.restricted_mip_calls, 1);
    assert!(heuristic.incumbent.is_finite());
    assert_eq!(heuristic.bound, plain.bound);
    let replayed: f64 = heuristic
        .routes
        .iter()
        .map(|r| r.quantity * r.route.route.original_cost)
        .sum();
    assert!((replayed - heuristic.incumbent).abs() < 1e-6);
}
